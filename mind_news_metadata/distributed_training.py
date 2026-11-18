"""Graph accumulation with sparse embeddings and one distributed Adam update.

Each replica processes its share of an effective batch sequentially in a graph.
Sparse gradients stay sparse; communication happens only after accumulation.
Loss scaling retries the same batch on overflow, without advancing Adam or the
sampling/LR cursors. All state needed to resume is checkpointed.
"""
import numpy as np
import tensorflow as tf

from .training_runtime import batch_layout


def make_strategy(replicas, allow_cpu=False):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if replicas < 1:
        raise ValueError("replicas must be positive")
    if len(gpus) < replicas:
        if allow_cpu and replicas == 1:
            return tf.distribute.OneDeviceStrategy("/CPU:0")
        raise ValueError("requested {} GPUs, TensorFlow sees {}; use --replicas 1 "
                         "--allow-cpu for a CPU smoke test".format(replicas, len(gpus)))
    return tf.distribute.MirroredStrategy(
        devices=["/GPU:{}".format(i) for i in range(replicas)])


class LossScaleState(tf.Module):
    def __init__(self, mixed):
        super().__init__(name="loss_scale")
        self.scale = tf.Variable(32768.0 if mixed else 1.0, trainable=False)
        self.good_updates = tf.Variable(0, dtype=tf.int64, trainable=False)


class DistributedTrainer:
    def __init__(self, strategy, model, optimizer, effective_batch, batch_size,
                 precision="mixed_float16", xla=False):
        self.strategy, self.model, self.optimizer = strategy, model, optimizer
        self.layout = batch_layout(effective_batch, batch_size, strategy.num_replicas_in_sync)
        self.variables = tuple(model.trainable_variables)
        self.batch_size, self.effective_batch = batch_size, effective_batch
        self.steps = self.layout["gradient_accumulation_steps"]
        self.mixed = precision == "mixed_float16"
        self.loss_scale = LossScaleState(self.mixed)
        self.overflow_retries = 0
        self.replica_gradients = tf.function(self._replica_gradients,
                                            jit_compile=xla, reduce_retracing=True)
        self.compute = tf.function(self._compute, reduce_retracing=True)
        self.apply = tf.function(self._apply, reduce_retracing=True)

    def _micro_gradients(self, features, labels, weights, normalizer, scale):
        with tf.GradientTape() as tape:
            logits = tf.cast(self.model(features, training=True)["logits"], tf.float32)
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits) * weights)
            scaled = (loss / normalizer) * scale
        grads = tape.gradient(scaled, self.variables)
        if any(g is None for g in grads):
            raise ValueError("trainable variable has no gradient")
        return loss, grads

    def _replica_gradients(self, features, labels, weights, normalizer, scale):
        def micro(index):
            start, stop = index * self.batch_size, (index + 1) * self.batch_size
            return self._micro_gradients(
                {k: v[start:stop] for k, v in features.items()},
                labels[start:stop], weights[start:stop], normalizer, scale)

        loss, first = micro(tf.constant(0))
        sparse = tuple(isinstance(g, tf.IndexedSlices) for g in first)
        state = []
        for is_sparse, grad in zip(sparse, first):
            if is_sparse:
                state.append((
                    tf.TensorArray(tf.float32, size=self.steps, infer_shape=False).write(
                        0, tf.cast(grad.values, tf.float32)),
                    tf.TensorArray(grad.indices.dtype, size=self.steps, infer_shape=False).write(
                        0, grad.indices)))
            else:
                state.append(tf.cast(grad, tf.float32))

        def body(index, total_loss, accumulated):
            part_loss, grads = micro(index)
            result = []
            for is_sparse, prior, grad in zip(sparse, accumulated, grads):
                if is_sparse:
                    result.append((prior[0].write(index, tf.cast(grad.values, tf.float32)),
                                   prior[1].write(index, grad.indices)))
                else:
                    result.append(prior + tf.cast(grad, tf.float32))
            return index + 1, total_loss + part_loss, tuple(result)

        _, loss, state = tf.while_loop(
            lambda index, *_: index < self.steps, body,
            (tf.constant(1), loss, tuple(state)), parallel_iterations=1)
        grads, checks = [], []
        for is_sparse, value, original in zip(sparse, state, first):
            if is_sparse:
                values, indices = value[0].concat() / scale, value[1].concat()
                # Coalesce locally before synchronization, without a dense
                # vocabulary-sized gradient buffer.
                unique, segments = tf.unique(indices)
                values = tf.math.unsorted_segment_sum(values, segments, tf.shape(unique)[0])
                grad = tf.IndexedSlices(values, unique, original.dense_shape)
                checks.append(tf.reduce_all(tf.math.is_finite(values)))
            else:
                grad = value / scale
                checks.append(tf.reduce_all(tf.math.is_finite(grad)))
            grads.append(grad)
        finite = tf.reduce_all(tf.stack(checks)) & tf.math.is_finite(loss)
        return loss, tuple(grads), tf.cast(finite, tf.int32)

    def _compute(self, features, labels, weights, normalizer, scale):
        loss, grads, finite = self.strategy.run(
            self.replica_gradients, args=(features, labels, weights, normalizer, scale))
        return (self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
                grads, self.strategy.reduce(tf.distribute.ReduceOp.SUM, finite, axis=None))

    def _apply(self, grads):
        def replica_apply(local_grads):
            self.optimizer.apply_gradients(zip(local_grads, self.variables))
        self.strategy.run(replica_apply, args=(grads,))

    def train_batch(self, features, labels):
        count = len(labels)
        if not 0 < count <= self.effective_batch:
            raise ValueError("invalid effective batch length")
        padding = self.effective_batch - count
        # Only the final short batch is padded. Zero-weight rows do not enter
        # the loss; the locked architecture has no batch norm or dropout.
        if padding:
            features = {k: np.pad(v, [(0, padding)] + [(0, 0)] * (v.ndim - 1))
                        for k, v in features.items()}
        labels = np.pad(np.asarray(labels, np.float32), (0, padding))
        weights = np.concatenate((np.ones(count, np.float32), np.zeros(padding, np.float32)))
        local_size = self.effective_batch // self.strategy.num_replicas_in_sync
        def distribute(context):
            i = int(context.replica_id_in_sync_group)
            start, stop = i * local_size, (i + 1) * local_size
            with tf.device(self.strategy.extended.worker_devices[i]):
                return ({k: tf.convert_to_tensor(v[start:stop], tf.int32) for k, v in features.items()},
                        tf.convert_to_tensor(labels[start:stop]),
                        tf.convert_to_tensor(weights[start:stop]))
        distributed = self.strategy.experimental_distribute_values_from_function(distribute)
        for _ in range(32):
            loss, grads, finite = self.compute(
                *distributed, tf.constant(float(count)), self.loss_scale.scale.read_value())
            if int(finite.numpy()) == self.strategy.num_replicas_in_sync:
                self.apply(grads)
                if self.mixed:
                    self.loss_scale.good_updates.assign_add(1)
                    if int(self.loss_scale.good_updates.numpy()) >= 2000:
                        self.loss_scale.scale.assign(tf.minimum(self.loss_scale.scale * 2, 16777216.0))
                        self.loss_scale.good_updates.assign(0)
                return float(loss.numpy())
            if not self.mixed or float(self.loss_scale.scale.numpy()) <= 1.0:
                raise FloatingPointError("non-finite loss/gradients at loss scale 1; batch was not applied")
            self.overflow_retries += 1
            self.loss_scale.scale.assign(tf.maximum(self.loss_scale.scale / 2, 1.0))
            self.loss_scale.good_updates.assign(0)
        raise FloatingPointError("loss scaling could not produce finite gradients")
