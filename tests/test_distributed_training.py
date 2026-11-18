"""Run with TF 2.15.1; uses two logical CPUs unless MIND_TEST_GPUS=1.

Checks distributed sparse Adam against a single full-batch reference, including
duplicate embedding indices, short batches, mixed precision, and checkpoint
continuation. Run this file in a fresh process so CPU devices can be configured.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HAS_TF = importlib.util.find_spec("tensorflow") is not None
if HAS_TF:
    import tensorflow as tf
    if os.environ.get("MIND_TEST_GPUS") == "1":
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) < 2:
            raise RuntimeError("MIND_TEST_GPUS=1 requires two visible GPUs")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICES = ["/GPU:0", "/GPU:1"]
    else:
        cpu = tf.config.list_physical_devices("CPU")[0]
        tf.config.set_logical_device_configuration(cpu, [
            tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()])
        DEVICES = ["/CPU:0", "/CPU:1"]
    from mind_news_metadata.distributed_training import DistributedTrainer
    from mind_news_metadata.field_encoder import MaskedAttention, DinAttention

    class TinyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.embedding = tf.keras.layers.Embedding(13, 4,
                embeddings_initializer=tf.keras.initializers.RandomUniform(seed=12))
            self.head = tf.keras.layers.Dense(1, dtype="float32",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=13))

        def call(self, features, training=False):
            value = tf.reduce_sum(self.embedding(features["ids"]), axis=1)
            return {"logits": tf.squeeze(self.head(value), axis=-1)}


@unittest.skipUnless(HAS_TF, "TensorFlow 2.15.1 is required")
class DistributedTrainingTest(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_inference_accepts_only_the_known_execution_migration(self):
        from mind_news_metadata.prediction import (
            _LEGACY_PREDICTION_SHA256, validate_inference_implementation)
        path = "mind_news_metadata/prediction.py"
        saved = {path: _LEGACY_PREDICTION_SHA256, "model.py": "original"}
        current = {path: "updated", "model.py": "original"}
        validate_inference_implementation(saved, current)
        validate_inference_implementation(current, current)
        for incompatible in (None, {path: "unknown", "model.py": "original"},
                             {path: _LEGACY_PREDICTION_SHA256, "model.py": "changed"}):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_inference_implementation(incompatible, current)

    def test_prediction_graph_parity_single_trace_and_memory(self):
        from test_kaggle_workflow import CompactStoreTest
        from mind_news_metadata.compact_data import CompactNewsBatchStore
        from mind_news_metadata.field_encoder import TitleAbstractMetadataEncoder
        from mind_news_metadata.recommendation_model import NoBodyRecommendationModel
        from mind_news_metadata.prediction import make_predict_step
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        with tempfile.TemporaryDirectory() as folder, tf.device(DEVICES[0]):
            store = CompactNewsBatchStore(CompactStoreTest()._fixture(Path(folder)))
            features, _, _ = store.batch("train", limit=3)
            encoder = TitleAbstractMetadataEncoder(store.vocab_sizes(),
                initialized_test_only=True, initializer_seed=12)
            model = NoBodyRecommendationModel(encoder)
            expected = model(features, training=False)["probabilities"].numpy()
            step = make_predict_step(model, features)
            self.assertEqual(dict(encoder.last_queries), {})
            self.assertIsNone(encoder.entity_din.last_weights)
            for count in (3, 1, 2, 3):
                batch = {k: tf.convert_to_tensor(v[:count]) for k, v in features.items()}
                np.testing.assert_allclose(step(batch).numpy(), expected[:count], rtol=2e-3, atol=2e-4)
            self.assertEqual(step.experimental_get_tracing_count(), 1)
            if os.environ.get("MIND_TEST_PREDICT_STRESS") == "1":
                if not DEVICES[0].startswith("/GPU"):
                    self.skipTest("prediction memory stress requires MIND_TEST_GPUS=1")
                batch = {k: tf.convert_to_tensor(np.tile(v, (22,) + (1,) * (v.ndim - 1))[:64])
                         for k, v in features.items()}
                for _ in range(100):
                    step(batch).numpy()
                baseline = tf.config.experimental.get_memory_info("GPU:0")["current"]
                for _ in range(6500):  # 416,000 candidates, beyond the observed OOM point
                    step(batch).numpy()
                final = tf.config.experimental.get_memory_info("GPU:0")["current"]
                self.assertLess(final - baseline, 64 * 1024 ** 2)

    def build(self, replicas=2, precision="float32", xla=False):
        tf.keras.mixed_precision.set_global_policy(precision)
        strategy = tf.distribute.MirroredStrategy(devices=DEVICES[:replicas])
        with strategy.scope():
            model = TinyModel()
            model({"ids": tf.zeros([1, 3], tf.int32)})
            optimizer = tf.keras.optimizers.Adam(0.001, jit_compile=False)
            optimizer.build(model.trainable_variables)
        trainer = DistributedTrainer(strategy, model, optimizer, 24, 4,
                                     precision=precision, xla=xla)
        return model, optimizer, trainer

    @staticmethod
    def batch(count):
        return {"ids": (np.arange(count * 3).reshape(count, 3) % 7).astype(np.int32)}, (
            np.arange(count) % 2).astype(np.float32)

    def test_sparse_distributed_adam_matches_full_batch(self):
        model, optimizer, trainer = self.build()
        reference, ref_optimizer, ref_trainer = self.build(replicas=1)
        reference.set_weights(model.get_weights())
        # Count=1 leaves the second replica with exclusively zero-weight padding.
        for count in (24, 19, 1, 24):
            x, y = self.batch(count)
            with tf.GradientTape() as tape:
                logits = reference(x)["logits"]
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
            grads = tape.gradient(loss, reference.trainable_variables)
            # Apply the reference under its own strategy, with one replica.
            ref_trainer.strategy.run(
                lambda: ref_optimizer.apply_gradients(zip(grads, reference.trainable_variables)))
            actual_loss = trainer.train_batch(x, y) / count
            self.assertAlmostEqual(actual_loss, float(loss.numpy()), places=5)
            for actual, expected in zip(model.get_weights(), reference.get_weights()):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(int(optimizer.iterations.numpy()), 4)

    def test_mixed_precision_and_empty_attention_are_finite(self):
        model, optimizer, trainer = self.build(precision="mixed_float16")
        x, y = self.batch(19)
        self.assertTrue(np.isfinite(trainer.train_batch(x, y)))
        self.assertEqual(int(optimizer.iterations.numpy()), 1)
        values = tf.zeros([2, 3, 100], tf.float16)
        mask = tf.zeros([2, 3], tf.bool)
        pooled, weights = MaskedAttention()(values, mask)
        din = DinAttention()(tf.zeros([2, 100], tf.float16), values, mask)
        np.testing.assert_array_equal(pooled.numpy(), 0)
        np.testing.assert_array_equal(weights.numpy(), 0)
        np.testing.assert_array_equal(din.numpy(), 0)

    def test_checkpoint_continuation_matches_uninterrupted(self):
        model, opt, trainer = self.build(precision="mixed_float16")
        x, y = self.batch(24)
        trainer.train_batch(x, y)
        with tempfile.TemporaryDirectory() as folder:
            prefix = tf.train.Checkpoint(model=model, optimizer=opt, loss_scale=trainer.loss_scale).save(folder + "/ckpt")
            trainer.train_batch(x, y)
            expected = model.get_weights()
            restored, restored_opt, restored_trainer = self.build(precision="mixed_float16")
            tf.train.Checkpoint(model=restored, optimizer=restored_opt,
                                loss_scale=restored_trainer.loss_scale).restore(prefix).assert_consumed()
            restored_trainer.train_batch(x, y)
            for actual, target in zip(restored.get_weights(), expected):
                np.testing.assert_allclose(actual, target, rtol=2e-5, atol=2e-6)
            self.assertEqual(int(restored_opt.iterations.numpy()), 2)

    def test_full_15field_model_mixed_precision_step(self):
        from test_kaggle_workflow import CompactStoreTest
        from mind_news_metadata.compact_data import CompactNewsBatchStore
        from mind_news_metadata.field_encoder import TitleAbstractMetadataEncoder
        from mind_news_metadata.recommendation_model import NoBodyRecommendationModel
        with tempfile.TemporaryDirectory() as folder:
            store = CompactNewsBatchStore(CompactStoreTest()._fixture(Path(folder)))
            features, labels, _ = store.batch("train", limit=3)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            strategy = tf.distribute.MirroredStrategy(devices=DEVICES)
            with strategy.scope():
                encoder = TitleAbstractMetadataEncoder(store.vocab_sizes(),
                    initialized_test_only=True, initializer_seed=12)
                model = NoBodyRecommendationModel(encoder)
                model({k: tf.convert_to_tensor(v) for k, v in features.items()})
                optimizer = tf.keras.optimizers.Adam(0.001, jit_compile=False)
                optimizer.build(model.trainable_variables)
            trainer = DistributedTrainer(strategy, model, optimizer, 4, 1)
            self.assertTrue(np.isfinite(trainer.train_batch(features, labels["click"])))
            self.assertEqual(int(optimizer.iterations.numpy()), 1)

    def test_overflow_retries_without_advancing_adam(self):
        _, optimizer, trainer = self.build(precision="mixed_float16")
        original = trainer.compute
        attempts = []
        def overflow_once(*args):
            attempts.append(int(optimizer.iterations.numpy()))
            loss, grads, finite = original(*args)
            if len(attempts) == 1:
                finite = tf.constant(0)
            return loss, grads, finite
        trainer.compute = overflow_once
        x, y = self.batch(24)
        trainer.train_batch(x, y)
        self.assertEqual(attempts[0:2], [0, 0])
        self.assertEqual(int(optimizer.iterations.numpy()), 1)
        self.assertGreaterEqual(trainer.overflow_retries, 1)

    def test_pipeline_resume_preserves_candidates_and_optimizer(self):
        from mind_news_metadata.full_pipeline import run_train

        class Store:
            compact = True
            def __init__(self, root):
                self.db_path = str(root)
            def impression_ids(self, split):
                return np.asarray([1, 2])
            def count_selected_partition(self, ids, partition, seed):
                return 27, 28, 2
            def batch(self, split, limit=None, impression_ids=None, **kwargs):
                ids = [1] if impression_ids is None else impression_ids
                features, labels, metadata = [], [], []
                for iid in ids:
                    start, stop = (0, 28) if int(iid) == 1 else (28, 55)
                    for i in range(start, stop):
                        features.append([i % 7, (i + 1) % 7, i % 7])
                        labels.append(i % 2)
                        metadata.append({"candidate_id": i + 1, "impression_id": str(iid),
                                         "position": i - start, "line_no": int(iid), "split": split})
                if limit is not None:
                    features, labels, metadata = features[:limit], labels[:limit], metadata[:limit]
                return {"ids": np.asarray(features, np.int32)}, {
                    "click": np.asarray(labels), "labeled": np.ones(len(labels), np.bool_)}, metadata

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def model_factory(db):
                return TinyModel(), Store(db), {"matrix_sha256": "fixture"}
            identity = {"fingerprint": {"fixture": True}, "location": {"fixture": True}}
            with patch("mind_news_metadata.full_pipeline.make_strategy",
                       side_effect=lambda *a, **k: tf.distribute.MirroredStrategy(devices=DEVICES)), patch(
                    "mind_news_metadata.full_pipeline.make_model", side_effect=model_factory), patch(
                    "mind_news_metadata.full_pipeline.locked_artifacts", return_value=identity):
                options = dict(effective_batch=24, microbatch=4, replicas=2, precision="float32",
                               db_path=root, checkpoint_every=1, prefetch=2)
                complete = run_train(output_dir=root / "complete", **options)
                first = run_train(output_dir=root / "resume", stop_after_updates_this_run=1, **options)
                resumed = run_train(output_dir=root / "resume", resume=True, **options)
            self.assertEqual(first["examples_seen_total"], 24)
            self.assertEqual(resumed["examples_seen_total"], 55)
            self.assertEqual(resumed["optimizer_iterations"], 3)
            self.assertEqual(resumed["sampling_cursor_last_candidate_key"],
                             complete["sampling_cursor_last_candidate_key"])
            a = tf.train.load_checkpoint(complete["latest_checkpoint"])
            b = tf.train.load_checkpoint(resumed["latest_checkpoint"])
            for key in a.get_variable_to_shape_map():
                if key.startswith(("model/", "optimizer/")):
                    np.testing.assert_allclose(a.get_tensor(key), b.get_tensor(key), rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(os.environ.get("MIND_TEST_XLA") == "1", "opt-in XLA compatibility test")
    def test_xla_sparse_step(self):
        _, _, trainer = self.build(xla=True)
        x, y = self.batch(24)
        self.assertTrue(np.isfinite(trainer.train_batch(x, y)))


if __name__ == "__main__":
    unittest.main()
