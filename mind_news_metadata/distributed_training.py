"""PyTorch gradient accumulation across one or more device replicas."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import torch

from .recommendation_model import pointwise_bce_sum
from .training_runtime import batch_layout


def training_devices(replicas, allow_cpu=False):
    if replicas < 1:
        raise ValueError("replicas must be positive")
    visible = torch.cuda.device_count()
    if visible >= replicas:
        return tuple(torch.device("cuda", i) for i in range(replicas))
    if allow_cpu and replicas == 1:
        return (torch.device("cpu"),)
    raise ValueError("requested {} GPUs, PyTorch sees {}; use --replicas 1 "
                     "--allow-cpu for a CPU smoke test".format(replicas, visible))


def to_tensors(features, device):
    return {name: torch.as_tensor(values, dtype=torch.int32, device=device)
            for name, values in features.items()}


def autocast(device, precision):
    return torch.autocast(device.type, dtype=torch.float16, enabled=precision == "mixed_float16")


def replicate_model(model, device):
    # Copy tensors directly to the target to avoid allocating a temporary
    # copy of all embedding tables on the primary accelerator.
    copies = {id(param): torch.nn.Parameter(param.detach().to(device, copy=True),
                                           requires_grad=param.requires_grad)
              for param in model.parameters()}
    copies.update({id(buffer): buffer.to(device, copy=True) for buffer in model.buffers()})
    return deepcopy(model, copies)


def finite_gradients(loss, gradients):
    checks = [loss.isfinite()]
    checks.extend((grad.values() if grad.is_sparse else grad).isfinite().all()
                  for grad in gradients)
    return bool(torch.stack(checks).all().item())


class DistributedTrainer:
    def __init__(self, devices, model, optimizer, effective_batch, batch_size,
                 precision="mixed_float16", compile_head=False):
        self.devices = tuple(devices)
        self.optimizer = optimizer
        batch_layout(effective_batch, batch_size, len(self.devices))
        self.models = [model, *(replicate_model(model, device) for device in self.devices[1:])]
        self.parameters = [tuple(p for p in replica.parameters() if p.requires_grad)
                           for replica in self.models]
        for replica in self.models:
            replica.train()
            if compile_head:
                replica.score_fields = torch.compile(replica.score_fields, dynamic=True)
        self.pool = (ThreadPoolExecutor(max_workers=len(self.devices), thread_name_prefix="mind-train")
                     if len(self.devices) > 1 else None)
        self.batch_size = batch_size
        self.effective_batch = effective_batch
        self.precision = precision
        self.mixed = precision == "mixed_float16"
        self.loss_scale = 32768.0 if self.mixed else 1.0
        self.good_updates = 0
        self.overflow_retries = 0
        self.updates = 0

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    @torch.no_grad()
    def _sync_replicas(self):
        for params in self.parameters[1:]:
            for target, source in zip(params, self.parameters[0]):
                target.copy_(source)

    def _compute(self, index, features, labels, count):
        model, device, params = self.models[index], self.devices[index], self.parameters[index]
        model.zero_grad(set_to_none=True)
        features = to_tensors(features, device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        loss = torch.zeros((), device=device)
        for start in range(0, len(labels), self.batch_size):
            stop = start + self.batch_size
            with autocast(device, self.precision):
                logits = model({name: v[start:stop] for name, v in features.items()})["logits"]
            micro_loss = pointwise_bce_sum(labels[start:stop], logits)
            (micro_loss * (self.loss_scale / count)).backward()
            loss += micro_loss.detach()
            # Limit accumulation to observed embedding rows, even when the
            # same IDs recur in many histories and microbatches.
            for param in params:
                if param.grad is not None and param.grad.is_sparse:
                    param.grad = param.grad.coalesce()
        grads = tuple(param.grad for param in params)
        if any(grad is None for grad in grads):
            raise ValueError("trainable parameter has no gradient")
        return loss, grads, finite_gradients(loss, grads)

    @torch.no_grad()
    def _apply(self, results):
        device = self.devices[0]
        loss, gradients, _ = results[0]
        gradients = list(gradients)
        for other_loss, other_grads, _ in results[1:]:
            loss = loss + other_loss.to(device)
            for index, grad in enumerate(other_grads):
                total = gradients[index] + grad.to(device)
                gradients[index] = total.coalesce() if total.is_sparse else total
        if len(results) > 1 and not finite_gradients(loss, gradients):
            return None
        for param, grad in zip(self.parameters[0], gradients):
            # Adam updates momentum even for rows with zero gradient. Use a
            # dense gradient only for the final optimizer step.
            param.grad = grad.to_dense() if grad.is_sparse else grad
            param.grad.div_(self.loss_scale)
        self.optimizer.step()
        self.updates += 1
        for model in self.models:
            model.zero_grad(set_to_none=True)
        self._sync_replicas()
        return float(loss.item())

    def train_batch(self, features, labels):
        count = len(labels)
        if not 0 < count <= self.effective_batch:
            raise ValueError("invalid effective batch length")
        # Each shard is contiguous. A short final batch can leave some
        # replicas idle, and its mean is normalized by real candidates.
        width = self.effective_batch // len(self.devices)
        shards = [
            (i, {k: v[start:start + width] for k, v in features.items()},
             labels[start:start + width], count)
            for i, start in enumerate(range(0, count, width))
        ]
        for _ in range(32):
            if self.pool is None:
                results = [self._compute(*shards[0])]
            else:
                futures = [self.pool.submit(self._compute, *shard) for shard in shards]
                results = [future.result() for future in futures]
            if all(finite for _, _, finite in results):
                loss = self._apply(results)
                if loss is not None:
                    if self.mixed:
                        self.good_updates += 1
                        if self.good_updates >= 2000:
                            self.loss_scale = min(self.loss_scale * 2, 16777216.0)
                            self.good_updates = 0
                    return loss
            if not self.mixed or self.loss_scale <= 1:
                raise FloatingPointError("non-finite loss/gradients at loss scale 1; batch was not applied")
            self.overflow_retries += 1
            self.loss_scale = max(self.loss_scale / 2, 1.0)
            self.good_updates = 0
        raise FloatingPointError("loss scaling could not produce finite gradients")
