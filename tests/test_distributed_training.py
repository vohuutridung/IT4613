"""Exercise single- and two-replica PyTorch paths on CPU or CUDA."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn

    from mind_news_metadata.checkpoint import latest_checkpoint, load_checkpoint
    from mind_news_metadata.distributed_training import DistributedTrainer, replicate_model, to_tensors
    from mind_news_metadata.field_encoder import DinAttention, MaskedAttention, TitleAbstractMetadataEncoder
    from mind_news_metadata.prediction import ParallelPredictor
    from mind_news_metadata.recommendation_model import NoBodyRecommendationModel, pointwise_bce_sum

    if os.environ.get("MIND_TEST_GPUS") == "1":
        if torch.cuda.device_count() < 2:
            raise RuntimeError("MIND_TEST_GPUS=1 requires two visible GPUs")
        DEVICES = (torch.device("cuda", 0), torch.device("cuda", 1))
    else:
        DEVICES = (torch.device("cpu"), torch.device("cpu"))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(13, 4, sparse=True)
            self.head = nn.Linear(4, 1)

        def forward(self, features):
            logits = self.head(self.embedding(features["ids"]).sum(dim=1)).squeeze(-1)
            return {"logits": logits, "probabilities": logits.float().sigmoid()}


@unittest.skipUnless(HAS_TORCH, "PyTorch is required")
class DistributedTrainingTest(unittest.TestCase):
    def build(self, replicas=2, precision="float32"):
        torch.manual_seed(12)
        model = TinyModel().to(DEVICES[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = DistributedTrainer(DEVICES[:replicas], model, optimizer, 24, 4,
                                     precision=precision)
        self.addCleanup(trainer.close)
        return model, optimizer, trainer

    @staticmethod
    def batch(count):
        return {"ids": (np.arange(count * 3).reshape(count, 3) % 7).astype(np.int32)}, (
            np.arange(count) % 2).astype(np.float32)

    def test_parallel_prediction_preserves_order_and_short_batches(self):
        class PredictModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.barrier = None

            def forward(self, features):
                if self.barrier:
                    self.barrier.wait(timeout=10)
                return {"probabilities": features["ids"][:, 0].float()}

        sample = np.arange(24, dtype=np.int32).reshape(8, 3)
        models = [PredictModel().to(device) for device in DEVICES]
        with ParallelPredictor(models, DEVICES) as parallel:
            for count in (8, 7, 1, 3, 8):
                features = {"ids": sample[:count]}
                np.testing.assert_array_equal(parallel(features), sample[:count, 0])
            # Both workers must enter forward before either can finish.
            barrier = Barrier(2)
            for model in models:
                model.barrier = barrier
            np.testing.assert_array_equal(parallel({"ids": sample}), sample[:, 0])
        with self.assertRaises(RuntimeError):
            parallel({"ids": sample})

    def test_sparse_distributed_adam_matches_full_batch(self):
        model, _, trainer = self.build()
        reference, optimizer, _ = self.build(replicas=1)
        reference.load_state_dict(model.state_dict())
        for count in (24, 19, 1, 24):
            x, y = self.batch(count)
            optimizer.zero_grad(set_to_none=True)
            logits = reference(to_tensors(x, DEVICES[0]))["logits"]
            loss = pointwise_bce_sum(torch.as_tensor(y, device=DEVICES[0]), logits) / count
            loss.backward()
            reference.embedding.weight.grad = reference.embedding.weight.grad.to_dense()
            optimizer.step()
            self.assertAlmostEqual(trainer.train_batch(x, y) / count, loss.item(), places=5)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(trainer.updates, 4)

    def test_mixed_precision_and_empty_attention_are_finite(self):
        _, _, trainer = self.build(precision="mixed_float16")
        x, y = self.batch(19)
        self.assertTrue(np.isfinite(trainer.train_batch(x, y)))
        self.assertEqual(trainer.updates, 1)
        device = DEVICES[0]
        values = torch.zeros((2, 3, 100), device=device, dtype=torch.float16)
        mask = torch.zeros((2, 3), device=device, dtype=torch.bool)
        with torch.autocast(device.type, dtype=torch.float16):
            pooled, weights = MaskedAttention().to(device)(values, mask)
            din = DinAttention().to(device)(values[:, 0], values, mask)
        for tensor in (pooled, weights, din):
            torch.testing.assert_close(tensor, torch.zeros_like(tensor))

    def test_full_15field_model_mixed_precision_step_and_prediction(self):
        from test_kaggle_workflow import CompactStoreTest
        from mind_news_metadata.compact_data import CompactNewsBatchStore

        with tempfile.TemporaryDirectory() as folder:
            store = CompactNewsBatchStore(CompactStoreTest()._fixture(Path(folder)))
            features, labels, _ = store.batch("train", limit=3)
            encoder = TitleAbstractMetadataEncoder(store.vocab_sizes(),
                initialized_test_only=True, initializer_seed=12)
            device = DEVICES[0]
            model = NoBodyRecommendationModel(encoder).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            trainer = DistributedTrainer(DEVICES, model, optimizer, 4, 1)
            self.addCleanup(trainer.close)
            self.assertTrue(np.isfinite(trainer.train_batch(features, labels["click"])))
            self.assertEqual(trainer.updates, 1)
            model.inference_precision = "mixed_float16"
            with ParallelPredictor([model], DEVICES[:1]) as single:
                expected = single(features)
            if os.environ.get("MIND_TEST_PREDICT_STRESS") == "1" and device.type == "cuda":
                batch = {name: np.tile(v, (22,) + (1,) * (v.ndim - 1))[:64]
                         for name, v in features.items()}
                with ParallelPredictor([model], DEVICES[:1]) as single:
                    for _ in range(100):
                        single(batch)
                    baseline = torch.cuda.memory_allocated(device)
                    for _ in range(6500):
                        single(batch)
                    self.assertLess(torch.cuda.memory_allocated(device) - baseline, 64 * 1024 ** 2)
            with ParallelPredictor([model, replicate_model(model, DEVICES[1])], DEVICES) as parallel:
                for count in (3, 1, 2, 3):
                    batch = {k: v[:count] for k, v in features.items()}
                    np.testing.assert_allclose(parallel(batch), expected[:count], rtol=2e-3, atol=2e-4)

    def test_overflow_retries_without_advancing_adam(self):
        _, optimizer, trainer = self.build(replicas=1, precision="mixed_float16")
        original = trainer._compute
        attempts = []

        def overflow_once(*args):
            attempts.append(bool(optimizer.state))
            loss, grads, finite = original(*args)
            return loss, grads, finite and len(attempts) > 1

        trainer._compute = overflow_once
        x, y = self.batch(24)
        trainer.train_batch(x, y)
        self.assertEqual(attempts[:2], [False, False])
        self.assertEqual(trainer.updates, 1)
        self.assertGreaterEqual(trainer.overflow_retries, 1)

    def test_pipeline_resume_preserves_candidates_and_optimizer(self):
        from mind_news_metadata.full_pipeline import run_train
        from mind_news_metadata.prediction import load_run

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
            with patch("mind_news_metadata.full_pipeline.training_devices", return_value=DEVICES), patch(
                    "mind_news_metadata.full_pipeline.make_model", side_effect=model_factory), patch(
                    "mind_news_metadata.full_pipeline.locked_artifacts", return_value=identity):
                options = dict(effective_batch=24, microbatch=4, replicas=2, precision="mixed_float16",
                               db_path=root, checkpoint_every=1, prefetch=2)
                complete = run_train(output_dir=root / "complete", **options)
                first = run_train(output_dir=root / "resume", stop_after_updates_this_run=1, **options)
                resumed = run_train(output_dir=root / "resume", resume=True, **options)
            self.assertEqual(first["examples_seen_total"], 24)
            self.assertEqual(resumed["examples_seen_total"], 55)
            self.assertEqual(resumed["optimizer_iterations"], 3)
            self.assertEqual(resumed["sampling_cursor_last_candidate_key"],
                             complete["sampling_cursor_last_candidate_key"])
            a = load_checkpoint(complete["latest_checkpoint"], complete["config_sha256"])
            b = load_checkpoint(resumed["latest_checkpoint"], resumed["config_sha256"])
            torch.testing.assert_close(a["model"], b["model"])
            torch.testing.assert_close(a["optimizer"], b["optimizer"])
            self.assertEqual(a["loss_scale"], b["loss_scale"])
            self.assertEqual(a["loss_scale_good_updates"], b["loss_scale_good_updates"])
            self.assertEqual(latest_checkpoint(root / "resume").name, "ckpt-3.pt")
            self.assertEqual(sorted(p.name for p in (root / "resume/checkpoints").glob("*.pt")),
                             ["ckpt-2.pt", "ckpt-3.pt"])
            with patch("mind_news_metadata.prediction.make_model", side_effect=model_factory), patch(
                    "mind_news_metadata.prediction.locked_artifacts", return_value=identity):
                restored, _, path, _ = load_run(root / "resume", root, device=DEVICES[0])
            self.assertEqual(path, resumed["latest_checkpoint"])
            torch.testing.assert_close({k: v.cpu() for k, v in restored.state_dict().items()}, b["model"])


if __name__ == "__main__":
    unittest.main()
