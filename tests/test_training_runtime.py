import unittest
import json
import io
from contextlib import redirect_stdout
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mind_news_metadata.training_runtime import batch_layout, effective_batches, prefetch_batches


class TrainingRuntimeTest(unittest.TestCase):
    def test_layout_and_invalid_combinations(self):
        self.assertEqual(batch_layout(1024, 64, 2)["gradient_accumulation_steps"], 8)
        self.assertEqual(batch_layout(1024, 128, 2)["gradient_accumulation_steps"], 4)
        for args in ((1000, 64, 2), (1024, 0, 2), (1024, 64, 0), (64, 64, 2)):
            with self.assertRaises(ValueError):
                batch_layout(*args)

    @staticmethod
    def groups():
        start = 0
        for count in (3, 11, 1, 8):
            ids = np.arange(start, start + count, dtype=np.int32)
            yield {"id": ids, "history": np.stack([ids, ids], axis=1)}, {
                "click": ids % 2, "labeled": np.ones(count, np.bool_)}, ids.tolist()
            start += count

    def test_prefetched_rebatching_keeps_order_and_final_tail(self):
        batches = list(prefetch_batches(effective_batches(self.groups(), 8), 2))
        self.assertEqual([len(y) for _, y, _ in batches], [8, 8, 7])
        np.testing.assert_array_equal(np.concatenate([f["id"] for f, _, _ in batches]), np.arange(23))
        self.assertEqual(sum([rows for _, _, rows in batches], []), list(range(23)))

    def test_prefetch_propagates_producer_error(self):
        def source():
            yield 1
            raise ValueError("producer failed")
        stream = prefetch_batches(source(), 2)
        self.assertEqual(next(stream), 1)
        with self.assertRaisesRegex(ValueError, "producer failed"):
            next(stream)

    def test_prefetch_can_close_early(self):
        stream = prefetch_batches(iter(range(10000)), 2)
        self.assertEqual(next(stream), 0)
        stream.close()

    def test_unlabeled_candidates_rejected(self):
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            list(effective_batches([({"id": np.zeros(1)}, {
                "click": np.zeros(1), "labeled": np.zeros(1, np.bool_)}, [0])], 8))

    def test_benchmark_records_compile_failure_and_uses_epoch_horizon(self):
        from scripts.benchmark_training import main
        with tempfile.TemporaryDirectory() as folder:
            def run(command, **kwargs):
                self.assertNotIn("--max-updates", command)
                self.assertIn("--stop-after-updates-this-run", command)
                work = Path(command[command.index("--work-root") + 1])
                if "--train-step-compile" in command:
                    return SimpleNamespace(returncode=1)
                report = work / "recommendation_runs/partition-0/training_progress.json"
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({"performance": {"steady_state_candidates_per_second": 123}}))
                return SimpleNamespace(returncode=0)
            with patch("sys.argv", ["benchmark", "--prepared-root", folder,
                                    "--work-root", str(Path(folder) / "benchmark")]), patch(
                    "scripts.benchmark_training.subprocess.run", side_effect=run):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
            rows = json.loads((Path(folder) / "benchmark/benchmark_summary.json").read_text())
            self.assertEqual([r["exit_code"] for r in rows], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
