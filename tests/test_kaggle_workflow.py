import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from deployment.kaggle.runner import Runner
from mind_news_metadata.submission_format import (
    format_impression_line,
    write_submission_zip,
)


ROOT = Path(__file__).resolve().parents[1]


class SubmissionFormatTest(unittest.TestCase):
    def test_official_example_shape(self):
        line = format_impression_line(
            "24481", [(0, 0.1), (1, 0.9), (2, 0.2), (3, 0.8)])
        self.assertEqual(line, "24481 [4,1,3,2]\n")

    def test_ties_use_candidate_position(self):
        self.assertEqual(
            format_impression_line("x", [(0, 0.5), (1, 0.5), (2, 0.2)]),
            "x [1,2,3]\n",
        )

    def test_positions_must_be_contiguous(self):
        with self.assertRaises(ValueError):
            format_impression_line("x", [(0, 0.5), (2, 0.4)])

    def test_zip_contains_only_prediction_txt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text = root / "source.txt"
            output = root / "submission.zip"
            text.write_text("1 [2,1]\n", encoding="utf-8")
            write_submission_zip(text, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["prediction.txt"])
                self.assertEqual(archive.read("prediction.txt"), b"1 [2,1]\n")


class RunnerCliTest(unittest.TestCase):
    def test_prepare_dry_run_needs_no_dataset(self):
        command = [
            sys.executable,
            str(ROOT / "deployment/kaggle/runner.py"),
            "--mode", "prepare",
            "--data-root", "/not-mounted/raw-mind",
            "--work-root", "/tmp/mind15-runner-test",
            "--dry-run",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY_RUN", result.stdout)

    def test_prepared_bundle_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "artifact_manifest.json").write_text(
                '{"artifacts": []}\n', encoding="utf-8")
            (prepared / "stage01.sqlite").write_bytes(b"fixture")
            args = SimpleNamespace(work_root=str(root / "work"), dry_run=False)
            runner = Runner(args)
            runner._finalize_bundle(prepared)
            manifest = runner.verify_bundle(prepared)
            self.assertEqual(
                manifest["status"], "MIND_15FIELD_PREPARED_BUNDLE_READY")
            self.assertTrue(manifest["payload_sha256"])


if __name__ == "__main__":
    unittest.main()
