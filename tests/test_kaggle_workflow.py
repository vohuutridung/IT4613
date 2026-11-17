import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deployment.kaggle.runner import Runner
from mind_data_pipeline.compact_preprocess import build_behaviors
from mind_data_pipeline.preprocess import init_db, pack_ids
from mind_news_metadata.batch_data import NewsBatchStore
from mind_news_metadata.compact_data import CompactNewsBatchStore
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
            compact = prepared / "compact"
            compact.mkdir()
            (compact / "compact_manifest.json").write_text(
                '{"status":"MIND_COMPACT_MEMMAP_READY"}\n', encoding="utf-8")
            args = SimpleNamespace(work_root=str(root / "work"), dry_run=False)
            runner = Runner(args)
            runner._finalize_bundle(prepared)
            manifest = runner.verify_bundle(prepared)
            self.assertEqual(
                manifest["status"], "MIND_15FIELD_PREPARED_BUNDLE_READY")
            self.assertTrue(manifest["payload_sha256"])


class CompactStoreTest(unittest.TestCase):
    def _fixture(self, root):
        prepared = root / "prepared"
        compact = prepared / "compact"
        news = compact / "news"
        behavior = compact / "behaviors/train"
        vocab = prepared / "vocab"
        news.mkdir(parents=True)
        behavior.mkdir(parents=True)
        vocab.mkdir(parents=True)
        (vocab / "did.tsv").write_text(
            "0\t<PAD>\t0\n1\t<UNK>\t0\n2\tN1\t1\n3\tN2\t1\n4\tN3\t1\n",
            encoding="utf-8")
        from mind_config import CONFIG_DATA
        shared = CONFIG_DATA["shared_architecture"]
        dids = 5
        values = {
            "present": np.asarray([0, 0, 1, 1, 1], np.bool_),
            "source_split": np.zeros(dids, np.uint8),
            "cat": np.asarray([1, 1, 2, 2, 2], np.int32),
            "sub_cat": np.asarray([1, 1, 2, 2, 2], np.int32),
            "title": np.zeros((dids, shared["max_title_tokens"]), np.int32),
            "abstract": np.zeros((dids, shared["max_abstract_tokens"]), np.int32),
            "title_entities": np.zeros((dids, shared["max_title_entities"]), np.int32),
            "title_entity_types": np.zeros((dids, shared["max_title_entities"]), np.int32),
            "abstract_entities": np.zeros((dids, shared["max_abstract_entities"]), np.int32),
            "abstract_entity_types": np.zeros((dids, shared["max_abstract_entities"]), np.int32),
        }
        values["title"][2:, 0] = [12, 13, 14]
        for name, value in values.items():
            np.save(news / (name + ".npy"), value, allow_pickle=False)
        np.save(behavior / "uid.npy", np.asarray([2, 3], np.int32))
        np.save(behavior / "impression_id.npy", np.asarray([b"10", b"11"], dtype="S32"))
        np.save(behavior / "history_offsets.npy", np.asarray([0, 1, 3], np.int64))
        np.save(behavior / "candidate_offsets.npy", np.asarray([0, 2, 3], np.int64))
        np.asarray([2, 3, 2], dtype="<i4").tofile(behavior / "history_did.i32")
        np.asarray([3, 4, 2], dtype="<i4").tofile(behavior / "candidate_did.i32")
        np.asarray([1, 0, 1], dtype="i1").tofile(behavior / "click.i8")
        layout = {
            "vocab": {
                "uid_embedding_rows": 4,
                "sizes": {
                    "uid": 4, "did": 5, "cat": 3, "sub_cat": 3,
                    "entity": 3, "entity_type": 3, "word": 20,
                },
            },
            "splits": {
                "train": {
                    "impressions": 2, "candidates": 3, "history_values": 3,
                    "impression_base": 1, "candidate_base": 1,
                }
            },
        }
        (compact / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
        (compact / "compact_manifest.json").write_text(
            json.dumps({"status": "MIND_COMPACT_MEMMAP_READY"}), encoding="utf-8")
        return prepared

    def _sqlite_fixture(self, root):
        path = root / "stage01.sqlite"
        conn = init_db(path)
        threshold = 1_000_010
        vocab_rows = [
            ("uid", 0, "<PAD>", 0), ("uid", 1, "<UNK>", 0),
            ("uid", 2, "U1", threshold), ("uid", 3, "U2", threshold),
        ]
        conn.executemany("INSERT INTO vocab VALUES(?,?,?,?)", vocab_rows)
        from mind_config import CONFIG_DATA
        shared = CONFIG_DATA["shared_architecture"]
        zeros = {
            "title": [0] * shared["max_title_tokens"],
            "abstract": [0] * shared["max_abstract_tokens"],
            "title_entities": [0] * shared["max_title_entities"],
            "abstract_entities": [0] * shared["max_abstract_entities"],
        }
        for did, news_id, first_token in ((2, "N1", 12), (3, "N2", 13), (4, "N3", 14)):
            title = list(zeros["title"])
            title[0] = first_token
            conn.execute(
                "INSERT INTO news VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (did, news_id, "train", 2, 2, pack_ids(title),
                 pack_ids(zeros["abstract"]), None,
                 pack_ids(zeros["title_entities"]), pack_ids(zeros["title_entities"]),
                 pack_ids(zeros["abstract_entities"]), pack_ids(zeros["abstract_entities"])))
        conn.executemany(
            "INSERT INTO impressions VALUES(?,?,?,?,?,?,?,?)",
            [(1, "train", 1, "10", 2, "T", pack_ids([2]), 1),
             (2, "train", 2, "11", 3, "T", pack_ids([3, 2]), 2)])
        conn.executemany(
            "INSERT INTO candidates VALUES(?,?,?,?)",
            [(1, 1, 3, 0), (2, 1, 4, 1), (3, 2, 2, 0)])
        conn.executemany(
            "INSERT INTO labels VALUES(?,?,?)",
            [(1, 1, 1), (2, 0, 1), (3, 1, 1)])
        conn.commit()
        conn.close()
        return path

    def test_compact_batch_order_and_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = CompactNewsBatchStore(self._fixture(Path(temporary)))
            features, labels, metadata = store.batch("train", limit=3)
            self.assertEqual([row["candidate_id"] for row in metadata], [1, 2, 3])
            self.assertEqual([row["position"] for row in metadata], [0, 1, 0])
            np.testing.assert_array_equal(labels["click"], [1, 0, 1])
            np.testing.assert_array_equal(features["cur_title_tokens"][:, 0], [13, 14, 12])
            np.testing.assert_array_equal(features["his_id"][0, :2], [2, 0])

    def test_compact_keyset_and_impression_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = CompactNewsBatchStore(self._fixture(Path(temporary)))
            _, _, metadata = store.batch("train", limit=2, after_key=(1, 0))
            self.assertEqual(
                [(row["line_no"], row["position"]) for row in metadata],
                [(1, 1), (2, 0)])
            _, _, shuffled = store.batch("train", impression_ids=[2, 1])
            self.assertEqual([row["impression_id"] for row in shuffled], ["11", "10", "10"])

    def test_compact_batch_matches_sqlite_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = CompactNewsBatchStore(self._fixture(root))
            sqlite = NewsBatchStore(self._sqlite_fixture(root))
            compact_batch = compact.batch("train", limit=3)
            sqlite_batch = sqlite.batch("train", limit=3)
            self.assertEqual(set(compact_batch[0]), set(sqlite_batch[0]))
            for key in compact_batch[0]:
                np.testing.assert_array_equal(
                    compact_batch[0][key], sqlite_batch[0][key], err_msg=key)
            np.testing.assert_array_equal(
                compact_batch[1]["click"], sqlite_batch[1]["click"])
            np.testing.assert_array_equal(
                compact_batch[1]["labeled"], sqlite_batch[1]["labeled"])
            self.assertEqual(compact_batch[2], sqlite_batch[2])
            for partition in range(5):
                compact_part = compact.batch(
                    "train", impression_ids=[1, 2], partition=partition)
                sqlite_part = sqlite.batch(
                    "train", impression_ids=[1, 2], partition=partition)
                self.assertEqual(compact_part[2], sqlite_part[2])
                np.testing.assert_array_equal(
                    compact_part[1]["click"], sqlite_part[1]["click"])

    def test_compact_behavior_builder_preserves_order_and_null_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            specs = {
                "train": "1\tU1\tT\tN1 N2\tN3-1 N2-0\n",
                "dev": "2\tU1\tT\tN1\tN2-1\n",
                "test": "3\tU1\tT\tN2\tN1\n",
            }
            manifest = {"splits": {}}
            for split, text in specs.items():
                folder = data / ("MINDlarge_" + split)
                folder.mkdir(parents=True)
                (folder / "behaviors.tsv").write_text(text, encoding="utf-8")
                history_count = 2 if split == "train" else 1
                candidate_count = 2 if split == "train" else 1
                manifest["splits"]["MINDlarge_" + split] = {
                    "files": {"behaviors.tsv": {
                        "rows": 1, "history_tokens": history_count,
                        "candidate_tokens": candidate_count,
                    }}
                }
            compact = root / "prepared/compact"
            metadata, membership = build_behaviors(
                data, compact,
                {"uid": {"U1": 2}, "did": {"N1": 2, "N2": 3, "N3": 4}},
                manifest)
            self.assertEqual(metadata["train"]["candidates"], 2)
            history = np.fromfile(
                compact / "behaviors/train/history_did.i32", dtype="<i4")
            candidates = np.fromfile(
                compact / "behaviors/train/candidate_did.i32", dtype="<i4")
            test_click = np.fromfile(
                compact / "behaviors/test/click.i8", dtype="i1")
            np.testing.assert_array_equal(history, [3, 2])
            np.testing.assert_array_equal(candidates, [4, 3])
            np.testing.assert_array_equal(test_click, [-1])
            np.testing.assert_array_equal(membership, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
