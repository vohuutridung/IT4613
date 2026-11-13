"""Small isolated regressions for resume fingerprints, UID cutoff, and SQL order."""
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from mind_data_pipeline.config import CONFIG
from mind_data_pipeline.data import Stage01Store, uid_embedding_id
from mind_data_pipeline.preprocess import build_counters, prepare


def _write_fixture(root):
    for split in ("train", "dev", "test"):
        base = root / ("MINDlarge_" + split)
        base.mkdir(parents=True)
        news = []
        for i in range(1, 5):
            news.append("N{}\tcat\tsub\tTitle {}\tAbstract {}\turl\t[]\t[]\n".format(i, i, i))
        (base / "news.tsv").write_text("".join(news), encoding="utf-8")
        if split == "train":
            rows = [
              "imp1\tu1\t2020-01-01T00:00:00Z\t\tN1-1 N2-0 N1-0\n",
              "imp2\tu2\t2020-01-01T00:01:00Z\tN1\tN3-1\n",
            ]
        else:
            rows = ["{}\tu{}\t2020-01-01T00:00:00Z\t\tN4\n".format(split, split)]
        (base / "behaviors.tsv").write_text("".join(rows), encoding="utf-8")


def _snapshot(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            result[str(path.relative_to(root))] = (path.stat().st_size, h)
    return result


def run():
    with tempfile.TemporaryDirectory(prefix="mind-data-regression-") as tmp:
        root = Path(tmp) / "raw"
        out = Path(tmp) / "out"
        _write_fixture(root)
        prepare(root, out, resume=False)

        # Same source/config passes preflight and resumes without rebuilding vocab.
        prepare(root, out, resume=True)
        store = Stage01Store(out / "stage01.sqlite")
        expected = [("imp1", 0), ("imp1", 1), ("imp1", 2), ("imp2", 0)]
        observed = []
        offset = 0
        while True:
            _, _, metadata = store.batch("train", limit=2, offset=offset)
            if metadata is None:
                break
            observed.extend((item["impression_id"], item["position"]) for item in metadata)
            offset += len(metadata)
        if observed != expected:
            raise AssertionError("ORDER BY/offset sequence mismatch: {}".format(observed))

        # Changed source must be rejected before any stage artifact is modified.
        before = _snapshot(out)
        behavior = root / "MINDlarge_train" / "behaviors.tsv"
        original = behavior.read_text(encoding="utf-8")
        behavior.write_text(original + "imp3\tu3\t2020-01-01T00:02:00Z\t\tN4\n", encoding="utf-8")
        try:
            try:
                prepare(root, out, resume=True)
            except ValueError as exc:
                if "fingerprint differs" not in str(exc):
                    raise
            else:
                raise AssertionError("changed source unexpectedly resumed")
            if _snapshot(out) != before:
                raise AssertionError("source mismatch changed an output artifact")
        finally:
            behavior.write_text(original, encoding="utf-8")

        # Changed config is likewise rejected before writes.
        import mind_data_pipeline.preprocess as pp
        before = _snapshot(out)
        old_config = pp.CONFIG
        try:
            class ChangedConfig(object):
                def __init__(self, base):
                    self.base = base
                    self.test_start = base.test_start
                    self.min_uid_count = base.min_uid_count
                    self.tokenizer_local = base.tokenizer_local
                    self.tokenizer_repo = base.tokenizer_repo
                    self.tokenizer_revision = base.tokenizer_revision
                    self.source_commit = base.source_commit
                def as_dict(self):
                    value = old_config.as_dict()
                    value["regression_only_changed_parameter"] = True
                    return value
            pp.CONFIG = ChangedConfig(old_config)
            try:
                prepare(root, out, resume=True)
            except ValueError as exc:
                if "fingerprint differs" not in str(exc):
                    raise
            else:
                raise AssertionError("changed config unexpectedly resumed")
            if _snapshot(out) != before:
                raise AssertionError("config mismatch changed an output artifact")
        finally:
            pp.CONFIG = old_config

        # A conflicting duplicate news row is an input error, not a silent
        # first-row choice. This synthetic conflict is isolated from the real raw data.
        dev_news = root / "MINDlarge_dev" / "news.tsv"
        dev_original = dev_news.read_text(encoding="utf-8")
        dev_news.write_text(dev_original.replace("Title 1", "Conflicting title", 1), encoding="utf-8")
        try:
            try:
                build_counters(root)
            except ValueError as exc:
                if "conflicting duplicate news metadata" not in str(exc):
                    raise
            else:
                raise AssertionError("conflicting duplicate news metadata was silently accepted")
        finally:
            dev_news.write_text(dev_original, encoding="utf-8")

        if uid_embedding_id(0, 10) != 0 or uid_embedding_id(9, 10) != 9:
            raise AssertionError("UID valid lower/boundary mapping failed")
        if uid_embedding_id(10, 10) != CONFIG.unk_id or uid_embedding_id(100, 10) != CONFIG.unk_id:
            raise AssertionError("UID out-of-table IDs must map to UNK")
        result = {"status": "PASS", "same_config_resume": True,
                "source_change_rejected_without_output_changes": True,
                "config_change_rejected_without_output_changes": True,
                "conflicting_duplicate_news_rejected": True,
                "ordered_batches_no_missing_or_duplicate_candidates": observed,
                "uid_id_boundary": {"0": uid_embedding_id(0, 10), "9_of_10": uid_embedding_id(9, 10),
                                    "10_of_10": uid_embedding_id(10, 10)}}
        full_db = Path("artifacts/mind_large_prepared/stage01.sqlite")
        if full_db.is_file():
            full_store = Stage01Store(full_db)
            rows = full_store.uid_embedding_rows()
            conn = sqlite3.connect("file:{}?mode=ro".format(full_db.resolve()), uri=True)
            try:
                boundary = conn.execute("SELECT id,source_count FROM vocab WHERE field='uid' AND id IN (?,?) ORDER BY id",
                                        (rows - 1, rows)).fetchall()
                test_only = conn.execute("SELECT id,source_count FROM vocab WHERE field='uid' AND source_count=1 ORDER BY id LIMIT 1").fetchone()
            finally:
                conn.close()
            if boundary != [(rows - 1, CONFIG.test_start + CONFIG.min_uid_count),
                            (rows, CONFIG.test_start + CONFIG.min_uid_count - 1)]:
                raise AssertionError("full DB UID cutoff boundary mismatch: {}".format(boundary))
            if not test_only or uid_embedding_id(test_only[0], rows, CONFIG.unk_id) != CONFIG.unk_id:
                raise AssertionError("test-only UID was not routed to UNK")
            result["full_database_uid_cutoff"] = {
              "threshold": CONFIG.test_start + CONFIG.min_uid_count,
              "embedding_rows": rows,
              "largest_valid": {"id": boundary[0][0], "source_count": boundary[0][1]},
              "first_excluded": {"id": boundary[1][0], "source_count": boundary[1][1], "maps_to": 1},
              "test_only_uid": {"id": test_only[0], "source_count": test_only[1], "maps_to": 1}
            }
        return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
