"""Build source-defined train-seen DID membership without using labels.

The source's ``did_vocab2`` is populated from both history and candidate DIDs
in train/behaviors.tsv. This utility streams that raw file and maps raw news
IDs through the retained model DB's DID vocabulary. The prepared DB's capped
history column is deliberately not used because it cannot represent full raw
history membership.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .batch_data import connect_readonly
from .full_pipeline import DEFAULT_DB, ROOT, verified_database_identity


MEMBERSHIP_DEFINITION = (
    "unique raw DIDs appearing in train behavior history or impression candidates; no labels used"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _database_identity(db_path):
    return verified_database_identity(db_path)


def _load_did_mapping(db_path):
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT id,token FROM vocab WHERE field='did' AND id>=2 ORDER BY id"
        )
        mapping = {}
        digest = hashlib.sha256()
        expected_id = 2
        for did_id, raw_did in rows:
            did_id = int(did_id)
            raw_did = str(raw_did)
            if did_id != expected_id or raw_did in mapping:
                raise ValueError("DID vocabulary IDs are not a contiguous unique sequence from ID 2")
            mapping[raw_did] = did_id
            digest.update("{}\t{}\n".format(did_id, raw_did).encode("utf-8"))
            expected_id += 1
    finally:
        conn.close()
    if not mapping:
        raise ValueError("DID vocabulary mapping is empty")
    return mapping, digest.hexdigest()


def _behavior_dids(behaviors_path, did_mapping):
    seen = set()
    rows = history_count = candidate_count = 0
    with Path(behaviors_path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for line_no, row in enumerate(reader, 1):
            if len(row) != 5:
                raise ValueError("expected five behaviors.tsv columns at line {}".format(line_no))
            _, _, _, history_raw, impressions_raw = row
            rows += 1
            history_values = history_raw.split() if history_raw and history_raw != "-" else ()
            for raw_did in history_values:
                if raw_did not in did_mapping:
                    raise ValueError("raw train history DID absent from retained vocabulary at line {}: {}".format(
                        line_no, raw_did))
                seen.add(did_mapping[raw_did])
                history_count += 1
            candidate_values = impressions_raw.split() if impressions_raw and impressions_raw != "-" else ()
            for candidate in candidate_values:
                raw_did = candidate.split("-", 1)[0]
                if not raw_did or raw_did not in did_mapping:
                    raise ValueError("raw train candidate DID absent from retained vocabulary at line {}: {}".format(
                        line_no, raw_did))
                seen.add(did_mapping[raw_did])
                candidate_count += 1
    return seen, rows, history_count, candidate_count


def build(db_path=DEFAULT_DB, train_behaviors_path=None, out_path=None):
    if train_behaviors_path is None:
        raise ValueError("--train-behaviors must point to the original train/behaviors.tsv")
    requested_db_path = Path(db_path).expanduser()
    train_behaviors_path = Path(train_behaviors_path).expanduser().resolve(strict=True)
    out_path = Path(out_path or ROOT / "artifacts/recommendation_runs/train_did_membership.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(out_path) + ".manifest.json")

    started = time.time()
    verified_database = _database_identity(requested_db_path)
    database_identity = verified_database["fingerprint"]
    database_location = verified_database["location"]
    db_path = requested_db_path.resolve(strict=True)
    behaviors_sha = sha256_file(train_behaviors_path)
    did_mapping, mapping_sha = _load_did_mapping(db_path)
    if mapping_sha != database_identity["did_mapping_sha256"]:
        raise ValueError("raw membership DID mapping does not match the verified database mapping")
    seen, behavior_rows, history_count, candidate_count = _behavior_dids(
        train_behaviors_path, did_mapping)
    values = np.asarray(sorted(seen), dtype=np.int32)
    if len(values) and (np.any(values[1:] <= values[:-1]) or int(values[0]) < 2):
        raise ValueError("constructed membership is not sorted, unique, and free of special rows")

    temp_path = out_path.with_name(out_path.name + ".tmp")
    with temp_path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, out_path)
    manifest = {
        "status": "TRAIN_BEHAVIOR_DID_MEMBERSHIP_BUILT",
        "schema_version": 2,
        "membership_definition": MEMBERSHIP_DEFINITION,
        "source_repository_commit": "f632bdf7b1136dff142835f9fe69c8ce5f8c166d",
        "source_evidence": [
            "upstream-mind/projects/ai/mind/prepare/gen-dict.py:41-50",
            "upstream-mind/projects/ai/mind/src/util.py:71-103",
        ],
        "split": "train",
        "behaviors_path": str(train_behaviors_path),
        "behaviors_sha256": behaviors_sha,
        "behavior_rows": behavior_rows,
        "history_did_occurrences": history_count,
        "candidate_did_occurrences": candidate_count,
        "labels_used": False,
        "candidate_label_suffix_used": False,
        "database_identity": database_identity,
        "database_location_at_build": database_location,
        "did_mapping_sha256": mapping_sha,
        "unique_train_dids": int(len(values)),
        "dtype": "int32",
        "sorted_unique": True,
        "matrix_path": str(out_path.resolve()),
        "matrix_bytes": int(out_path.stat().st_size),
        "matrix_sha256": sha256_file(out_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "preprocessing_or_db_write": False,
    }
    temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--train-behaviors", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(build(args.db, args.train_behaviors, args.output), indent=2))


if __name__ == "__main__":
    main()
