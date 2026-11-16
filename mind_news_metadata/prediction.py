"""Local-only candidate prediction and keyed mean-probability ensemble."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from .full_pipeline import (DEFAULT_DB, ROOT, TRAINING_CONFIG, EXECUTION_CONFIG,
                            _candidate_mask, _json_hash, _to_tensors,
                            implementation_fingerprint, iter_ordered_batches,
                            locked_artifacts, make_model)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_digest(prefix):
    prefix = str(prefix)
    files = sorted(Path(prefix).parent.glob(Path(prefix).name + ".*"))
    h = hashlib.sha256()
    items = []
    for path in files:
        part_hash = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                part_hash.update(block)
        part = part_hash.hexdigest()
        items.append({"path": str(path), "sha256": part, "bytes": path.stat().st_size})
        h.update(path.name.encode()); h.update(bytes.fromhex(part))
    return {"sha256": h.hexdigest(), "files": items}


def load_run(run_dir, db_path=DEFAULT_DB):
    run_dir = Path(run_dir)
    requested_db_path = Path(db_path).expanduser()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    current_identity = locked_artifacts(requested_db_path)
    current_inputs = current_identity["fingerprint"]
    if config.get("config", {}).get("inputs") != current_inputs:
        raise ValueError("checkpoint input DB/mapping/artifact fingerprint differs from current files")
    if config.get("config", {}).get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("checkpoint source code fingerprint differs from current implementation")
    dynamic = {"effective_batch_size", "microbatch_size"}
    for key, value in TRAINING_CONFIG.items():
        if key in dynamic:
            continue
        if config.get("config", {}).get(key) != value:
            raise ValueError("checkpoint model/protocol contract differs at {}".format(key))
    db_path = requested_db_path.resolve(strict=True)
    model, store, word_manifest = make_model(db_path)
    if Path(store.db_path).resolve(strict=True) != db_path:
        raise ValueError("inference loader opened a different database than the fingerprinted DB")
    sample, _, _ = store.batch("dev", limit=1)
    model(_to_tensors(sample), training=False)
    prefix = tf.train.latest_checkpoint(str(run_dir / "checkpoints"))
    if prefix is None:
        raise ValueError("no TensorFlow checkpoint found")
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(prefix)
    status.assert_existing_objects_matched()
    status.expect_partial()  # optimizer slots are intentionally unused for inference
    config["current_database_location"] = current_identity["location"]
    return model, store, prefix, config


def train_did_set(path, expected_database):
    path = Path(path)
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("train DID membership provenance manifest is missing: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "TRAIN_BEHAVIOR_DID_MEMBERSHIP_BUILT":
        raise ValueError("train DID membership was not built from the full train behaviors stream")
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError("train DID membership artifact uses a legacy/incompatible schema")
    if manifest.get("membership_definition") != "unique raw DIDs appearing in train behavior history or impression candidates; no labels used":
        raise ValueError("train DID membership definition is incompatible")
    identity = manifest.get("database_identity", {})
    for key in ("bytes", "sha256", "schema_sha256", "mapping_sha256", "did_mapping_sha256"):
        if identity.get(key) != expected_database.get(key):
            raise ValueError("train DID membership DB identity differs at {}".format(key))
    if manifest.get("did_mapping_sha256") != expected_database.get("did_mapping_sha256"):
        raise ValueError("train DID membership DID mapping checksum differs from the active DB")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 1 or arr.dtype != np.int32:
        raise ValueError("train DID membership artifact must be sorted int32 vector")
    if len(arr) and (np.any(arr[1:] <= arr[:-1]) or int(arr[0]) < 2):
        raise ValueError("train DID membership IDs must be sorted, unique, and exclude PAD/UNK")
    if manifest.get("matrix_sha256") != file_sha256(path):
        raise ValueError("train DID membership checksum differs from its provenance manifest")
    return set(map(int, arr))


def predict(run_dir, split="test", profile="test", train_dids_path=None,
            output=None, db_path=DEFAULT_DB,
            batch_size=EXECUTION_CONFIG.evaluation_batch_size, max_candidates=None):
    model, store, prefix, run_config = load_run(run_dir, db_path)
    train_dids = (train_did_set(train_dids_path, run_config["config"]["inputs"]["database"])
                  if train_dids_path else None)
    if split == "test" and profile == "test" and train_dids is None:
        raise ValueError("test candidate masking requires membership built from raw train behavior history and candidates")
    out_path = Path(output or Path(run_dir) / "predictions-{}.csv.gz".format(split))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    count = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "impression_id", "candidate_position", "news_id", "probability"])
        for features, labels, meta in iter_ordered_batches(store, split, batch_size):
            if max_candidates is not None and count >= max_candidates:
                break
            take = len(meta) if max_candidates is None else min(len(meta), max_candidates - count)
            features = {k: v[:take] for k, v in features.items()}
            labels = {k: v[:take] for k, v in labels.items()}
            meta = meta[:take]
            if split == "test" and bool(labels["labeled"].any()):
                raise ValueError("test labels must remain null")
            if split != "test" and not bool(labels["labeled"].all()):
                raise ValueError("labeled split contains unlabeled rows")
            features = _candidate_mask(features, meta, profile, train_dids)
            probs = model(_to_tensors(features), training=False)["probabilities"].numpy()
            if not np.isfinite(probs).all():
                raise ValueError("non-finite probability")
            for row, prob in zip(meta, probs):
                writer.writerow([split, row["impression_id"], row["position"], row["news_id"], "{:.9g}".format(float(prob))])
            count += take
    os.replace(tmp, out_path)
    manifest = {"status": "LOCAL_PREDICTIONS_WRITTEN" if max_candidates is None else "PILOT_ONLY_PARTIAL_PREDICTIONS",
                "full_split": max_candidates is None, "split": split,
                "mask_profile": profile, "rows": count, "output": str(out_path),
                "checkpoint_prefix": prefix, "checkpoint": checkpoint_digest(prefix),
                "run_config_sha256": run_config["config_sha256"],
                "model_contract_sha256": _json_hash(TRAINING_CONFIG),
                "input_artifact_fingerprint": run_config["config"]["inputs"],
                "database_location": run_config.get("current_database_location"),
                "implementation_fingerprint": run_config["config"]["implementation_fingerprint"],
                "field_order": list(TRAINING_CONFIG["field_order"]),
                "output_bytes": out_path.stat().st_size,
                "output_sha256": file_sha256(out_path),
                "test_metrics": "unavailable; labels are null" if split == "test" else None,
                "candidate_subsampling": False, "uploaded": False}
    manpath = Path(str(out_path) + ".manifest.json")
    manpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _read_prediction(path):
    stream = gzip.open(path, "rt", encoding="utf-8", newline="")
    reader = csv.DictReader(stream)
    if reader.fieldnames != ["split", "impression_id", "candidate_position", "news_id", "probability"]:
        stream.close()
        raise ValueError("prediction schema mismatch: {}".format(path))
    return stream, reader


def ensemble(inputs, output):
    if len(inputs) < 2:
        raise ValueError("ensemble requires at least two separately trained model outputs")
    sidecars = []
    for path in inputs:
        manifest_path = Path(str(path) + ".manifest.json")
        if not manifest_path.is_file():
            raise ValueError("ensemble input missing prediction manifest: {}".format(manifest_path))
        sidecars.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if any(m.get("status") != "LOCAL_PREDICTIONS_WRITTEN" or not m.get("full_split") for m in sidecars):
        raise ValueError("ensemble input is not a completed local prediction artifact")
    if len({m.get("checkpoint", {}).get("sha256") for m in sidecars}) != len(sidecars):
        raise ValueError("ensemble inputs must come from distinct model checkpoints")
    if len({json.dumps(m.get("input_artifact_fingerprint"), sort_keys=True) for m in sidecars}) != 1:
        raise ValueError("ensemble input data/mapping fingerprints differ")
    if len({m.get("model_contract_sha256") for m in sidecars}) != 1:
        raise ValueError("ensemble model contracts differ")
    if len({json.dumps(m.get("implementation_fingerprint"), sort_keys=True) for m in sidecars}) != 1:
        raise ValueError("ensemble implementations differ")
    readers = [_read_prediction(p) for p in inputs]
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    rows = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["split", "impression_id", "candidate_position", "news_id", "mean_probability", "rank"])
        current_imp = None
        bucket = []
        sentinels = [object() for _ in readers]
        iterators = [iter(r) for _, r in readers]
        while True:
            records = [next(it, sentinel) for it, sentinel in zip(iterators, sentinels)]
            if all(rec is sentinel for rec, sentinel in zip(records, sentinels)):
                break
            if any(rec is sentinel for rec, sentinel in zip(records, sentinels)):
                raise ValueError("prediction key-set length mismatch")
            keys = [(r["split"], r["impression_id"], int(r["candidate_position"]), r["news_id"]) for r in records]
            if any(key != keys[0] for key in keys[1:]):
                raise ValueError("prediction candidate key mismatch: {}".format(keys))
            probs = [float(r["probability"]) for r in records]
            if not all(np.isfinite(probs)):
                raise ValueError("non-finite ensemble component")
            imp_key = keys[0][:2]
            if current_imp is not None and imp_key != current_imp:
                bucket.sort(key=lambda x: x[0])
                rank_order = sorted(bucket, key=lambda x: (-x[1], x[0]))
                rank_by_pos = {x[0]: rank + 1 for rank, x in enumerate(rank_order)}
                for pos, prob, key in bucket:
                    writer.writerow([key[0], key[1], pos, key[3], "{:.9g}".format(prob), rank_by_pos[pos]])
                rows += len(bucket); bucket = []
            current_imp = imp_key
            bucket.append((keys[0][2], float(np.mean(probs)), keys[0]))
        if bucket:
            bucket.sort(key=lambda x: x[0])
            rank_order = sorted(bucket, key=lambda x: (-x[1], x[0]))
            rank_by_pos = {x[0]: rank + 1 for rank, x in enumerate(rank_order)}
            for pos, prob, key in bucket:
                writer.writerow([key[0], key[1], pos, key[3], "{:.9g}".format(prob), rank_by_pos[pos]])
            rows += len(bucket)
    for stream, _ in readers:
        stream.close()
    os.replace(tmp, out)
    manifest = {"status": "LOCAL_MEAN_PROBABILITY_ENSEMBLE", "model_count": len(inputs),
                "inputs": [str(x) for x in inputs], "rows": rows,
                "input_checkpoints": [m["checkpoint"]["sha256"] for m in sidecars],
                "model_contract_sha256": sidecars[0]["model_contract_sha256"],
                "implementation_fingerprint": sidecars[0]["implementation_fingerprint"],
                "input_artifact_fingerprint": sidecars[0]["input_artifact_fingerprint"],
                "key": ["split", "impression_id", "candidate_position", "news_id"],
                "rank_tie_break": "candidate_position ascending", "output": str(out), "uploaded": False}
    manifest["output_bytes"] = out.stat().st_size
    manifest["output_sha256"] = file_sha256(out)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
