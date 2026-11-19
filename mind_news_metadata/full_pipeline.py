"""Train/evaluate the authorized 15-field no-body MIND reimplementation.

No feature interaction/prediction code from the private upstream dependency is
called. This module performs local training and prediction only.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .batch_data import connect_readonly, make_batch_store
from .checkpoint import (CHECKPOINT_SCHEMA_VERSION, atomic_write, latest_checkpoint,
                         load_checkpoint, save_checkpoint)
from .field_encoder import (TitleAbstractMetadataEncoder, load_glove_artifact,
                            load_pretrained_matrix, sha256_file)
from .recommendation_model import NoBodyRecommendationModel
from .training_runtime import batch_layout, effective_batches, prefetch_batches
from .distributed_training import DistributedTrainer, autocast, to_tensors, training_devices
from .recommendation_protocol import (aggregate_impression_metrics,
                                      learning_rate_at, partition_for_impression)
from mind_config import ENCODER_CONFIG, EXECUTION_CONFIG, TRAINING_CONFIG


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED = ROOT / "artifacts/mind_large_prepared"
DEFAULT_DB = (DEFAULT_PREPARED if
              (DEFAULT_PREPARED / "compact/compact_manifest.json").is_file()
              else DEFAULT_PREPARED / "stage01.sqlite")
DEFAULT_OUT = ROOT / "artifacts/recommendation_runs"
FIELD_ORDER = TitleAbstractMetadataEncoder.FIELD_ORDER


def _json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical_hash(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def implementation_fingerprint():
    paths = ["mind_config.py", "configs/mind_no_body_15field.json",
             "mind_data_pipeline/config.py", "mind_news_metadata/batch_data.py",
             "mind_news_metadata/compact_data.py",
             "mind_news_metadata/training_runtime.py", "mind_news_metadata/distributed_training.py",
             "mind_news_metadata/checkpoint.py",
             "mind_news_metadata/encoder_config.py",
             "mind_news_metadata/field_encoder.py", "mind_news_metadata/recommendation_model.py",
             "mind_news_metadata/recommendation_protocol.py", "mind_news_metadata/full_pipeline.py",
             "mind_news_metadata/prediction.py", "mind_news_metadata/submission_format.py",
             "mind_news_metadata/build_train_did_membership.py"]
    return {path: sha256_file(ROOT / path) for path in paths}


def _file_signature(stat_result):
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
            stat_result.st_mtime_ns, stat_result.st_ctime_ns)


def _sha256_stable_file(path):
    """Hash a file and reject replacement or mutation during the read."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current_path = path.stat()
    if (_file_signature(before) != _file_signature(after)
            or _file_signature(after) != _file_signature(current_path)):
        raise ValueError("database changed while its content hash was being verified: {}".format(path))
    return digest.hexdigest(), after


def verified_database_identity(db_path):
    """Return path-independent DB compatibility hashes and separate trace paths.

    Sidecar hashes are expectations. The DB bytes, SQL schema, and mapping are
    read and hashed at invocation time; no size-only cache is used.
    """
    # Keep the caller's absolute-but-unresolved path as well as its physical
    # target. Kaggle exposes immutable inputs through a working-directory
    # overlay: stage01.sqlite and artifact_manifest.json are sibling symlinks,
    # while resolving only the DB symlink moves it back to /kaggle/input where
    # that sibling manifest is not present.
    requested_db = Path(db_path).expanduser()
    if not requested_db.is_absolute():
        requested_db = Path.cwd() / requested_db
    requested_db = Path(os.path.abspath(str(requested_db)))
    if requested_db.is_dir():
        resolved_root = requested_db.resolve(strict=True)
        manifest_path = resolved_root / "compact/compact_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("compact input manifest is missing: {}".format(manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "MIND_COMPACT_MEMMAP_READY":
            raise ValueError("compact input status is incompatible")
        observed = {}
        for row in manifest.get("files", []):
            rel = Path(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("unsafe compact manifest path: {}".format(rel))
            path = resolved_root / "compact" / rel
            actual, stat = _sha256_stable_file(path)
            if int(row["bytes"]) != stat.st_size or row["sha256"] != actual:
                raise ValueError("compact input file differs: {}".format(path))
            observed[rel.as_posix()] = actual
        payload = hashlib.sha256(json.dumps(
            observed, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        identity = manifest.get("identity", {})
        observed_bytes = sum(int(row["bytes"]) for row in manifest.get("files", []))
        if (payload != identity.get("sha256")
                or observed_bytes != int(identity.get("bytes", -1))):
            raise ValueError("compact input aggregate fingerprint differs")
        layout = json.loads(
            (resolved_root / "compact/layout.json").read_text(encoding="utf-8"))
        mapping_digest = hashlib.sha256()
        for field in ("uid", "did", "cat", "sub_cat", "entity", "entity_type", "word"):
            vocab_path = resolved_root / "vocab" / (field + ".tsv")
            actual_vocab_sha, _ = _sha256_stable_file(vocab_path)
            if actual_vocab_sha != layout["vocab"]["tsv_sha256"][field]:
                raise ValueError("compact vocabulary differs for {}".format(field))
            mapping_digest.update(field.encode("utf-8") + b"\0")
            mapping_digest.update(bytes.fromhex(actual_vocab_sha))
        did_digest = hashlib.sha256()
        with (resolved_root / "vocab/did.tsv").open(encoding="utf-8") as stream:
            for line in stream:
                did_id, raw_did, *_ = line.rstrip("\n").split("\t")
                if int(did_id) >= 2:
                    did_digest.update(
                        "{}\t{}\n".format(int(did_id), raw_did).encode("utf-8"))
        if (mapping_digest.hexdigest() != identity.get("mapping_sha256")
                or did_digest.hexdigest() != identity.get("did_mapping_sha256")):
            raise ValueError("compact vocabulary identity differs")
        return {
            "fingerprint": {
                key: identity[key] for key in (
                    "bytes", "sha256", "schema_sha256",
                    "mapping_sha256", "did_mapping_sha256")
            },
            "location": {
                "storage_format": manifest.get("storage_format"),
                "resolved_database_path": str(resolved_root),
                "resolved_manifest_path": str(manifest_path),
                "manifest_sha256_for_trace": sha256_file(manifest_path),
            },
        }
    if not requested_db.is_file():
        raise FileNotFoundError("database is not a regular file: {}".format(requested_db))
    resolved_db = requested_db.resolve(strict=True)
    if not resolved_db.is_file():
        raise FileNotFoundError("database is not a regular file: {}".format(resolved_db))

    db_manifest_path = requested_db.parent / "artifact_manifest.json"
    if not db_manifest_path.is_file() and resolved_db.parent != requested_db.parent:
        db_manifest_path = resolved_db.parent / "artifact_manifest.json"
    if not db_manifest_path.is_file():
        raise ValueError(
            "database sidecar artifact_manifest.json is missing beside requested path {} "
            "and resolved target {}".format(requested_db, resolved_db))
    db_manifest = json.loads(db_manifest_path.read_text(encoding="utf-8"))
    try:
        db_relative = requested_db.relative_to(db_manifest_path.parent).as_posix()
    except ValueError:
        try:
            db_relative = resolved_db.relative_to(db_manifest_path.parent.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(
                "database path is not covered by its sidecar manifest: {}".format(db_manifest_path)
            ) from exc
    db_entries = {item["path"]: item for item in db_manifest.get("artifacts", [])}
    db_entry = db_entries.get(db_relative)
    if db_entry is None:
        raise ValueError("database manifest does not cover the actual DB path {}".format(db_relative))
    actual_db_sha256, db_stat = _sha256_stable_file(resolved_db)
    if int(db_entry.get("bytes", -1)) != db_stat.st_size:
        raise ValueError("database size differs from its manifest: {}".format(resolved_db))
    if db_entry.get("sha256") != actual_db_sha256:
        raise ValueError("database content checksum differs from its manifest: {}".format(resolved_db))

    reconciliation = db_manifest.get("legacy_artifact_reconciliation", {})
    expected_schema_sha = (db_manifest.get("schema_sha256")
                           or reconciliation.get("schema_sha256"))
    expected_vocab_hashes = reconciliation.get("vocabulary_sha256", {})
    artifact_entries = {item["path"]: item for item in db_manifest.get("artifacts", [])}
    vocab_fields = ("uid", "did", "cat", "sub_cat", "entity", "entity_type", "word")

    if not expected_schema_sha:
        resolved_config_path = db_manifest_path.parent / "resolved_config.json"
        if resolved_config_path.is_file():
            resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
            preprocessing_fingerprint = resolved_config.get("preprocessing_fingerprint", {})
            expected_schema_sha = preprocessing_fingerprint.get("schema_sha256")
    if not expected_schema_sha:
        raise ValueError("database manifest lacks an expected SQL schema fingerprint")

    conn = connect_readonly(resolved_db)
    try:
        schema_rows = conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        ).fetchall()
        actual_schema_sha = _canonical_hash(schema_rows)
        if actual_schema_sha != expected_schema_sha:
            raise ValueError("database SQL schema differs from its expected fingerprint")

        mapping_digest = hashlib.sha256()
        for field in vocab_fields:
            rel = "vocab/{}.tsv".format(field)
            path = db_manifest_path.parent / rel
            expected = expected_vocab_hashes.get(rel)
            if expected is None:
                expected = artifact_entries.get(rel, {}).get("sha256")

            # The database stores every vocabulary row. Reconstruct the exact
            # Stage 01 TSV byte stream from that table so a transfer package
            # may omit standalone uid/did/category maps without weakening
            # provenance: the DB itself has already matched its manifest SHA.
            exported_hash = hashlib.sha256()
            expected_id = 0
            row_count = 0
            for row_id, token, source_count in conn.execute(
                    "SELECT id,token,source_count FROM vocab WHERE field=? ORDER BY id", (field,)):
                row_id = int(row_id)
                token = str(token)
                if row_id != expected_id:
                    raise ValueError("database vocabulary IDs are not contiguous for {} at {}".format(
                        field, row_id))
                if field == "word":
                    line = "{}\t{}\t\n".format(row_id, token)
                else:
                    if source_count is None:
                        raise ValueError("database vocabulary count is missing for {} ID {}".format(
                            field, row_id))
                    line = "{}\t{}\t{}\n".format(row_id, token.replace("\t", " "),
                                                   int(source_count))
                exported_hash.update(line.encode("utf-8"))
                expected_id += 1
                row_count += 1
            if not row_count:
                raise ValueError("database vocabulary is empty for {}".format(field))
            reconstructed_hash = exported_hash.hexdigest()

            if path.is_file():
                actual_hash, vocab_stat = _sha256_stable_file(path)
                if _file_signature(vocab_stat) != _file_signature(path.stat()):
                    raise ValueError("database vocabulary mapping changed during fingerprint verification: {}".format(
                        path))
                if actual_hash != reconstructed_hash:
                    raise ValueError("standalone vocabulary file differs from DB rows: {}".format(path))
            else:
                actual_hash = reconstructed_hash

            if expected is not None and actual_hash != expected:
                raise ValueError("database vocabulary mapping checksum differs for {}".format(rel))
            mapping_digest.update(field.encode("utf-8") + b"\0")
            mapping_digest.update(bytes.fromhex(actual_hash))
        for row in conn.execute(
                "SELECT field,id,token,source_count FROM vocab ORDER BY field,id"):
            mapping_digest.update(json.dumps(
                row, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8") + b"\n")
        did_digest = hashlib.sha256()
        for did_id, raw_did in conn.execute(
                "SELECT id,token FROM vocab WHERE field='did' AND id>=2 ORDER BY id"):
            did_digest.update("{}\t{}\n".format(int(did_id), str(raw_did)).encode("utf-8"))
    finally:
        conn.close()

    # Bind the SQLite reads to the exact file that was hashed above. This is
    # metadata-only (no second multi-GB hash) and rejects replacement or
    # mutation during identity construction, including same-size rewrites.
    if _file_signature(db_stat) != _file_signature(resolved_db.stat()):
        raise ValueError("database changed while schema/mapping identity was being verified: {}".format(
            resolved_db))

    actual_mapping_sha = mapping_digest.hexdigest()
    expected_mapping_sha = db_manifest.get("mapping_sha256")
    if expected_mapping_sha is None:
        resolved_config_path = db_manifest_path.parent / "resolved_config.json"
        if resolved_config_path.is_file():
            resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
            expected_mapping_sha = resolved_config.get("preprocessing_fingerprint", {}).get(
                "mapping_sha256")
    if expected_mapping_sha and actual_mapping_sha != expected_mapping_sha:
        raise ValueError("database vocabulary mapping differs from its expected fingerprint")

    return {
        "fingerprint": {
            "bytes": int(db_stat.st_size),
            "sha256": actual_db_sha256,
            "schema_sha256": actual_schema_sha,
            "mapping_sha256": actual_mapping_sha,
            "did_mapping_sha256": did_digest.hexdigest(),
        },
        "location": {
            "resolved_database_path": str(resolved_db),
            "resolved_manifest_path": str(db_manifest_path.resolve()),
            "manifest_sha256_for_trace": sha256_file(db_manifest_path),
        },
    }


def locked_artifacts(db_path):
    """Bind a run to content compatibility while keeping physical paths trace-only."""
    database = verified_database_identity(db_path)

    input_path = Path(db_path)
    out = (input_path.resolve(strict=True) if input_path.is_dir()
           else ROOT / "artifacts/mind_large_prepared")
    manifest_path = out / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [out / "embeddings/word_glove_title_abstract_l2.npy",
             out / "embeddings/word_glove_title_abstract_l2.manifest.json",
             out / "embeddings/entity.npy",
             out / "embeddings/entity_matrix_manifest.json",
             out / "vocab/word.tsv", out / "vocab/entity.tsv"]
    entries = {item["path"]: item for item in manifest.get("artifacts", [])}
    locked = {}
    for p in paths:
        rel = str(p.relative_to(out))
        item = entries.get(rel)
        if item is None or not p.is_file() or p.stat().st_size != int(item["bytes"]):
            raise ValueError("existing artifact manifest does not cover current input {}".format(rel))
        # Compatibility is content-based and portable across local/Kaggle
        # install roots. Physical paths belong only in the trace section.
        locked["artifacts/mind_large_prepared/{}".format(rel)] = {
            "bytes": int(item["bytes"]), "sha256": item["sha256"]}
    return {
        "fingerprint": {
            "database": database["fingerprint"],
            "artifacts": locked,
        },
        "location": dict(database["location"],
                         prepared_artifact_root=str(out.resolve()),
                         prepared_artifact_manifest_path=str(manifest_path.resolve()),
                         prepared_artifact_manifest_sha256_for_trace=sha256_file(manifest_path)),
    }


def make_model(db_path=DEFAULT_DB):
    store = make_batch_store(db_path)
    sizes = store.vocab_sizes()
    input_path = Path(db_path)
    prepared = (input_path.resolve(strict=True) if input_path.is_dir()
                else ROOT / "artifacts/mind_large_prepared")
    emb = prepared / "embeddings"
    vocab = prepared / "vocab"
    entity_manifest = json.loads((emb / "entity_matrix_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(emb / "entity.npy") != entity_manifest["matrix_sha256"]:
        raise ValueError("entity matrix checksum differs from its provenance manifest")
    entity_mapping_path = Path(entity_manifest["mapping_path"])
    if not entity_mapping_path.is_absolute():
        entity_mapping_path = ROOT / entity_mapping_path
    entity_values = load_pretrained_matrix(
        emb / "entity.npy", sizes["entity"], ENCODER_CONFIG.embedding_dim,
        entity_manifest["mapping_sha256"], sha256_file(entity_mapping_path))
    # Entity type table is small and initialized by the active encoder; only the
    # entity vector table is the derived MIND artifact.
    word_values, word_manifest = load_glove_artifact(
        emb / "word_glove_title_abstract_l2.npy",
        emb / "word_glove_title_abstract_l2.manifest.json",
        vocab / "word.tsv", sizes["word"], ENCODER_CONFIG.embedding_dim)
    encoder = TitleAbstractMetadataEncoder(sizes, word_matrix=word_values,
                                            entity_matrix=entity_values)
    return NoBodyRecommendationModel(encoder), store, word_manifest


def _candidate_mask(features, metadata, profile, train_dids=None,
                    unk_id=ENCODER_CONFIG.unk_id,
                    base_seed=TRAINING_CONFIG["dev_mask_seed"]):
    """Mask only candidate DID IDs, leaving the content lookup and other fields intact."""
    did = features["did"].copy()
    if profile == "train":
        return features
    if profile == "dev_new_did_masked":
        if train_dids is None:
            raise ValueError("dev_new_did_masked requires source-defined train behavior DID membership")
        for i, row in enumerate(metadata):
            # Report-based choice: mask only DIDs absent from train. The pinned
            # executable util.mask_dids(valid) masks all DIDs instead; see the
            # config/source matrix for this intentional source conflict.
            # Membership is defined by source did_vocab2: IDs found in either
            # train behavior history or train impression candidates.
            if int(did[i]) in train_dids:
                continue
            # A draw is keyed per candidate occurrence, not per distinct DID.
            token = "{}\0{}\0{}".format(base_seed, row["impression_id"], row["position"]).encode()
            value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") / float(1 << 64)
            if value < TRAINING_CONFIG["dev_mask_ratio"]:
                did[i] = unk_id
    elif profile == "dev_raw":
        pass
    elif profile == "test" or profile == "test_all_mask":
        if profile == "test_all_mask":
            did[:] = unk_id
        else:
            if train_dids is None:
                raise ValueError("test inference requires train candidate DID membership")
            did = np.asarray([v if int(v) in train_dids else unk_id for v in did], dtype=np.int32)
    else:
        raise ValueError("unknown DID mask profile {}".format(profile))
    copied = dict(features)
    copied["did"] = did
    return copied


def iter_train_batches(store, partition, seed=TRAINING_CONFIG["partition_base_seed"],
                       impression_group=EXECUTION_CONFIG.training_impression_group_size,
                       max_impressions=None):
    """Yield sampled arrays in deterministic shuffled-impression order."""
    ids = store.impression_ids("train")
    rng = np.random.RandomState(seed)
    order = ids[rng.permutation(len(ids))]
    if max_impressions is not None:
        order = order[:int(max_impressions)]
    for start in range(0, len(order), int(impression_group)):
        f, y, metadata = store.batch(
            "train", impression_ids=order[start:start + int(impression_group)],
            partition=partition, partition_seed=seed, include_metadata=True)
        if f is None:
            if y is not None or metadata is not None:
                raise ValueError("empty training batch returned inconsistent labels/metadata")
            continue
        if y is None or metadata is None or len(metadata) != len(y["click"]):
            raise ValueError("training features, labels, and cursor metadata are not aligned")
        candidate_ids = [int(row["candidate_id"]) for row in metadata]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("training batch contains duplicate candidate IDs")
        yield f, y, metadata


def count_partition_candidates(store, partition, seed=TRAINING_CONFIG["partition_base_seed"],
                               impression_group=EXECUTION_CONFIG.partition_count_group_size,
                               max_impressions=None):
    """Exact train-only selector count for fixing the one-epoch LR schedule."""
    ids = store.impression_ids("train")
    rng = np.random.RandomState(seed)
    order = ids[rng.permutation(len(ids))]
    if max_impressions is not None:
        order = order[:int(max_impressions)]
    if len(order) >= 20000:
        print("counting selected train candidates for partition {} ({} impressions)".format(
            int(partition), len(order)), flush=True)
    counter = getattr(store, "count_selected_partition", None)
    if counter is not None:
        positives, negatives, impression_count = counter(order, partition, seed)
    else:
        positives = negatives = impression_count = 0
        group_starts = range(0, len(order), int(impression_group))
        if len(order) >= 20000:
            group_count = (len(order) + int(impression_group) - 1) // int(impression_group)
            group_starts = tqdm(
                group_starts, total=group_count, desc="count train partition {}".format(int(partition)),
                unit="group", mininterval=1.0, dynamic_ncols=True)
        for start in group_starts:
            part_ids = order[start:start + int(impression_group)].tolist()
            by_imp = store.candidate_groups("train", part_ids)
            for iid in part_ids:
                group = by_imp.get(iid)
                if group is None:
                    continue
                if len(group["click"]) == 0:
                    continue
                if not bool(np.asarray(group["labeled"]).all()):
                    raise ValueError("unlabeled candidate in train partition count")
                keep = partition_for_impression(
                    group["click"], group["positions"], group["impression_id"],
                    partition, partitions=TRAINING_CONFIG["partition_count"],
                    base_seed=seed, split="train")
                selected = np.asarray(group["click"])[keep]
                positives += sum(int(x) == 1 for x in selected)
                negatives += sum(int(x) == 0 for x in selected)
                impression_count += 1
    return {"partition": int(partition), "impressions_with_candidates": impression_count,
            "positives": positives, "negatives": negatives,
            "selected_candidates": positives + negatives,
            "partition_count": TRAINING_CONFIG["partition_count"], "base_seed": int(seed),
            "max_impressions": max_impressions}


def iter_ordered_batches(store, split, batch_size=EXECUTION_CONFIG.evaluation_batch_size):
    """Keyset stream for unsampled evaluation/inference, no OFFSET scans."""
    after = (0, -1)
    while True:
        f, y, m = store.batch(split, limit=batch_size, after_key=after)
        if f is None:
            return
        yield f, y, m
        after = (int(m[-1]["line_no"]), int(m[-1]["position"]))


def _candidate_cursor_key(row):
    required = ("candidate_id", "impression_id", "position", "line_no", "split")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError("candidate cursor metadata is missing keys: {}".format(missing))
    return json.dumps({
        "candidate_id": int(row["candidate_id"]),
        "impression_id": str(row["impression_id"]),
        "candidate_position": int(row["position"]),
        "impression_line_no": int(row["line_no"]),
        "split": str(row["split"]),
    }, sort_keys=True, separators=(",", ":"))


def run_train(partition=TRAINING_CONFIG["default_partition"], max_impressions=None,
              effective_batch=TRAINING_CONFIG["effective_batch_size"],
              microbatch=TRAINING_CONFIG["microbatch_size"],
              max_updates=None, seed=TRAINING_CONFIG["partition_base_seed"], resume=False, output_dir=None,
              db_path=DEFAULT_DB, stop_after_updates_this_run=None,
              replicas=TRAINING_CONFIG["replicas"], precision=TRAINING_CONFIG["precision"],
              prefetch=EXECUTION_CONFIG.prefetch_batches,
              checkpoint_every=EXECUTION_CONFIG.checkpoint_every_updates,
              train_step_compile=TRAINING_CONFIG["train_step_compile"], allow_cpu=False):
    layout = batch_layout(effective_batch, microbatch, replicas)
    if precision not in ("float32", "mixed_float16"):
        raise ValueError("unsupported precision")
    if checkpoint_every < 1 or prefetch < 0:
        raise ValueError("checkpoint interval must be positive and prefetch nonnegative")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max-updates must be positive")
    if stop_after_updates_this_run is not None and stop_after_updates_this_run < 1:
        raise ValueError("stop-after-updates-this-run must be positive")
    devices = training_devices(replicas, allow_cpu=allow_cpu)
    torch.manual_seed(seed)
    out_dir = Path(output_dir or DEFAULT_OUT / "partition-{}".format(partition))
    # Fail early when a run directory contains a checkpoint from another backend.
    checkpoint_path = latest_checkpoint(out_dir)
    if resume and checkpoint_path is None:
        raise ValueError("cannot resume: no PyTorch checkpoint found")
    if not resume and checkpoint_path is not None:
        raise ValueError("run directory already contains checkpoints; pass --resume or choose a new output directory")
    # Keep the caller path for external manifests next to a symlink.
    requested_db_path = Path(db_path).expanduser()
    input_identity = locked_artifacts(requested_db_path)
    locked = input_identity["fingerprint"]
    database_location = input_identity["location"]
    db_path = requested_db_path.resolve(strict=True)
    model, store, word_manifest = make_model(db_path)
    if Path(store.db_path).resolve(strict=True) != db_path:
        raise ValueError("model loader opened a different database than the fingerprinted DB")
    if store.batch("train", limit=min(2, effective_batch))[0] is None:
        raise ValueError("training DB is empty")
    model.to(devices[0])
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=TRAINING_CONFIG["learning_rate_peak"],
        betas=(TRAINING_CONFIG["beta_1"], TRAINING_CONFIG["beta_2"]),
        eps=TRAINING_CONFIG["epsilon"], weight_decay=TRAINING_CONFIG["weight_decay"],
        amsgrad=TRAINING_CONFIG["amsgrad"], fused=devices[0].type == "cuda", foreach=False)
    count_path = out_dir / "partition_counts.json"
    config_path = out_dir / "run_config.json"
    prior = None
    if resume:
        if not config_path.is_file():
            raise ValueError("cannot resume: run_config.json missing")
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        prior_config = prior.get("config", {})
        request = {
            "partition": int(partition), "seed": int(seed),
            "effective_batch_size": int(effective_batch), "microbatch_size": int(microbatch),
            "replicas": int(replicas), "precision": precision, "train_step_compile": bool(train_step_compile),
            "prefetch_batches": int(prefetch), "checkpoint_every_updates": int(checkpoint_every),
            "max_impressions": max_impressions, "max_updates": max_updates,
            "inputs": locked, "word_matrix_sha256": word_manifest["matrix_sha256"],
        }
        for key, value in request.items():
            if prior_config.get(key) != value:
                raise ValueError("resume refused before writes: fingerprint mismatch at {}".format(key))
        count = prior_config.get("exact_partition_counts")
        if count is None:
            raise ValueError("resume refused: prior run has no exact sampling count evidence")
        if count_path.exists():
            count_info = json.loads(count_path.read_text(encoding="utf-8"))
            if (count_info.get("inputs") != locked or count_info.get("partition") != partition
                    or count_info.get("seed") != seed or count_info.get("counts") != count):
                raise ValueError("resume refused before writes: partition count fingerprint mismatch")
        if not out_dir.is_dir():
            raise ValueError("cannot resume: output directory is missing")
    elif count_path.exists():
        count_info = json.loads(count_path.read_text(encoding="utf-8"))
        if count_info.get("inputs") != locked or count_info.get("partition") != partition or count_info.get("seed") != seed or count_info.get("max_impressions") != max_impressions:
            raise ValueError("stored partition count fingerprint mismatch")
        count = count_info["counts"]
    else:
        count = count_partition_candidates(store, partition, seed,
                                           max_impressions=max_impressions)
        count_info = {"inputs": locked, "partition": partition, "seed": seed,
                      "max_impressions": max_impressions, "counts": count}
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_count = count_path.with_suffix(".json.tmp")
        temp_count.write_text(json.dumps(count_info, indent=2, sort_keys=True) + "\n")
        temp_count.replace(count_path)
    total_updates = (int(max_updates) if max_updates is not None else
                     max(1, int(np.ceil(count["selected_candidates"] / float(effective_batch)))))
    run_config = dict(TRAINING_CONFIG, partition=int(partition), seed=int(seed),
                      effective_batch_size=int(effective_batch), microbatch_size=int(microbatch),
                      replicas=int(replicas), precision=precision, train_step_compile=bool(train_step_compile),
                      gradient_accumulation_steps=layout["gradient_accumulation_steps"],
                      global_microbatch_size=layout["global_microbatch_size"],
                      prefetch_batches=int(prefetch), checkpoint_every_updates=int(checkpoint_every),
                      word_matrix_sha256=word_manifest["matrix_sha256"],
                      inputs=locked, max_impressions=max_impressions,
                      max_updates=max_updates, exact_partition_counts=count,
                      sampling_state_contract={
                          "kind": "deterministic replay from locked DB content, implementation, seed, impression order, and partition",
                          "examples_seen": "number of selected candidate examples whose gradients were applied",
                          "cursor": "zero-based ordinal of the next selected candidate in the deterministic stream",
                          "cursor_validation": "replay the consumed prefix and compare its final full candidate key with the checkpoint key",
                          "cursor_count_invariant": "examples_seen equals cursor count because each selected candidate contributes one example",
                          "order": "seeded permutation of train impression IDs; candidates ordered by position",
                          "partition_count": int(TRAINING_CONFIG["partition_count"]),
                          "partition_seed": int(seed),
                      },
                      checkpoint_state_schema_version=CHECKPOINT_SCHEMA_VERSION,
                      schedule_state_contract={
                          "checkpoint_schedule_index": "zero-based index of the next optimizer update",
                          "lr_assignment": "learning_rate_at(schedule_index) before optimizer.step",
                          "saved_optimizer_lr": "peak LR at update 0; otherwise learning_rate_at(update - 1)",
                          "saved_lr_restore_tolerance": {"rtol": 1e-6, "atol": 1e-12},
                          "schedule_index_equals_optimizer_iterations": True,
                          "short_final_batch": "one final update normalized by its actual selected candidate count",
                      },
                      total_optimizer_updates=total_updates,
                      implementation_fingerprint=implementation_fingerprint())
    config_hash = _json_hash(run_config)
    if resume:
        if prior.get("config_sha256") != config_hash or prior.get("config") != run_config:
            raise ValueError("resume refused before checkpoint writes: model/data/config fingerprint mismatch")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_config = config_path.with_suffix(".json.tmp")
        temp_config.write_text(json.dumps({"config": run_config, "config_sha256": config_hash,
                                          "database_location_at_creation": database_location},
                                          indent=2, sort_keys=True) + "\n")
        temp_config.replace(config_path)
    update = examples_seen = sampling_cursor = schedule_index = 0
    saved_last_candidate_key = ""
    if resume:
        state = load_checkpoint(checkpoint_path, config_hash)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        update = state["update"]
        examples_seen = state["examples_seen"]
        sampling_cursor = state["sampling_cursor"]
        saved_last_candidate_key = state["sampling_cursor_last_candidate_key"]
        schedule_index = state["schedule_index"]
        if update and (len(optimizer.state) != len(trainable)
                       or any(int(optimizer.state[param]["step"].item()) != update
                              for param in trainable)):
            raise ValueError("cannot resume: Adam step counters differ from the checkpoint update")
    if examples_seen != sampling_cursor:
        raise ValueError("cannot resume: examples_seen count differs from sampling cursor ordinal")
    if schedule_index != update:
        raise ValueError("cannot resume: schedule index {} != next zero-based update {}".format(
            schedule_index, update))
    expected_saved_lr = (float(TRAINING_CONFIG["learning_rate_peak"]) if update == 0 else
                         learning_rate_at(
                             update - 1, total_updates,
                             peak=TRAINING_CONFIG["learning_rate_peak"],
                             minimum=TRAINING_CONFIG["learning_rate_min"],
                             warmup_fraction=TRAINING_CONFIG["warmup_fraction"]))
    restored_lr = optimizer.param_groups[0]["lr"]
    if not np.isclose(restored_lr, expected_saved_lr, rtol=1e-6, atol=1e-12):
        raise ValueError("cannot resume: saved optimizer LR {} != expected {} after update {}".format(
            restored_lr, expected_saved_lr, update))
    if update < 0 or update > total_updates:
        raise ValueError("cannot resume: checkpoint update is outside the locked epoch schedule")
    expected_cursor = min(update * int(effective_batch), int(count["selected_candidates"]))
    if sampling_cursor != expected_cursor:
        raise ValueError(
            "cannot resume: sampling cursor {} != expected {} from update {}, "
            "effective batch {}, and selected candidate count {}".format(
                sampling_cursor, expected_cursor, update, effective_batch,
                count["selected_candidates"]))
    if (sampling_cursor == 0 and saved_last_candidate_key) or (
            sampling_cursor > 0 and not saved_last_candidate_key):
        raise ValueError("cannot resume: cursor ordinal and last candidate key disagree")
    if saved_last_candidate_key:
        try:
            saved_key_object = json.loads(saved_last_candidate_key)
        except ValueError as exc:
            raise ValueError("cannot resume: saved candidate cursor key is malformed") from exc
        if saved_key_object.get("split") != "train":
            raise ValueError("cannot resume: saved candidate cursor key is not from train")
    trainer = DistributedTrainer(devices, model, optimizer, effective_batch, microbatch,
                                 precision=precision, compile_head=train_step_compile)
    trainer.updates = update
    if resume:
        trainer.loss_scale = state["loss_scale"]
        trainer.good_updates = state["loss_scale_good_updates"]
        del state

    def save():
        return save_checkpoint(out_dir, {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": config_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "examples_seen": examples_at_start + examples,
            "sampling_cursor": examples_at_start + examples,
            "sampling_cursor_last_candidate_key": last_candidate_key,
            "schedule_index": update,
            "loss_scale": trainer.loss_scale,
            "loss_scale_good_updates": trainer.good_updates,
        })

    examples_at_start = examples_seen
    last_candidate_key = saved_last_candidate_key
    examples = updates_this_run = 0
    loss_sum = 0.0
    last_effective_loss = None
    label_counts = [0, 0]
    profile = {"data_batch_wait_seconds": 0.0, "train_step_seconds": 0.0,
               "checkpoint_save_seconds": 0.0, "first_train_step_seconds": None,
               "benchmark_warmup_updates": 5, "steady_state_updates": 0,
               "steady_state_candidates": 0, "steady_state_seconds": 0.0}

    def remaining_groups():
        replay = sampling_cursor if resume else 0
        replay_key = None
        if update >= total_updates and not replay:
            return
        for features, labels, metadata in iter_train_batches(
                store, partition, seed, max_impressions=max_impressions):
            if replay:
                skipped = min(replay, len(metadata))
                replay_key = _candidate_cursor_key(metadata[skipped - 1])
                replay -= skipped
                if replay == 0 and replay_key != saved_last_candidate_key:
                    raise ValueError("cannot resume: deterministic replay candidate key differs")
                if skipped == len(metadata):
                    continue
                features = {k: v[skipped:] for k, v in features.items()}
                labels = {k: v[skipped:] for k, v in labels.items()}
                metadata = metadata[skipped:]
            if update < total_updates:
                yield features, labels, metadata
            elif not replay:
                break
        if replay:
            raise ValueError("cannot resume: iterator ended before saved sampling cursor")

    # SQLite's legacy reader is thread-affine. The production compact reader
    # uses read-only memmaps and can run on the prefetch producer thread.
    active_prefetch = prefetch if hasattr(store, "compact") else 0
    batches = prefetch_batches(effective_batches(remaining_groups(), effective_batch), active_prefetch)
    start_time = time.time()
    print(json.dumps({"training_layout": layout, "precision": precision,
                      "train_step_compile": bool(train_step_compile), "prefetch_batches": active_prefetch,
                      "checkpoint_every_updates": checkpoint_every}), flush=True)
    progress = tqdm(total=total_updates, initial=update, unit="update",
                    desc="train partition {}".format(partition), dynamic_ncols=True, mininterval=1.0)
    try:
        while True:
            fetch_started = time.perf_counter()
            try:
                features, y, metadata = next(batches)
            except StopIteration:
                break
            profile["data_batch_wait_seconds"] += time.perf_counter() - fetch_started
            if update >= total_updates:
                break
            lr = learning_rate_at(update, total_updates,
                                  peak=TRAINING_CONFIG["learning_rate_peak"],
                                  minimum=TRAINING_CONFIG["learning_rate_min"],
                                  warmup_fraction=TRAINING_CONFIG["warmup_fraction"])
            optimizer.param_groups[0]["lr"] = lr
            step_started = time.perf_counter()
            batch_loss_sum = trainer.train_batch(features, y)
            step_elapsed = time.perf_counter() - step_started
            profile["train_step_seconds"] += step_elapsed
            if profile["first_train_step_seconds"] is None:
                profile["first_train_step_seconds"] = step_elapsed
            if trainer.updates != update + 1:
                raise ValueError("optimizer iteration did not advance exactly once")
            n = len(y)
            examples += n
            label_counts[1] += int(y.sum())
            label_counts[0] += n - int(y.sum())
            loss_sum += batch_loss_sum
            last_effective_loss = batch_loss_sum / n
            last_candidate_key = _candidate_cursor_key(metadata[-1])
            update += 1
            updates_this_run += 1
            if update % checkpoint_every == 0:
                save_started = time.perf_counter()
                checkpoint_path = save()
                profile["checkpoint_save_seconds"] += time.perf_counter() - save_started
            if updates_this_run > profile["benchmark_warmup_updates"]:
                profile["steady_state_updates"] += 1
                profile["steady_state_candidates"] += n
                profile["steady_state_seconds"] += time.perf_counter() - fetch_started
            progress.update(1)
            progress.set_postfix(loss="{:.5f}".format(last_effective_loss),
                                 lr="{:.3g}".format(lr), examples=examples_at_start + examples,
                                 ex_s="{:.1f}".format(examples / max(time.time() - start_time, 1e-9)),
                                 ckpt=checkpoint_path.name if checkpoint_path else "-",
                                 refresh=False)
            if (update >= total_updates or (stop_after_updates_this_run is not None
                    and updates_this_run >= stop_after_updates_this_run)):
                break
    finally:
        batches.close()
        trainer.close()
        progress.close()
    profile["overflow_retries"] = trainer.overflow_retries
    profile["loss_scale"] = trainer.loss_scale
    profile["steady_state_candidates_per_second"] = (
        profile["steady_state_candidates"] / profile["steady_state_seconds"]
        if profile["steady_state_seconds"] else None)
    if max_updates is None and max_impressions is None and stop_after_updates_this_run is None:
        if examples_at_start + examples != int(count["selected_candidates"]):
            raise ValueError("full partition iterator ended at a different sampling cursor than its exact count")
        if update != total_updates:
            raise ValueError("full partition iterator ended before the locked optimizer update count")
    if checkpoint_path is None or checkpoint_path.name != "ckpt-{}.pt".format(update):
        save_started = time.perf_counter()
        checkpoint_path = save()
        profile["checkpoint_save_seconds"] += time.perf_counter() - save_started
    elapsed = time.time() - start_time
    full_epoch = max_updates is None and max_impressions is None and stop_after_updates_this_run is None
    report = {"status": "EPOCH_ITERATOR_EXHAUSTED" if full_epoch else "PARTIAL_COMPUTE",
              "full_epoch": full_epoch,
              "partition": partition, "examples_this_run": examples,
              "examples_seen_total": examples_at_start + examples,
              "sampling_cursor_count": examples_at_start + examples,
              "sampling_cursor_last_candidate_key": json.loads(last_candidate_key)
                  if last_candidate_key else None,
              "positives": label_counts[1], "negatives": label_counts[0],
              "partition_counts_expected": count,
              "optimizer_updates_this_run": updates_this_run, "optimizer_update_total": update,
              "optimizer_iterations": trainer.updates,
              "schedule_index_next_update": update,
              "effective_batch_size": effective_batch, "microbatch_size": microbatch,
              "training_layout": layout, "precision": precision, "train_step_compile": bool(train_step_compile),
              "loss_mean": loss_sum / max(1, examples), "elapsed_seconds": elapsed,
              "examples_per_second": examples / max(elapsed, 1e-9),
              "latest_checkpoint": str(checkpoint_path),
              "database_location": database_location,
              "checkpoint_bytes": sum(p.stat().st_size for p in (out_dir / "checkpoints").glob("*")),
              "performance": profile,
              "resume_command": ("python -m mind_news_metadata.full_pipeline train "
                                 "--partition {} --seed {} --effective-batch {} --batch-size {} "
                                 "--replicas {} --precision {} --prefetch-batches {} --checkpoint-every {}"
                                 "{}{}{} --resume --output-dir {} --db {}".format(
                                     partition, seed, effective_batch, microbatch, replicas, precision,
                                     prefetch, checkpoint_every, " --train-step-compile" if train_step_compile else "",
                                     " --max-impressions {}".format(max_impressions) if max_impressions is not None else "",
                                     " --max-updates {}".format(max_updates) if max_updates is not None else "",
                                     out_dir, db_path)),
              "config_sha256": config_hash}
    report_bytes = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    atomic_write(out_dir / "training_progress.json", lambda stream: stream.write(report_bytes))
    return report


def evaluate(model, store, split="dev", profile="dev_raw", train_dids=None,
             batch_size=EXECUTION_CONFIG.evaluation_batch_size, max_candidates=None):
    model.eval()
    device = next(model.parameters()).device
    precision = getattr(model, "inference_precision", "float32")
    current_id = None
    labels, scores, positions = [], [], []
    seen = 0

    def completed_groups():
        nonlocal current_id, labels, scores, positions, seen
        for features, y, metadata in iter_ordered_batches(store, split, batch_size):
            if max_candidates is not None and seen >= int(max_candidates):
                break
            take = len(metadata) if max_candidates is None else min(len(metadata), int(max_candidates) - seen)
            features = {k: v[:take] for k, v in features.items()}
            y = {k: v[:take] for k, v in y.items()}
            metadata = metadata[:take]
            if split != "test" and not bool(y["labeled"].all()):
                raise ValueError("labeled evaluation split contains unlabeled candidates")
            if split == "test" and bool(y["labeled"].any()):
                raise ValueError("test DB unexpectedly contains labels")
            masked = _candidate_mask(features, metadata, profile, train_dids)
            with torch.inference_mode(), autocast(device, precision):
                prob = model(to_tensors(masked, device))["probabilities"].cpu().numpy()
            if not np.isfinite(prob).all():
                raise ValueError("non-finite predicted probabilities")
            seen += len(metadata)
            if split == "test":
                continue
            for row, label, score in zip(metadata, y["click"], prob):
                imp = row["impression_id"]
                if current_id is not None and imp != current_id:
                    yield (current_id, labels, scores, positions)
                    labels, scores, positions = [], [], []
                current_id = imp
                labels.append(int(label)); scores.append(float(score)); positions.append(int(row["position"]))
        if split != "test" and current_id is not None:
            yield (current_id, labels, scores, positions)

    if split == "test":
        for _ in completed_groups():
            pass
        return {"status": "PREDICTION_FORWARD_PASS", "candidates": seen,
                "metrics": "unavailable: test labels are null"}
    metrics = aggregate_impression_metrics(completed_groups())
    return {"status": "PASS" if max_candidates is None else "PILOT_ONLY_NOT_FULL_SPLIT",
            "split": split, "profile": profile, "candidates": seen,
            "candidate_limit": max_candidates, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--db", default=str(DEFAULT_DB)); tr.add_argument("--partition", type=int, default=TRAINING_CONFIG["default_partition"])
    tr.add_argument("--seed", type=int, default=TRAINING_CONFIG["partition_base_seed"]); tr.add_argument("--effective-batch", type=int, default=TRAINING_CONFIG["effective_batch_size"])
    tr.add_argument("--batch-size", "--microbatch", dest="microbatch", type=int,
                    default=TRAINING_CONFIG["microbatch_size"], help="candidates per GPU per microstep")
    tr.add_argument("--replicas", type=int, default=TRAINING_CONFIG["replicas"])
    tr.add_argument("--precision", choices=("float32", "mixed_float16"), default=TRAINING_CONFIG["precision"])
    tr.add_argument("--prefetch-batches", type=int, default=EXECUTION_CONFIG.prefetch_batches)
    tr.add_argument("--checkpoint-every", type=int, default=EXECUTION_CONFIG.checkpoint_every_updates)
    tr.add_argument("--train-step-compile", action=argparse.BooleanOptionalAction,
                    default=TRAINING_CONFIG["train_step_compile"],
                    help="use torch.compile for the MLP during training")
    tr.add_argument("--allow-cpu", action="store_true")
    tr.add_argument("--max-impressions", type=int)
    tr.add_argument("--max-updates", type=int); tr.add_argument("--resume", action="store_true")
    tr.add_argument("--stop-after-updates-this-run", type=int)
    tr.add_argument("--output-dir")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--run-dir", required=True); ev.add_argument("--db", default=str(DEFAULT_DB))
    ev.add_argument("--split", default="dev", choices=("dev", "test"))
    ev.add_argument("--profile", default=TRAINING_CONFIG["dev_mask_profile"],
                    choices=("dev_new_did_masked", "dev_raw", "test", "test_all_mask"))
    ev.add_argument("--train-dids"); ev.add_argument("--batch-size", type=int, default=EXECUTION_CONFIG.evaluation_batch_size)
    ev.add_argument("--max-candidates", type=int)
    pr = sub.add_parser("predict")
    pr.add_argument("--run-dir", required=True); pr.add_argument("--db", default=str(DEFAULT_DB))
    pr.add_argument("--split", default="test", choices=("dev", "test")); pr.add_argument("--profile")
    pr.add_argument("--train-dids"); pr.add_argument("--output"); pr.add_argument("--batch-size", type=int, default=EXECUTION_CONFIG.evaluation_batch_size)
    pr.add_argument("--max-candidates", type=int)
    pr.add_argument("--replicas", type=int, default=1,
                    help="GPU replicas for prediction; batch-size is global across devices")
    en = sub.add_parser("ensemble")
    en.add_argument("--inputs", nargs="+", required=True); en.add_argument("--output", required=True)
    su = sub.add_parser("submit")
    su.add_argument("--inputs", nargs="+", required=True,
                    help="one or more completed full-test prediction CSV.gz files")
    su.add_argument("--output", required=True)
    su.add_argument("--db", default=str(DEFAULT_DB))
    mb = sub.add_parser("build-membership")
    mb.add_argument("--db", default=str(DEFAULT_DB)); mb.add_argument("--train-behaviors", required=True)
    mb.add_argument("--output")
    a = ap.parse_args()
    if a.command == "train":
        print(json.dumps(run_train(a.partition, a.max_impressions, a.effective_batch,
                                   a.microbatch, a.max_updates, a.seed, a.resume,
                                   a.output_dir, a.db, a.stop_after_updates_this_run,
                                   replicas=a.replicas, precision=a.precision,
                                   prefetch=a.prefetch_batches, checkpoint_every=a.checkpoint_every,
                                   train_step_compile=a.train_step_compile, allow_cpu=a.allow_cpu), indent=2))
    elif a.command == "build-membership":
        from .build_train_did_membership import build
        print(json.dumps(build(a.db, a.train_behaviors, a.output), indent=2))
    elif a.command == "evaluate":
        from .prediction import load_run, train_did_set
        model, store, prefix, config = load_run(a.run_dir, a.db)
        membership = (train_did_set(a.train_dids, config["config"]["inputs"]["database"])
                      if a.train_dids else None)
        if a.profile == "dev_new_did_masked" and membership is None:
            raise ValueError("dev_new_did_masked requires --train-dids built from full train behaviors")
        result = evaluate(model, store, a.split, a.profile, membership,
                          a.batch_size, a.max_candidates)
        result["checkpoint"] = prefix
        result["run_config_sha256"] = config["config_sha256"]
        result["database_location"] = config.get("current_database_location")
        print(json.dumps(result, indent=2))
    elif a.command == "predict":
        from .prediction import predict
        profile = a.profile or ("test" if a.split == "test" else "dev_raw")
        print(json.dumps(predict(a.run_dir, a.split, profile, a.train_dids,
                                 a.output, a.db, a.batch_size, a.max_candidates,
                                 replicas=a.replicas), indent=2))
    elif a.command == "ensemble":
        from .prediction import ensemble
        print(json.dumps(ensemble(a.inputs, a.output), indent=2))
    elif a.command == "submit":
        from .prediction import submission
        print(json.dumps(submission(a.inputs, a.output, a.db), indent=2))


if __name__ == "__main__":
    main()
