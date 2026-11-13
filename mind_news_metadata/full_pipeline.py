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
import tensorflow as tf

from .batch_data import NewsBatchStore, connect_readonly
from .field_encoder import (TitleAbstractMetadataEncoder, load_glove_artifact,
                            load_pretrained_matrix, sha256_file)
from .recommendation_model import NoBodyRecommendationModel, pointwise_bce_sum
from .recommendation_protocol import (aggregate_impression_metrics,
                                      learning_rate_at, partition_for_impression)
from mind_config import ENCODER_CONFIG, EXECUTION_CONFIG, TRAINING_CONFIG


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "artifacts/mind_large_prepared/stage01.sqlite"
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
             "mind_news_metadata/encoder_config.py",
             "mind_news_metadata/field_encoder.py", "mind_news_metadata/recommendation_model.py",
             "mind_news_metadata/recommendation_protocol.py", "mind_news_metadata/full_pipeline.py",
             "mind_news_metadata/prediction.py", "mind_news_metadata/build_train_did_membership.py"]
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
    resolved_db = Path(db_path).expanduser().resolve(strict=True)
    if not resolved_db.is_file():
        raise FileNotFoundError("database is not a regular file: {}".format(resolved_db))

    db_manifest_path = resolved_db.parent / "artifact_manifest.json"
    if not db_manifest_path.is_file():
        raise ValueError("database sidecar artifact_manifest.json is missing beside {}".format(resolved_db))
    db_manifest = json.loads(db_manifest_path.read_text(encoding="utf-8"))
    db_relative = resolved_db.relative_to(db_manifest_path.parent.resolve()).as_posix()
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
    vocab_hashes = {}
    for field in vocab_fields:
        rel = "vocab/{}.tsv".format(field)
        path = db_manifest_path.parent / rel
        expected = expected_vocab_hashes.get(rel)
        if expected is None:
            expected = artifact_entries.get(rel, {}).get("sha256")
        if expected is None or not path.is_file():
            raise ValueError("database manifest lacks mapping provenance for {}".format(rel))
        actual, vocab_stat = _sha256_stable_file(path)
        if _file_signature(vocab_stat) != _file_signature(path.stat()):
            raise ValueError("database vocabulary mapping changed during fingerprint verification: {}".format(
                path))
        if actual != expected:
            raise ValueError("database vocabulary mapping checksum differs for {}".format(path))
        vocab_hashes[rel] = actual

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
            mapping_digest.update(field.encode("utf-8") + b"\0")
            mapping_digest.update(bytes.fromhex(vocab_hashes["vocab/{}.tsv".format(field)]))
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

    out = ROOT / "artifacts/mind_large_prepared"
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
    store = NewsBatchStore(db_path)
    sizes = store.vocab_sizes()
    emb = ROOT / "artifacts/mind_large_prepared/embeddings"
    vocab = ROOT / "artifacts/mind_large_prepared/vocab"
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
    con = connect_readonly(store.db_path)
    ids = np.fromiter((r[0] for r in con.execute(
        "SELECT id FROM impressions WHERE split='train' ORDER BY line_no")), dtype=np.int64)
    con.close()
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
    con = connect_readonly(store.db_path)
    ids = np.fromiter((r[0] for r in con.execute(
        "SELECT id FROM impressions WHERE split='train' ORDER BY line_no")), dtype=np.int64)
    rng = np.random.RandomState(seed)
    order = ids[rng.permutation(len(ids))]
    if max_impressions is not None:
        order = order[:int(max_impressions)]
    positives = negatives = impression_count = 0
    for start in range(0, len(order), int(impression_group)):
        part_ids = order[start:start + int(impression_group)].tolist()
        marks = ",".join("?" for _ in part_ids)
        rows = con.execute("""
          SELECT i.id,i.impression_id,c.position,l.click,l.labeled
          FROM impressions i JOIN candidates c ON c.impression_id=i.id
          JOIN labels l ON l.candidate_id=c.id
          WHERE i.id IN (""" + marks + ") ORDER BY i.id,c.position", part_ids).fetchall()
        by_imp = {}
        for row in rows:
            by_imp.setdefault(row[0], []).append(row)
        for iid in part_ids:
            group = by_imp.get(iid, [])
            if not group:
                continue
            if not all(row[4] for row in group):
                raise ValueError("unlabeled candidate in train partition count")
            keep = partition_for_impression(
                [int(row[3]) for row in group], [int(row[2]) for row in group],
                group[0][1], partition, partitions=TRAINING_CONFIG["partition_count"],
                base_seed=seed, split="train")
            selected = [row[3] for row, use in zip(group, keep) if use]
            positives += sum(int(x) == 1 for x in selected)
            negatives += sum(int(x) == 0 for x in selected)
            impression_count += 1
    con.close()
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


def _to_tensors(features):
    return {key: tf.convert_to_tensor(value, dtype=tf.int32) for key, value in features.items()}


def _slice_features(features, start, stop):
    return {k: v[start:stop] for k, v in features.items()}


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


def make_checkpoint(model, optimizer, update, examples, sampling_cursor,
                    last_candidate_key, out_dir):
    update_var = tf.Variable(int(update), trainable=False, dtype=tf.int64, name="optimizer_update")
    examples_var = tf.Variable(int(examples), trainable=False, dtype=tf.int64, name="examples_seen")
    sampling_cursor_var = tf.Variable(int(sampling_cursor), trainable=False,
                                      dtype=tf.int64, name="sampling_cursor")
    last_candidate_key_var = tf.Variable(
        str(last_candidate_key), trainable=False, dtype=tf.string,
        name="sampling_cursor_last_candidate_key")
    schedule_var = tf.Variable(int(update), trainable=False, dtype=tf.int64, name="schedule_index")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, update=update_var,
                               examples_seen=examples_var, sampling_cursor=sampling_cursor_var,
                               sampling_cursor_last_candidate_key=last_candidate_key_var,
                               schedule_index=schedule_var)
    manager = tf.train.CheckpointManager(ckpt, str(out_dir / "checkpoints"), max_to_keep=2)
    return (ckpt, manager, update_var, examples_var, sampling_cursor_var,
            last_candidate_key_var, schedule_var)


def run_train(partition=TRAINING_CONFIG["default_partition"], max_impressions=None,
              effective_batch=TRAINING_CONFIG["effective_batch_size"],
              microbatch=TRAINING_CONFIG["microbatch_size"],
              max_updates=None, seed=TRAINING_CONFIG["partition_base_seed"], resume=False, output_dir=None,
              db_path=DEFAULT_DB, stop_after_updates_this_run=None):
    if effective_batch < 1 or microbatch < 1:
        raise ValueError("batch sizes must be positive")
    tf.keras.utils.set_random_seed(seed)
    db_path = Path(db_path).expanduser().resolve(strict=True)
    input_identity = locked_artifacts(db_path)
    locked = input_identity["fingerprint"]
    database_location = input_identity["location"]
    model, store, word_manifest = make_model(db_path)
    if Path(store.db_path).resolve(strict=True) != db_path:
        raise ValueError("model loader opened a different database than the fingerprinted DB")
    first = store.batch("train", limit=min(2, effective_batch))
    if first[0] is None:
        raise ValueError("training DB is empty")
    model(_to_tensors(first[0]), training=False)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=TRAINING_CONFIG["learning_rate_peak"],
        beta_1=TRAINING_CONFIG["beta_1"], beta_2=TRAINING_CONFIG["beta_2"],
        epsilon=TRAINING_CONFIG["epsilon"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        amsgrad=TRAINING_CONFIG["amsgrad"],
        clipnorm=TRAINING_CONFIG["clipnorm"],
        clipvalue=TRAINING_CONFIG["clipvalue"],
        global_clipnorm=TRAINING_CONFIG["global_clipnorm"],
        use_ema=TRAINING_CONFIG["use_ema"],
        jit_compile=TRAINING_CONFIG["jit_compile"],
        name="adam_reimplementation")
    out_dir = Path(output_dir or DEFAULT_OUT / "partition-{}".format(partition))
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
    elif (out_dir / "checkpoints").exists() and list((out_dir / "checkpoints").glob("ckpt-*")):
        raise ValueError("run directory already contains checkpoints; pass --resume or choose a new output directory")
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
                      checkpoint_state_schema_version=2,
                      schedule_state_contract={
                          "checkpoint_schedule_index": "zero-based index of the next optimizer update",
                          "lr_assignment": "learning_rate_at(schedule_index) before apply_gradients",
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
    optimizer.build(model.trainable_variables)
    (ckpt, manager, update_var, examples_var, sampling_cursor_var,
     last_candidate_key_var, schedule_var) = make_checkpoint(
        model, optimizer, 0, 0, 0, "", out_dir)
    if resume:
        if not manager.latest_checkpoint:
            raise ValueError("cannot resume: no checkpoint found")
        checkpoint_prefix = manager.latest_checkpoint
        checkpoint_dir = Path(checkpoint_prefix).parent
        checkpoint_files = sorted(checkpoint_dir.glob("ckpt-*.index"))
        index_prefixes = {path.with_suffix("").name for path in checkpoint_files}
        data_prefixes = {path.name.split(".data-", 1)[0]
                         for path in checkpoint_dir.glob("ckpt-*.data-*-of-*")}
        if not checkpoint_files or data_prefixes - index_prefixes:
            raise ValueError("cannot resume: checkpoint index/data shard set is incomplete")
        complete_updates = []
        incomplete = []
        for index_path in checkpoint_files:
            prefix = index_path.with_suffix("")
            shard_paths = sorted(checkpoint_dir.glob(prefix.name + ".data-*-of-*"))
            if not shard_paths:
                incomplete.append(prefix.name)
                continue
            totals = set()
            shard_ids = set()
            for shard in shard_paths:
                try:
                    shard_prefix, shard_suffix = shard.name.split(".data-", 1)
                    shard_id_text, total_text = shard_suffix.split("-of-", 1)
                    if shard_prefix != prefix.name:
                        raise ValueError("checkpoint shard prefix mismatch")
                    shard_id, total = shard_id_text, total_text
                    shard_ids.add(int(shard_id))
                    totals.add(int(total))
                except (IndexError, ValueError):
                    incomplete.append(prefix.name)
                    break
            if prefix.name in incomplete:
                continue
            if len(totals) != 1 or shard_ids != set(range(next(iter(totals)))):
                incomplete.append(prefix.name)
                continue
            try:
                complete_updates.append(int(prefix.name.rsplit("-", 1)[1]))
            except ValueError:
                incomplete.append(prefix.name)
        if incomplete:
            raise ValueError("cannot resume: incomplete checkpoint files found: {}".format(incomplete))
        if not complete_updates or int(checkpoint_prefix.rsplit("-", 1)[1]) != max(complete_updates):
            raise ValueError("cannot resume: checkpoint manager does not point to the newest complete checkpoint")
        restore_status = ckpt.restore(checkpoint_prefix)
        restore_status.assert_existing_objects_matched()
        restore_status.assert_consumed()
    update = int(update_var.numpy())
    optimizer_iteration = int(optimizer.iterations.numpy())
    if optimizer_iteration != update:
        raise ValueError("cannot resume: optimizer.iterations {} != update counter {}".format(
            optimizer_iteration, update))
    if resume and int(Path(checkpoint_prefix).name.rsplit("-", 1)[1]) != update:
        raise ValueError("cannot resume: checkpoint filename update != restored update counter")
    examples_seen = int(examples_var.numpy())
    sampling_cursor = int(sampling_cursor_var.numpy())
    saved_last_candidate_key = last_candidate_key_var.numpy().decode("utf-8")
    if examples_seen != sampling_cursor:
        raise ValueError("cannot resume: examples_seen count differs from sampling cursor ordinal")
    schedule_index = int(schedule_var.numpy())
    if schedule_index != update:
        raise ValueError("cannot resume: schedule index {} != next zero-based update {}".format(
            schedule_index, update))
    expected_saved_lr = (float(TRAINING_CONFIG["learning_rate_peak"]) if update == 0 else
                         learning_rate_at(
                             update - 1, total_updates,
                             peak=TRAINING_CONFIG["learning_rate_peak"],
                             minimum=TRAINING_CONFIG["learning_rate_min"],
                             warmup_fraction=TRAINING_CONFIG["warmup_fraction"]))
    restored_lr = float(optimizer.learning_rate.numpy())
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
    replay_remaining = sampling_cursor if resume else 0
    replay_last_candidate_key = None
    examples_at_start = examples_seen
    last_candidate_key = saved_last_candidate_key
    pending = 0
    accumulated_count = 0
    accum = [None] * len(model.trainable_variables)
    groups = (iter_train_batches(store, partition, seed, max_impressions=max_impressions)
              if update < total_updates or (resume and replay_remaining > 0) else ())
    examples = 0
    updates_this_run = 0
    loss_sum = 0.0
    loss_parts = []
    label_counts = [0, 0]
    profile = {"trace_seconds": 0.0, "data_batch_wait_seconds": 0.0, "tensor_conversion_seconds": 0.0,
               "gradient_dispatch_seconds": 0.0, "gradient_accumulation_dispatch_seconds": 0.0,
               "optimizer_apply_dispatch_seconds": 0.0,
               "device_sync_and_loss_read_seconds": 0.0,
               "checkpoint_save_seconds": 0.0,
               "gradient_calls": 0, "first_gradient_dispatch_seconds": None}

    # The model/optimizer variables are built before tracing. This keeps Python
    # dispatch out of the per-microbatch hot path; the math and effective batch
    # accumulation remain unchanged.
    feature_signature = {
        key: tf.TensorSpec([None] + list(value.shape[1:]), tf.int32, name=key)
        for key, value in first[0].items()
    }

    @tf.function(input_signature=[feature_signature,
                                  tf.TensorSpec([None], tf.float32, name="labels")])
    def compute_gradients(batch_features, batch_labels):
        with tf.GradientTape() as tape:
            output = model(batch_features, training=True)
            batch_loss = pointwise_bce_sum(batch_labels, output["logits"])
        batch_grads = tape.gradient(batch_loss, model.trainable_variables)
        return batch_loss, batch_grads

    trace_started = time.perf_counter()
    compute_gradients.get_concrete_function()
    profile["trace_seconds"] = time.perf_counter() - trace_started
    start_time = time.time()

    def apply_accumulated():
        nonlocal update, schedule_index, pending, accumulated_count, accum
        nonlocal updates_this_run, loss_sum, loss_parts
        if accumulated_count == 0:
            return
        if schedule_index != update:
            raise ValueError("schedule cursor diverged from optimizer update before optimizer step")
        # Tiny runs use an explicit finite update horizon; full runs use the
        # source-guided one-epoch cosine-free warmup/linear schedule.
        lr = learning_rate_at(
            min(update, total_updates - 1), total_updates,
            peak=TRAINING_CONFIG["learning_rate_peak"],
            minimum=TRAINING_CONFIG["learning_rate_min"],
            warmup_fraction=TRAINING_CONFIG["warmup_fraction"])
        optimizer.learning_rate.assign(lr)
        grads = []
        for grad, var in zip(accum, model.trainable_variables):
            if grad is None:
                grads.append((None, var))
            elif isinstance(grad, list):
                idx = tf.concat([g.indices for g in grad], 0)
                val = tf.concat([g.values for g in grad], 0) / float(accumulated_count)
                grads.append((tf.IndexedSlices(val, idx, grad[0].dense_shape), var))
            else:
                grads.append((grad / float(accumulated_count), var))
        apply_started = time.perf_counter()
        optimizer.apply_gradients((g, v) for g, v in grads if g is not None)
        profile["optimizer_apply_dispatch_seconds"] += time.perf_counter() - apply_started
        if int(optimizer.iterations.numpy()) != update + 1:
            raise ValueError("optimizer iteration did not advance exactly once for the effective batch")
        # Avoid synchronizing the GPU once per microbatch. Read the scalar loss
        # once at the effective-batch boundary; this wait also accounts for
        # outstanding device work rather than timing only asynchronous enqueue.
        sync_started = time.perf_counter()
        if loss_parts:
            loss_values = np.asarray(tf.stack(loss_parts).numpy(), dtype=np.float64)
            loss_sum += float(loss_values.sum(dtype=np.float64))
        profile["device_sync_and_loss_read_seconds"] += time.perf_counter() - sync_started
        loss_parts = []
        update += 1
        schedule_index = update
        update_var.assign(update)
        schedule_var.assign(schedule_index)
        examples_var.assign(examples_at_start + examples)
        sampling_cursor_var.assign(examples_at_start + examples)
        last_candidate_key_var.assign(last_candidate_key)
        updates_this_run += 1
        pending = 0
        accumulated_count = 0
        accum = [None] * len(model.trainable_variables)
        if update % EXECUTION_CONFIG.checkpoint_every_updates == 0:
            save_started = time.perf_counter()
            manager.save(checkpoint_number=update)
            profile["checkpoint_save_seconds"] += time.perf_counter() - save_started

    group_iter = iter(groups)
    while True:
        fetch_started = time.perf_counter()
        try:
            features, labels, metadata = next(group_iter)
        except StopIteration:
            profile["data_batch_wait_seconds"] += time.perf_counter() - fetch_started
            break
        profile["data_batch_wait_seconds"] += time.perf_counter() - fetch_started
        if not bool(labels["labeled"].all()):
            raise ValueError("train iterator returned unlabeled rows")
        y = labels["click"].astype(np.float32)
        if metadata is None or len(metadata) != len(y):
            raise ValueError("training candidate metadata is not aligned with labels")
        if replay_remaining:
            skipped = min(replay_remaining, len(y))
            if skipped:
                replay_last_candidate_key = _candidate_cursor_key(metadata[skipped - 1])
            replay_remaining -= skipped
            if replay_remaining == 0 and sampling_cursor > 0:
                if replay_last_candidate_key != saved_last_candidate_key:
                    raise ValueError(
                        "cannot resume: deterministic replay reached a different candidate key "
                        "at the saved sampling cursor")
            if skipped == len(y):
                continue
            features = {k: v[skipped:] for k, v in features.items()}
            y = y[skipped:]
            metadata = metadata[skipped:]
        convert_started = time.perf_counter()
        x = _to_tensors(features)
        profile["tensor_conversion_seconds"] += time.perf_counter() - convert_started
        stop_training = False
        for begin in range(0, len(y), microbatch):
            end = min(begin + microbatch, len(y))
            cursor = begin
            while cursor < end:
                # Split at the optimizer boundary when an impression-group
                # tail would otherwise make the effective batch overshoot.
                take = min(end - cursor, effective_batch - pending)
                stop = cursor + take
                batch_x = _slice_features(x, cursor, stop)
                batch_y = tf.convert_to_tensor(y[cursor:stop], dtype=tf.float32)
                compute_started = time.perf_counter()
                loss, grads = compute_gradients(batch_x, batch_y)
                call_elapsed = time.perf_counter() - compute_started
                if profile["first_gradient_dispatch_seconds"] is None:
                    profile["first_gradient_dispatch_seconds"] = call_elapsed
                profile["gradient_dispatch_seconds"] += call_elapsed
                profile["gradient_calls"] += 1
                loss_parts.append(loss)
                n = stop - cursor
                label_counts[1] += int(y[cursor:stop].sum())
                label_counts[0] += int(n - y[cursor:stop].sum())
                accumulation_started = time.perf_counter()
                for i, grad in enumerate(grads):
                    if grad is None:
                        continue
                    if isinstance(grad, tf.IndexedSlices):
                        if accum[i] is None:
                            accum[i] = [grad]
                        elif isinstance(accum[i], list):
                            accum[i].append(grad)
                        else:
                            raise TypeError("gradient type changed during accumulation")
                    else:
                        if accum[i] is None:
                            accum[i] = grad
                        elif isinstance(accum[i], list):
                            raise TypeError("gradient type changed during accumulation")
                        else:
                            accum[i] = accum[i] + grad
                profile["gradient_accumulation_dispatch_seconds"] += time.perf_counter() - accumulation_started
                pending += n
                accumulated_count += n
                examples += n
                last_candidate_key = _candidate_cursor_key(metadata[stop - 1])
                cursor = stop
                if pending == effective_batch:
                    apply_accumulated()
                    if ((max_updates is not None and update >= max_updates) or
                            (stop_after_updates_this_run is not None and updates_this_run >= stop_after_updates_this_run)):
                        stop_training = True
                        break
            if stop_training:
                break
        if stop_training:
            break
    if replay_remaining:
        raise ValueError("cannot resume: selected candidate stream ended before the saved cursor")
    if resume and sampling_cursor > 0 and replay_last_candidate_key != saved_last_candidate_key:
        raise ValueError("cannot resume: saved candidate key was not verified during deterministic replay")
    apply_accumulated()
    if max_updates is None and max_impressions is None and stop_after_updates_this_run is None:
        if int(sampling_cursor_var.numpy()) != int(count["selected_candidates"]):
            raise ValueError("full partition iterator ended at a different sampling cursor than its exact count")
        if update != total_updates:
            raise ValueError("full partition iterator ended before the locked optimizer update count")
    if updates_this_run or manager.latest_checkpoint is None:
        save_started = time.perf_counter()
        manager.save(checkpoint_number=update)
        profile["checkpoint_save_seconds"] += time.perf_counter() - save_started
    elapsed = time.time() - start_time
    report = {"status": "PARTIAL_COMPUTE" if (max_updates is not None or max_impressions is not None or stop_after_updates_this_run is not None) else "EPOCH_ITERATOR_EXHAUSTED",
              "partition": partition, "examples_this_run": examples,
              "examples_seen_total": int(examples_var.numpy()),
              "sampling_cursor_count": int(sampling_cursor_var.numpy()),
              "sampling_cursor_last_candidate_key": json.loads(last_candidate_key)
                  if last_candidate_key else None,
              "positives": label_counts[1], "negatives": label_counts[0],
              "partition_counts_expected": count,
              "optimizer_updates_this_run": updates_this_run, "optimizer_update_total": update,
              "optimizer_iterations": int(optimizer.iterations.numpy()),
              "schedule_index_next_update": int(schedule_var.numpy()),
              "effective_batch_size": effective_batch, "microbatch_size": microbatch,
              "loss_mean": loss_sum / max(1, examples), "elapsed_seconds": elapsed,
              "examples_per_second": examples / max(elapsed, 1e-9),
              "latest_checkpoint": manager.latest_checkpoint,
              "database_location": database_location,
              "checkpoint_bytes": sum(p.stat().st_size for p in (out_dir / "checkpoints").glob("*")),
              "performance": profile,
              "resume_command": ("PYTHONPATH=. /tmp/mind-train-tf215/bin/python -m mind_news_metadata.full_pipeline train "
                                 "--partition {} --seed {} --effective-batch {} --microbatch {}{}{} --resume --output-dir {}".format(
                                     partition, seed, effective_batch, microbatch,
                                     " --max-impressions {}".format(max_impressions) if max_impressions is not None else "",
                                     " --max-updates {}".format(max_updates) if max_updates is not None else "", out_dir)),
              "full_epoch": max_updates is None and max_impressions is None and stop_after_updates_this_run is None,
              "config_sha256": config_hash}
    (out_dir / "training_progress.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate(model, store, split="dev", profile="dev_raw", train_dids=None,
             batch_size=EXECUTION_CONFIG.evaluation_batch_size, max_candidates=None):
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
            out = model(_to_tensors(masked), training=False)
            prob = out["probabilities"].numpy()
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
    tr.add_argument("--microbatch", type=int, default=TRAINING_CONFIG["microbatch_size"]); tr.add_argument("--max-impressions", type=int)
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
    en = sub.add_parser("ensemble")
    en.add_argument("--inputs", nargs="+", required=True); en.add_argument("--output", required=True)
    mb = sub.add_parser("build-membership")
    mb.add_argument("--db", default=str(DEFAULT_DB)); mb.add_argument("--train-behaviors", required=True)
    mb.add_argument("--output")
    a = ap.parse_args()
    if a.command == "train":
        print(json.dumps(run_train(a.partition, a.max_impressions, a.effective_batch,
                                   a.microbatch, a.max_updates, a.seed, a.resume,
                                   a.output_dir, a.db, a.stop_after_updates_this_run), indent=2))
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
                                 a.output, a.db, a.batch_size, a.max_candidates), indent=2))
    elif a.command == "ensemble":
        from .prediction import ensemble
        print(json.dumps(ensemble(a.inputs, a.output), indent=2))


if __name__ == "__main__":
    main()
