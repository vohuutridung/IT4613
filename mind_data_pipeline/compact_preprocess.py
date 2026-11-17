"""Direct raw-MIND to compact typed-array preprocessing for the 15-field model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import CONFIG
from .preprocess import (
    PAD_ID,
    UNK_ID,
    SPLITS,
    VOCAB_FIELDS,
    BEHAVIOR_COLUMNS,
    NEWS_COLUMNS,
    build_counters,
    encode_text,
    iter_tsv,
    make_vocabularies,
    parse_entities,
    row_click,
    vocab_maps,
)


SCHEMA_VERSION = 1
STORAGE_FORMAT = "mind_compact_memmap_v1"
SPLIT_CODE = {"train": 0, "dev": 1, "test": 2}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def write_vocabularies(vocabs, out_dir):
    vocab_dir = Path(out_dir) / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    source_counts = {}
    hashes = {}
    for field, entries in vocabs.items():
        path = vocab_dir / (field + ".tsv")
        if field == "word":
            lines = ["{}\t{}\t\n".format(token_id, token)
                     for token, token_id in entries]
            source_counts[field] = None
        else:
            lines = ["0\t<PAD>\t0\n", "1\t<UNK>\t0\n"]
            lines.extend(
                "{}\t{}\t{}\n".format(index + 2, token.replace("\t", " "), count)
                for index, (token, count) in enumerate(entries[2:]))
            source_counts[field] = [0, 0] + [
                int(count) for _, count in entries[2:]]
        path.write_text("".join(lines), encoding="utf-8", newline="")
        sizes[field] = len(entries)
        hashes[field] = sha256_file(path)
    mapping = hashlib.sha256()
    for field in VOCAB_FIELDS + ("word",):
        mapping.update(field.encode("utf-8") + b"\0")
        mapping.update(bytes.fromhex(hashes[field]))
    did_digest = hashlib.sha256()
    for did_id, (raw_did, _) in enumerate(vocabs["did"][2:], 2):
        did_digest.update("{}\t{}\n".format(did_id, raw_did).encode("utf-8"))
    threshold = int(CONFIG.test_start + CONFIG.min_uid_count)
    uid_counts = source_counts["uid"][2:]
    eligible = 0
    for count in uid_counts:
        if count >= threshold:
            eligible += 1
        else:
            break
    if any(count >= threshold for count in uid_counts[eligible:]):
        raise ValueError("UID vocabulary threshold rows are not a contiguous prefix")
    return {
        "sizes": sizes,
        "tsv_sha256": hashes,
        "mapping_sha256": mapping.hexdigest(),
        "did_mapping_sha256": did_digest.hexdigest(),
        "uid_embedding_rows": 2 + eligible,
    }


def _open_npy(path, dtype, shape, fill=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.dtype(dtype), shape=shape)
    if fill is not None:
        array[...] = fill
    return array


def _open_bin(path, dtype, count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="w+", dtype=np.dtype(dtype), shape=(int(count),))


def build_news(data_root, compact_dir, maps, tokenizer, did_size, limit_per_split=None):
    news_dir = Path(compact_dir) / "news"
    arrays = {
        "present": _open_npy(news_dir / "present.npy", np.bool_, (did_size,), False),
        "source_split": _open_npy(news_dir / "source_split.npy", np.uint8, (did_size,), 0),
        "cat": _open_npy(news_dir / "cat.npy", np.int32, (did_size,), UNK_ID),
        "sub_cat": _open_npy(news_dir / "sub_cat.npy", np.int32, (did_size,), UNK_ID),
        "title": _open_npy(news_dir / "title.npy", np.int32,
                           (did_size, int(CONFIG.max_title_tokens)), PAD_ID),
        "abstract": _open_npy(news_dir / "abstract.npy", np.int32,
                              (did_size, int(CONFIG.max_abstract_tokens)), PAD_ID),
        "title_entities": _open_npy(
            news_dir / "title_entities.npy", np.int32,
            (did_size, int(CONFIG.max_title_entities)), PAD_ID),
        "title_entity_types": _open_npy(
            news_dir / "title_entity_types.npy", np.int32,
            (did_size, int(CONFIG.max_title_entities)), PAD_ID),
        "abstract_entities": _open_npy(
            news_dir / "abstract_entities.npy", np.int32,
            (did_size, int(CONFIG.max_abstract_entities)), PAD_ID),
        "abstract_entity_types": _open_npy(
            news_dir / "abstract_entity_types.npy", np.int32,
            (did_size, int(CONFIG.max_abstract_entities)), PAD_ID),
    }
    seen = {}
    rows = 0
    for split in SPLITS:
        path = Path(data_root) / ("MINDlarge_" + split) / "news.tsv"
        for line_no, row in tqdm(
                iter_tsv(path, NEWS_COLUMNS, limit_per_split),
                desc="compact news {}".format(split), unit="row", dynamic_ncols=True):
            did, cat, sub_cat, title, abstract, _, title_raw, abstract_raw = row
            if did in seen:
                if tuple(row) != seen[did]:
                    raise ValueError(
                        "conflicting duplicate news metadata for {} at {}:{}".format(
                            did, path, line_no))
                continue
            seen[did] = tuple(row)
            did_id = int(maps["did"].get(did, UNK_ID))
            title_ents = parse_entities(
                title_raw, path, line_no)[:int(CONFIG.max_title_entities)]
            abstract_ents = parse_entities(
                abstract_raw, path, line_no)[:int(CONFIG.max_abstract_entities)]
            arrays["present"][did_id] = True
            arrays["source_split"][did_id] = SPLIT_CODE[split]
            arrays["cat"][did_id] = maps["cat"].get(cat, UNK_ID)
            arrays["sub_cat"][did_id] = maps["sub_cat"].get(sub_cat, UNK_ID)
            arrays["title"][did_id] = encode_text(
                tokenizer, title, int(CONFIG.max_title_tokens))
            arrays["abstract"][did_id] = encode_text(
                tokenizer, abstract, int(CONFIG.max_abstract_tokens))
            for index, (entity, typ) in enumerate(title_ents):
                arrays["title_entities"][did_id, index] = maps["entity"].get(
                    entity, UNK_ID)
                arrays["title_entity_types"][did_id, index] = (
                    maps["entity_type"].get(typ, UNK_ID) if typ else PAD_ID)
            for index, (entity, typ) in enumerate(abstract_ents):
                arrays["abstract_entities"][did_id, index] = maps["entity"].get(
                    entity, UNK_ID)
                arrays["abstract_entity_types"][did_id, index] = (
                    maps["entity_type"].get(typ, UNK_ID) if typ else PAD_ID)
            rows += 1
    for array in arrays.values():
        array.flush()
    return rows


def _split_counts(data_manifest, split, limit_per_split):
    entry = data_manifest["splits"]["MINDlarge_" + split]["files"]["behaviors.tsv"]
    rows = int(entry["rows"])
    candidates = int(entry["candidate_tokens"])
    histories = int(entry["history_tokens"])
    if limit_per_split is not None:
        raise ValueError(
            "compact bounded builds require a dedicated audit manifest; "
            "--limit-per-split is not supported")
    return rows, candidates, histories


def build_behaviors(data_root, compact_dir, maps, data_manifest, limit_per_split=None):
    split_metadata = {}
    membership = set()
    impression_base = 1
    candidate_base = 1
    for split in SPLITS:
        rows, candidate_count, history_capacity = _split_counts(
            data_manifest, split, limit_per_split)
        split_dir = Path(compact_dir) / "behaviors" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        uid = _open_npy(split_dir / "uid.npy", np.int32, (rows,))
        impression_id = _open_npy(
            split_dir / "impression_id.npy", "S32", (rows,))
        history_offsets = _open_npy(
            split_dir / "history_offsets.npy", np.int64, (rows + 1,), 0)
        candidate_offsets = _open_npy(
            split_dir / "candidate_offsets.npy", np.int64, (rows + 1,), 0)
        history_values = _open_bin(
            split_dir / "history_did.i32", "<i4", history_capacity)
        candidate_did = _open_bin(
            split_dir / "candidate_did.i32", "<i4", candidate_count)
        click = _open_bin(split_dir / "click.i8", "i1", candidate_count)
        history_cursor = candidate_cursor = impression_cursor = 0
        path = Path(data_root) / ("MINDlarge_" + split) / "behaviors.tsv"
        for line_no, row in tqdm(
                iter_tsv(path, BEHAVIOR_COLUMNS, limit_per_split),
                total=rows, desc="compact behaviors {}".format(split),
                unit="row", dynamic_ncols=True):
            raw_impression, raw_uid, _, history_raw, impressions_raw = row
            if len(raw_impression.encode("utf-8")) > 32:
                raise ValueError("impression ID exceeds 32 bytes at {}:{}".format(
                    path, line_no))
            history_raw_values = history_raw.split()
            missing_history = [
                value for value in history_raw_values if value not in maps["did"]]
            if missing_history:
                raise ValueError(
                    "raw history DID absent from compact vocabulary at {}:{}: {}".format(
                        path, line_no, missing_history[0]))
            history = [
                maps["did"][value] for value in reversed(history_raw_values)
            ][:int(CONFIG.max_history)]
            candidates = impressions_raw.split()
            uid[impression_cursor] = maps["uid"].get(raw_uid, UNK_ID)
            impression_id[impression_cursor] = raw_impression.encode("utf-8")
            if history:
                history_values[history_cursor:history_cursor + len(history)] = history
            history_cursor += len(history)
            history_offsets[impression_cursor + 1] = history_cursor
            for token in candidates:
                raw_did, label = row_click(token)
                if raw_did not in maps["did"]:
                    raise ValueError(
                        "raw candidate DID absent from compact vocabulary at {}:{}: {}".format(
                            path, line_no, raw_did))
                did = maps["did"][raw_did]
                candidate_did[candidate_cursor] = did
                click[candidate_cursor] = -1 if label is None else int(label)
                candidate_cursor += 1
                if split == "train" and did >= 2:
                    membership.add(int(did))
            if split == "train":
                membership.update(
                    int(maps["did"][value]) for value in history_raw_values
                    if int(maps["did"][value]) >= 2)
            candidate_offsets[impression_cursor + 1] = candidate_cursor
            impression_cursor += 1
        if impression_cursor != rows or candidate_cursor != candidate_count:
            raise ValueError(
                "{} count mismatch: impressions {}/{}, candidates {}/{}".format(
                    split, impression_cursor, rows, candidate_cursor, candidate_count))
        uid.flush()
        impression_id.flush()
        history_offsets.flush()
        candidate_offsets.flush()
        history_values.flush()
        candidate_did.flush()
        click.flush()
        del history_values, candidate_did, click
        history_path = split_dir / "history_did.i32"
        with history_path.open("r+b") as stream:
            stream.truncate(history_cursor * np.dtype("<i4").itemsize)
        split_metadata[split] = {
            "impressions": rows,
            "candidates": candidate_count,
            "history_values": history_cursor,
            "impression_base": impression_base,
            "candidate_base": candidate_base,
        }
        impression_base += rows
        candidate_base += candidate_count
    return split_metadata, np.asarray(sorted(membership), dtype=np.int32)


def _compact_files(compact_dir):
    files = []
    for path in sorted(Path(compact_dir).rglob("*")):
        if path.is_file() and path.name != "compact_manifest.json":
            files.append({
                "path": path.relative_to(compact_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return files


def build(data_root, out_dir, data_manifest_path, limit_per_split=None):
    started = time.time()
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    compact_dir = out_dir / "compact"
    if compact_dir.exists() and any(compact_dir.iterdir()):
        raise FileExistsError(
            "{} already contains a compact build; use a new output directory".format(
                compact_dir))
    compact_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads(
        Path(data_manifest_path).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.tokenizer_local, local_files_only=True, use_fast=False)
    counters = build_counters(data_root, limit_per_split)
    vocabs = make_vocabularies(counters, tokenizer)
    maps = vocab_maps(vocabs)
    vocab_meta = write_vocabularies(vocabs, out_dir)
    del counters
    news_rows = build_news(
        data_root, compact_dir, maps, tokenizer,
        vocab_meta["sizes"]["did"], limit_per_split)
    split_metadata, membership = build_behaviors(
        data_root, compact_dir, maps, data_manifest, limit_per_split)
    del maps, vocabs
    layout = {
        "storage_format": STORAGE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "source_commit": CONFIG.source_commit,
        "vocab": vocab_meta,
        "splits": split_metadata,
        "news_rows": news_rows,
        "limits": {
            key: int(getattr(CONFIG, key))
            for key in (
                "max_title_tokens", "max_abstract_tokens", "max_history",
                "max_history_titles", "max_history_abstracts",
                "max_title_entities", "max_abstract_entities",
                "max_history_title_entities_per_news",
                "max_history_abstract_entities_per_news")
        },
        "raw_data_manifest_sha256": sha256_file(data_manifest_path),
    }
    layout["schema_sha256"] = hashlib.sha256(json.dumps(
        {key: layout[key] for key in (
            "storage_format", "schema_version", "limits")},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    _atomic_json(compact_dir / "layout.json", layout)
    files = _compact_files(compact_dir)
    payload = hashlib.sha256(json.dumps(
        {row["path"]: row["sha256"] for row in files},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    identity = {
        "bytes": sum(int(row["bytes"]) for row in files),
        "sha256": payload,
        "schema_sha256": layout["schema_sha256"],
        "mapping_sha256": vocab_meta["mapping_sha256"],
        "did_mapping_sha256": vocab_meta["did_mapping_sha256"],
    }
    manifest = {
        "status": "MIND_COMPACT_MEMMAP_READY",
        "storage_format": STORAGE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "files": files,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _atomic_json(compact_dir / "compact_manifest.json", manifest)
    membership_dir = out_dir / "membership"
    membership_dir.mkdir(parents=True, exist_ok=True)
    membership_path = membership_dir / "train_did_membership.npy"
    np.save(membership_path, membership, allow_pickle=False)
    membership_manifest = {
        "status": "TRAIN_BEHAVIOR_DID_MEMBERSHIP_BUILT",
        "schema_version": 2,
        "membership_definition":
            "unique raw DIDs appearing in train behavior history or impression candidates; no labels used",
        "source_repository_commit": CONFIG.source_commit,
        "split": "train",
        "labels_used": False,
        "candidate_label_suffix_used": False,
        "database_identity": identity,
        "did_mapping_sha256": identity["did_mapping_sha256"],
        "unique_train_dids": int(len(membership)),
        "dtype": "int32",
        "sorted_unique": True,
        "matrix_path": str(membership_path),
        "matrix_bytes": membership_path.stat().st_size,
        "matrix_sha256": sha256_file(membership_path),
        "preprocessing_or_db_write": False,
    }
    _atomic_json(
        Path(str(membership_path) + ".manifest.json"), membership_manifest)
    artifact_files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name not in {
                "artifact_manifest.json", "prepared_bundle_manifest.json"}:
            artifact_files.append({
                "path": path.relative_to(out_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    artifact_manifest = {
        "scope": "full",
        "storage_format": STORAGE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": identity["schema_sha256"],
        "mapping_sha256": identity["mapping_sha256"],
        "artifacts": artifact_files,
        "state": "COMPACT_STAGE01_READY; EMBEDDINGS_MISSING",
    }
    _atomic_json(out_dir / "artifact_manifest.json", artifact_manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-manifest", default="artifacts/audit/data_manifest.json")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    print(json.dumps(build(
        args.data_root, args.out, args.data_manifest, args.limit_per_split),
        indent=2))


if __name__ == "__main__":
    main()
