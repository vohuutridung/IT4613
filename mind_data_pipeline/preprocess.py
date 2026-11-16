"""Streaming, restartable stage-01 preprocessing for the checked MIND-large files.

This is an explicitly documented reimplementation, not the author's private code.
It stores news content once by did and stores histories once per impression; each
candidate references its impression and has a separate label row.
"""
import argparse
import array
import csv
import hashlib
import json
import os
import sqlite3
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

from transformers import AutoTokenizer

from .config import CONFIG


PAD_ID = int(CONFIG.pad_id)
UNK_ID = int(CONFIG.unk_id)
SPLITS = ("train", "dev", "test")
VOCAB_FIELDS = ("uid", "did", "cat", "sub_cat", "entity", "entity_type")
NEWS_COLUMNS = 8
BEHAVIOR_COLUMNS = 5
PREPROCESSING_VERSION = "2.0.0"
SCHEMA_VERSION = "1"


class StableCounter:
    def __init__(self):
        self.counts = OrderedDict()

    def add(self, token, count=1):
        if token not in self.counts:
            self.counts[token] = 0
        self.counts[token] += int(count)

    def ordered(self):
        return [item for _, item in sorted(enumerate(self.counts.items()),
                                           key=lambda x: (-x[1][1], x[0]))]


def iter_tsv(path, expected, limit=None):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_no, row in enumerate(reader, 1):
            if limit is not None and line_no > limit:
                break
            if len(row) != expected:
                raise ValueError("{}:{} expected {} columns, got {}".format(path, line_no, expected, len(row)))
            yield line_no, row


def parse_entities(raw, source, line_no):
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except Exception as e:
        raise ValueError("invalid entity JSON at {}:{}: {}".format(source, line_no, e))
    if not isinstance(value, list):
        raise ValueError("entity JSON is not a list at {}:{}".format(source, line_no))
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        wid = str(item.get("WikidataId", ""))
        typ = str(item.get("Type", ""))
        if wid:
            out.append((wid, typ))
    return out


def row_click(token):
    if "-" not in token:
        return token, None
    did, suffix = token.rsplit("-", 1)
    if suffix in ("0", "1"):
        return did, int(suffix)
    return token, None


def build_counters(data_root, limit_per_split=None):
    counters = {k: StableCounter() for k in VOCAB_FIELDS}
    seen_uids, seen_dids, seen_news = set(), set(), {}
    for split in SPLITS:
        started = time.time()
        print("[vocab] behaviors {} start".format(split), file=sys.stderr, flush=True)
        p = Path(data_root) / ("MINDlarge_" + split) / "behaviors.tsv"
        for line_no, row in iter_tsv(p, BEHAVIOR_COLUMNS, limit_per_split):
            _, uid, _, history_raw, impressions_raw = row
            weighted_uid = 1 if split == "test" or uid in seen_uids else CONFIG.test_start
            counters["uid"].add(uid, weighted_uid)
            history = history_raw.split()
            # The source notebook counts the stored history order. Reversal is
            # applied later when materializing model history records.
            for raw_did in history:
                if uid not in seen_uids:
                    _add_behavior_did(counters["did"], seen_dids, raw_did, split)
                _add_behavior_did(counters["did"], seen_dids, raw_did, split)
            for candidate in impressions_raw.split():
                raw_did, _ = row_click(candidate)
                _add_behavior_did(counters["did"], seen_dids, raw_did, split)
            seen_uids.add(uid)
            if line_no % 250000 == 0:
                print("[vocab] behaviors {} line={} users={} dids={}".format(
                    split, line_no, len(seen_uids), len(seen_dids)), file=sys.stderr, flush=True)
        print("[vocab] behaviors {} done in {:.1f}s (uids={}, dids={})".format(
            split, time.time() - started, len(counters['uid'].counts), len(counters['did'].counts)),
            file=sys.stderr, flush=True)

    # Reproduce the notebook's content counter weighting for cat/subcat/entities.
    # The only deliberate mapping extension is to append metadata-only news IDs
    # after interaction IDs so they cannot all collide at UNK in the lookup.
    for split in SPLITS:
        started = time.time()
        print("[vocab] news {} start".format(split), file=sys.stderr, flush=True)
        p = Path(data_root) / ("MINDlarge_" + split) / "news.tsv"
        for line_no, row in iter_tsv(p, NEWS_COLUMNS, limit_per_split):
            did, cat, sub_cat, _, _, _, title_raw, abstract_raw = row
            if did in seen_news:
                if tuple(row) != seen_news[did]:
                    raise ValueError("conflicting duplicate news metadata for {} at {}:{}".format(did, p, line_no))
                continue
            seen_news[did] = tuple(row)
            if did not in counters["did"].counts:
                counters["did"].add(did, 0)
            cat_weight = 1 if split == "test" or cat in counters["cat"].counts else CONFIG.test_start
            sub_weight = 1 if split == "test" or sub_cat in counters["sub_cat"].counts else CONFIG.test_start
            counters["cat"].add(cat, cat_weight)
            counters["sub_cat"].add(sub_cat, sub_weight)
            raw_did_count = counters["did"].counts.get(did, 0)
            did_count = raw_did_count - CONFIG.test_start + 1 if raw_did_count >= CONFIG.test_start else raw_did_count
            for raw, target in ((title_raw, "entity"), (abstract_raw, "entity")):
                for wid, typ in parse_entities(raw, p, line_no):
                    weight = (0 if split == "test" or wid in counters["entity"].counts else CONFIG.test_start) + did_count
                    counters["entity"].add(wid, weight)
                    if typ:
                        type_weight = (0 if split == "test" or typ in counters["entity_type"].counts else CONFIG.test_start) + did_count
                        counters["entity_type"].add(typ, type_weight)
            if line_no % 50000 == 0:
                print("[vocab] news {} line={} unique_news={} entities={}".format(
                    split, line_no, len(seen_news), len(counters['entity'].counts)),
                    file=sys.stderr, flush=True)
        print("[vocab] news {} done in {:.1f}s (news={}, entities={})".format(
            split, time.time() - started, len(seen_news), len(counters['entity'].counts)),
            file=sys.stderr, flush=True)
    return counters


def _add_behavior_did(counter, seen_dids, did, split):
    weight = 1 if split == "test" or did in seen_dids else CONFIG.test_start
    counter.add(did, weight)
    seen_dids.add(did)


def make_vocabularies(counters, tokenizer):
    if (PAD_ID, UNK_ID) != (0, 1):
        raise ValueError("the active vocabulary format requires configured PAD=0 and UNK=1")
    vocabs = {}
    for field, counter in counters.items():
        entries = [("<PAD>", PAD_ID), ("<UNK>", UNK_ID)]
        entries.extend(counter.ordered())
        # Special spellings in raw data must not shadow the reserved rows.
        entries = entries[:2] + [(t, c) for t, c in entries[2:] if t not in ("<PAD>", "<UNK>")]
        vocabs[field] = entries
    word_vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    if word_vocab[0][0] != "[PAD]" or word_vocab[0][1] != PAD_ID:
        raise ValueError("candidate tokenizer PAD ID differs from the configured PAD ID")
    vocabs["word"] = [(token, int(token_id)) for token, token_id in word_vocab]
    return vocabs


def vocab_maps(vocabs):
    maps = {}
    for field, entries in vocabs.items():
        if field == "word":
            maps[field] = {token: token_id for token, token_id in entries}
        else:
            maps[field] = {token: i for i, (token, _) in enumerate(entries)}
    return maps


def pack_ids(values):
    if not values:
        return b""
    return struct.pack("<{}I".format(len(values)), *[int(x) for x in values])


def unpack_ids(blob, length=None):
    if blob is None or not blob:
        return [] if length is None else [PAD_ID] * length
    n = len(blob) // 4
    values = list(struct.unpack("<{}I".format(n), blob))
    if length is not None:
        values = values[:length] + [PAD_ID] * max(0, length - len(values))
    return values


def encode_text(tokenizer, text, max_len):
    # Match the notebook path: encode with special boundaries, strip [1:-1],
    # then right-pad/truncate. Empty text becomes an all-PAD row.
    if not text:
        return [PAD_ID] * max_len
    raw = tokenizer.encode(text, add_special_tokens=True)
    ids = raw[1:-1][:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS vocab(field TEXT NOT NULL, id INTEGER NOT NULL, token TEXT NOT NULL, source_count INTEGER, PRIMARY KEY(field,id), UNIQUE(field,token));
      CREATE TABLE IF NOT EXISTS news(did INTEGER PRIMARY KEY, news_id TEXT NOT NULL UNIQUE, source_split TEXT NOT NULL, cat_id INTEGER NOT NULL, sub_cat_id INTEGER NOT NULL,
        title_ids BLOB NOT NULL, abstract_ids BLOB NOT NULL, body_ids BLOB,
        title_entity_ids BLOB NOT NULL, title_entity_types BLOB NOT NULL,
        abstract_entity_ids BLOB NOT NULL, abstract_entity_types BLOB NOT NULL);
      CREATE TABLE IF NOT EXISTS impressions(id INTEGER PRIMARY KEY, split TEXT NOT NULL, line_no INTEGER NOT NULL, impression_id TEXT NOT NULL,
        uid INTEGER NOT NULL, time_text TEXT, history_ids BLOB NOT NULL, history_len INTEGER NOT NULL, UNIQUE(split,line_no));
      CREATE TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY, impression_id INTEGER NOT NULL, did INTEGER NOT NULL, position INTEGER NOT NULL,
        FOREIGN KEY(impression_id) REFERENCES impressions(id), UNIQUE(impression_id,position));
      CREATE TABLE IF NOT EXISTS labels(candidate_id INTEGER PRIMARY KEY, click INTEGER, labeled INTEGER NOT NULL,
        FOREIGN KEY(candidate_id) REFERENCES candidates(id));
      CREATE INDEX IF NOT EXISTS candidates_impression_idx ON candidates(impression_id);
      CREATE INDEX IF NOT EXISTS news_id_idx ON news(news_id);
    """)
    return conn


def write_vocabularies(conn, vocabs, out_dir):
    vdir = Path(out_dir) / "vocab"
    vdir.mkdir(parents=True, exist_ok=True)
    for field, entries in vocabs.items():
        path = vdir / (field + ".tsv")
        if field == "word":
            lines = ["{}\t{}\t\n".format(token_id, token) for token, token_id in entries]
        else:
            lines = ["{}\t<PAD>\t0\n".format(PAD_ID), "{}\t<UNK>\t0\n".format(UNK_ID)]
            lines.extend("{}\t{}\t{}\n".format(i + 2, token.replace("\t", " "), count) for i, (token, count) in enumerate(entries[2:]))
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        rows = []
        if field == "word":
            rows = [(field, int(token_id), token, None) for token, token_id in entries]
        else:
            rows = [(field, PAD_ID, "<PAD>", 0), (field, UNK_ID, "<UNK>", 0)]
            rows.extend((field, i + 2, token, int(count)) for i, (token, count) in enumerate(entries[2:]))
        conn.executemany("INSERT OR REPLACE INTO vocab(field,id,token,source_count) VALUES(?,?,?,?)", rows)
        conn.commit()


def load_external_body(body_json, nid_map_path):
    """Optional exact-artifact interface; requires an explicit nid->news-id map."""
    if not body_json:
        return {}
    if not nid_map_path:
        raise ValueError("--nid-map is required with --body-json")
    nid_map = json.load(open(nid_map_path, encoding="utf-8"))
    root = json.load(open(body_json, encoding="utf-8"))
    entries = root if isinstance(root, list) else root.values()
    result = {}
    for obj in entries:
        nid = str(obj.get("nid", obj.get("id", "")))
        news_id = nid_map.get(nid)
        body = obj.get("body", "")
        if isinstance(body, list):
            body = " ".join(str(x.get("text", "") if isinstance(x, dict) else x) for x in body)
        if news_id and body:
            result[str(news_id)] = str(body)
    return result


def insert_news(conn, data_root, maps, tokenizer, body_map, limit_per_split=None):
    # Re-scan safely after interruption. INSERT OR IGNORE makes this phase
    # idempotent while avoiding a false "complete" marker after a partial run.
    seen = {}
    insert_sql = "INSERT OR IGNORE INTO news VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
    n_total = 0
    for split in SPLITS:
        started = time.time()
        print("[lookup] news {} start".format(split), file=sys.stderr, flush=True)
        path = Path(data_root) / ("MINDlarge_" + split) / "news.tsv"
        batch = []
        for line_no, row in iter_tsv(path, NEWS_COLUMNS, limit_per_split):
            did, cat, sub_cat, title, abstract, url, title_raw, abstract_raw = row
            if did in seen:
                if tuple(row) != seen[did]:
                    raise ValueError("conflicting duplicate news metadata for {} at {}:{}".format(did, path, line_no))
                continue
            seen[did] = tuple(row)
            did_id = maps["did"].get(did, UNK_ID)
            title_ents = parse_entities(title_raw, path, line_no)[:CONFIG.max_title_entities]
            abstract_ents = parse_entities(abstract_raw, path, line_no)[:CONFIG.max_abstract_entities]
            title_e = [maps["entity"].get(x, UNK_ID) for x, _ in title_ents]
            title_t = [maps["entity_type"].get(t, UNK_ID) if t else PAD_ID for _, t in title_ents]
            abstract_e = [maps["entity"].get(x, UNK_ID) for x, _ in abstract_ents]
            abstract_t = [maps["entity_type"].get(t, UNK_ID) if t else PAD_ID for _, t in abstract_ents]
            title_e = title_e + [PAD_ID] * (CONFIG.max_title_entities - len(title_e))
            title_t = title_t + [PAD_ID] * (CONFIG.max_title_entities - len(title_t))
            abstract_e = abstract_e + [PAD_ID] * (CONFIG.max_abstract_entities - len(abstract_e))
            abstract_t = abstract_t + [PAD_ID] * (CONFIG.max_abstract_entities - len(abstract_t))
            body = body_map.get(did)
            body_ids = pack_ids(encode_text(tokenizer, body, CONFIG.max_body_tokens)) if body else None
            batch.append((did_id, did, split, maps["cat"].get(cat, UNK_ID), maps["sub_cat"].get(sub_cat, UNK_ID),
                          pack_ids(encode_text(tokenizer, title, CONFIG.max_title_tokens)),
                          pack_ids(encode_text(tokenizer, abstract, CONFIG.max_abstract_tokens)), body_ids,
                          pack_ids(title_e), pack_ids(title_t), pack_ids(abstract_e), pack_ids(abstract_t)))
            if len(batch) >= 512:
                conn.executemany(insert_sql, batch)
                if body_map:
                    conn.executemany("UPDATE news SET body_ids=? WHERE did=?",
                                     [(row[7], row[0]) for row in batch if row[7] is not None])
                conn.commit(); n_total += len(batch); batch.clear()
        if batch:
            conn.executemany(insert_sql, batch)
            if body_map:
                conn.executemany("UPDATE news SET body_ids=? WHERE did=?",
                                 [(row[7], row[0]) for row in batch if row[7] is not None])
            conn.commit(); n_total += len(batch)
        print("[lookup] news {} done in {:.1f}s (rows attempted={})".format(
            split, time.time() - started, n_total), file=sys.stderr, flush=True)
    actual = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.execute("INSERT OR REPLACE INTO meta VALUES('news_lookup_rows',?)", (str(actual),)); conn.commit()


def insert_records(conn, data_root, maps, limit_per_split=None, resume=False):
    result = {}
    for split in SPLITS:
        started = time.time()
        print("[records] {} start".format(split), file=sys.stderr, flush=True)
        max_line = 0
        if resume:
            max_line = conn.execute("SELECT COALESCE(MAX(line_no),0) FROM impressions WHERE split=?", (split,)).fetchone()[0]
        path = Path(data_root) / ("MINDlarge_" + split) / "behaviors.tsv"
        batch_impressions, batch_candidates, batch_labels = [], [], []
        n_imp = n_cand = 0
        for line_no, row in iter_tsv(path, BEHAVIOR_COLUMNS, limit_per_split):
            if line_no <= max_line:
                continue
            impression, uid, time_text, history_raw, impressions_raw = row
            history = [maps["did"].get(x, UNK_ID) for x in reversed(history_raw.split())]
            history = history[:CONFIG.max_history]
            if len(batch_impressions) > 2000:
                _flush_records(conn, batch_impressions, batch_candidates, batch_labels)
                n_imp += len(batch_impressions); n_cand += len(batch_candidates)
                batch_impressions, batch_candidates, batch_labels = [], [], []
            # IDs are assigned in a deterministic second insert so child rows can
            # reference the impression without duplicating its history payload.
            batch_impressions.append((split, line_no, impression, maps["uid"].get(uid, UNK_ID), time_text,
                                      pack_ids(history), len(history)))
            for pos, token in enumerate(impressions_raw.split()):
                raw_did, click = row_click(token)
                did = maps["did"].get(raw_did, UNK_ID)
                batch_candidates.append((line_no, pos, did, click, int(click is not None)))
        _flush_records(conn, batch_impressions, batch_candidates, batch_labels, split)
        n_imp += len(batch_impressions); n_cand += len(batch_candidates)
        result[split] = {'impressions_added': n_imp, 'candidates_added': n_cand, 'resumed_after_line': max_line}
        print("[records] {} done in {:.1f}s (impressions added={}, candidates added={})".format(
            split, time.time() - started, n_imp, n_cand), file=sys.stderr, flush=True)
    return result


def _flush_records(conn, impressions, candidates, labels_unused, split=None):
    if not impressions:
        return
    conn.execute("BEGIN")
    conn.executemany("INSERT OR IGNORE INTO impressions(split,line_no,impression_id,uid,time_text,history_ids,history_len) VALUES(?,?,?,?,?,?,?)", impressions)
    line_ids = [x[1] for x in impressions]
    q = "SELECT id,line_no FROM impressions WHERE split=? AND line_no BETWEEN ? AND ?"
    ids = {line_no: impression_id for impression_id, line_no in
           conn.execute(q, (split or impressions[0][0], min(line_ids), max(line_ids))).fetchall()}
    by_line = {line: ids[line] for line in line_ids if line in ids}
    cand_rows, label_rows = [], []
    # candidates passed as (line_no, position, did, click, labeled).
    for line_no, pos, did, click, labeled in candidates:
        imp_id = by_line[line_no]
        cand_rows.append((imp_id, did, pos))
    conn.executemany("INSERT OR IGNORE INTO candidates(impression_id,did,position) VALUES(?,?,?)", cand_rows)
    # Bulk-read candidate IDs (avoid one SQL query per impression).
    candidate_ids = {}
    imp_values = list(by_line.values())
    for start in range(0, len(imp_values), 800):
        part = imp_values[start:start + 800]
        marks = ",".join("?" for _ in part)
        sql = "SELECT id,impression_id,position FROM candidates WHERE impression_id IN (" + marks + ")"
        candidate_ids.update({(imp_id, pos): cand_id for cand_id, imp_id, pos in conn.execute(sql, part)})
    candidates_by_imp = {}
    for (imp_id, pos), cand_id in candidate_ids.items():
        candidates_by_imp.setdefault(imp_id, []).append((pos, cand_id))
    # Join labels by unique impression+position; labels remain a separate table.
    label_lookup = {(line, pos): (click, labeled) for line, pos, _, click, labeled in candidates}
    for line_no in line_ids:
        imp_id = by_line[line_no]
        for pos, cand_id in candidates_by_imp.get(imp_id, []):
            click, labeled = label_lookup.get((line_no, pos), (None, 0))
            label_rows.append((cand_id, click, labeled))
    conn.executemany("INSERT OR REPLACE INTO labels(candidate_id,click,labeled) VALUES(?,?,?)", label_rows)
    conn.commit()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def _fingerprint_sha(value):
    return _canonical_sha({key: value.get(key) for key in
                           ("core_sha256", "schema_sha256", "mapping_sha256")})


def _file_fingerprint(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def _tree_fingerprint(path):
    path = Path(path)
    result = {}
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        result[str(item.relative_to(path))] = _file_fingerprint(item)
    return result


def _input_fingerprint(data_root, body_json, nid_map, limit_per_split):
    source = {}
    for split in SPLITS:
        base = Path(data_root) / ("MINDlarge_" + split)
        for name in ("behaviors.tsv", "news.tsv"):
            path = base / name
            if not path.is_file():
                raise FileNotFoundError(path)
            source[split + "/" + name] = _file_fingerprint(path)
    body = None
    if body_json or nid_map:
        if not body_json or not nid_map:
            raise ValueError("--body-json and --nid-map must be supplied together")
        body = {"body_json": _file_fingerprint(body_json), "nid_map": _file_fingerprint(nid_map)}
    tok = Path(CONFIG.tokenizer_local)
    if not tok.is_dir():
        raise FileNotFoundError("local tokenizer directory not found: {}".format(tok))
    tokenizer = {"repo": CONFIG.tokenizer_repo, "revision": CONFIG.tokenizer_revision,
                 "files": _tree_fingerprint(tok)}
    code = {name: _file_fingerprint(Path(__file__).resolve().parent / name)
            for name in ("preprocess.py", "config.py")}
    return {
      "preprocessing_version": PREPROCESSING_VERSION,
      "schema_version": SCHEMA_VERSION,
      "source_commit": CONFIG.source_commit,
      "parameters": CONFIG.as_dict(),
      "limit_per_split": limit_per_split,
      "source_files": source,
      "body_inputs": body,
      "tokenizer": tokenizer,
      "code": code,
    }


def _mapping_fingerprint(out_dir, conn):
    h = hashlib.sha256()
    for field in VOCAB_FIELDS + ("word",):
        path = Path(out_dir) / "vocab" / (field + ".tsv")
        if not path.is_file():
            raise ValueError("vocabulary file missing: {}".format(path))
        h.update(field.encode("utf-8") + b"\0")
        h.update(bytes.fromhex(sha256(path)))
    for row in conn.execute("SELECT field,id,token,source_count FROM vocab ORDER BY field,id"):
        h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    return h.hexdigest()


def _schema_fingerprint(conn):
    rows = conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master "
                        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
    return _canonical_sha(rows)


def _legacy_reconciliation(out_dir, db_path, core):
    """Reconcile the previously built full DB from retained hashes, without rewriting it."""
    out_dir = Path(out_dir)
    db_path = Path(db_path)
    prior_path = out_dir / "resolved_config.json"
    manifest_path = out_dir / "artifact_manifest.json"
    lineage_path = Path("artifacts/project_manifest.json")
    if not (prior_path.is_file() and manifest_path.is_file() and lineage_path.is_file() and db_path.is_file()):
        raise ValueError("resume refused: artifact has no fingerprint and retained reconciliation evidence is incomplete")
    prior = json.load(open(prior_path, encoding="utf-8"))
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    lineage = json.load(open(lineage_path, encoding="utf-8"))
    if prior.get("preprocessing_fingerprint"):
        raise ValueError("legacy reconciliation called for artifact that already has a fingerprint")
    if prior.get("parameters") != CONFIG.as_dict() or prior.get("limit_per_split") != core["limit_per_split"]:
        raise ValueError("resume refused: legacy config does not match current parameters/scope")
    if prior.get("source_commit") != CONFIG.source_commit:
        raise ValueError("resume refused: legacy source commit differs")
    if prior.get("tokenizer", {}).get("repo") != CONFIG.tokenizer_repo or prior.get("tokenizer", {}).get("revision") != CONFIG.tokenizer_revision:
        raise ValueError("resume refused: legacy tokenizer revision differs")
    prior_body = prior.get("body_input")
    current_body = core["body_inputs"]
    if prior_body:
        if current_body is None:
            raise ValueError("resume refused: legacy artifact used body inputs that were not supplied for reconciliation")
        if (prior_body.get("body_json_sha256") != current_body["body_json"]["sha256"] or
                prior_body.get("nid_map_sha256") != current_body["nid_map"]["sha256"]):
            raise ValueError("resume refused: legacy body input checksums differ")
    elif current_body is not None:
        raise ValueError("resume refused: legacy artifact did not record the supplied body inputs")
    tokenizer_provenance_path = Path("tokenizer_artifact.json")
    if not tokenizer_provenance_path.is_file():
        raise ValueError("resume refused: no retained tokenizer file checksums for legacy artifact")
    tokenizer_provenance = json.load(open(tokenizer_provenance_path, encoding="utf-8"))
    if tokenizer_provenance.get("revision") != CONFIG.tokenizer_revision:
        raise ValueError("resume refused: retained tokenizer artifact revision differs")
    expected_token_files = {
      Path(item["path"]).name: {"bytes": item["bytes"], "sha256": item["sha256"]}
      for item in tokenizer_provenance.get("downloaded_files", [])
    }
    current_token_files = {Path(name).name: item for name, item in core["tokenizer"]["files"].items()}
    if expected_token_files != current_token_files:
        raise ValueError("resume refused: local tokenizer bytes differ from retained provenance")

    # Verify raw bytes against the still-present data manifest, then compare their
    # freshly computed fingerprints to the manifest entries.
    data_manifest_path = Path("artifacts/audit/data_manifest.json")
    if not data_manifest_path.is_file():
        raise ValueError("resume refused: no retained raw-data manifest for legacy artifact")
    data_manifest = json.load(open(data_manifest_path, encoding="utf-8"))
    data_manifest_sha = sha256(data_manifest_path)
    if prior.get("data_manifest", {}).get("sha256") != data_manifest_sha:
        raise ValueError("resume refused: legacy data-manifest checksum differs")
    for split in SPLITS:
        entry = data_manifest["splits"]["MINDlarge_" + split]["files"]
        for name in ("behaviors.tsv", "news.tsv"):
            actual = core["source_files"][split + "/" + name]
            expected = entry[name]
            if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
                raise ValueError("resume refused: current raw source differs from retained manifest: {}/{}".format(split, name))

    files = {item["path"]: item for item in manifest.get("artifacts", [])}
    required = ["stage01.sqlite", "resolved_config.json"] + ["vocab/" + f + ".tsv" for f in VOCAB_FIELDS + ("word",)]
    for rel in required:
        item = files.get(rel)
        path = out_dir / rel
        if not item or not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("resume refused: retained artifact checksum does not verify: {}".format(rel))
    code_lineage = {item["path"]: item for item in lineage.get("code", [])}
    for rel in ("mind_stage12/preprocess.py", "mind_stage12/config.py", "mind_stage12/data.py"):
        if rel not in code_lineage:
            raise ValueError("resume refused: no retained code lineage for {}".format(rel))

    conn = sqlite3.connect("file:{}?mode=ro".format(Path(db_path).resolve()), uri=True)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not set(("meta", "vocab", "news", "impressions", "candidates", "labels")).issubset(tables):
            raise ValueError("resume refused: legacy database schema is incomplete")
        counts = {split: {
            "impressions": conn.execute("SELECT COUNT(*) FROM impressions WHERE split=?", (split,)).fetchone()[0],
            "candidates": conn.execute("SELECT COUNT(*) FROM candidates c JOIN impressions i ON i.id=c.impression_id WHERE i.split=?", (split,)).fetchone()[0]
        } for split in SPLITS}
        expected = {split: {"impressions": data_manifest["splits"]["MINDlarge_" + split]["files"]["behaviors.tsv"]["rows"],
                            "candidates": data_manifest["splits"]["MINDlarge_" + split]["files"]["behaviors.tsv"]["candidate_tokens"]}
                    for split in SPLITS}
        if counts != expected:
            raise ValueError("resume refused: legacy DB split counts differ from retained raw manifest")
        schema_sha = _schema_fingerprint(conn)
    finally:
        conn.close()
    return {"status": "RECONCILED_FROM_RETAINED_EVIDENCE",
            "database_sha256": files["stage01.sqlite"]["sha256"],
            "resolved_config_sha256": files["resolved_config.json"]["sha256"],
            "vocabulary_sha256": {rel: files[rel]["sha256"] for rel in required if rel.startswith("vocab/")},
            "pre_patch_code_sha256": {rel: code_lineage[rel]["sha256"] for rel in
                                       ("mind_stage12/preprocess.py", "mind_stage12/config.py", "mind_stage12/data.py")},
            "raw_manifest_sha256": data_manifest_sha,
            "tokenizer_provenance_sha256": sha256(tokenizer_provenance_path),
            "split_counts": counts,
            "schema_sha256": schema_sha}


def _preflight_resume(out_dir, db_path, core):
    """Read and compare all resume fingerprints before the first write."""
    out_dir = Path(out_dir)
    config_path, state_path = out_dir / "resolved_config.json", out_dir / "state.json"
    db_exists = Path(db_path).is_file()
    if not (config_path.exists() or db_exists or state_path.exists()):
        return None, None
    if not (config_path.is_file() and db_exists):
        raise ValueError("resume refused: incomplete output has no verifiable config/database pair")
    prior = json.load(open(config_path, encoding="utf-8"))
    saved = prior.get("preprocessing_fingerprint")
    if saved is None:
        proof = _legacy_reconciliation(out_dir, db_path, core)
        return None, proof
    if saved.get("core_sha256") != _canonical_sha(core):
        raise ValueError("resume refused: source/config/tokenizer/schema/code fingerprint differs")
    conn = sqlite3.connect("file:{}?mode=ro".format(Path(db_path).resolve()), uri=True)
    try:
        actual_schema = _schema_fingerprint(conn)
        if actual_schema != saved.get("schema_sha256"):
            raise ValueError("resume refused: SQLite schema fingerprint differs")
        actual_mapping = (_mapping_fingerprint(out_dir, conn)
                          if saved.get("mapping_sha256") is not None else None)
    finally:
        conn.close()
    if saved.get("mapping_sha256") is not None and actual_mapping != saved.get("mapping_sha256"):
        raise ValueError("resume refused: vocabulary mapping fingerprint differs")
    meta = sqlite3.connect("file:{}?mode=ro".format(Path(db_path).resolve()), uri=True)
    try:
        db_saved = meta.execute("SELECT value FROM meta WHERE key='preprocessing_fingerprint'").fetchone()
    finally:
        meta.close()
    if not db_saved or db_saved[0] != saved.get("fingerprint_sha256"):
        raise ValueError("resume refused: DB fingerprint marker missing or differs")
    # A null mapping hash is a legitimate pre-vocabulary checkpoint created by
    # this version. The run regenerates vocabularies before writing lookup rows.
    return saved, None


def prepare(data_root, out_dir, limit_per_split=None, resume=False, body_json=None, nid_map=None):
    data_root, out_dir = Path(data_root), Path(out_dir)
    db_path = out_dir / "stage01.sqlite"
    state_path = out_dir / "state.json"
    if (state_path.exists() or db_path.exists()) and not resume:
        raise FileExistsError("{} exists; use --resume or a new --out".format(state_path))
    core_fingerprint = _input_fingerprint(data_root, body_json, nid_map, limit_per_split)
    saved_fingerprint, legacy_proof = None, None
    if resume and (state_path.exists() or db_path.exists() or (out_dir / "resolved_config.json").exists()):
        saved_fingerprint, legacy_proof = _preflight_resume(out_dir, db_path, core_fingerprint)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_local, local_files_only=True, use_fast=False)
    cfg_path = out_dir / "resolved_config.json"
    resolved = {
      'implementation': 'independent stage 01 reimplementation',
      'source_commit': CONFIG.source_commit,
      'profile': CONFIG.profile,
      'parameters': CONFIG.as_dict(),
      'parameter_sources': {
        'embedding_dim': 'upstream-mind/projects/ai/mind/src/train/v17/base.sh --emb_size=100',
        'token_lengths_history': 'upstream-mind/projects/ai/mind/prepare/config.py and src/train/v17/common.sh',
        'reverse_history_and_test_start': 'upstream-mind/projects/ai/mind/prepare/gen-records.py:109 and juypter/behaviors.ipynb cell 2',
        'tokenizer': 'juypter/dump-doc-lookup-Copy1.ipynb cells 4,19,23; official HF candidate revision in tokenizer_artifact.json',
        'din_activation_normalization': 'src/config.py att_activation=sigmoid, din_normalize=true',
        'attention_hidden_activation': 'public wenzheng candidate Pooling source; this reimplementation uses hidden={}, {}'.format(CONFIG.attention_hidden, CONFIG.attention_activation),
      },
      'decisions': {
        'pad_unk_for_nonword_vocabs': "<PAD>={}, <UNK>={}; exact author vocabulary special rows are unverified".format(PAD_ID, UNK_ID),
        'vocabulary_tie_break': 'stable first-seen order after source-derived weighted counts',
        'metadata_only_news': 'append after behavior news IDs to avoid all metadata-only articles sharing one UNK lookup row',
        'abstract_entities': 'read abstract_entities array correctly; intentional fix to confirmed upstream stale-title-variable bug',
        'freshness': 'omit start_times/timestamps from encoder input because use_fresh=false in profile; no freshness feature added',
        'label_isolation': 'candidate labels stored in separate SQLite table; batch feature dictionary excludes labels',
        'body': 'NULL when original body corpus is absent; no abstract fallback',
        'glove': 'loader interface supported; absent artifact is only synthesized in smoke test',
        'news_duplicate_policy': 'train -> dev -> test; first occurrence wins when full raw rows match; conflicting duplicate metadata raises an error',
        'news_duplicate_audit': {'duplicate_occurrences': 164132, 'conflicting_rows': 0,
                                 'evidence': 'retained independent review raw comparison; no rescan during remediation'},
      },
      'tokenizer': {'repo': CONFIG.tokenizer_repo, 'revision': CONFIG.tokenizer_revision,
                    'candidate_not_proven_original_local_artifact': True, 'use_fast': False},
      'limit_per_split': limit_per_split,
      'resumable': True,
      'preprocessing_fingerprint': None,
      'body_input': ({'body_json': str(Path(body_json).resolve()), 'body_json_sha256': sha256(body_json),
                      'nid_map': str(Path(nid_map).resolve()), 'nid_map_sha256': sha256(nid_map)}
                     if body_json and nid_map else None),
      'data_manifest': {'path': 'artifacts/audit/data_manifest.json',
                        'sha256': sha256('artifacts/audit/data_manifest.json')
                        if Path('artifacts/audit/data_manifest.json').is_file() else None},
    }
    if cfg_path.exists() and resume and saved_fingerprint is None and legacy_proof is None:
        prior = json.load(open(cfg_path, encoding='utf-8'))
        if prior.get('limit_per_split') != limit_per_split:
            raise ValueError('resume scope differs from saved run')
        if not body_json:
            resolved['body_input'] = prior.get('body_input')
    conn = init_db(str(db_path))
    schema_sha = _schema_fingerprint(conn)
    if saved_fingerprint and schema_sha != saved_fingerprint.get("schema_sha256"):
        conn.close()
        raise ValueError("resume refused: SQLite schema fingerprint differs")
    fingerprint = {
      'core': core_fingerprint,
      'core_sha256': _canonical_sha(core_fingerprint),
      'schema_sha256': schema_sha,
      'mapping_sha256': None,
      'legacy_reconciliation': legacy_proof or (saved_fingerprint or {}).get('legacy_reconciliation'),
    }
    fingerprint['fingerprint_sha256'] = _fingerprint_sha(fingerprint)
    if not saved_fingerprint:
        conn.execute("INSERT OR REPLACE INTO meta VALUES('preprocessing_fingerprint',?)",
                     (fingerprint['fingerprint_sha256'],)); conn.commit()
    resolved['preprocessing_fingerprint'] = fingerprint
    if legacy_proof:
        resolved['legacy_artifact_reconciliation'] = legacy_proof
    json.dump(resolved, open(cfg_path, "w", encoding="utf-8"), indent=2); open(cfg_path,"a").write("\n")

    vocab_complete = (conn.execute("SELECT COUNT(DISTINCT field) FROM vocab").fetchone()[0] ==
                      len(VOCAB_FIELDS) + 1 and
                      all((out_dir / 'vocab' / (field + '.tsv')).is_file()
                          for field in VOCAB_FIELDS + ('word',)))
    has_vocab = vocab_complete
    maps = None
    if not has_vocab:
        conn.execute("DELETE FROM vocab"); conn.commit()
        started = time.time()
        counters = build_counters(data_root, limit_per_split)
        vocabs = make_vocabularies(counters, tokenizer)
        maps = vocab_maps(vocabs)
        write_vocabularies(conn, vocabs, out_dir)
        del counters
        state = {'phase':'vocab_complete','vocab_fields':list(vocabs),'limit_per_split':limit_per_split,
                 'preprocessing_fingerprint':fingerprint['fingerprint_sha256'],
                 'elapsed_seconds':time.time()-started}
        json.dump(state, open(state_path,'w'), indent=2); open(state_path,'a').write('\n')
    else:
        vocabs = {}
        for field in VOCAB_FIELDS + ('word',):
            path = out_dir / 'vocab' / (field + '.tsv')
            entries=[]
            with open(path, encoding='utf-8') as f:
                for line in f:
                    idx, token, *rest = line.rstrip('\n').split('\t')
                    entries.append((token, int(rest[0]) if rest and rest[0] else 0, int(idx)))
            vocabs[field]=entries
        maps = {f:{token:idx for token,_,idx in entries} for f,entries in vocabs.items()}
    mapping_sha = _mapping_fingerprint(out_dir, conn)
    if saved_fingerprint and saved_fingerprint.get('mapping_sha256') is not None and mapping_sha != saved_fingerprint.get('mapping_sha256'):
        conn.close()
        raise ValueError("resume refused: vocabulary mapping fingerprint differs")
    fingerprint['mapping_sha256'] = mapping_sha
    fingerprint['fingerprint_sha256'] = _fingerprint_sha(fingerprint)
    if saved_fingerprint and saved_fingerprint.get('mapping_sha256') is not None and saved_fingerprint.get('fingerprint_sha256') != fingerprint['fingerprint_sha256']:
        conn.close()
        raise ValueError("resume refused: preprocessing fingerprint differs")
    conn.execute("INSERT OR REPLACE INTO meta VALUES('preprocessing_fingerprint',?)",
                 (fingerprint['fingerprint_sha256'],)); conn.commit()
    resolved['preprocessing_fingerprint'] = fingerprint
    json.dump(resolved, open(cfg_path, "w", encoding="utf-8"), indent=2); open(cfg_path,"a").write("\n")
    body_map = load_external_body(body_json, nid_map)
    insert_news(conn, data_root, maps, tokenizer, body_map, limit_per_split)
    news_count = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    json.dump({'phase':'news_lookup_complete','news_rows':news_count,
               'preprocessing_fingerprint':fingerprint['fingerprint_sha256']},open(state_path,'w'),indent=2); open(state_path,'a').write('\n')
    record_counts = insert_records(conn, data_root, maps, limit_per_split, resume=resume)
    for split in SPLITS:
        record_counts[split]['impressions_total'] = conn.execute('SELECT COUNT(*) FROM impressions WHERE split=?',(split,)).fetchone()[0]
        record_counts[split]['candidates_total'] = conn.execute('SELECT COUNT(*) FROM candidates c JOIN impressions i ON i.id=c.impression_id WHERE i.split=?',(split,)).fetchone()[0]
    json.dump({'phase':'complete','record_counts':record_counts,'news_rows':news_count,
               'preprocessing_fingerprint':fingerprint['fingerprint_sha256']},open(state_path,'w'),indent=2); open(state_path,'a').write('\n')
    body_rows = conn.execute("SELECT COUNT(*) FROM news WHERE body_ids IS NOT NULL").fetchone()[0]
    conn.execute("INSERT OR REPLACE INTO meta VALUES('body_available',?)",('1' if body_rows else '0',))
    conn.execute("INSERT OR REPLACE INTO meta VALUES('split_scope',?)",('sample' if limit_per_split else 'full',)); conn.commit(); conn.close()
    # Hash only generated artifacts; no source-data re-inventory is performed.
    files=[]
    for p in sorted(out_dir.rglob('*')):
        if p.is_file() and p.name not in ('artifact_manifest.json','-wal','-shm'):
            files.append({'path':str(p.relative_to(out_dir)),'bytes':p.stat().st_size,'sha256':sha256(p)})
    manifest={'scope':'sample' if limit_per_split else 'full','source_commit':CONFIG.source_commit,
              'schema_version':SCHEMA_VERSION,'schema_sha256':schema_sha,
              'mapping_sha256':mapping_sha,
              'tokenizer_repo':CONFIG.tokenizer_repo,'tokenizer_revision':CONFIG.tokenizer_revision,
              'body_available':bool(body_rows),'body_rows':body_rows,'glove_available':False,'artifacts':files,
              'state':'READY_FOR_LOOKUP; BODY/GLOVE ARTIFACTS MISSING' if not body_rows else 'BODY_LOADED_GLOVE_ARTIFACT_MISSING'}
    json.dump(manifest,open(out_dir/'artifact_manifest.json','w'),indent=2); open(out_dir/'artifact_manifest.json','a').write('\n')
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--out',required=True)
    ap.add_argument('--limit-per-split',type=int,default=None,help='bounded parser smoke only; omit for full processing')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--body-json')
    ap.add_argument('--nid-map')
    args=ap.parse_args()
    result=prepare(args.data_root,args.out,args.limit_per_split,args.resume,args.body_json,args.nid_map)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
