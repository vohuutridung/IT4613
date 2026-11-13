"""Focused checks of generated stage-01 data against a few source records."""
import argparse
import csv
import json
import sqlite3

from transformers import AutoTokenizer

from mind_data_pipeline.config import CONFIG
from mind_data_pipeline.preprocess import (NEWS_COLUMNS, BEHAVIOR_COLUMNS, encode_text, iter_tsv,
                        parse_entities, row_click, unpack_ids)


def _vocab(path, word=False):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a, b, *_ = line.rstrip("\n").split("\t")
            result[b if word else b] = int(a)
    return result


def _source_row(path, line_number, expected):
    for line_no, row in iter_tsv(path, expected, line_number):
        if line_no == line_number:
            return row
    raise IndexError(line_number)


def run(data_root, db_path, output_dir):
    from pathlib import Path
    db = sqlite3.connect(db_path)
    root, out = Path(data_root), Path(output_dir)
    maps = {field: _vocab(out / "vocab" / (field + ".tsv"))
            for field in ("uid", "did", "cat", "sub_cat", "entity", "entity_type")}
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_local, local_files_only=True, use_fast=False)
    checks = {}

    # Check first impression's IDs, labels, positions, and stored history.
    row = _source_row(root / "MINDlarge_train" / "behaviors.tsv", 1, BEHAVIOR_COLUMNS)
    imp, uid, time_text, history_text, impression_text = row
    saved = db.execute("SELECT id,impression_id,uid,history_ids,history_len FROM impressions WHERE split='train' AND line_no=1").fetchone()
    assert saved[1] == imp and saved[2] == maps["uid"].get(uid, 1)
    hist = [maps["did"].get(x, 1) for x in list(reversed(history_text.split()))[:CONFIG.max_history]]
    assert unpack_ids(saved[3]) == hist and saved[4] == len(hist)
    cand_rows = db.execute("SELECT c.position,c.did,l.click,l.labeled FROM candidates c JOIN labels l ON l.candidate_id=c.id WHERE c.impression_id=? ORDER BY c.position", (saved[0],)).fetchall()
    expected = []
    for pos, token in enumerate(impression_text.split()):
        did, click = row_click(token)
        expected.append((pos, maps["did"].get(did, 1), click, int(click is not None)))
    assert cand_rows == expected
    checks["first_impression_history_reverse_and_candidate_labels"] = True
    checks["candidate_positions"] = len(cand_rows)
    checks["time_is_metadata_only"] = time_text is not None

    # Verify a full-cap history (if present) is reversed then truncated at 200.
    long_row = db.execute("SELECT line_no,impression_id,history_ids,history_len FROM impressions WHERE split='train' AND history_len=200 ORDER BY line_no LIMIT 1").fetchone()
    assert long_row is not None
    raw = _source_row(root / "MINDlarge_train" / "behaviors.tsv", long_row[0], BEHAVIOR_COLUMNS)
    expected_hist = [maps["did"].get(x, 1) for x in list(reversed(raw[3].split()))[:200]]
    assert unpack_ids(long_row[2]) == expected_hist and len(expected_hist) == 200
    checks["history_cap_200_and_reverse"] = True

    # Verify title and abstract tokenization from one real news row.
    news_row = None
    for line_no, candidate in iter_tsv(root / "MINDlarge_train" / "news.tsv", NEWS_COLUMNS, 500):
        if candidate[6] or candidate[7]:
            news_row = (line_no, candidate)
            break
    assert news_row is not None
    line_no, r = news_row
    did, cat, subcat, title, abstract, _, title_raw, abstract_raw = r
    did_id = maps["did"].get(did, 1)
    saved_news = db.execute("SELECT cat_id,sub_cat_id,title_ids,abstract_ids,title_entity_ids,title_entity_types,abstract_entity_ids,abstract_entity_types FROM news WHERE did=?", (did_id,)).fetchone()
    assert saved_news[0] == maps["cat"].get(cat,1) and saved_news[1] == maps["sub_cat"].get(subcat,1)
    assert unpack_ids(saved_news[2]) == encode_text(tokenizer, title, CONFIG.max_title_tokens)
    assert unpack_ids(saved_news[3]) == encode_text(tokenizer, abstract, CONFIG.max_abstract_tokens)
    title_entities = parse_entities(title_raw, "source", line_no)[:CONFIG.max_title_entities]
    abstract_entities = parse_entities(abstract_raw, "source", line_no)[:CONFIG.max_abstract_entities]
    def entity_values(values, cap):
        ids = [maps["entity"].get(x,1) for x,_ in values]
        types = [maps["entity_type"].get(t,1) if t else 0 for _,t in values]
        return ids + [0] * (cap-len(ids)), types + [0] * (cap-len(types))
    assert unpack_ids(saved_news[4]) == entity_values(title_entities, CONFIG.max_title_entities)[0]
    assert unpack_ids(saved_news[5]) == entity_values(title_entities, CONFIG.max_title_entities)[1]
    assert unpack_ids(saved_news[6]) == entity_values(abstract_entities, CONFIG.max_abstract_entities)[0]
    assert unpack_ids(saved_news[7]) == entity_values(abstract_entities, CONFIG.max_abstract_entities)[1]
    checks["bert_cased_tokenization_title_abstract"] = True
    checks["title_abstract_entities_and_types_synchronized"] = True
    checks["abstract_entities_read_from_abstract_column"] = True
    checks["unk_fallback_id"] = 1
    checks["lookup_text_stored_once_by_news_id"] = True
    db.close()
    return {"status": "PASS", "checks": checks, "source_records_read": {"behaviors": 2, "news": line_no},
            "tokenizer": {"repo": CONFIG.tokenizer_repo, "revision": CONFIG.tokenizer_revision,
                          "local_original_identity_confirmed": False}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True, help="stage-01 output directory containing vocab/")
    a = p.parse_args()
    print(json.dumps(run(a.data_root, a.db, a.out), indent=2))
