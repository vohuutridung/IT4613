#!/usr/bin/env python3
"""Read-only inventory and schema audit for the supplied MIND-large files.

This intentionally does not tokenize, remap IDs, materialize records, or build
model inputs. Those steps require the locked upstream implementation and its
private dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

SPLITS = ("MINDlarge_train", "MINDlarge_dev", "MINDlarge_test")
EXPECTED = {
    "behaviors.tsv": 5,
    "news.tsv": 8,
    "entity_embedding.vec": 102,  # ID + 100 values + trailing empty TSV cell
    "relation_embedding.vec": 102,
}


def digest_and_rows(path: Path, expected_columns: int) -> dict:
    sha = hashlib.sha256()
    rows = 0
    bad_width = 0
    invalid_utf8 = 0
    invalid_json = 0
    bad_labels = 0
    candidates = clicks = unlabeled = histories = empty_histories = 0
    news_ids: set[str] = set()
    duplicate_news_ids = 0
    embedding_dims: dict[str, int] = {}
    trailing_empty = 0
    started = time.monotonic()
    with path.open("rb") as stream:
        for raw in stream:
            sha.update(raw)
            rows += 1
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                invalid_utf8 += 1
                continue
            cells = line.split("\t")
            if len(cells) != expected_columns:
                bad_width += 1
                continue
            if path.name == "news.tsv":
                if cells[0] in news_ids:
                    duplicate_news_ids += 1
                news_ids.add(cells[0])
                for raw_json in cells[6:8]:
                    if raw_json:
                        try:
                            value = json.loads(raw_json)
                            if not isinstance(value, list):
                                invalid_json += 1
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            invalid_json += 1
            elif path.name == "behaviors.tsv":
                history = cells[3].split() if cells[3] else []
                histories += len(history)
                empty_histories += not bool(history)
                if cells[4]:
                    for item in cells[4].split():
                        if "-" in item:
                            _, label = item.rsplit("-", 1)
                            candidates += 1
                            if label in ("0", "1"):
                                clicks += label == "1"
                            else:
                                bad_labels += 1
                        else:
                            candidates += 1
                            unlabeled += 1
            else:
                values = cells[:-1] if cells[-1] == "" else cells
                trailing_empty += cells[-1] == ""
                dim = len(values) - 1
                embedding_dims[str(dim)] = embedding_dims.get(str(dim), 0) + 1
                if dim != 100:
                    bad_width += 1
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha.hexdigest(),
        "rows": rows,
        "expected_columns": expected_columns,
        "bad_width_rows": bad_width,
        "invalid_utf8_rows": invalid_utf8,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if path.name == "news.tsv":
        result.update(unique_news_ids=len(news_ids), duplicate_news_ids=duplicate_news_ids,
                      invalid_entity_json_fields=invalid_json)
    elif path.name == "behaviors.tsv":
        result.update(candidate_tokens=candidates, positive_labels=clicks,
                      unlabeled_candidate_tokens=unlabeled, bad_label_tokens=bad_labels,
                      history_tokens=histories, empty_histories=empty_histories)
    else:
        result.update(embedding_dimensions=embedding_dims,
                      trailing_empty_rows=trailing_empty)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()
    began = time.monotonic()
    inventory = {"dataset_root": str(args.data.resolve()), "splits": {},
                 "archives": [], "audit": "read_only"}
    for split in SPLITS:
        root = args.data / split
        if not root.is_dir():
            inventory["splits"][split] = {"status": "MISSING"}
            continue
        files = {}
        for name, width in EXPECTED.items():
            path = root / name
            files[name] = digest_and_rows(path, width) if path.is_file() else {"status": "MISSING"}
        inventory["splits"][split] = {"directory_bytes": sum(
            f.stat().st_size for f in root.iterdir() if f.is_file()), "files": files}
    inventory["archives"] = [
        {"path": str(p), "bytes": p.stat().st_size}
        for p in sorted(args.data.parent.glob("*.zip"))
    ]
    inventory["runtime"] = {
        "elapsed_seconds": round(time.monotonic() - began, 3),
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_rss_unit_note": "KiB on Linux; raw getrusage value recorded",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "data_manifest.json"
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(inventory["runtime"], indent=2))


if __name__ == "__main__":
    main()
