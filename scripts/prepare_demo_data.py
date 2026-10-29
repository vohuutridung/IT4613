#!/usr/bin/env python3
"""Validate the immutable PENS subset and prepare deterministic static data.

Timestamps are naive calendar values: no timezone is assumed or introduced.
Only lightweight previews and profiles load initially; full bodies are separate files.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INTERACTION_HEADERS = ["UserID", "ClicknewsID", "dwelltime", "exposure_time", "pos", "neg", "start", "end", "dwelltime_pos"]
NEWS_HEADERS = ["News ID", "Category", "Topic", "Headline", "News body"]
TIMESTAMP = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2}):(\d{2}) (AM|PM)")


class DataIntegrityError(ValueError):
    """An error in the supplied dataset that must not be hidden by the UI."""


def parse_timestamp(value: str, context: str = "timestamp") -> datetime:
    """Explicitly parse PENS calendar timestamps, without timezone conversion."""
    match = TIMESTAMP.fullmatch(value)
    if not match:
        raise DataIntegrityError(f"{context}: invalid timestamp {value!r}; expected M/D/YYYY h:mm:ss AM/PM")
    month, day, year, hour, minute, second = map(int, match.groups()[:6])
    if not 1 <= hour <= 12:
        raise DataIntegrityError(f"{context}: invalid 12-hour clock hour in {value!r}")
    hour = hour % 12 + (12 if match.group(7) == "PM" else 0)
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise DataIntegrityError(f"{context}: invalid calendar timestamp {value!r}") from exc


def read_tsv(path: Path, headers: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames != headers:
                raise DataIntegrityError(f"{path.name}: expected headers {headers!r}, got {reader.fieldnames!r}")
            rows = []
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise DataIntegrityError(f"{path.name}, record {line}: incorrect TSV field count")
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataIntegrityError(f"Cannot read {path}: {exc}") from exc


def durations(value: str, context: str) -> list[int | float]:
    result = []
    for token in value.split():
        try:
            number = float(token)
        except ValueError as exc:
            raise DataIntegrityError(f"{context}: invalid duration {token!r}") from exc
        if not math.isfinite(number) or number < 0:
            raise DataIntegrityError(f"{context}: durations must be finite and nonnegative, got {token!r}")
        result.append(int(number) if number.is_integer() else number)
    return result


def excerpt(body: str, limit: int = 220) -> str:
    """Whitespace-only normalization; never paraphrase supplied article text."""
    compact = " ".join(body.split())
    if len(compact) <= limit:
        return compact
    end = compact.rfind(" ", 0, limit + 1)
    return compact[:end if end > 0 else limit].rstrip() + "…"


def assert_id(value: str, prefix: str, context: str) -> None:
    if not re.fullmatch(prefix + r"\d+", value):
        raise DataIntegrityError(f"{context}: invalid identifier {value!r}")


def build_data(source_dir: Path, expected_readers: int | None = 57,
               expected_articles: int | None = 1669) -> dict:
    news_path = source_dir / "display_news.tsv"
    interactions_path = source_dir / "display.tsv"
    articles = {}
    previews = {}
    for line, row in enumerate(read_tsv(news_path, NEWS_HEADERS), 2):
        news_id = row["News ID"]
        context = f"display_news.tsv record {line} ({news_id})"
        assert_id(news_id, "N", context)
        if news_id in articles:
            raise DataIntegrityError(f"{context}: duplicate News ID")
        if not row["Headline"].strip() or not row["News body"].strip():
            raise DataIntegrityError(f"{context}: headline and body must be nonempty")
        if not row["Category"].strip() or not row["Topic"].strip():
            raise DataIntegrityError(f"{context}: category and topic must be nonempty")
        article = dict(id=news_id, category=row["Category"], topic=row["Topic"],
                       headline=row["Headline"], body=row["News body"])
        articles[news_id] = article
        word_count = len(row["News body"].split())
        previews[news_id] = {key: value for key, value in article.items() if key != "body"}
        previews[news_id].update(excerpt=excerpt(article["body"]), wordCount=word_count,
                                 estimatedReadingMinutes=max(1, math.ceil(word_count / 220)))
    if expected_articles is not None and len(articles) != expected_articles:
        raise DataIntegrityError(f"Expected {expected_articles} articles, found {len(articles)}")

    profiles = {}
    readers = []
    for line, row in enumerate(read_tsv(interactions_path, INTERACTION_HEADERS), 2):
        user_id = row["UserID"]
        context = f"display.tsv record {line} ({user_id})"
        assert_id(user_id, "U", context)
        if user_id in profiles:
            raise DataIntegrityError(f"{context}: duplicate User ID")
        history_ids = row["ClicknewsID"].split()
        history_durations = durations(row["dwelltime"], context + " dwelltime")
        exposures = row["exposure_time"].split("#TAB#") if row["exposure_time"] else []
        if not len(history_ids) == len(history_durations) == len(exposures):
            raise DataIntegrityError(f"{context}: history, dwelltime, and exposure_time lengths do not align")
        positives, negatives = row["pos"].split(), row["neg"].split()
        positive_durations = durations(row["dwelltime_pos"], context + " dwelltime_pos")
        if len(positives) != len(positive_durations):
            raise DataIntegrityError(f"{context}: positive candidate and dwelltime_pos lengths do not align")
        if set(positives) & set(negatives):
            raise DataIntegrityError(f"{context}: positive and negative candidates overlap")
        if len(set(positives + negatives)) != len(positives + negatives):
            raise DataIntegrityError(f"{context}: duplicate candidate ID within impression")
        for news_id in history_ids + positives + negatives:
            if news_id not in articles:
                raise DataIntegrityError(f"{context}: unresolved News ID {news_id!r}")
        start, end = parse_timestamp(row["start"], context + " start"), parse_timestamp(row["end"], context + " end")
        if end < start:
            raise DataIntegrityError(f"{context}: impression ends before it starts")
        history = [dict(newsId=news_id, exposureTime=timestamp,
                        exposureSortKey=parse_timestamp(timestamp, context + " exposure_time").isoformat(),
                        dwellTimeSeconds=dwell)
                   for news_id, timestamp, dwell in zip(history_ids, exposures, history_durations)]
        candidates = [dict(newsId=news_id, label=1, originalIndex=index, dwellTimeSeconds=dwell)
                      for index, (news_id, dwell) in enumerate(zip(positives, positive_durations))]
        candidates.extend(dict(newsId=news_id, label=0, originalIndex=len(positives) + index, dwellTimeSeconds=None)
                          for index, news_id in enumerate(negatives))
        profiles[user_id] = dict(userId=user_id, history=history,
                                impression=dict(id=f"{user_id}-impression", userId=user_id,
                                                start=row["start"], end=row["end"], startDate=start.date().isoformat(),
                                                candidates=candidates))
        counts = collections.Counter(articles[news_id]["category"] for news_id in history_ids)
        top_categories = [dict(category=category, count=count)
                          for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]
        readers.append(dict(userId=user_id, historyCount=len(history), candidateCount=len(candidates),
                            topCategories=top_categories,
                            previewHeadline=articles[history_ids[-1]]["headline"] if history_ids else None))
    if expected_readers is not None and len(profiles) != expected_readers:
        raise DataIntegrityError(f"Expected {expected_readers} readers, found {len(profiles)}")
    referenced = {entry["newsId"] for profile in profiles.values()
                  for entry in profile["history"] + profile["impression"]["candidates"]}
    report = dict(schemaVersion=1, readers=len(profiles), articles=len(articles),
                  referencedArticles=len(referenced),
                  historyEntries=sum(len(profile["history"]) for profile in profiles.values()),
                  candidates=sum(len(profile["impression"]["candidates"]) for profile in profiles.values()),
                  timezone="unspecified; timestamps retained as naive calendar values",
                  sourceSha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [interactions_path, news_path]})
    return dict(readers=readers, profiles=profiles, previews=previews, articles=articles, report=report)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def write_data(data: dict, output: Path, source_dir: Path) -> None:
    output = output.resolve()
    source_dir = source_dir.resolve()
    # Never allow output cleanup to target input data, the project root, or a parent.
    if output == source_dir or output in source_dir.parents or output == ROOT:
        raise DataIntegrityError("Output directory must be separate from source data and project root")
    output.mkdir(parents=True, exist_ok=True)
    for name in ["profiles", "articles"]:
        if (output / name).is_symlink():
            raise DataIntegrityError(f"Refusing to replace symlink {output / name}")
    with tempfile.TemporaryDirectory(prefix=".daily-data-", dir=output.parent) as stage_name:
        stage = Path(stage_name)
        write_json(stage / "readers-index.json", data["readers"])
        write_json(stage / "news-index.json", data["previews"])
        write_json(stage / "data-report.json", data["report"])
        for name, records in [("profiles", data["profiles"]), ("articles", data["articles"])]:
            (stage / name).mkdir(exist_ok=True)
            for identifier, record in records.items():
                write_json(stage / name / f"{identifier}.json", record)
        # Cleanup is restricted to known generated subdirectories, never raw TSVs.
        for name in ["profiles", "articles"]:
            target = output / name
            if target.exists():
                shutil.rmtree(target)
            (stage / name).replace(target)
        for name in ["readers-index.json", "news-index.json", "data-report.json"]:
            (stage / name).replace(output / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data", help="Directory containing source TSV files")
    parser.add_argument("--output", type=Path, default=ROOT / "public" / "data", help="Generated static data directory")
    parser.add_argument("--check", action="store_true", help="Validate without writing generated files")
    args = parser.parse_args()
    try:
        data = build_data(args.source)
        if not args.check:
            write_data(data, args.output, args.source)
    except DataIntegrityError as exc:
        print(f"Data integrity error: {exc}", file=sys.stderr)
        return 1
    report = data["report"]
    action = "Validated" if args.check else "Prepared"
    print(f"{action} {report['readers']} readers, {report['articles']} articles, "
          f"{report['historyEntries']} history entries, and {report['candidates']} candidates. "
          "All references, timestamps, and positional lists are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
