"""Data contract regression tests. Run: python3 -m unittest discover -s scripts -p 'test_*.py'."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_demo_data import (
    DataIntegrityError, INTERACTION_HEADERS, NEWS_HEADERS, ROOT,
    build_data, excerpt, parse_timestamp, write_data,
)


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        self.news = [
            {"News ID": "N1", "Category": "news", "Topic": "newspolitics", "Headline": "A headline, faithfully retained.", "News body": "An original article body. U.S. policy, quotations, and Unicode — stay unchanged."},
            {"News ID": "N2", "Category": "foodanddrink", "Topic": "beverages", "Headline": "Second headline", "News body": "A short description."},
            {"News ID": "N3", "Category": "sports", "Topic": "basketball_nba", "Headline": "Third headline", "News body": "The complete text."},
        ]
        self.rows = [{"UserID": "U1", "ClicknewsID": "N1 N2", "dwelltime": "45 125", "exposure_time": "6/9/2019 11:59:59 PM#TAB#6/10/2019 12:00:00 AM", "pos": "N3", "neg": "N2", "start": "6/24/2019 12:00:00 PM", "end": "6/24/2019 1:00:00 PM", "dwelltime_pos": "318"}]
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for name, headers, rows in [("display.tsv", INTERACTION_HEADERS, self.rows), ("display_news.tsv", NEWS_HEADERS, self.news)]:
            with (self.source / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)

    def load(self):
        return build_data(self.source, expected_readers=1, expected_articles=3)

    def test_normalizes_aligned_lists_preserves_original_content_and_labels(self):
        data = self.load()
        profile = data["profiles"]["U1"]
        self.assertEqual(profile["history"][1], {"newsId": "N2", "exposureTime": "6/10/2019 12:00:00 AM", "exposureSortKey": "2019-06-10T00:00:00", "dwellTimeSeconds": 125})
        self.assertEqual(profile["impression"]["candidates"], [{"newsId": "N3", "label": 1, "originalIndex": 0, "dwellTimeSeconds": 318}, {"newsId": "N2", "label": 0, "originalIndex": 1, "dwellTimeSeconds": None}])
        self.assertEqual(data["articles"]["N1"]["body"], self.news[0]["News body"])
        self.assertNotIn("body", data["previews"]["N1"])
        self.assertEqual(data["previews"]["N1"]["estimatedReadingMinutes"], 1)
        self.assertEqual(data["readers"][0]["historyCount"], 2)
        self.assertEqual(data["readers"][0]["candidateCount"], 2)

    def test_timestamp_is_calendar_aware_without_timezone(self):
        midnight = parse_timestamp("6/10/2019 12:00:00 AM")
        noon = parse_timestamp("6/10/2019 12:00:00 PM")
        self.assertIsNone(midnight.tzinfo)
        self.assertEqual(midnight.hour, 0)
        self.assertEqual(noon.hour, 12)
        self.assertGreater(midnight, parse_timestamp("6/9/2019 11:59:59 PM"))
        for invalid in ["2019-06-10T12:00:00Z", "2/29/2019 1:00:00 AM", "6/10/2019 0:00:00 AM", "6/10/2019 13:00:00 PM"]:
            with self.subTest(invalid=invalid), self.assertRaises(DataIntegrityError):
                parse_timestamp(invalid)

    def test_bad_interactions_fail_visibly(self):
        variants = [
            ("dwelltime", "45", "lengths do not align"),
            ("exposure_time", "6/9/2019 11:59:59 PM", "lengths do not align"),
            ("dwelltime_pos", "", "lengths do not align"),
            ("dwelltime", "nan 125", "finite and nonnegative"),
            ("dwelltime_pos", "-5", "finite and nonnegative"),
            ("dwelltime", "invalid 125", "invalid duration"),
            ("neg", "N3", "overlap"),
            ("neg", "N2 N2", "duplicate candidate"),
            ("neg", "N999", "unresolved News ID"),
            ("start", "today", "invalid timestamp"),
            ("end", "6/23/2019 1:00:00 PM", "ends before"),
            ("UserID", "../U1", "invalid identifier"),
        ]
        original = copy.deepcopy(self.rows)
        for field, value, message in variants:
            with self.subTest(field=field, value=value):
                self.rows = copy.deepcopy(original)
                self.rows[0][field] = value
                self.save()
                with self.assertRaisesRegex(DataIntegrityError, message):
                    self.load()

    def test_duplicate_readers_and_articles_fail(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.save()
        with self.assertRaisesRegex(DataIntegrityError, "duplicate User ID"):
            self.load()
        self.rows.pop()
        self.news.append(copy.deepcopy(self.news[0]))
        self.save()
        with self.assertRaisesRegex(DataIntegrityError, "duplicate News ID"):
            self.load()

    def test_missing_content_and_wrong_headers_fail(self):
        self.news[0]["News body"] = "  "
        self.save()
        with self.assertRaisesRegex(DataIntegrityError, "nonempty"):
            self.load()
        (self.source / "display_news.tsv").write_text("Wrong\tHeaders\n", encoding="utf-8")
        with self.assertRaisesRegex(DataIntegrityError, "expected headers"):
            self.load()

    def test_output_is_deterministic_cleans_stale_generated_files_and_keeps_source(self):
        data = self.load()
        source_hash = hashlib.sha256((self.source / "display_news.tsv").read_bytes()).hexdigest()
        output = self.root / "generated"
        write_data(data, output, self.source)
        first = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*.json")}
        (output / "articles" / "N999.json").write_text("{}")
        write_data(self.load(), output, self.source)
        second = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*.json")}
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256((self.source / "display_news.tsv").read_bytes()).hexdigest(), source_hash)
        self.assertEqual(json.loads((output / "data-report.json").read_text())["sourceSha256"]["display_news.tsv"], source_hash)
        with self.assertRaisesRegex(DataIntegrityError, "separate from source"):
            write_data(data, self.source, self.source)

    def test_excerpt_only_normalizes_whitespace_and_truncates_at_word_boundary(self):
        self.assertEqual(excerpt("Actual\n\twords.  Unchanged!"), "Actual words. Unchanged!")
        self.assertEqual(excerpt("A real sentence contains several words.", 15), "A real sentence…")

    def test_curated_dataset_contract_all_profiles_and_references(self):
        data = build_data(ROOT / "data")
        self.assertEqual(len(data["profiles"]), 57)
        self.assertEqual(len(data["articles"]), 1669)
        default = data["profiles"]["U216199"]
        self.assertEqual(len(default["history"]), 34)
        self.assertEqual(len(default["impression"]["candidates"]), 23)
        self.assertEqual(sum(candidate["label"] for candidate in default["impression"]["candidates"]), 7)
        self.assertEqual(default["impression"]["candidates"][0]["newsId"], "N75626")
        for profile in data["profiles"].values():
            for entry in profile["history"] + profile["impression"]["candidates"]:
                self.assertIn(entry["newsId"], data["articles"])
        # Repeated builds preserve byte-for-byte input-derived structures.
        self.assertEqual(data, build_data(ROOT / "data"))


if __name__ == "__main__":
    unittest.main()
