import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import {
  categoryLabel,
  topicLabel,
  formatArchiveDate,
  formatArchiveTime,
  formatDuration,
  sortHistory,
} from "../src/lib/format";
import { fallbackSentences, formatParagraphs } from "../src/lib/paragraphs";
import { createRankingProvider, LabelRanker } from "../src/services/ranking";
import type { ReaderProfile } from "../src/types";

const normalize = (value: string) => value.replace(/\s+/g, " ").trim();

test("ranking is stable within labels, complete, deterministic, and immutable", async () => {
  const profile: ReaderProfile = {
    userId: "example",
    history: [],
    impression: {
      id: "example-1",
      userId: "example",
      start: "",
      end: "",
      startDate: "",
      candidates: [
        { newsId: "N2", label: 0, originalIndex: 0, dwellTimeSeconds: null },
        { newsId: "N3", label: 1, originalIndex: 1, dwellTimeSeconds: 45 },
        { newsId: "N4", label: 1, originalIndex: 2, dwellTimeSeconds: 65 },
        { newsId: "N1", label: 0, originalIndex: 3, dwellTimeSeconds: null },
      ],
    },
  };
  const original = JSON.stringify(profile);
  const ranker = new LabelRanker();
  const actual = await ranker.rank(profile);
  assert.deepEqual(
    actual.map((item) => item.newsId),
    ["N3", "N4", "N2", "N1"],
  );
  assert.deepEqual(
    actual.map((item) => item.rank),
    [1, 2, 3, 4],
  );
  assert.deepEqual(await ranker.rank(profile), actual);
  assert.equal(JSON.stringify(profile), original);
  assert.deepEqual(
    await ranker.rank({
      ...profile,
      impression: { ...profile.impression, candidates: [] },
    }),
    [],
  );
});

test("a replacement ranker can be injected without changing presentation", async () => {
  const replacement = {
    rank: async () => [{ newsId: "replacement", rank: 1 }],
  };
  assert.equal(createRankingProvider(replacement), replacement);
});

test("every generated reader retains the exact complete impression order", async () => {
  const directory = join(process.cwd(), "public/data/profiles");
  assert.ok(existsSync(directory), "Run npm run prepare:data before tests");
  const files = readdirSync(directory).filter((name) => name.endsWith(".json"));
  assert.equal(files.length, 57);
  const ranker = new LabelRanker();
  for (const file of files) {
    const profile = JSON.parse(
      readFileSync(join(directory, file), "utf8"),
    ) as ReaderProfile;
    const result = await ranker.rank(profile);
    const expected = [1, 0].flatMap((label) =>
      profile.impression.candidates
        .filter((candidate) => candidate.label === label)
        .map((candidate) => candidate.newsId),
    );
    assert.deepEqual(
      result.map((candidate) => candidate.newsId),
      expected,
      profile.userId,
    );
    assert.equal(
      new Set(result.map((candidate) => candidate.newsId)).size,
      expected.length,
    );
  }
});

test("durations distinguish seconds and recorded minutes without rounding away seconds", () => {
  assert.equal(formatDuration(45), "45 sec");
  assert.equal(formatDuration(125), "2 min 5 sec");
  assert.equal(formatDuration(318), "5 min 18 sec");
  assert.equal(formatDuration(120), "2 min");
  assert.equal(formatDuration(0), "0 sec");
  assert.equal(formatDuration(null), "Not recorded");
});

test("naive archive calendar values are rendered without timezone conversion", () => {
  assert.equal(formatArchiveDate("6/20/2019 12:00:11 AM"), "June 20, 2019");
  assert.equal(
    formatArchiveDate("2019-06-20T23:59:59", { short: true }),
    "Jun 20, 2019",
  );
  assert.equal(formatArchiveTime("6/20/2019 12:00:11 AM"), "12:00 AM");
  assert.equal(formatArchiveTime("6/20/2019 12:00:11 PM"), "12:00 PM");
  assert.equal(formatArchiveTime("2019-06-20T23:59:59"), "11:59 PM");
});

test("history sorting uses parsed naive keys and retains aligned fields", () => {
  const entries = [
    {
      newsId: "A",
      exposureTime: "6/2/2019 8:00:00 AM",
      exposureSortKey: "2019-06-02T08:00:00",
      dwellTimeSeconds: 12,
    },
    {
      newsId: "B",
      exposureTime: "6/15/2019 8:00:00 AM",
      exposureSortKey: "2019-06-15T08:00:00",
      dwellTimeSeconds: 24,
    },
  ];
  assert.deepEqual(
    sortHistory(entries).map((entry) => [entry.newsId, entry.dwellTimeSeconds]),
    [
      ["B", 24],
      ["A", 12],
    ],
  );
  assert.equal(entries[0].newsId, "A");
});

test("human-readable metadata covers concatenated and unfamiliar identifiers", () => {
  assert.equal(categoryLabel("foodanddrink"), "Food & Drink");
  assert.equal(topicLabel("basketball_nba"), "NBA Basketball");
  assert.equal(topicLabel("newspolitics"), "Politics");
  assert.equal(topicLabel("lifestyleroyals"), "Royals");
  assert.equal(topicLabel("newsfuture_topic"), "Future Topic");
  assert.equal(topicLabel("future_topic"), "Future Topic");
});

test("paragraph formatting preserves all wording and handles brief or blank bodies", () => {
  assert.deepEqual(formatParagraphs(""), []);
  assert.deepEqual(formatParagraphs("A short video description."), [
    "A short video description.",
  ]);
  const text =
    "Dr. J. R. Smith traveled to the U.S. in June. He met Mr. Jones. “What a trip!” she said. This is the fourth sentence. Here comes another sentence. Then the story ends.";
  for (const useNativeSegmenter of [true, false]) {
    const paragraphs = formatParagraphs(text, { useNativeSegmenter });
    assert.equal(normalize(paragraphs.join(" ")), normalize(text));
    assert.ok(paragraphs.length >= 2);
    assert.ok(
      paragraphs[0].includes("Dr. J. R. Smith traveled to the U.S. in June."),
    );
  }
  assert.deepEqual(
    formatParagraphs("Original first paragraph.\n\nOriginal second paragraph."),
    ["Original first paragraph.", "Original second paragraph."],
  );
});

test("fallback boundaries avoid titles, initials, dotted acronyms, and decimals", () => {
  const sentences = fallbackSentences(
    "Dr. J. R. Smith visited the U.S. in May. Prices rose 2.5 percent. A final sentence.",
  );
  assert.deepEqual(sentences, [
    "Dr. J. R. Smith visited the U.S. in May.",
    "Prices rose 2.5 percent.",
    "A final sentence.",
  ]);
});

test("native sentence boundaries do not insert spaces into contiguous source punctuation", () => {
  const text =
    "We both realized it….Sometimes you just realize it. More words follow. Another sentence follows. One more sentence. And the end.";
  assert.equal(normalize(formatParagraphs(text).join(" ")), normalize(text));
});

test("paragraph formatting preserves normalized source text for every article in both modes", () => {
  const directory = join(process.cwd(), "public/data/articles");
  assert.ok(existsSync(directory), "Run npm run prepare:data before tests");
  const files = readdirSync(directory).filter((name) => name.endsWith(".json"));
  assert.equal(files.length, 1669);
  for (const file of files) {
    const article = JSON.parse(readFileSync(join(directory, file), "utf8")) as {
      body: string;
    };
    for (const useNativeSegmenter of [true, false]) {
      const formatted = formatParagraphs(article.body, { useNativeSegmenter });
      assert.equal(
        normalize(formatted.join(" ")),
        normalize(article.body),
        `${file}: ${useNativeSegmenter ? "native" : "fallback"} segmentation changed source text`,
      );
    }
  }
});
