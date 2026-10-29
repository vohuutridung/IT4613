const abbreviations =
  /\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Rev|Hon|Sen|Rep|Gov|Gen|Lt|Col|Capt|Sgt|No|Nos|Fig|Figs|Vol|Vs|vs|etc|Inc|Ltd|Co|Corp|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$/i;

/** A conservative fallback: never cut after titles, initials, or dotted abbreviations. */
export function fallbackSentences(text: string): string[] {
  const sentences: string[] = [];
  const boundary = /[.!?]+["'”’\)\]]*(?=\s+|$)/g;
  let start = 0;
  for (const match of text.matchAll(boundary)) {
    const end = (match.index ?? 0) + match[0].length;
    const prefix = text.slice(0, end).replace(/["'”’\)\]]+$/, "");
    if (
      match[0].startsWith(".") &&
      (abbreviations.test(prefix) ||
        /(?:^|[\s("“])(?:[A-Za-z]\.){2,}$/.test(prefix) ||
        /(?:^|[\s("“])[A-Z]\.$/.test(prefix))
    )
      continue;
    const sentence = text.slice(start, end).trim();
    if (sentence) sentences.push(sentence);
    start = end;
  }
  const tail = text.slice(start).trim();
  if (tail) sentences.push(tail);
  return sentences;
}

function sentencesFor(text: string, useNative: boolean): string[] {
  if (useNative && typeof Intl.Segmenter === "function") {
    const segments = new Intl.Segmenter("en", {
      granularity: "sentence",
    }).segment(text);
    // Segmenters can differ between engines; keep accidental title/initial splits joined.
    const sentences: string[] = [];
    let pending = "";
    for (const segment of segments) {
      pending += segment.segment;
      const end = pending.trimEnd().replace(/["'”’\)\]]+$/, "");
      const hasFollowingText =
        segment.index + segment.segment.length < text.length;
      if (
        abbreviations.test(end) ||
        /(?:^|[\s("“])(?:[A-Z]\.)+$/.test(end) ||
        (hasFollowingText && !/\s$/.test(segment.segment))
      )
        continue;
      if (pending.trim()) sentences.push(pending.trim());
      pending = "";
    }
    if (pending.trim()) sentences.push(pending.trim());
    return sentences;
  }
  return fallbackSentences(text);
}

/**
 * Adds paragraph boundaries to plain source text without rewriting its words.
 * Existing line boundaries are preserved; short text remains a single paragraph.
 */
export function formatParagraphs(
  body: string,
  options: { useNativeSegmenter?: boolean } = {},
): string[] {
  const paragraphs: string[] = [];
  for (const block of body
    .split(/\r?\n+/)
    .map((part) => part.trim())
    .filter(Boolean)) {
    const sentences = sentencesFor(block, options.useNativeSegmenter !== false);
    let current: string[] = [];
    let length = 0;
    for (const sentence of sentences) {
      current.push(sentence);
      length += sentence.length;
      if (
        current.length >= 5 ||
        (current.length >= 3 && length >= 450) ||
        (current.length >= 2 && length >= 800)
      ) {
        paragraphs.push(current.join(" "));
        current = [];
        length = 0;
      }
    }
    if (current.length) paragraphs.push(current.join(" "));
  }
  return paragraphs;
}
