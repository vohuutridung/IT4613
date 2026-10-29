import type { Ranker, RankingContext, RankedCandidate } from "./types";

/** First-version ordering only. No prediction scores are inferred from labels. */
export class LabelRanker implements Ranker {
  async rank({ impression }: RankingContext): Promise<RankedCandidate[]> {
    return [...impression.candidates]
      .sort((a, b) => b.label - a.label || a.originalIndex - b.originalIndex)
      .map((candidate, index) => ({
        newsId: candidate.newsId,
        rank: index + 1,
      }));
  }
}
