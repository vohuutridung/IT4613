import { LabelRanker } from "./labelRanker";
import type { Ranker } from "./types";

/** Inject a backend/model adapter here later without changing feed components. */
export function createRankingProvider(provider?: Ranker): Ranker {
  return provider ?? new LabelRanker();
}

export const rankingProvider = createRankingProvider();
export { LabelRanker };
export type { Ranker, RankingContext, RankedCandidate } from "./types";
