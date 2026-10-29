import type { HistoryEntry, Impression } from "../../types";

export interface RankingContext {
  userId: string;
  impression: Impression;
  history: HistoryEntry[];
}

export interface RankedCandidate {
  newsId: string;
  rank: number;
  score?: number;
}

export interface Ranker {
  rank(context: RankingContext): Promise<RankedCandidate[]>;
}
