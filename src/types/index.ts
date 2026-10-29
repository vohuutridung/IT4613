export interface NewsPreview {
  id: string;
  category: string;
  topic: string;
  headline: string;
  excerpt: string;
  wordCount: number;
  estimatedReadingMinutes: number;
}

export interface NewsArticle {
  id: string;
  category: string;
  topic: string;
  headline: string;
  body: string;
}

export interface ReaderSummary {
  userId: string;
  historyCount: number;
  candidateCount: number;
  topCategories: { category: string; count: number }[];
  previewHeadline?: string;
}

export interface HistoryEntry {
  newsId: string;
  exposureTime: string;
  /** Local calendar value, deliberately without an invented timezone. */
  exposureSortKey: string;
  dwellTimeSeconds: number;
}

export interface Candidate {
  newsId: string;
  label: 0 | 1;
  originalIndex: number;
  dwellTimeSeconds: number | null;
}

export interface Impression {
  id: string;
  userId: string;
  start: string;
  end: string;
  startDate: string;
  candidates: Candidate[];
}

export interface ReaderProfile {
  userId: string;
  history: HistoryEntry[];
  impression: Impression;
}

export type NewsIndex = Record<string, NewsPreview>;
