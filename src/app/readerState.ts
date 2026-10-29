import { createContext, useContext } from "react";
import type { NewsPreview, ReaderProfile, ReaderSummary } from "../types";

interface ReaderContextValue {
  readers: ReaderSummary[];
  selectedId: string | null;
  selectReader: (id: string) => void;
  profile?: ReaderProfile;
  news: Record<string, NewsPreview>;
  loading: boolean;
  profileLoading: boolean;
  error?: string;
  readersError?: string;
  retry: () => void;
}

// Keep the context outside the component's Fast Refresh boundary.
export const ReaderContext = createContext<ReaderContextValue | null>(null);

export function useReader() {
  const context = useContext(ReaderContext);
  if (!context) throw new Error("ReaderProvider is required.");
  return context;
}
