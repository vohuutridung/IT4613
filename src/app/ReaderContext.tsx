import { useEffect, useState, type ReactNode } from "react";
import { getNewsIndex, getProfile, getReaders } from "../services/data";
import { useResource } from "../hooks/useResource";
import { ReaderContext } from "./readerState";

const STORAGE_KEY = "the-daily.reader";

export function ReaderProvider({ children }: { children: ReactNode }) {
  const [selectedId, setSelectedId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  });
  const readers = useResource("readers", getReaders);
  const edition = useResource(`edition:${selectedId}`, async () => {
    if (!selectedId || !/^U\d+$/.test(selectedId))
      return { profile: undefined, news: {} };
    const [profile, news] = await Promise.all([
      getProfile(selectedId),
      getNewsIndex(),
    ]);
    return { profile, news };
  });
  useEffect(() => {
    if (
      readers.data &&
      selectedId &&
      !readers.data.some((reader) => reader.userId === selectedId)
    ) {
      setSelectedId(null);
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        /* Reading works without storage. */
      }
    }
  }, [readers.data, selectedId]);
  const selectReader = (id: string) => {
    if (!readers.data?.some((reader) => reader.userId === id)) return;
    setSelectedId(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* Keep selection for this session. */
    }
  };
  return (
    <ReaderContext.Provider
      value={{
        readers: readers.data ?? [],
        selectedId,
        selectReader,
        profile: edition.data?.profile,
        news: edition.data?.news ?? {},
        loading: readers.loading,
        profileLoading: edition.loading,
        error: readers.error || (selectedId ? edition.error : undefined),
        readersError: readers.error,
        retry: () => {
          readers.retry();
          edition.retry();
        },
      }}
    >
      {children}
    </ReaderContext.Provider>
  );
}
