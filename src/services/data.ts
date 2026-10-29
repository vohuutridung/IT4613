import type {
  NewsArticle,
  NewsPreview,
  ReaderProfile,
  ReaderSummary,
} from "../types";

const requests = new Map<string, Promise<unknown>>();

function load<T>(path: string): Promise<T> {
  if (!requests.has(path)) {
    const request = fetch(`${import.meta.env.BASE_URL}data/${path}`)
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            response.status === 404
              ? "This story or reader could not be found."
              : "We couldn’t load this edition. Please try again.",
          );
        if (!response.headers.get("content-type")?.includes("application/json"))
          throw new Error("This story or reader could not be found.");
        try {
          return (await response.json()) as T;
        } catch {
          throw new Error(
            "We couldn’t read this part of the archive. Please try again.",
          );
        }
      })
      .catch((error) => {
        requests.delete(path);
        throw error;
      });
    requests.set(path, request);
  }
  return requests.get(path) as Promise<T>;
}

export const getReaders = () => load<ReaderSummary[]>("readers-index.json");
export const getNewsIndex = () =>
  load<Record<string, NewsPreview>>("news-index.json");
export const getProfile = (id: string) =>
  load<ReaderProfile>(`profiles/${encodeURIComponent(id)}.json`);
export const getArticle = (id: string) => {
  if (!/^N\d+$/.test(id))
    return Promise.reject(new Error("This story could not be found."));
  return load<NewsArticle>(`articles/${id}.json`);
};
