import { useEffect, useState } from "react";

export function useResource<T>(key: string, loader: () => Promise<T>) {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{
    key: string;
    data?: T;
    error?: string;
  }>({ key: "" });
  useEffect(() => {
    let active = true;
    setResult({ key });
    loader()
      .then((data) => {
        if (active) setResult({ key, data });
      })
      .catch((error) => {
        if (active)
          setResult({
            key,
            error:
              error instanceof Error
                ? error.message
                : "Something went wrong. Please try again.",
          });
      });
    return () => {
      active = false;
    };
    // A resource key identifies its loader; callbacks need not be memoized by callers.
  }, [key, attempt]);
  const current = result.key === key ? result : { key };
  return {
    ...current,
    loading: !current.data && !current.error,
    retry: () => setAttempt((value) => value + 1),
  };
}
