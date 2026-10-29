import { useEffect } from "react";
import { useLocation, useNavigationType } from "react-router-dom";

const positions = new Map<string, number>();

export function ScrollRestoration() {
  const { key, hash, pathname, search } = useLocation();
  const positionKey = `${pathname}${search}:${key}`;
  const navigationType = useNavigationType();
  useEffect(() => {
    const previous = history.scrollRestoration;
    history.scrollRestoration = "manual";
    return () => {
      history.scrollRestoration = previous;
    };
  }, []);
  useEffect(() => {
    let target = 0;
    let savedPosition: number | undefined;
    if (navigationType === "POP") {
      savedPosition = positions.get(positionKey);
      try {
        const stored = sessionStorage.getItem(`daily.scroll.${positionKey}`);
        if (savedPosition === undefined && stored !== null)
          savedPosition = Number(stored);
      } catch {
        /* Browser storage is optional. */
      }
      target = Number.isFinite(savedPosition) ? savedPosition! : 0;
    }
    let restoring = true;
    let frame = 0;
    const started = performance.now();
    const restore = () => {
      if (hash && savedPosition === undefined) {
        let id = hash.slice(1);
        try {
          id = decodeURIComponent(id);
        } catch {
          /* Keep an invalid hash literal. */
        }
        const element = document.getElementById(id);
        if (element) {
          element.scrollIntoView();
          restoring = false;
          return;
        }
        if (performance.now() - started < 5000) {
          frame = requestAnimationFrame(restore);
          return;
        }
      }
      window.scrollTo(0, target);
      if (
        document.documentElement.scrollHeight - innerHeight >= target ||
        performance.now() - started > 5000
      )
        restoring = false;
      else frame = requestAnimationFrame(restore);
    };
    restore();
    const remember = () => {
      if (!restoring) positions.set(positionKey, window.scrollY);
    };
    window.addEventListener("scroll", remember, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", remember);
      const position = positions.get(positionKey) ?? window.scrollY;
      positions.set(positionKey, position);
      try {
        sessionStorage.setItem(`daily.scroll.${positionKey}`, String(position));
      } catch {
        /* Optional persistence. */
      }
    };
  }, [positionKey, navigationType, hash]);
  return null;
}
