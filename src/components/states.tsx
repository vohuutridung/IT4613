import { ArrowLeft, ArrowRight, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";

export function LoadingState({ article = false }: { article?: boolean }) {
  return (
    <main
      id="main-content"
      className={`page-width loading-layout ${article ? "loading-article" : ""}`}
      role="status"
      aria-label="Loading your edition"
    >
      <span className="sr-only">Loading your edition…</span>
      <div className="skeleton skeleton-eyebrow" />
      <div className="skeleton skeleton-title" />
      <div className="skeleton skeleton-subtitle" />
      <div className="loading-columns">
        <div>
          {Array.from({ length: 4 }, (_, index) => (
            <div key={index} className="skeleton-story">
              <div className="skeleton skeleton-eyebrow" />
              <div className="skeleton skeleton-headline" />
              <div className="skeleton skeleton-line" />
              <div className="skeleton skeleton-line short" />
            </div>
          ))}
        </div>
        <div className="skeleton-sidebar">
          <div className="skeleton skeleton-headline" />
          <div className="skeleton skeleton-line" />
        </div>
      </div>
    </main>
  );
}

export function ErrorState({
  message,
  retry,
}: {
  message?: string;
  retry?: () => void;
}) {
  return (
    <main id="main-content" className="page-width message-state" role="alert">
      <span className="eyebrow">A SMALL INTERRUPTION</span>
      <h1>Let’s turn the page.</h1>
      <p>{message || "We couldn’t load this edition. Please try again."}</p>
      <div className="message-actions">
        {retry && (
          <button className="button button-dark" onClick={retry}>
            Try again <RefreshCw size={15} />
          </button>
        )}
        <Link className="text-link" to="/readers">
          Choose a reader <ArrowRight size={16} />
        </Link>
      </div>
    </main>
  );
}

export function NotFound({ article = false }: { article?: boolean }) {
  return (
    <main id="main-content" className="page-width message-state">
      <span className="eyebrow">
        {article ? "STORY NOT FOUND" : "404 / A MISSING PAGE"}
      </span>
      <h1>
        This page slipped
        <br />
        out of the edition.
      </h1>
      <p>
        {article
          ? "This story isn’t available in our archive. There are more good reads waiting in your edition."
          : "Let’s find you something worth reading."}
      </p>
      <Link className="button button-dark" to="/for-you">
        <ArrowLeft size={16} /> Back to your edition
      </Link>
    </main>
  );
}
