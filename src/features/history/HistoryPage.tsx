import { ArrowRight, ArrowUpRight, BookOpen } from "lucide-react";
import { Link } from "react-router-dom";
import { useReader } from "../../app/readerState";
import { HistoryItem } from "../../components/editorial";
import { EditionMark } from "../../components/layout";
import { ErrorState, LoadingState } from "../../components/states";
import {
  categoryLabel,
  formatArchiveDate,
  sortHistory,
} from "../../lib/format";
import type { HistoryEntry } from "../../types";

export function HistoryPage() {
  const { profile, profileLoading, error, retry, news, readers } = useReader();
  if (error) return <ErrorState message={error} retry={retry} />;
  if (profileLoading || !profile) return <LoadingState />;
  const history = sortHistory(profile.history);
  const groups = new Map<string, HistoryEntry[]>();
  for (const entry of history) {
    const day = entry.exposureSortKey.slice(0, 10);
    groups.set(day, [...(groups.get(day) ?? []), entry]);
  }
  const summary = readers.find((reader) => reader.userId === profile.userId);
  return (
    <main id="main-content" className="page-width history-page">
      <div className="edition-topline">
        <EditionMark label="THE STORIES THAT STAY WITH YOU" />
        <span>READER {profile.userId}</span>
      </div>
      <section className="history-intro">
        <div>
          <span className="eyebrow">READING HISTORY</span>
          <h1>
            A trail of
            <br />
            <em>good reads.</em>
          </h1>
          <p>
            A thought revisited. A story worth a second look.
            <br />
            Your personal reading archive, all in one place.
          </p>
        </div>
        <div className="history-count">
          <BookOpen size={24} strokeWidth={1.2} />
          <strong>{history.length}</strong>
          <span>stories along the way</span>
        </div>
      </section>
      <div className="feed-edition-strip">
        <span>YOUR READING ARCHIVE</span>
        <span>
          Most recent first <ArrowDownIcon />
        </span>
      </div>
      <div className="editorial-layout history-layout">
        <div className="history-main">
          {groups.size ? (
            [...groups].map(([date, entries]) => (
              <section className="history-date-group" key={date}>
                <div className="history-date">
                  <span className="tiny-diamond" />
                  <h2>{formatArchiveDate(date)}</h2>
                  <span>
                    {entries.length}{" "}
                    {entries.length === 1 ? "story" : "stories"}
                  </span>
                </div>
                <div className="history-entries">
                  {entries.map((entry, index) => (
                    <HistoryItem
                      key={`${entry.newsId}-${index}`}
                      entry={entry}
                      article={news[entry.newsId]}
                    />
                  ))}
                </div>
              </section>
            ))
          ) : (
            <div className="empty-search">
              <h2>Your next chapter starts here.</h2>
              <p>This reader has no previous reading history.</p>
            </div>
          )}
          <div className="history-end">
            <span className="tiny-diamond" />
            The beginning of your reading trail.
          </div>
        </div>
        <aside className="editorial-sidebar">
          <section className="history-aside-note">
            <span className="eyebrow">ONE STORY LEADS TO ANOTHER</span>
            <h2>
              What’s on <br />
              <em>your mind?</em>
            </h2>
            <p>
              Your reading history offers a glimpse of the things that catch
              your eye.
            </p>
            <div className="reader-interests">
              {summary?.topCategories.map((item) => (
                <span key={item.category}>{categoryLabel(item.category)}</span>
              ))}
            </div>
            <p className="history-aside-small">
              Reading times here are recorded time spent with each story, as
              preserved in the archive.
            </p>
            <Link to="/for-you" className="text-link">
              Find your next read <ArrowUpRight size={16} />
            </Link>
          </section>
          <Link to="/readers" className="history-switch">
            A different reader.
            <br />A different perspective.
            <ArrowRight size={18} />
          </Link>
        </aside>
      </div>
    </main>
  );
}

function ArrowDownIcon() {
  return <span aria-hidden="true">↓</span>;
}
