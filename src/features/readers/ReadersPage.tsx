import { ArrowDown, ArrowRight, ArrowUpRight, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useReader } from "../../app/readerState";
import { ErrorState, LoadingState } from "../../components/states";
import { categoryLabel } from "../../lib/format";
import type { ReaderSummary } from "../../types";

export function ReadersPage() {
  const { readers, loading, readersError, retry, selectedId, selectReader } =
    useReader();
  const [search, setSearch] = useState("");
  const [visible, setVisible] = useState(12);
  const navigate = useNavigate();
  const filtered = useMemo(
    () =>
      readers.filter((reader) =>
        reader.userId.toLowerCase().includes(search.trim().toLowerCase()),
      ),
    [readers, search],
  );
  const choose = (id: string) => {
    selectReader(id);
    navigate("/for-you");
  };
  if (loading) return <LoadingState />;
  if (readersError) return <ErrorState message={readersError} retry={retry} />;
  return (
    <main id="main-content" className="page-width readers-page">
      <div className="readers-hero">
        <div>
          <h1>
            A world of stories.
            <br />
            <em>An edition of your own.</em>
          </h1>
        </div>
      </div>
      <section id="choose-reader" className="reader-selection">
        <div className="reader-section-top">
          <div>
            <h2>Find your point of view.</h2>
          </div>
          <label className="search-field">
            <Search size={17} aria-hidden="true" />
            <input
              aria-label="Search readers by ID"
              placeholder="Search by reader ID"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setVisible(12);
              }}
            />
            {search && (
              <button aria-label="Clear search" onClick={() => setSearch("")}>
                <X size={16} />
              </button>
            )}
          </label>
        </div>
        <div className="profile-list-meta">
          <span>
            {search
              ? `${filtered.length} ${filtered.length === 1 ? "reader" : "readers"} found`
              : `${readers.length} readers`}
          </span>
        </div>
        {filtered.length ? (
          <>
            <div className="profile-grid">
              {filtered.slice(0, visible).map((reader, index) => (
                <ReaderCard
                  key={reader.userId}
                  reader={reader}
                  number={index + 1}
                  selected={selectedId === reader.userId}
                  choose={choose}
                />
              ))}
            </div>
            {visible < filtered.length && (
              <div className="load-more">
                <span>
                  Showing {Math.min(visible, filtered.length)} of{" "}
                  {filtered.length} readers
                </span>
                <button
                  className="button button-outline"
                  onClick={() => setVisible((value) => value + 12)}
                >
                  Meet more readers <ArrowDown size={15} />
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="empty-search">
            <h3>No reader by that name.</h3>
            <p>Try another ID, or explore the whole collection.</p>
            <button className="text-link" onClick={() => setSearch("")}>
              Show all readers <ArrowRight size={16} />
            </button>
          </div>
        )}
      </section>
    </main>
  );
}

function ReaderCard({
  reader,
  number,
  selected,
  choose,
}: {
  reader: ReaderSummary;
  number: number;
  selected: boolean;
  choose: (id: string) => void;
}) {
  return (
    <article className={`reader-card ${selected ? "is-selected" : ""}`}>
      <div className="reader-card-top">
        <span className="reader-id-label">READER {reader.userId}</span>
        <span className="reader-card-number">
          {String(number).padStart(2, "0")}
        </span>
      </div>
      <div className="reader-card-counts">
        <span>
          <strong>{reader.historyCount}</strong> articles read
        </span>
        <span>{reader.candidateCount} stories waiting</span>
      </div>
      <div className="reader-card-interests">
        {reader.topCategories
          .slice(0, 3)
          .map((item) => categoryLabel(item.category))
          .join(" · ")}
      </div>
      <button
        className="reader-card-button"
        onClick={() => choose(reader.userId)}
      >
        {selected ? "Return to your edition" : "Start reading"}
        <ArrowUpRight size={17} />
      </button>
    </article>
  );
}
