import { ArrowDown, ArrowRight, ArrowUpRight, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useReader } from "../../app/readerState";
import { EditionMark, ProfileSelected } from "../../components/layout";
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
  const featured =
    readers.find((reader) => reader.userId === "U216199") ?? readers[0];
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
          <EditionMark label="A DIFFERENT KIND OF DAILY" />
          <h1>
            A world of stories.
            <br />
            <em>An edition of your own.</em>
          </h1>
          <p>
            Different interests. Different perspectives. Find a reader
            <br className="desktop-break" /> like you, and make a little room
            for a good read.
          </p>
          <Link to="#choose-reader" className="text-link hero-scroll">
            Find your edition <ArrowDown size={16} />
          </Link>
        </div>
        <aside className="welcome-note">
          <div className="editorial-sun" aria-hidden="true">
            ✳
          </div>
          <span className="eyebrow">HELLO, CURIOUS READER.</span>
          <p>
            There’s more than one
            <br />
            way to see the world.
          </p>
          <span>
            Choose an anonymous reader below.
            <br />
            Their reading history shapes your edition.
          </span>
        </aside>
      </div>
      <div className="reader-intro-line">
        <span>
          <span className="tiny-diamond" />
          {readers.length} readers. Countless perspectives.
        </span>
        <span>No account. Just curiosity.</span>
      </div>
      {!search && featured && (
        <section className="featured-reader">
          <div className="featured-reader-intro">
            <span className="eyebrow">A PLACE TO BEGIN</span>
            <h2>
              Meet your next <br />
              <em>reading habit.</em>
            </h2>
            <p>Take a look through a reader’s lens.</p>
          </div>
          <div className="featured-reader-detail">
            <ProfileSelected selected={selectedId === featured.userId} />
            <span className="reader-id-label">READER {featured.userId}</span>
            <div className="reader-counts">
              <span>
                <strong>{featured.historyCount}</strong> articles read
              </span>
              <span>
                <strong>{featured.candidateCount}</strong> stories in this
                edition
              </span>
            </div>
            <div className="reader-interests">
              {featured.topCategories.slice(0, 3).map((item) => (
                <span key={item.category}>{categoryLabel(item.category)}</span>
              ))}
            </div>
            <button
              className="button button-dark"
              onClick={() => choose(featured.userId)}
            >
              Start reading <ArrowUpRight size={17} />
            </button>
          </div>
          <div className="featured-reader-quote">
            <span className="eyebrow">FROM THIS READER’S HISTORY</span>
            <p>“{featured.previewHeadline}”</p>
            <span>One story can lead to another.</span>
          </div>
        </section>
      )}
      <section id="choose-reader" className="reader-selection">
        <div className="reader-section-top">
          <div>
            <h2>Find your point of view.</h2>
            <p>Interests are drawn from each reader’s past reading.</p>
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
              : "THE READER COLLECTION"}
          </span>
          <span>
            {search
              ? "Find a familiar perspective"
              : `01 — ${String(readers.length).padStart(2, "0")}`}
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
      <div className="readers-closing">
        <span className="tiny-diamond" />
        <p>
          The same world. <em>A different read.</em>
        </p>
        <span className="tiny-diamond" />
      </div>
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
