import { ArrowDown, ArrowUpRight } from "lucide-react";
import { Link } from "react-router-dom";
import { useReader } from "../../app/readerState";
import {
  FeaturedStory,
  HistoryItem,
  HistoryLink,
  SectionHeading,
  StoryCard,
  StoryRow,
} from "../../components/editorial";
import { EditionMark } from "../../components/layout";
import { ErrorState, LoadingState } from "../../components/states";
import { useResource } from "../../hooks/useResource";
import {
  categoryLabel,
  formatArchiveDate,
  sortHistory,
} from "../../lib/format";
import { rankingProvider } from "../../services/ranking";

export function FeedPage() {
  const { profile, selectedId, news, profileLoading, error, retry, readers } =
    useReader();
  const ranked = useResource(`ranking:${profile?.userId ?? ""}`, async () =>
    profile
      ? rankingProvider.rank({
          userId: profile.userId,
          impression: profile.impression,
          history: profile.history,
        })
      : [],
  );
  if (error) return <ErrorState message={error} retry={retry} />;
  if (profileLoading || !profile || ranked.loading) return <LoadingState />;
  if (ranked.error)
    return <ErrorState message={ranked.error} retry={ranked.retry} />;
  const stories = (ranked.data ?? []).map(
    (candidate) => news[candidate.newsId],
  );
  const history = sortHistory(profile.history);
  const reader = readers.find((item) => item.userId === selectedId);
  return (
    <main id="main-content" className="page-width feed-page">
      <div className="edition-topline">
        <EditionMark />
        <span>
          FROM THE ARCHIVE <i /> {formatArchiveDate(profile.impression.start)}
        </span>
      </div>
      <section className="feed-intro">
        <div>
          <span className="eyebrow">FOR YOU</span>
          <h1>
            Your daily dose
            <br />
            of <em>perspective.</em>
          </h1>
        </div>
        <div className="feed-intro-note">
          <p>
            A few familiar interests.
            <br />A few new discoveries.
            <br />
            All in your own edition.
          </p>
          <Link to="#your-stories" className="text-link">
            {stories.length} stories to explore <ArrowDown size={15} />
          </Link>
        </div>
      </section>
      <div className="feed-edition-strip">
        <span>THE {selectedId} EDITION</span>
        <span>
          Chosen around your reading interests <span className="tiny-diamond" />
        </span>
      </div>
      <div className="editorial-layout" id="your-stories">
        <div className="feed-main">
          {stories.length ? (
            <>
              <FeaturedStory article={stories[0]} />
              <div className="secondary-stories">
                {stories.slice(1, 3).map((article, index) => (
                  <StoryCard
                    key={article.id}
                    article={article}
                    number={index + 2}
                  />
                ))}
              </div>
              {stories.length > 3 && (
                <section className="more-stories">
                  <SectionHeading
                    aside={<span className="quiet-label">TAKE YOUR TIME</span>}
                  >
                    More to your story
                  </SectionHeading>
                  {stories.slice(3).map((article, index) => (
                    <StoryRow
                      key={article.id}
                      article={article}
                      number={index + 4}
                    />
                  ))}
                </section>
              )}
              <div className="edition-end">
                <span className="end-ornament">✳</span>
                <h2>And that’s your edition.</h2>
                <p>Good reading is time well spent.</p>
                <Link to="/readers" className="text-link">
                  Explore another perspective <ArrowUpRight size={16} />
                </Link>
              </div>
            </>
          ) : (
            <div className="empty-search">
              <h2>A quiet edition.</h2>
              <p>There are no stories available for this reader.</p>
              <Link className="text-link" to="/readers">
                Try another reader <ArrowUpRight size={16} />
              </Link>
            </div>
          )}
        </div>
        <aside className="editorial-sidebar">
          <section className="history-sidebar">
            <div className="sidebar-heading">
              <span className="eyebrow">YOUR READING TRAIL</span>
              <span className="sidebar-symbol" aria-hidden="true">
                ↗
              </span>
            </div>
            <h2>
              Recently read<span>.</span>
            </h2>
            <p className="sidebar-description">Pick up a familiar thread.</p>
            {history.slice(0, 4).map((entry) => (
              <HistoryItem
                key={`${entry.newsId}-${entry.exposureSortKey}`}
                entry={entry}
                article={news[entry.newsId]}
                compact
              />
            ))}
            <HistoryLink />
          </section>
          <section className="interest-note">
            <span className="eyebrow">A LITTLE MORE YOU</span>
            <h2>
              Follow your <br />
              <em>curiosity.</em>
            </h2>
            <p>A few interests from your reading history.</p>
            <div className="interest-list">
              {reader?.topCategories.slice(0, 3).map((interest, index) => (
                <div key={interest.category}>
                  <span className="interest-number">0{index + 1}</span>
                  <span>{categoryLabel(interest.category)}</span>
                  <span className="tiny-diamond" />
                </div>
              ))}
            </div>
            <Link to="/history" className="text-link">
              See your reading history <ArrowUpRight size={15} />
            </Link>
          </section>
          <div className="sidebar-colophon">
            <span className="colophon-mark">D.</span>
            <p>
              Less noise.
              <br />
              More perspective.
            </p>
            <span>THE DAILY.</span>
          </div>
        </aside>
      </div>
    </main>
  );
}
