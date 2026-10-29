import { ArrowUpRight } from "lucide-react";
import { Link } from "react-router-dom";
import { useReader } from "../../app/readerState";
import {
  FeaturedStory,
  HistoryItem,
  HistoryLink,
  StoryCard,
  StoryRow,
} from "../../components/editorial";
import { ErrorState, LoadingState } from "../../components/states";
import { useResource } from "../../hooks/useResource";
import { sortHistory } from "../../lib/format";
import { rankingProvider } from "../../services/ranking";

export function FeedPage() {
  const { profile, news, profileLoading, error, retry } = useReader();
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
  return (
    <main id="main-content" className="page-width feed-page">
      <section className="feed-intro">
        <div>
          <h1>
            Your daily dose
            <br />
            of <em>perspective.</em>
          </h1>
        </div>
      </section>
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
                <div className="more-stories">
                  {stories.slice(3).map((article, index) => (
                    <StoryRow
                      key={article.id}
                      article={article}
                      number={index + 4}
                    />
                  ))}
                </div>
              )}
              <div className="edition-end">
                <span className="end-ornament">✳</span>
                <h2>And that’s your edition.</h2>
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
            <h2>
              Recently read<span>.</span>
            </h2>
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
        </aside>
      </div>
    </main>
  );
}
