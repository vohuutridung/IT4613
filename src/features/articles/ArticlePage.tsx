import { ArrowLeft, ArrowRight } from "lucide-react";
import { useEffect, useRef } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { useReader } from "../../app/readerState";
import {
  CategoryLabel,
  SectionHeading,
  StoryCard,
} from "../../components/editorial";
import { ErrorState, LoadingState, NotFound } from "../../components/states";
import { useResource } from "../../hooks/useResource";
import { formatParagraphs } from "../../lib/paragraphs";
import { getArticle } from "../../services/data";
import { rankingProvider } from "../../services/ranking";

export function ArticlePage() {
  const { newsId = "" } = useParams();
  const article = useResource(`article:${newsId}`, () => getArticle(newsId));
  const { profile, news } = useReader();
  const ranked = useResource(`continue:${profile?.userId ?? ""}`, async () =>
    profile
      ? rankingProvider.rank({
          userId: profile.userId,
          impression: profile.impression,
          history: profile.history,
        })
      : [],
  );
  const location = useLocation();
  const navigate = useNavigate();
  const contentRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!article.data) return;
    const previous = document.title;
    document.title = `${article.data.headline} — The Daily.`;
    return () => {
      document.title = previous;
    };
  }, [article.data]);
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const content = contentRef.current;
      if (!content || !progressRef.current) return;
      const bottom = content.getBoundingClientRect().bottom + window.scrollY;
      const distance = bottom - window.innerHeight;
      const value =
        distance <= 0 ? 1 : Math.max(0, Math.min(1, window.scrollY / distance));
      progressRef.current.style.transform = `scaleX(${value})`;
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    const observer = new ResizeObserver(schedule);
    if (contentRef.current) observer.observe(contentRef.current);
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    update();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [article.data]);
  if (article.loading) return <LoadingState article />;
  if (article.error)
    return article.error.includes("found") ? (
      <NotFound article />
    ) : (
      <ErrorState message={article.error} retry={article.retry} />
    );
  if (!article.data) return <NotFound article />;
  const story = article.data;
  const paragraphs = formatParagraphs(story.body);
  const continued = (ranked.data ?? [])
    .filter((item) => item.newsId !== newsId)
    .slice(0, 2)
    .map((item) => news[item.newsId])
    .filter(Boolean);
  const from =
    typeof location.state?.from === "string" ? location.state.from : null;
  return (
    <main id="main-content" className="article-page">
      <div className="reading-progress-track" aria-hidden="true">
        <div ref={progressRef} className="reading-progress" />
      </div>
      <div className="page-width article-topbar">
        {from ? (
          <button className="text-link" onClick={() => navigate(-1)}>
            <ArrowLeft size={16} />
            {from === "/history"
              ? "Back to reading history"
              : from.startsWith("/article/")
                ? "Back to previous story"
                : "Back to your edition"}
          </button>
        ) : (
          <Link className="text-link" to="/for-you">
            <ArrowLeft size={16} /> Back to your edition
          </Link>
        )}
      </div>
      <article className="article-content" ref={contentRef}>
        <header className="article-header">
          <CategoryLabel article={story} />
          <h1>{story.headline}</h1>
        </header>
        <div className="article-body">
          {paragraphs.map((paragraph, index) => (
            <p key={index}>{paragraph}</p>
          ))}
        </div>
        <div className="article-endmark" aria-label="End of article">
          <span className="tiny-diamond" />
          <span className="tiny-diamond" />
          <span className="tiny-diamond" />
        </div>
        <div className="article-closing">
          <Link to="/for-you" className="text-link">
            Your edition <ArrowRight size={15} />
          </Link>
        </div>
      </article>
      {continued.length > 0 && (
        <section className="continue-reading page-width">
          <SectionHeading
            aside={
              <Link to="/for-you" className="text-link">
                All your stories <ArrowRight size={15} />
              </Link>
            }
          >
            There’s more to the story.
          </SectionHeading>
          <div className="continue-grid">
            {continued.map((item, index) => (
              <StoryCard key={item.id} article={item} number={index + 1} />
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
