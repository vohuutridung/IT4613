import { ArrowRight, ArrowUpRight, Clock3 } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import {
  categoryLabel,
  topicLabel,
  formatArchiveDate,
  formatArchiveTime,
  formatDuration,
} from "../lib/format";
import type { HistoryEntry, NewsPreview } from "../types";

export function CategoryLabel({
  article,
  topic = true,
}: {
  article: Pick<NewsPreview, "category" | "topic">;
  topic?: boolean;
}) {
  const category = categoryLabel(article.category);
  const detail = topicLabel(article.topic, article.category);
  return (
    <div className="category-label">
      <span>{category}</span>
      {topic && detail && detail !== category && (
        <>
          <i />
          {detail}
        </>
      )}
    </div>
  );
}

export function ReadingTime({ minutes }: { minutes: number }) {
  return (
    <span className="reading-time">
      <Clock3 size={12} aria-hidden="true" /> Est. {minutes} min read
    </span>
  );
}

export function ArticleLink({
  article,
  children,
  className = "",
}: {
  article: Pick<NewsPreview, "id" | "headline">;
  children: React.ReactNode;
  className?: string;
}) {
  const location = useLocation();
  return (
    <Link
      to={`/article/${article.id}`}
      state={{ from: location.pathname }}
      className={className}
    >
      {children}
    </Link>
  );
}

export function FeaturedStory({ article }: { article: NewsPreview }) {
  return (
    <article className="featured-story">
      <div className="featured-topline">
        <span className="eyebrow">THE LEAD STORY</span>
        <span className="feature-number">01</span>
      </div>
      <CategoryLabel article={article} />
      <h2>
        <ArticleLink article={article}>{article.headline}</ArticleLink>
      </h2>
      <p className="featured-excerpt">{article.excerpt}</p>
      <div className="featured-bottom">
        <ArticleLink article={article} className="text-link accent-link">
          Read the story <ArrowUpRight size={17} />
        </ArticleLink>
        <ReadingTime minutes={article.estimatedReadingMinutes} />
      </div>
      <span className="feature-flourish" aria-hidden="true">
        D.
      </span>
    </article>
  );
}

export function StoryCard({
  article,
  number,
}: {
  article: NewsPreview;
  number: number;
}) {
  return (
    <article className="story-card">
      <div className="story-card-top">
        <CategoryLabel article={article} topic={false} />
        <span className="story-index">{String(number).padStart(2, "0")}</span>
      </div>
      <h2>
        <ArticleLink article={article}>{article.headline}</ArticleLink>
      </h2>
      <p>{article.excerpt}</p>
      <div className="story-card-bottom">
        <ReadingTime minutes={article.estimatedReadingMinutes} />
        <ArticleLink article={article} className="story-arrow">
          <span className="sr-only">Read {article.headline}</span>
          <ArrowUpRight size={19} />
        </ArticleLink>
      </div>
    </article>
  );
}

export function StoryRow({
  article,
  number,
}: {
  article: NewsPreview;
  number: number;
}) {
  return (
    <article className="story-row">
      <span className="story-index">{String(number).padStart(2, "0")}</span>
      <div>
        <CategoryLabel article={article} />
        <h2>
          <ArticleLink article={article}>{article.headline}</ArticleLink>
        </h2>
        <p>{article.excerpt}</p>
        <ReadingTime minutes={article.estimatedReadingMinutes} />
      </div>
      <ArticleLink article={article} className="story-arrow">
        <span className="sr-only">Read {article.headline}</span>
        <ArrowUpRight size={20} />
      </ArticleLink>
    </article>
  );
}

export function HistoryItem({
  entry,
  article,
  compact = false,
}: {
  entry: HistoryEntry;
  article: NewsPreview;
  compact?: boolean;
}) {
  return (
    <article className={compact ? "history-preview-item" : "history-item"}>
      <CategoryLabel article={article} topic={!compact} />
      <h3>
        <ArticleLink article={article}>{article.headline}</ArticleLink>
      </h3>
      {!compact && <p className="history-excerpt">{article.excerpt}</p>}
      <div className="history-metadata">
        {compact ? (
          <span>{formatArchiveDate(entry.exposureTime, { short: true })}</span>
        ) : (
          <span>{formatArchiveTime(entry.exposureTime)}</span>
        )}
        <span className="meta-dot">·</span>
        <span>Read for {formatDuration(entry.dwellTimeSeconds)}</span>
      </div>
    </article>
  );
}

export function SectionHeading({
  children,
  aside,
}: {
  children: React.ReactNode;
  aside?: React.ReactNode;
}) {
  return (
    <div className="section-heading">
      <h2>{children}</h2>
      {aside}
    </div>
  );
}

export function HistoryLink() {
  return (
    <Link to="/history" className="text-link">
      Your full reading history <ArrowRight size={15} />
    </Link>
  );
}
