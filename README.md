# The Daily.

A responsive, text-first personalized newspaper built from the curated PENS subset in this repository. Choose an anonymous reader, explore their edition, revisit their reading history, and open the complete supplied articles.

## Run locally

Requires Node.js 22.12+ and Python 3.10+; Python uses only its standard library.

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. `npm run dev` automatically validates and prepares the data before starting. Reader selection persists in local storage; the original reading history remains read-only.

For a complete validation and production-build workflow:

```sh
npm run prepare:data
npm run check:data
npm test
npm run build
npm run preview
```

| Command | Purpose |
| --- | --- |
| `npm run prepare:data` | Validate source TSVs and regenerate static JSON. |
| `npm run check:data` | Validate source TSVs without writing generated data. |
| `npm test` | Automatically prepare data, then run Python data-contract and TypeScript domain tests. |
| `npm run build` | Automatically prepare data, type-check TypeScript, and build into `dist/`. |
| `npm run preview` | Serve the existing production build locally. |

## Data and reading semantics

The immutable source files are `data/display.tsv` and `data/display_news.tsv`. Validation confirms **57 unique readers**, **1,669 unique articles**, **1,427 historical interactions**, and **1,252 impression candidates**. Every referenced article resolves. Each reader has one selected impression. Reader `U216199` has 34 history entries and 23 candidates: 7 clicked and 16 unclicked.

Recommendations contain exactly the selected impression's candidates. The initial ordering places `pos` before `neg`, preserving the supplied order within each group. It includes both groups and does not retrieve unrelated corpus articles. Historical entries come exclusively from `ClicknewsID`; opening an article does not change them. Interest labels are inferred from history category frequencies.

Source timestamps use `M/D/YYYY h:mm:ss AM/PM`. Preprocessing parses them explicitly into naive calendar sort keys while preserving the original strings. No timezone is inferred or converted. History appears most recent first with aligned exposure times and recorded dwell durations. Archive dates describe interactions or the impression, **not article publication dates**. Estimated reading time uses article word count at 220 words per minute and is distinct from recorded dwell time.

The dataset supplies no article photographs, authors, verified publishers, publication dates, or original URLs. The interface uses the supplied text without inventing those fields. Article formatting adds conservative paragraph breaks while retaining wording, punctuation, and order; content renders as plain text.

## Architecture

React, TypeScript, Vite, Tailwind CSS, React Router, and Lucide React form the frontend. Newsreader and DM Sans have local serif/sans-serif fallbacks.

```text
src/app/                 Routing, reader state, and scroll restoration
src/components/          Shared editorial blocks, layout, and loading/error states
src/features/readers/    Profile collection and ID search
src/features/feed/       Personalized edition
src/features/history/    Chronological reading archive
src/features/articles/   Article reader and reading progress
src/services/data.ts     Cached static-data loading
src/services/ranking/    Ranking contract and replaceable provider
src/lib/                 Category, date, duration, and paragraph formatting
src/types/               Normalized data contracts
src/styles/              Editorial design and responsive behavior
scripts/                 Deterministic preprocessing and Python validation tests
tests/                   TypeScript domain tests
```

Preprocessing generates the following ignored files; rebuild them instead of editing them:

```text
public/data/
├── readers-index.json        Lightweight reader summaries and interests
├── news-index.json           Article previews; no full article bodies
├── data-report.json          Counts, source SHA256 checksums, and timestamp policy
├── profiles/<UserID>.json    One reader's history and impression
└── articles/<NewsID>.json    Original full article text, loaded on demand
```

The profile-selection screen loads reader summaries. Selecting a reader loads their profile and the shared preview index. The article page fetches its body separately. Successfully requested JSON is cached for the browser session, and failed requests can be retried. Full bodies are neither embedded in the JavaScript bundle nor loaded with the initial reader collection.

The processor checks exact headers, unique identifiers, nonempty article text, resolvable references, positional list alignment, finite nonnegative durations, valid calendar timestamps, and disjoint positive/negative candidate sets. It preserves positive candidate dwell times, produces deterministic internal impression IDs, and fails visibly on integrity errors.

## Replace the ranking provider

`src/services/ranking/types.ts` defines the boundary:

```ts
interface Ranker {
  rank(context: RankingContext): Promise<RankedCandidate[]>;
}
```

`RankingContext` contains the reader ID, impression, and history. Results contain `newsId`, `rank`, and an optional internal score. `LabelRanker` supplies the initial deterministic ordering. `createRankingProvider()` in `src/services/ranking/index.ts` accepts a replacement `Ranker`; inject a model or API adapter there and keep the feed and article components unchanged.

A future adapter can call `POST /api/rank` with that context. It must return **every candidate in the current impression exactly once**, with no history-only or unrelated IDs. Validate that contract at the adapter boundary and keep scores and provider configuration out of the reading interface. No trained-model backend is included in this version.

## Static hosting

Deploy the contents of `dist/` after `npm run build`. Serve real files under `/data/` directly, and configure a single-page-app fallback to `/index.html` for frontend routes such as `/for-you`, `/history`, and `/article/N75626`. This allows refreshes and direct article links to work. Unknown JSON files should return HTTP 404 rather than the HTML fallback. Vite's development and preview servers provide frontend-route fallback locally.

Routes are `/readers`, `/for-you`, `/history`, and `/article/:newsId`. `/` returns to the selected reader's edition, or opens reader selection. The article reader supports direct links without a selected profile; continued-reading suggestions appear when a reader is selected and remain within that reader's candidate set.

See [verification notes and desktop/mobile screenshots](docs/QA.md) for the checked behavior and validation boundaries.
