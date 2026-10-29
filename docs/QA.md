# V1 verification

Verified locally on branch `UI`, 29 October 2026.

## Automated checks

- `npm run check:data`: 57 readers, 1,669 articles, 1,427 historical interactions, and 1,252 candidates. All joins, positional lists, and timestamps validate.
- `npm test`: 8 Python tests and 11 TypeScript tests pass.
- The ranking test covers every reader's complete impression, including both candidate labels and stable original order.
- The paragraph regression test covers every article in both native and fallback sentence-segmentation modes, preserving normalized source wording, punctuation, and order.
- `npm run build`: TypeScript and the Vite production build pass. Full article bodies remain separate static JSON assets.
- Source TSV files have no Git diff.

## Browser verification

Tested the development app and the final production build with the in-app browser.

| Check | Observed result |
| --- | --- |
| Desktop, 1440 × 1050 | Editorial layout with lead story, secondary stories, and history sidebar. |
| Tablet, 768 × 1024 | Adapted two-column layout; no horizontal overflow. |
| Mobile, 390 × 844 | Single-column layout, article reading, collapsible navigation, and no horizontal overflow. |
| Reader selection | Featured reader, ID search, empty-search recovery, and pagination through all 57 profiles work. |
| Reader switching | U216199 displays 23 candidates; switching to U194652 displays its 20 candidates and different lead story. |
| Persistence | Refresh retains the selected reader and edition. |
| Reading history | U216199 displays 34 historical entries, latest calendar date first, with aligned recorded durations. |
| Article navigation | Opens at the top; history-origin articles offer the history return action. |
| Browser back | Restores the scrolled feed position, including after the feed's story jump link. |
| Deep links | Article route refresh works in the production preview. |
| Reading progress | Increases with article scroll; a short article that already fits displays completed progress. |
| Short text | N72642 renders its actual seven-word body without filler. |
| Missing article | A nonexistent News ID displays the designed not-found state. |
| Keyboard navigation | Mobile menu opens with Enter; its history link navigates with Enter. |
| Production console | No application errors or warnings observed during final production checks. |

## Screenshots

- [Desktop edition](screenshots/feed-desktop.jpg)
- [Tablet edition](screenshots/feed-tablet.jpg)
- [Mobile edition](screenshots/feed-mobile.jpg)
- [Desktop reader selection](screenshots/readers-desktop.jpg)
- [Desktop article reader](screenshots/article-desktop.jpg)
- [Mobile article reader](screenshots/article-mobile.jpg)
- [Mobile reading history](screenshots/history-mobile.jpg)

## Boundaries

V1 uses the deterministic label-based ranking provider. A trained model is not connected. Validation covers the supplied curated subset; it does not establish recommendation-model quality. Static production hosting needs the frontend route fallback documented in the README. Web fonts use external font hosting with local fallbacks.
