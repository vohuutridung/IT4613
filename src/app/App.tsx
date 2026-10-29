import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
} from "react-router-dom";
import { ReaderProvider } from "./ReaderContext";
import { useReader } from "./readerState";
import { ScrollRestoration } from "./ScrollRestoration";
import { SiteFooter, SiteHeader } from "../components/layout";
import { ErrorState, LoadingState, NotFound } from "../components/states";
import { ReadersPage } from "../features/readers/ReadersPage";
import { FeedPage } from "../features/feed/FeedPage";
import { HistoryPage } from "../features/history/HistoryPage";
import { ArticlePage } from "../features/articles/ArticlePage";

function HomeRoute() {
  const { selectedId, loading } = useReader();
  return loading ? (
    <LoadingState />
  ) : (
    <Navigate to={selectedId ? "/for-you" : "/readers"} replace />
  );
}

function RequiresReader() {
  const { selectedId, loading, error, retry } = useReader();
  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} retry={retry} />;
  return selectedId ? <Outlet /> : <Navigate to="/readers" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <ReaderProvider>
        <ScrollRestoration />
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <SiteHeader />
        <Routes>
          <Route path="/" element={<HomeRoute />} />
          <Route path="/readers" element={<ReadersPage />} />
          <Route element={<RequiresReader />}>
            <Route path="/for-you" element={<FeedPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Route>
          <Route path="/article/:newsId" element={<ArticlePage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
        <SiteFooter />
      </ReaderProvider>
    </BrowserRouter>
  );
}
