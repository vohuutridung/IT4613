import { ArrowDown, ArrowUpRight, ChevronDown, Menu, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useReader } from "../app/readerState";

export function Wordmark({ small = false }: { small?: boolean }) {
  return (
    <span className={`wordmark ${small ? "wordmark-small" : ""}`}>
      The Daily<span>.</span>
    </span>
  );
}

export function SiteHeader() {
  const { selectedId } = useReader();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  return (
    <header className="site-header">
      <div className="header-inner">
        <Link
          className="brand-link"
          to={selectedId ? "/for-you" : "/readers"}
          aria-label="The Daily home"
        >
          <Wordmark small />
        </Link>
        <nav className="desktop-nav" aria-label="Main navigation">
          <NavLink to="/for-you">For You</NavLink>
          <NavLink to="/history">Reading History</NavLink>
        </nav>
        <Link className="reader-switch" to="/readers">
          <span className="reader-monogram">
            {selectedId ? selectedId.slice(-2) : "D."}
          </span>
          <span>{selectedId || "Choose a reader"}</span>
          <ChevronDown size={14} />
        </Link>
        <button
          className="icon-button mobile-menu-button"
          aria-expanded={menuOpen}
          aria-controls="mobile-navigation"
          aria-label={menuOpen ? "Close navigation" : "Open navigation"}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X size={21} /> : <Menu size={21} />}
        </button>
      </div>
      {menuOpen && (
        <nav
          id="mobile-navigation"
          className="mobile-nav"
          aria-label="Mobile navigation"
        >
          <NavLink to="/for-you">
            For You
            <ArrowUpRight size={18} />
          </NavLink>
          <NavLink to="/history">
            Reading History
            <ArrowUpRight size={18} />
          </NavLink>
          <NavLink to="/readers">
            Switch reader
            <ArrowUpRight size={18} />
          </NavLink>
        </nav>
      )}
    </header>
  );
}

export function SiteFooter() {
  const { selectedId } = useReader();
  return (
    <footer className="site-footer page-width">
      <div>
        <Link
          to={selectedId ? "/for-you" : "/readers"}
          aria-label="The Daily home"
        >
          <Wordmark small />
        </Link>
      </div>
      <button
        className="text-link footer-top"
        onClick={() =>
          window.scrollTo({
            top: 0,
            behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
              ? "instant"
              : "smooth",
          })
        }
      >
        Back to top <ArrowDown className="rotate-180" size={14} />
      </button>
    </footer>
  );
}
