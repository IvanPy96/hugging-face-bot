"""Web content reader for URLs and arxiv papers.

Regular URLs are processed with trafilatura (article extraction).
Arxiv links use the arxiv library for structured paper metadata.
"""

from __future__ import annotations

import asyncio
import logging
import re

from hf_bot.clients.base import BaseHTTPClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d+\.\d+(?:v\d+)?)")


def extract_urls(text: str) -> list[str]:
    """Return all HTTP(S) URLs found in text."""
    return _URL_RE.findall(text)


def is_arxiv_url(url: str) -> bool:
    """Return True when the URL points to arxiv.org."""
    return "arxiv" in url.lower()


def _extract_arxiv_id(url: str) -> str | None:
    """Extract the paper ID from an arxiv URL."""
    match = _ARXIV_ID_RE.search(url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Synchronous helpers (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _trafilatura_extract(raw_html: str, url: str) -> str | None:
    """Extract main article text from raw HTML using trafilatura."""
    try:
        import trafilatura  # noqa: PLC0415

        return trafilatura.extract(
            raw_html,
            url=url,
            include_comments=False,
            include_tables=True,
        )
    except Exception:
        logger.exception("trafilatura extraction failed for %s", url)
        return None


def _fetch_arxiv_sync(paper_id: str) -> str | None:
    """Fetch arxiv paper metadata with the arxiv library."""
    try:
        import arxiv  # noqa: PLC0415

        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        if not results:
            return None

        paper = results[0]
        lines: list[str] = [
            f"=== Arxiv Paper: {paper_id} ===",
            f"Title: {paper.title}",
            f"Authors: {', '.join(a.name for a in paper.authors)}",
            f"Published: {paper.published.strftime('%Y-%m-%d')}",
            f"URL: {paper.entry_id}",
        ]
        if paper.categories:
            lines.append(f"Categories: {', '.join(paper.categories)}")
        if paper.comment:
            lines.append(f"Comment: {paper.comment}")
        if paper.journal_ref:
            lines.append(f"Journal ref: {paper.journal_ref}")
        if paper.doi:
            lines.append(f"DOI: {paper.doi}")
        lines += ["", "--- Abstract ---", paper.summary]
        return "\n".join(lines)
    except Exception:
        logger.exception("arxiv fetch failed for %s", paper_id)
        return None


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class WebReaderClient(BaseHTTPClient):
    """Async client for reading web pages and arxiv papers."""

    async def fetch_url_text(self, url: str, *, max_length: int = 8000) -> str | None:
        """Fetch a URL and return extracted article text."""
        try:
            async with self.session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    logger.warning("URL returned %d: %s", resp.status, url)
                    return None
                raw_html = await resp.text()
        except Exception:
            logger.exception("Error downloading URL: %s", url)
            return None

        text = await asyncio.to_thread(_trafilatura_extract, raw_html, url)
        if not text:
            return None
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[...текст обрезан...]"
        return text

    async def fetch_arxiv_paper(self, url: str) -> str | None:
        """Fetch arxiv paper metadata and abstract from a URL."""
        paper_id = _extract_arxiv_id(url)
        if not paper_id:
            logger.warning("Could not extract arxiv ID from: %s", url)
            return None
        return await asyncio.to_thread(_fetch_arxiv_sync, paper_id)
