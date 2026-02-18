"""Brave Search API client."""

from __future__ import annotations

import logging

from hf_bot.clients.base import BaseHTTPClient

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Triggers that indicate a user query needs a web search
# (tuple — iterated for substring matching, not set lookup)
_TRIGGERS: tuple[str, ...] = (
    "новост", "последн", "свеж", "актуальн", "недавн", "сегодня", "вчера",
    "news", "latest", "recent", "today",
    "вышл", "релиз", "выпустил", "анонс", "release", "announced",
    "лучш", "топ ", "рейтинг", "benchmark", "leaderboard",
    "сейчас", "на данный момент", "в 2024", "в 2025", "в 2026",
)

_AI_KEYWORDS: tuple[str, ...] = (
    "llm", "model", "ai", "gpt", "qwen", "deepseek", "mistral",
    "llama", "модел", "нейросет", "искусственн",
)


class SearchClient(BaseHTTPClient):
    """Brave Search API async client."""

    def __init__(self, *, api_key: str, timeout_seconds: int = 10) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._api_key = api_key

    @property
    def available(self) -> bool:
        """Whether the API key is configured."""
        return bool(self._api_key)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        country: str = "RU",
    ) -> list[dict[str, str]]:
        """Return web search results as a list of title/body/href dicts."""
        if not self._api_key:
            logger.warning("BRAVE_SEARCH_API_KEY not set — web search unavailable")
            return []

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": min(max_results, 20),
            "country": country,
            "search_lang": "ru",
            "ui_lang": "ru-RU",
        }

        try:
            async with self.session.get(
                _SEARCH_URL, headers=headers, params=params,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "body": r.get("description", ""),
                    "href": r.get("url", ""),
                }
                for r in data.get("web", {}).get("results", [])[:max_results]
            ]
        except Exception:
            logger.exception("Brave Search request failed")
            return []

    # ----- static helpers -------------------------------------------------

    @staticmethod
    def needs_search(text: str) -> bool:
        """Return True when the message likely needs a web search."""
        lower = text.lower()
        return any(t in lower for t in _TRIGGERS)

    @staticmethod
    def build_query(user_text: str) -> str:
        """Add AI context to the query when no AI keywords are present."""
        lower = user_text.lower()
        if any(kw in lower for kw in _AI_KEYWORDS):
            return user_text
        return f"AI LLM {user_text}"

    @staticmethod
    def format_results(results: list[dict[str, str]]) -> str:
        """Format search results as plain text for LLM context."""
        if not results:
            return ""
        lines = ["[Результаты поиска в интернете:]"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('title', '')}")
            body = r.get("body", "")
            if body:
                lines.append(f"   {body[:300]}{'...' if len(body) > 300 else ''}")
            href = r.get("href", "")
            if href:
                lines.append(f"   URL: {href}")
        return "\n".join(lines)
