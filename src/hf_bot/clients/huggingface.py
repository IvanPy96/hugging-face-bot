"""Hugging Face API client with proper pagination."""

from __future__ import annotations

import logging
import random
import re
from typing import Any

import aiohttp

from hf_bot.clients.base import BaseHTTPClient
from hf_bot.models import ModelInfo

logger = logging.getLogger(__name__)

_HF_API = "https://huggingface.co/api/models"

# HF API returns at most ~1000 items per request regardless of the `limit` value.
# We must paginate via the ``Link: <url>; rel="next"`` header to get all results.
_PAGE_SIZE = 1000
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')

# ---------------------------------------------------------------------------
# Image extraction from README
# ---------------------------------------------------------------------------

_IMG_PATTERNS = (
    re.compile(r'!\[[^\]]*\]\(([^)]+)\)'),
    re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE),
)
_RELEVANT_IMG_KW = frozenset((
    "benchmark", "performance", "comparison", "chart", "graph",
    "result", "eval", "score", "accuracy", "metrics", "leaderboard", "table",
))

# GigaChat model IDs for /random easter egg
_GIGACHAT_IDS = (
    "ai-sage/GigaChat-20B-A3B-instruct",
    "ai-sage/GigaChat-20B-A3B-instruct-v1.5",
    "ai-sage/GigaChat3-10B-A1.8B",
    "ai-sage/GigaChat3-702B-A36B-preview",
)


class HuggingFaceClient(BaseHTTPClient):
    """Async client for the Hugging Face model API with automatic pagination."""

    # ----- internal pagination helper ------------------------------------

    async def _paginate(
        self,
        params: dict[str, Any],
        *,
        max_pages: int = 50,
        per_request_timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all pages from the HF models API by following the Link header.

        The HF API silently caps a single response at ~1000 items.  This method
        follows the rel="next" link until all results are collected.
        """
        all_items: list[dict[str, Any]] = []
        url: str = _HF_API
        current_params: dict[str, Any] | None = params
        extra: dict[str, Any] = {}
        if per_request_timeout:
            extra["timeout"] = aiohttp.ClientTimeout(total=per_request_timeout)

        for _ in range(max_pages):
            async with self.session.get(url, params=current_params, **extra) as resp:
                resp.raise_for_status()
                batch: list[dict[str, Any]] = await resp.json()
                link_header = resp.headers.get("Link", "")

            if not batch:
                break

            all_items.extend(batch)

            # If fewer than PAGE_SIZE → this was the last page
            if len(batch) < _PAGE_SIZE:
                break

            # Follow the next-page cursor from the Link header
            match = _LINK_NEXT_RE.search(link_header)
            if not match:
                break

            url = match.group(1)
            current_params = None  # next URL already contains all query params

        return all_items

    # ----- Organisation monitoring ----------------------------------------

    async def fetch_models_for_org(self, org: str) -> list[dict[str, str]]:
        """Fetch all models for an organization sorted by lastModified.

        Paginates automatically — no artificial limit.
        """
        params: dict[str, Any] = {
            "author": org,
            "sort": "lastModified",
            "direction": -1,
            "limit": _PAGE_SIZE,
        }
        raw = await self._paginate(params)
        return [
            {
                "id": item.get("modelId") or item.get("id") or item.get("_id", ""),
                "last_modified": item.get("lastModified", ""),
            }
            for item in raw
            if item.get("modelId") or item.get("id") or item.get("_id")
        ]

    async def fetch_org_model_count(self, org: str) -> int:
        """Return the real total model count for an organization."""
        params: dict[str, Any] = {"author": org, "limit": _PAGE_SIZE}
        try:
            raw = await self._paginate(params, per_request_timeout=30)
            return len(raw)
        except Exception:
            return 0

    # ----- Single-model operations ----------------------------------------

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Return model information, or None when the model is not found."""
        async with self.session.get(f"{_HF_API}/{model_id}") as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return ModelInfo.from_api(await resp.json())

    async def search_models(self, query: str, limit: int = 5) -> list[ModelInfo]:
        """Search models by text query, sorted by downloads."""
        params = {
            "search": query, "sort": "downloads", "direction": -1,
            "limit": limit, "full": "true",
        }
        try:
            async with self.session.get(_HF_API, params=params) as resp:
                resp.raise_for_status()
                return [
                    ModelInfo.from_api(item)
                    for item in await resp.json()
                    if item.get("modelId") or item.get("id")
                ]
        except Exception:
            return []

    # ----- README / model card -------------------------------------------

    async def get_model_readme(
        self, model_id: str, *, max_length: int = 6000,
    ) -> str | None:
        """Fetch the raw README (model card) text."""
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        try:
            async with self.session.get(url) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                text = await resp.text()
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[...README обрезан...]"
            return text
        except Exception:
            return None

    async def get_readme_with_images(
        self, model_id: str, *, max_length: int = 6000, max_images: int = 3,
    ) -> tuple[str | None, list[str]]:
        """Return a model README and extracted image URLs."""
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        try:
            async with self.session.get(url) as resp:
                if resp.status == 404:
                    return None, []
                resp.raise_for_status()
                full = await resp.text()
            images = _extract_images(full, model_id, max_images)
            text = full[:max_length] + "\n\n[...README обрезан...]" if len(full) > max_length else full
            return text, images
        except Exception:
            return None, []

    # ----- Random models --------------------------------------------------

    async def get_random_model(self) -> ModelInfo | None:
        """Return a random model from the top-downloaded list."""
        params = {
            "sort": "downloads", "direction": -1, "limit": 20,
            "skip": random.randint(0, 500), "full": "true",
        }
        async with self.session.get(_HF_API, params=params) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return ModelInfo.from_api(random.choice(payload)) if payload else None

    async def get_random_gigachat(self) -> ModelInfo | None:
        """Return a random GigaChat model for the /random easter egg."""
        ids = list(_GIGACHAT_IDS)
        random.shuffle(ids)
        for mid in ids:
            try:
                model = await self.get_model_info(mid)
                if model:
                    return model
            except Exception:
                continue
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_images(readme: str, model_id: str, max_images: int) -> list[str]:
    """Extract and prioritise image URLs from a README."""
    raw: list[str] = []
    for pat in _IMG_PATTERNS:
        raw.extend(pat.findall(readme))

    normalized: list[str] = []
    for url in raw:
        url = url.strip()
        if any(s in url.lower() for s in ("badge", "shield", "icon", ".svg", "logo")):
            continue
        if not url.startswith("http"):
            url = f"https://huggingface.co/{model_id}/resolve/main/{url.lstrip('./')}"
        normalized.append(url)

    relevant, others = [], []
    for url in normalized:
        bucket = relevant if any(kw in url.lower() for kw in _RELEVANT_IMG_KW) else others
        bucket.append(url)

    seen: set[str] = set()
    unique: list[str] = []
    for url in relevant + others:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique[:max_images]
