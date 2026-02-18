"""Lightweight user-intent analysis (regex-based, no LLM call required)."""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Known model name patterns
# ---------------------------------------------------------------------------

_FULL_ID_RE = re.compile(r"([A-Za-z0-9_-]+/[A-Za-z0-9_.-]+)")

_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(Qwen[23]?(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(DeepSeek(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(deepseek(?:-[a-z0-9.-]+)?)\b",
        r"\b(Mistral(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(Mixtral(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(Llama(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(llama(?:-[a-z0-9.-]+)?)\b",
        r"\b(GigaChat[0-9]*(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(GLM(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(Gemma(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(Claude(?:-[A-Za-z0-9.-]+)?)\b",
        r"\b(GPT-?[0-9]+(?:-[A-Za-z0-9.-]+)?)\b",
    )
]

# ---------------------------------------------------------------------------
# Trigger phrases (tuples — iterated for substring matching, not set lookup)
# ---------------------------------------------------------------------------

_COMPARE: tuple[str, ...] = (
    "сравни", "сравнить", "сравнение", "versus", "vs", "против",
    "лучше", "хуже", "разница", "отличия", "или", "выбрать между",
    "что лучше", "какая лучше", "какой лучше",
)

_INFO: tuple[str, ...] = (
    "что за", "что такое", "расскажи о", "расскажи про", "инфа о", "инфо о",
    "информация о", "характеристики", "бенчмарки", "benchmark",
    "сколько параметров", "какой размер", "на чём обучена", "на чем обучена",
)

_NEWS: tuple[str, ...] = (
    "новости", "новость", "news", "что нового", "что происходит",
    "последние события", "свежее", "актуальное", "тренды",
    "что слышно", "обсудим", "расскажи про события",
    "что там в мире", "что случилось", "последние новости",
    "новинки", "обновления в мире", "что творится",
)

# ---------------------------------------------------------------------------
# Russian → canonical name aliases
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    "квен": "Qwen",
    "кьювен": "Qwen",
    "дипсик": "DeepSeek",
    "дипсек": "DeepSeek",
    "мистраль": "Mistral",
    "лама": "Llama",
    "гигачат": "GigaChat",
    "джемма": "Gemma",
    "клод": "Claude",
    "гпт": "GPT",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(text: str) -> dict[str, Any]:
    """Determine user intent, extract model IDs and search queries.

    Returns a dict::

        {
            "intent": "chat" | "info" | "compare" | "news",
            "models": ["org/name", ...],          # exact HF model IDs
            "normalized_queries": ["Qwen", ...],  # search keywords
        }
    """
    lower = text.lower()

    # Exact model IDs (org/model)
    models: list[str] = list(dict.fromkeys(_FULL_ID_RE.findall(text)))

    # Set of bare model names from full IDs for deduplication
    model_names: set[str] = {mid.split("/", 1)[-1] for mid in models}

    # Normalise Russian aliases → canonical names
    queries: list[str] = []
    for alias, canonical in _ALIASES.items():
        if alias in lower:
            queries.append(canonical)

    # Match known model-name patterns
    for pat in _MODEL_PATTERNS:
        for m in pat.findall(text):
            if m and m not in queries and m not in model_names:
                queries.append(m)

    queries = list(dict.fromkeys(queries))

    # --- Intent classification ---
    intent = "chat"
    n_refs = len(models) + len(queries)

    if any(t in lower for t in _COMPARE):
        intent = "compare" if n_refs >= 2 else ("info" if n_refs else "chat")
    elif any(t in lower for t in _INFO):
        if n_refs:
            intent = "info"
    elif any(t in lower for t in _NEWS):
        intent = "info" if n_refs else "news"
    elif n_refs:
        intent = "info"

    return {"intent": intent, "models": models, "normalized_queries": queries}
