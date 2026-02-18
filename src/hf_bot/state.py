"""Persistent state management (JSON file with atomic writes)."""

from __future__ import annotations

import json
import os
from typing import Any

from hf_bot.models import DERIVATIVE_SUFFIXES


def _empty_state() -> dict[str, Any]:
    """Return a fresh default state dict (avoids shared-mutable-default pitfalls)."""
    return {"orgs": {}, "chat_users": {}, "question_bank": []}


def load_state(path: str) -> dict[str, Any]:
    """Load state from path or return the default state."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_state()

    if not isinstance(data, dict):
        return _empty_state()
    data.setdefault("orgs", {})
    data.setdefault("chat_users", {})
    data.setdefault("question_bank", [])
    return data


def save_state(state: dict[str, Any], path: str) -> None:
    """Persist state atomically using a temp file and rename."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def get_example_models(
    state: dict[str, Any],
    orgs: list[str] | None = None,
) -> dict[str, str]:
    """Return {org: latest_main_model_id} for dynamic prompt examples."""
    orgs = orgs or ["Qwen", "deepseek-ai", "mistralai", "meta-llama"]
    examples: dict[str, str] = {}
    for org in orgs:
        models: list[str] = state.get("orgs", {}).get(org, {}).get("models", [])
        for m in models:
            if not m.split("/", 1)[-1].lower().endswith(DERIVATIVE_SUFFIXES):
                examples[org] = m
                break
    return examples
