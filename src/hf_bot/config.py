"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Missing or invalid configuration."""


def _int_env(name: str, default: str) -> int:
    """Read an environment variable as int and validate the value."""
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, validated application settings."""

    # --- Required --------------------------------------------------------
    bot_token: str
    chat_id: str

    # --- HuggingFace monitoring ------------------------------------------
    poll_seconds: int = 60
    hf_timeout_seconds: int = 10
    state_path: str = "data/state.json"

    # --- LLM (OpenRouter) ------------------------------------------------
    openrouter_api_key: str = ""
    llm_model: str = "google/gemini-2.5-flash-lite"
    llm_timeout_seconds: int = 90

    # --- Web search (Brave) ----------------------------------------------
    brave_search_api_key: str = ""

    # --- Logging ---------------------------------------------------------
    log_level: str = "INFO"

    # --- Monitored organisations -----------------------------------------
    monitored_orgs: tuple[str, ...] = (
        "moonshotai",
        "Qwen",
        "deepseek-ai",
        "zai-org",
        "mistralai",
        "ai-sage",
        "yandex",
        "t-tech",
        "google",
        "meta-llama",
        "tencent",
        "nvidia",
        "xai-org",
        "openai",
        "Anthropic",
        "MiniMaxAI",
        "inclusionAI",
    )

    # -----------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables with early validation."""
        bot_token = os.getenv("BOT_TOKEN", "")
        chat_id = os.getenv("CHAT_ID", "")

        if not bot_token or not chat_id:
            raise ConfigError("BOT_TOKEN and CHAT_ID environment variables are required")

        return cls(
            bot_token=bot_token,
            chat_id=chat_id,
            poll_seconds=_int_env("POLL_SECONDS", "60"),
            hf_timeout_seconds=_int_env("HF_TIMEOUT_SECONDS", "10"),
            state_path=os.getenv("STATE_PATH", "data/state.json"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite"),
            llm_timeout_seconds=_int_env("LLM_TIMEOUT_SECONDS", "90"),
            brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
