"""Application builder, lifecycle management, and entry point."""

from __future__ import annotations

import asyncio
import logging
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from hf_bot.clients.huggingface import HuggingFaceClient
from hf_bot.clients.llm import LLMClient
from hf_bot.clients.search import SearchClient
from hf_bot.clients.web_reader import WebReaderClient
from hf_bot.config import ConfigError, Settings
from hf_bot.handlers import (
    cmd_agi,
    cmd_battle,
    cmd_deploy,
    cmd_help,
    cmd_hero,
    cmd_info,
    cmd_orgs,
    cmd_random,
    cmd_start,
    cmd_stats,
    handle_message,
    track_user,
)
from hf_bot.monitoring import monitoring_job
from hf_bot.state import load_state, save_state

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle hooks
# ═══════════════════════════════════════════════════════════════════════════

async def _post_init(application: Application) -> None:
    """Create shared HTTP clients after the event loop is running."""
    settings: Settings = application.bot_data["settings"]

    hf = HuggingFaceClient(timeout_seconds=settings.hf_timeout_seconds)
    llm = LLMClient(
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    search = SearchClient(
        api_key=settings.brave_search_api_key,
    )
    web_reader = WebReaderClient(timeout_seconds=30)

    application.bot_data["hf_client"] = hf
    application.bot_data["llm_client"] = llm
    application.bot_data["search_client"] = search
    application.bot_data["web_reader_client"] = web_reader

    logger.info("HTTP clients initialised (HF, LLM, Search, WebReader)")

    # Pre-fill question bank in background if needed
    bank: list[dict[str, str]] = application.bot_data.get("question_bank", [])
    if len(bank) < 10 and llm.available:
        asyncio.create_task(_prefill_question_bank(llm, bank, application.bot_data))


async def _prefill_question_bank(
    llm: LLMClient,
    bank: list[dict[str, str]],
    bot_data: dict,
) -> None:
    """Background task: fill the question bank to 10 at startup."""
    needed = 10 - len(bank)
    if needed <= 0:
        return
    try:
        questions = await llm.generate_question_bank(count=needed)
        if questions:
            bank.extend(questions)
            settings = bot_data.get("settings")
            state = bot_data.get("state", {})
            if settings:
                save_state(state, settings.state_path)
            logger.info("Question bank pre-filled to %d at startup", len(bank))
    except Exception:
        logger.exception("Failed to pre-fill question bank at startup")


async def _post_shutdown(application: Application) -> None:
    """Gracefully close all HTTP sessions."""
    for key in ("hf_client", "llm_client", "search_client", "web_reader_client"):
        client = application.bot_data.get(key)
        if client:
            await client.close()
    logger.info("HTTP clients closed")


# ═══════════════════════════════════════════════════════════════════════════
# Application factory
# ═══════════════════════════════════════════════════════════════════════════

def build_application(settings: Settings) -> Application:
    """Build and return a fully configured Telegram application."""
    state = load_state(settings.state_path)
    state.setdefault("orgs", {})

    app = (
        Application.builder()
        .token(settings.bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    # Store shared data
    app.bot_data["settings"] = settings
    app.bot_data["orgs"] = list(settings.monitored_orgs)
    app.bot_data["state"] = state
    # Load persisted chat users for /hero command
    app.bot_data["chat_users"] = state.get("chat_users", {})
    # Load persisted question bank for /battle command
    app.bot_data["question_bank"] = state.get("question_bank", [])

    # --- Register handlers -----------------------------------------------

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("info", cmd_info))
    app.add_handler(CommandHandler("deploy", cmd_deploy))
    app.add_handler(CommandHandler("orgs", cmd_orgs))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("random", cmd_random))
    app.add_handler(CommandHandler("hero", cmd_hero))
    app.add_handler(CommandHandler("battle", cmd_battle))
    app.add_handler(CommandHandler("agi", cmd_agi))

    # Track chat participants (separate handler group so it doesn't block commands)
    app.add_handler(MessageHandler(filters.ALL, track_user), group=-1)

    # AI assistant (non-command text messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # --- Monitoring job ---------------------------------------------------

    # Note: hf_client / llm_client are read from bot_data inside the job
    # callback (they are initialised by _post_init, which runs before any job).
    app.job_queue.run_repeating(
        monitoring_job,
        interval=settings.poll_seconds,
        first=5,
        data={
            "chat_id": settings.chat_id,
            "orgs": list(settings.monitored_orgs),
            "state": state,
            "state_path": settings.state_path,
        },
        name="monitoring",
    )

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Load settings, build application, and start polling."""
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        logging.basicConfig()
        logging.error("Configuration error: %s", exc)
        sys.exit(1)

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Monitored orgs: %s", ", ".join(settings.monitored_orgs))
    logger.info("Poll interval: %ds", settings.poll_seconds)
    logger.info("Starting bot…")

    app = build_application(settings)
    app.run_polling(allowed_updates=Update.ALL_TYPES)
