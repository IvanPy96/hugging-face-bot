"""Periodic monitoring of HuggingFace organisations for new models."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from telegram import Bot
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hf_bot.clients.huggingface import HuggingFaceClient
from hf_bot.clients.llm import LLMClient
from hf_bot.formatter import format_deploy_info, format_new_model_notification, sanitize_html
from hf_bot.models import DeployInfo, is_derivative_model
from hf_bot.state import save_state

logger = logging.getLogger(__name__)


async def poll_once(
    bot: Bot,
    chat_id: str,
    orgs: list[str],
    state: dict[str, Any],
    state_path: str,
    *,
    hf: HuggingFaceClient,
    llm: LLMClient,
) -> None:
    """Execute a single monitoring cycle across all organisations."""
    # Fetch all orgs in parallel (HF client paginates automatically)
    async def _fetch(org: str) -> tuple[str, list[dict[str, str]]]:
        try:
            return org, await hf.fetch_models_for_org(org)
        except Exception:
            logger.exception("Monitoring error for org=%s", org)
            return org, []

    results = await asyncio.gather(*(_fetch(o) for o in orgs))
    changed = False

    for org, models in results:
        if not models:
            continue

        # Deduplicate
        seen: set[str] = set()
        current_ids: list[str] = []
        for m in models:
            mid = m["id"]
            if mid not in seen:
                seen.add(mid)
                current_ids.append(mid)

        org_state = state["orgs"].get(org)
        if not org_state:
            # First sync — no notifications
            state["orgs"][org] = {"models": current_ids}
            changed = True
            logger.info("Baseline sync for %s (%d models)", org, len(current_ids))
            continue

        prev_set = set(org_state.get("models", []))
        new_ids = [mid for mid in current_ids if mid not in prev_set]

        if new_ids:
            main_models = [mid for mid in new_ids if not is_derivative_model(mid)]

            for mid in reversed(main_models):
                await _notify_new_model(bot, chat_id, org, mid, hf=hf, llm=llm)

            if main_models:
                logger.info(
                    "Sent %d notifications for %s (skipped %d derivatives)",
                    len(main_models), org, len(new_ids) - len(main_models),
                )
            elif new_ids:
                logger.info("Skipped %d derivative models for %s", len(new_ids), org)

            state["orgs"][org] = {"models": current_ids}
            changed = True
        elif current_ids != org_state.get("models", []):
            state["orgs"][org] = {"models": current_ids}
            changed = True

    if changed:
        save_state(state, state_path)


async def _notify_new_model(
    bot: Bot,
    chat_id: str,
    org: str,
    model_id: str,
    *,
    hf: HuggingFaceClient,
    llm: LLMClient,
) -> None:
    """Send new-model notification with summary and deploy info."""
    notification = format_new_model_notification(org, model_id)

    # Parallel: send notification + fetch README + model info
    send_task = bot.send_message(
        chat_id=chat_id, text=notification,
        parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )
    readme_task = hf.get_model_readme(model_id)
    info_task = hf.get_model_info(model_id)

    try:
        _, readme, model_info = await asyncio.gather(send_task, readme_task, info_task)
    except Exception:
        logger.exception("Failed to send notification for %s", model_id)
        return

    # Auto-summary from README
    try:
        if readme:
            summary = await llm.generate_model_summary(model_id, readme)
            if summary:
                summary = sanitize_html(summary)
                await _safe_send(bot, chat_id, summary)
        else:
            await bot.send_message(
                chat_id=chat_id,
                text="🤷 <i>README пока нет. Ждём, когда авторы расскажут, что это за зверь...</i>",
                parse_mode=ParseMode.HTML,
            )
    except Exception:
        logger.exception("Summary generation error for %s", model_id)

    # Deploy estimation
    try:
        if model_info:
            deploy = DeployInfo.from_model(model_info)
            if deploy:
                await bot.send_message(
                    chat_id=chat_id,
                    text=format_deploy_info(deploy, model_id),
                    parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                )
    except Exception:
        logger.exception("Deploy calculation error for %s", model_id)


async def _safe_send(bot: Bot, chat_id: str, text: str) -> None:
    """Send a message with HTML; fall back to plain text on parse error."""
    try:
        await bot.send_message(
            chat_id=chat_id, text=text,
            parse_mode=ParseMode.HTML, disable_web_page_preview=True,
        )
    except Exception:
        plain = re.sub(r"<[^>]+>", "", text)
        await bot.send_message(chat_id=chat_id, text=plain, disable_web_page_preview=True)


# ═══════════════════════════════════════════════════════════════════════════
# Job callback (called by Application's job queue)
# ═══════════════════════════════════════════════════════════════════════════

async def monitoring_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Repeating-job callback invoked by the application scheduler."""
    data = context.job.data
    bd = context.application.bot_data
    await poll_once(
        bot=context.bot,
        chat_id=data["chat_id"],
        orgs=data["orgs"],
        state=data["state"],
        state_path=data["state_path"],
        hf=bd["hf_client"],
        llm=bd["llm_client"],
    )
