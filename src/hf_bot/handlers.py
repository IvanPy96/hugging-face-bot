"""Telegram bot command and message handlers."""

from __future__ import annotations

import asyncio
import html
import logging
import random
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from telegram import Message, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hf_bot.clients.huggingface import HuggingFaceClient
from hf_bot.clients.llm import LLMClient
from hf_bot.clients.search import SearchClient
from hf_bot.clients.web_reader import WebReaderClient, extract_urls, is_arxiv_url
from hf_bot.formatter import (
    THINKING_PHRASES,
    format_agi_check,
    format_battle_already_active,
    format_battle_no_llm,
    format_battle_question,
    format_battle_reminder,
    format_battle_timeout,
    format_deploy_info,
    format_deploy_usage,
    format_error,
    format_help_message,
    format_hero_message,
    format_info_usage,
    format_model_card,
    format_model_not_found,
    format_orgs_list,
    format_random_model,
    format_start_message,
    format_stats,
    sanitize_html,
)
from hf_bot.intent import analyze as analyze_intent
from hf_bot.models import DeployInfo
from hf_bot.state import get_example_models, save_state

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper to access shared clients from bot_data
# ═══════════════════════════════════════════════════════════════════════════

def _hf(ctx: ContextTypes.DEFAULT_TYPE) -> HuggingFaceClient:
    """Return the shared Hugging Face client from bot_data."""
    return ctx.bot_data["hf_client"]


def _llm(ctx: ContextTypes.DEFAULT_TYPE) -> LLMClient:
    """Return the shared LLM client from bot_data."""
    return ctx.bot_data["llm_client"]


def _search(ctx: ContextTypes.DEFAULT_TYPE) -> SearchClient:
    """Return the shared web-search client from bot_data."""
    return ctx.bot_data["search_client"]


def _web_reader(ctx: ContextTypes.DEFAULT_TYPE) -> WebReaderClient:
    """Return the shared web-reader client from bot_data."""
    return ctx.bot_data["web_reader_client"]


# ═══════════════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start command handler.

    Sends a welcome message with a brief description of the bot and available commands.
    """
    await update.message.reply_text(
        format_start_message(), parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help command handler.

    Sends a detailed help message describing the bot's features, monitoring logic,
    and available commands with examples.
    """
    await update.message.reply_text(
        format_help_message(), parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/info command handler.

    Fetches and displays information about a specific Hugging Face model,
    including downloads, likes, tags, and a link to the hub.
    """
    if not context.args:
        await update.message.reply_text(format_info_usage(), parse_mode=ParseMode.HTML)
        return

    model_id = context.args[0]
    try:
        model = await _hf(context).get_model_info(model_id)
    except Exception:
        logger.exception("Error fetching model info: %s", model_id)
        await update.message.reply_text(format_error(), parse_mode=ParseMode.HTML)
        return

    if not model:
        await update.message.reply_text(format_model_not_found(model_id), parse_mode=ParseMode.HTML)
        return

    await update.message.reply_text(
        format_model_card(model), parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_deploy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/deploy command handler.

    Calculates and displays the estimated GPU requirements (VRAM) for deploying
    a specific model, based on its parameter count and precision.
    """
    if not context.args:
        await update.message.reply_text(format_deploy_usage(), parse_mode=ParseMode.HTML)
        return

    model_id = context.args[0]
    try:
        model = await _hf(context).get_model_info(model_id)
    except Exception:
        logger.exception("Error fetching model for deploy: %s", model_id)
        await update.message.reply_text(format_error(), parse_mode=ParseMode.HTML)
        return

    if not model:
        await update.message.reply_text(format_model_not_found(model_id), parse_mode=ParseMode.HTML)
        return

    deploy = DeployInfo.from_model(model)
    if not deploy:
        await update.message.reply_text(
            f"🖥️ Не удалось определить размер модели <code>{html.escape(model_id)}</code>.\n\n"
            "<i>У модели нет данных safetensors с информацией о количестве параметров и точности весов.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.message.reply_text(
        format_deploy_info(deploy, model_id),
        parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_orgs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/orgs command handler.

    Displays the list of Hugging Face organisations currently being monitored
    for new model releases.
    """
    orgs: list[str] = context.bot_data.get("orgs", [])
    if not orgs:
        await update.message.reply_text("🤔 Список организаций пуст.")
        return
    await update.message.reply_text(
        format_orgs_list(orgs), parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats command handler.

    Fetches the current model counts for all monitored organisations from the
    Hugging Face API and displays a leaderboard with statistics.
    """
    orgs: list[str] = context.bot_data.get("orgs", [])
    if not orgs:
        await update.message.reply_text(
            "📊 Статистика пока пуста.\n\n<i>Нет отслеживаемых организаций.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    msg = await update.message.reply_text(
        "📊 <i>Загружаю актуальную статистику с Hugging Face...</i>",
        parse_mode=ParseMode.HTML,
    )

    hf = _hf(context)
    results = await asyncio.gather(
        *(hf.fetch_org_model_count(o) for o in orgs), return_exceptions=True,
    )
    org_stats: dict[str, int] = {}
    total = 0
    for org, result in zip(orgs, results):
        cnt = result if isinstance(result, int) else 0
        org_stats[org] = cnt
        total += cnt

    await msg.edit_text(
        format_stats(org_stats, total),
        parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_random(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/random command handler.

    Fetches and displays a random popular model from Hugging Face.
    Every 10th invocation forces a GigaChat model for comedic effect.
    """
    counter = context.bot_data.get("random_counter", 0) + 1
    context.bot_data["random_counter"] = counter
    forced_gc = counter % 10 == 0

    hf = _hf(context)
    try:
        if forced_gc:
            model = await hf.get_random_gigachat()
            if not model:
                model = await hf.get_random_model()
                forced_gc = False
        else:
            model = await hf.get_random_model()
    except Exception:
        logger.exception("Error fetching random model")
        await update.message.reply_text(format_error(), parse_mode=ParseMode.HTML)
        return

    if not model:
        await update.message.reply_text(
            "🎲 Не удалось найти случайную модель. Попробуйте ещё раз!",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.message.reply_text(
        format_random_model(model, forced_gigachat=forced_gc),
        parse_mode=ParseMode.HTML, disable_web_page_preview=True,
    )


async def cmd_agi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/agi command handler.

    Simulates a check for Artificial General Intelligence (AGI) presence,
    always resulting in a humorous failure message.
    """
    msg = await update.message.reply_text(
        "🤖 <b>Проверяю наличие AGI...</b>\n\n<code>░░░░░░░░░░░░░░░░░░░░</code> 0%",
        parse_mode=ParseMode.HTML,
    )
    await asyncio.sleep(1.5)
    await msg.edit_text(format_agi_check(), parse_mode=ParseMode.HTML)


async def cmd_hero(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/hero command handler.

    Selects a random user from the chat (who has previously interacted with the bot)
    and generates a personalized motivational message using the LLM.
    """
    chat = update.message.chat
    sender_id = str(update.message.from_user.id)  # String for JSON key comparison
    bot_id = str(context.bot.id)

    if chat.type == "private":
        user = update.message.from_user
        mention = f'<a href="tg://user?id={user.id}">{html.escape(user.first_name)}</a>'
    else:
        chat_id = str(chat.id)
        tracked = context.bot_data.get("chat_users", {}).get(chat_id, {})
        # Filter out bot and sender (keys are strings in JSON)
        candidates = {uid: info for uid, info in tracked.items() if uid != bot_id and uid != sender_id}

        # Always try to add administrators to expand the candidate pool
        try:
            for admin in await chat.get_administrators():
                admin_id = str(admin.user.id)
                if admin_id != bot_id and admin_id != sender_id and not admin.user.is_bot:
                    # Only add if not already tracked (avoid overwriting fresher data)
                    if admin_id not in candidates:
                        candidates[admin_id] = {
                            "first_name": admin.user.first_name or "Аноним",
                            "username": admin.user.username or "",
                        }
        except Exception:
            pass

        if not candidates:
            await update.message.reply_text(
                "🦸 Пока не знаю никого в этом чате! "
                "Пусть люди пообщаются немного, и я обязательно найду героя. 💬",
                parse_mode=ParseMode.HTML,
            )
            return

        # Use secrets.choice for cryptographically secure randomness
        candidate_ids = list(candidates.keys())
        hero_id = secrets.choice(candidate_ids)
        hero_info = candidates[hero_id]
        mention = f'<a href="tg://user?id={hero_id}">{html.escape(hero_info["first_name"])}</a>'

    thinking_msg = await update.message.reply_text(
        "🦸 <i>Ищу героя и подбираю слова...</i>", parse_mode=ParseMode.HTML,
    )

    hero_text = await _llm(context).generate_hero_message()
    if hero_text:
        hero_text = sanitize_html(hero_text)

    result = format_hero_message(mention, hero_text or "")
    await _safe_edit(thinking_msg, result, update)


# ═══════════════════════════════════════════════════════════════════════════
# /battle — quiz duel with GigaChat
# ═══════════════════════════════════════════════════════════════════════════


async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/battle command handler.

    Picks a pre-generated question from the question bank (serious, Gemini 3 Pro)
    or generates an absurd question on the fly (~20 % chance).  Sends it with a
    "Гигачат," prefix to trigger the GigaChat bot.

    Timeline: 60 s → reminder nudge → 60 s → battle auto-cancelled (2 min total).
    """
    chat_id = str(update.message.chat_id)
    battles = context.bot_data.setdefault("battles", {})

    if chat_id in battles:
        await update.message.reply_text(
            format_battle_already_active(), parse_mode=ParseMode.HTML,
        )
        return

    thinking_msg = await update.message.reply_text(
        "⚔️ <i>Готовлю каверзный вопрос...</i>", parse_mode=ParseMode.HTML,
    )

    # ~20% absurd, ~80% serious from the bank
    llm = _llm(context)
    if random.random() < 0.2:
        qdata = await llm.generate_absurd_question()
        qtype = "absurd"
    else:
        qdata = await _pop_bank_question(context)
        qtype = "serious"

    if not qdata:
        await thinking_msg.edit_text(
            format_battle_no_llm(), parse_mode=ParseMode.HTML,
        )
        return

    battles[chat_id] = {
        "question": qdata["question"],
        "answer": qdata["answer"],
        "type": qtype,
    }

    await thinking_msg.edit_text(
        format_battle_question(qdata["question"]),
        parse_mode=ParseMode.HTML,
    )

    context.job_queue.run_once(
        _battle_reminder,
        when=60,
        chat_id=int(chat_id),
        name=f"battle_reminder_{chat_id}",
        data={"chat_id": chat_id},
    )
    logger.info("Battle started in chat %s, type=%s", chat_id, qtype)


# ----- question bank helpers ---------------------------------------------

async def _pop_bank_question(
    context: ContextTypes.DEFAULT_TYPE,
) -> dict[str, str] | None:
    """Take one question from the pre-generated bank, refilling as needed.

    The bank is persisted to state.json so it survives restarts.
    On first call (empty bank) it is populated with 10 questions from Gemini 3 Pro.
    After each pop a background task tops the bank back up to 10.
    """
    bank: list[dict[str, str]] = context.bot_data.setdefault("question_bank", [])
    llm = _llm(context)
    state = context.bot_data.get("state", {})
    settings = context.bot_data.get("settings")

    if not bank:
        logger.info("Question bank empty — generating initial batch")
        questions = await llm.generate_question_bank(count=10)
        if questions:
            bank.extend(questions)
    if not bank:
        return None

    question = bank.pop(0)
    logger.info("Question bank: popped 1, %d remaining", len(bank))

    # Persist updated bank to disk
    if settings:
        save_state(state, settings.state_path)

    # Fire-and-forget: top the bank back up to 10
    state_path = settings.state_path if settings else ""
    asyncio.create_task(_refill_bank(llm, bank, state, state_path))
    return question


async def _refill_bank(
    llm: LLMClient,
    bank: list[dict[str, str]],
    state: dict[str, Any],
    state_path: str,
) -> None:
    """Background task: generate one question to keep the bank at 10."""
    if len(bank) >= 10:
        return
    try:
        new = await llm.generate_question_bank(count=1)
        if new:
            bank.extend(new)
            if state_path:
                save_state(state, state_path)
            logger.info("Question bank refilled to %d", len(bank))
    except Exception:
        logger.exception("Failed to refill question bank")


async def _battle_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a nudge after 60 s of silence; schedule the final timeout for another 60 s."""
    chat_id = context.job.data["chat_id"]
    battles = context.application.bot_data.get("battles", {})
    if chat_id not in battles:
        return

    await context.bot.send_message(
        chat_id=int(chat_id),
        text=format_battle_reminder(),
        parse_mode=ParseMode.HTML,
    )

    context.job_queue.run_once(
        _battle_final_timeout,
        when=60,
        chat_id=int(chat_id),
        name=f"battle_timeout_{chat_id}",
        data={"chat_id": chat_id},
    )
    logger.info("Battle reminder sent in chat %s", chat_id)


async def _battle_final_timeout(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel battle after 2 minutes total (60 s reminder + 60 s final)."""
    chat_id = context.job.data["chat_id"]
    battles = context.application.bot_data.get("battles", {})
    if chat_id in battles:
        del battles[chat_id]
        await context.bot.send_message(
            chat_id=int(chat_id),
            text=format_battle_timeout(),
            parse_mode=ParseMode.HTML,
        )
        logger.info("Battle timed out in chat %s", chat_id)


async def _handle_battle_evaluate(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    battle: dict[str, str],
    gigachat_response: str,
    chat_id: str,
) -> None:
    """Evaluate GigaChat's response and deliver the verdict."""
    thinking_msg = await update.message.reply_text(
        "⚔️ <i>Проверяю ответ GigaChat... это не займёт много времени.</i>",
        parse_mode=ParseMode.HTML,
    )

    evaluation = await _llm(context).evaluate_battle_answer(
        question=battle["question"],
        correct_answer=battle["answer"],
        gigachat_response=gigachat_response,
        question_type=battle["type"],
    )

    # Clear battle state and cancel pending reminder / timeout jobs
    battles = context.bot_data.get("battles", {})
    if chat_id in battles:
        del battles[chat_id]
    for suffix in ("reminder", "timeout"):
        for job in context.job_queue.get_jobs_by_name(f"battle_{suffix}_{chat_id}"):
            job.schedule_removal()

    if evaluation:
        evaluation = sanitize_html(evaluation)
        result = f"⚔️ <b>BATTLE MODE — результат</b>\n\n{evaluation}"
        await _safe_edit(thinking_msg, result, update)
    else:
        await thinking_msg.edit_text(
            "⚔️ <i>Не удалось оценить ответ. Даже профессора иногда устают...</i>",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Battle evaluated in chat %s", chat_id)


# ═══════════════════════════════════════════════════════════════════════════
# Chat-user tracking (for /hero)
# ═══════════════════════════════════════════════════════════════════════════

async def track_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Track chat participants for /hero.

    Passively records user IDs and names from incoming messages to build a list
    of candidates for the /hero command. Data is persisted to disk.
    """
    if not update.message or not update.message.from_user or update.message.from_user.is_bot:
        return

    user = update.message.from_user
    cid = str(update.message.chat_id)
    uid = str(user.id)  # JSON keys are strings

    chat_users = context.bot_data.setdefault("chat_users", {})
    users = chat_users.setdefault(cid, {})

    # Check if this is a new user or name changed — only then persist
    user_info = {"first_name": user.first_name or "Аноним", "username": user.username or ""}
    if users.get(uid) != user_info:
        users[uid] = user_info
        # Persist to state file
        state = context.bot_data.get("state", {})
        state["chat_users"] = chat_users
        settings = context.bot_data.get("settings")
        if settings:
            save_state(state, settings.state_path)


# ═══════════════════════════════════════════════════════════════════════════
# Free-text message handler (AI assistant)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _GatheredContext:
    """Collected data from parallel fetching for LLM context assembly."""

    hf_context: str | None = None
    url_context: str | None = None
    web_context: str | None = None
    image_urls: list[str] = field(default_factory=list)


async def _gather_context(
    user_text: str,
    intent_data: dict[str, Any],
    *,
    hf: HuggingFaceClient,
    search_client: SearchClient,
    reader: WebReaderClient,
) -> _GatheredContext:
    """Fetch HF model data, URL contents, and web search results in parallel."""
    user_urls = extract_urls(user_text)[:3]

    tasks: list[tuple[str, asyncio.Task[Any]]] = []
    all_images: list[str] = []

    # --- HF model / search tasks ------------------------------------------
    if intent_data["intent"] in ("compare", "info"):
        is_compare = intent_data["intent"] == "compare"

        async def _fetch_model(mid: str) -> tuple[str | None, list[str]]:
            try:
                if is_compare:
                    m_task = hf.get_model_info(mid)
                    r_task = hf.get_readme_with_images(mid)
                    model, (readme, imgs) = await asyncio.gather(m_task, r_task)
                else:
                    model = await hf.get_model_info(mid)
                    readme, imgs = None, []
                if not model:
                    return None, []
                return model.to_context(readme=readme), imgs
            except Exception:
                return None, []

        async def _search_and_fetch(q: str) -> tuple[str | None, list[str]]:
            try:
                found = await hf.search_models(q, 1)
                if not found:
                    return None, []
                model = found[0]
                if is_compare:
                    readme, imgs = await hf.get_readme_with_images(model.id)
                else:
                    readme, imgs = None, []
                return model.to_context(readme=readme), imgs
            except Exception:
                return None, []

        for mid in intent_data["models"]:
            tasks.append(("model", asyncio.ensure_future(_fetch_model(mid))))
        for q in intent_data["normalized_queries"]:
            tasks.append(("search", asyncio.ensure_future(_search_and_fetch(q))))

    # --- URL content (arxiv vs regular web pages) -------------------------
    for url in user_urls:
        if is_arxiv_url(url):
            tasks.append(("arxiv", asyncio.ensure_future(reader.fetch_arxiv_paper(url))))
        else:
            tasks.append(("url", asyncio.ensure_future(reader.fetch_url_text(url))))

    # --- Web search -------------------------------------------------------
    is_news = intent_data["intent"] == "news"
    should_search = is_news or search_client.needs_search(user_text)
    if should_search and search_client.available:
        async def _do_web_search() -> list[dict[str, str]]:
            try:
                query = (
                    f"AI LLM новости {user_text}"
                    if is_news
                    else search_client.build_query(user_text)
                )
                return await search_client.search(query, max_results=3)
            except Exception:
                logger.exception("Web search error")
                return []

        tasks.append(("web", asyncio.ensure_future(_do_web_search())))

    # --- Await & classify results -----------------------------------------
    models_data: list[str] = []
    search_results: list[dict[str, str]] = []
    url_contents: list[str] = []

    if tasks:
        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        for (task_type, _), result in zip(tasks, results):
            if isinstance(result, BaseException):
                continue
            if task_type in ("model", "search"):
                text, imgs = result
                if text:
                    models_data.append(text)
                if imgs:
                    all_images.extend(imgs)
            elif task_type == "web":
                search_results = result
            elif task_type in ("url", "arxiv") and result:
                url_contents.append(result)

    # --- Assemble context strings -----------------------------------------
    ctx = _GatheredContext(image_urls=all_images[:3])

    if models_data:
        seen: set[str] = set()
        unique: list[str] = []
        for d in models_data:
            key = d.split("\n")[0]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        ctx.hf_context = "\n\n".join(unique[:4])

    if url_contents:
        ctx.url_context = "\n\n".join(url_contents)

    if search_results:
        ctx.web_context = search_client.format_results(search_results)

    return ctx


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """AI assistant — responds to non-command text messages."""
    message = update.message
    if not message or not message.text:
        return

    # In groups: only respond when mentioned or replied-to
    is_reply_to_us = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == context.bot.id
    )
    if message.chat.type != "private":
        bot_username = context.bot.username
        mentioned = bot_username and f"@{bot_username}" in message.text
        if not mentioned and not is_reply_to_us:
            return

    user_text = message.text
    if context.bot.username:
        user_text = user_text.replace(f"@{context.bot.username}", "").strip()

    # Capture reply context from ANY replied-to message (ours, other bots, humans)
    reply_context = None
    if message.reply_to_message and message.reply_to_message.text:
        reply_context = message.reply_to_message.text

    # --- Battle mode ---
    # During an active battle, replies are ignored so that users can freely
    # interact with GigaChat via reply (GigaChat bot can't see bot messages).
    # Evaluation is triggered by a direct mention (no reply) — the user's
    # text is treated as GigaChat's response.
    chat_id = str(message.chat_id)
    battle = context.bot_data.get("battles", {}).get(chat_id)
    if battle:
        if message.reply_to_message:
            return  # let the user talk to GigaChat via replies
        await _handle_battle_evaluate(update, context, battle, user_text, chat_id)
        return

    if not user_text:
        return

    # Show "thinking" placeholder
    thinking_msg = await message.reply_text(
        random.choice(THINKING_PHRASES), parse_mode=ParseMode.HTML,
    )

    # Analyse intent & gather data
    intent_data = analyze_intent(user_text)
    logger.info(
        "Intent: %s, models: %s, queries: %s",
        intent_data.get("intent"), intent_data.get("models"), intent_data.get("normalized_queries"),
    )

    gathered = await _gather_context(
        user_text,
        intent_data,
        hf=_hf(context),
        search_client=_search(context),
        reader=_web_reader(context),
    )

    # Generate LLM response
    state = context.bot_data.get("state", {})

    try:
        response = await _llm(context).generate_response(
            user_text,
            gathered.hf_context,
            gathered.image_urls or None,
            reply_context=reply_context,
            search_context=gathered.web_context,
            url_context=gathered.url_context,
            model_examples=get_example_models(state),
        )
    except Exception:
        logger.exception("LLM generation error")
        response = None

    if response:
        response = sanitize_html(response)
        await _safe_edit(thinking_msg, response, update)
    else:
        await thinking_msg.edit_text(
            "🤖 <i>Что-то пошло не так... Даже я иногда ломаюсь. Попробуй ещё раз.</i>",
            parse_mode=ParseMode.HTML,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _safe_edit(msg: Message, text: str, update: Update) -> None:
    """Edit a message with HTML and fall back to plain text."""
    try:
        await msg.edit_text(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        plain = re.sub(r"<[^>]+>", "", text)
        try:
            await msg.edit_text(plain, disable_web_page_preview=True)
        except Exception:
            await update.message.reply_text(plain, disable_web_page_preview=True)
