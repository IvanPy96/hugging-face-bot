"""Telegram message formatting (HTML).

All bot-facing text formatting lives here: model cards, notifications,
stats, commands, and HTML sanitisation. Content strings (org comments,
random phrases, hero messages) live in content.py.
"""

from __future__ import annotations

import datetime
import html
import random
import re

from hf_bot.content import (
    BATTLE_INTRO_PHRASES,
    BATTLE_REMINDER_MESSAGE,
    BATTLE_TIMEOUT_MESSAGE,
    BATTLE_WAITING_PHRASES,
    DEFAULT_ORG_COMMENTS,
    HERO_FALLBACKS,
    ORG_COMMENTS,
    ORG_PHRASES,
    RANDOM_MODEL_COMMENTS,
    RANDOM_PHRASES,
    THINKING_PHRASES,
    gigachat_roast,
    is_gigachat,
    stat_comment,
)
from hf_bot.models import DeployInfo, ModelInfo

# Re-export so existing imports from handlers.py keep working
__all__ = [
    "THINKING_PHRASES",
    "format_agi_check",
    "format_battle_already_active",
    "format_battle_no_llm",
    "format_battle_question",
    "format_battle_reminder",
    "format_battle_timeout",
    "format_deploy_info",
    "format_deploy_usage",
    "format_error",
    "format_help_message",
    "format_hero_message",
    "format_info_usage",
    "format_model_card",
    "format_model_not_found",
    "format_new_model_notification",
    "format_number",
    "format_orgs_list",
    "format_random_model",
    "format_start_message",
    "format_stats",
    "sanitize_html",
]


# ═══════════════════════════════════════════════════════════════════════════
# HTML sanitisation pipeline (markdown → HTML, unclosed-tag fix, escaping)
# ═══════════════════════════════════════════════════════════════════════════


def _fix_markdown_to_html(text: str) -> str:
    """Convert common Markdown markup to Telegram-compatible HTML."""
    if not text:
        return text

    # --- protect code blocks ---
    code_blocks: list[str] = []

    def _save_block(m: re.Match[str]) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```(?:\w+)?\n?(.*?)```", _save_block, text, flags=re.DOTALL)

    inline_codes: list[str] = []

    def _save_inline(m: re.Match[str]) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    if "<code>" not in text:
        text = re.sub(r"`([^`\n]+)`", _save_inline, text)

    # --- bold / italic ---
    if "<b>" not in text and "<strong>" not in text:
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    if "<i>" not in text and "<em>" not in text:
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"<i>\1</i>", text)

    # --- links ---
    if "<a href" not in text:
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # --- headings ---
    text = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # --- restore protected sections ---
    for i, code in enumerate(inline_codes):
        text = text.replace(f"\x00IC{i}\x00", f"<code>{html.escape(code)}</code>")
    for i, code in enumerate(code_blocks):
        text = text.replace(f"\x00CB{i}\x00", f"<pre>{html.escape(code.strip())}</pre>")

    return text


def _fix_unclosed_tags(text: str) -> str:
    """Balance opening/closing tags for safe Telegram HTML."""
    if not text:
        return text
    for tag in ("b", "i", "u", "s", "code", "pre", "a"):
        opens = len(re.findall(rf"<{tag}(?:\s|>)", text, re.IGNORECASE))
        closes = len(re.findall(rf"</{tag}>", text, re.IGNORECASE))
        while opens > closes:
            text += f"</{tag}>"
            closes += 1
        while closes > opens:
            text = re.sub(rf"</{tag}>$", "", text.rstrip())
            closes -= 1
    return text


def _escape_outside_tags(text: str) -> str:
    """Escape &, <, and > outside HTML tags for Telegram safety."""
    if not text:
        return text
    tag_re = re.compile(r"<(/?)(\w+)([^>]*)>")
    result: list[str] = []
    last = 0
    for m in tag_re.finditer(text):
        between = text[last : m.start()]
        between = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", between)
        between = re.sub(r"<(?!/?\w)", "&lt;", between)
        between = re.sub(r"(?<!\w)>(?!/)", "&gt;", between)
        result.append(between)
        result.append(m.group(0))
        last = m.end()
    tail = text[last:]
    tail = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", tail)
    tail = re.sub(r"<(?!/?\w)", "&lt;", tail)
    tail = re.sub(r"(?<!\w)>(?!/)", "&gt;", tail)
    result.append(tail)
    return "".join(result)


def sanitize_html(text: str) -> str:
    """Full post-processing pipeline: markdown->HTML, escape, fix tags."""
    if not text:
        return text
    text = _fix_markdown_to_html(text)
    text = _escape_outside_tags(text)
    text = _fix_unclosed_tags(text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Number formatting
# ═══════════════════════════════════════════════════════════════════════════


def format_number(n: int) -> str:
    """Format a count as 1.2M, 45.3K, or a plain number."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════════
# Visual dividers
# ═══════════════════════════════════════════════════════════════════════════

_DIV = "─" * 24
_SEP = "━━━━━━━━━━━━━━━━━━━━"


# ═══════════════════════════════════════════════════════════════════════════
# Public formatters
# ═══════════════════════════════════════════════════════════════════════════


# ----- Model card (/info) ------------------------------------------------

def format_model_card(model: ModelInfo) -> str:
    """Format a model info card as an HTML message."""
    is_gc = is_gigachat(model.id)
    lines: list[str] = [f"🤖 <b>{model.id}</b>"]
    if is_gc:
        lines.append("<i>(⚠️ Осторожно, GigaChat! Возможны галлюцинации и разочарование.)</i>")
    lines.append(f"<code>{_DIV}</code>")

    if model.downloads or model.likes:
        parts = []
        if model.downloads:
            parts.append(f"📥 <b>{format_number(model.downloads)}</b> downloads")
        if model.likes:
            parts.append(f"❤️ <b>{format_number(model.likes)}</b>")
        lines.append("   ".join(parts))

    meta = []
    if model.pipeline_tag:
        meta.append(f"🎯 {model.pipeline_tag}")
    if model.library_name:
        meta.append(f"📚 {model.library_name}")
    if meta:
        lines.append("  ".join(meta))

    useful = model.useful_tags(6)
    if useful:
        lines.append("🏷 " + " · ".join(f"<code>{t}</code>" for t in useful))

    lines.append(f"<code>{_DIV}</code>")

    if is_gc:
        lines += [f"💬 <i>{gigachat_roast()}</i>", ""]

    lines.append(f'🔗 <a href="{model.url}">Открыть на Hugging Face</a>')
    return "\n".join(lines)


# ----- New-model notification --------------------------------------------

def format_new_model_notification(org: str, model_id: str) -> str:
    """Format a notification message for a newly detected model."""
    url = f"https://huggingface.co/{model_id}"
    pool = RANDOM_PHRASES + ORG_PHRASES.get(org, [])
    phrase = random.choice(pool)
    is_gc = is_gigachat(model_id)

    if is_gc:
        comment = gigachat_roast()
        header = "🚨 <b>Тревога!</b> (ложная, это просто GigaChat)"
        intro = f"Сбер выкатил очередной <b>GigaChat</b>... опять. 🙄"
    else:
        comment = random.choice(ORG_COMMENTS.get(org, DEFAULT_ORG_COMMENTS))
        header = "🚨 <b>Ахтунг!</b>"
        intro = f"Вышла новая модель от <b>{org}</b>!"

    return (
        f"{header}\n\n{intro}\n<i>{phrase}</i>\n\n{_SEP}\n\n"
        f"🤖 <b>{model_id}</b>\n\n"
        f"💬 {comment}\n\n"
        f'🔗 <a href="{url}">Смотреть веса</a>'
    )


# ----- /start, /help -----------------------------------------------------

def format_start_message() -> str:
    """Return the welcome message for /start."""
    return (
        "👋 <b>Привет!</b>\n\n"
        "Я слежу за новыми моделями на <b>Hugging Face</b> "
        "и присылаю уведомления, когда появляется что-то новое.\n\n"
        f"{_SEP}\n\n"
        "🤖 <b>AI-ассистент</b>\n\n"
        "Просто напиши мне — отвечу на вопросы о моделях, "
        "сравню их по бенчмаркам, посоветую что выбрать.\n\n"
        f"{_SEP}\n\n"
        "📋 <b>Команды:</b>\n\n"
        "  /orgs — отслеживаемые организации\n"
        "  /info <code>model_id</code> — карточка модели\n"
        "  /deploy <code>model_id</code> — расчёт GPU для деплоя\n"
        "  /stats — статистика\n"
        "  /random — случайная модель\n"
        "  /hero — мотивация для случайного участника\n"
        "  /battle — викторина-дуэль с GigaChat\n"
        "  /help — справка\n\n"
        f"{_SEP}\n\n"
        "💡 <i>Уведомления приходят автоматически</i>"
    )


def format_help_message() -> str:
    """Return the help message for /help."""
    return (
        f"📖 <b>Справка</b>\n\n{_SEP}\n\n"
        "🔔 <b>Мониторинг</b>\n\n"
        "Каждую минуту проверяю Hugging Face на наличие новых моделей. "
        "Уведомления приходят автоматически.\n\n"
        f"{_SEP}\n\n"
        "🤖 <b>AI-ассистент</b>\n\n"
        "Просто напиши сообщение (без команды) — отвечу!\n\n"
        "Примеры:\n"
        "• <i>Сравни Qwen3 и DeepSeek V3</i>\n"
        "• <i>Что за модель Mistral Large?</i>\n"
        "• <i>Посоветуй модель для кода</i>\n"
        "• <i>Когда будет AGI?</i>\n\n"
        "Читаю карточки моделей с HF, сравниваю по бенчмаркам.\n\n"
        f"{_SEP}\n\n"
        "📋 <b>Команды</b>\n\n"
        "  /orgs — отслеживаемые организации\n"
        "  /info <code>автор/модель</code> — карточка модели\n"
        "  /deploy <code>автор/модель</code> — расчёт GPU для деплоя\n"
        "  /stats — статистика мониторинга\n"
        "  /random — случайная модель\n"
        "  /hero — мотивация для случайного участника\n"
        "  /battle — викторина-дуэль с GigaChat\n\n"
        f"{_SEP}\n\n"
        "⚔️ <b>Battle Mode</b>\n\n"
        "Напиши /battle — бот задаст каверзный вопрос GigaChat-боту.\n"
        "Перешли вопрос GigaChat-у (через реплай на его сообщение).\n"
        "Когда GigaChat ответит — тегни меня и скинь его ответ (без реплая).\n"
        "Я оценю ответ (спойлер: будет больно). Таймаут — 2 минуты.\n\n"
        f"{_SEP}\n\n"
        "💡 <b>Пример команды</b>\n\n"
        "<code>/info Qwen/Qwen2-72B-Instruct</code>"
    )


# ----- /orgs -------------------------------------------------------------

def format_orgs_list(orgs: list[str]) -> str:
    """Format the list of monitored organisations."""
    lines = [f"🏢 <b>Отслеживаемые организации</b>\n\n{_SEP}\n"]
    for o in orgs:
        lines.append(f'  • <a href="https://huggingface.co/{o}">{o}</a>')
    lines += ["", _SEP, "", f"📊 Всего: <b>{len(orgs)}</b>"]
    return "\n".join(lines)


# ----- /stats ------------------------------------------------------------

def format_stats(org_stats: dict[str, int], total: int) -> str:
    """Format the statistics message with medals and comments."""
    medals = {0: "🥇 ", 1: "🥈 ", 2: "🥉 "}
    sorted_orgs = sorted(org_stats.items(), key=lambda x: x[1], reverse=True)
    lines = [f"📊 <b>Статистика мониторинга</b>\n\n{_SEP}\n"]
    for i, (org, cnt) in enumerate(sorted_orgs):
        m = medals.get(i, "") if cnt > 0 else ""
        pct = cnt / total * 100 if total > 0 else 0
        comment = stat_comment(cnt, org)
        lines += [f"  {m}<b>{org}</b>: {cnt:,} ({pct:.1f}%)", f"  <i>└ {comment}</i>", ""]
    days = (datetime.date.today() - datetime.date(2023, 3, 14)).days
    lines += [
        _SEP, "",
        f"🤖 Всего моделей: <b>{total:,}</b>",
        f"🏢 Организаций: <b>{len(sorted_orgs)}</b>",
        f"⏳ Дней без AGI: <b>{days}</b>",
    ]
    return "\n".join(lines)


# ----- /agi (easter egg) -------------------------------------------------

def format_agi_check() -> str:
    """Format the AGI check easter egg message."""
    pct = random.randint(65, 95)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    excuses = [
        "Попробуйте завтра.", "Ждите следующий релиз от DeepSeek.",
        "OpenAI обещали скоро.", "Илон говорит, что уже почти.",
        "Нужно больше H100.", "Ещё пару эпох training'а.",
        "Scale is all you need.", "Демис Хассабис работает над этим.",
    ]
    return (
        f"🤖 <b>Проверяю наличие AGI...</b>\n\n"
        f"<code>{bar}</code> {pct}%\n\n"
        f"{_SEP}\n\n"
        f"❌ <b>AGI не обнаружен.</b>\n\n"
        f"💬 <i>{random.choice(excuses)}</i>"
    )


# ----- /random -----------------------------------------------------------

def format_random_model(model: ModelInfo, *, forced_gigachat: bool = False) -> str:
    """Format a random model card with a humorous header."""
    is_gc = is_gigachat(model.id)
    if is_gc:
        sub = '(вам "повезло" — GigaChat! 🎰 Соболезнуем.)' if forced_gigachat else "(увы, это GigaChat... крутите ещё раз 😅)"
        header = f"🎲 <b>Случайная модель дня</b>\n<i>{sub}</i>"
        comment = gigachat_roast()
    else:
        header = "🎲 <b>Случайная модель дня</b>"
        comment = random.choice(RANDOM_MODEL_COMMENTS)

    lines = [header, "", _SEP, "", f"🤖 <b>{model.id}</b>", f"<code>{_DIV}</code>"]
    if model.downloads or model.likes:
        parts = []
        if model.downloads:
            parts.append(f"📥 <b>{format_number(model.downloads)}</b>")
        if model.likes:
            parts.append(f"❤️ <b>{format_number(model.likes)}</b>")
        lines.append("   ".join(parts))
    if model.pipeline_tag:
        lines.append(f"🎯 {model.pipeline_tag}")
    lines += [f"<code>{_DIV}</code>", f"💬 <i>{comment}</i>", "", f'🔗 <a href="{model.url}">Посмотреть</a>']
    return "\n".join(lines)


# ----- /hero -------------------------------------------------------------

def format_hero_message(mention: str, message: str = "") -> str:
    """Format the hero of the day message."""
    if not message:
        message = random.choice(HERO_FALLBACKS)
    return f"🦸 <b>Герой дня!</b>\n\n{mention}, это тебе:\n\n{_SEP}\n\n💌 {message}\n\n{_SEP}"


# ----- /deploy -----------------------------------------------------------

def format_deploy_info(deploy: DeployInfo, model_id: str) -> str:
    """Format GPU deployment requirements."""
    if deploy.total_params >= 1e9:
        ps = f"{deploy.total_params / 1e9:.1f}B"
    elif deploy.total_params >= 1e6:
        ps = f"{deploy.total_params / 1e6:.0f}M"
    else:
        ps = f"{deploy.total_params:,}"

    lines = [
        f"🖥️ <b>Расчёт деплоя</b>: <code>{html.escape(model_id)}</code>",
        "", _SEP, "",
        f"📊 Параметры: <b>{ps}</b>",
        f"💾 Точность: <b>{deploy.dtype}</b>",
        f"📦 Размер весов: <b>~{deploy.weight_gb:.1f} ГБ</b>",
        f"📈 С запасом на инференс (~20%): <b>~{deploy.total_gb:.1f} ГБ</b>",
        "", _SEP, "",
    ]

    # H200
    if deploy.h200_count == 1:
        spare = 140 - deploy.total_gb
        note = f"→ <b>1 × H200</b> (запас ~{spare:.0f} ГБ{' — шикарно!' if spare > 70 else ''})"
        emoji = "🟢"
    elif deploy.h200_count <= 8:
        note = f"→ <b>{deploy.h200_count} × H200</b> (одна HGX-нода)"
        emoji = "🟡"
    else:
        nodes = (deploy.h200_count + 7) // 8
        note = f"→ <b>{deploy.h200_count} × H200</b> ({nodes} нод — серьёзная заявка!)"
        emoji = "🔴"
    lines += [f"{emoji} <b>NVIDIA H200</b> (140 ГБ VRAM):", f"  {note}", ""]

    # L40s
    if deploy.l40s_fits:
        spare = 48 - deploy.total_gb
        lines += [f"🟢 <b>NVIDIA L40s</b> (48 ГБ VRAM):", f"  → <b>1 × L40s</b> (запас ~{spare:.0f} ГБ)"]
    else:
        lines += ["🔴 <b>NVIDIA L40s</b> (48 ГБ VRAM):", "  → Сюда никак не влезет! 😤"]
    return "\n".join(lines)


# ----- Error / usage messages --------------------------------------------

def format_model_not_found(model_id: str) -> str:
    """Format a 'model not found' error message."""
    return f"❌ Модель не найдена\n\n<code>{model_id}</code>\n\n💡 Проверьте правильность написания.\nФормат: <code>автор/название-модели</code>"


def format_info_usage() -> str:
    """Format usage instructions for /info."""
    return "ℹ️ <b>Укажите модель</b>\n\nФормат: <code>/info автор/модель</code>\n\n💡 Пример:\n<code>/info Qwen/Qwen2-72B-Instruct</code>"


def format_deploy_usage() -> str:
    """Format usage instructions for /deploy."""
    return "🖥️ <b>Расчёт деплоя</b>\n\nФормат: <code>/deploy автор/модель</code>\n\n💡 Пример:\n<code>/deploy Qwen/Qwen3-32B</code>"


def format_error() -> str:
    """Format a generic error message."""
    return "⚠️ Произошла ошибка. Попробуйте позже."


# ----- /battle -----------------------------------------------------------

def format_battle_question(question: str) -> str:
    """Format the battle question message that triggers GigaChat."""
    intro = random.choice(BATTLE_INTRO_PHRASES)
    hint = random.choice(BATTLE_WAITING_PHRASES)
    return (
        f"⚔️ <b>BATTLE MODE</b>\n\n"
        f"{intro}\n\n"
        f"{_SEP}\n\n"
        f"Гигачат, {question}\n\n"
        f"{_SEP}\n\n"
        f"<i>{hint}</i>"
    )


def format_battle_reminder() -> str:
    """Format the battle reminder (sent after 1 minute of silence)."""
    return f"⚔️ {BATTLE_REMINDER_MESSAGE}"


def format_battle_timeout() -> str:
    """Format the final battle timeout message (sent after 2 minutes total)."""
    return f"⚔️ <b>BATTLE MODE — завершён</b>\n\n{BATTLE_TIMEOUT_MESSAGE}"


def format_battle_already_active() -> str:
    """Format a message when battle is already in progress."""
    return "⚔️ Battle уже идёт! Дождитесь окончания текущего раунда."


def format_battle_no_llm() -> str:
    """Format a message when LLM failed to generate a question."""
    return "⚔️ Не удалось придумать вопрос. Даже я иногда туплю. Попробуйте ещё раз!"
