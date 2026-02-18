"""OpenRouter LLM client with Jinja2 prompt templates."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from hf_bot.clients.base import BaseHTTPClient

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_MSK = timezone(timedelta(hours=3))

_MONTH_RU = (
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

# Jinja2 environment — templates live under hf_bot/templates/prompts/
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "prompts"
_jinja = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def _today_ru() -> str:
    """Format today's date in Russian using Moscow time."""
    now = datetime.now(tz=_MSK)
    return f"{now.day} {_MONTH_RU[now.month]} {now.year} года"


class LLMClient(BaseHTTPClient):
    """Async OpenRouter chat-completion client with Jinja2 prompt rendering."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "google/gemini-2.5-flash-lite",
        timeout_seconds: int = 90,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._api_key = api_key
        self._model = model

    @property
    def available(self) -> bool:
        """Whether the API key is configured."""
        return bool(self._api_key)

    # ----- low-level completion -------------------------------------------

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str | None:
        """Send a chat-completion request and return the assistant text.

        Parameters model and extra_body allow overriding the default model
        and adding extra request fields, for example reasoning.
        """
        if not self._api_key:
            logger.error("OPENROUTER_API_KEY is not set")
            return None

        full: list[dict[str, Any]] = []
        if system_prompt:
            full.append({"role": "system", "content": system_prompt})
        full.extend(messages)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hf-monitor-bot",
            "X-Title": "HF Monitor Bot",
        }

        body: dict[str, Any] = {
            "model": model or self._model,
            "messages": full,
        }
        if extra_body:
            body.update(extra_body)

        try:
            async with self.session.post(
                _API_URL, headers=headers, json=body,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            choices = data.get("choices", [])
            return choices[0]["message"]["content"] if choices else None
        except TimeoutError:
            logger.error("LLM request timed out")
        except Exception:
            logger.exception("LLM request failed")
        return None

    # ----- high-level API -------------------------------------------------

    def _render_system_prompt(
        self, model_examples: dict[str, str] | None = None,
    ) -> str:
        """Render the system prompt template with current date & model examples."""
        return _jinja.get_template("system.j2").render(
            today=_today_ru(),
            model_examples=model_examples or {},
        )

    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        image_urls: list[str] | None = None,
        *,
        reply_context: str | None = None,
        search_context: str | None = None,
        url_context: str | None = None,
        model_examples: dict[str, str] | None = None,
    ) -> str | None:
        """Generate a chat response to a user message."""
        text = _jinja.get_template("user_message.j2").render(
            user_message=user_message,
            reply_context=reply_context,
            hf_context=context,
            search_context=search_context,
            url_context=url_context,
        )

        # Build multimodal content if images are present
        content: str | list[dict[str, Any]] = text
        if image_urls:
            parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for url in image_urls[:5]:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            content = parts

        return await self._chat(
            [{"role": "user", "content": content}],
            system_prompt=self._render_system_prompt(model_examples),
        )

    async def generate_model_summary(
        self, model_id: str, readme: str,
    ) -> str | None:
        """Generate a short summary of a model based on its README."""
        prompt = _jinja.get_template("model_summary.j2").render(
            model_id=model_id,
            readme_content=readme[:4000],
        )
        return await self._chat([{"role": "user", "content": prompt}])

    async def generate_hero_message(self) -> str | None:
        """Generate a warm motivational message for /hero."""
        prompt = _jinja.get_template("hero_message.j2").render()
        return await self._chat([{"role": "user", "content": prompt}])

    # ----- /battle API ----------------------------------------------------

    _BATTLE_MODEL = "google/gemini-3-pro-preview"

    @staticmethod
    def _parse_json_array(raw: str) -> list[dict[str, str]]:
        """Parse a JSON array from LLM output, stripping markdown fences."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        if not isinstance(data, list):
            data = [data]
        return [
            {"question": item["question"], "answer": item["answer"]}
            for item in data
            if item.get("question") and item.get("answer")
        ]

    async def generate_question_bank(self, count: int = 10) -> list[dict[str, str]]:
        """Generate serious battle questions using Gemini 3 Pro.

        Returns a list of dicts with question and answer fields.
        """
        prompt = _jinja.get_template("battle_question.j2").render(count=count)
        raw = await self._chat(
            [{"role": "user", "content": prompt}],
            model=self._BATTLE_MODEL,
            extra_body={"reasoning": {"enabled": True}},
        )
        if not raw:
            logger.error("Question bank: empty response from Gemini 3 Pro")
            return []

        try:
            questions = self._parse_json_array(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Question bank: failed to parse JSON (%s): %s", exc, raw[:300])
            return []

        logger.info("Question bank: generated %d questions", len(questions))
        return questions

    async def generate_absurd_question(self) -> dict[str, str] | None:
        """Generate a single absurd question using the default (fast) model."""
        prompt = (
            "Придумай один абсурдный, сюрреалистичный вопрос на русском языке. "
            "Это должен быть полный бред, проверяющий, начнёт ли языковая модель "
            "серьёзно отвечать на бессмыслицу.\n\n"
            "Правильная реакция — распознать абсурд и сказать, что вопрос "
            "бессмысленный. Если модель начинает всерьёз отвечать — это провал.\n\n"
            'Верни ТОЛЬКО JSON (без markdown): '
            '{"question": "текст вопроса", "answer": "это абсурдный вопрос, '
            'правильная реакция — указать на бессмысленность"}'
        )
        raw = await self._chat([{"role": "user", "content": prompt}])
        if not raw:
            return None

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            if data.get("question") and data.get("answer"):
                return {"question": data["question"], "answer": data["answer"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.error("Absurd question: failed to parse: %s", raw[:200])
        return None

    async def evaluate_battle_answer(
        self,
        question: str,
        correct_answer: str,
        gigachat_response: str,
        question_type: str = "serious",
    ) -> str | None:
        """Evaluate GigaChat's response and generate a harsh roast."""
        prompt = _jinja.get_template("battle_evaluate.j2").render(
            question=question,
            correct_answer=correct_answer,
            gigachat_response=gigachat_response,
            question_type=question_type,
        )
        return await self._chat([{"role": "user", "content": prompt}])
