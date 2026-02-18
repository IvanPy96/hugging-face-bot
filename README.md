<div align="center">

<br>

```
 ██╗  ██╗███████╗    ███╗   ███╗ ██████╗ ███╗   ██╗██╗████████╗ ██████╗ ██████╗ 
 ██║  ██║██╔════╝    ████╗ ████║██╔═══██╗████╗  ██║██║╚══██╔══╝██╔═══██╗██╔══██╗
 ███████║█████╗      ██╔████╔██║██║   ██║██╔██╗ ██║██║   ██║   ██║   ██║██████╔╝
 ██╔══██║██╔══╝      ██║╚██╔╝██║██║   ██║██║╚██╗██║██║   ██║   ██║   ██║██╔══██╗
 ██║  ██║██║         ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║   ██║   ╚██████╔╝██║  ██║
 ╚═╝  ╚═╝╚═╝         ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

### Telegram-бот для мониторинга AI-моделей на Hugging Face

AI-уведомления · саммари · `/battle`-режим · саркастичный ассистент

<br>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Bot_API-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Ruff](https://img.shields.io/badge/Code_style-Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)

<br>

[**🇷🇺 Русский**](#-русский) · [**🇬🇧 English**](#-english)

<br>

</div>

---

## 🇷🇺 Русский

### Что это?

Бот следит за **17 ведущими AI-организациями** на Hugging Face и моментально сообщает о новых моделях в Telegram. Под капотом — AI-саммари, оценка VRAM для деплоя, умный ассистент с веб-поиском и легендарный Battle Mode, где GigaChat отвечает на каверзные вопросы.

> **100% vibecode** — написан с помощью **Claude Opus 4.6** 🧠

<br>

### Возможности

<table>
<tr>
<td width="50%">

**🔔 Автомониторинг**
Отслеживает новые модели от Meta, Google, OpenAI, Anthropic, Qwen, DeepSeek, Mistral, NVIDIA, xAI и других — уведомления приходят мгновенно

</td>
<td width="50%">

**🧠 AI-саммари**
Автоматически генерирует краткое описание новой модели на основе её README — чтобы не читать самому

</td>
</tr>
<tr>
<td>

**🖥️ Оценка деплоя**
Считает количество параметров, VRAM и подсказывает подходящий GPU для запуска модели

</td>
<td>

**💬 AI-ассистент**
Отвечает на вопросы о моделях, читает arXiv-статьи, ищет в интернете через Brave Search

</td>
</tr>
<tr>
<td>

**⚔️ Battle Mode**
Викторина: Gemini 3 Pro генерирует вопрос, GigaChat отвечает, бот выносит вердикт

</td>
<td>

**🦸 Герой дня**
Выбирает случайного участника чата и генерирует ему персональное мотивационное сообщение

</td>
</tr>
</table>

<br>

### Команды

| Команда | Описание |
|:---|:---|
| `/start` | Приветствие и краткое описание бота |
| `/help` | Подробная справка по всем командам |
| `/info <model_id>` | Карточка модели: скачивания, лайки, теги, ссылка |
| `/deploy <model_id>` | Оценка VRAM и GPU для деплоя модели |
| `/orgs` | Список отслеживаемых организаций |
| `/stats` | Лидерборд организаций по количеству моделей |
| `/random` | Случайная модель с HuggingFace |
| `/hero` | Выбрать героя дня и отправить мотивацию |
| `/battle` | Запустить викторину против GigaChat |
| `/agi` | Проверить наличие AGI (спойлер: нет) |

<br>

### Быстрый старт

#### Docker (рекомендуется)

```bash
git clone <repo_url> && cd hf_bot
cp .env.example .env      # заполните BOT_TOKEN и CHAT_ID
docker compose up -d --build
```

```bash
docker compose logs -f     # логи
docker compose down        # остановка
```

#### Локально

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
cp .env.example .env       # заполните BOT_TOKEN и CHAT_ID
python -m hf_bot
```

<br>

### Переменные окружения

| Переменная | Обязательная | По умолчанию | Описание |
|:---|:---:|:---|:---|
| `BOT_TOKEN` | ✅ | — | Токен Telegram-бота от [@BotFather](https://t.me/BotFather) |
| `CHAT_ID` | ✅ | — | ID чата для уведомлений |
| `OPENROUTER_API_KEY` | — | — | Ключ [OpenRouter](https://openrouter.ai/) для AI-функций |
| `BRAVE_SEARCH_API_KEY` | — | — | Ключ [Brave Search](https://brave.com/search/api/) (2000 req/мес бесплатно) |
| `LLM_MODEL` | — | `google/gemini-2.5-flash-lite` | Модель для LLM-ответов |
| `LLM_TIMEOUT_SECONDS` | — | `90` | Таймаут LLM-запросов |
| `POLL_SECONDS` | — | `60` | Интервал опроса HuggingFace (сек) |
| `HF_TIMEOUT_SECONDS` | — | `10` | Таймаут HF API (сек) |
| `STATE_PATH` | — | `data/state.json` | Путь к файлу состояния |
| `LOG_LEVEL` | — | `INFO` | Уровень логирования |

<br>

### Архитектура

```
hf_bot/
├── app.py              ← точка входа, lifecycle, регистрация хендлеров
├── handlers.py         ← команды бота и AI-ассистент
├── monitoring.py       ← периодический мониторинг HF + уведомления
├── intent.py           ← NLU: анализ намерений пользователя
├── formatter.py        ← форматирование сообщений (HTML)
├── content.py          ← тексты, фразы, рандомные комментарии
├── models.py           ← модели данных (DeployInfo и т.д.)
├── state.py            ← персистентное JSON-состояние с атомарной записью
├── config.py           ← настройки из env-переменных
├── clients/
│   ├── huggingface.py  ← API HuggingFace (модели, README, поиск)
│   ├── llm.py          ← OpenRouter API (Gemini, саммари, battle)
│   ├── search.py       ← Brave Search API
│   └── web_reader.py   ← парсинг веб-страниц и arXiv
└── templates/prompts/  ← Jinja2-шаблоны для LLM-промптов
```

<br>

### Стек

| Компонент | Технология |
|:---|:---|
| Telegram Bot | [python-telegram-bot](https://python-telegram-bot.org/) 21.x |
| HTTP-клиент | [aiohttp](https://docs.aiohttp.org/) 3.x |
| LLM | [OpenRouter](https://openrouter.ai/) → Gemini 2.5 Flash Lite |
| Веб-поиск | [Brave Search API](https://brave.com/search/api/) |
| Шаблоны | [Jinja2](https://jinja.palletsprojects.com/) 3.x |
| Парсинг | [trafilatura](https://trafilatura.readthedocs.io/) + [arxiv](https://pypi.org/project/arxiv/) |
| Контейнеризация | Docker + Docker Compose |
| Линтер | [Ruff](https://docs.astral.sh/ruff/) |

<br>

### Отслеживаемые организации

<div align="center">

`Meta` · `Google` · `OpenAI` · `Anthropic` · `Qwen` · `DeepSeek` · `Mistral` · `NVIDIA` · `xAI` · `Tencent` · `MiniMax` · `Moonshot` · `inclusionAI` · `Yandex` · `T-Tech` · `Sber (ai-sage)` · `Z.ai (zai-org)`

</div>

<br>

---

## 🇬🇧 English

### What is this?

A Telegram bot that monitors **17 leading AI organizations** on Hugging Face and instantly reports new models. Under the hood — AI summaries, VRAM estimation for deployment, a smart assistant with web search, and the legendary Battle Mode where GigaChat answers tricky questions.

> **100% vibecoded** with **Claude Opus 4.6** 🧠

<br>

### Features

<table>
<tr>
<td width="50%">

**🔔 Auto-monitoring**
Tracks new models from Meta, Google, OpenAI, Anthropic, Qwen, DeepSeek, Mistral, NVIDIA, xAI and more — instant Telegram alerts

</td>
<td width="50%">

**🧠 AI Summaries**
Auto-generates concise model descriptions from README files — so you don't have to read them

</td>
</tr>
<tr>
<td>

**🖥️ Deploy Estimation**
Calculates parameter count, VRAM requirements, and suggests the right GPU for running a model

</td>
<td>

**💬 AI Assistant**
Answers questions about models, reads arXiv papers, searches the web via Brave Search

</td>
</tr>
<tr>
<td>

**⚔️ Battle Mode**
Quiz: Gemini 3 Pro generates a question, GigaChat answers, the bot delivers the verdict

</td>
<td>

**🦸 Hero of the Day**
Picks a random chat member and generates a personalized motivational message

</td>
</tr>
</table>

<br>

### Commands

| Command | Description |
|:---|:---|
| `/start` | Welcome message and bot overview |
| `/help` | Detailed help for all commands |
| `/info <model_id>` | Model card: downloads, likes, tags, link |
| `/deploy <model_id>` | VRAM and GPU estimation for deployment |
| `/orgs` | List of monitored organizations |
| `/stats` | Org leaderboard by model count |
| `/random` | Random model from HuggingFace |
| `/hero` | Pick hero of the day and send motivation |
| `/battle` | Start a quiz battle against GigaChat |
| `/agi` | Check for AGI (spoiler: nope) |

<br>

### Quick Start

#### Docker (recommended)

```bash
git clone <repo_url> && cd hf_bot
cp .env.example .env      # fill in BOT_TOKEN and CHAT_ID
docker compose up -d --build
```

```bash
docker compose logs -f     # logs
docker compose down        # stop
```

#### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
cp .env.example .env       # fill in BOT_TOKEN and CHAT_ID
python -m hf_bot
```

<br>

### Environment Variables

| Variable | Required | Default | Description |
|:---|:---:|:---|:---|
| `BOT_TOKEN` | ✅ | — | Telegram bot token from [@BotFather](https://t.me/BotFather) |
| `CHAT_ID` | ✅ | — | Chat ID for notifications |
| `OPENROUTER_API_KEY` | — | — | [OpenRouter](https://openrouter.ai/) key for AI features |
| `BRAVE_SEARCH_API_KEY` | — | — | [Brave Search](https://brave.com/search/api/) key (2000 req/mo free) |
| `LLM_MODEL` | — | `google/gemini-2.5-flash-lite` | Model for LLM responses |
| `LLM_TIMEOUT_SECONDS` | — | `90` | LLM request timeout |
| `POLL_SECONDS` | — | `60` | HuggingFace polling interval (sec) |
| `HF_TIMEOUT_SECONDS` | — | `10` | HF API timeout (sec) |
| `STATE_PATH` | — | `data/state.json` | Path to state file |
| `LOG_LEVEL` | — | `INFO` | Logging level |

<br>

### Project Layout

```
hf_bot/
├── app.py              ← entry point, lifecycle, handler registration
├── handlers.py         ← bot commands and AI assistant
├── monitoring.py       ← periodic HF monitoring + notifications
├── intent.py           ← NLU: user intent analysis
├── formatter.py        ← message formatting (HTML)
├── content.py          ← text banks, phrases, random comments
├── models.py           ← data models (DeployInfo, etc.)
├── state.py            ← persistent JSON state with atomic writes
├── config.py           ← settings from environment variables
├── clients/
│   ├── huggingface.py  ← HuggingFace API (models, README, search)
│   ├── llm.py          ← OpenRouter API (Gemini, summaries, battle)
│   ├── search.py       ← Brave Search API
│   └── web_reader.py   ← web page and arXiv parsing
└── templates/prompts/  ← Jinja2 templates for LLM prompts
```

<br>

### Tech Stack

| Component | Technology |
|:---|:---|
| Telegram Bot | [python-telegram-bot](https://python-telegram-bot.org/) 21.x |
| HTTP Client | [aiohttp](https://docs.aiohttp.org/) 3.x |
| LLM | [OpenRouter](https://openrouter.ai/) → Gemini 2.5 Flash Lite |
| Web Search | [Brave Search API](https://brave.com/search/api/) |
| Templates | [Jinja2](https://jinja.palletsprojects.com/) 3.x |
| Parsing | [trafilatura](https://trafilatura.readthedocs.io/) + [arxiv](https://pypi.org/project/arxiv/) |
| Containerization | Docker + Docker Compose |
| Linter | [Ruff](https://docs.astral.sh/ruff/) |

<br>

---

<div align="center">

**MIT License** · Ivan Mordovets · 2026

</div>
