"""HF Monitor Bot — Telegram bot for monitoring new AI models on Hugging Face."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hf-monitor-bot")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
