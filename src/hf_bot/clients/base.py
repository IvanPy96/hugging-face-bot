"""Base class for async HTTP clients with session lifecycle management."""

from __future__ import annotations

import logging
from typing import Any, Self

import aiohttp

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    """Manage one aiohttp.ClientSession with a configurable timeout.

    Usage::

        async with MyClient() as client:
            data = await client.fetch(...)

    Or call close explicitly during application shutdown.
    """

    def __init__(self, *, timeout_seconds: int = 10) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Lazy-initialised session (created on first access)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """Gracefully close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
