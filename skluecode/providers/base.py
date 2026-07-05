from __future__ import annotations

from typing import AsyncIterator, Protocol

from ..models import AppConfig, SessionRecord, StreamEvent


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderConfigError(ProviderError):
    """Raised when a provider receives unsupported configuration."""


class ProviderAuthError(ProviderError):
    """Raised when authentication fails."""


class ProviderNetworkError(ProviderError):
    """Raised when the upstream request fails due to network conditions."""


class ProviderProtocolError(ProviderError):
    """Raised when the provider response shape is not supported."""


class Provider(Protocol):
    async def stream_chat(
        self,
        config: AppConfig,
        session: SessionRecord,
    ) -> AsyncIterator[StreamEvent]:
        ...
