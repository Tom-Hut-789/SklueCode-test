from __future__ import annotations

from typing import AsyncIterator

from .agent import AgentRunner
from .models import (
    AgentMode,
    AppConfig,
    ChatMessage,
    MessageRole,
    SessionMode,
    SessionRecord,
    StreamEvent,
    StreamEventType,
)
from .session_store.base import SessionStore, SessionStoreError


class ChatController:
    def __init__(
        self,
        config: AppConfig,
        session_store: SessionStore,
        agent: AgentRunner,
    ) -> None:
        self.config = config
        self.session_store = session_store
        self.agent = agent
        self.session: SessionRecord | None = None
        self._mode: AgentMode = AgentMode.NORMAL

    @property
    def mode(self) -> AgentMode:
        return self._mode

    def set_mode(self, mode: AgentMode) -> None:
        self._mode = mode

    async def load_or_create_session(self) -> SessionRecord:
        if self.session is not None:
            return self.session

        restored: SessionRecord | None = None
        if self.config.session_mode == SessionMode.PERSISTENT:
            restored = await self.session_store.load_latest()

        self.session = restored or SessionRecord(mode=self.config.session_mode)
        return self.session

    async def send_user_message(self, text: str) -> AsyncIterator[StreamEvent]:
        session = await self.load_or_create_session()
        session.add_message(ChatMessage(role=MessageRole.USER, content=text))

        yield StreamEvent(type=StreamEventType.MESSAGE_START)

        save_error: str | None = None
        try:
            async for event in self.agent.run(session, self._mode):
                yield event
        finally:
            if self.config.session_mode == SessionMode.PERSISTENT:
                try:
                    await self.session_store.save(session)
                except SessionStoreError as error:
                    save_error = str(error)

        if save_error is not None:
            yield StreamEvent(type=StreamEventType.ERROR, text=save_error, is_final=True)
        yield StreamEvent(type=StreamEventType.MESSAGE_END, is_final=True)

    async def close(self) -> None:
        if self.session and self.config.session_mode == SessionMode.PERSISTENT:
            await self.session_store.save(self.session)
