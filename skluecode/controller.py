from __future__ import annotations

from typing import AsyncIterator

from .models import AppConfig, ChatMessage, MessageRole, SessionMode, SessionRecord, StreamEvent, StreamEventType
from .providers.base import Provider, ProviderError
from .session_store.base import SessionStore, SessionStoreError


class ChatController:
    def __init__(
        self,
        config: AppConfig,
        provider: Provider,
        session_store: SessionStore,
    ) -> None:
        self.config = config
        self.provider = provider
        self.session_store = session_store
        self.session: SessionRecord | None = None

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

        assistant_answer = ""
        assistant_thinking = ""
        did_error = False

        try:
            async for event in self.provider.stream_chat(self.config, session):
                if event.type == StreamEventType.ANSWER_DELTA:
                    assistant_answer += event.text
                elif event.type == StreamEventType.THINKING_DELTA:
                    assistant_thinking += event.text
                elif event.type == StreamEventType.ERROR:
                    did_error = True
                yield event
        except (ProviderError, SessionStoreError) as error:
            did_error = True
            yield StreamEvent(type=StreamEventType.ERROR, text=str(error), is_final=True)

        if not did_error and assistant_answer:
            session.add_message(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=assistant_answer,
                    thinking=assistant_thinking or None,
                )
            )

        if self.config.session_mode == SessionMode.PERSISTENT:
            try:
                await self.session_store.save(session)
            except SessionStoreError as error:
                yield StreamEvent(type=StreamEventType.ERROR, text=str(error), is_final=True)

    async def close(self) -> None:
        if self.session and self.config.session_mode == SessionMode.PERSISTENT:
            await self.session_store.save(self.session)
