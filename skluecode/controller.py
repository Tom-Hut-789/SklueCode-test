from __future__ import annotations

from typing import Any, AsyncIterator

from .models import (
    AppConfig,
    ChatMessage,
    MessageKind,
    MessageRole,
    SessionMode,
    SessionRecord,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from .providers.base import Provider, ProviderError
from .session_store.base import SessionStore, SessionStoreError
from .tools.base import NeutralToolDef
from .tools.registry import ToolRegistry
from .tools.scheduler import ToolScheduler


class ChatController:
    def __init__(
        self,
        config: AppConfig,
        provider: Provider,
        session_store: SessionStore,
        registry: ToolRegistry | None = None,
        scheduler: ToolScheduler | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.session_store = session_store
        self.registry = registry
        self.scheduler = scheduler
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
        tool_round_used = False

        try:
            async for event in self._run_round(session, allow_tools=True):
                if event.type == StreamEventType.TOOL_CALL_BATCH:
                    calls = _as_tool_call_list(event.raw)
                    if not calls:
                        continue

                    yield event  # 让 TUI 渲染工具调用块（running 状态）

                    session.add_message(
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            kind=MessageKind.TOOL_CALL,
                            content=assistant_answer or "",
                            thinking=assistant_thinking or None,
                            tool_calls=calls,
                        )
                    )
                    assistant_answer = ""
                    assistant_thinking = ""

                    results = await self._execute_tools(calls)
                    session.add_message(
                        ChatMessage(
                            role=MessageRole.USER,
                            kind=MessageKind.TOOL_RESULT,
                            content="",
                            tool_results=results,
                        )
                    )
                    for result in results:
                        yield StreamEvent(type=StreamEventType.TOOL_RESULT_READY, raw=result)

                    if tool_round_used:
                        did_error = True
                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            text=(
                                "This version does not support automatic multi-round agent loops. "
                                "The tool result was delivered once; ask for a follow-up as a new message."
                            ),
                            is_final=True,
                        )
                        break
                    tool_round_used = True

                    # 第二次调用 Provider：不允许再要工具，只取最终文字回答。
                    async for event2 in self._run_round(session, allow_tools=False):
                        if event2.type == StreamEventType.TOOL_CALL_BATCH:
                            did_error = True
                            yield StreamEvent(
                                type=StreamEventType.ERROR,
                                text=(
                                    "The model requested tools again after receiving results, "
                                    "which exceeds this version's single tool round. Ending the turn."
                                ),
                                is_final=True,
                            )
                            break
                        if event2.type == StreamEventType.ANSWER_DELTA:
                            assistant_answer += event2.text
                        elif event2.type == StreamEventType.THINKING_DELTA:
                            assistant_thinking += event2.text
                        elif event2.type == StreamEventType.ERROR:
                            did_error = True
                        yield event2
                    break

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

    async def _run_round(
        self,
        session: SessionRecord,
        allow_tools: bool,
    ) -> AsyncIterator[StreamEvent]:
        tools: list[NeutralToolDef] | None = None
        if allow_tools and self.registry is not None:
            tools = self.registry.to_neutral_definitions()
        async for event in self.provider.stream_chat(self.config, session, tools=tools):
            yield event

    async def _execute_tools(self, calls: list[ToolCall]) -> Any:
        if self.scheduler is None:
            return [
                _tool_result_for(
                    call,
                    is_error=True,
                    error_type="permission_denied",
                    content="Tool system is not initialized in this session.",
                )
                for call in calls
            ]
        return await self.scheduler.schedule(calls)

    async def close(self) -> None:
        if self.session and self.config.session_mode == SessionMode.PERSISTENT:
            await self.session_store.save(self.session)


def _as_tool_call_list(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, ToolCall)]


def _tool_result_for(
    call: ToolCall,
    is_error: bool,
    error_type: str | None,
    content: str,
) -> Any:
    from .models import ToolResult

    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        is_error=is_error,
        error_type=error_type,
        content=content,
    )
