from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator

from .models import (
    AgentMode,
    AgentProgress,
    ChatMessage,
    MessageKind,
    MessageRole,
    SessionRecord,
    StopReason,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from .providers.base import Provider
from .tools.base import NeutralToolDef
from .tools.registry import ToolRegistry
from .tools.scheduler import ToolScheduler

READ_ONLY_TOOL_NAMES: tuple[str, ...] = ("read_file", "find_files", "search_code")

PLAN_SYSTEM_PROMPT = (
    "You are running in PLAN MODE. You may only read files, search code, and inspect "
    "the project with the read-only tools available to you. Do not attempt to modify "
    "files or run commands. Analyze what you find and reply with a concrete plan or "
    "recommendation for the user. The user will switch back to normal mode with /do "
    "when they want you to execute the plan."
)


@dataclass(slots=True)
class CollectedTurn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    had_error: bool = False
    error_message: str | None = None


class StreamCollector:
    """Splits a provider stream into two paths: real-time forwarding and accumulation."""

    def __init__(self) -> None:
        self.text = ""
        self.thinking = ""
        self.tool_calls: list[ToolCall] = []
        self.usage: TokenUsage | None = None
        self.had_error = False
        self.error_message: str | None = None

    async def collect(self, source: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
        async for event in source:
            self.consume(event)
            yield event

    def consume(self, event: StreamEvent) -> None:
        if event.type == StreamEventType.ANSWER_DELTA:
            self.text += event.text
        elif event.type == StreamEventType.THINKING_DELTA:
            self.thinking += event.text
        elif event.type == StreamEventType.TOOL_CALL_BATCH:
            calls = _as_tool_call_list(event.raw)
            if calls:
                self.tool_calls.extend(calls)
        elif event.type == StreamEventType.TOKEN_USAGE:
            if isinstance(event.raw, TokenUsage):
                self.usage = event.raw
        elif event.type == StreamEventType.ERROR:
            self.had_error = True
            if event.text:
                self.error_message = event.text

    def finish(self) -> CollectedTurn:
        return CollectedTurn(
            text=self.text,
            thinking=self.thinking,
            tool_calls=list(self.tool_calls),
            usage=self.usage,
            had_error=self.had_error,
            error_message=self.error_message,
        )


class AgentRunner:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        scheduler: ToolScheduler,
        config: Any,
        max_iterations: int = 10,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.scheduler = scheduler
        self.config = config
        self.max_iterations = max_iterations

    async def run(
        self,
        session: SessionRecord,
        mode: AgentMode = AgentMode.NORMAL,
    ) -> AsyncIterator[StreamEvent]:
        unknown_streak = 0

        for iteration in range(1, self.max_iterations + 1):
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                raw=AgentProgress(
                    iteration=iteration,
                    max_iterations=self.max_iterations,
                    phase="thinking",
                ),
            )

            tools = self._tools_for(mode)
            system_prompt = PLAN_SYSTEM_PROMPT if mode == AgentMode.PLAN else None
            collector = StreamCollector()

            try:
                stream = self.provider.stream_chat(
                    self.config,
                    session,
                    tools=tools,
                    system_prompt=system_prompt,
                )
                async for event in collector.collect(stream):
                    yield event
            except asyncio.CancelledError:
                yield StreamEvent(type=StreamEventType.LOOP_END, raw=StopReason.USER_CANCELLED)
                return
            except Exception as error:  # noqa: BLE001 - the loop must contain any provider failure
                yield StreamEvent(type=StreamEventType.ERROR, text=str(error), is_final=True)
                yield StreamEvent(type=StreamEventType.LOOP_END, raw=StopReason.PROVIDER_ERROR)
                return

            turn = collector.finish()

            if turn.had_error:
                # A failed stream can leave partial tool calls; persist only the text
                # so the history never ends with an unanswered tool_calls message.
                self._append_assistant(session, replace(turn, tool_calls=[]))
                yield StreamEvent(type=StreamEventType.LOOP_END, raw=StopReason.PROVIDER_ERROR)
                return

            self._append_assistant(session, turn)

            if not turn.tool_calls:
                yield StreamEvent(type=StreamEventType.LOOP_END, raw=StopReason.MODEL_DONE)
                return

            if all(self.registry.get(call.tool_name) is None for call in turn.tool_calls):
                unknown_streak += 1
            else:
                unknown_streak = 0
            if unknown_streak >= 2:
                self._append_not_executed(session, turn.tool_calls, "Tool is not available.")
                yield StreamEvent(
                    type=StreamEventType.LOOP_END,
                    raw=StopReason.UNKNOWN_TOOL_REPEATED,
                )
                return

            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                raw=AgentProgress(
                    iteration=iteration,
                    max_iterations=self.max_iterations,
                    phase="executing",
                ),
            )

            try:
                results = await self.scheduler.schedule(turn.tool_calls)
            except asyncio.CancelledError:
                # Cancelled mid-execution: record placeholder results so the persisted
                # history stays a valid tool_calls/tool pairing across restarts.
                self._append_not_executed(
                    session, turn.tool_calls, "Cancelled before the tool finished."
                )
                yield StreamEvent(type=StreamEventType.LOOP_END, raw=StopReason.USER_CANCELLED)
                return

            self._append_tool_results(session, results)
            for result in results:
                yield StreamEvent(type=StreamEventType.TOOL_RESULT_READY, raw=result)

        yield StreamEvent(type=StreamEventType.LOOP_END, raw=StopReason.MAX_ITERATIONS)

    def _tools_for(self, mode: AgentMode) -> list[NeutralToolDef] | None:
        if mode == AgentMode.PLAN:
            return self.registry.to_neutral_definitions(READ_ONLY_TOOL_NAMES)
        return self.registry.to_neutral_definitions()

    def _append_assistant(self, session: SessionRecord, turn: CollectedTurn) -> None:
        if not turn.text and not turn.tool_calls:
            return
        session.add_message(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.TOOL_CALL if turn.tool_calls else MessageKind.TEXT,
                content=turn.text,
                thinking=turn.thinking or None,
                tool_calls=turn.tool_calls or None,
            )
        )

    def _append_tool_results(self, session: SessionRecord, results: list[ToolResult]) -> None:
        session.add_message(
            ChatMessage(
                role=MessageRole.USER,
                kind=MessageKind.TOOL_RESULT,
                content="",
                tool_results=list(results),
            )
        )

    def _append_not_executed(
        self,
        session: SessionRecord,
        calls: list[ToolCall],
        reason: str,
    ) -> None:
        """Record placeholder results so tool_calls never dangle in the history."""
        self._append_tool_results(
            session,
            [
                ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    is_error=True,
                    error_type="not_executed",
                    content=reason,
                )
                for call in calls
            ],
        )


def _as_tool_call_list(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, ToolCall)]
