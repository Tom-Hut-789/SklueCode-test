from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..models import (
    AppConfig,
    MessageKind,
    MessageRole,
    SessionRecord,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from ..tools.base import NeutralToolDef
from .base import (
    ProviderAuthError,
    ProviderConfigError,
    ProviderNetworkError,
    ProviderProtocolError,
)


class AnthropicProvider:
    async def stream_chat(
        self,
        config: AppConfig,
        session: SessionRecord,
        tools: list[NeutralToolDef] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if config.enable_extended_thinking and not config.thinking_budget_tokens:
            raise ProviderConfigError("Anthropic extended thinking requires thinking_budget_tokens.")

        yield StreamEvent(type=StreamEventType.MESSAGE_START)

        url = f"{config.base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": config.model,
            "stream": True,
            "max_tokens": 4096,
            "messages": _build_anthropic_messages(session),
        }
        if config.enable_extended_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget_tokens,
            }
        if tools:
            payload["tools"] = [
                {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "input_schema": tool_def.parameters,
                }
                for tool_def in tools
            ]

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    await _raise_for_status(response)
                    parser = _AnthropicStreamParser()
                    async for event_name, event_data in _iter_sse_events(response):
                        if event_name == "error":
                            raise ProviderProtocolError(_extract_anthropic_error(event_data))
                        for stream_event in parser.handle(event_name, event_data):
                            yield stream_event
        except httpx.TimeoutException as error:
            raise ProviderNetworkError("Anthropic request timed out.") from error
        except httpx.NetworkError as error:
            raise ProviderNetworkError("Anthropic network request failed.") from error

        yield StreamEvent(type=StreamEventType.MESSAGE_END, is_final=True)


def _build_anthropic_messages(session: SessionRecord) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in session.messages:
        if message.role == MessageRole.SYSTEM:
            continue

        if message.kind == MessageKind.TOOL_RESULT:
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                }
                for result in (message.tool_results or [])
            ]
            if not blocks:
                continue
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": "user", "content": blocks})
            continue

        if message.kind == MessageKind.TOOL_CALL:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.tool_call_id,
                        "name": call.tool_name,
                        "input": call.arguments or {},
                    }
                )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue

        if message.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
            continue
        text_block = [{"type": "text", "text": message.content}]
        if messages and messages[-1]["role"] == message.role.value and message.role == MessageRole.USER:
            messages[-1]["content"].extend(text_block)
        else:
            messages.append({"role": message.role.value, "content": text_block})
    return messages


class _AnthropicStreamParser:
    """Stateful mapper from Anthropic SSE events to unified StreamEvents."""

    def __init__(self) -> None:
        self._pending_tool_uses: dict[int, dict[str, str]] = {}
        self._tool_calls: list[ToolCall] | None = None
        self._next_id = 0

    def handle(self, event_name: str, payload: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if event_name == "content_block_start":
            index = payload.get("index")
            block = payload.get("content_block", {})
            if block.get("type") == "tool_use" and isinstance(index, int):
                self._pending_tool_uses[index] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "args_acc": "",
                }
        elif event_name == "content_block_delta":
            index = payload.get("index")
            delta = payload.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                if isinstance(text, str) and text:
                    events.append(StreamEvent(StreamEventType.ANSWER_DELTA, text=text, raw=payload))
            elif delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                if isinstance(text, str) and text:
                    events.append(StreamEvent(StreamEventType.THINKING_DELTA, text=text, raw=payload))
            elif delta_type == "input_json_delta":
                if isinstance(index, int) and index in self._pending_tool_uses:
                    partial = delta.get("partial_json", "")
                    if isinstance(partial, str):
                        self._pending_tool_uses[index]["args_acc"] += partial
        elif event_name == "content_block_stop":
            index = payload.get("index")
            if isinstance(index, int) and index in self._pending_tool_uses:
                entry = self._pending_tool_uses.pop(index)
                self._finalize_tool_use(entry)
        elif event_name == "message_stop":
            if self._tool_calls:
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_BATCH,
                        raw=self._tool_calls,
                        is_final=False,
                    )
                )

        return events

    def _finalize_tool_use(self, entry: dict[str, str]) -> None:
        arguments: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            parsed = json.loads(entry["args_acc"] or "{}")
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                parse_error = f"Tool arguments must be a JSON object, got {type(parsed).__name__}."
        except json.JSONDecodeError as error:
            parse_error = f"JSON decode error: {error}"

        call = ToolCall(
            tool_call_id=entry["id"] or f"tool-use-{self._next_id}",
            tool_name=entry["name"] or "",
            arguments=arguments,
            parse_error=parse_error,
        )
        self._next_id += 1
        if self._tool_calls is None:
            self._tool_calls = []
        self._tool_calls.append(call)


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_name = "message"
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line.startswith("event: "):
            event_name = line[7:].strip()
            continue
        if line.startswith("data: "):
            data_lines.append(line[6:])
            continue
        if line == "":
            if data_lines:
                raw_payload = "\n".join(data_lines)
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError as error:
                    raise ProviderProtocolError("Anthropic stream returned invalid JSON.") from error
                yield event_name, payload
            event_name = "message"
            data_lines = []

    if data_lines:
        raw_payload = "\n".join(data_lines)
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as error:
            raise ProviderProtocolError("Anthropic stream returned invalid JSON.") from error
        yield event_name, payload


async def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    message = await _read_error_message(response)
    if response.status_code in {401, 403}:
        raise ProviderAuthError(message)
    raise ProviderProtocolError(message)


async def _read_error_message(response: httpx.Response) -> str:
    try:
        data = await response.aread()
    except httpx.HTTPError:
        data = b""

    if not data:
        return f"Anthropic request failed with status {response.status_code}."

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return f"Anthropic request failed with status {response.status_code}."

    return _extract_anthropic_error(payload)


def _extract_anthropic_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return "Anthropic request failed."
