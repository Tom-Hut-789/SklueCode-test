from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..models import AppConfig, MessageRole, SessionRecord, StreamEvent, StreamEventType
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

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    await _raise_for_status(response)
                    async for event_name, event_data in _iter_sse_events(response):
                        if event_name == "error":
                            raise ProviderProtocolError(_extract_anthropic_error(event_data))

                        stream_event = _map_event(event_name, event_data)
                        if stream_event is not None:
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
        if message.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
            continue
        messages.append(
            {
                "role": message.role.value,
                "content": [{"type": "text", "text": message.content}],
            }
        )
    return messages


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


def _map_event(event_name: str, payload: dict[str, Any]) -> StreamEvent | None:
    if event_name == "content_block_delta":
        delta = payload.get("delta", {})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text", "")
            if isinstance(text, str) and text:
                return StreamEvent(StreamEventType.ANSWER_DELTA, text=text, raw=payload)
        if delta_type == "thinking_delta":
            text = delta.get("thinking", "")
            if isinstance(text, str) and text:
                return StreamEvent(StreamEventType.THINKING_DELTA, text=text, raw=payload)
    return None


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
