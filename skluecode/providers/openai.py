from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..models import AppConfig, MessageRole, SessionRecord, StreamEvent, StreamEventType
from .base import ProviderAuthError, ProviderNetworkError, ProviderProtocolError


class OpenAIProvider:
    async def stream_chat(
        self,
        config: AppConfig,
        session: SessionRecord,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.MESSAGE_START)

        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.model,
            "stream": True,
            "messages": _build_openai_messages(session),
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    await _raise_for_status(response)
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break

                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError as error:
                            raise ProviderProtocolError("OpenAI stream returned invalid JSON.") from error

                        text = _extract_text_delta(data)
                        if text:
                            yield StreamEvent(
                                type=StreamEventType.ANSWER_DELTA,
                                text=text,
                                raw=data,
                            )
        except httpx.TimeoutException as error:
            raise ProviderNetworkError("OpenAI request timed out.") from error
        except httpx.NetworkError as error:
            raise ProviderNetworkError("OpenAI network request failed.") from error

        yield StreamEvent(type=StreamEventType.MESSAGE_END, is_final=True)


def _build_openai_messages(session: SessionRecord) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in session.messages:
        if message.role not in {MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT}:
            continue
        messages.append({"role": message.role.value, "content": message.content})
    return messages


def _extract_text_delta(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    delta = choices[0].get("delta", {})
    content = delta.get("content", "")
    return content if isinstance(content, str) else ""


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
        return f"OpenAI request failed with status {response.status_code}."

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return f"OpenAI request failed with status {response.status_code}."

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message

    return f"OpenAI request failed with status {response.status_code}."
