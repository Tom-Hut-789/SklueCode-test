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
    TokenUsage,
    ToolCall,
)
from ..tools.base import NeutralToolDef
from .base import ProviderAuthError, ProviderNetworkError, ProviderProtocolError


class OpenAIProvider:
    async def stream_chat(
        self,
        config: AppConfig,
        session: SessionRecord,
        tools: list[NeutralToolDef] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.MESSAGE_START)

        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": config.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": _build_openai_messages(session, system_prompt),
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.parameters,
                    },
                }
                for tool_def in tools
            ]

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    await _raise_for_status(response)
                    pending_tool_calls: dict[int, dict[str, str]] = {}
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

                        usage = _extract_usage(data)
                        if usage is not None:
                            yield StreamEvent(type=StreamEventType.TOKEN_USAGE, raw=usage)

                        _accumulate_tool_call_delta(data, pending_tool_calls)

                        if _is_final_tool_call_chunk(data):
                            calls = _build_tool_calls_from_pending(pending_tool_calls)
                            if calls:
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_BATCH,
                                    raw=calls,
                                    is_final=False,
                                )
                            pending_tool_calls = {}
        except httpx.TimeoutException as error:
            raise ProviderNetworkError("OpenAI request timed out.") from error
        except httpx.NetworkError as error:
            raise ProviderNetworkError("OpenAI network request failed.") from error

        yield StreamEvent(type=StreamEventType.MESSAGE_END, is_final=True)


def _build_openai_messages(
    session: SessionRecord,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in session.messages:
        if message.kind == MessageKind.TOOL_RESULT:
            for result in message.tool_results or []:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                )
            continue

        if message.role == MessageRole.ASSISTANT and message.kind == MessageKind.TOOL_CALL:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            tool_calls = []
            for call in message.tool_calls or []:
                tool_calls.append(
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(call.arguments or {}),
                        },
                    }
                )
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            continue

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


def _extract_usage(chunk: dict[str, Any]) -> TokenUsage | None:
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None

    total_tokens = usage.get("total_tokens")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )


def _accumulate_tool_call_delta(chunk: dict[str, Any], pending: dict[int, dict[str, str]]) -> None:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    delta = choices[0].get("delta", {})
    if not isinstance(delta, dict):
        return
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return

    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int):
            continue
        entry = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
        tool_call_id = item.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            entry["id"] = tool_call_id
        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                entry["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry["arguments"] += arguments


def _is_final_tool_call_chunk(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    return choices[0].get("finish_reason") == "tool_calls"


def _build_tool_calls_from_pending(pending: dict[int, dict[str, str]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(pending):
        entry = pending[index]
        tool_call_id = entry["id"] or f"tool-call-{index}"
        name = entry["name"] or ""
        arguments: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            parsed = json.loads(entry["arguments"] or "{}")
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                parse_error = f"Tool arguments must be a JSON object, got {type(parsed).__name__}."
        except json.JSONDecodeError as error:
            parse_error = f"JSON decode error: {error}"
        calls.append(
            ToolCall(
                tool_call_id=tool_call_id,
                tool_name=name,
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return calls


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
