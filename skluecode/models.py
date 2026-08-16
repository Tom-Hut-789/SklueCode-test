from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProtocolType(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class SessionMode(StrEnum):
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageKind(StrEnum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class StreamEventType(StrEnum):
    MESSAGE_START = "message_start"
    THINKING_DELTA = "thinking_delta"
    ANSWER_DELTA = "answer_delta"
    MESSAGE_END = "message_end"
    ERROR = "error"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_BATCH = "tool_call_batch"
    TOOL_RESULT_READY = "tool_result_ready"


@dataclass(slots=True)
class ToolCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolCall":
        return cls(
            tool_call_id=payload["tool_call_id"],
            tool_name=payload["tool_name"],
            arguments=payload.get("arguments"),
            parse_error=payload.get("parse_error"),
        )


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    is_error: bool
    content: str
    error_type: str | None = None
    truncated: bool = False
    target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "is_error": self.is_error,
            "error_type": self.error_type,
            "content": self.content,
            "truncated": self.truncated,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolResult":
        return cls(
            tool_call_id=payload["tool_call_id"],
            tool_name=payload["tool_name"],
            is_error=bool(payload.get("is_error", False)),
            error_type=payload.get("error_type"),
            content=payload.get("content", ""),
            truncated=bool(payload.get("truncated", False)),
            target=payload.get("target"),
        )


@dataclass(slots=True)
class AppConfig:
    protocol: ProtocolType
    model: str
    base_url: str
    api_key: str
    session_mode: SessionMode = SessionMode.EPHEMERAL
    enable_extended_thinking: bool = False
    thinking_budget_tokens: int | None = 1024
    storage_path: str | None = "data/sessions"


@dataclass(slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    kind: MessageKind = MessageKind.TEXT
    created_at: datetime = field(default_factory=utc_now)
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "kind": self.kind.value,
            "created_at": self.created_at.isoformat(),
            "thinking": self.thinking,
            "tool_calls": [call.to_dict() for call in self.tool_calls] if self.tool_calls else None,
            "tool_results": [result.to_dict() for result in self.tool_results] if self.tool_results else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        raw_calls = payload.get("tool_calls")
        raw_results = payload.get("tool_results")
        return cls(
            role=MessageRole(payload["role"]),
            content=payload.get("content", ""),
            kind=MessageKind(payload.get("kind", MessageKind.TEXT.value)),
            created_at=datetime.fromisoformat(payload["created_at"]),
            thinking=payload.get("thinking"),
            tool_calls=[ToolCall.from_dict(item) for item in raw_calls] if raw_calls else None,
            tool_results=[ToolResult.from_dict(item) for item in raw_results] if raw_results else None,
        )


@dataclass(slots=True)
class SessionRecord:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    mode: SessionMode = SessionMode.EPHEMERAL
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "messages": [message.to_dict() for message in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=payload["session_id"],
            mode=SessionMode(payload["mode"]),
            messages=[ChatMessage.from_dict(item) for item in payload.get("messages", [])],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
        )


@dataclass(slots=True)
class StreamEvent:
    type: StreamEventType
    text: str = ""
    raw: Any | None = None
    is_final: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload
