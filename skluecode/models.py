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


class StreamEventType(StrEnum):
    MESSAGE_START = "message_start"
    THINKING_DELTA = "thinking_delta"
    ANSWER_DELTA = "answer_delta"
    MESSAGE_END = "message_end"
    ERROR = "error"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "kind": self.kind.value,
            "created_at": self.created_at.isoformat(),
            "thinking": self.thinking,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=MessageRole(payload["role"]),
            content=payload["content"],
            kind=MessageKind(payload.get("kind", MessageKind.TEXT.value)),
            created_at=datetime.fromisoformat(payload["created_at"]),
            thinking=payload.get("thinking"),
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
    raw: dict[str, Any] | None = None
    is_final: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload
