from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from ..controller import ChatController
from ..models import (
    ChatMessage,
    MessageKind,
    MessageRole,
    SessionRecord,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)
from ..tools.confirmation import (
    ConfirmationRequest,
    ConfirmationResponse,
    DelegatingConfirmationBridge,
)


class ConfirmationModal(ModalScreen[ConfirmationResponse]):
    """Modal that asks the user to approve a risky tool operation."""

    def __init__(self, request: ConfirmationRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        detail_widget: Any = Label("", id="modal_detail")
        if self.request.detail:
            detail_widget = Label(self.request.detail, id="modal_detail")
        yield Vertical(
            Label(f"[b]{self.request.title}[/b]", id="modal_title"),
            Label(self.request.description, id="modal_description"),
            Label(f"Target: {self.request.target}", id="modal_target"),
            detail_widget,
            Horizontal(
                Button("Approve", variant="primary", id="btn_approve"),
                Button("Cancel", variant="error", id="btn_cancel"),
                id="modal_buttons",
            ),
            id="modal_box",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_approve":
            self.dismiss(ConfirmationResponse(approved=True))
        else:
            self.dismiss(ConfirmationResponse(approved=False, reason="User cancelled the operation."))


class ChatApp(App[None]):
    TITLE = "SklueCode"
    SUB_TITLE = "AI Coding Agent"
    ICON = "🤖"

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #status {
        height: auto;
        padding: 0 1;
    }

    #transcript_scroll {
        height: 1fr;
        border: round $accent;
    }

    #transcript {
        padding: 1;
    }

    #prompt {
        dock: bottom;
    }

    ConfirmationModal {
        align: center middle;
    }

    #modal_box {
        width: 70%;
        height: auto;
        border: thick $error;
        padding: 1 2;
        background: $surface;
    }

    #modal_title {
        text-style: bold;
        margin-bottom: 1;
    }

    #modal_description {
        margin-bottom: 1;
    }

    #modal_target {
        color: $warning;
        margin-bottom: 1;
    }

    #modal_buttons {
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, controller: ChatController) -> None:
        super().__init__()
        self.controller = controller
        self._blocks: list[str] = []
        self._tool_block_index: dict[str, int] = {}
        self._assistant_active = False
        self._assistant_thinking = ""
        self._assistant_answer = ""
        self._assistant_block_index: int | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Static("Loading session...", id="status")
            with VerticalScroll(id="transcript_scroll"):
                yield Static("", id="transcript")
            yield Input(placeholder="Type a message and press Enter. Use /quit to exit.", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._bind_confirmation_gateway()
        self.run_worker(self._initialize(), exclusive=False)

    def _bind_confirmation_gateway(self) -> None:
        scheduler = self.controller.scheduler
        if scheduler is None:
            return
        bridge = getattr(scheduler, "request_confirmation", None)
        if isinstance(bridge, DelegatingConfirmationBridge):
            bridge.callback = self.confirm

    async def confirm(self, request: ConfirmationRequest) -> ConfirmationResponse:
        return await self.push_screen_wait(ConfirmationModal(request))

    async def _initialize(self) -> None:
        session = await self.controller.load_or_create_session()
        self._render_history(session)
        self._set_status(
            f"Provider: {self.controller.config.protocol.value} | "
            f"Mode: {session.mode.value} | Session: {session.session_id}"
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._busy:
            return

        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text in {"/quit", "/exit"}:
            self.exit()
            return

        self._append_user_message(text)
        self._busy = True
        event.input.disabled = True
        self._set_status("Streaming reply...")
        self.run_worker(self._handle_message(text), exclusive=True)

    async def _handle_message(self, text: str) -> None:
        prompt = self.query_one("#prompt", Input)
        try:
            async for event in self.controller.send_user_message(text):
                self._apply_stream_event(event)
        finally:
            self._busy = False
            prompt.disabled = False
            prompt.focus()
            self._assistant_active = False
            self._assistant_thinking = ""
            self._assistant_answer = ""
            self._assistant_block_index = None
            session = await self.controller.load_or_create_session()
            self._set_status(
                f"Provider: {self.controller.config.protocol.value} | "
                f"Mode: {session.mode.value} | Session: {session.session_id}"
            )

    def _render_history(self, session: SessionRecord) -> None:
        self._blocks = []
        self._tool_block_index = {}
        self._assistant_block_index = None
        for message in session.messages:
            if message.role == MessageRole.USER and message.kind == MessageKind.TOOL_RESULT:
                for result in message.tool_results or []:
                    self._blocks.append(_format_tool_result_block(result))
            elif message.kind == MessageKind.TOOL_CALL:
                for call in message.tool_calls or []:
                    self._blocks.append(_format_tool_call_block(call))
            elif message.role == MessageRole.USER:
                self._blocks.append(_format_user_block(message.content))
            elif message.role == MessageRole.ASSISTANT:
                self._blocks.append(_format_assistant_block(message))
        self._refresh_transcript()

    def _append_user_message(self, text: str) -> None:
        self._blocks.append(_format_user_block(text))
        self._refresh_transcript()

    def _apply_stream_event(self, event: StreamEvent) -> None:
        if event.type == StreamEventType.MESSAGE_START:
            self._finalize_pending_assistant_block()
            self._assistant_active = True
            self._assistant_thinking = ""
            self._assistant_answer = ""
            self._assistant_block_index = len(self._blocks)
            self._blocks.append("")
        elif event.type == StreamEventType.THINKING_DELTA:
            self._assistant_thinking += event.text
            self._update_live_assistant_block()
        elif event.type == StreamEventType.ANSWER_DELTA:
            self._assistant_answer += event.text
            self._update_live_assistant_block()
        elif event.type == StreamEventType.TOOL_CALL_BATCH:
            for call in _as_tool_call_list(event.raw):
                self._finalize_pending_assistant_block()
                self._blocks.append(_format_tool_call_block(call))
                self._tool_block_index[call.tool_call_id] = len(self._blocks) - 1
            self._set_status("Executing tools...")
        elif event.type == StreamEventType.TOOL_RESULT_READY:
            result = event.raw
            if isinstance(result, ToolResult):
                block = _format_tool_result_block(result)
                index = self._tool_block_index.get(result.tool_call_id)
                if index is not None:
                    self._blocks[index] = block
                else:
                    self._blocks.append(block)
                if result.is_error:
                    self._set_status("A tool failed; see the result block.")
                else:
                    self._set_status("Tool completed.")
        elif event.type == StreamEventType.ERROR:
            self._assistant_active = False
            self._finalize_pending_assistant_block()
            self._blocks.append(_format_error_block(event.text))
            self._set_status("Last request failed.")
        elif event.type == StreamEventType.MESSAGE_END:
            self._assistant_active = False
            self._update_live_assistant_block()

        self._refresh_transcript()

    def _update_live_assistant_block(self) -> None:
        """Update the assistant block only if we are tracking an in-flight one."""
        if self._assistant_block_index is None:
            return
        index = self._assistant_block_index
        if 0 <= index < len(self._blocks):
            self._blocks[index] = _format_live_assistant_block(
                self._assistant_thinking,
                self._assistant_answer,
            )

    def _finalize_pending_assistant_block(self) -> None:
        """Stop tracking the current assistant block. Drop it when it is empty so
        a stray 'Assistant:' header is never rendered between a tool round."""
        if self._assistant_block_index is None:
            return
        index = self._assistant_block_index
        self._assistant_block_index = None
        if 0 <= index < len(self._blocks) and not self._blocks[index].strip():
            self._blocks.pop(index)

    def _refresh_transcript(self) -> None:
        transcript = self.query_one("#transcript", Static)
        scroll = self.query_one("#transcript_scroll", VerticalScroll)
        transcript.update(Text("\n\n".join(block for block in self._blocks if block)))
        scroll.scroll_end(animate=False)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(Text(text))


def _as_tool_call_list(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, ToolCall)]


def _format_user_block(text: str) -> str:
    return f"You:\n{text}"


def _format_assistant_block(message: ChatMessage) -> str:
    return _format_live_assistant_block(message.thinking or "", message.content)


def _format_live_assistant_block(thinking: str, answer: str) -> str:
    sections: list[str] = ["Assistant:"]
    if thinking:
        sections.append(f"[Thinking]\n{thinking}")
    if answer:
        sections.append(f"[Answer]\n{answer}")
    return "\n".join(sections)


def _format_tool_call_block(call: ToolCall) -> str:
    try:
        args_preview = json.dumps(call.arguments or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        args_preview = str(call.arguments)
    if call.parse_error:
        args_preview = f"{args_preview}  [parse error: {call.parse_error}]"
    return f"[Tool] {call.tool_name} {args_preview}\nStatus: running..."


def _format_tool_result_block(result: ToolResult) -> str:
    preview = result.content
    if len(preview) > 2000:
        preview = f"{preview[:2000]}\n[truncated preview; full result was {len(result.content)} chars]"
    if result.is_error:
        error_type = result.error_type or "unknown"
        return (
            f"[Tool result] {result.tool_name} [failed: {error_type}]\n{preview}\n"
            f"建议：修正参数重试 / 改用其他工具 / 手动操作"
        )
    return f"[Tool result] {result.tool_name} [ok]\n{preview}"


def _format_error_block(text: str) -> str:
    return f"[Error]\n{text}"
