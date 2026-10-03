from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Static
from textual.worker import Worker

from ..controller import ChatController
from ..models import (
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
from ..tools.confirmation import (
    ConfirmationRequest,
    ConfirmationResponse,
    DelegatingConfirmationBridge,
)
from .clipboard import read_clipboard_text


def _clipboard_single_line() -> str:
    """Read the OS clipboard as a single line (Input is a one-line widget)."""
    text = read_clipboard_text()
    return " ".join(text.splitlines()) if text else ""


class PromptInput(Input):
    """Input that pastes from the OS clipboard.

    ``Input`` binds ``ctrl+v`` to ``action_paste``, but that uses Textual's
    in-app clipboard buffer rather than the OS clipboard. On Windows terminals
    ``Ctrl+Shift+V`` often arrives as plain ``ctrl+v``, so we override the
    action to read the real system clipboard instead.
    """

    def action_paste(self) -> None:
        text = _clipboard_single_line()
        if not text:
            super().action_paste()
            return
        self.replace(text, *self.selection)


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

    #prompt_bar {
        dock: bottom;
        height: auto;
    }

    #prompt {
        width: 1fr;
    }

    #btn_abort {
        width: auto;
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
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+shift+v", "paste_clipboard", "Paste", priority=True),
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
        self._active_worker: Worker[None] | None = None
        self._iteration = 0
        self._max_iterations = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._total_tokens = 0
        self._status_note = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Static("Loading session...", id="status")
            with VerticalScroll(id="transcript_scroll"):
                yield Static("", id="transcript")
            with Horizontal(id="prompt_bar"):
                yield PromptInput(
                    placeholder="Type a message and press Enter. Ctrl+V pastes from the clipboard.",
                    id="prompt",
                )
                yield Button("Cancel", variant="error", id="btn_abort", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._bind_confirmation_gateway()
        self.run_worker(self._initialize(), exclusive=False)

    def _bind_confirmation_gateway(self) -> None:
        scheduler = self.controller.agent.scheduler
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
        self._refresh_status()

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
        if text == "/plan":
            self._switch_mode(AgentMode.PLAN)
            return
        if text == "/do":
            self._switch_mode(AgentMode.NORMAL)
            return
        if text == "/paste":
            self.action_paste_clipboard()
            return

        self._append_user_message(text)
        self._busy = True
        self._iteration = 0
        self._max_iterations = 0
        self._status_note = "Streaming reply..."
        event.input.disabled = True
        self._set_cancel_enabled(True)
        self._refresh_status()
        self._active_worker = self.run_worker(self._handle_message(text), exclusive=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_abort":
            self.action_cancel_session()

    def _switch_mode(self, mode: AgentMode) -> None:
        self.controller.set_mode(mode)
        if mode == AgentMode.PLAN:
            note = (
                "计划模式已开启：我可以读取文件、分析内容、提供建议并给出方案，"
                "但不会修改任何文件。\n输入 /do 返回正常模式开始执行。"
            )
        else:
            note = (
                "正常模式已开启：所有工具均可用，我会按需调用工具并执行修改。\n"
                "输入 /plan 切换到计划模式。"
            )
        self._blocks.append(_format_mode_block(note))
        self._refresh_transcript()
        self._refresh_status("")

    async def _handle_message(self, text: str) -> None:
        prompt = self.query_one("#prompt", Input)
        try:
            async for event in self.controller.send_user_message(text):
                self._apply_stream_event(event)
        finally:
            self._busy = False
            self._active_worker = None
            prompt.disabled = False
            prompt.focus()
            self._set_cancel_enabled(False)
            self._assistant_active = False
            self._assistant_thinking = ""
            self._assistant_answer = ""
            self._assistant_block_index = None
            self._refresh_status("")

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
            self._begin_assistant_block()
        elif event.type == StreamEventType.THINKING_DELTA:
            self._assistant_thinking += event.text
            self._update_live_assistant_block()
        elif event.type == StreamEventType.ANSWER_DELTA:
            self._assistant_answer += event.text
            self._update_live_assistant_block()
        elif event.type == StreamEventType.TOOL_CALL_BATCH:
            self._finalize_pending_assistant_block()
            for call in _as_tool_call_list(event.raw):
                self._blocks.append(_format_tool_call_block(call))
                self._tool_block_index[call.tool_call_id] = len(self._blocks) - 1
            self._assistant_answer = ""
            self._assistant_thinking = ""
            self._refresh_status("Executing tools...")
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
                    self._refresh_status("A tool failed; see the result block.")
                else:
                    self._refresh_status("Tool completed.")
        elif event.type == StreamEventType.TOKEN_USAGE:
            self._record_usage(event.raw)
        elif event.type == StreamEventType.PROGRESS:
            progress = event.raw
            if isinstance(progress, AgentProgress):
                self._iteration = progress.iteration
                self._max_iterations = progress.max_iterations
                if progress.phase == "thinking":
                    self._begin_assistant_block()
                    self._refresh_status("Thinking...")
                else:
                    self._refresh_status("Executing tools...")
        elif event.type == StreamEventType.LOOP_END:
            self._handle_loop_end(event.raw)
        elif event.type == StreamEventType.ERROR:
            self._assistant_active = False
            self._finalize_pending_assistant_block()
            self._blocks.append(_format_error_block(event.text))
            self._refresh_status("Last request failed.")
        elif event.type == StreamEventType.MESSAGE_END:
            self._assistant_active = False
            self._update_live_assistant_block()

        self._refresh_transcript()

    def _record_usage(self, raw: Any) -> None:
        if not isinstance(raw, TokenUsage):
            return
        self._last_input_tokens = raw.input_tokens
        self._last_output_tokens = raw.output_tokens
        if raw.total_tokens is not None:
            self._total_tokens += raw.total_tokens
        else:
            self._total_tokens += raw.input_tokens + raw.output_tokens
        self._refresh_status()

    def _handle_loop_end(self, reason: Any) -> None:
        if reason == StopReason.MAX_ITERATIONS:
            self._blocks.append(
                _format_notice_block(
                    f"本轮循环已结束：已达迭代上限 {self._max_iterations} 轮。可继续输入下一条消息。"
                )
            )
        elif reason == StopReason.USER_CANCELLED:
            self._blocks.append(_format_notice_block("本轮循环已结束：已取消当前循环。"))
        elif reason == StopReason.UNKNOWN_TOOL_REPEATED:
            self._blocks.append(
                _format_notice_block("本轮循环已结束：模型连续请求未知工具，已停止本次循环。")
            )
        # MODEL_DONE ends naturally and PROVIDER_ERROR is already rendered as an
        # ERROR block, so neither needs an extra end-of-loop notice.
        self._refresh_status()

    def _begin_assistant_block(self) -> None:
        """Start a fresh assistant block for a new loop round."""
        self._finalize_pending_assistant_block()
        self._assistant_active = True
        self._assistant_thinking = ""
        self._assistant_answer = ""
        self._assistant_block_index = len(self._blocks)
        self._blocks.append("")

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

    def _refresh_status(self, note: str | None = None) -> None:
        if note is not None:
            self._status_note = note
        mode_label = "PLAN" if self.controller.mode == AgentMode.PLAN else "NORMAL"
        round_text = f"{self._iteration}/{self._max_iterations}" if self._max_iterations else "-"
        last_text = (
            f"{self._last_input_tokens}/{self._last_output_tokens}"
            if (self._last_input_tokens or self._last_output_tokens)
            else "unknown"
        )
        parts = [
            f"Provider: {self.controller.config.protocol.value}",
            f"Mode: {mode_label}",
            f"Round: {round_text}",
            f"Last call: {last_text}",
            f"Session tokens: {self._total_tokens}",
        ]
        if self._status_note:
            parts.append(self._status_note)
        self.query_one("#status", Static).update(Text(" | ".join(parts)))

    def action_paste_clipboard(self) -> None:
        """Insert OS clipboard text into the prompt.

        Reads the clipboard directly from Python so non-ASCII input works even
        when the terminal's raw mode cannot deliver it as keystrokes.
        """
        prompt = self.query_one("#prompt", Input)
        if prompt.disabled:
            return
        text = _clipboard_single_line()
        if not text:
            self._status_note = "Clipboard is empty or unavailable."
            self._refresh_status()
            return
        prompt.focus()
        prompt.insert_text_at_cursor(text)
        self._status_note = ""
        self._refresh_status()

    def action_cancel_session(self) -> None:
        """Abort the running loop. Bound to the in-app Cancel button.

        Ctrl+C is deliberately not bound here so it stays available for copying
        selected text; quitting is Ctrl+Q.
        """
        if not self._busy:
            return
        self._cancel_current_worker()

    def _cancel_current_worker(self) -> None:
        worker = self._active_worker
        if worker is not None:
            worker.cancel()
        self._close_confirmation_modal()

    def _close_confirmation_modal(self) -> bool:
        """Dismiss an open confirmation modal, returning True if one was closed."""
        if isinstance(self.screen, ConfirmationModal):
            self.screen.dismiss(
                ConfirmationResponse(approved=False, reason="Cancelled by the user.")
            )
            return True
        return False

    def _set_cancel_enabled(self, enabled: bool) -> None:
        self.query_one("#btn_abort", Button).disabled = not enabled


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


def _format_notice_block(text: str) -> str:
    return f"[Notice]\n{text}"


def _format_mode_block(text: str) -> str:
    return f"[Mode]\n{text}"
