from __future__ import annotations

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from ..controller import ChatController
from ..models import ChatMessage, MessageRole, SessionRecord, StreamEvent, StreamEventType


class ChatApp(App[None]):
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
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, controller: ChatController) -> None:
        super().__init__()
        self.controller = controller
        self._blocks: list[str] = []
        self._assistant_active = False
        self._assistant_thinking = ""
        self._assistant_answer = ""
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
        self.run_worker(self._initialize(), exclusive=False)

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
            session = await self.controller.load_or_create_session()
            self._set_status(
                f"Provider: {self.controller.config.protocol.value} | "
                f"Mode: {session.mode.value} | Session: {session.session_id}"
            )

    def _render_history(self, session: SessionRecord) -> None:
        self._blocks = []
        for message in session.messages:
            if message.role == MessageRole.USER:
                self._blocks.append(_format_user_block(message.content))
            elif message.role == MessageRole.ASSISTANT:
                self._blocks.append(_format_assistant_block(message))
        self._refresh_transcript()

    def _append_user_message(self, text: str) -> None:
        self._blocks.append(_format_user_block(text))
        self._refresh_transcript()

    def _apply_stream_event(self, event: StreamEvent) -> None:
        if event.type == StreamEventType.MESSAGE_START:
            self._assistant_active = True
            self._assistant_thinking = ""
            self._assistant_answer = ""
            self._blocks.append("")
        elif event.type == StreamEventType.THINKING_DELTA:
            self._assistant_thinking += event.text
        elif event.type == StreamEventType.ANSWER_DELTA:
            self._assistant_answer += event.text
        elif event.type == StreamEventType.ERROR:
            self._assistant_active = False
            self._blocks.append(_format_error_block(event.text))
            self._set_status("Last request failed.")
        elif event.type == StreamEventType.MESSAGE_END:
            self._assistant_active = False

        if self._assistant_active and self._blocks:
            self._blocks[-1] = _format_live_assistant_block(
                self._assistant_thinking,
                self._assistant_answer,
            )
        elif event.type == StreamEventType.MESSAGE_END and self._blocks:
            self._blocks[-1] = _format_live_assistant_block(
                self._assistant_thinking,
                self._assistant_answer,
            )

        self._refresh_transcript()

    def _refresh_transcript(self) -> None:
        transcript = self.query_one("#transcript", Static)
        scroll = self.query_one("#transcript_scroll", VerticalScroll)
        transcript.update(Text("\n\n".join(block for block in self._blocks if block)))
        scroll.scroll_end(animate=False)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(Text(text))


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


def _format_error_block(text: str) -> str:
    return f"[Error]\n{text}"
