from __future__ import annotations

import json
from pathlib import Path

from ..models import SessionRecord
from .base import SessionStoreError


class FileSessionStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.storage_dir / "latest.json"

    async def load_latest(self) -> SessionRecord | None:
        if not self.latest_path.exists():
            return None

        try:
            payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return SessionRecord.from_dict(payload)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
            raise SessionStoreError(f"Failed to load session from {self.latest_path}") from error

    async def save(self, session: SessionRecord) -> None:
        try:
            self.latest_path.write_text(
                json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            raise SessionStoreError(f"Failed to save session to {self.latest_path}") from error
