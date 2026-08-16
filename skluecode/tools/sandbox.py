from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ResolvedPath:
    absolute: Path
    relative_to_root: str | None
    is_outside: bool
    target_display: str


class WorkspaceSandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, raw_path: str) -> ResolvedPath:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root / path
        absolute = path.resolve()

        try:
            relative = str(absolute.relative_to(self.root))
            is_outside = False
        except ValueError:
            relative = None
            is_outside = True

        return ResolvedPath(
            absolute=absolute,
            relative_to_root=relative,
            is_outside=is_outside,
            target_display=str(absolute),
        )
