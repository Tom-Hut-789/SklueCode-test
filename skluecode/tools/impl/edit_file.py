from __future__ import annotations

from typing import Any

from ..base import (
    ToolExecutionContext,
    ToolExecutionError,
    ToolResultContent,
    ToolSchemaError,
    ToolTargetError,
)


class EditFileTool:
    name = "edit_file"
    description = (
        "Edit a file by replacing a single exact occurrence of `old_str` with `new_str`. "
        "`old_str` must match exactly once in the target file. When it matches 0 or more than 1 times, "
        "the tool fails with structured context (match count, surrounding lines of the first few hits) "
        "so you can refine `old_str` and retry."
    )
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Target file path. Relative paths are resolved against the workspace root.",
            },
            "old_str": {
                "type": "string",
                "description": "Exact string to match. Must appear exactly once in the file.",
            },
            "new_str": {
                "type": "string",
                "description": "Replacement string that will overwrite the single matched occurrence.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding. Defaults to utf-8.",
                "default": "utf-8",
            },
        },
        "required": ["path", "old_str", "new_str"],
        "additionalProperties": False,
    }
    default_timeout_seconds = 15.0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,  # noqa: ARG002
    ) -> ToolResultContent:
        raw_path = arguments.get("path")
        old_str = arguments.get("old_str")
        new_str = arguments.get("new_str")
        encoding = arguments.get("encoding", "utf-8") or "utf-8"

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolSchemaError("edit_file requires a non-empty string 'path'.")
        if not isinstance(old_str, str):
            raise ToolSchemaError("edit_file requires a string 'old_str'.")
        if not isinstance(new_str, str):
            raise ToolSchemaError("edit_file requires a string 'new_str'.")
        if not isinstance(encoding, str):
            raise ToolSchemaError("edit_file 'encoding' must be a string.")
        if not old_str:
            raise ToolSchemaError("edit_file 'old_str' must be non-empty.")

        target_path = _resolve_path(raw_path, context.workspace_root)
        if not target_path.exists():
            raise ToolTargetError(f"File does not exist: {target_path}")
        if not target_path.is_file():
            raise ToolTargetError(f"Target is not a regular file: {target_path}")

        try:
            content = target_path.read_text(encoding=encoding)
        except (OSError, LookupError, UnicodeDecodeError) as error:
            raise ToolTargetError(f"Unable to read {target_path}: {error}") from error

        count = content.count(old_str)
        if count == 1:
            updated = content.replace(old_str, new_str, 1)
            try:
                target_path.write_text(updated, encoding=encoding)
            except (OSError, LookupError) as error:
                raise ToolExecutionError(f"Unable to write {target_path}: {error}") from error
            return ToolResultContent(
                text=(
                    f"Successfully replaced 1 occurrence in {target_path}.\n"
                    f"--- Before ---\n{old_str}\n--- After ---\n{new_str}"
                ),
                truncated=False,
                target=str(target_path),
            )

        lines = content.splitlines(keepends=True)
        occurrences = _locate_occurrences(lines, old_str)
        snippet = _format_context(lines, occurrences[:5], old_str)
        if count == 0:
            message = (
                f"edit_file failed: old_str was not found in {target_path}. "
                f"Check that whitespace and quotes match exactly.\nContext:\n{snippet}"
            )
        else:
            message = (
                f"edit_file failed: old_str matched {count} times (expected exactly 1) in {target_path}. "
                f"Make old_str more specific by including surrounding context lines.\n"
                f"First {min(5, len(occurrences))} occurrence(s):\n{snippet}"
            )
        raise ToolTargetError(message)


def _locate_occurrences(lines: list[str], old_str: str) -> list[tuple[int, int]]:
    """Return list of (line_index_1based, column_offset)."""
    hits: list[tuple[int, int]] = []
    for line_index, line in enumerate(lines, start=1):
        start = 0
        while True:
            pos = line.find(old_str, start)
            if pos < 0:
                break
            hits.append((line_index, pos))
            start = pos + len(old_str)
    return hits


def _format_context(lines: list[str], occurrences: list[tuple[int, int]], old_str: str) -> str:
    if not occurrences:
        total_lines = len(lines)
        sample_from = max(1, total_lines - 20)
        if total_lines == 0:
            return "(file is empty)"
        excerpt = lines[sample_from - 1 :]
        return _render_excerpt(excerpt, sample_from)

    line_numbers: set[int] = set()
    for line_no, _col in occurrences:
        for n in range(line_no - 2, line_no + 3):
            if 1 <= n <= len(lines):
                line_numbers.add(n)

    ordered = sorted(line_numbers)
    snippet_lines: list[str] = []
    snippet_lines.append(
        f"[matches: {len(occurrences)} | needle_length: {len(old_str)} | needle preview: {old_str[:120]!r}]"
    )
    snippet_lines.append(_render_excerpt([lines[n - 1] for n in ordered], ordered))
    return "\n".join(snippet_lines)


def _render_excerpt(lines: list[str], line_numbers: list[int] | int) -> str:
    start: int = line_numbers if isinstance(line_numbers, int) else line_numbers[0]
    rendered: list[str] = []
    for offset, line in enumerate(lines):
        no = start + offset
        rendered.append(f"{no:4} | {line.rstrip()}")
    return "\n".join(rendered)


def _resolve_path(raw_path: str, workspace_root: Any) -> Any:
    from pathlib import Path

    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()
