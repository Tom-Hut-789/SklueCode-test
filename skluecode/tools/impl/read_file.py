from __future__ import annotations

from typing import Any

from ..base import (
    ToolExecutionContext,
    ToolResultContent,
    ToolSchemaError,
    ToolTargetError,
)


class ReadFileTool:
    name = "read_file"
    description = (
        "Read the full contents of a text file. Paths can be absolute or relative to the workspace root. "
        "This tool reports files that do not exist, are not regular files, or are larger than the byte limit."
    )
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Target file path. Relative paths are resolved against the workspace root.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding. Defaults to utf-8.",
                "default": "utf-8",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum bytes to read before truncation. Defaults to 500000.",
                "default": 500_000,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    default_timeout_seconds = 10.0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,  # noqa: ARG002
    ) -> ToolResultContent:
        raw_path = arguments.get("path")
        encoding = arguments.get("encoding", "utf-8") or "utf-8"
        max_bytes = arguments.get("max_bytes", 500_000)

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolSchemaError("read_file requires a non-empty string 'path'.")
        if not isinstance(encoding, str):
            raise ToolSchemaError("read_file 'encoding' must be a string.")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ToolSchemaError("read_file 'max_bytes' must be a positive integer.")

        target_path = _resolve_path(raw_path, context.workspace_root)
        if not target_path.exists():
            raise ToolTargetError(f"File does not exist: {target_path}")
        if not target_path.is_file():
            raise ToolTargetError(f"Target is not a regular file: {target_path}")

        try:
            raw_bytes = target_path.read_bytes()
        except OSError as error:
            raise ToolTargetError(f"Unable to read {target_path}: {error}") from error

        truncated = len(raw_bytes) > max_bytes
        if truncated:
            raw_bytes = raw_bytes[:max_bytes]

        try:
            text = raw_bytes.decode(encoding=encoding, errors="replace")
        except LookupError as error:
            raise ToolTargetError(f"Unknown encoding '{encoding}': {error}") from error

        content = text
        if truncated:
            content = f"{content}\n[truncated: only the first {max_bytes} bytes of the file are shown]"

        return ToolResultContent(text=content, truncated=truncated, target=str(target_path))


def _resolve_path(raw_path: str, workspace_root: Any) -> Any:
    from pathlib import Path

    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()
