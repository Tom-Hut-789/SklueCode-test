from __future__ import annotations

from typing import Any

from ..base import (
    ToolExecutionContext,
    ToolExecutionError,
    ToolResultContent,
    ToolSchemaError,
)


class WriteFileTool:
    name = "write_file"
    description = (
        "Write text content to a file, creating the file and its parent directories when necessary. "
        "If the target file already exists, the scheduler will ask the user for confirmation before running."
    )
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination file path. Relative paths are resolved against the workspace root.",
            },
            "content": {
                "type": "string",
                "description": "Full text content to write into the destination file.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding. Defaults to utf-8.",
                "default": "utf-8",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    default_timeout_seconds = 15.0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,  # noqa: ARG002
    ) -> ToolResultContent:
        raw_path = arguments.get("path")
        content = arguments.get("content")
        encoding = arguments.get("encoding", "utf-8") or "utf-8"

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolSchemaError("write_file requires a non-empty string 'path'.")
        if not isinstance(content, str):
            raise ToolSchemaError("write_file requires a string 'content'.")
        if not isinstance(encoding, str):
            raise ToolSchemaError("write_file 'encoding' must be a string.")

        target_path = _resolve_path(raw_path, context.workspace_root)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding=encoding)
        except (OSError, LookupError) as error:
            raise ToolExecutionError(f"Unable to write {target_path}: {error}") from error

        return ToolResultContent(
            text=f"Successfully wrote {len(content)} characters to {target_path}.",
            truncated=False,
            target=str(target_path),
        )


def _resolve_path(raw_path: str, workspace_root: Any) -> Any:
    from pathlib import Path

    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()
