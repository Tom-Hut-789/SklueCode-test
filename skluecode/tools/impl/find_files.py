from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import (
    ToolExecutionContext,
    ToolResultContent,
    ToolSchemaError,
    ToolTargetError,
)


class FindFilesTool:
    name = "find_files"
    description = (
        "List files under the workspace root that match a glob-style pattern such as `**/*.py` or `tests/*_test.py`. "
        "Results are returned as relative paths; very large result sets are truncated."
    )
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern. Examples: **/*.py, docs/**/*.md, src/*.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to search recursively. Defaults to true.",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of paths to return before truncation. Defaults to 200.",
                "default": 200,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    default_timeout_seconds = 30.0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResultContent:
        pattern = arguments.get("pattern")
        recursive = arguments.get("recursive", True)
        max_results = arguments.get("max_results", 200)

        if not isinstance(pattern, str) or not pattern.strip():
            raise ToolSchemaError("find_files requires a non-empty string 'pattern'.")
        if not isinstance(recursive, bool):
            raise ToolSchemaError("find_files 'recursive' must be a boolean.")
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or max_results <= 0
        ):
            raise ToolSchemaError("find_files 'max_results' must be a positive integer.")

        root = Path(context.workspace_root)
        if not root.exists() or not root.is_dir():
            raise ToolTargetError(f"Workspace root does not exist or is not a directory: {root}")

        try:
            if recursive:
                iterator = root.rglob(pattern)
            else:
                iterator = root.glob(pattern)
            matches = [path for path in iterator if path.is_file()]
        except ValueError as error:
            raise ToolTargetError(f"Invalid glob pattern {pattern!r}: {error}") from error

        total = len(matches)
        truncated = total > max_results
        if truncated:
            matches = matches[:max_results]

        relative_paths: list[str] = []
        for path in matches:
            try:
                relative_paths.append(str(path.relative_to(root)))
            except ValueError:
                relative_paths.append(str(path))

        if not relative_paths:
            content = f"No files matched pattern {pattern!r} under {root}."
        else:
            header = (
                f"Found {total} file(s) matching pattern {pattern!r} under {root}. "
                f"Showing {len(relative_paths)} result(s)."
                if not truncated
                else (
                    f"Found {total} file(s) matching pattern {pattern!r} under {root}. "
                    f"Only the first {max_results} are shown below."
                )
            )
            content = "\n".join([header, *relative_paths])

        return ToolResultContent(text=content, truncated=truncated, target=str(root))
