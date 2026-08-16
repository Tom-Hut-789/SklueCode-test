from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..base import (
    ToolExecutionContext,
    ToolResultContent,
    ToolSchemaError,
    ToolTargetError,
)


class SearchCodeTool:
    name = "search_code"
    description = (
        "Search file contents under the workspace root, either with a literal substring or a regex. "
        "Each match is reported with its file path, line number and surrounding context lines. "
        "Too many matches or too many output lines are truncated so the result stays compact."
    )
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Substring or regex pattern to search for.",
            },
            "mode": {
                "type": "string",
                "enum": ["literal", "regex"],
                "description": "Whether `query` is a plain literal or a regular expression. Defaults to literal.",
                "default": "literal",
            },
            "include_globs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of glob patterns used to restrict candidate files, e.g. [\"**/*.py\"].",
            },
            "context_lines": {
                "type": "integer",
                "description": "How many lines before and after each match to include. Defaults to 2.",
                "default": 2,
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum number of matches before truncation. Defaults to 100.",
                "default": 100,
            },
            "max_total_lines": {
                "type": "integer",
                "description": "Soft cap on total rendered output lines. Defaults to 400.",
                "default": 400,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    default_timeout_seconds = 60.0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResultContent:
        query = arguments.get("query")
        mode = arguments.get("mode", "literal") or "literal"
        include_globs = arguments.get("include_globs")
        context_lines = arguments.get("context_lines", 2)
        max_matches = arguments.get("max_matches", 100)
        max_total_lines = arguments.get("max_total_lines", 400)

        if not isinstance(query, str) or not query:
            raise ToolSchemaError("search_code requires a non-empty string 'query'.")
        if mode not in {"literal", "regex"}:
            raise ToolSchemaError("search_code 'mode' must be either 'literal' or 'regex'.")
        if include_globs is not None:
            if not isinstance(include_globs, list) or not all(
                isinstance(item, str) and item for item in include_globs
            ):
                raise ToolSchemaError(
                    "search_code 'include_globs' must be a list of non-empty strings when provided."
                )
        if (
            not isinstance(context_lines, int)
            or isinstance(context_lines, bool)
            or context_lines < 0
        ):
            raise ToolSchemaError("search_code 'context_lines' must be a non-negative integer.")
        if (
            not isinstance(max_matches, int)
            or isinstance(max_matches, bool)
            or max_matches <= 0
        ):
            raise ToolSchemaError("search_code 'max_matches' must be a positive integer.")
        if (
            not isinstance(max_total_lines, int)
            or isinstance(max_total_lines, bool)
            or max_total_lines <= 0
        ):
            raise ToolSchemaError("search_code 'max_total_lines' must be a positive integer.")

        root = Path(context.workspace_root)
        if not root.exists() or not root.is_dir():
            raise ToolTargetError(f"Workspace root does not exist or is not a directory: {root}")

        if mode == "regex":
            try:
                pattern = re.compile(query)
            except re.error as error:
                raise ToolTargetError(f"Invalid regex pattern {query!r}: {error}") from error
        else:
            pattern = None

        candidate_files = _collect_candidate_files(root, include_globs or None)

        matches: list[tuple[Path, int, str]] = []
        truncated_matches = False
        truncated_lines = False
        for file_path in candidate_files:
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if mode == "regex":
                    assert pattern is not None
                    if not pattern.search(line):
                        continue
                else:
                    if query not in line:
                        continue
                matches.append((file_path, line_no, line))
                if len(matches) > max_matches:
                    truncated_matches = True
                    break
            if truncated_matches:
                break

        rendered: list[str] = []
        header = f"Searched {len(candidate_files)} file(s) for {mode} pattern {query!r} under {root}."
        if truncated_matches:
            header += (
                f" Match count exceeded max_matches={max_matches}; only the first "
                f"{max_matches} matches are reported."
            )
        rendered.append(header)

        total_lines = 0
        last_file: Path | None = None
        reported = 0
        for file_path, line_no, line in matches[:max_matches]:
            reported += 1
            if total_lines >= max_total_lines:
                truncated_lines = True
                break
            if file_path != last_file:
                relative = str(file_path.relative_to(root)) if file_path.is_absolute() and str(file_path).startswith(str(root)) else str(file_path)
                rendered.append("")
                rendered.append(f"## {relative}")
                last_file = file_path
                total_lines += 2
                if total_lines >= max_total_lines:
                    truncated_lines = True
                    break
            start = max(1, line_no - context_lines)
            end = min(line_no + context_lines, 9_999_999)
            try:
                all_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            end = min(end, len(all_lines))
            rendered.append(f"-- around line {line_no} (match #{reported}) --")
            total_lines += 1
            for n in range(start, end + 1):
                if total_lines >= max_total_lines:
                    truncated_lines = True
                    break
                marker = ">>" if n == line_no else "  "
                rendered.append(f"{marker}{n:4} | {all_lines[n - 1]}")
                total_lines += 1

        if truncated_lines:
            rendered.append(
                f"[truncated: output exceeded max_total_lines={max_total_lines}. "
                "Consider a more specific query or larger max_total_lines/max_matches.]"
            )

        if reported == 0:
            rendered.append("No matches found.")

        truncated = truncated_matches or truncated_lines
        return ToolResultContent(text="\n".join(rendered), truncated=truncated, target=str(root))


def _collect_candidate_files(root: Path, include_globs: list[str] | None) -> list[Path]:
    if not include_globs:
        # Use a reasonable default: search all files but skip heavy dirs at top level.
        skipped = {".venv", ".git", "__pycache__", "node_modules", ".pytest_cache"}
        results: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skipped for part in path.relative_to(root).parts):
                continue
            results.append(path)
        return results

    results_set: set[Path] = set()
    for pattern in include_globs:
        for path in root.rglob(pattern):
            if path.is_file():
                results_set.add(path)
    # Skip the same heavy dirs even with include_globs if path happens to hit them? Keep simple: don't skip.
    return sorted(results_set)
