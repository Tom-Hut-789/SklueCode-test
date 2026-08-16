from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..base import (
    ToolExecutionContext,
    ToolResultContent,
    ToolSchemaError,
)


class RunCommandTool:
    name = "run_command"
    description = (
        "Run an arbitrary shell command in the current user's default shell. "
        "This tool is marked dangerous; the user is always asked to confirm before execution. "
        "Long-running commands are killed after `timeout_seconds`; stdout/stderr may be truncated."
    )
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. It is forwarded to the default shell.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command. Defaults to the workspace root.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional custom timeout in seconds. Overrides the default 300s budget.",
            },
            "max_output_bytes": {
                "type": "integer",
                "description": "Optional total bytes cap (combined stdout + stderr) before truncation. Defaults to 65536.",
                "default": 65536,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    default_timeout_seconds = 300.0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResultContent:
        command = arguments.get("command")
        cwd_raw = arguments.get("cwd")
        timeout_override = arguments.get("timeout_seconds")
        max_output_bytes = arguments.get("max_output_bytes", 65536)

        if not isinstance(command, str) or not command.strip():
            raise ToolSchemaError("run_command requires a non-empty string 'command'.")
        if cwd_raw is not None and (not isinstance(cwd_raw, str) or not cwd_raw.strip()):
            raise ToolSchemaError("run_command 'cwd' must be a non-empty string if provided.")
        if timeout_override is not None:
            if isinstance(timeout_override, bool) or not isinstance(timeout_override, (int, float)) or timeout_override <= 0:
                raise ToolSchemaError("run_command 'timeout_seconds' must be a positive number.")
        if (
            not isinstance(max_output_bytes, int)
            or isinstance(max_output_bytes, bool)
            or max_output_bytes <= 0
        ):
            raise ToolSchemaError("run_command 'max_output_bytes' must be a positive integer.")

        if cwd_raw:
            cwd_path = Path(cwd_raw)
            if not cwd_path.is_absolute():
                cwd_path = context.workspace_root / cwd_path
            cwd = str(cwd_path.resolve())
        else:
            cwd = str(context.workspace_root.resolve())

        self._timeout_seconds_override = float(timeout_override) if timeout_override is not None else None

        proc = await asyncio.create_subprocess_shell(
            command,
            shell=True,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_data, stderr_data = await proc.communicate()
        except Exception as error:  # pragma: no cover - defensive
            return ToolResultContent(
                text=f"[exit_code=-1]\n--- COMMAND EXCEPTION ---\n{error}\n[command: {command!r}]\n[cwd: {cwd}]",
                truncated=False,
                target=command,
                exit_code=-1,
            )

        exit_code = proc.returncode if proc.returncode is not None else -1
        total_bytes = len(stdout_data) + len(stderr_data)
        truncated = total_bytes > max_output_bytes
        if truncated:
            stdout_bytes = stdout_data[: max_output_bytes // 2]
            stderr_bytes = stderr_data[: max_output_bytes // 2]
        else:
            stdout_bytes = stdout_data
            stderr_bytes = stderr_data

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")
        sections = [f"[exit_code={exit_code}]", f"[command] {command}", f"[cwd] {cwd}"]
        if truncated:
            sections.append(
                f"[truncated: output capped at about {max_output_bytes} combined bytes "
                f"(original total was {total_bytes} bytes)]"
            )
        sections.append("--- STDOUT ---")
        sections.append(stdout_text)
        sections.append("--- STDERR ---")
        sections.append(stderr_text)
        return ToolResultContent(
            text="\n".join(sections),
            truncated=truncated,
            target=command,
            exit_code=exit_code,
        )

    @property
    def effective_timeout_seconds(self) -> float:
        return self._timeout_seconds_override or self.default_timeout_seconds
