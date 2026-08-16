from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall, ToolResult
from .base import (
    ToolExecutionContext,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchemaError,
    ToolTargetError,
)
from .confirmation import (
    ConfirmationGateway,
    ConfirmationRequest,
    ConfirmationType,
)
from .registry import ToolRegistry
from .sandbox import ResolvedPath, WorkspaceSandbox


@dataclass(frozen=True)
class _GroupKey:
    """Determines whether two tool calls must be serialized.

    category: file_write / file_read / shell / neutral
    target:   normalized absolute path for file operations, None otherwise.
    """

    category: str
    target: str | None = None

    def conflicts_with(self, other: "_GroupKey") -> bool:
        # A shell command is exclusive: never overlap with anything else.
        if self.category == "shell" or other.category == "shell":
            return True
        # Reads with a concrete target conflict with writes of the same target,
        # and writes of the same target conflict with each other.
        if self.category in {"file_write", "file_read"} and other.category in {
            "file_write",
            "file_read",
        }:
            return self.target is not None and self.target == other.target
        return False


@dataclass(slots=True)
class _ExecutionPlan:
    call: ToolCall
    index: int = -1
    tool: Any = None
    arguments: dict[str, Any] | None = None
    target: str | None = None
    group_key: _GroupKey | None = None
    confirmation_types: list[ConfirmationType] | None = None
    error_result: ToolResult | None = None


class ToolScheduler:
    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: WorkspaceSandbox,
        request_confirmation: ConfirmationGateway,
    ) -> None:
        self.registry = registry
        self.sandbox = sandbox
        self.request_confirmation = request_confirmation

    async def schedule(self, calls: list[ToolCall]) -> list[ToolResult]:
        plans = [self._preflight(call, index) for index, call in enumerate(calls)]

        # 若批次内含 shell 命令，为保证"命令独占、不与任何调用重叠"，整个批次串行执行。
        if any(
            plan.tool is not None and plan.tool.name == "run_command" and plan.error_result is None
            for plan in plans
        ):
            groups: list[list[int]] = [[plan.index for plan in plans]]
        else:
            groups = self._group_plans(plans)

        results_by_index: dict[int, ToolResult] = {}

        async def _run_group(group: list[int]) -> None:
            # 组内串行：逐个执行，保证写冲突的先后顺序。
            for index in group:
                _index, result = await self._run_single(plans[index])
                results_by_index[_index] = result

        # 组间并行：无冲突的调用（如多个 read_file）同时执行。
        await asyncio.gather(*(_run_group(group) for group in groups))

        return [results_by_index[i] for i in range(len(calls))]

    # ---------- preflight ----------

    def _preflight(self, call: ToolCall, index: int) -> _ExecutionPlan:
        plan = _ExecutionPlan(call=call, index=index)

        try:
            plan.tool = self.registry.require(call.tool_name)
        except ToolNotFoundError as error:
            plan.error_result = self._build_error_result(call, "not_found", str(error), None)
            return plan

        if call.parse_error:
            plan.error_result = self._build_error_result(
                call,
                "schema_violation",
                f"Tool arguments JSON is invalid: {call.parse_error}",
                None,
            )
            return plan

        if not isinstance(call.arguments, dict):
            plan.error_result = self._build_error_result(
                call,
                "schema_violation",
                "Tool arguments must be a JSON object at the top level.",
                None,
            )
            return plan

        plan.arguments = call.arguments

        target_info = _derive_target(call.tool_name, plan.arguments, self.sandbox)
        plan.target = target_info.display
        plan.group_key = target_info.group_key
        plan.confirmation_types = target_info.confirmations
        return plan

    def _group_plans(self, plans: list[_ExecutionPlan]) -> list[list[int]]:
        groups: list[list[int]] = []
        for plan in plans:
            # 已失败的调用单独成组（_run_single 会直接返回 error result，不执行）。
            if plan.error_result is not None or plan.group_key is None:
                groups.append([plan.index])
                continue
            placed = False
            for group in groups:
                group_conflicts = any(
                    plan.group_key.conflicts_with(plans[other].group_key)
                    for other in group
                    if plans[other].group_key is not None
                )
                if group_conflicts:
                    group.append(plan.index)
                    placed = True
                    break
            if not placed:
                groups.append([plan.index])
        return groups

    async def _run_single(self, plan: _ExecutionPlan) -> tuple[int, ToolResult]:
        if plan.error_result is not None:
            return plan.index, plan.error_result

        assert plan.tool is not None
        assert plan.arguments is not None

        # 用户确认：可能同时含越界与覆盖两类风险，合并成一次弹窗。
        if plan.confirmation_types:
            request = _build_confirmation_request(plan.confirmation_types, plan.call, plan.target)
            try:
                response = await self.request_confirmation.confirm(request)
            except Exception as error:  # pragma: no cover - defensive
                return plan.index, self._build_error_result(
                    plan.call,
                    "permission_denied",
                    f"Confirmation flow failed: {error}",
                    plan.target,
                )
            if not getattr(response, "approved", False):
                reason = getattr(response, "reason", None) or "User cancelled the operation."
                return plan.index, self._build_error_result(
                    plan.call,
                    "user_cancelled",
                    reason,
                    plan.target,
                )

        context = ToolExecutionContext(
            workspace_root=self.sandbox.root,
            request_confirmation=self.request_confirmation,
        )
        timeout = _effective_timeout(plan.tool, plan.arguments)

        try:
            async with asyncio.timeout(timeout):
                output = await plan.tool.execute(plan.arguments, context)
        except asyncio.TimeoutError:
            return plan.index, self._build_error_result(
                plan.call,
                "timeout",
                f"Tool '{plan.call.tool_name}' exceeded its timeout of {timeout}s.",
                plan.target,
            )
        except ToolSchemaError as error:
            return plan.index, self._build_error_result(plan.call, "schema_violation", str(error), plan.target)
        except ToolTargetError as error:
            return plan.index, self._build_error_result(plan.call, "bad_target", str(error), plan.target)
        except ToolExecutionError as error:
            return plan.index, self._build_error_result(plan.call, "unknown", str(error), plan.target)
        except Exception as error:  # pragma: no cover - defensive
            return plan.index, self._build_error_result(
                plan.call,
                "unknown",
                f"Unexpected error during tool execution: {error}",
                plan.target,
            )

        is_error = False
        error_type: str | None = None
        if plan.call.tool_name == "run_command":
            # 退出码非零不抛异常，而是标记为失败，让模型读到退出码后自行调整。
            if output.exit_code is not None and output.exit_code != 0:
                is_error = True
                error_type = "command_failed"

        return plan.index, ToolResult(
            tool_call_id=plan.call.tool_call_id,
            tool_name=plan.call.tool_name,
            is_error=is_error,
            error_type=error_type,
            content=output.text,
            truncated=bool(output.truncated),
            target=output.target or plan.target,
        )

    @staticmethod
    def _build_error_result(
        call: ToolCall,
        error_type: str,
        message: str,
        target: str | None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            is_error=True,
            error_type=error_type,
            content=f"[{error_type}] {call.tool_name}: {message}",
            truncated=False,
            target=target,
        )


@dataclass(frozen=True)
class _TargetInfo:
    display: str | None
    group_key: _GroupKey | None
    confirmations: list[ConfirmationType]


def _derive_target(
    tool_name: str,
    arguments: dict[str, Any],
    sandbox: WorkspaceSandbox,
) -> _TargetInfo:
    if tool_name in {"read_file", "write_file", "edit_file"}:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str):
            return _TargetInfo(None, _GroupKey("neutral"), [])
        resolved: ResolvedPath = sandbox.resolve(raw_path)
        category = "file_read" if tool_name == "read_file" else "file_write"
        group_key = _GroupKey(category, str(resolved.absolute))
        confirmations: list[ConfirmationType] = []
        if resolved.is_outside:
            confirmations.append(ConfirmationType.PATH_OUTSIDE_WORKSPACE)
        if tool_name == "write_file" and resolved.absolute.exists():
            confirmations.append(ConfirmationType.OVERWRITE_EXISTING_FILE)
        return _TargetInfo(resolved.target_display, group_key, confirmations)

    if tool_name == "run_command":
        command = arguments.get("command")
        return _TargetInfo(
            command if isinstance(command, str) else None,
            _GroupKey("shell"),
            [ConfirmationType.DANGEROUS_COMMAND],
        )

    if tool_name in {"find_files", "search_code"}:
        return _TargetInfo(str(sandbox.root), _GroupKey("file_read", None), [])

    return _TargetInfo(None, _GroupKey("neutral"), [])


def _effective_timeout(tool: Any, arguments: dict[str, Any]) -> float:
    default = getattr(tool, "default_timeout_seconds", 30.0)
    if tool.name == "run_command":
        override = arguments.get("timeout_seconds")
        if isinstance(override, (int, float)) and not isinstance(override, bool) and override > 0:
            return float(override)
    return float(default)


def _build_confirmation_request(
    confirmation_types: list[ConfirmationType],
    call: ToolCall,
    target: str | None,
) -> ConfirmationRequest:
    if confirmation_types == [ConfirmationType.DANGEROUS_COMMAND]:
        return ConfirmationRequest(
            type=ConfirmationType.DANGEROUS_COMMAND,
            title="Run shell command",
            description=(
                "The model requested executing a shell command. This can modify files on disk, "
                "consume resources, or leak data. Approve only if you understand the command."
            ),
            target=target or call.tool_name,
            detail=f"Tool: {call.tool_name}\nArguments: {call.arguments}",
        )

    risks: list[str] = []
    for kind in confirmation_types:
        if kind == ConfirmationType.PATH_OUTSIDE_WORKSPACE:
            risks.append("Path is outside the workspace root")
        if kind == ConfirmationType.OVERWRITE_EXISTING_FILE:
            risks.append("Target file already exists and will be overwritten")
    return ConfirmationRequest(
        type=confirmation_types[0],
        title="Confirm file operation",
        description="The model requested a file operation that requires user confirmation:\n- "
        + "\n- ".join(risks),
        target=target or call.tool_name,
        detail=f"Tool: {call.tool_name}\nArguments: {call.arguments}",
    )
