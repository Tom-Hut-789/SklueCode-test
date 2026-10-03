from __future__ import annotations

from typing import Iterable

from .base import NeutralToolDef, Tool, ToolNotFoundError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool not registered: {name}")
        return tool

    def to_neutral_definitions(self, names: Iterable[str] | None = None) -> list[NeutralToolDef]:
        if names is None:
            selected = list(self._tools.values())
        else:
            wanted = set(names)
            selected = [tool for tool in self._tools.values() if tool.name in wanted]
        return [
            NeutralToolDef(
                name=tool.name,
                description=tool.description,
                parameters=dict(tool.parameters_json_schema),
            )
            for tool in selected
        ]
