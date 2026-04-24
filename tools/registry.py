import logging
from typing import Any

from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered. Overwriting.", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Tool registered: '%s'", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            logger.error("Attempted to dispatch unknown tool: '%s'", name)
            return ToolResult.fail(f"Unknown tool: '{name}'")

        return tool.run(raw_input)

    def get_gemini_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_gemini_schema() for tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
