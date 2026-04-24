import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

logger = logging.getLogger(__name__)


class ToolResult:
    def __init__(
        self, success: bool, output: str | None = None, error: str | None = None
    ) -> None:
        self.success = success
        self.output = output
        self.error = error

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        return cls(success=True, output=output)

    @classmethod
    def fail(cls, message: str) -> "ToolResult":
        return cls(success=False, error=message)

    def to_agent_str(self) -> str:
        if self.success:
            return self.output or ""
        return f"ERROR: {self.error}"


class BaseTool(ABC, Generic[InputT, OutputT]):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]

    def run(self, raw_input: dict[str, Any]) -> ToolResult:
        try:
            payload = self.input_model.model_validate(raw_input)
        except Exception as exc:
            logger.warning("Tool '%s' — invalid input: %s", self.name, exc)
            return ToolResult.fail(f"Invalid input: {exc}")

        try:
            output = self._execute(payload)  # type: ignore[arg-type]
            return ToolResult.ok(output.model_dump_json(indent=2))
        except Exception as exc:
            logger.error(
                "Tool '%s' — execution error: %s", self.name, exc, exc_info=True
            )
            return ToolResult.fail(str(exc))

    @abstractmethod
    def _execute(self, payload: InputT) -> OutputT: ...

    def to_gemini_schema(self) -> dict[str, Any]:
        raw = self.input_model.model_json_schema()
        defs = raw.pop("$defs", {})

        def _clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_name = obj["$ref"].split("/")[-1]
                    return _clean(defs.get(ref_name, obj))
                return {k: _clean(v) for k, v in obj.items() if k != "title"}
            if isinstance(obj, list):
                return [_clean(item) for item in obj]
            return obj

        parameters = _clean(raw)
        parameters.pop("title", None)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }
