from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationInfo, field_validator

if TYPE_CHECKING:
    from tiny_coder.file_agent import AgentContext


class ExecutableScriptOutput(ABC):
    """Structured output that provides source code for an executable script."""

    async def prepare_executable_script(self, context: AgentContext) -> None:
        """Normalize and validate the script before optional execution."""
        _ = context

    @abstractmethod
    def to_executable_script(self, context: AgentContext) -> str: ...


class GeneratedScriptOutput(BaseModel, ExecutableScriptOutput):
    """Default structured response for one generated executable script."""

    summary: str = Field(
        ...,
        min_length=1,
        description="Concise plain-text summary of the generated script's behavior.",
    )
    script: str = Field(
        ...,
        min_length=1,
        description="Complete executable script source code to validate and run.",
    )

    @field_validator("summary", "script")
    @classmethod
    def validate_non_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"generated {info.field_name} must not be blank")
        return value

    def to_executable_script(self, context: AgentContext) -> str:
        _ = context
        return self.script


@dataclass(frozen=True, slots=True)
class ScriptExecutionResult:
    """Captured result from executing one generated script."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


__all__ = [
    "ExecutableScriptOutput",
    "GeneratedScriptOutput",
    "ScriptExecutionResult",
]
