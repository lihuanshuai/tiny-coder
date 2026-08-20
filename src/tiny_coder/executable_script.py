from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from tiny_coder.file_agent import AgentContext


class ExecutableScriptOutput(ABC):
    """Structured output that provides source code for an executable script."""

    @abstractmethod
    def to_executable_script(self, context: AgentContext) -> str: ...


class GeneratedScriptOutput(BaseModel, ExecutableScriptOutput):
    """Default structured response for one generated executable script."""

    summary: str = ""
    script: str

    @field_validator("script")
    @classmethod
    def validate_script(cls, script: str) -> str:
        if not script.strip():
            raise ValueError("generated script must not be blank")
        return script

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
    timed_out: bool = False


__all__ = [
    "ExecutableScriptOutput",
    "GeneratedScriptOutput",
    "ScriptExecutionResult",
]
