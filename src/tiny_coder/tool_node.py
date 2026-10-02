"""`Tool` argument models, `ToolNode` execution, and `has_tool_calls`."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pydantic import BaseModel, ValidationError

from tiny_coder.state import State, Update

ToolHandler: TypeAlias = Callable[..., Awaitable[str]]


def _snake_case(name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _default_description(model: type[BaseModel]) -> str:
    doc = (model.__doc__ or "").strip()
    return doc or model.__name__


@dataclass(frozen=True)
class Tool:
    """One callable tool described by a Pydantic argument model."""

    name: str
    description: str
    model: type[BaseModel]
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("tool name must not be blank")
        if not isinstance(self.model, type) or not issubclass(self.model, BaseModel):
            raise TypeError("tool model must be a BaseModel subclass")
        if not callable(self.handler):
            raise TypeError("tool handler must be an async callable(state, **kwargs)")

    @property
    def parameters(self) -> dict[str, Any]:
        return self.model.model_json_schema()

    @classmethod
    def from_model(
        cls,
        model: type[BaseModel],
        handler: ToolHandler,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError("model must be a BaseModel subclass")
        return cls(
            name=name or _snake_case(model.__name__),
            description=description or _default_description(model),
            model=model,
            handler=handler,
        )

    async def invoke(self, state: State, arguments: str) -> str:
        try:
            payload = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as error:
            return f"error: invalid JSON arguments: {error}"
        if not isinstance(payload, dict):
            return "error: tool arguments must be a JSON object"
        try:
            validated = self.model.model_validate(payload)
        except ValidationError as error:
            return f"error: invalid arguments: {error}"
        try:
            result = await self.handler(state, **validated.model_dump())
        except Exception as error:
            return f"error: {type(error).__name__}: {error}"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


def _message_tool_calls(message: object) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        raw = message.get("tool_calls")
    else:
        raw = getattr(message, "tool_calls", None)
    if not isinstance(raw, (list, tuple)):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            calls.append(
                {
                    "id": str(item.get("id", "")),
                    "name": str(function.get("name", "")),
                    "arguments": str(function.get("arguments", "")),
                }
            )
    return calls


def has_tool_calls(state: State) -> bool:
    """Return True when the last message requests one or more tool calls."""
    messages = state.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str) or not messages:
        return False
    return bool(_message_tool_calls(messages[-1]))


@dataclass
class ToolNode:
    """A graph node that executes the last assistant message's tool calls."""

    tools: list[Tool]
    _by_name: dict[str, Tool] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for tool in self.tools:
            if not isinstance(tool, Tool):
                raise TypeError("tools must contain Tool instances")
            if tool.name in self._by_name:
                raise ValueError(f"duplicate tool name: {tool.name!r}")
            self._by_name[tool.name] = tool

    async def __call__(self, state: State) -> Update:
        messages = state.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, str) or not messages:
            return {}
        calls = _message_tool_calls(messages[-1])
        if not calls:
            return {}
        results: list[dict[str, Any]] = []
        for call in calls:
            name = call["name"]
            tool = self._by_name.get(name)
            if tool is None:
                content = f"error: unknown tool: {name or '<missing>'}"
            else:
                content = await tool.invoke(state, call["arguments"])
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": content,
                }
            )
        return {"messages": results}


__all__ = [
    "Tool",
    "ToolHandler",
    "ToolNode",
    "has_tool_calls",
]
