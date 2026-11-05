"""`Tool` argument models, `ToolNode` execution, and `has_tool_calls`."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter
from typing import (
    Any,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from tiny_coder.state import State, Update
from tiny_coder.tool_arguments import ToolArgumentsError, validate_tool_payload

_P = ParamSpec("_P")
_R = TypeVar("_R")
ToolHandler: TypeAlias = Callable[_P, Coroutine[Any, Any, _R]]


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
class Tool(Generic[_P, _R]):
    """One callable tool described by a Pydantic argument model."""

    name: str
    description: str
    model: type[BaseModel]
    handler: ToolHandler[_P, _R]
    parameters_schema: dict[str, Any] | None = None
    _state_mode: Literal["positional", "keyword", "none"] = field(default="positional", repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("tool name must not be blank")
        if not isinstance(self.model, type) or not issubclass(self.model, BaseModel):
            raise TypeError("tool model must be a BaseModel subclass")
        if not inspect.iscoroutinefunction(self.handler):
            raise TypeError("tool handler must be an async def function; define it with async def")

    @property
    def parameters(self) -> dict[str, Any]:
        return (
            self.parameters_schema
            if self.parameters_schema is not None
            else self.model.model_json_schema()
        )

    @classmethod
    def from_model(
        cls,
        model: type[BaseModel],
        handler: ToolHandler[_P, _R],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool[_P, _R]:
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError("model must be a BaseModel subclass")
        return cls(
            name=name or _snake_case(model.__name__),
            description=description or _default_description(model),
            model=model,
            handler=handler,
        )

    def with_parameters(self, **constraints: dict[str, Any]) -> Self:
        """Override named field schemas on a copy; keep the Pydantic validators.

        Each value is a complete JSON Schema for that field, e.g.
        ``path={"type": "string", "enum": allowed_paths}``. Overrides are
        also checked locally before the handler executes.
        """
        if not constraints:
            return self
        schema = deepcopy(self.parameters)
        properties = schema.get("properties", {})
        for name, constraint in constraints.items():
            if name not in properties:
                raise ValueError(f"unknown parameter {name!r} for tool {self.name!r}")
            properties[name] = deepcopy(constraint)
        return replace(self, parameters_schema=schema)

    async def invoke(self, state: State, arguments: str) -> str:
        try:
            payload = validate_tool_payload(self.name, self.parameters, arguments)
        except ToolArgumentsError as error:
            return str(error)
        try:
            validated = self.model.model_validate(payload)
        except ValidationError as error:
            return "error: invalid arguments: " + json.dumps(
                {
                    "tool": self.name,
                    "errors": [
                        {
                            **failure,
                            "suggestion": f"{self.name}: Correct field "
                            f"`{'.'.join(map(str, failure['loc'])) or '$'}`: {failure['msg']}. "
                            "Resubmit the corrected argument object; the tool was not executed.",
                        }
                        for failure in error.errors(
                            include_url=False, include_input=False, include_context=False
                        )
                    ],
                    "required_fields": self.parameters.get("required", []),
                    "provided_fields": sorted(payload),
                    "executed": False,
                },
                ensure_ascii=False,
            )
        try:
            kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
            # At the JSON boundary the schema, rather than a static call site, validates kwargs.
            handler = cast(ToolHandler[..., _R], self.handler)
            if self._state_mode == "positional":
                result = await handler(state, **kwargs)
            else:
                if self._state_mode == "keyword":
                    kwargs["state"] = state
                result = await handler(**kwargs)
        except Exception as error:
            return f"error: {type(error).__name__}: {error}"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


class _ToolDecorator(Protocol):
    def __call__(self, function: ToolHandler[_P, _R], /) -> Tool[_P, _R]: ...


@overload
def tool(
    function: Tool[_P, _R] | ToolHandler[_P, _R],
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, dict[str, Any]] | None = None,
) -> Tool[_P, _R]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, dict[str, Any]] | None = None,
) -> _ToolDecorator: ...


def tool(
    function: Tool[..., Any] | ToolHandler[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, dict[str, Any]] | None = None,
) -> Tool[..., Any] | _ToolDecorator:
    """Define a tool from a typed async function and its docstring.

    An optional ``state: State`` parameter is injected and never exposed to the
    model. Use Annotated[T, Field(...)] for field constraints and Pydantic model
    parameters for custom validators. Handlers receive typed values,
    including nested models. Handlers must use async def.
    """

    if isinstance(function, Tool):
        definition = function
        if name is not None or description is not None:
            definition = replace(
                definition,
                name=definition.name if name is None else name,
                description=definition.description if description is None else description,
            )
        return definition.with_parameters(**parameters) if parameters else definition

    def define(handler: ToolHandler[_P, _R]) -> Tool[_P, _R]:
        signature = inspect.signature(handler)
        hints = get_type_hints(handler, include_extras=True)
        fields: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise TypeError(f"tool parameter {parameter.name!r} must be a named argument")
            if parameter.name == "state":
                if hints.get("state") != State:
                    raise TypeError("injected tool parameter 'state' must be annotated as State")
                continue
            if parameter.name not in hints:
                raise TypeError(f"tool parameter {parameter.name!r} requires a type annotation")
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (hints[parameter.name], default)
        model = create_model(
            f"{handler.__name__}Arguments", __config__=ConfigDict(extra="forbid"), **fields
        )

        definition = Tool(
            name=handler.__name__ if name is None else name,
            description=(inspect.getdoc(handler) or handler.__name__)
            if description is None
            else description,
            model=model,
            handler=handler,
            _state_mode="keyword" if "state" in signature.parameters else "none",
        )
        return definition.with_parameters(**parameters) if parameters else definition

    return define(function) if function is not None else define


ToolDefinition: TypeAlias = Tool[..., Any] | ToolHandler[..., Any]


def _register_tools(
    tools: list[Tool[..., Any]], *definitions: ToolDefinition, replace: bool = False
) -> None:
    """Register a batch atomically; replacement preserves existing name order."""
    updated = {definition.name: definition for definition in tools}
    if len(updated) != len(tools):
        raise ValueError("duplicate tool names in existing tools")
    batch = [tool(definition) for definition in definitions]
    names: set[str] = set()
    for definition in batch:
        if definition.name in names or (not replace and definition.name in updated):
            raise ValueError(f"duplicate tool name: {definition.name!r}")
        names.add(definition.name)
        updated[definition.name] = definition
    tools[:] = updated.values()


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


@dataclass(frozen=True)
class ToolExecution:
    """Execution metadata kept out of the model's tool result."""

    tool_call_id: str
    name: str
    started_at: str
    completed_at: str
    duration_ms: float
    status: Literal["completed", "error", "cancelled"]
    error: str | None = None


@dataclass
class ToolNode:
    """A graph node that executes the last assistant message's tool calls."""

    tools: list[Tool[..., Any]]
    on_tool_finished: Callable[[ToolExecution], Awaitable[None]] | None = None
    _by_name: dict[str, Tool[..., Any]] = field(default_factory=dict, init=False, repr=False)

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
            started_at = datetime.now(timezone.utc).isoformat()
            started = perf_counter()
            status: Literal["completed", "error", "cancelled"] = "completed"
            error: str | None = None
            try:
                if tool is None:
                    content = (
                        f"error: unknown tool: {name or '<missing>'}; "
                        f"choose exactly one of the registered tool names: {', '.join(self._by_name)}"
                    )
                else:
                    content = await tool.invoke(state, call["arguments"])
                if content.startswith("error:"):
                    status, error = "error", content
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exception:
                status, error = "error", f"{type(exception).__name__}: {exception}"
                raise
            finally:
                execution = ToolExecution(
                    tool_call_id=call["id"],
                    name=name,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=(perf_counter() - started) * 1000,
                    status=status,
                    error=error,
                )
                if self.on_tool_finished is not None:
                    await self.on_tool_finished(execution)
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
    "tool",
    "ToolHandler",
    "ToolDefinition",
    "ToolNode",
    "ToolExecution",
    "has_tool_calls",
]
