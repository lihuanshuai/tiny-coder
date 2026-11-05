from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Any, Literal

import pytest
from mypy import api as mypy_api
from pydantic import BaseModel, Field, field_validator

from tiny_coder.llm import tool_openai_spec
from tiny_coder.state import State
from tiny_coder.tool_node import Tool, ToolExecution, ToolNode, has_tool_calls, tool


@tool
async def add_typed(
    state: State, left: Annotated[int, Field(ge=1)], *, right: int = 2
) -> dict[str, int]:
    """Add positive numbers and record the result."""
    state["sum"] = left + right
    return {"sum": left + right}


def test_decorated_tool_infers_schema_defaults_and_injects_state() -> None:
    spec = tool_openai_spec(add_typed)
    assert spec["name"] == "add_typed"
    assert spec["description"] == "Add positive numbers and record the result."
    assert set(spec["parameters"]["properties"]) == {"left", "right"}
    assert spec["parameters"]["properties"]["left"]["minimum"] == 1
    assert spec["parameters"]["required"] == ["left", "right"]
    state: State = {}
    assert json.loads(asyncio.run(add_typed.invoke(state, '{"left":3}'))) == {"sum": 5}
    assert state == {"sum": 5}
    for invalid in ('{"left":0}', '{"left":"3"}', '{"left":3,"state":{}}'):
        assert asyncio.run(add_typed.invoke({}, invalid)).startswith("error: invalid arguments")


class AddArgs(BaseModel):
    """Add two integers."""

    left: int
    right: int


class NoopArgs(BaseModel):
    """Do nothing."""


class PositiveBatch(BaseModel):
    """A validated group of additions."""

    additions: list[AddArgs]

    @field_validator("additions")
    @classmethod
    def nonempty(cls, value: list[AddArgs]) -> list[AddArgs]:
        if not value:
            raise ValueError("provide at least one addition")
        return value


def test_decorator_infers_nested_models_and_preserves_their_validators() -> None:
    @tool(name="batch")
    async def add(batch: PositiveBatch) -> int:
        return sum(item.left + item.right for item in batch.additions)

    assert add.name == "batch"
    assert "at least one addition" in asyncio.run(add.invoke({}, '{"batch":{"additions":[]}}'))
    assert asyncio.run(add.invoke({}, '{"batch":{"additions":[{"left":1,"right":2}]}}')) == "3"


def test_async_tool_accepts_keyword_state() -> None:
    @tool(description="Record a value.")
    async def worker(*, state: State) -> int:
        state["value"] = 42
        return state["value"]

    async def run() -> None:
        state: State = {}
        result = await worker.invoke(state, "{}")
        assert int(result) == state["value"] == 42

    asyncio.run(run())


def test_typed_async_handler_preserves_signature_and_return_type() -> None:
    @tool
    async def later(value: int) -> int:
        return value + 1

    assert asyncio.run(later.invoke({}, '{"value":1}')) == "2"

    async def run() -> None:
        assert await later.handler(value=2) == 3
        assert await add_typed.handler({}, left=2, right=3) == {"sum": 5}

    asyncio.run(run())


def _sync_handler(value: int) -> int:
    return value


def _sync_coroutine_handler(value: int) -> Coroutine[Any, Any, int]:
    async def result() -> int:
        return value

    return result()


@pytest.mark.parametrize("handler", [_sync_handler, _sync_coroutine_handler])
def test_all_tool_definition_paths_reject_synchronous_handlers(handler: Callable[..., Any]) -> None:
    for define in (
        lambda: tool(handler),
        lambda: tool(handler, name="invalid"),
        lambda: tool(name="invalid")(handler),
        lambda: Tool.from_model(AddArgs, handler),
        lambda: Tool(name="invalid", description="Invalid", model=AddArgs, handler=handler),
    ):
        with pytest.raises(TypeError, match="async def"):
            define()


def test_generic_tool_types_preserve_parameters_and_results(tmp_path: Path) -> None:
    source = tmp_path / "tool_typing.py"
    with source.open("w", encoding="utf-8", newline="\n") as target:
        target.write(
            dedent('''\
            from typing import assert_type
            from pydantic import BaseModel
            from tiny_coder.state import State
            from tiny_coder.tool_node import Tool, tool

            @tool
            async def add(left: int, right: int = 2) -> int:
                return left + right

            @tool(name="echo")
            async def echo(value: str) -> str:
                return value

            @tool()
            async def flag(*, enabled: bool = False) -> bool:
                return enabled

            @tool
            async def stateful(state: State, value: int) -> int:
                return value

            class Args(BaseModel):
                value: str

            async def handle(state: State, value: str) -> str:
                return value

            class Worker:
                async def handle(self, *, value: str) -> str:
                    return value

            def synchronous(value: int) -> int:
                return value

            explicit = Tool.from_model(Args, handle)
            bound = tool(Worker().handle)
            updated = tool(bound, description="Updated")
            restricted = echo.with_parameters(value={"type": "string", "enum": ["ok"]})

            async def check() -> None:
                assert_type(await add.handler(1, right=2), int)
                assert_type(await flag.handler(enabled=True), bool)
                assert_type(await stateful.handler({}, value=1), int)
                assert_type(await explicit.handler({}, value="ok"), str)
                assert_type(await bound.handler(value="ok"), str)
                assert_type(await restricted.handler(value="ok"), str)
                assert_type(await updated.handler(value="ok"), str)
                await add.handler("bad")
                await restricted.handler(unknown="bad")
                incorrect: str = await add.handler(1)
                incompatible: Tool[[str], str] = add
                tool(synchronous)
                tool(name="invalid")(synchronous)
                Tool.from_model(Args, synchronous)
                await bound.handler(value=1)
            ''')
        )
    stdout, stderr, status = mypy_api.run(
        [
            "--no-incremental",
            "--no-error-summary",
            str(source),
            str(Path(__file__).parent.parent / "src" / "tiny_coder"),
        ]
    )
    assert status == 1 and not stderr, stdout + stderr
    errors = [line for line in stdout.splitlines() if ": error:" in line]
    assert len(errors) == 8, stdout
    assert all("tool_typing.py:" in error for error in errors), stdout
    assert 'incompatible type "str"' in errors[0]
    assert 'Unexpected keyword argument "unknown"' in errors[1]
    assert all("Incompatible types in assignment" in error for error in errors[2:4])
    assert all("incompatible type" in error and "Coroutine" in error for error in errors[4:7])
    assert 'incompatible type "int"' in errors[7]


def test_parameter_overrides_are_isolated_and_enforced() -> None:
    restricted = add_typed.with_parameters(left={"type": "integer", "enum": [1, 2]})
    assert "enum" not in add_typed.parameters["properties"]["left"]
    assert "[1, 2]" in asyncio.run(restricted.invoke({}, '{"left":3}'))
    assert asyncio.run(restricted.invoke({}, '{"left":2}')) == '{"sum": 4}'
    with pytest.raises(ValueError, match="unknown parameter"):
        restricted.with_parameters(typo={"enum": [1]})
    # An overridden wire field cannot disable the model's original validation.
    invalid = add_typed.with_parameters(left={"type": "integer", "enum": [0]})
    state: State = {}
    assert "greater than or equal to 1" in asyncio.run(invalid.invoke(state, '{"left":0}'))
    assert not state


def test_existing_tool_updates_reuse_model_and_preserve_original_constraints() -> None:
    assert tool(add_typed) is add_typed
    assert add_typed.with_parameters() is add_typed
    updated = tool(
        add_typed,
        name="restricted_add",
        description="Only add one.",
        parameters={"left": {"type": "integer", "enum": [1]}},
    )
    assert updated.model is add_typed.model
    assert updated.handler is add_typed.handler
    assert updated.description == "Only add one."
    assert updated.name == "restricted_add"
    assert add_typed.name == "add_typed"
    assert "enum" not in add_typed.parameters["properties"]["left"]
    assert "[1]" in asyncio.run(updated.invoke({}, '{"left":2}'))
    assert json.loads(asyncio.run(updated.invoke({}, '{"left":1}'))) == {"sum": 3}


def test_decorator_rejects_ambiguous_definitions_before_registration() -> None:
    async def missing_annotation(value):  # type: ignore[no-untyped-def]
        return value

    async def variadic(*args: str) -> str:
        return ""

    async def positional(value: str, /) -> str:
        return value

    async def incorrect_state(state: int) -> str:
        return str(state)

    for handler, message in (
        (missing_annotation, "type annotation"),
        (variadic, "named argument"),
        (positional, "named argument"),
        (incorrect_state, "annotated as State"),
    ):
        with pytest.raises(TypeError, match=message):
            tool(handler)


def _add_tool(calls: list[tuple[int, int]] | None = None) -> Tool[..., str]:
    async def handler(state: State, left: int, right: int) -> str:
        _ = state
        if calls is not None:
            calls.append((left, right))
        return str(left + right)

    return Tool.from_model(AddArgs, handler, name="add")


def _assistant_msg(tool_name: str, arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }


def test_tool_from_model_builds_openai_schema() -> None:
    tool = _add_tool()
    assert tool.name == "add"
    assert tool.description == "Add two integers."
    params = tool.parameters
    assert params["properties"]["left"]["type"] == "integer"
    assert params["properties"]["right"]["type"] == "integer"
    assert set(params["required"]) == {"left", "right"}


def test_tool_from_model_infers_snake_name() -> None:
    async def handler(state: State, **kw: object) -> str:
        _ = state, kw
        return "0"

    tool = Tool.from_model(AddArgs, handler)
    assert tool.name == "add_args"


def test_tool_from_model_requires_base_model_subclass() -> None:
    with pytest.raises(TypeError, match="BaseModel"):
        Tool.from_model(object(), lambda s, **kw: "")  # type: ignore[arg-type]


def test_tool_from_model_rejects_blank_name() -> None:
    async def handler(state: State, **kw: object) -> str:
        _ = state, kw
        return ""

    with pytest.raises(ValueError, match="tool name must not be blank"):
        Tool(name="", description="", model=AddArgs, handler=handler)


def test_tool_invoke_validates_and_executes_handler() -> None:
    tool = _add_tool()
    result = asyncio.run(tool.invoke({}, json.dumps({"left": 2, "right": 3})))
    assert result == "5"


def test_tool_invoke_returns_error_for_invalid_json() -> None:
    tool = _add_tool()
    result = asyncio.run(tool.invoke({}, "not-json"))
    assert result.startswith("error: invalid JSON arguments")


def test_tool_invoke_returns_error_for_validation_failure() -> None:
    tool = _add_tool()
    result = asyncio.run(tool.invoke({}, json.dumps({"left": "x", "right": 3})))
    assert result.startswith("error: invalid arguments")


def test_tool_validation_feedback_identifies_missing_fields_without_echoing_large_input() -> None:
    calls: list[tuple[int, int]] = []
    tool = _add_tool(calls)
    state: State = {"pending": ["existing edit"]}
    result = asyncio.run(tool.invoke(state, json.dumps({"left": "x" * 5000})))
    feedback = json.loads(result.removeprefix("error: invalid arguments: "))
    assert feedback["tool"] == "add"
    assert {(tuple(error["loc"]), error["type"]) for error in feedback["errors"]} == {
        (("left",), "type"),
        (("right",), "missing"),
    }
    assert feedback["required_fields"] == ["left", "right"]
    assert feedback["provided_fields"] == ["left"]
    assert feedback["executed"] is False
    assert "x" * 100 not in result
    assert not calls
    assert state == {"pending": ["existing edit"]}


def test_tool_invoke_returns_error_for_handler_exception() -> None:
    async def broken_handler(state: State) -> str:
        _ = state
        raise RuntimeError("boom")

    tool = Tool.from_model(NoopArgs, broken_handler, name="broken")
    result = asyncio.run(tool.invoke({}, "{}"))
    assert result.startswith("error: RuntimeError")


def test_tool_node_executes_tool_calls_and_appends_results() -> None:
    calls: list[tuple[int, int]] = []
    node = ToolNode([_add_tool(calls)])
    state: dict[str, list[dict[str, Any]]] = {
        "messages": [
            {"role": "user", "content": "add"},
            _assistant_msg("add", json.dumps({"left": 2, "right": 3})),
        ]
    }
    update = asyncio.run(node(state))
    assert calls == [(2, 3)]
    assert update["messages"] == [
        {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "5"}
    ]


@pytest.mark.parametrize("cancel", [False, True])
def test_tool_execution_records_handler_errors_and_cancellation(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    clock = [0.0]
    executions: list[ToolExecution] = []

    async def handler(state: State) -> str:
        clock[0] += 1.5
        if cancel:
            raise asyncio.CancelledError
        raise RuntimeError("handler failed")

    async def record(execution: ToolExecution) -> None:
        executions.append(execution)

    monkeypatch.setattr("tiny_coder.tool_node.perf_counter", lambda: clock[0])
    node = ToolNode([Tool.from_model(NoopArgs, handler, name="fail")], on_tool_finished=record)
    state: State = {"messages": [_assistant_msg("fail", "{}")]}
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(node(state))
    else:
        update = asyncio.run(node(state))
        assert update["messages"][0]["content"] == "error: RuntimeError: handler failed"
    assert len(executions) == 1
    execution = executions[0]
    assert execution.tool_call_id == "call_1"
    assert execution.name == "fail"
    assert execution.started_at and execution.completed_at
    assert execution.duration_ms == 1500.0
    assert execution.status == ("cancelled" if cancel else "error")
    assert execution.error == (None if cancel else "error: RuntimeError: handler failed")


def test_tool_node_returns_error_for_unknown_tool() -> None:
    node = ToolNode([_add_tool()])
    state: dict[str, list[dict[str, Any]]] = {
        "messages": [{"role": "user", "content": "hi"}, _assistant_msg("nonexistent", "{}")]
    }
    update = asyncio.run(node(state))
    assert "unknown tool" in update["messages"][0]["content"]


def test_tool_node_returns_error_for_invalid_arguments() -> None:
    node = ToolNode([_add_tool()])
    state: dict[str, list[dict[str, Any]]] = {
        "messages": [
            {"role": "user", "content": "hi"},
            _assistant_msg("add", json.dumps({"left": "x", "right": 3})),
        ]
    }
    update = asyncio.run(node(state))
    assert "error" in update["messages"][0]["content"].lower()


def test_tool_node_returns_empty_when_no_tool_calls() -> None:
    node = ToolNode([_add_tool()])
    update = asyncio.run(node({"messages": [{"role": "user", "content": "hi"}]}))
    assert update == {}


def test_tool_node_returns_empty_without_messages_key() -> None:
    node = ToolNode([_add_tool()])
    update = asyncio.run(node({}))
    assert update == {}


def test_tool_node_returns_empty_when_messages_not_a_list() -> None:
    node = ToolNode([_add_tool()])
    update = asyncio.run(node({"messages": "not a list"}))
    assert update == {}


def test_has_tool_calls_detects_assistant_tool_calls() -> None:
    assert has_tool_calls(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "add"}}],
                }
            ]
        }
    )
    assert not has_tool_calls({"messages": [{"role": "user", "content": "hi"}]})
    assert not has_tool_calls({"messages": []})
    assert not has_tool_calls({})


def test_tool_node_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolNode([_add_tool(), _add_tool()])


def test_tool_node_rejects_non_tool_instance() -> None:
    with pytest.raises(TypeError, match="Tool instances"):
        ToolNode([object()])  # type: ignore[list-item]


class ItemArgs(BaseModel):
    name: str = Field(min_length=2)
    count: int = Field(ge=1, le=3)
    mode: Literal["read", "write"]


class BatchArgs(BaseModel):
    items: list[ItemArgs] = Field(min_length=1)
    limit: int = 10


@pytest.mark.parametrize(
    ("payload", "location", "advice"),
    [
        ({}, ["items"], "Add `items`"),
        ({"items": []}, ["items"], "at least 1 items"),
        ({"items": "[]"}, ["items"], "JSON type array"),
        (
            {"items": [{"name": "ok", "count": "2", "mode": "read"}]},
            ["items", 0, "count"],
            "JSON type integer",
        ),
        (
            {"items": [{"name": "ok", "count": True, "mode": "read"}]},
            ["items", 0, "count"],
            "JSON type integer",
        ),
        ({"items": [{"name": "ok", "count": 0, "mode": "read"}]}, ["items", 0, "count"], ">= 1"),
        ({"items": [{"name": "ok", "count": 4, "mode": "read"}]}, ["items", 0, "count"], "<= 3"),
        (
            {"items": [{"name": "x", "count": 2, "mode": "read"}]},
            ["items", 0, "name"],
            "at least 2 characters",
        ),
        (
            {"items": [{"name": "ok", "count": 2, "mode": "other"}]},
            ["items", 0, "mode"],
            '["read", "write"]',
        ),
        (
            {"items": [{"name": "ok", "count": 2, "mode": "read", "stray": 1}]},
            ["items", 0, "stray"],
            "Remove `items.0.stray`",
        ),
        (
            {"items": [{"name": "ok", "count": 2, "mode": "read"}], "stray": 1},
            ["stray"],
            "Remove `stray`",
        ),
    ],
)
def test_all_tool_arguments_are_checked_before_execution_with_field_specific_advice(
    payload: dict[str, Any],
    location: list[str | int],
    advice: str,
) -> None:
    async def handler(state: State, **kwargs: Any) -> str:
        state["executed"] = True
        return "unexpected"

    state: State = {"pending": ["retained"]}
    result = asyncio.run(
        Tool.from_model(BatchArgs, handler, name="batch").invoke(state, json.dumps(payload))
    )
    feedback = json.loads(result.removeprefix("error: invalid arguments: "))
    failure = next(error for error in feedback["errors"] if error["loc"] == location)
    assert advice in failure["suggestion"]
    assert "batch:" in failure["suggestion"]
    assert feedback["executed"] is False
    assert state == {"pending": ["retained"]}


def test_tool_wire_schema_is_strict_recursively_and_defaults_remain_usable_locally() -> None:
    calls = []

    async def handler(state: State, **kwargs: Any) -> str:
        calls.append(kwargs)
        return "accepted"

    tool = Tool.from_model(BatchArgs, handler)
    wire = tool_openai_spec(tool)
    assert wire["strict"] is True
    parameters = wire["parameters"]
    assert parameters["required"] == ["items", "limit"]
    assert parameters["additionalProperties"] is False
    item_schema = parameters["properties"]["items"]["items"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["required"] == ["name", "count", "mode"]
    assert tool.parameters["required"] == ["items"]
    assert (
        asyncio.run(tool.invoke({}, '{"items":[{"name":"ok","count":2,"mode":"read"}]}'))
        == "accepted"
    )
    assert calls[0]["limit"] == 10


def test_custom_parameter_enum_is_enforced_locally() -> None:
    async def handler(state: State, left: int, right: int) -> str:
        state["called"] = True
        return "unexpected"

    schema = AddArgs.model_json_schema()
    schema["properties"]["left"]["enum"] = [1, 2]
    tool = Tool(
        name="add", description="Add", model=AddArgs, handler=handler, parameters_schema=schema
    )
    state: State = {}
    result = asyncio.run(tool.invoke(state, '{"left":3,"right":1}'))
    assert "[1, 2]" in result
    assert state == {}


@pytest.mark.parametrize(
    ("arguments", "advice"),
    [
        ('{"left":1,', "line 1, column"),
        ('[1,2]', "JSON object with named arguments"),
        ('{"left":1,"left":2,"right":1}', "duplicate argument key"),
        ('{"left":NaN,"right":1}', "NaN is not a JSON number"),
    ],
)
def test_invalid_argument_json_reports_the_exact_repair(arguments: str, advice: str) -> None:
    calls: list[tuple[int, int]] = []
    result = asyncio.run(_add_tool(calls).invoke({}, arguments))
    assert advice in result
    assert not calls


def test_multiple_missing_arguments_are_reported_once_each() -> None:
    result = asyncio.run(_add_tool().invoke({}, '{}'))
    feedback = json.loads(result.removeprefix("error: invalid arguments: "))
    assert [failure["loc"] for failure in feedback["errors"]] == [["left"], ["right"]]
