from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from tiny_coder.llm import tool_openai_spec
from tiny_coder.state import State
from tiny_coder.tool_node import Tool, ToolExecution, ToolNode, has_tool_calls


class AddArgs(BaseModel):
    """Add two integers."""

    left: int
    right: int


class NoopArgs(BaseModel):
    """Do nothing."""


def _add_tool(calls: list[tuple[int, int]] | None = None) -> Tool:
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
