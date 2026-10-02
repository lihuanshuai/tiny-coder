from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from tiny_coder.state import State
from tiny_coder.tool_node import Tool, ToolNode, has_tool_calls


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


def test_tool_invoke_returns_error_for_handler_exception() -> None:
    async def broken_handler(state: State) -> str:
        _ = state
        raise RuntimeError("boom")

    tool = Tool.from_model(NoopArgs, broken_handler, name="broken")
    result = asyncio.run(tool.invoke({}, json.dumps({"x": 1})))
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
