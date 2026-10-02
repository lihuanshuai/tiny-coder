from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tiny_coder.checkpoint import Checkpoint, Checkpointer, JsonCheckpointer
from tiny_coder.graph import END, Graph, GraphError
from tiny_coder.state import State


def _run(
    graph: Graph,
    state: State | None = None,
    *,
    checkpointer: Checkpointer | None = None,
    resume: bool = False,
) -> State:
    return asyncio.run(graph.run(state, checkpointer=checkpointer, resume=resume))


def test_graph_runs_nodes_in_edge_order() -> None:
    calls: list[str] = []

    async def a(state: State) -> dict[str, object]:
        calls.append("a")
        return {"value": state.get("value", 0) + 1}

    async def b(state: State) -> dict[str, object]:
        calls.append("b")
        return {"value": state["value"] * 10}

    graph = Graph()
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge("a", "b")

    result = _run(graph, {"value": 0})
    assert calls == ["a", "b"]
    assert result == {"value": 10}


def test_graph_conditional_edges_route_by_state() -> None:
    graph = Graph()

    async def start(state: State) -> dict[str, object]:
        return {"kind": state["kind"]}

    async def yes_fn(state: State) -> dict[str, object]:
        _ = state
        return {"result": "yes"}

    async def no_fn(state: State) -> dict[str, object]:
        _ = state
        return {"result": "no"}

    graph.add_node("start", start)
    graph.add_node("yes", yes_fn)
    graph.add_node("no", no_fn)
    graph.add_conditional_edges("start", lambda s: s["kind"], {"yes": "yes", "no": "no"})

    result1 = _run(graph, {"kind": "yes"})
    result2 = _run(graph, {"kind": "no"})
    assert result1["result"] == "yes"
    assert result2["result"] == "no"


def test_first_added_node_becomes_entry() -> None:
    graph = Graph()

    async def entry(state: State) -> dict[str, object]:
        return {"result": "ran"}

    graph.add_node("entry", entry)
    result = _run(graph, {})
    assert result["result"] == "ran"


def test_explicit_entry_overrides_default() -> None:
    graph = Graph()

    async def a(state: State) -> dict[str, object]:
        return {"result": "a"}

    async def b(state: State) -> dict[str, object]:
        return {"result": "b"}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.set_entry("b")
    result = _run(graph, {})
    assert result["result"] == "b"


def test_graph_reaches_end_when_no_edges() -> None:
    calls: list[str] = []

    async def a(state: State) -> dict[str, object]:
        calls.append("a")
        return {}

    graph = Graph()
    graph.add_node("a", a)
    _run(graph, {})
    assert calls == ["a"]


def test_graph_appends_messages_by_default() -> None:
    graph = Graph()

    async def first(state: State) -> dict[str, object]:
        return {"messages": [{"role": "user", "content": "hi"}]}

    async def second(state: State) -> dict[str, object]:
        return {"messages": [{"role": "assistant", "content": "yo"}]}

    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge("first", "second")

    result = _run(graph, {})
    assert len(result["messages"]) == 2
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]


def test_graph_custom_reducer_merges_updates() -> None:
    graph = Graph()
    graph.set_reducer("count", lambda current, update: (current or 0) + update)

    async def a(state: State) -> dict[str, object]:
        return {"count": 1}

    async def b(state: State) -> dict[str, object]:
        return {"count": 2}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge("a", "b")
    assert _run(graph, {})["count"] == 3


def test_graph_stops_at_max_steps() -> None:
    graph = Graph(max_steps=3)

    async def loop(state: State) -> dict[str, object]:
        return {"n": state.get("n", 0) + 1}

    graph.add_node("loop", loop)
    graph.add_edge("loop", "loop")
    with pytest.raises(GraphError, match="max_steps=3"):
        _run(graph, {})


def test_graph_rejects_unknown_edge_target() -> None:
    graph = Graph()

    async def a(state: State) -> dict[str, object]:
        return {}

    graph.add_node("a", a)
    with pytest.raises(GraphError, match="undefined node"):
        graph.add_edge("a", "missing")


def test_graph_rejects_duplicate_nodes() -> None:
    graph = Graph()

    async def a(state: State) -> dict[str, object]:
        return {}

    graph.add_node("a", a)
    with pytest.raises(ValueError, match="duplicate node"):
        graph.add_node("a", a)


def test_graph_requires_state_when_not_resuming() -> None:
    graph = Graph()

    async def a(state: State) -> dict[str, object]:
        return {}

    graph.add_node("a", a)
    with pytest.raises(GraphError, match="requires a checkpointer"):
        _run(graph, resume=True)

    with pytest.raises(GraphError, match="state is required"):
        _run(graph, resume=False)


def test_graph_resumes_without_rerunning_previous_nodes(tmp_path: Path) -> None:
    calls: list[str] = []

    async def a(state: State) -> dict[str, object]:
        calls.append("a")
        return {"messages": [{"role": "user", "content": "hi"}]}

    async def b(state: State) -> dict[str, object]:
        calls.append("b")
        return {"messages": [{"role": "assistant", "content": "yo"}]}

    graph = Graph()
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge("a", "b")
    cp = JsonCheckpointer(tmp_path / "cp.json")

    asyncio.run(
        cp.save(
            Checkpoint(
                step=1, next_node="b", state={"messages": [{"role": "user", "content": "hi"}]}
            )
        )
    )
    result = _run(graph, resume=True, checkpointer=cp)
    assert calls == ["b"]
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]


def test_graph_saves_checkpoint_during_normal_run(tmp_path: Path) -> None:
    calls: list[str] = []

    async def a(state: State) -> dict[str, object]:
        calls.append("a")
        return {"n": 1}

    async def b(state: State) -> dict[str, object]:
        calls.append("b")
        return {"n": 2}

    graph = Graph()
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge("a", "b")
    cp = JsonCheckpointer(tmp_path / "cp.json")

    _run(graph, {"n": 0}, checkpointer=cp)
    assert calls == ["a", "b"]
    loaded = asyncio.run(cp.load())
    assert loaded is not None
    assert loaded.step == 2
    assert loaded.next_node == END
    assert loaded.state["n"] == 2
