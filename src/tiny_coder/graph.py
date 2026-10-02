"""Graph execution: nodes, edges, conditional routing, reducers, and step limits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tiny_coder.checkpoint import Checkpoint, Checkpointer
from tiny_coder.state import NodeFn, Reducer, State, add_messages

END = "__end__"


class GraphError(RuntimeError):
    """Invalid graph configuration or execution failure."""


@dataclass(kw_only=True)
class Graph:
    """Orchestrate node execution with edges and checkpointing."""

    max_steps: int = 100
    _nodes: dict[str, NodeFn] = field(default_factory=dict, init=False, repr=False)
    _edges: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _conditional: dict[str, tuple[Callable[[State], str], dict[str, str]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _reducers: dict[str, Reducer] = field(
        default_factory=lambda: {"messages": add_messages}, init=False, repr=False
    )
    _entry: str | None = field(default=None, init=False, repr=False)

    def add_node(self, name: str, fn: NodeFn) -> None:
        if not name or not name.strip():
            raise ValueError("node name must not be blank")
        if name in self._nodes:
            raise ValueError(f"duplicate node: {name!r}")
        self._nodes[name] = fn
        if self._entry is None:
            self._entry = name

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._require_node(from_node)
        if to_node != END:
            self._require_node(to_node)
        self._edges[from_node] = to_node

    def add_conditional_edges(
        self,
        from_node: str,
        router: Callable[[State], str],
        path_map: dict[str, str],
    ) -> None:
        self._require_node(from_node)
        for key, target in path_map.items():
            if target != END:
                self._require_node(target)
        self._conditional[from_node] = (router, path_map)

    def set_entry(self, name: str) -> None:
        self._require_node(name)
        self._entry = name

    def set_reducer(self, key: str, reducer: Reducer) -> None:
        self._reducers[key] = reducer

    async def run(
        self,
        state: State | None = None,
        *,
        checkpointer: Checkpointer | None = None,
        resume: bool = False,
        max_steps: int | None = None,
    ) -> State:
        limit = self.max_steps if max_steps is None else max_steps
        if limit < 1:
            raise ValueError("max_steps must be at least 1")
        self._validate()
        assert self._entry is not None  # _validate() ensures this
        if resume:
            if checkpointer is None:
                raise GraphError("resume requires a checkpointer")
            checkpoint = await checkpointer.load()
            if checkpoint is None:
                raise GraphError("no checkpoint found to resume")
            merged = dict(checkpoint.state)
            next_node = checkpoint.next_node
            step = checkpoint.step
        else:
            if state is None:
                raise GraphError("state is required when resume=False")
            merged = dict(state)
            next_node = self._entry
            step = 0
            if checkpointer is not None:
                await checkpointer.save(Checkpoint(step=0, next_node=next_node, state=merged))
        steps = 0
        while next_node != END:
            if next_node not in self._nodes:
                raise GraphError(f"unknown node: {next_node!r}")
            if steps >= limit:
                raise GraphError(f"graph exceeded max_steps={limit} at node {next_node!r}")
            fn = self._nodes[next_node]
            update = await fn(merged)
            self._apply(merged, update)
            step += 1
            steps += 1
            following = self._resolve_next(next_node, merged)
            if checkpointer is not None:
                await checkpointer.save(Checkpoint(step=step, next_node=following, state=merged))
            next_node = following
        return merged

    def _require_node(self, name: str) -> None:
        if name not in self._nodes and name != END:
            raise GraphError(f"undefined node: {name!r}")

    def _resolve_next(self, current: str, state: State) -> str:
        if current in self._conditional:
            router, path_map = self._conditional[current]
            key = router(state)
            if key not in path_map:
                raise GraphError(f"router for {current!r} returned {key!r} not in path map")
            return path_map[key]
        if current in self._edges:
            return self._edges[current]
        return END

    def _apply(self, state: State, update: Mapping[str, Any]) -> None:
        if not isinstance(update, Mapping):
            raise GraphError("node must return a dict-like update")
        for key, value in update.items():
            reducer = self._reducers.get(key)
            if reducer is not None:
                state[key] = reducer(state.get(key), value)
            else:
                state[key] = value

    def _validate(self) -> None:
        if self._entry is None:
            raise GraphError("graph has no nodes; add at least one node")
        for node in self._edges:
            self._require_node(node)
        for node, (_, path_map) in self._conditional.items():
            self._require_node(node)


__all__ = [
    "END",
    "Graph",
    "GraphError",
]
