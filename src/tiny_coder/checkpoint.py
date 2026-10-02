"""`Checkpointer` protocol and `JsonCheckpointer` for resuming graph runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tiny_coder.state import State


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One resumable snapshot taken after a node finished."""

    step: int
    next_node: str
    state: State

    def to_json(self) -> dict[str, Any]:
        return {"step": self.step, "next_node": self.next_node, "state": self.state}

    @classmethod
    def from_json(cls, payload: object) -> Checkpoint:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a JSON object")
        step = payload.get("step")
        next_node = payload.get("next_node")
        state = payload.get("state")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("checkpoint step must be an integer")
        if not isinstance(next_node, str) or not next_node:
            raise ValueError("checkpoint next_node must be a non-blank string")
        if not isinstance(state, dict):
            raise ValueError("checkpoint state must be a JSON object")
        return cls(step=step, next_node=next_node, state=state)


@runtime_checkable
class Checkpointer(Protocol):
    """Persist and restore one graph run."""

    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def load(self) -> Checkpoint | None: ...


@dataclass
class JsonCheckpointer:
    """Store a single checkpoint as one UTF-8 JSON file."""

    path: Path

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve()

    async def save(self, checkpoint: Checkpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(checkpoint.to_json(), file, ensure_ascii=False, indent=2)
            file.write("\n")

    async def load(self) -> Checkpoint | None:
        if not self.path.is_file():
            return None
        with self.path.open("r", encoding="utf-8", newline="\n") as file:
            payload = json.load(file)
        return Checkpoint.from_json(payload)


__all__ = [
    "Checkpoint",
    "Checkpointer",
    "JsonCheckpointer",
]
