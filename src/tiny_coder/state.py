"""`State` type aliases and the `add_messages` reducer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeAlias

State: TypeAlias = dict[str, Any]
Update: TypeAlias = Mapping[str, Any]
NodeFn: TypeAlias = Callable[[State], Awaitable[Update]]
RouterFn: TypeAlias = Callable[[State], str]
Reducer: TypeAlias = Callable[[Any, Any], Any]


def add_messages(current: Sequence[Any] | None, update: Sequence[Any] | None) -> list[Any]:
    """Reducer that appends new messages to the existing conversation."""
    return [*(current or []), *(update or [])]


__all__ = [
    "NodeFn",
    "Reducer",
    "RouterFn",
    "State",
    "Update",
    "add_messages",
]
