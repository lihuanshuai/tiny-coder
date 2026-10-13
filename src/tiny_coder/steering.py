from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Steering:
    """Supplementary instructions shared by calls within one task scope.

    Child asyncio tasks inherit the scope, including calls to reused agents.
    Instructions apply to future calls only; completed work is not replayed.
    """

    _instructions: list[str] = field(default_factory=list, init=False, repr=False)
    _delivered: int = field(default=0, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    @property
    def pending(self) -> int:
        return len(self._instructions) - self._delivered

    def add(self, instruction: str) -> None:
        if not self._active:
            raise RuntimeError("steering scope is not active")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("steering instruction must not be blank")
        self._instructions.append(instruction)

    def snapshot(self) -> tuple[str, ...]:
        """Mark instructions as delivered to a call without consuming them."""
        if not self._active:
            return ()
        self._delivered = len(self._instructions)
        return tuple(self._instructions)

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Bind a fresh task scope; close it even on cancellation or failure."""
        if self._active:
            raise RuntimeError("steering scope is already active")
        self._instructions.clear()
        self._delivered = 0
        self._active = True
        token = _CURRENT.set(self)
        try:
            yield
        finally:
            self._active = False
            _CURRENT.reset(token)


_CURRENT: ContextVar[Steering | None] = ContextVar("agent_steering", default=None)


def current_instructions() -> tuple[str, ...]:
    steering = _CURRENT.get()
    return steering.snapshot() if steering is not None else ()
