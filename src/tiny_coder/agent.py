from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType, TracebackType
from typing import Generic, Self, TypeVar, final

from tiny_coder.eventbus import Event, EventBus

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

_CHILD_RUNNING: ContextVar[bool] = ContextVar("child_agent_running", default=False)


@dataclass(eq=False)
class Invocation(Generic[InputT, OutputT]):
    """One input and its output, populated by the agent's event handler."""

    input: InputT
    output: OutputT = field(init=False, repr=False)


@dataclass(eq=False, repr=False)
class Agent(Generic[InputT, OutputT]):
    """An invokable agent that owns reusable, serially scheduled child agents."""

    name: str = field(default="main", init=False, repr=False, compare=False)
    events: EventBus = field(default_factory=EventBus, init=False, repr=False, compare=False)
    invoked: Event[Invocation[InputT, OutputT]] = field(
        default_factory=lambda: Event("agent.invoked"), init=False, repr=False, compare=False
    )
    _invoke_event: Event[Invocation[InputT, OutputT]] = field(
        default_factory=lambda: Event("agent.invoke"), init=False, repr=False, compare=False
    )
    _parent: Agent[InputT, OutputT] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _children: dict[str, Agent[InputT, OutputT]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _pending: set[asyncio.Task[OutputT]] = field(
        default_factory=set, init=False, repr=False, compare=False
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.events.subscribe(self._invoke_event, self._receive)

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def children(self) -> Mapping[str, Agent[InputT, OutputT]]:
        return MappingProxyType(self._children)

    @final
    async def invoke(self, input: InputT) -> OutputT:
        """Execute one input with the agent's lifecycle and child restrictions."""
        invocation = Invocation[InputT, OutputT](input)
        await self.events.emit(self._invoke_event, invocation)
        return invocation.output

    async def _receive(self, invocation: Invocation[InputT, OutputT]) -> None:
        self._ensure_open()
        token = _CHILD_RUNNING.set(self._parent is not None or _CHILD_RUNNING.get())
        try:
            invocation.output = await self._invoke(invocation.input)
            await self.events.emit(self.invoked, invocation)
        finally:
            _CHILD_RUNNING.reset(token)

    async def _invoke(self, input: InputT, /) -> OutputT:
        """Handle one task. Subclasses implement the actual work."""
        raise NotImplementedError("this agent has no task handler")

    def spawn(
        self, name: str, factory: Callable[[], Agent[InputT, OutputT]]
    ) -> Agent[InputT, OutputT]:
        """Create a child once per name. Only a main agent may spawn children."""
        self._ensure_open()
        if self._parent is not None or _CHILD_RUNNING.get():
            raise RuntimeError("child agents cannot spawn agents")
        if not name.strip():
            raise ValueError("agent name must not be blank")
        if name not in self._children:
            child = factory()
            if not isinstance(child, Agent):
                raise TypeError("factory must return an Agent")
            if child is self or child._parent is not None or child._children:
                raise ValueError("child must be an unowned agent without children")
            child._ensure_open()
            child.name = name
            child._parent = self
            self._children[name] = child
        return self._children[name]

    def send(self, input: InputT) -> asyncio.Task[OutputT]:
        """Schedule invoke and return its task. Calls to this agent run serially."""
        self._ensure_open()
        task = asyncio.create_task(self._execute(input), name=f"agent:{self.name}")
        self._pending.add(task)
        task.add_done_callback(self._finished)
        return task

    async def _execute(self, input: InputT) -> OutputT:
        async with self._lock:
            return await self.invoke(input)

    def _finished(self, task: asyncio.Task[OutputT]) -> None:
        self._pending.discard(task)
        # Observe abandoned failures; awaiting the task still raises its error.
        if not task.cancelled():
            task.exception()

    async def close(self) -> None:
        """Cancel and join outstanding tasks and all owned children."""
        agents = (self, *self._children.values())
        pending = [task for agent in agents for task in agent._pending]
        for agent in agents:
            agent._closed = True
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _ensure_open(self) -> None:
        if self._closed or (self._parent is not None and self._parent._closed):
            raise RuntimeError("agent is closed")


__all__ = ["Agent", "Invocation"]
