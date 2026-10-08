from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, eq=False)
class Event(Generic[PayloadT]):
    """An identity-based event key with a statically typed payload."""

    name: str


class EventBus:
    """Dispatch events in subscription order, awaiting each handler.

    Dispatch uses a subscriber snapshot. Handler errors propagate to the publisher;
    no background tasks or queues are created by the bus.
    """

    def __init__(self) -> None:
        # Each key fixes its handlers' payload type through subscribe().
        self._handlers: dict[Event[Any], list[Callable[[Any], Awaitable[None]]]] = {}

    def subscribe(
        self, event: Event[PayloadT], handler: Callable[[PayloadT], Awaitable[None]]
    ) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def unsubscribe(
        self, event: Event[PayloadT], handler: Callable[[PayloadT], Awaitable[None]]
    ) -> None:
        handlers = self._handlers.get(event)
        if handlers is not None and handler in handlers:
            handlers.remove(handler)
            if not handlers:
                del self._handlers[event]

    def has_subscribers(self, event: Event[PayloadT]) -> bool:
        return bool(self._handlers.get(event))

    async def emit(self, event: Event[PayloadT], payload: PayloadT) -> None:
        for handler in tuple(self._handlers.get(event, ())):
            await handler(payload)
