from __future__ import annotations

import asyncio

import pytest

from tiny_coder.eventbus import Event, EventBus


def test_handlers_are_awaited_in_order_and_events_are_distinct() -> None:
    bus = EventBus()
    changed = Event[list[int]]("changed")
    unrelated = Event[list[int]]("changed")
    observed: list[list[int]] = []

    async def update(values: list[int]) -> None:
        await asyncio.sleep(0)
        values.append(2)

    async def observe(values: list[int]) -> None:
        observed.append(list(values))

    bus.subscribe(changed, update)
    bus.subscribe(changed, observe)
    bus.subscribe(unrelated, observe)

    async def run() -> None:
        await bus.emit(changed, [1])
        assert observed == [[1, 2]]
        await bus.emit(unrelated, [3])
        assert observed == [[1, 2], [3]]

    asyncio.run(run())


def test_subscriber_changes_apply_to_the_next_dispatch() -> None:
    bus = EventBus()
    event = Event[int]("number")
    observed: list[int] = []

    async def observe(value: int) -> None:
        observed.append(value)

    async def replace(value: int) -> None:
        bus.unsubscribe(event, replace)
        bus.unsubscribe(event, observe)

    bus.subscribe(event, replace)
    bus.subscribe(event, observe)

    async def run() -> None:
        assert bus.has_subscribers(event)
        await bus.emit(event, 1)
        assert observed == [1]
        assert not bus.has_subscribers(event)
        await bus.emit(event, 2)
        bus.unsubscribe(event, observe)
        assert observed == [1]

    asyncio.run(run())


@pytest.mark.parametrize("error", [ValueError("rejected"), asyncio.CancelledError()])
def test_handler_failure_propagates_and_stops_dispatch(error: BaseException) -> None:
    bus = EventBus()
    event = Event[int]("number")
    observed: list[int] = []

    async def reject(value: int) -> None:
        raise error

    async def observe(value: int) -> None:
        observed.append(value)

    bus.subscribe(event, reject)
    bus.subscribe(event, observe)

    async def run() -> None:
        with pytest.raises(type(error)):
            await bus.emit(event, 1)
        assert not observed
        bus.unsubscribe(event, reject)
        await bus.emit(event, 2)
        assert observed == [2]

    asyncio.run(run())


def test_handler_can_await_a_nested_event() -> None:
    bus = EventBus()
    source = Event[int]("source")
    target = Event[str]("target")
    observed: list[str] = []

    async def forward(value: int) -> None:
        await bus.emit(target, str(value))
        observed.append("completed")

    async def receive(value: str) -> None:
        observed.append(value)

    bus.subscribe(source, forward)
    bus.subscribe(target, receive)
    asyncio.run(bus.emit(source, 3))
    assert observed == ["3", "completed"]
