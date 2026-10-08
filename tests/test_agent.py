from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest
from test_structured_agent import SampleLlmConfig, SampleStructured, _outcome

from tiny_coder.agent import Agent, Invocation
from tiny_coder.llm import LlmChatOutcome
from tiny_coder.structured_agent import (
    AFTER_CALL,
    AgentCall,
    StructuredAgent,
    StructuredInput,
    create_structured_agent,
)


class Double(Agent[int, int]):
    async def _invoke(self, prompt: int) -> int:
        return prompt * 2


def test_invoke_and_send_publish_completed_tasks_on_each_agents_bus() -> None:
    observed: list[tuple[int, int]] = []

    async def receive(invocation: Invocation[int, int]) -> None:
        observed.append((invocation.input, invocation.output))

    async def run() -> None:
        async with Main() as parent:
            child = parent.spawn("calculator", Double)
            child.events.subscribe(child.invoked, receive)
            assert await parent.invoke(3) == 7
            assert await parent.send(4) == 9
            assert observed == [(3, 6), (4, 8)]
            assert parent.spawn("calculator", Double) is child
            assert child.events is not parent.events

    asyncio.run(run())


def test_completion_subscriber_can_dispatch_and_await_a_reused_sibling() -> None:
    reviewed: list[int] = []

    async def run() -> None:
        async with Agent[int, int]() as main:
            writer = main.spawn("writer", Double)
            reviewer = main.spawn("reviewer", Double)

            async def review(invocation: Invocation[int, int]) -> None:
                reviewed.append(await reviewer.send(invocation.output))

            writer.events.subscribe(writer.invoked, review)
            assert await writer.send(3) == 6
            assert reviewed == [12]
            assert await writer.invoke(4) == 8
            assert reviewed == [12, 16]
            assert main.spawn("reviewer", Double) is reviewer

    asyncio.run(run())


def test_completion_subscribers_keep_child_restrictions() -> None:
    async def run() -> None:
        async with Agent[int, int]() as main:
            child = main.spawn("writer", Double)

            async def spawn(invocation: Invocation[int, int]) -> None:
                main.spawn("forbidden", Double)

            child.events.subscribe(child.invoked, spawn)
            with pytest.raises(RuntimeError, match="child agents cannot spawn"):
                await child.invoke(3)
            assert tuple(main.children) == ("writer",)

    asyncio.run(run())


@dataclass
class Labeled(Agent[int, int]):
    label: str

    async def _invoke(self, prompt: int) -> int:
        return prompt


def test_agent_instances_keep_identity_equality_and_separate_children() -> None:
    left = Agent[int, int]()
    right = Agent[int, int]()

    assert left is not right
    assert left != right
    left.spawn("worker", Double)
    assert tuple(left.children) == ("worker",)
    assert not right.children


def test_dataclass_subclass_equality_ignores_agent_lifecycle_state() -> None:
    left = Labeled("same")
    right = Labeled("same")
    different = Labeled("different")

    assert left == right
    assert left != different
    left.name = "renamed"
    left.spawn("worker", Double)
    assert left == right


def test_spawn_reuses_child_and_awaits_results() -> None:
    created: list[Double] = []

    def factory() -> Double:
        child = Double()
        created.append(child)
        return child

    async def run() -> None:
        async with Agent[int, int]() as main:
            for value in range(10):
                child = main.spawn("worker", factory)
                assert child is created[0]
                assert child.name == "worker"
                task = child.send(value)
                assert await task == value * 2
                assert task.done()
            assert tuple(main.children) == ("worker",)

    asyncio.run(run())
    assert len(created) == 1


class Blocking(Agent[int, int]):
    def __init__(self, release: asyncio.Event, started: list[int]) -> None:
        super().__init__()
        self.release = release
        self.started = started
        self.finished = False

    async def _invoke(self, prompt: int) -> int:
        self.started.append(prompt)
        try:
            if prompt == 1:
                await self.release.wait()
            return prompt
        finally:
            self.finished = True


def test_children_are_serial_and_independent_children_run_concurrently() -> None:
    async def run() -> None:
        release = asyncio.Event()
        started: list[int] = []
        async with Agent[int, int]() as main:
            left = main.spawn("left", partial(Blocking, release, started))
            right = main.spawn("right", partial(Blocking, release, started))
            first = left.send(1)
            second = left.send(2)
            third = right.send(3)
            assert await asyncio.wait_for(third, 1) == 3
            assert started == [1, 3]
            release.set()
            results = await asyncio.gather(first, second)
            assert list(results) == [1, 2]
            assert started == [1, 3, 2]

    asyncio.run(run())


class Main(Agent[int, int]):
    async def _invoke(self, prompt: int) -> int:
        calculator = self.spawn("calculator", Double)
        return await calculator.send(prompt) + 1


def test_main_agent_can_spawn_and_wait_during_invoke() -> None:
    async def run() -> None:
        async with Main() as main:
            assert await main.send(3) == 7
            assert await main.send(4) == 9
            assert tuple(main.children) == ("calculator",)

    asyncio.run(run())


class SpawnAttempt(Agent[int, int]):
    def __init__(self, target: Agent[int, int] | None = None) -> None:
        super().__init__()
        self.target = target

    async def _invoke(self, prompt: int) -> int:
        target = self if self.target is None else self.target
        target.spawn("grandchild", Double)
        return prompt


@pytest.mark.parametrize("through_parent", [False, True])
@pytest.mark.parametrize("method", ["invoke", "send"])
def test_child_cannot_spawn_even_through_parent_reference(
    through_parent: bool, method: str
) -> None:
    async def run() -> None:
        async with Agent[int, int]() as main:
            child = main.spawn("child", partial(SpawnAttempt, main if through_parent else None))
            with pytest.raises(RuntimeError, match="child agents cannot spawn"):
                child.spawn("grandchild", Double)
            with pytest.raises(RuntimeError, match="child agents cannot spawn"):
                if method == "invoke":
                    await child.invoke(input=1)
                else:
                    await child.send(input=1)
            assert not child.children
            assert tuple(main.children) == ("child",)
            assert await main.spawn("calculator", Double).send(2) == 4

    asyncio.run(run())


@pytest.mark.parametrize("error", [ValueError("invalid task"), asyncio.CancelledError()])
def test_failed_task_preserves_following_tasks_and_child(error: BaseException) -> None:
    class Failing(Agent[int, int]):
        async def _invoke(self, prompt: int) -> int:
            if prompt == 1:
                raise error
            return prompt

    async def run() -> None:
        async with Agent[int, int]() as main:
            child = main.spawn("worker", Failing)
            failed = child.send(1)
            following = child.send(2)
            with pytest.raises(type(error)):
                await failed
            assert await asyncio.wait_for(following, 1) == 2
            assert main.spawn("worker", Failing) is child

    asyncio.run(run())


def test_shielded_wait_can_be_cancelled_without_discarding_task() -> None:
    async def run() -> None:
        release = asyncio.Event()
        started: list[int] = []
        async with Agent[int, int]() as main:
            child = main.spawn("worker", partial(Blocking, release, started))
            task = child.send(1)
            waiter = asyncio.shield(task)
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not task.done()
            release.set()
            assert await task == 1

    asyncio.run(run())


@pytest.mark.parametrize("start_child", [False, True])
def test_close_cancels_active_and_queued_tasks_and_joins_children(start_child: bool) -> None:
    async def run() -> None:
        release = asyncio.Event()
        started: list[int] = []
        child = Blocking(release, started)
        main = Agent[int, int]()
        main.spawn("worker", lambda: child)
        active = child.send(1)
        queued = child.send(2)
        if start_child:
            await asyncio.sleep(0)
            assert started == [1]
        await main.close()
        assert child.finished == start_child
        for task in (active, queued):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not any(task.get_name() == "agent:worker" for task in asyncio.all_tasks())
        with pytest.raises(RuntimeError, match="closed"):
            main.spawn("other", Double)
        with pytest.raises(RuntimeError, match="closed"):
            _ = child.send(3)
        await main.close()

    asyncio.run(run())


def test_spawn_rejects_blank_names_shared_ownership_and_existing_descendants() -> None:
    async def run() -> None:
        async with Agent[int, int]() as left, Agent[int, int]() as right:
            with pytest.raises(ValueError, match="must not be blank"):
                left.spawn(" ", Double)
            child = left.spawn("worker", Double)
            with pytest.raises(ValueError, match="unowned"):
                right.spawn("shared", lambda: child)
            with pytest.raises(ValueError, match="unowned"):
                right.spawn("nested", lambda: left)
            assert await child.send(1) == 2

    asyncio.run(run())


def test_structured_agent_spawns_reusable_structured_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        return _outcome(kwargs["messages"][-1]["content"].upper())

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)

    async def run() -> None:
        async with create_structured_agent(llm_config=SampleLlmConfig()) as main:
            factory = partial(
                create_structured_agent,
                llm_config=SampleLlmConfig(),
            )
            child = main.spawn("writer", factory)
            assert (await child.send(StructuredInput(prompt="one"))).text == "ONE"
            assert main.spawn("writer", factory) is child
            assert (await child.send(StructuredInput(prompt="two"))).text == "TWO"
            with pytest.raises(RuntimeError, match="child agents cannot spawn"):
                child.spawn("nested", factory)

    asyncio.run(run())


def test_invoke_and_send_accept_the_same_input_for_all_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[tuple[object, int]] = []

    async def observe(call: AgentCall) -> None:
        contexts.append((call.context, call.attempt))

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        assert kwargs["messages"] == [
            {"role": "system", "content": "custom system"},
            {"role": "user", "content": "history"},
            {"role": "user", "content": "prompt"},
        ]
        assert kwargs["llm_cfg"].temperature == 0.7
        assert kwargs["response_format"]["json_schema"]["schema"]["title"] == "SampleStructured"
        return _outcome('{"name":"sample","count":1}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)

    def factory() -> StructuredAgent:
        child = create_structured_agent(llm_config=SampleLlmConfig(), max_steps=1)
        child.events.subscribe(AFTER_CALL, observe)
        return child

    async def run() -> None:
        async with Double() as plain:
            assert await plain.invoke(input=3) == 6
            assert await plain.send(input=3) == 6
        with pytest.raises(RuntimeError, match="closed"):
            await plain.invoke(input=3)

        async with create_structured_agent(llm_config=SampleLlmConfig()) as main:
            child = main.spawn("worker", factory)
            input = StructuredInput(
                prompt="prompt",
                messages=[{"role": "user", "content": "history"}],
                system_prompt="custom system",
                llm_config=SampleLlmConfig(temperature=0.7),
                response_model=SampleStructured,
                context="task",
            )
            direct = await child.invoke(input=input)
            scheduled = await child.send(input=input)
            assert direct.model(SampleStructured) == SampleStructured(name="sample", count=1)
            assert scheduled.model(SampleStructured) == direct.model(SampleStructured)
            assert input.messages == [{"role": "user", "content": "history"}]
        with pytest.raises(RuntimeError, match="closed"):
            await child.invoke(input=input)

    asyncio.run(run())
    assert contexts == [("task", 1), ("task", 1)]
