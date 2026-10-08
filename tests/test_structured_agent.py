from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from tiny_coder.graph import GraphError
from tiny_coder.llm import LlmChatOutcome, LlmConfig, LlmExchange, ToolCall
from tiny_coder.state import State
from tiny_coder.structured_agent import (
    AFTER_CALL,
    BEFORE_CALL,
    CALL_FAILED,
    EXCHANGE_RECEIVED,
    AgentCall,
    StructuredInput,
    StructuredResult,
    create_structured_agent,
)
from tiny_coder.tool_node import Tool


class SampleLlmConfig(LlmConfig):
    base_url: str = "http://localhost:11434/v1"
    llm_model: str = "test"
    num_ctx: int = 4096
    temperature: float = 0.0
    repeat_penalty: float = 1.0
    think: bool = False
    timeout: float = 30.0
    max_output_tokens: int = 100


class SampleStructured(BaseModel):
    """A tiny structured response."""

    name: str
    count: int


class EchoArgs(BaseModel):
    """Repeat text back."""

    text: str


async def _echo(state: State, text: str) -> str:
    _ = state
    return f"echo:{text}"


echo_tool = Tool.from_model(EchoArgs, _echo, name="echo")


def _outcome(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
) -> LlmChatOutcome:
    return LlmChatOutcome(
        text=text,
        tool_calls=tool_calls,
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=10.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )


def _echo_call(text: str) -> ToolCall:
    return ToolCall(
        id="c1",
        type="function",
        function={"name": "echo", "arguments": f'{{"text": "{text}"}}'},
    )


def _fake_llm(
    fake_calls: list[list[dict[str, Any]]],
    queue: list[LlmChatOutcome],
) -> Callable[..., Any]:
    async def fake_stream_llm_chat(*, messages: Any, **_kwargs: Any) -> LlmChatOutcome:
        fake_calls.append(list(messages))
        return queue.pop(0)

    return fake_stream_llm_chat


def test_create_structured_agent_returns_parsed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_calls: list[list[dict[str, Any]]] = []
    queue = [_outcome(text='{"name": "srt", "count": 2}')]
    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", _fake_llm(fake_calls, queue))

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        system_prompt="Return JSON.",
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="go")))

    assert result.model(SampleStructured) == SampleStructured(name="srt", count=2)
    assert [m["role"] for m in result.messages] == ["user", "assistant"]
    assert fake_calls[0][0] == {"role": "system", "content": "Return JSON."}
    assert fake_calls[0][1] == {"role": "user", "content": "go"}


def test_create_structured_agent_accepts_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_calls: list[list[dict[str, Any]]] = []
    queue = [_outcome(text='{"name": "a", "count": 1}')]
    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", _fake_llm(fake_calls, queue))

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
    )
    asyncio.run(
        agent.invoke(StructuredInput(messages=[{"role": "user", "content": "from messages"}]))
    )

    assert fake_calls[0][1] == {"role": "user", "content": "from messages"}


def test_create_structured_agent_overrides_config_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs_seen: list[LlmConfig] = []

    async def recording_fake(*, llm_cfg: LlmConfig, **_kwargs: Any) -> LlmChatOutcome:
        configs_seen.append(llm_cfg)
        return _outcome(text='{"name": "a", "count": 1}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", recording_fake)

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
    )

    retry_config = SampleLlmConfig(temperature=0.5)
    asyncio.run(agent.invoke(StructuredInput(prompt="first")))
    asyncio.run(agent.invoke(StructuredInput(prompt="second", llm_config=retry_config)))

    assert configs_seen[0].temperature == 0.0
    assert configs_seen[1].temperature == 0.5


def test_structured_result_model_returns_model_for_valid_json() -> None:
    result = StructuredResult(
        state={"messages": [{"role": "assistant", "content": '{"name":"a","count":1}'}]}
    )
    assert result.model(SampleStructured) == SampleStructured(name="a", count=1)
    assert result.text == '{"name":"a","count":1}'


def test_structured_result_model_returns_none_for_non_json() -> None:
    result = StructuredResult(state={"messages": [{"role": "assistant", "content": "not json"}]})
    assert result.model(SampleStructured) is None
    assert result.text == "not json"


def test_structured_result_returns_empty_text_without_assistant() -> None:
    assert StructuredResult(state={}).text == ""
    assert StructuredResult(state={}).model(SampleStructured) is None


def test_create_structured_agent_state_excludes_internal_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = [_outcome(text='{"name": "a", "count": 1}')]
    monkeypatch.setattr(
        "tiny_coder.structured_agent.stream_llm_chat",
        _fake_llm([], queue),
    )

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="go")))
    assert "_llm_config" not in result.state


def test_create_structured_agent_works_with_checkpointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tiny_coder.checkpoint import JsonCheckpointer

    queue = [_outcome(text='{"name": "a", "count": 1}')]
    monkeypatch.setattr(
        "tiny_coder.structured_agent.stream_llm_chat",
        _fake_llm([], queue),
    )

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        checkpointer=JsonCheckpointer(tmp_path / "cp.json"),
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="hello")))
    assert result.model(SampleStructured) is not None


def test_create_structured_agent_requires_prompt() -> None:
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
    )
    with pytest.raises(ValueError, match="prompt or messages"):
        asyncio.run(agent.invoke(StructuredInput(prompt="")))


def test_create_structured_agent_routes_tool_calls_then_returns_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_calls: list[list[dict[str, Any]]] = []
    queue = [
        _outcome(tool_calls=(_echo_call("hello"),)),
        _outcome(text='{"name": "done", "count": 1}'),
    ]
    monkeypatch.setattr(
        "tiny_coder.structured_agent.stream_llm_chat",
        _fake_llm(fake_calls, queue),
    )

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        system_prompt="Use echo when asked.",
        tools=[echo_tool],
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="go")))

    assert [tool.name for tool in agent.tools] == ["echo"]
    assert result.model(SampleStructured) == SampleStructured(name="done", count=1)
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    tool_message = next(m for m in result.messages if m["role"] == "tool")
    assert tool_message["content"] == "echo:hello"
    assert len(fake_calls) == 2
    assert fake_calls[1][-1]["role"] == "tool"


def test_create_structured_agent_without_tools_sends_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[Any] = []

    async def recording_fake(*, tools: Any, **_kwargs: Any) -> LlmChatOutcome:
        seen_tools.append(tools)
        return _outcome(text='{"name": "a", "count": 1}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", recording_fake)

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
    )
    asyncio.run(agent.invoke(StructuredInput(prompt="go")))

    assert agent.tools == []
    assert seen_tools == [None]


def test_structured_agent_publishes_exchanges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[LlmExchange] = []
    exchange = LlmExchange(messages=[], response=_outcome(text='{"name": "a", "count": 1}'))

    async def recording_fake(*, on_exchange: Any, **_kwargs: Any) -> LlmChatOutcome:
        await on_exchange(exchange)
        return _outcome(text='{"name": "a", "count": 1}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", recording_fake)

    async def recorder(exchange: LlmExchange) -> None:
        seen.append(exchange)

    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
    )
    agent.events.subscribe(EXCHANGE_RECEIVED, recorder)
    asyncio.run(agent.invoke(StructuredInput(prompt="go")))

    assert seen == [exchange]


def test_hooks_prepare_validate_and_retry_with_updated_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    requests: list[tuple[float, str]] = []
    failures: list[str] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        requests.append((kwargs["llm_cfg"].temperature, kwargs["messages"][-1]["content"]))
        assert kwargs["response_format"]["type"] == "json_schema"
        return _outcome("invalid" if len(requests) == 1 else '{"name":"done","count":1}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        max_steps=2,
    )

    async def prepare(call: AgentCall) -> None:
        events.append(("prepare", call.attempt))
        call.messages = [{"role": "user", "content": "retry" if failures else "first"}]
        call.llm_config = SampleLlmConfig(temperature=0.2 if failures else 0.0)

    async def validate(call: AgentCall) -> None:
        events.append(("validate", call.attempt))
        assert call.outcome is not None
        if call.outcome.model(SampleStructured) is None:
            raise ValueError("invalid response")

    async def retry(call: AgentCall) -> None:
        events.append(("retry", call.attempt))
        assert isinstance(call.error, ValueError)
        assert call.outcome is not None
        failures.append(call.outcome.text)
        call.retry = True

    agent.events.subscribe(BEFORE_CALL, prepare)
    agent.events.subscribe(AFTER_CALL, validate)
    agent.events.subscribe(CALL_FAILED, retry)
    result = asyncio.run(agent.invoke(StructuredInput()))

    assert result.model(SampleStructured) == SampleStructured(name="done", count=1)
    assert requests == [(0.0, "first"), (0.2, "retry")]
    assert failures == ["invalid"]
    assert events == [
        ("prepare", 1),
        ("validate", 1),
        ("retry", 1),
        ("prepare", 2),
        ("validate", 2),
    ]
    assert [message["content"] for message in result.messages] == ["retry", result.text]


def test_error_hook_appends_feedback_after_the_rejected_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(
        "tiny_coder.structured_agent.stream_llm_chat",
        _fake_llm(requests, [_outcome("invalid"), _outcome("accepted")]),
    )
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        max_steps=2,
    )

    async def validate(call: AgentCall) -> None:
        if call.attempt == 1:
            raise ValueError("retry")

    async def retry(call: AgentCall) -> None:
        call.messages.append({"role": "user", "content": str(call.error)})
        call.retry = True

    agent.events.subscribe(AFTER_CALL, validate)
    agent.events.subscribe(CALL_FAILED, retry)
    result = asyncio.run(agent.invoke(StructuredInput(prompt="go")))

    assert requests[1][-2:] == [
        {"role": "assistant", "content": "invalid"},
        {"role": "user", "content": "retry"},
    ]
    assert result.text == "accepted"


def test_hook_loop_preserves_tool_results_and_json_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tiny_coder.checkpoint import JsonCheckpointer

    requests: list[list[dict[str, Any]]] = []
    queue = [
        _outcome(tool_calls=(_echo_call("hello"),)),
        _outcome('{"name":"interim","count":1}'),
        _outcome('{"name":"done","count":2}'),
    ]
    checkpointer = JsonCheckpointer(tmp_path / "hooks.json")
    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", _fake_llm(requests, queue))
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        tools=[echo_tool],
        checkpointer=checkpointer,
        max_steps=4,
    )

    async def continue_after_interim(call: AgentCall) -> None:
        assert call.outcome is not None
        call.retry = "interim" in call.outcome.text

    agent.events.subscribe(AFTER_CALL, continue_after_interim)
    result = asyncio.run(agent.invoke(StructuredInput(prompt="go")))
    checkpoint = asyncio.run(checkpointer.load())

    assert result.model(SampleStructured) == SampleStructured(name="done", count=2)
    assert requests[1][-1]["content"] == "echo:hello"
    assert requests[2][-2]["content"] == "echo:hello"
    assert checkpoint is not None and checkpoint.step == 4
    assert set(result.state) == {"messages"}


def test_hook_loop_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []
    monkeypatch.setattr(
        "tiny_coder.structured_agent.stream_llm_chat",
        _fake_llm([], [_outcome("a"), _outcome("b")]),
    )
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        max_steps=2,
    )

    async def keep_going(call: AgentCall) -> None:
        attempts.append(call.attempt)
        call.retry = True

    agent.events.subscribe(AFTER_CALL, keep_going)
    with pytest.raises(GraphError, match="max_steps=2"):
        asyncio.run(agent.invoke(StructuredInput(prompt="go")))
    assert attempts == [1, 2]


@pytest.mark.parametrize("error", [ValueError("bad request"), asyncio.CancelledError()])
def test_hook_errors_propagate_without_retry(
    error: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def llm_call(**_kwargs: Any) -> LlmChatOutcome:
        raise error

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(llm_config=SampleLlmConfig())
    with pytest.raises(type(error)):
        asyncio.run(agent.invoke(StructuredInput(prompt="go")))


def test_hooks_are_ordered_and_call_contexts_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str, int]] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        await asyncio.sleep(0)
        return _outcome(kwargs["messages"][-1]["content"])

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(llm_config=SampleLlmConfig())

    async def first(call: AgentCall) -> None:
        events.append(("first", call.messages[-1]["content"], call.attempt))

    async def second(call: AgentCall) -> None:
        events.append(("second", call.messages[-1]["content"], call.attempt))

    agent.events.subscribe(BEFORE_CALL, first)
    agent.events.subscribe(BEFORE_CALL, second)

    async def run() -> list[StructuredResult]:
        return list(
            await asyncio.gather(
                agent.invoke(StructuredInput(prompt="a")),
                agent.invoke(StructuredInput(prompt="b")),
            )
        )

    results = asyncio.run(run())
    again = asyncio.run(agent.invoke(StructuredInput(prompt="c")))
    assert [result.text for result in results] == ["a", "b"]
    assert again.text == "c"
    assert events == [
        ("first", "a", 1),
        ("second", "a", 1),
        ("first", "b", 1),
        ("second", "b", 1),
        ("first", "c", 1),
        ("second", "c", 1),
    ]


def test_repeated_invocations_reset_overrides_and_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[float, str, str, int]] = []
    attempts: list[int] = []
    queue = [
        _outcome("invalid"),
        _outcome('{"name":"first","count":1}'),
        _outcome('{"text":"second"}'),
        _outcome('{"name":"default","count":3}'),
    ]
    results: list[StructuredResult] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        requests.append(
            (
                kwargs["llm_cfg"].temperature,
                kwargs["messages"][0]["content"],
                kwargs["response_format"]["json_schema"]["schema"]["title"],
                len(kwargs["messages"]),
            )
        )
        return queue.pop(0)

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        system_prompt="default system",
        max_steps=2,
    )

    async def validate(call: AgentCall) -> None:
        attempts.append(call.attempt)
        assert call.outcome is not None
        if call.outcome.text == "invalid":
            call.retry = True

    agent.events.subscribe(AFTER_CALL, validate)

    async def run() -> None:
        results.append(
            await agent.invoke(StructuredInput(prompt="first", system_prompt="first system"))
        )
        results.append(
            await agent.invoke(
                StructuredInput(
                    prompt="second",
                    llm_config=SampleLlmConfig(temperature=0.7),
                    response_model=EchoArgs,
                )
            )
        )
        results.append(await agent.invoke(StructuredInput(prompt="third")))

    asyncio.run(run())

    assert results[0].model(SampleStructured) == SampleStructured(name="first", count=1)
    assert results[1].model(EchoArgs) == EchoArgs(text="second")
    assert results[2].model(SampleStructured) == SampleStructured(name="default", count=3)
    assert attempts == [1, 2, 1, 1]
    assert requests == [
        (0.0, "first system", "SampleStructured", 2),
        (0.0, "first system", "SampleStructured", 3),
        (0.7, "default system", "EchoArgs", 2),
        (0.0, "default system", "SampleStructured", 2),
    ]
    assert agent.graph.max_steps == 2
    assert all(set(result.state) == {"messages"} for result in results)


def test_failed_invocation_does_not_poison_later_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[dict[str, Any]]] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            raise ValueError("failed")
        return _outcome("accepted")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(llm_config=SampleLlmConfig())
    with pytest.raises(ValueError, match="failed"):
        asyncio.run(agent.invoke(StructuredInput(prompt="first")))
    assert asyncio.run(agent.invoke(StructuredInput(prompt="second"))).text == "accepted"
    assert [call[-1]["content"] for call in calls] == ["first", "second"]
    assert all(len(call) == 2 for call in calls)


def test_concurrent_runs_keep_schemas_separate_with_a_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        await asyncio.sleep(0)
        title = kwargs["response_format"]["json_schema"]["schema"]["title"]
        return _outcome(
            '{"name":"sample","count":1}' if title == "SampleStructured" else '{"text":"echo"}'
        )

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=SampleStructured,
        max_steps=2,
    )

    async def continue_echo(call: AgentCall) -> None:
        assert call.outcome is not None
        call.retry = '"echo"' in call.outcome.text and call.attempt == 1

    agent.events.subscribe(AFTER_CALL, continue_echo)

    async def run() -> list[StructuredResult]:
        return list(
            await asyncio.gather(
                agent.invoke(StructuredInput(prompt="sample")),
                agent.invoke(StructuredInput(prompt="echo", response_model=EchoArgs)),
            )
        )

    results = asyncio.run(run())
    assert results[0].model(SampleStructured) == SampleStructured(name="sample", count=1)
    assert results[1].model(EchoArgs) == EchoArgs(text="echo")
    again = asyncio.run(agent.invoke(StructuredInput(prompt="echo", response_model=EchoArgs)))
    assert again.model(EchoArgs) == EchoArgs(text="echo")
    assert agent.graph.max_steps == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_agent_step_limit_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="max_steps must be at least 1"):
        create_structured_agent(llm_config=SampleLlmConfig(), max_steps=limit)
