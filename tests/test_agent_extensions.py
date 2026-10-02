from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import ReferenceType, ref

import pytest
from httpx import Request
from jinja2 import Environment
from openai import APIConnectionError
from pydantic import BaseModel
from test_structured_agent import SampleLlmConfig, _echo_call, _outcome, echo_tool

from tiny_coder.agent_extensions import (
    CallRecording,
    JinjaPrompt,
    RetryPolicy,
    StreamOutput,
    StructuredOutput,
    read_file_snapshots,
)
from tiny_coder.checkpoint import JsonCheckpointer
from tiny_coder.llm import LlmChatOutcome
from tiny_coder.structured_agent import (
    AgentCall,
    AgentExtension,
    AgentHooks,
    StructuredInput,
    create_structured_agent,
)


class Output(BaseModel):
    summary: str


def test_call_recording_preserves_rejected_output_and_resolves_each_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(("invalid", '{"summary":"done"}', '{"summary":"next"}'))

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        return _outcome(next(responses))

    def metadata(call: AgentCall) -> dict[str, object]:
        return {"task": call.context, "attempt": call.attempt}

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    path = tmp_path / "records" / "calls.jsonl"
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        max_steps=2,
        extensions=(RetryPolicy(2), CallRecording(path, metadata), StructuredOutput()),
    )

    async def run() -> None:
        async with agent:
            await agent.invoke(StructuredInput(prompt="first", context="one"))
            await agent.invoke(StructuredInput(prompt="second", context="two"))

    asyncio.run(run())
    with path.open("r", encoding="utf-8", newline="\n") as file:
        records = [json.loads(line) for line in file]
    assert [record["output"] for record in records] == [
        "invalid",
        '{"summary":"done"}',
        '{"summary":"next"}',
    ]
    assert [(record["task"], record["attempt"], record["user"]) for record in records] == [
        ("one", 1, "first"),
        ("one", 2, "first"),
        ("two", 1, "second"),
    ]
    assert records[0]["stats"] == {"prompt_eval_count": 1, "eval_count": 1}


def test_structured_output_works_with_plain_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    accepted: list[tuple[object, BaseModel]] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        assert kwargs["messages"][-1]["content"] == "plain prompt"
        return _outcome('{"summary":"done"}')

    async def accept(call: AgentCall, output: BaseModel) -> None:
        accepted.append((call.context, output))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        extensions=(StructuredOutput(accept),),
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="plain prompt", context="task")))
    assert result.output == Output(summary="done")
    assert result.model(Output) is result.output
    assert accepted == [("task", Output(summary="done"))]


def test_jinja_prompt_works_without_other_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Environment()

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        assert kwargs["messages"] == [
            {"role": "system", "content": "system Ada\n"},
            {"role": "user", "content": "hello Ada\n"},
        ]
        return _outcome("hello")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        extensions=(
            JinjaPrompt(
                env.from_string("system {{ name }}"),
                env.from_string("hello {{ name }}"),
                {"name": "Ada"},
            ),
        ),
    )
    result = asyncio.run(agent.invoke(StructuredInput()))
    assert result.text == "hello"


@pytest.mark.parametrize("field, expected", [(None, '{"summary":"done"}'), ("summary", "done")])
def test_stream_output_works_without_templates_or_schema(
    field: str | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks: list[str] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        for chunk in ('{"summary":"', 'done"}'):
            await kwargs["on_chunk"](chunk)
        return _outcome('{"summary":"done"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(llm_config=SampleLlmConfig()).use(
        StreamOutput(field=field, sink=chunks.append)
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="prompt")))
    assert result.text == '{"summary":"done"}'
    assert "".join(chunks) == expected + "\n"


def test_retry_policy_works_without_other_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[float] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        attempts.append(kwargs["llm_cfg"].temperature)
        if len(attempts) == 1:
            raise APIConnectionError(request=Request("POST", "http://localhost"))
        return _outcome("done")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(llm_config=SampleLlmConfig()).use(RetryPolicy(2))
    assert asyncio.run(agent.invoke(StructuredInput(prompt="prompt"))).text == "done"
    assert attempts == [0.0, 0.0]


@dataclass
class Events(AgentExtension):
    name: str
    events: list[str]

    def register(self, hooks: AgentHooks) -> None:
        hooks.add_hook("before_call", self.before)
        hooks.add_hook("after_call", self.after)

    async def before(self, call: AgentCall) -> None:
        self.events.append(f"{self.name}:before:{call.context}")

    async def after(self, call: AgentCall) -> None:
        self.events.append(f"{self.name}:after:{call.context}")


@dataclass
class ScopedResource(AgentExtension):
    name: str
    active: ContextVar[str]
    events: list[str]
    fail_on_enter: bool = False

    def register(self, hooks: AgentHooks) -> None:
        hooks.add_scope(self.scope)

    @contextmanager
    def scope(self, call: AgentCall) -> Iterator[None]:
        self.events.append(f"enter:{self.name}")
        token = self.active.set(self.name)
        try:
            if self.fail_on_enter:
                self.fail_on_enter = False
                raise ValueError("scope failed")
            yield
        finally:
            self.active.reset(token)
            self.events.append(f"exit:{self.name}")


@pytest.mark.parametrize("failure", ["model", "cancel", "scope"])
def test_call_scopes_unwind_before_error_hooks_and_can_be_reused(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    active: ContextVar[str] = ContextVar("test_scope", default="idle")
    events: list[str] = []
    failed = False

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        nonlocal failed
        assert active.get() == "inner"
        if not failed and failure in ("model", "cancel"):
            failed = True
            if failure == "cancel":
                raise asyncio.CancelledError()
            raise ValueError("model failed")
        return _outcome("done")

    async def retry(call: AgentCall) -> None:
        assert active.get() == "idle"
        events.append("retry")
        call.retry = True

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        extensions=(
            ScopedResource("outer", active, events),
            ScopedResource("inner", active, events, fail_on_enter=failure == "scope"),
        ),
    )
    agent.add_hook("on_error", retry)

    async def run() -> None:
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await agent.invoke(StructuredInput(prompt="prompt"))
            assert active.get() == "idle"
        assert (await agent.invoke(StructuredInput(prompt="prompt"))).text == "done"
        assert active.get() == "idle"

    asyncio.run(run())
    cycle = ["enter:outer", "enter:inner", "exit:inner", "exit:outer"]
    assert events == cycle + ([] if failure == "cancel" else ["retry"]) + cycle


def test_extension_requires_register_implementation() -> None:
    with pytest.raises(TypeError, match="abstract"):
        AgentExtension()  # type: ignore[abstract]


@pytest.mark.parametrize("entrypoint", ["use", "factory"])
def test_registration_rejects_duck_typed_extensions(entrypoint: str) -> None:
    class DuckExtension:
        def register(self, hooks: AgentHooks) -> None:
            raise AssertionError("invalid extension must not be registered")

    agent = create_structured_agent(llm_config=SampleLlmConfig())
    extension = DuckExtension()
    with pytest.raises(TypeError, match="AgentExtension instances"):
        if entrypoint == "use":
            agent.use(extension)  # type: ignore[arg-type]
        else:
            create_structured_agent(
                llm_config=SampleLlmConfig(),
                extensions=(extension,),  # type: ignore[arg-type]
            )


def test_factory_and_use_extensions_compose_without_accumulating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        return _outcome("done")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        extensions=(Events("factory", events),),
    )
    assert agent.use(Events("use", events)) is agent

    async def run() -> None:
        await agent.invoke(StructuredInput(prompt="one", context=1))
        await agent.invoke(StructuredInput(prompt="two", context=2))

    asyncio.run(run())
    assert events == [
        "factory:before:1",
        "use:before:1",
        "factory:after:1",
        "use:after:1",
        "factory:before:2",
        "use:before:2",
        "factory:after:2",
        "use:after:2",
    ]


def test_retry_refreshes_variables_and_file_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.md"
    source.write_text("first", encoding="utf-8", newline="\n")
    previous: list[str | None] = []
    calls: list[list[dict[str, str]]] = []

    def variables(call: AgentCall) -> dict[str, object]:
        previous.append(call.previous_outcome.text if call.previous_outcome is not None else None)
        return {"attempt": call.attempt, "input_files": read_file_snapshots(tmp_path, [source])}

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            source.write_text("second", encoding="utf-8", newline="\n")
            return _outcome("invalid")
        return _outcome('{"summary":"done"}')

    template = Environment().from_string(
        "{{ attempt }}:{{ input_files[0].path }}:{{ input_files[0].content }}:{{ retry_errors | join(',') }}"
    )
    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        extensions=(JinjaPrompt(template, template, variables), RetryPolicy(2), StructuredOutput()),
    )
    result = asyncio.run(agent.invoke(StructuredInput()))
    assert result.output == Output(summary="done")
    assert previous == [None, "invalid"]
    assert calls[0][0]["content"] == "1:source.md:first:\n"
    assert calls[1][0]["content"].startswith("2:source.md:second:invalid structured output")
    assert calls[1][1]["content"].startswith("2:source.md:second:invalid structured output")


@pytest.mark.parametrize("failure", ["network", "stream", "validation", "accept"])
def test_retry_rules_and_stream_reset(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    temperatures: list[float] = []
    chunks: list[str] = []
    accepted: list[BaseModel] = []

    async def accept(call: AgentCall, output: BaseModel) -> None:
        if failure == "accept" and call.attempt == 1:
            raise ValueError("candidate rejected")
        accepted.append(output)

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        temperatures.append(kwargs["llm_cfg"].temperature)
        if len(temperatures) == 1:
            if failure == "network":
                raise APIConnectionError(request=Request("POST", "http://localhost"))
            if failure == "stream":
                await kwargs["on_chunk"]("abcd")
                raise AssertionError("stream limit was not enforced")
            if failure == "validation":
                return _outcome('{"summary":null}')
        await kwargs["on_chunk"]("ok")
        return _outcome('{"summary":"done"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        extensions=(
            JinjaPrompt("system", "user"),
            StreamOutput(max_chars=3, sink=chunks.append),
            RetryPolicy(2, SampleLlmConfig(temperature=0.2)),
            StructuredOutput(accept),
        ),
    )
    result = asyncio.run(agent.invoke(StructuredInput()))
    assert result.output == Output(summary="done")
    assert accepted == [Output(summary="done")]
    assert temperatures == [0.0, 0.0 if failure == "network" else 0.2]
    assert chunks[-2:] == ["ok", "\n"]


def test_agent_recovers_from_failure_with_fresh_invocation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, str]] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        calls.append((kwargs["llm_cfg"].temperature, kwargs["messages"][-1]["content"]))
        return _outcome("invalid" if len(calls) == 1 else '{"summary":"recovered"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        extensions=(RetryPolicy(1), StructuredOutput()),
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="failed after 1 attempt"):
            await agent.invoke(StructuredInput(prompt="first"))
        for _ in range(2):
            result = await agent.invoke(
                StructuredInput(prompt="next", llm_config=SampleLlmConfig(temperature=0.7))
            )
            assert result.output == Output(summary="recovered")

    asyncio.run(run())
    assert calls == [(0.0, "first"), (0.7, "next"), (0.7, "next")]


@pytest.mark.parametrize("right_enabled", [True, False])
def test_concurrent_invocations_isolate_context_feedback_and_streams(
    right_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Environment()
    chunks: list[str] = []
    attempts: dict[str, int] = {}

    def variables(call: AgentCall) -> dict[str, object]:
        return {"name": call.context}

    def stream_field(call: AgentCall) -> str | None:
        return "summary" if call.context == "left" else None

    def stream_limit(call: AgentCall) -> int:
        return 18 if call.context == "left" else 19

    def stream_enabled(call: AgentCall) -> bool:
        return call.context == "left" or right_enabled

    def retry_limit(call: AgentCall) -> int:
        return 2 if call.context == "left" else 1

    def retry_config(call: AgentCall) -> SampleLlmConfig:
        return SampleLlmConfig(temperature=0.2 if call.context == "left" else 0.7)

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        name = kwargs["messages"][0]["content"].strip()
        attempts[name] = attempts.get(name, 0) + 1
        await kwargs["on_chunk"]('{"summary":"')
        await asyncio.sleep(0)
        await kwargs["on_chunk"](name + '"}')
        if name == "left" and attempts[name] == 1:
            return _outcome("invalid")
        if name == "right":
            assert kwargs["llm_cfg"].temperature == 0.0
            assert kwargs["messages"][1]["content"] == "\n"
        elif attempts[name] == 2:
            assert kwargs["llm_cfg"].temperature == 0.2
            assert "invalid structured output" in kwargs["messages"][1]["content"]
        return _outcome('{"summary":"' + name + '"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
    ).use(
        JinjaPrompt(
            env.from_string("{{ name }}"),
            env.from_string("{{ retry_errors | join(',') }}"),
            variables,
        ),
        StreamOutput(
            field=stream_field, max_chars=stream_limit, enabled=stream_enabled, sink=chunks.append
        ),
        RetryPolicy(retry_limit, retry_config),
        StructuredOutput(),
    )

    async def run() -> None:
        results = await asyncio.gather(
            agent.invoke(StructuredInput(context="left")),
            agent.invoke(StructuredInput(context="right")),
        )
        assert [result.output for result in results] == [
            Output(summary="left"),
            Output(summary="right"),
        ]

    asyncio.run(run())
    assert attempts == {"left": 2, "right": 1}
    assert chunks.count("left") == 2
    assert chunks.count('right"}') == int(right_enabled)
    assert chunks.count('{"summary":"') == int(right_enabled)
    assert chunks.count("\n") == 2 + int(right_enabled)


def test_nested_invocation_preserves_the_outer_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: list[str] = []
    nested = False

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        name = kwargs["messages"][-1]["content"]
        if name == "outer":
            await kwargs["on_chunk"]('{"summary":"out')
            await kwargs["on_chunk"]('er"}')
        else:
            await kwargs["on_chunk"]('{"summary":"inner"}')
        return _outcome('{"summary":"' + name + '"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        extensions=(StreamOutput(field="summary", sink=chunks.append),),
    )

    async def invoke_inner(call: AgentCall) -> None:
        nonlocal nested
        if call.context == "outer" and not nested:
            nested = True
            result = await agent.invoke(StructuredInput(prompt="inner"))
            assert result.model(Output) == Output(summary="inner")

    agent.add_hook("on_chunk", invoke_inner)
    result = asyncio.run(agent.invoke(StructuredInput(prompt="outer", context="outer")))
    assert result.model(Output) == Output(summary="outer")
    assert chunks == ["out", "inner", "\n", "er", "\n"]


@pytest.mark.parametrize("termination", ["success", "error", "cancel"])
def test_stream_extension_does_not_retain_finished_calls(
    termination: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[ReferenceType[AgentCall]] = []

    async def observe(call: AgentCall) -> None:
        calls.append(ref(call))

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        await kwargs["on_chunk"]("content")
        if termination == "error":
            raise RuntimeError("failed")
        if termination == "cancel":
            raise asyncio.CancelledError()
        return _outcome("content")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(), extensions=(StreamOutput(sink=None),)
    )
    agent.add_hook("before_call", observe)

    async def run() -> None:
        if termination == "success":
            assert (await agent.invoke(StructuredInput(prompt="prompt"))).text == "content"
        else:
            error = RuntimeError if termination == "error" else asyncio.CancelledError
            with pytest.raises(error):
                await agent.invoke(StructuredInput(prompt="prompt"))

    asyncio.run(run())
    gc.collect()
    assert len(calls) == 1
    assert calls[0]() is None


@pytest.mark.parametrize("option", ["stream_limit", "retry_limit"])
def test_invalid_context_limits_are_rejected_before_llm_call(
    option: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_limit(call: AgentCall) -> int:
        return -1

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        raise AssertionError("invalid limits must fail before calling the model")

    extension = (
        StreamOutput(max_chars=invalid_limit)
        if option == "stream_limit"
        else RetryPolicy(invalid_limit)
    )
    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(llm_config=SampleLlmConfig(), extensions=(extension,))
    with pytest.raises(ValueError, match="max_chars|max_attempts"):
        asyncio.run(agent.invoke(StructuredInput(prompt="prompt")))


def test_prompt_and_output_extensions_preserve_tool_rounds_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict[str, Any]]] = []

    async def llm_call(**kwargs: Any) -> LlmChatOutcome:
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return _outcome(tool_calls=(_echo_call("hello"),))
        return _outcome('{"summary":"done"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm_call)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        tools=[echo_tool],
        checkpointer=JsonCheckpointer(tmp_path / "checkpoint.json"),
        extensions=(JinjaPrompt("system", "prompt"), StructuredOutput()),
    )
    result = asyncio.run(agent.invoke(StructuredInput(context=object())))
    assert result.output == Output(summary="done")
    assert [message["role"] for message in calls[1]] == ["system", "user", "assistant", "tool"]
    assert calls[1][-1]["content"] == "echo:hello"


@pytest.mark.parametrize("absolute", [False, True])
def test_file_snapshots_use_workspace_relative_paths(tmp_path: Path, absolute: bool) -> None:
    source = tmp_path / "plan" / "overview.md"
    source.parent.mkdir()
    source.write_text("# Plan\n", encoding="utf-8", newline="\n")

    snapshots = read_file_snapshots(tmp_path, [source if absolute else Path("plan/overview.md")])

    assert snapshots == [{"path": "plan/overview.md", "language": "markdown", "content": "# Plan"}]


def test_snapshot_path_must_stay_inside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_file_snapshots(tmp_path, [Path("../outside.md")])
