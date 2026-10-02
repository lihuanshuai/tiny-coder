from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Literal, TypeVar

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from tiny_coder import llm as llm_module
from tiny_coder.llm import (
    LlmChatOutcome,
    LlmConfig,
    LlmExchange,
    nonnegative_int_from_llm_field,
    schema_response_format,
    stream_llm_chat,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class SampleOpenAIConfig(LlmConfig):
    base_url: str = "http://localhost:11434/v1"
    llm_model: str = "sample-model"
    num_ctx: int = 4096
    temperature: float = 0.5
    repeat_penalty: float = 1.1
    think: bool = False
    timeout: float = 37.0
    max_output_tokens: int | None = 8192


class _SummaryOutput(BaseModel):
    summary: str


class PlainPydanticConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"


class FakeStream:
    def __init__(self, packets: list[ChatCompletionChunk]) -> None:
        self._packets = iter(packets)

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._packets)
        except StopIteration as error:
            raise StopAsyncIteration from error


class FakeCompletions:
    kwargs: dict[str, Any] = {}
    packets: list[ChatCompletionChunk] = []

    @classmethod
    async def create(cls, **kwargs: Any) -> FakeStream:
        cls.kwargs = kwargs
        return FakeStream(list(cls.packets))


class FakeAsyncOpenAI:
    closed = False
    timeout = 0.0

    def __init__(self, **kwargs: Any) -> None:
        self.base_url = kwargs.get("base_url", "")
        type(self).timeout = kwargs.get("timeout", 0.0)
        self.chat = SimpleNamespace(completions=FakeCompletions())

    async def close(self) -> None:
        type(self).closed = True


_FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


def _chunk(
    *,
    delta: ChoiceDelta | None = None,
    finish_reason: _FinishReason | None = None,
    usage: CompletionUsage | None = None,
    no_choices: bool = False,
    **extras: Any,
) -> ChatCompletionChunk:
    choices: list[Choice] = []
    if not no_choices:
        choices.append(Choice(index=0, delta=delta or ChoiceDelta(), finish_reason=finish_reason))
    return ChatCompletionChunk(
        id="chunk",
        choices=choices,
        created=0,
        model="sample-model",
        object="chat.completion.chunk",
        usage=usage,
        **extras,
    )


PACKETS_TEXT = [
    _chunk(delta=ChoiceDelta.model_validate({"content": "", "reasoning": "hidden"})),
    _chunk(delta=ChoiceDelta(content="hello")),
    _chunk(delta=ChoiceDelta(content=" world"), finish_reason="stop"),
    _chunk(
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        no_choices=True,
        prompt_eval_duration=1_500_000_000,
        eval_duration=2_000_000_000,
    ),
]

TOOL_CALL_PACKETS = [
    _chunk(
        delta=ChoiceDelta(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call_1",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name="add", arguments='{"left"'),
                )
            ]
        )
    ),
    _chunk(
        delta=ChoiceDelta(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    function=ChoiceDeltaToolCallFunction(arguments=':2,"right":3}'),
                )
            ]
        ),
        finish_reason="tool_calls",
    ),
    _chunk(
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=6, total_tokens=11),
        no_choices=True,
    ),
]


def _patch(monkeypatch: pytest.MonkeyPatch, packets: Sequence[ChatCompletionChunk]) -> None:
    FakeCompletions.packets = list(packets)
    FakeCompletions.kwargs = {}
    FakeAsyncOpenAI.closed = False
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)


def test_llm_config_declares_required_fields() -> None:
    assert set(LlmConfig.model_fields) == {
        "base_url",
        "llm_model",
        "num_ctx",
        "temperature",
        "repeat_penalty",
        "think",
        "timeout",
        "max_output_tokens",
    }


def test_nonnegative_int_from_llm_field_accepts_only_nonnegative_integral_values() -> None:
    assert nonnegative_int_from_llm_field(3) == 3
    assert nonnegative_int_from_llm_field(3.0) == 3
    assert nonnegative_int_from_llm_field(True) is None
    assert nonnegative_int_from_llm_field(-1) is None
    assert nonnegative_int_from_llm_field(3.5) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"summary": "ok"}', {"summary": "ok"}),
        ("not json", None),
    ],
)
def test_llm_chat_outcome_json(text: str, expected: dict[str, object] | None) -> None:
    outcome = LlmChatOutcome(
        text=text,
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    assert outcome.json() == expected


def test_llm_chat_outcome_model_returns_requested_type() -> None:
    outcome = LlmChatOutcome(
        text='{"summary":"ok"}',
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    model = outcome.model(_SummaryOutput)
    assert model == _SummaryOutput(summary="ok")


def test_llm_chat_outcome_model_preserves_failures() -> None:
    outcome = LlmChatOutcome(
        text="not json",
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    assert outcome.model(_SummaryOutput) is None

    outcome2 = LlmChatOutcome(
        text='{"missing":true}',
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    with pytest.raises(ValidationError, match="summary"):
        outcome2.model(_SummaryOutput)


def test_llm_chat_outcome_message_appends_tool_calls() -> None:
    outcome = LlmChatOutcome(
        text="",
        tool_calls=(
            llm_module.ToolCall(
                id="c1", type="function", function={"name": "add", "arguments": "{}"}
            ),
        ),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="",
        completed_at="",
        llm={},
    )
    msg = outcome.message
    assert msg["role"] == "assistant"
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["id"] == "c1"


def test_stream_llm_chat_forwards_messages_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, PACKETS_TEXT)
    streamed: list[str] = []

    async def record_chunk(chunk: str) -> None:
        streamed.append(chunk)

    outcome = asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "say hello"}],
            on_chunk=record_chunk,
        )
    )

    assert streamed == ["hello", " world"]
    assert outcome.text == "hello world"
    assert outcome.prompt_eval_count == 3
    assert outcome.eval_count == 4
    assert outcome.llm["provider"] == "openai-compatible"
    assert FakeCompletions.kwargs["messages"] == [{"role": "user", "content": "say hello"}]
    assert "response_format" not in FakeCompletions.kwargs
    assert "tools" not in FakeCompletions.kwargs


def test_stream_llm_chat_records_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, PACKETS_TEXT)
    seen: list[LlmExchange] = []

    async def record_exchange(exchange: LlmExchange) -> None:
        seen.append(exchange)

    outcome = asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "say hello"}],
            on_exchange=record_exchange,
        )
    )

    assert len(seen) == 1
    assert seen[0].messages == [{"role": "user", "content": "say hello"}]
    assert seen[0].response is outcome


def test_stream_llm_chat_with_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, PACKETS_TEXT)
    rf = schema_response_format(_SummaryOutput)

    asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "go"}],
            response_format=rf,
        )
    )

    assert FakeCompletions.kwargs["response_format"] == rf
    assert FakeAsyncOpenAI.closed


def test_stream_llm_chat_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, PACKETS_TEXT)

    class FakeTool:
        name = "add"
        description = "Add numbers"
        parameters = {"type": "object", "properties": {"x": {"type": "integer"}}}

    asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "add"}],
            tools=[FakeTool()],
        )
    )

    assert "tools" in FakeCompletions.kwargs
    assert FakeCompletions.kwargs["tools"][0]["function"]["name"] == "add"


def test_stream_llm_chat_omits_max_tokens_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, PACKETS_TEXT)

    asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(max_output_tokens=None),
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert "max_tokens" not in FakeCompletions.kwargs


def test_stream_llm_chat_raises_server_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, [_chunk(no_choices=True, error={"message": "rate limited"})])

    with pytest.raises(RuntimeError, match="rate limited"):
        asyncio.run(
            stream_llm_chat(
                llm_cfg=SampleOpenAIConfig(),
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert FakeAsyncOpenAI.closed


def test_stream_llm_chat_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(
            stream_llm_chat(
                llm_cfg=SampleOpenAIConfig(),
                messages=[],
            )
        )


def test_stream_llm_chat_requires_llm_config_subclass() -> None:
    with pytest.raises(TypeError, match="llm_cfg must inherit LlmConfig"):
        asyncio.run(
            stream_llm_chat(
                llm_cfg=PlainPydanticConfig(),  # type: ignore[arg-type]
                messages=[{"role": "user", "content": "hi"}],
            )
        )


def test_stream_llm_chat_passes_tool_specs_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, PACKETS_TEXT)

    class FakeTool:
        name = "get_weather"
        description = "Get current weather"
        parameters = {"type": "object", "properties": {"city": {"type": "string"}}}

    asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "what's the weather"}],
            tools=[FakeTool()],
        )
    )

    tool_spec = FakeCompletions.kwargs["tools"][0]
    assert tool_spec["type"] == "function"
    assert tool_spec["function"]["name"] == "get_weather"


def test_stream_llm_chat_collects_tool_calls_from_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, TOOL_CALL_PACKETS)

    outcome = asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "add 2 and 3"}],
        )
    )

    assert len(outcome.tool_calls) == 1
    tc = outcome.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.function["name"] == "add"
    assert tc.function["arguments"] == '{"left":2,"right":3}'
    assert outcome.text == ""
    # message dict should include tool_calls
    msg = outcome.message
    assert "tool_calls" in msg


def test_stream_llm_chat_collects_tool_calls_from_sdk_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The OpenAI SDK parses SSE chunks into ChatCompletionChunk objects.
    tool_call = ChoiceDeltaToolCall(
        index=0,
        id="call_1",
        type="function",
        function=ChoiceDeltaToolCallFunction(name="add", arguments='{"left":2,"right":3}'),
    )
    packets: list[ChatCompletionChunk] = [
        ChatCompletionChunk(
            id="chunk-1",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="", tool_calls=[tool_call]),
                    finish_reason=None,
                )
            ],
            created=0,
            model="sample-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chunk-2",
            choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason="tool_calls")],
            created=0,
            model="sample-model",
            object="chat.completion.chunk",
        ),
    ]
    _patch(monkeypatch, packets)

    outcome = asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(),
            messages=[{"role": "user", "content": "add 2 and 3"}],
        )
    )

    assert len(outcome.tool_calls) == 1
    tc = outcome.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.function["name"] == "add"
    assert tc.function["arguments"] == '{"left":2,"right":3}'


def test_schema_response_format_validates_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rf = schema_response_format(_SummaryOutput)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "structured_output"

    with pytest.raises(TypeError, match="BaseModel"):
        schema_response_format(object())  # type: ignore[arg-type]
