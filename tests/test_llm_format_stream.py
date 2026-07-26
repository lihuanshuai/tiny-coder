from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from tiny_coder import llm_format_stream
from tiny_coder.llm_format_stream import (
    _llm_stream_chunk,
    _nonneg_int_from_llm_field,
    stream_llm_chat_format,
)


class SampleOpenAIConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    llm_model: str = "sample-model"
    num_ctx: int = 4096
    temperature: float = 0.5
    repeat_penalty: float = 1.1
    think: bool = False


class FakeStream:
    def __init__(self, packets: list[object]) -> None:
        self._packets = iter(packets)

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._packets)
        except StopIteration as error:
            raise StopAsyncIteration from error


class FakeCompletions:
    kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> FakeStream:
        type(self).kwargs = kwargs
        return FakeStream(
            [
                {
                    "choices": [
                        {
                            "delta": {"content": "hello"},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "model": "sample-model",
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                    "choices": [
                        {
                            "delta": {"content": " world"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ]
        )


class FakeAsyncOpenAI:
    closed = False

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=FakeCompletions())

    async def close(self) -> None:
        type(self).closed = True


def test_nonneg_int_from_llm_field_accepts_only_nonnegative_integral_values() -> None:
    assert _nonneg_int_from_llm_field(3) == 3
    assert _nonneg_int_from_llm_field(3.0) == 3
    assert _nonneg_int_from_llm_field(True) is None
    assert _nonneg_int_from_llm_field(-1) is None
    assert _nonneg_int_from_llm_field(3.5) is None


def test_llm_stream_chunk_reads_openai_and_local_fields() -> None:
    assert _llm_stream_chunk(
        {
            "usage": {"prompt_tokens": 2, "completion_tokens": 5},
            "choices": [{"delta": {"content": "chunk"}}],
        }
    ) == (2, 5, "chunk")
    assert _llm_stream_chunk(
        {
            "prompt_eval_count": 7,
            "eval_count": 9,
            "message": {"content": "fallback"},
        }
    ) == (7, 9, "fallback")


def test_stream_llm_chat_format_forwards_schema_chunks_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_format_stream, "AsyncOpenAI", FakeAsyncOpenAI)
    streamed: list[str] = []
    schema = {"type": "object"}

    outcome = asyncio.run(
        stream_llm_chat_format(
            llm_cfg=SampleOpenAIConfig(),
            system="system",
            prompt="prompt",
            response_format=schema,
            on_chunk=streamed.append,
        )
    )

    assert streamed == ["hello", " world"]
    assert outcome.text == "hello world"
    assert outcome.prompt_eval_count == 3
    assert outcome.eval_count == 4
    assert outcome.llm == {
        "prompt_eval_count": 3,
        "eval_count": 4,
        "model": "sample-model",
        "done_reason": "stop",
        "provider": "openai-compatible",
    }
    assert FakeCompletions.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "strict": True,
            "schema": schema,
        },
    }
    assert FakeCompletions.kwargs["extra_body"] == {
        "num_ctx": 4096,
        "repeat_penalty": 1.1,
        "think": False,
    }
    assert FakeAsyncOpenAI.closed is True
