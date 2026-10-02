"""OpenAI-compatible streaming chat with tool calls and JSON schema output."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeAlias, TypeVar, cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel, ConfigDict, Field

from tiny_coder.json_utils import JsonProtocolError, load_json_object

logger = logging.getLogger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_CompletionStream: TypeAlias = AsyncStream[ChatCompletionChunk]

UNSUPPORTED_SCHEMA_HINT = (
    "Current OpenAI-compatible LLM server does not support JSON Schema structured output "
    "(response_format.json_schema). Please confirm that the server and loaded model support it."
)


class LlmConfig(BaseModel):
    """Configuration required by the OpenAI-compatible chat stream."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
    )

    base_url: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    num_ctx: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    repeat_penalty: float = Field(ge=0.0)
    think: bool
    timeout: float = Field(gt=0.0)
    max_output_tokens: int | None = Field(default=None, gt=0)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation from the LLM."""

    id: str
    type: str
    function: dict[str, str]


@dataclass(frozen=True, slots=True)
class LlmChatOutcome:
    """Result from one complete LLM chat stream."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_eval_count: int
    eval_count: int
    client_wall_time_ms: float
    started_at: str
    completed_at: str
    llm: dict[str, Any]

    @property
    def message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.get("name", ""),
                        "arguments": tc.function.get("arguments", ""),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg

    def json(self) -> dict[str, Any] | None:
        try:
            return load_json_object(self.text)
        except JsonProtocolError:
            return None

    def model(self, model_type: type[_ModelT]) -> _ModelT | None:
        return parse_structured(self.text, model_type)


@dataclass(frozen=True, slots=True)
class LlmExchange:
    """One LLM request and response captured for recording and inspection."""

    messages: list[Mapping[str, Any]]
    response: LlmChatOutcome


def parse_structured(text: str, model_type: type[_ModelT]) -> _ModelT | None:
    """Parse a JSON object text into a model, or return None when it is not JSON."""
    try:
        payload = load_json_object(text)
    except JsonProtocolError:
        return None
    return model_type.model_validate(payload)


def nonnegative_int_from_llm_field(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0 and value == int(value):
        return int(value)
    return None


def _positive_float_from_llm_field(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 else None


def _tokens_per_second(token_count: int, duration_seconds: float | None) -> float | None:
    if duration_seconds is None or duration_seconds <= 0:
        return None
    return token_count / duration_seconds


def schema_response_format(model: type[BaseModel]) -> dict[str, Any]:
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("model must be a BaseModel subclass")
    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "strict": True,
            "schema": schema,
        },
    }


def _chunk_delta_text(packet: ChatCompletionChunk) -> tuple[str, str]:
    if not packet.choices:
        return "", ""
    delta = packet.choices[0].delta
    if delta is None:
        return "", ""
    content = delta.content
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning is None:
        reasoning = getattr(delta, "reasoning", None)
    return (
        "" if content is None else str(content),
        "" if reasoning is None else str(reasoning),
    )


def _chunk_tool_call_deltas(packet: ChatCompletionChunk) -> list[dict[str, Any]]:
    tool_calls: list[ChoiceDeltaToolCall] | None = None
    if packet.choices:
        delta = packet.choices[0].delta
        if delta is not None:
            tool_calls = delta.tool_calls
    if not tool_calls:
        return []
    result: list[dict[str, Any]] = []
    for item in tool_calls:
        function = item.function
        result.append(
            {
                "index": item.index,
                "id": str(item.id or ""),
                "type": str(item.type or "function"),
                "function": {
                    "name": "" if function is None else str(function.name or ""),
                    "arguments": "" if function is None else str(function.arguments or ""),
                },
            }
        )
    return result


def _merge_tool_call_deltas(
    slots: dict[int, dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> None:
    for delta in deltas:
        index = delta["index"]
        slot: dict[str, Any] = slots.get(index) or {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
        slots[index] = slot
        if delta.get("id"):
            slot["id"] = str(delta["id"])
        if delta.get("type"):
            slot["type"] = str(delta["type"])
        fn = delta.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += str(fn["name"])
        if fn.get("arguments"):
            slot["function"]["arguments"] += str(fn["arguments"])


def _slots_to_tool_calls(slots: dict[int, dict[str, Any]]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(
            id=str(slot["id"]) if slot["id"] else f"call_{index}",
            type=str(slot.get("type", "function")),
            function={
                "name": str(slot["function"].get("name", "")),
                "arguments": str(slot["function"].get("arguments", "")),
            },
        )
        for index, slot in sorted(slots.items())
    )


def _stream_error_message(packet: ChatCompletionChunk) -> str | None:
    error = getattr(packet, "error", None)
    if error is None:
        return None
    message = error.get("message") if isinstance(error, dict) else getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(error)


def _llm_tail_metrics_dict(
    packet: ChatCompletionChunk,
    *,
    done_reason: str | None,
    reasoning_chars: int,
    prompt_eval_count: int,
    eval_count: int,
    observed_prompt_seconds: float,
    observed_eval_seconds: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    usage = packet.usage
    if usage is not None:
        for value, target_key in (
            (nonnegative_int_from_llm_field(usage.prompt_tokens), "prompt_eval_count"),
            (nonnegative_int_from_llm_field(usage.completion_tokens), "eval_count"),
        ):
            if value is not None:
                metrics[target_key] = value
    if packet.model:
        metrics["model"] = packet.model
    if done_reason:
        metrics["done_reason"] = done_reason
    if reasoning_chars:
        metrics["reasoning_chars"] = reasoning_chars
    prompt_duration_ns = _positive_float_from_llm_field(
        getattr(packet, "prompt_eval_duration", None)
    )
    eval_duration_ns = _positive_float_from_llm_field(getattr(packet, "eval_duration", None))
    prompt_speed = _tokens_per_second(
        prompt_eval_count,
        prompt_duration_ns / 1_000_000_000
        if prompt_duration_ns is not None
        else observed_prompt_seconds,
    )
    eval_speed = _tokens_per_second(
        eval_count,
        eval_duration_ns / 1_000_000_000 if eval_duration_ns is not None else observed_eval_seconds,
    )
    if prompt_speed is not None:
        metrics["prompt_tokens_per_second"] = prompt_speed
    if eval_speed is not None:
        metrics["eval_tokens_per_second"] = eval_speed
    metrics["provider"] = "openai-compatible"
    return metrics


def openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or "not-needed"


def openai_extra_body(llm_cfg: LlmConfig) -> dict[str, Any]:
    return {
        "num_ctx": llm_cfg.num_ctx,
        "repeat_penalty": llm_cfg.repeat_penalty,
        "think": llm_cfg.think,
    }


def tool_openai_spec(tool: object) -> dict[str, Any]:
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", "")
    parameters = getattr(tool, "parameters", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"tool {tool!r} must have a 'name' attribute")
    if not isinstance(parameters, dict):
        raise TypeError(f"tool {name!r} 'parameters' must be a JSON schema dict")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(description or ""),
            "parameters": parameters,
        },
    }


async def stream_llm_chat(
    *,
    llm_cfg: LlmConfig,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[object] | None = None,
    response_format: dict[str, Any] | None = None,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    on_exchange: Callable[[LlmExchange], Awaitable[None]] | None = None,
) -> LlmChatOutcome:
    """Stream an OpenAI-compatible chat completion with optional tools or structured output."""
    if not isinstance(llm_cfg, LlmConfig):
        raise TypeError("llm_cfg must inherit LlmConfig")
    if not messages:
        raise ValueError("messages must not be empty")
    config = llm_cfg
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=openai_api_key(),
        timeout=config.timeout,
    )
    prompt_eval_count = 0
    eval_count = 0
    had_prompt_metric = False
    had_eval_metric = False
    last_packet: ChatCompletionChunk | None = None
    done_reason: str | None = None
    reasoning_chars = 0
    first_model_token_at: float | None = None
    collected: list[str] = []
    tool_call_slots: dict[int, dict[str, Any]] = {}

    async def dispatch_chunk(chunk: str) -> None:
        collected.append(chunk)
        if on_chunk is not None:
            await on_chunk(chunk)

    stream_response: _CompletionStream | None = None
    try:
        try:
            create_completion = cast(
                Callable[..., Awaitable[_CompletionStream]],
                client.chat.completions.create,
            )
            reasoning_options = {} if config.think else {"reasoning_effort": "none"}
            output_limit_options = (
                {"max_tokens": config.max_output_tokens}
                if config.max_output_tokens is not None
                else {}
            )
            kwargs: dict[str, Any] = {
                "model": config.llm_model,
                "messages": list(messages),
                "temperature": config.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": openai_extra_body(llm_cfg),
                **reasoning_options,
                **output_limit_options,
            }
            if tools is not None:
                kwargs["tools"] = [tool_openai_spec(t) for t in tools]
            if response_format is not None:
                kwargs["response_format"] = response_format
            stream_response = await create_completion(**kwargs)
        except Exception as error:
            message = str(error).lower()
            if any(
                marker in message for marker in ("format", "grammar", "schema", "response_format")
            ):
                raise RuntimeError(UNSUPPORTED_SCHEMA_HINT) from error
            raise
        assert stream_response is not None
        async for packet in stream_response:
            last_packet = packet
            error_message = _stream_error_message(packet)
            if error_message is not None:
                raise RuntimeError(f"OpenAI-compatible LLM stream error: {error_message}")
            prompt_count, completion_count, content, reasoning, finish_reason = (
                _parse_stream_packet(packet)
            )
            if prompt_count is not None:
                prompt_eval_count = prompt_count
                had_prompt_metric = True
            if completion_count is not None:
                eval_count = completion_count
                had_eval_metric = True

            deltas = _chunk_tool_call_deltas(packet)
            if deltas:
                _merge_tool_call_deltas(tool_call_slots, deltas)
            has_output = bool(content) or bool(reasoning) or bool(deltas)
            if first_model_token_at is None and has_output:
                first_model_token_at = time.perf_counter()
            reasoning_chars += len(reasoning)
            if finish_reason is not None:
                done_reason = finish_reason
            if content:
                await dispatch_chunk(content)
    finally:
        if stream_response is not None:
            try:
                await stream_response.close()
            except Exception:
                pass
        try:
            await client.close()
        except Exception:
            pass
    if not had_prompt_metric or not had_eval_metric:
        logger.debug(
            "OpenAI-compatible chat stream missing usage metrics (prompt=%s eval=%s); using zeros",
            had_prompt_metric,
            had_eval_metric,
        )
    tool_calls = _slots_to_tool_calls(tool_call_slots)
    if not collected and not tool_calls:
        raise RuntimeError(
            "OpenAI-compatible LLM stream returned no content "
            f"(finish_reason={done_reason or 'unknown'}, "
            f"prompt_eval_count={prompt_eval_count}, eval_count={eval_count}, "
            f"reasoning_chars={reasoning_chars})"
        )
    assert first_model_token_at is not None
    assert last_packet is not None
    completed = time.perf_counter()
    completed_at = datetime.now(timezone.utc).isoformat()
    outcome = LlmChatOutcome(
        text="".join(collected),
        tool_calls=tool_calls,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        client_wall_time_ms=(completed - started) * 1000.0,
        started_at=started_at,
        completed_at=completed_at,
        llm=_llm_tail_metrics_dict(
            last_packet,
            done_reason=done_reason,
            reasoning_chars=reasoning_chars,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            observed_prompt_seconds=first_model_token_at - started,
            observed_eval_seconds=completed - first_model_token_at,
        ),
    )
    if on_exchange is not None:
        await on_exchange(LlmExchange(messages=list(messages), response=outcome))
    return outcome


def _parse_stream_packet(
    packet: ChatCompletionChunk,
) -> tuple[int | None, int | None, str, str, str | None]:
    usage = packet.usage
    prompt_eval_count = (
        nonnegative_int_from_llm_field(usage.prompt_tokens) if usage is not None else None
    )
    eval_count = (
        nonnegative_int_from_llm_field(usage.completion_tokens) if usage is not None else None
    )
    if prompt_eval_count is None:
        prompt_eval_count = nonnegative_int_from_llm_field(
            getattr(packet, "prompt_eval_count", None)
        )
    if eval_count is None:
        eval_count = nonnegative_int_from_llm_field(getattr(packet, "eval_count", None))
    content, reasoning = _chunk_delta_text(packet)
    finish_reason: str | None = None
    if packet.choices:
        finish_reason = packet.choices[0].finish_reason
    return prompt_eval_count, eval_count, content, reasoning, finish_reason


__all__ = [
    "LlmChatOutcome",
    "LlmConfig",
    "LlmExchange",
    "ToolCall",
    "nonnegative_int_from_llm_field",
    "openai_api_key",
    "openai_extra_body",
    "parse_structured",
    "schema_response_format",
    "stream_llm_chat",
    "tool_openai_spec",
]
