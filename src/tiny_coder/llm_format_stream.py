from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, cast

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

UNSUPPORTED_SCHEMA_HINT = (
    "Current OpenAI-compatible LLM server does not support JSON Schema structured output "
    "(response_format.json_schema). Please confirm that the server and loaded model support it."
)


class _OpenAIChatConfig(Protocol):
    base_url: str
    llm_model: str
    num_ctx: int
    temperature: float
    repeat_penalty: float
    think: bool
    timeout: float


@dataclass(frozen=True, slots=True)
class LlmChatStreamOutcome:
    """Summary and metrics from one OpenAI-compatible chat stream."""

    text: str
    prompt_eval_count: int
    eval_count: int
    client_wall_time_ms: float
    started_at: str
    completed_at: str
    llm: dict[str, Any]


def _get_value(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def nonnegative_int_from_llm_field(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0 and value == int(value):
        return int(value)
    return None


def _schema_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "strict": True,
            "schema": schema,
        },
    }


def _chunk_delta_text(packet: object) -> tuple[str, str]:
    choices = _get_value(packet, "choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    choice = choices[0]
    delta = _get_value(choice, "delta")
    content = _get_value(delta, "content")
    reasoning = _get_value(delta, "reasoning_content")
    if reasoning is None:
        reasoning = _get_value(delta, "reasoning")
    return (
        "" if content is None else str(content),
        "" if reasoning is None else str(reasoning),
    )


def parse_llm_stream_chunk(
    packet: object,
) -> tuple[int | None, int | None, str, str, str | None]:
    """Read metrics, output text, reasoning, and finish reason from one stream packet."""
    usage = _get_value(packet, "usage")
    prompt_eval_count = nonnegative_int_from_llm_field(_get_value(usage, "prompt_tokens"))
    eval_count = nonnegative_int_from_llm_field(_get_value(usage, "completion_tokens"))
    if prompt_eval_count is None:
        prompt_eval_count = nonnegative_int_from_llm_field(_get_value(packet, "prompt_eval_count"))
    if eval_count is None:
        eval_count = nonnegative_int_from_llm_field(_get_value(packet, "eval_count"))
    content, reasoning = _chunk_delta_text(packet)
    choices = _get_value(packet, "choices")
    finish_reason: str | None = None
    if isinstance(choices, list) and choices:
        finish_value = _get_value(choices[0], "finish_reason")
        if isinstance(finish_value, str) and finish_value:
            finish_reason = finish_value
    if not content:
        message = _get_value(packet, "message")
        value = _get_value(message, "content")
        content = "" if value is None else str(value)
    return prompt_eval_count, eval_count, content, reasoning, finish_reason


def _stream_error_message(packet: object) -> str | None:
    error = _get_value(packet, "error")
    if error is None:
        return None
    message = _get_value(error, "message")
    if isinstance(message, str) and message:
        return message
    return str(error)


def _llm_tail_metrics_dict(
    packet: object | None,
    *,
    done_reason: str | None,
    reasoning_chars: int,
) -> dict[str, Any]:
    if packet is None:
        return {}
    metrics: dict[str, Any] = {}
    usage = _get_value(packet, "usage")
    for source_key, target_key in (
        ("prompt_tokens", "prompt_eval_count"),
        ("completion_tokens", "eval_count"),
    ):
        value = nonnegative_int_from_llm_field(_get_value(usage, source_key))
        if value is not None:
            metrics[target_key] = value
    model = _get_value(packet, "model")
    if isinstance(model, str) and model:
        metrics["model"] = model
    if done_reason:
        metrics["done_reason"] = done_reason
    if reasoning_chars:
        metrics["reasoning_chars"] = reasoning_chars
    metrics["provider"] = "openai-compatible"
    return metrics


def openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or "not-needed"


def _require_openai_chat_config(llm_cfg: BaseModel) -> _OpenAIChatConfig:
    required = (
        "base_url",
        "llm_model",
        "num_ctx",
        "temperature",
        "repeat_penalty",
        "think",
        "timeout",
    )
    missing = [name for name in required if not hasattr(llm_cfg, name)]
    if missing:
        raise TypeError("llm_cfg is missing OpenAI-compatible fields: " + ", ".join(missing))
    config = cast(_OpenAIChatConfig, llm_cfg)
    if config.timeout <= 0:
        raise ValueError("llm_cfg.timeout must be greater than zero")
    return config


def openai_extra_body(llm_cfg: BaseModel) -> dict[str, Any]:
    config = _require_openai_chat_config(llm_cfg)
    return {
        "num_ctx": config.num_ctx,
        "repeat_penalty": config.repeat_penalty,
        "think": config.think,
    }


async def stream_llm_chat_format(
    *,
    llm_cfg: BaseModel,
    system: str,
    prompt: str,
    response_format: dict[str, Any],
    on_chunk: Callable[[str], None],
) -> LlmChatStreamOutcome:
    """Stream an OpenAI-compatible chat completion with JSON Schema output."""
    if not isinstance(response_format, dict) or not response_format:
        raise RuntimeError("response_format must be a non-empty JSON Schema object.")
    config = _require_openai_chat_config(llm_cfg)
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
    last_packet: object | None = None
    done_reason: str | None = None
    reasoning_chars = 0
    collected: list[str] = []

    def dispatch_chunk(chunk: str) -> None:
        collected.append(chunk)
        on_chunk(chunk)

    stream_response: AsyncIterator[object] | None = None
    try:
        try:
            create_completion = cast(Callable[..., Any], client.chat.completions.create)
            reasoning_options = {} if config.think else {"reasoning_effort": "none"}
            stream_response = await create_completion(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                response_format=_schema_response_format(response_format),
                stream=True,
                stream_options={"include_usage": True},
                extra_body=openai_extra_body(llm_cfg),
                **reasoning_options,
            )
        except Exception as error:
            message = str(error).lower()
            if any(
                marker in message for marker in ("format", "grammar", "schema", "response_format")
            ):
                raise RuntimeError(UNSUPPORTED_SCHEMA_HINT) from error
            raise
        async for packet in stream_response:
            last_packet = packet
            error_message = _stream_error_message(packet)
            if error_message is not None:
                raise RuntimeError(f"OpenAI-compatible LLM stream error: {error_message}")
            prompt_count, completion_count, content, reasoning, finish_reason = (
                parse_llm_stream_chunk(packet)
            )
            if prompt_count is not None:
                prompt_eval_count = prompt_count
                had_prompt_metric = True
            if completion_count is not None:
                eval_count = completion_count
                had_eval_metric = True
            reasoning_chars += len(reasoning)
            if finish_reason is not None:
                done_reason = finish_reason
            if content:
                dispatch_chunk(content)
    finally:
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
    if not collected:
        raise RuntimeError(
            "OpenAI-compatible LLM stream returned no content "
            f"(finish_reason={done_reason or 'unknown'}, "
            f"prompt_eval_count={prompt_eval_count}, eval_count={eval_count}, "
            f"reasoning_chars={reasoning_chars})"
        )
    completed_at = datetime.now(timezone.utc).isoformat()
    return LlmChatStreamOutcome(
        text="".join(collected),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        client_wall_time_ms=(time.perf_counter() - started) * 1000.0,
        started_at=started_at,
        completed_at=completed_at,
        llm=_llm_tail_metrics_dict(
            last_packet,
            done_reason=done_reason,
            reasoning_chars=reasoning_chars,
        ),
    )


__all__ = [
    "LlmChatStreamOutcome",
    "nonnegative_int_from_llm_field",
    "openai_api_key",
    "openai_extra_body",
    "parse_llm_stream_chunk",
    "stream_llm_chat_format",
]
