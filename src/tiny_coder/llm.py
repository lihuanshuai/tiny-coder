"""Responses streaming with tool calls, structured output and local conversation state."""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias, TypeVar, cast

from openai import AsyncOpenAI, AsyncStream, pydantic_function_tool
from openai.types.responses import Response, ResponseStreamEvent, ToolChoiceFunctionParam
from pydantic import BaseModel, ConfigDict, Field

from tiny_coder.json_utils import JsonProtocolError, load_json_object
from tiny_coder.tool_arguments import tool_parameter_schema

logger = logging.getLogger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_ResponseStream: TypeAlias = AsyncStream[ResponseStreamEvent]
ToolChoice: TypeAlias = Literal['none', 'auto', 'required'] | ToolChoiceFunctionParam


class LlmConfig(BaseModel):
    """Configuration required by a Responses-compatible server."""

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
class ToolCallDelta:
    """Provisional tool arguments for display, never for execution."""

    output_index: int
    name: str
    delta: str


@dataclass(frozen=True, slots=True)
class LlmChatOutcome:
    """Result from one complete Responses stream."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_eval_count: int
    eval_count: int
    client_wall_time_ms: float
    started_at: str
    completed_at: str
    llm: dict[str, Any]
    reasoning_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.reasoning_items:
            msg['reasoning_items'] = deepcopy(list(self.reasoning_items))
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
    return number if math.isfinite(number) and number > 0 else None


def _tokens_per_second(token_count: int, duration_seconds: float | None) -> float | None:
    if (
        token_count <= 0
        or duration_seconds is None
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        return None
    return token_count / duration_seconds


def schema_response_format(model: type[BaseModel]) -> dict[str, Any]:
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("model must be a BaseModel subclass")
    schema = pydantic_function_tool(model)['function']['parameters']
    return {
        "type": "json_schema",
        "name": "structured_output",
        "strict": True,
        "schema": schema,
    }


def _llm_tail_metrics_dict(
    packet: Response,
    *,
    done_reason: str | None,
    reasoning_chars: int,
    prompt_eval_count: int,
    eval_count: int,
    server_timings: Mapping[str, float],
    request_seconds: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(server_timings)
    usage = packet.usage
    if usage is not None:
        for value, target_key in (
            (nonnegative_int_from_llm_field(usage.input_tokens), "prompt_eval_count"),
            (nonnegative_int_from_llm_field(usage.output_tokens), "eval_count"),
        ):
            if value is not None:
                metrics[target_key] = value
        if usage.output_tokens_details is not None:
            reasoning_tokens = nonnegative_int_from_llm_field(
                usage.output_tokens_details.reasoning_tokens
            )
            if reasoning_tokens is not None:
                metrics['reasoning_tokens'] = reasoning_tokens
    if packet.model:
        metrics["model"] = packet.model
    if done_reason:
        metrics["done_reason"] = done_reason
    if reasoning_chars:
        metrics["reasoning_chars"] = reasoning_chars
    prompt_duration_ns = server_timings.get("prompt_eval_duration")
    eval_duration_ns = server_timings.get("eval_duration")
    prompt_speed = server_timings.get("prompt_per_second") or _tokens_per_second(
        prompt_eval_count,
        prompt_duration_ns / 1_000_000_000 if prompt_duration_ns is not None else None,
    )
    eval_speed = server_timings.get("predicted_per_second") or _tokens_per_second(
        eval_count,
        eval_duration_ns / 1_000_000_000 if eval_duration_ns is not None else None,
    )
    if prompt_speed is not None:
        metrics["prompt_tokens_per_second"] = prompt_speed
    if eval_speed is not None:
        metrics["eval_tokens_per_second"] = eval_speed
    elif (average := _tokens_per_second(eval_count, request_seconds)) is not None:
        metrics["request_tokens_per_second"] = average
    metrics["provider"] = "openai-responses"
    metrics["response_id"] = packet.id
    metrics["status"] = packet.status
    return metrics


def openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or "not-needed"


def openai_extra_body(llm_cfg: LlmConfig) -> dict[str, Any]:
    return {
        "num_ctx": llm_cfg.num_ctx,
        "repeat_penalty": llm_cfg.repeat_penalty,
    }


def _inline_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Expose nested argument fields to tool renderers that do not follow local refs.

    Recursive or unsupported references retain their original schema. Validation
    still uses the original Pydantic model, including all runtime constraints.
    """
    definitions = schema.get("$defs", {})
    unresolved = False
    annotations = {"title", "description", "default", "examples", "example"}
    data_keywords = {"default", "examples", "example", "enum", "const"}

    def expand(value: Any, trail: tuple[str, ...]) -> Any:
        nonlocal unresolved
        if isinstance(value, list):
            return [expand(item, trail) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
                unresolved = True
                return deepcopy(value)
            name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
            target = definitions.get(name)
            if reference in trail or not isinstance(target, dict):
                unresolved = True
                return deepcopy(value)
            resolved = expand(target, (*trail, reference))
            siblings = expand({key: item for key, item in value.items() if key != "$ref"}, trail)
            if (resolved.keys() & siblings.keys()) - annotations:
                return {"allOf": [resolved, siblings]}
            return {**resolved, **siblings}
        result = {}
        for key, item in value.items():
            if key == "$defs":
                continue
            if key in {"properties", "patternProperties", "dependentSchemas"}:
                result[key] = {name: expand(child, trail) for name, child in item.items()}
            else:
                result[key] = deepcopy(item) if key in data_keywords else expand(item, trail)
        return result

    result = expand(schema, ())
    return deepcopy(schema) if unresolved else result


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
        "name": name,
        "description": str(description or ""),
        "parameters": _inline_tool_schema(tool_parameter_schema(parameters)),
        "strict": True,
    }


def responses_input(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Translate graph history into native Responses items without duplicating tool data.

    Keep the graph's editable messages as the source of truth: retry validation may
    redact rejected function arguments. Reasoning items retain their opaque payload.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get('role')
        if role == 'tool':
            items.append(
                {
                    'type': 'function_call_output',
                    'call_id': message['tool_call_id'],
                    'output': message['content'],
                }
            )
            continue
        if role == 'assistant':
            items.extend(deepcopy(message.get('reasoning_items', [])))
        content = message.get('content')
        if content:
            items.append({'role': role, 'content': content})
        for tool in message.get('tool_calls', []):
            items.append(
                {
                    'type': 'function_call',
                    'call_id': tool['id'],
                    'name': tool['function']['name'],
                    'arguments': tool['function']['arguments'],
                }
            )
    return items


async def stream_llm_chat(
    *,
    llm_cfg: LlmConfig,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[object] | None = None,
    tool_choice: ToolChoice | None = None,
    response_format: dict[str, Any] | None = None,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    on_tool_chunk: Callable[[ToolCallDelta], Awaitable[None]] | None = None,
    on_exchange: Callable[[LlmExchange], Awaitable[None]] | None = None,
    client: AsyncOpenAI | None = None,
) -> LlmChatOutcome:
    """Stream /responses; an injected client remains owned by the caller.

    Conversation state is replayed locally with store=False so retries and resumed
    checkpoints never depend on server-side response retention.
    """
    if not isinstance(llm_cfg, LlmConfig):
        raise TypeError('llm_cfg must inherit LlmConfig')
    if not messages:
        raise ValueError('messages must not be empty')
    if tool_choice is not None and not tools:
        raise ValueError('tool_choice requires tools')
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    owns_client = client is None
    client = (
        client
        if client is not None
        else AsyncOpenAI(
            base_url=llm_cfg.base_url,
            api_key=openai_api_key(),
            timeout=llm_cfg.timeout,
        )
    )
    final: Response | None = None
    reasoning_chars = 0
    server_timings: dict[str, float] = {}
    streamed_text = False
    tool_names: dict[int, str] = {}
    streamed_tools: set[int] = set()
    first_output: float | None = None
    stream: _ResponseStream | None = None
    try:
        kwargs: dict[str, Any] = {
            'model': llm_cfg.llm_model,
            'input': responses_input(messages),
            'temperature': llm_cfg.temperature,
            'stream': True,
            'store': False,
            'extra_body': openai_extra_body(llm_cfg),
            'reasoning': {'effort': 'medium' if llm_cfg.think else 'none'},
        }
        if llm_cfg.max_output_tokens is not None:
            kwargs['max_output_tokens'] = llm_cfg.max_output_tokens
        if tools is not None:
            kwargs['tools'] = [tool_openai_spec(tool) for tool in tools]
        if tool_choice is not None:
            kwargs['tool_choice'] = tool_choice
        if response_format is not None:
            kwargs['text'] = {'format': response_format}
        create_response = cast(Callable[..., Awaitable[_ResponseStream]], client.responses.create)
        stream = await create_response(**kwargs)
        async for event in stream:
            if (
                first_output is None
                and (
                    event.type == 'response.output_text.delta'
                    or event.type == 'response.function_call_arguments.delta'
                    or event.type == 'response.reasoning_text.delta'
                    or event.type == 'response.reasoning_summary_text.delta'
                )
                and event.delta
            ):
                first_output = time.perf_counter()
            if event.type == 'error':
                raise RuntimeError(f'Responses stream error: {event.message}')
            if event.type == 'response.failed':
                error = event.response.error
                raise RuntimeError(
                    f'Responses failed: {error.message if error else "unknown error"}'
                )
            if event.type == 'response.output_text.delta':
                streamed_text = True
                if on_chunk is not None:
                    await on_chunk(event.delta)
            elif event.type == 'response.output_item.added' and event.item.type == 'function_call':
                tool_names[event.output_index] = event.item.name
                if on_tool_chunk is not None:
                    await on_tool_chunk(ToolCallDelta(event.output_index, event.item.name, ''))
            elif event.type == 'response.function_call_arguments.delta':
                name = tool_names.get(event.output_index)
                if on_tool_chunk is not None and name is not None and event.delta:
                    await on_tool_chunk(ToolCallDelta(event.output_index, name, event.delta))
                    streamed_tools.add(event.output_index)
            elif (
                event.type == 'response.reasoning_text.delta'
                or event.type == 'response.reasoning_summary_text.delta'
            ):
                reasoning_chars += len(event.delta)
            elif event.type == 'response.completed' or event.type == 'response.incomplete':
                final = event.response
            # Local servers may attach timing metadata to progress or terminal events.
            for packet in (event, getattr(event, 'response', None)):
                timings = getattr(packet, 'timings', None)
                values = {
                    key: getattr(packet, key, None)
                    for key in ('prompt_eval_duration', 'eval_duration')
                }
                if isinstance(timings, dict):
                    values.update(
                        (key, timings.get(key))
                        for key in (
                            'prompt_per_second',
                            'predicted_per_second',
                            'prompt_ms',
                            'predicted_ms',
                        )
                    )
                for key, value in values.items():
                    if (valid := _positive_float_from_llm_field(value)) is not None:
                        server_timings[key] = valid
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass
        if owns_client:
            try:
                await client.close()
            except Exception:
                pass
    if final is None:
        raise JsonProtocolError(
            'Responses stream ended without a terminal response; partial output rejected'
        )
    if final.status != 'completed':
        reason = final.incomplete_details.reason if final.incomplete_details else 'unknown'
        raise JsonProtocolError(
            f'Responses {final.status}: {reason}; partial text and tool calls rejected'
        )
    tool_calls = tuple(
        ToolCall(
            id=item.call_id,
            type='function',
            function={'name': item.name, 'arguments': item.arguments},
        )
        for item in final.output
        if item.type == 'function_call'
    )
    done_reason = 'tool_calls' if tool_calls else 'stop'
    output_text = final.output_text
    if not output_text and not tool_calls:
        raise RuntimeError(
            f'Responses returned no text or tool calls (finish_reason={done_reason})'
        )
    # The terminal output is authoritative, including providers that buffer text.
    if not streamed_text and output_text and on_chunk is not None:
        await on_chunk(output_text)
    if on_tool_chunk is not None:
        for index, item in enumerate(final.output):
            if item.type == 'function_call' and index not in streamed_tools:
                await on_tool_chunk(ToolCallDelta(index, item.name, item.arguments))
    usage = final.usage
    prompt_eval_count = nonnegative_int_from_llm_field(usage.input_tokens) if usage else None
    eval_count = nonnegative_int_from_llm_field(usage.output_tokens) if usage else None
    if prompt_eval_count is None or eval_count is None:
        logger.debug('Responses usage metrics missing; using zeros for missing counts')
    completed = time.perf_counter()
    outcome = LlmChatOutcome(
        text=output_text,
        tool_calls=tool_calls,
        prompt_eval_count=prompt_eval_count or 0,
        eval_count=eval_count or 0,
        client_wall_time_ms=(completed - started) * 1000.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        reasoning_items=tuple(
            item.model_dump(mode='json', exclude_none=True)
            for item in final.output
            if item.type == 'reasoning'
        ),
        llm=_llm_tail_metrics_dict(
            final,
            done_reason=done_reason,
            reasoning_chars=reasoning_chars,
            prompt_eval_count=prompt_eval_count or 0,
            eval_count=eval_count or 0,
            server_timings=server_timings,
            request_seconds=completed - started,
        ),
    )
    if first_output is not None:
        outcome.llm['time_to_first_output_ms'] = (first_output - started) * 1000
        outcome.llm['stream_output_ms'] = (completed - first_output) * 1000
    if on_exchange is not None:
        await on_exchange(LlmExchange(messages=list(messages), response=outcome))
    return outcome


__all__ = [
    'LlmChatOutcome',
    'LlmConfig',
    'LlmExchange',
    'ToolCall',
    'ToolCallDelta',
    'ToolChoice',
    'nonnegative_int_from_llm_field',
    'openai_api_key',
    'openai_extra_body',
    'parse_structured',
    'responses_input',
    'schema_response_format',
    'stream_llm_chat',
    'tool_openai_spec',
]
