from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import NoReturn, TypedDict, TypeVar

from jinja2 import Template
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from tiny_coder.apply_patch import resolve_agent_file_path, resolve_agent_root
from tiny_coder.eventbus import EventBus
from tiny_coder.json_utils import JsonStringFieldStreamer
from tiny_coder.llm import LlmConfig
from tiny_coder.structured_agent import (
    AFTER_CALL,
    BEFORE_CALL,
    CALL_FAILED,
    CALL_STARTED,
    CHUNK_RECEIVED,
    AgentCall,
    AgentExtension,
)

_T = TypeVar("_T")


class ResponseValidationError(ValueError):
    """A rejected response with actionable guidance for the next attempt."""

    def __init__(
        self,
        message: str,
        *,
        suggestion: str = (
            "根据字段错误逐项修正上一份结果，保留已满足的约束；"
            "重新校验后提交完整 JSON 对象，不要只返回修改说明、局部字段或原样重复无效内容。"
        ),
    ) -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message}\n响应建议：{suggestion}")


def _response_suggestion(error: Exception) -> str:
    if isinstance(error, ResponseValidationError):
        return error.suggestion
    if isinstance(error, OSError):
        return "检查失败的文件路径、访问权限和输入文件，解决读写问题后重试。"
    return (
        "根据错误修正上一份结果，保留已满足的约束并遵守原输出格式；"
        "重新校验后提交完整结果，不要只返回修改说明或原样重复无效内容。"
    )


def _error_feedback(error: Exception) -> str:
    if isinstance(error, ResponseValidationError):
        return str(error)
    return f"{error}\n响应建议：{_response_suggestion(error)}"


def _validation_failure_message(failure: ErrorDetails) -> str:
    original = failure.get("ctx", {}).get("error")
    return original.message if isinstance(original, ResponseValidationError) else failure["msg"]


def _validation_error_message(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, failure['loc'])) or '$'}: {_validation_failure_message(failure)}"
        for failure in error.errors(include_url=False)
    )


def _reraise_response_validation_error(error: ValidationError) -> NoReturn:
    """Preserve validator-owned errors and advice across Pydantic's wrapper."""
    failures = error.errors(include_url=False)
    suggestions: list[str] = []
    for failure in failures:
        original = failure.get("ctx", {}).get("error")
        if isinstance(original, ResponseValidationError):
            if len(failures) == 1:
                raise original from error
            location = ".".join(map(str, failure["loc"])) or "$"
            suggestions.append(f"{location}: {original.suggestion}")
    message = f"invalid structured output: {_validation_error_message(error)}"
    if suggestions:
        raise ResponseValidationError(message, suggestion="\n".join(suggestions)) from error
    raise ResponseValidationError(message) from error


def _resolve(value: _T | Callable[[AgentCall], _T], call: AgentCall) -> _T:
    return value(call) if callable(value) else value


class FileSnapshot(TypedDict):
    path: str
    language: str
    content: str


def read_file_snapshots(cwd: Path, paths: Sequence[Path]) -> list[FileSnapshot]:
    root = resolve_agent_root(cwd)
    snapshots: list[FileSnapshot] = []
    for source in paths:
        path = resolve_agent_file_path(root, source)
        with path.open("r", encoding="utf-8", newline="\n") as file:
            content = file.read()
        snapshots.append(
            {
                "path": path.relative_to(root).as_posix(),
                "language": "markdown" if path.suffix.lower() == ".md" else "text",
                "content": content.rstrip("\n"),
            }
        )
    return snapshots


async def render_prompt(prompt: str | Template, variables: Mapping[str, object]) -> str:
    if isinstance(prompt, str):
        return prompt
    rendered = await asyncio.to_thread(prompt.render, **variables)
    return rendered.rstrip() + "\n"


@dataclass
class JinjaPrompt(AgentExtension):
    """Refresh the initial prompts while retaining conversation history."""

    system: str | Template | Callable[[AgentCall], str | Template]
    user: str | Template | Callable[[AgentCall], str | Template]
    variables: Mapping[str, object] | Callable[[AgentCall], Mapping[str, object]] = dataclass_field(
        default_factory=dict
    )

    def register(self, events: EventBus) -> None:
        events.subscribe(BEFORE_CALL, self.prepare)

    async def prepare(self, call: AgentCall) -> None:
        source = _resolve(self.variables, call)
        variables = {**source, "retry_errors": list(call.feedback)}
        system_prompt = self.system(call) if callable(self.system) else self.system
        user_prompt = self.user(call) if callable(self.user) else self.user
        call.system_prompt = await render_prompt(system_prompt, variables)
        user = {"role": "user", "content": await render_prompt(user_prompt, variables)}
        for index, message in enumerate(call.messages):
            if message.get("role") == "user":
                call.messages[index] = user
                return
        call.messages.insert(0, user)


def _print_chunk(text: str) -> None:
    print(text, end="", flush=True)


@dataclass
class _Stream:
    streamer: JsonStringFieldStreamer | None
    max_chars: int | None
    sink: Callable[[str], None] | None
    raw_len: int = 0
    displayed: bool = False


@dataclass
class StreamOutput(AgentExtension):
    """Display raw text or one JSON string field and optionally limit stream size."""

    field: str | None | Callable[[AgentCall], str | None] = None
    max_chars: int | None | Callable[[AgentCall], int | None] = None
    sink: Callable[[str], None] | None = _print_chunk
    enabled: bool | Callable[[AgentCall], bool] = True
    _stream: ContextVar[_Stream] = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._stream = ContextVar("stream_output")
        if isinstance(self.max_chars, int) and self.max_chars < 0:
            raise ValueError("max_chars must not be negative")

    def register(self, events: EventBus) -> None:
        events.subscribe(CALL_STARTED, self.start)
        events.subscribe(CHUNK_RECEIVED, self.feed)
        events.subscribe(AFTER_CALL, self.finish)

    async def start(self, call: AgentCall) -> None:
        field = _resolve(self.field, call)
        max_chars = _resolve(self.max_chars, call)
        if max_chars is not None and max_chars < 0:
            raise ValueError("max_chars must not be negative")
        token = self._stream.set(
            _Stream(
                JsonStringFieldStreamer(field) if field else None,
                max_chars,
                self.sink if _resolve(self.enabled, call) else None,
            )
        )
        call.resources.callback(self._stream.reset, token)

    async def feed(self, call: AgentCall) -> None:
        stream = self._stream.get()
        stream.raw_len += len(call.chunk)
        displayed = stream.streamer.feed(call.chunk) if stream.streamer is not None else call.chunk
        if displayed and stream.sink is not None:
            stream.sink(displayed)
            stream.displayed = True
        if stream.max_chars is not None and stream.raw_len > stream.max_chars:
            raise ValueError(
                f"stream aborted at {stream.raw_len} characters (limit {stream.max_chars})"
            )

    async def finish(self, call: AgentCall) -> None:
        stream = self._stream.get()
        if stream.sink is not None and (
            stream.displayed
            or (stream.streamer is None and call.outcome is not None and call.outcome.text)
        ):
            stream.sink("\n")


@dataclass
class StructuredOutput(AgentExtension):
    """Parse final responses and accept them through an optional async callback."""

    accept: Callable[[AgentCall, BaseModel], Awaitable[None]] | None = None

    def register(self, events: EventBus) -> None:
        events.subscribe(AFTER_CALL, self.parse)

    async def parse(self, call: AgentCall) -> None:
        assert call.outcome is not None
        if call.outcome.tool_calls:
            return
        if call.response_model is None:
            raise ValueError("structured output requires a response_model")
        try:
            output = call.outcome.model(call.response_model)
        except ValidationError as error:
            _reraise_response_validation_error(error)
        if output is None:
            raise ResponseValidationError(
                "invalid structured output: response is not a JSON object",
                suggestion=(
                    "返回单个完整、合法且符合响应 Schema 的 JSON 对象；"
                    "检查引号、转义和括号，不要添加代码块、解释文字或额外对象。"
                ),
            )
        if self.accept is not None:
            await self.accept(call, output)
        call.output = output


@dataclass
class CallRecording(AgentExtension):
    """Record raw responses before parsing or acceptance, including rejected candidates."""

    path: Path
    metadata: Mapping[str, object] | Callable[[AgentCall], Mapping[str, object]] = dataclass_field(
        default_factory=dict
    )

    def register(self, events: EventBus) -> None:
        events.subscribe(AFTER_CALL, self.record)

    async def record(self, call: AgentCall) -> None:
        assert call.outcome is not None
        record = {
            "system": call.system_prompt,
            "user": next(
                (
                    message.get("content", "")
                    for message in call.messages
                    if message.get("role") == "user"
                ),
                "",
            ),
            "output": call.outcome.text,
            "stats": {
                "prompt_eval_count": call.outcome.prompt_eval_count,
                "eval_count": call.outcome.eval_count,
            },
            **_resolve(self.metadata, call),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class RetryPolicy(AgentExtension):
    """Bound retries and select an alternate config after validation failures."""

    max_attempts: int | Callable[[AgentCall], int]
    validation_llm_config: LlmConfig | None | Callable[[AgentCall], LlmConfig | None] = None

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, int) and self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def register(self, events: EventBus) -> None:
        events.subscribe(BEFORE_CALL, self.configure)
        events.subscribe(CALL_FAILED, self.retry)

    async def configure(self, call: AgentCall) -> None:
        self._attempt_limit(call)
        if call.feedback:
            config = _resolve(self.validation_llm_config, call)
            if config is not None:
                call.llm_config = config

    def _attempt_limit(self, call: AgentCall) -> int:
        limit = _resolve(self.max_attempts, call)
        if limit < 1:
            raise ValueError("max_attempts must be at least 1")
        return limit

    def _check_validation_progress(self, call: AgentCall, error: ValueError) -> None:
        if call.outcome is None or call.previous_outcome is None:
            return
        current = error.__cause__
        previous = call.previous_error.__cause__ if call.previous_error is not None else None
        if isinstance(current, ValidationError) and isinstance(previous, ValidationError):
            failures = current.errors(include_url=False, include_context=False)
            if failures != previous.errors(include_url=False, include_context=False):
                return
            explanation = _validation_error_message(current)
        elif (
            isinstance(error, ResponseValidationError)
            and isinstance(call.previous_error, ResponseValidationError)
            and error.message == call.previous_error.message
            and call.outcome.text == call.previous_outcome.text
        ):
            explanation = error.message
        else:
            return
        raise RuntimeError(
            f"repeated validation failure after {call.attempt} attempt(s): {explanation}\n"
            f"响应建议：{_response_suggestion(error)}"
        ) from error

    def _add_feedback(self, call: AgentCall) -> None:
        if call.outcome is None or call.outcome.tool_calls:
            return
        assert call.error is not None
        cause = call.error.__cause__
        errors: list[dict[str, object]] = (
            [
                {
                    "loc": failure["loc"],
                    "type": failure["type"],
                    "msg": _validation_failure_message(failure),
                }
                for failure in cause.errors(include_url=False)
            ]
            if isinstance(cause, ValidationError)
            else [
                {
                    "msg": call.error.message
                    if isinstance(call.error, ResponseValidationError)
                    else str(call.error)
                }
            ]
        )
        call.messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "validation_errors": errors,
                        "response_suggestion": _response_suggestion(call.error),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    async def retry(self, call: AgentCall) -> None:
        error = call.error
        detail = str(error)
        if isinstance(
            error, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        ):
            failure = "LLM call failed"
        elif isinstance(error, (ValueError, OSError)):
            if isinstance(error, ValueError):
                self._check_validation_progress(call, error)
            detail = _error_feedback(error)
            call.feedback.append(detail)
            failure = "failed"
        else:
            return
        limit = self._attempt_limit(call)
        if call.attempt >= limit:
            raise RuntimeError(f"{failure} after {limit} attempt(s): {detail}") from error
        if isinstance(error, (ValueError, OSError)):
            self._add_feedback(call)
        call.retry = True


__all__ = [
    "ResponseValidationError",
    "JinjaPrompt",
    "StreamOutput",
    "StructuredOutput",
    "CallRecording",
    "RetryPolicy",
    "FileSnapshot",
    "read_file_snapshots",
    "render_prompt",
]
