from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TypedDict, TypeVar

from jinja2 import Template
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError

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
    """Render fresh prompts before each call, retaining completed tool exchanges."""

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
        if any(message.get("role") == "tool" for message in call.messages):
            for index, message in enumerate(call.messages):
                if message.get("role") == "user":
                    call.messages[index] = user
                    break
        else:
            call.messages = [user]


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
            raise ValueError(f"invalid structured output: {error}") from error
        if output is None:
            raise ValueError("invalid structured output: response is not a JSON object")
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

    async def retry(self, call: AgentCall) -> None:
        error = call.error
        if isinstance(
            error, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        ):
            failure = "LLM call failed"
        elif isinstance(error, (ValueError, OSError)):
            call.feedback.append(str(error))
            failure = "failed"
        else:
            return
        limit = self._attempt_limit(call)
        if call.attempt >= limit:
            raise RuntimeError(f"{failure} after {limit} attempt(s): {error}") from error
        call.retry = True


__all__ = [
    "JinjaPrompt",
    "StreamOutput",
    "StructuredOutput",
    "CallRecording",
    "RetryPolicy",
    "FileSnapshot",
    "read_file_snapshots",
    "render_prompt",
]
