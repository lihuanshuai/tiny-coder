from __future__ import annotations

from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict, TypeVar, cast

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError

from tiny_coder.executable_script import ScriptExecutionResult
from tiny_coder.json_utils import JsonProtocolError, load_json_object
from tiny_coder.llm_format_stream import stream_llm_chat_format
from tiny_coder.plugins import (
    AfterConversationHandler,
    BeforeConversationHandler,
    ConversationConditionHandler,
    ConversationPlugin,
    require_output_model_type,
    resolve_agent_file_path,
    resolve_agent_root,
)

_OutputT = TypeVar("_OutputT", bound=BaseModel)
_SystemPromptCallback: TypeAlias = Callable[["ConversationContext"], str]
_UserPromptCallback: TypeAlias = Callable[["ConversationContext", Sequence[str]], str]
_OutputWriterCallback: TypeAlias = Callable[["ConversationContext", BaseModel], list[Path]]
_ConversationCompletedCallback: TypeAlias = Callable[
    ["ConversationContext", "ConversationResult"], Awaitable[None]
]
_AfterLlmCallCallback: TypeAlias = Callable[
    ["ConversationContext", "LlmCallOutcome"], Coroutine[Any, Any, None]
]

_CODE_FENCE_LANG_BY_SUFFIX = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonl": "jsonl",
    ".txt": "text",
}


class _AgentInputSnapshot(TypedDict):
    label: str
    language: str
    content: str


class LlmCallOutcome(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def prompt_eval_count(self) -> int: ...

    @property
    def eval_count(self) -> int: ...


class LlmCall(Protocol):
    def __call__(
        self,
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        on_chunk: Callable[[str], None],
    ) -> Coroutine[Any, Any, LlmCallOutcome]: ...


def _default_llm_call() -> LlmCall:
    return stream_llm_chat_format


@dataclass
class ConversationResult:
    """Validated result and model metadata produced by one conversation."""

    key: str
    summary: str
    written_paths: list[Path]
    output: BaseModel
    outcome: LlmCallOutcome
    script_execution: ScriptExecutionResult | None = None


@dataclass
class AgentResult:
    """Ordered results from the conversations completed by one agent run."""

    conversations: dict[str, ConversationResult]

    @property
    def written_paths(self) -> list[Path]:
        return [
            path
            for conversation in self.conversations.values()
            for path in conversation.written_paths
        ]


@dataclass
class AgentContext:
    """State shared across conversations in one agent run."""

    cwd: Path
    conversation_results: dict[str, ConversationResult] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Configuration and runtime state for one conversation."""

    agent: AgentContext
    key: str
    llm_config: BaseModel | None = None
    llm_call: LlmCall = field(default_factory=_default_llm_call)
    max_attempts: int = 1
    llm_call_outcome: LlmCallOutcome | None = None
    llm_call_system_prompt: str = ""
    llm_call_user_prompt: str = ""
    input_paths: list[Path] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)
    before_conversation_hooks: list[BeforeConversationHandler] = field(default_factory=list)
    after_conversation_hooks: list[AfterConversationHandler] = field(default_factory=list)
    conversation_completed_hooks: list[_ConversationCompletedCallback] = field(default_factory=list)
    system_prompt_hooks: list[_SystemPromptCallback] = field(default_factory=list)
    user_prompt_hooks: list[_UserPromptCallback] = field(default_factory=list)
    after_llm_call_hooks: list[_AfterLlmCallCallback] = field(default_factory=list)
    llm_response_output_type: type[BaseModel] | None = None
    llm_response_output: BaseModel | None = None
    output_writer: _OutputWriterCallback | None = None
    allow_overwrite_existing_paths: bool = True
    clean_up_paths: list[Path] = field(default_factory=list)
    should_run: ConversationConditionHandler | None = None
    script_execution_result: ScriptExecutionResult | None = None

    @property
    def cwd(self) -> Path:
        return self.agent.cwd

    @property
    def previous(self) -> Mapping[str, ConversationResult]:
        """Expose completed earlier conversations as read-only shared context."""
        return self.agent.conversation_results

    @property
    def extras(self) -> dict[str, Any]:
        return self.agent.extras


@dataclass
class Conversation:
    """One configured LLM conversation in a multi-turn agent flow."""

    key: str
    plugins: list[ConversationPlugin]
    context: ConversationContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("conversation key must not be blank")

    def register(self, agent_context: AgentContext) -> None:
        """Build this conversation context and register all conversation plugins."""
        self.context = ConversationContext(agent=agent_context, key=self.key)
        for plugin in self.plugins:
            if not isinstance(plugin, ConversationPlugin):
                raise TypeError("plugin must inherit ConversationPlugin")
            plugin.on_registered(self.context)
        self._validate_registration()

    def _validate_registration(self) -> None:
        if self.context.llm_config is None:
            raise ValueError(f"conversation {self.key!r} requires LlmConfigPlugin")
        if self.context.llm_response_output_type is None:
            raise ValueError(f"conversation {self.key!r} requires ResponseOutputTypePlugin")
        if self.context.output_writer is None:
            raise ValueError(f"conversation {self.key!r} requires an output writer plugin")
        if not self.context.system_prompt_hooks:
            raise ValueError(f"conversation {self.key!r} requires a system prompt plugin")
        if not self.context.user_prompt_hooks:
            raise ValueError(f"conversation {self.key!r} requires a user prompt plugin")


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(resolve_agent_root(root)).as_posix()
    except ValueError:
        return str(path)


def read_agent_file(root: Path, path: Path) -> str:
    """Read a UTF-8 text file from the agent workspace."""
    target = resolve_agent_file_path(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"file not found: {_relative_label(root, target)}")
    with target.open("r", encoding="utf-8", newline="\n") as f:
        return f.read()


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as f:
        return f.read()


def _code_fence_language(path: Path) -> str:
    return _CODE_FENCE_LANG_BY_SUFFIX.get(path.suffix.lower(), "text")


def agent_input_snapshots(root: Path, input_paths: list[Path]) -> list[_AgentInputSnapshot]:
    """Read agent input files and return template-ready snapshot objects."""
    return [
        {
            "label": _relative_label(root, path),
            "language": _code_fence_language(path),
            "content": _read_text_file(path).rstrip("\n"),
        }
        for path in input_paths
    ]


def parse_structured_output(raw_output: str, *, output_type: type[_OutputT]) -> _OutputT:
    """Parse the model JSON response into the conversation's output model."""
    try:
        payload = load_json_object(raw_output)
        model = require_output_model_type(output_type)
        return cast(_OutputT, model.model_validate(payload))
    except (JsonProtocolError, ValidationError) as error:
        raise ValueError(f"invalid structured output: {error}") from error


def _print_llm_chunk(chunk: str) -> None:
    print(chunk, end="", flush=True)


_RETRYABLE_LLM_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


@dataclass(kw_only=True)
class BasicFileAgent:
    """Lazily run ordered conversations and integrate their context."""

    cwd: Path
    conversations: Iterable[Conversation] | AsyncIterable[Conversation]
    context: AgentContext = field(init=False)

    def __post_init__(self) -> None:
        self.cwd = resolve_agent_root(self.cwd)
        self.context = AgentContext(cwd=self.cwd)

    @staticmethod
    def _validate_conversation(
        conversation: Conversation,
        seen_keys: set[str],
    ) -> None:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversations must yield Conversation instances")
        if conversation.key in seen_keys:
            raise ValueError(f"duplicate conversation key: {conversation.key}")
        seen_keys.add(conversation.key)

    async def run(self) -> AgentResult:
        """Consume and run synchronous or asynchronous conversations in order."""
        self.context.conversation_results.clear()
        seen_keys: set[str] = set()
        async for conversation in self._iter_conversations():
            self._validate_conversation(conversation, seen_keys)
            conversation.register(self.context)
            context = conversation.context
            self._reset_conversation(context)
            if context.should_run is not None and not context.should_run(context):
                continue
            result = await self._run_conversation(context)
            self.context.conversation_results[conversation.key] = result
        if not seen_keys:
            raise ValueError("BasicFileAgent requires at least one Conversation")
        return AgentResult(conversations=dict(self.context.conversation_results))

    async def _iter_conversations(self) -> AsyncIterator[Conversation]:
        if isinstance(self.conversations, AsyncIterable):
            async for conversation in self.conversations:
                yield conversation
            return
        for conversation in self.conversations:
            yield conversation

    @staticmethod
    def _reset_conversation(context: ConversationContext) -> None:
        context.llm_call_outcome = None
        context.llm_call_system_prompt = ""
        context.llm_call_user_prompt = ""
        context.llm_response_output = None
        context.script_execution_result = None

    async def _run_conversation(self, context: ConversationContext) -> ConversationResult:
        self._clean_up_files(context)
        for before_hook in context.before_conversation_hooks:
            await before_hook(context)
        self._guard_existing_output_paths(context)

        system_prompt = self._build_system_prompt(context)
        retry_errors: list[str] = []
        for attempt in range(1, context.max_attempts + 1):
            user_prompt = self._build_user_prompt(context, retry_errors)
            try:
                result = await self._call_and_apply_output(
                    context,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                for after_hook in context.after_conversation_hooks:
                    await after_hook(context, result)
                result.script_execution = context.script_execution_result
            except _RETRYABLE_LLM_ERRORS as error:
                if attempt >= context.max_attempts:
                    raise RuntimeError(
                        f"conversation {context.key!r} LLM call failed after "
                        f"{context.max_attempts} attempt(s): {error}"
                    ) from error
                self._print_attempt_error(context, attempt, str(error))
                continue
            except (ValueError, OSError) as error:
                error_text = str(error)
                retry_errors.append(error_text)
                if attempt >= context.max_attempts:
                    raise RuntimeError(
                        f"conversation {context.key!r} failed after "
                        f"{context.max_attempts} attempt(s): {error_text}"
                    ) from error
                self._print_attempt_error(context, attempt, error_text)
                continue
            for completed_hook in context.conversation_completed_hooks:
                await completed_hook(context, result)
            if result.summary:
                print(result.summary.strip())
            return result
        raise RuntimeError(f"conversation {context.key!r} unexpectedly exhausted retries")

    async def _call_and_apply_output(
        self,
        context: ConversationContext,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ConversationResult:
        llm_config = context.llm_config
        output_type = context.llm_response_output_type
        output_writer = context.output_writer
        if llm_config is None or output_type is None or output_writer is None:
            raise RuntimeError(f"conversation {context.key!r} is not fully configured")

        context.llm_call_system_prompt = system_prompt
        context.llm_call_user_prompt = user_prompt
        outcome = await context.llm_call(
            llm_cfg=llm_config,
            system=system_prompt,
            prompt=user_prompt,
            response_format=require_output_model_type(output_type).model_json_schema(),
            on_chunk=_print_llm_chunk,
        )
        context.llm_call_outcome = outcome
        for hook in context.after_llm_call_hooks:
            await hook(context, outcome)
        if outcome.text:
            print()

        output = parse_structured_output(outcome.text, output_type=output_type)
        context.llm_response_output = output
        written_paths = output_writer(context, output)
        self._validate_written_paths(context, written_paths)
        summary = getattr(output, "summary", "")
        return ConversationResult(
            key=context.key,
            summary=summary if isinstance(summary, str) else "",
            written_paths=written_paths,
            output=output,
            outcome=outcome,
        )

    @staticmethod
    def _build_system_prompt(context: ConversationContext) -> str:
        prompt = ""
        for hook in context.system_prompt_hooks:
            prompt = hook(context)
        return prompt

    @staticmethod
    def _build_user_prompt(context: ConversationContext, retry_errors: Sequence[str]) -> str:
        prompt = ""
        for hook in context.user_prompt_hooks:
            prompt = hook(context, retry_errors)
        return prompt

    @staticmethod
    def _guard_existing_output_paths(context: ConversationContext) -> None:
        if context.allow_overwrite_existing_paths:
            return
        for path in context.output_paths:
            if path.exists():
                label = _relative_label(context.cwd, path)
                raise RuntimeError(f"refusing to overwrite existing path: {label}")

    @staticmethod
    def _clean_up_files(context: ConversationContext) -> None:
        for path in context.clean_up_paths:
            if path.is_dir():
                label = _relative_label(context.cwd, path)
                raise RuntimeError(f"cleanup path is a directory: {label}")
            path.unlink(missing_ok=True)

    @staticmethod
    def _validate_written_paths(
        context: ConversationContext,
        written_paths: list[Path],
    ) -> None:
        allowed = set(context.output_paths)
        extra = set(written_paths) - allowed
        if allowed and extra:
            labels = ", ".join(_relative_label(context.cwd, path) for path in sorted(extra))
            raise ValueError(f"output writer returned unexpected paths: extra={labels}")

    @staticmethod
    def _print_attempt_error(
        context: ConversationContext,
        attempt: int,
        error: str,
    ) -> None:
        print(
            f"\nConversation {context.key!r} failed "
            f"(attempt {attempt}/{context.max_attempts}): {error}"
        )


__all__ = [
    "AgentContext",
    "AgentResult",
    "BasicFileAgent",
    "Conversation",
    "ConversationContext",
    "ConversationResult",
    "LlmCall",
    "LlmCallOutcome",
    "agent_input_snapshots",
    "parse_structured_output",
    "read_agent_file",
]
