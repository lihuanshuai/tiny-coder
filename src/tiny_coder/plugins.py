from __future__ import annotations

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypedDict,
)

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from tiny_coder.apply_patch import (
    ApplyPatchOutput,
    apply_patches,
    resolve_agent_file_path,
    resolve_agent_root,
)
from tiny_coder.executable_script import ExecutableScriptOutput, ScriptExecutionResult
from tiny_coder.json_utils import JsonStringFieldStreamer

if TYPE_CHECKING:
    from tiny_coder.file_agent import (
        ConversationContext,
        ConversationResult,
        LlmCall,
        LlmCallOutcome,
    )


_SystemPromptProvider: TypeAlias = str | Callable[["ConversationContext"], str]
_UserPromptProvider: TypeAlias = str | Callable[["ConversationContext", Sequence[str]], str]
_TemplateVars: TypeAlias = Mapping[str, Any] | Callable[["ConversationContext"], Mapping[str, Any]]
UserTemplateVars: TypeAlias = (
    Mapping[str, Any] | Callable[["ConversationContext"], Mapping[str, Any]]
)
_BeforeConversationCallback: TypeAlias = Callable[["ConversationContext"], Awaitable[None]]
_AfterConversationCallback: TypeAlias = Callable[
    ["ConversationContext", "ConversationResult"], Awaitable[None]
]
_ExecutionConfirmation: TypeAlias = Callable[["ConversationContext", BaseModel], Awaitable[bool]]
_ConversationCondition: TypeAlias = Callable[["ConversationContext"], bool]
_JsonScalar: TypeAlias = str | int | float | bool | None
_JsonValue: TypeAlias = _JsonScalar | list["_JsonValue"] | dict[str, "_JsonValue"]
_LlmCallTokenStatsHandler: TypeAlias = Callable[
    ["ConversationContext", "LlmCallTokenStats"], Awaitable[None]
]


class LlmCallJsonlRecorderExtraHandler(ABC):
    """Provide JSON-compatible metadata for one recorded LLM call."""

    @abstractmethod
    def __call__(
        self,
        context: ConversationContext,
        outcome: LlmCallOutcome,
        /,
    ) -> Mapping[str, _JsonValue]: ...


class _LlmCallStats(TypedDict):
    prompt_eval_count: int
    eval_count: int


class _LlmCallJsonlRecord(TypedDict):
    conversation_key: str
    system: str
    user: str
    output: str
    stats: _LlmCallStats
    extra: dict[str, _JsonValue]


class ConversationPlugin(ABC):
    """Register one focused capability on a conversation context."""

    @abstractmethod
    def on_registered(self, context: ConversationContext) -> None: ...


@dataclass
class StaticInputPathsPlugin(ConversationPlugin):
    """Provide a fixed set of readable input paths for one conversation."""

    paths: list[Path]

    def on_registered(self, context: ConversationContext) -> None:
        """Resolve configured paths once the core agent context is available."""
        context.input_paths = _resolve_agent_file_paths(context, self.paths)


@dataclass
class StaticOutputPathsPlugin(ConversationPlugin):
    """Provide a fixed set of writable output paths for one conversation."""

    paths: list[Path]

    def on_registered(self, context: ConversationContext) -> None:
        """Resolve configured paths once the core agent context is available."""
        context.output_paths = _resolve_agent_file_paths(context, self.paths)


@dataclass(kw_only=True)
class LlmCallJsonlRecorderPlugin(ConversationPlugin):
    """Append every completed LLM call's raw input and output to one JSONL file."""

    path: Path
    extra_handler: LlmCallJsonlRecorderExtraHandler | None = None
    target_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.extra_handler is not None and not isinstance(
            self.extra_handler,
            LlmCallJsonlRecorderExtraHandler,
        ):
            raise TypeError("extra_handler must inherit LlmCallJsonlRecorderExtraHandler")

    def on_registered(self, context: ConversationContext) -> None:
        """Resolve the log path and register the call recorder for this scope."""
        self.target_path = resolve_agent_file_path(context.cwd, self.path)
        context.after_llm_call_hooks.append(self._record_call)

    async def _record_call(
        self,
        context: ConversationContext,
        outcome: LlmCallOutcome,
    ) -> None:
        extra = dict(self.extra_handler(context, outcome)) if self.extra_handler else {}
        record = _LlmCallJsonlRecord(
            conversation_key=context.key,
            system=context.llm_call_system_prompt,
            user=context.llm_call_user_prompt,
            output=outcome.text,
            stats=_LlmCallStats(
                prompt_eval_count=int(outcome.prompt_eval_count),
                eval_count=int(outcome.eval_count),
            ),
            extra=extra,
        )
        self._append(record)

    def _append(self, record: _LlmCallJsonlRecord) -> None:
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        with self.target_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _token_speed(outcome: LlmCallOutcome, key: str) -> float | None:
    metadata = getattr(outcome, "llm", None)
    value = metadata.get(key) if isinstance(metadata, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _format_token_speed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.1f} token/s"


@dataclass
class LlmCallTokenStats:
    """Numeric token usage and speed values from one successful LLM call."""

    prompt_eval_count: int
    eval_count: int
    prompt_tokens_per_second: float | None
    eval_tokens_per_second: float | None


def format_llm_call_token_stats(stats: LlmCallTokenStats) -> str:
    """Format token statistics for compact command-line display."""
    prompt_speed = _format_token_speed(stats.prompt_tokens_per_second)
    eval_speed = _format_token_speed(stats.eval_tokens_per_second)
    return (
        f"LLM Token｜输入 {stats.prompt_eval_count:,}（{prompt_speed}）"
        f"｜输出 {stats.eval_count:,}（{eval_speed}）"
    )


@dataclass
class LlmCallTokenStatsPlugin(ConversationPlugin):
    """Print final token counts and speeds after a conversation succeeds."""

    handler: _LlmCallTokenStatsHandler | None = None

    def on_registered(self, context: ConversationContext) -> None:
        """Register a compact success-only footer for this conversation."""

        async def print_call_stats(
            _context: ConversationContext,
            _result: ConversationResult,
        ) -> None:
            outcome = context.llm_call_outcome
            if outcome is None:
                raise RuntimeError("successful conversation has no LLM call outcome")
            stats = LlmCallTokenStats(
                prompt_eval_count=outcome.prompt_eval_count,
                eval_count=outcome.eval_count,
                prompt_tokens_per_second=_token_speed(outcome, "prompt_tokens_per_second"),
                eval_tokens_per_second=_token_speed(outcome, "eval_tokens_per_second"),
            )
            if self.handler is None:
                print(format_llm_call_token_stats(stats))
            else:
                await self.handler(context, stats)

        context.conversation_completed_hooks.append(print_call_stats)


@dataclass
class BeforeConversationPlugin(ConversationPlugin):
    """Prepare one conversation before prompting the model."""

    handler: _BeforeConversationCallback

    def on_registered(self, context: ConversationContext) -> None:
        context.before_conversation_hooks.append(self.handler)


@dataclass
class AfterConversationPlugin(ConversationPlugin):
    """Inspect one parsed and handled conversation inside its retry loop."""

    handler: _AfterConversationCallback

    def on_registered(self, context: ConversationContext) -> None:
        context.after_conversation_hooks.append(self.handler)


@dataclass
class ConditionalConversationPlugin(ConversationPlugin):
    """Run this conversation only when its runtime condition is true."""

    should_run: _ConversationCondition

    def on_registered(self, context: ConversationContext) -> None:
        if context.should_run is not None:
            raise RuntimeError("conversation condition is already configured")
        context.should_run = self.should_run


class DynamicOutputPathsPlugin(ConversationPlugin):
    """Use the currently registered input paths as writable output paths."""

    def on_registered(self, context: ConversationContext) -> None:
        """Copy input paths once the core agent context is available."""
        context.output_paths = list(context.input_paths)


def _resolve_agent_file_paths(
    context: ConversationContext,
    paths: list[Path],
) -> list[Path]:
    return [resolve_agent_file_path(context.cwd, path) for path in paths]


class ExistingPathGuardPlugin(ConversationPlugin):
    """Abort a run before any configured output path is overwritten."""

    def on_registered(self, context: ConversationContext) -> None:
        """Disable replacing existing output files for this agent."""
        context.allow_overwrite_existing_paths = False


@dataclass
class FileCleanupPlugin(ConversationPlugin):
    """Register workspace files that the core runtime removes before each run."""

    paths: list[Path]

    def on_registered(self, context: ConversationContext) -> None:
        """Resolve and append cleanup paths to the shared runtime context."""
        context.clean_up_paths.extend(_resolve_agent_file_paths(context, self.paths))


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(resolve_agent_root(root)).as_posix()
    except ValueError:
        return str(path)


@dataclass
class StaticSystemPromptPlugin(ConversationPlugin):
    """Replace the system prompt with static text or context-derived text."""

    prompt: _SystemPromptProvider

    def on_registered(self, context: ConversationContext) -> None:
        """Register this plugin as a system prompt hook."""
        context.system_prompt_hooks.append(self.build_system_prompt)

    def build_system_prompt(
        self,
        context: ConversationContext,
    ) -> str:
        """Return the configured system prompt for this run."""
        prompt = self.prompt(context) if callable(self.prompt) else self.prompt
        return prompt.rstrip() + "\n"


@dataclass
class StaticUserPromptPlugin(ConversationPlugin):
    """Build a user prompt from static text or the current conversation context."""

    prompt: _UserPromptProvider

    def on_registered(self, context: ConversationContext) -> None:
        context.user_prompt_hooks.append(self.build_user_prompt)

    def build_user_prompt(
        self,
        context: ConversationContext,
        retry_errors: Sequence[str],
    ) -> str:
        prompt = self.prompt(context, retry_errors) if callable(self.prompt) else self.prompt
        return prompt.rstrip() + "\n"


@dataclass
class TemplateSystemPromptPlugin(ConversationPlugin):
    """Render a template-backed system prompt using the registered agent context."""

    template_name: str
    template_root: Path = field(kw_only=True)
    template_vars: _TemplateVars | None = field(default=None, kw_only=True)

    def on_registered(self, context: ConversationContext) -> None:
        """Register this template as a system prompt hook."""
        context.system_prompt_hooks.append(self.build_system_prompt)

    def build_system_prompt(
        self,
        context: ConversationContext,
    ) -> str:
        """Replace the current system prompt with the rendered template contract."""
        variables = self._template_vars(context)
        return (
            _jinja_env(self.template_root).get_template(self.template_name).render(**variables)
        ).rstrip() + "\n"

    def _template_vars(self, context: ConversationContext) -> Mapping[str, Any]:
        if self.template_vars is None:
            return {}
        if callable(self.template_vars):
            return self.template_vars(context)
        return self.template_vars


@dataclass
class TemplateUserPromptPlugin(ConversationPlugin):
    """Render a template-backed user prompt using the registered agent context."""

    template_name: str
    template_root: Path = field(kw_only=True)
    template_vars: UserTemplateVars | None = field(default=None, kw_only=True)

    def on_registered(self, context: ConversationContext) -> None:
        """Register this template as a user prompt hook."""
        context.user_prompt_hooks.append(self.build_user_prompt)

    def build_user_prompt(
        self,
        context: ConversationContext,
        retry_errors: Sequence[str],
    ) -> str:
        """Replace the current user prompt with the rendered template."""
        from tiny_coder.file_agent import agent_input_snapshots

        custom_vars = (
            self.template_vars(context)
            if callable(self.template_vars)
            else dict(self.template_vars or {})
        )
        variables = {
            **custom_vars,
            "input_files": agent_input_snapshots(context.cwd, context.input_paths),
            "retry_errors": list(retry_errors),
        }
        rendered = (
            _jinja_env(self.template_root).get_template(self.template_name).render(**variables)
        )
        return rendered.rstrip() + "\n"


def _jinja_env(template_root: Path) -> Environment:
    root = template_root.expanduser().resolve()
    return Environment(
        loader=FileSystemLoader(str(root), encoding="utf-8"),
        undefined=StrictUndefined,
        autoescape=False,
    )


@dataclass
class FileTreeInputPathsPlugin(ConversationPlugin):
    """Provide readable input paths from a directory tree."""

    root: Path
    first_paths: list[Path] = field(default_factory=list, kw_only=True)
    patterns: list[str] = field(default_factory=lambda: ["**/*"], kw_only=True)

    def on_registered(self, context: ConversationContext) -> None:
        """Resolve configured tree paths once the core agent context is available."""
        root = resolve_agent_file_path(context.cwd, self.root)
        paths = [
            resolve_agent_file_path(context.cwd, path if path.is_absolute() else root / path)
            for path in self.first_paths
        ]
        for pattern in self.patterns:
            paths.extend(
                resolve_agent_file_path(context.cwd, path)
                for path in sorted(root.glob(pattern))
                if path.is_file()
            )
        context.input_paths = list(dict.fromkeys(paths))


def require_output_model_type(output_type: Any) -> type[BaseModel]:
    """Validate and return a BaseModel subclass."""
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise TypeError("output_type must be a BaseModel subclass")
    return output_type


@dataclass
class ResponseOutputTypePlugin(ConversationPlugin):
    """Provide the Pydantic response model used for JSON Schema and parsing."""

    output_type: type[BaseModel]

    def __post_init__(self) -> None:
        self.output_type = require_output_model_type(self.output_type)

    def on_registered(self, context: ConversationContext) -> None:
        """Register the response model on the shared agent context."""
        context.llm_response_output_type = self.output_type


@dataclass
class LlmConfigPlugin(ConversationPlugin):
    """Provide the model configuration for one conversation."""

    llm_config: BaseModel

    def on_registered(self, context: ConversationContext) -> None:
        context.llm_config = self.llm_config


class SilentLlmCallPlugin(ConversationPlugin):
    """Suppress streamed chunks while preserving the configured LLM call."""

    def on_registered(self, context: ConversationContext) -> None:
        base_llm_call = context.llm_call

        async def silent_llm_call(
            *,
            llm_cfg: BaseModel,
            system: str,
            prompt: str,
            response_format: dict[str, Any],
            on_chunk: Callable[[str], None],
        ) -> Any:
            _ = on_chunk
            return await base_llm_call(
                llm_cfg=llm_cfg,
                system=system,
                prompt=prompt,
                response_format=response_format,
                on_chunk=lambda _chunk: None,
            )

        context.llm_call = silent_llm_call


@dataclass
class ConversationRetryPlugin(ConversationPlugin):
    """Configure the attempt limit for one conversation."""

    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def on_registered(self, context: ConversationContext) -> None:
        context.max_attempts = self.max_attempts


@dataclass(kw_only=True)
class JsonFieldStreamLlmCallPlugin(ConversationPlugin):
    """Call the configured LLM while printing one JSON string field."""

    field_name: str
    raw_abort_threshold: int

    def on_registered(self, context: ConversationContext) -> None:
        """Wrap the shared LLM call to stream only the configured JSON field."""
        base_llm_call = context.llm_call

        async def stream_field_llm_call(
            *,
            llm_cfg: BaseModel,
            system: str,
            prompt: str,
            response_format: dict[str, Any],
            on_chunk: Callable[[str], None],
        ) -> Any:
            _ = on_chunk
            return await self._call_and_stream_field(
                base_llm_call,
                llm_cfg=llm_cfg,
                system=system,
                prompt=prompt,
                response_format=response_format,
            )

        context.llm_call = stream_field_llm_call

    async def _call_and_stream_field(
        self,
        llm_call: LlmCall,
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
    ) -> Any:
        """Call the model while printing only the configured JSON string field."""
        streamer = JsonStringFieldStreamer(self.field_name)
        raw_len = 0

        def on_chunk(chunk: str) -> None:
            nonlocal raw_len
            raw_len += len(chunk)
            streamed = streamer.feed(chunk)
            if streamed:
                print(streamed, end="", flush=True)
            if raw_len > self.raw_abort_threshold:
                raise ValueError(
                    f"stream aborted at {raw_len} characters (limit {self.raw_abort_threshold})"
                )

        outcome = await llm_call(
            llm_cfg=llm_cfg,
            system=system,
            prompt=prompt,
            response_format=response_format,
            on_chunk=on_chunk,
        )
        if streamer.streamed_text:
            print()
        return outcome


class ApplyPatchWriterPlugin(ConversationPlugin):
    """Apply an output model's generic patches to workspace text files."""

    def on_registered(self, context: ConversationContext) -> None:
        """Register this writer after validating the configured response model."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError(
                "ApplyPatchWriterPlugin requires ResponseOutputTypePlugin to be registered first"
            )
        if not issubclass(output_type, ApplyPatchOutput):
            raise TypeError("response output type must inherit ApplyPatchOutput")
        context.output_writer = self.write_output

    def write_output(
        self,
        context: ConversationContext,
        output: BaseModel,
    ) -> list[Path]:
        """Convert the validated output to patches and apply them."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError("ApplyPatchWriterPlugin is not registered")
        if type(output) is not output_type or not isinstance(output, ApplyPatchOutput):
            raise ValueError(
                "structured output type mismatch: "
                f"expected {output_type.__qualname__} inheriting ApplyPatchOutput, "
                f"got {type(output).__qualname__}"
            )
        return apply_patches(
            context.cwd,
            output.to_apply_patches(context),
            allowed_paths=context.output_paths,
        )


@dataclass(kw_only=True)
class ExecutableScriptPlugin(ConversationPlugin):
    """Normalize, validate, confirm, and execute structured script content."""

    command: list[str] = field(default_factory=lambda: [sys.executable, "-"])
    arguments: list[str] = field(default_factory=list)
    execution_confirmation: _ExecutionConfirmation | None = None
    environment: Mapping[str, str] | None = field(default=None, repr=False)
    capture_output: bool = True

    def __post_init__(self) -> None:
        if not self.command or any(not part for part in self.command):
            raise ValueError("script command must contain non-blank arguments")

    def on_registered(self, context: ConversationContext) -> None:
        """Register content validation followed by script execution."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError(
                "ExecutableScriptPlugin requires ResponseOutputTypePlugin to be registered first"
            )
        if not issubclass(output_type, ExecutableScriptOutput):
            raise TypeError("response output type must inherit ExecutableScriptOutput")
        if context.output_writer is not None:
            raise RuntimeError("ExecutableScriptPlugin requires an unconfigured output writer")

        context.output_writer = self.validate_output
        context.after_conversation_hooks.append(self.process_output)

    def validate_output(
        self,
        context: ConversationContext,
        output: BaseModel,
    ) -> list[Path]:
        """Validate the structured script output without writing a workspace file."""
        self._validated_script(context, output)
        return []

    def _validated_output(
        self,
        context: ConversationContext,
        output: BaseModel,
    ) -> ExecutableScriptOutput:
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError("ExecutableScriptPlugin is not registered")
        if type(output) is not output_type or not isinstance(output, ExecutableScriptOutput):
            raise ValueError(
                "structured output type mismatch: "
                f"expected {output_type.__qualname__} inheriting ExecutableScriptOutput, "
                f"got {type(output).__qualname__}"
            )
        return output

    def _validated_script(self, context: ConversationContext, output: BaseModel) -> str:
        executable_output = self._validated_output(context, output)
        script = executable_output.to_executable_script(context)
        if not isinstance(script, str):
            raise ValueError("executable script output must return str content")
        script = _normalize_newlines(script)
        if not script.strip():
            raise ValueError("generated script must not be blank")
        return script if script.endswith("\n") else script + "\n"

    async def process_output(
        self,
        context: ConversationContext,
        result: ConversationResult,
    ) -> None:
        """Run asynchronous validation and optionally execute the prepared script."""
        _ = result
        output = context.llm_response_output
        if output is None:
            raise RuntimeError("generated script processing requires validated output")
        executable_output = self._validated_output(context, output)
        await executable_output.prepare_executable_script(context)
        script = self._validated_script(context, output)
        if self.execution_confirmation is not None and not await self.execution_confirmation(
            context,
            output,
        ):
            return
        command = (*self.command, *self.arguments)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=context.cwd,
            env=self.environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE if self.capture_output else None,
            stderr=asyncio.subprocess.PIPE if self.capture_output else None,
        )
        stdout_bytes, stderr_bytes = await process.communicate(script.encode("utf-8"))

        stdout_bytes = stdout_bytes or b""
        stderr_bytes = stderr_bytes or b""

        returncode = process.returncode
        if returncode is None:
            raise RuntimeError("generated script process did not terminate")
        execution = ScriptExecutionResult(
            command=command,
            returncode=returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )
        context.script_execution_result = execution
        result.script_execution = execution
        if execution.returncode != 0:
            raise ValueError(
                f"generated script failed with exit code {execution.returncode}"
                + _script_output_details(execution)
            )


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _script_output_details(result: ScriptExecutionResult) -> str:
    details = []
    if result.stdout:
        details.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        details.append(f"stderr:\n{result.stderr.rstrip()}")
    return "" if not details else "\n" + "\n".join(details)


class NoopOutputWriterPlugin(ConversationPlugin):
    """Accept a validated response without writing workspace files."""

    def on_registered(self, context: ConversationContext) -> None:
        context.output_writer = self.write_output

    def write_output(
        self,
        context: ConversationContext,
        output: BaseModel,
    ) -> list[Path]:
        _ = context, output
        return []


__all__ = [
    "AfterConversationPlugin",
    "ApplyPatchWriterPlugin",
    "BeforeConversationPlugin",
    "ConditionalConversationPlugin",
    "ConversationPlugin",
    "ConversationRetryPlugin",
    "DynamicOutputPathsPlugin",
    "ExistingPathGuardPlugin",
    "ExecutableScriptPlugin",
    "FileCleanupPlugin",
    "FileTreeInputPathsPlugin",
    "JsonFieldStreamLlmCallPlugin",
    "LlmConfigPlugin",
    "LlmCallJsonlRecorderExtraHandler",
    "LlmCallJsonlRecorderPlugin",
    "LlmCallTokenStats",
    "LlmCallTokenStatsPlugin",
    "NoopOutputWriterPlugin",
    "ResponseOutputTypePlugin",
    "SilentLlmCallPlugin",
    "StaticInputPathsPlugin",
    "StaticOutputPathsPlugin",
    "StaticSystemPromptPlugin",
    "StaticUserPromptPlugin",
    "TemplateSystemPromptPlugin",
    "TemplateUserPromptPlugin",
    "UserTemplateVars",
    "format_llm_call_token_stats",
    "require_output_model_type",
    "resolve_agent_file_path",
    "resolve_agent_root",
]
