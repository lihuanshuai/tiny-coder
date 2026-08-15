from __future__ import annotations

import json
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
from tiny_coder.json_utils import JsonStringFieldStreamer

if TYPE_CHECKING:
    from tiny_coder.file_agent import (
        AgentContext,
        LlmCall,
        LlmCallOutcome,
        SyncAgentResult,
    )


_SystemPromptProvider: TypeAlias = str | Callable[["AgentContext"], str]
_TemplateVars: TypeAlias = Mapping[str, Any] | Callable[["AgentContext", str], Mapping[str, Any]]
UserTemplateVars: TypeAlias = Mapping[str, Any] | Callable[["AgentContext"], Mapping[str, Any]]
_BeforeRunCallback: TypeAlias = Callable[["AgentContext"], Awaitable[None]]
_LlmOutputResultCallback: TypeAlias = Callable[["AgentContext", "SyncAgentResult"], Awaitable[None]]
_AfterRunCallback: TypeAlias = Callable[["AgentContext", "SyncAgentResult"], Awaitable[None]]
_BeforeIterationCallback: TypeAlias = Callable[["AgentContext", object], Awaitable[None]]
_AfterIterationCallback: TypeAlias = Callable[
    ["AgentContext", object, "SyncAgentResult"], Awaitable[None]
]
_BeforeLlmRequestCallback: TypeAlias = Callable[["AgentContext", str], Awaitable[None]]
_AfterLlmRequestCallback: TypeAlias = Callable[
    ["AgentContext", str, "SyncAgentResult"], Awaitable[None]
]
_LlmRequestCondition: TypeAlias = Callable[["AgentContext", str], bool]
_JsonScalar: TypeAlias = str | int | float | bool | None
_JsonValue: TypeAlias = _JsonScalar | list["_JsonValue"] | dict[str, "_JsonValue"]
_LlmCallExtraHandler: TypeAlias = Callable[
    ["AgentContext", "LlmCallOutcome"], Mapping[str, _JsonValue]
]


class _LlmCallStats(TypedDict):
    prompt_eval_count: int
    eval_count: int


class _LlmCallJsonlRecord(TypedDict):
    request_key: str | None
    system: str
    user: str
    output: str
    stats: _LlmCallStats
    extra: dict[str, _JsonValue]


class FileAgentPlugin(ABC):
    """Register one focused capability on an agent context."""

    @abstractmethod
    def on_registered(self, context: AgentContext) -> None: ...


@dataclass
class StaticInputPathsPlugin(FileAgentPlugin):
    """Provide a fixed set of readable input paths for the core agent lifecycle."""

    paths: list[Path]

    def on_registered(self, context: AgentContext) -> None:
        """Resolve configured paths once the core agent context is available."""
        context.input_paths = _resolve_agent_file_paths(context, self.paths)


@dataclass
class StaticOutputPathsPlugin(FileAgentPlugin):
    """Provide a fixed set of writable output paths for the core agent lifecycle."""

    paths: list[Path]

    def on_registered(self, context: AgentContext) -> None:
        """Resolve configured paths once the core agent context is available."""
        context.output_paths = _resolve_agent_file_paths(context, self.paths)


@dataclass(kw_only=True)
class LlmCallJsonlRecorderPlugin(FileAgentPlugin):
    """Append every completed LLM call's raw input and output to one JSONL file."""

    path: Path
    extra_handler: _LlmCallExtraHandler | None = None
    target_path: Path = field(init=False)

    def on_registered(self, context: AgentContext) -> None:
        """Resolve the log path and register the call recorder for this scope."""
        self.target_path = resolve_agent_file_path(context.cwd, self.path)
        context.after_llm_call_hooks.append(self._record_call)

    async def _record_call(self, context: AgentContext, outcome: LlmCallOutcome) -> None:
        extra = dict(self.extra_handler(context, outcome)) if self.extra_handler else {}
        record = _LlmCallJsonlRecord(
            request_key=context.llm_request_key,
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


@dataclass
class BeforeRunPlugin(FileAgentPlugin):
    """Register an asynchronous callback that prepares one agent run."""

    handler: _BeforeRunCallback

    def on_registered(self, context: AgentContext) -> None:
        """Append the callback to the shared run-preparation hooks."""
        context.before_run_hooks.append(self.handler)


@dataclass
class LlmOutputResultPlugin(FileAgentPlugin):
    """Inspect one parsed and written LLM output inside the retry loop."""

    handler: _LlmOutputResultCallback

    def on_registered(self, context: AgentContext) -> None:
        """Bind the context and register the output-result hook."""

        async def handle_output_result(result: SyncAgentResult) -> None:
            await self.handler(context, result)

        context.llm_output_result_hooks.append(handle_output_result)


@dataclass
class AfterRunPlugin(FileAgentPlugin):
    """Run a callback after the retrying agent run succeeds."""

    handler: _AfterRunCallback

    def on_registered(self, context: AgentContext) -> None:
        """Bind the context and register the post-run hook."""

        async def after_run(result: SyncAgentResult) -> None:
            await self.handler(context, result)

        context.after_run_hooks.append(after_run)


@dataclass
class IterativeRunPlugin(FileAgentPlugin):
    """Repeat the retrying agent run for context-owned iteration items."""

    items: list[object]

    def on_registered(self, context: AgentContext) -> None:
        """Register one iteration source on the shared agent context."""
        if context.iteration_items is not None:
            raise RuntimeError("only one IterativeRunPlugin can be registered")
        context.iteration_items = list(self.items)


@dataclass
class BeforeIterationPlugin(FileAgentPlugin):
    """Run a callback before each context-owned iteration."""

    handler: _BeforeIterationCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the shared iteration lifecycle."""
        context.before_iteration_hooks.append(self.handler)


@dataclass
class AfterIterationPlugin(FileAgentPlugin):
    """Run a callback after each context-owned iteration succeeds."""

    handler: _AfterIterationCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the shared iteration lifecycle."""
        context.after_iteration_hooks.append(self.handler)


@dataclass
class LlmRequestGroupPlugin(FileAgentPlugin):
    """Start one flat, keyed request configuration block in the agent plugin list."""

    key: str

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("LLM request key must not be blank")

    def on_registered(self, context: AgentContext) -> None:
        """Create and register this keyed request context on the agent context."""
        # Local import avoids the file_agent/plugins module cycle during import.
        from tiny_coder.file_agent import AgentContext

        if self.key in context.llm_request_contexts:
            raise ValueError(f"duplicate LLM request key: {self.key}")

        request_context = AgentContext(cwd=context.cwd, llm_config=context.llm_config)
        request_context.llm_request_key = self.key
        request_context.llm_call_outcomes = context.llm_call_outcomes
        request_context.llm_response_outputs = context.llm_response_outputs
        request_context.llm_request_results = context.llm_request_results
        request_context.extras = context.extras
        request_context.after_llm_call_hooks = list(context.after_llm_call_hooks)
        context.llm_request_contexts[self.key] = request_context


@dataclass
class ConditionalLlmRequestPlugin(FileAgentPlugin):
    """Run the active keyed request only when its runtime condition is true."""

    should_run: _LlmRequestCondition

    def on_registered(self, context: AgentContext) -> None:
        """Register one condition on the active request context."""
        if context.llm_request_key is None:
            raise RuntimeError(
                "ConditionalLlmRequestPlugin requires a preceding LlmRequestGroupPlugin"
            )
        if context.llm_request_condition is not None:
            raise RuntimeError("LLM request condition is already configured")
        context.llm_request_condition = self.should_run


@dataclass
class BeforeLlmRequestPlugin(FileAgentPlugin):
    """Prepare the active keyed LLM request before each call."""

    handler: _BeforeLlmRequestCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the active request context."""
        if context.llm_request_key is None:
            raise RuntimeError("BeforeLlmRequestPlugin requires a preceding LlmRequestGroupPlugin")
        context.before_llm_request_hooks.append(self.handler)


@dataclass
class AfterLlmRequestPlugin(FileAgentPlugin):
    """Inspect the active keyed LLM request after it succeeds."""

    handler: _AfterLlmRequestCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the active request context."""
        if context.llm_request_key is None:
            raise RuntimeError("AfterLlmRequestPlugin requires a preceding LlmRequestGroupPlugin")
        context.after_llm_request_hooks.append(self.handler)


def validate_llm_request_registration(context: AgentContext) -> None:
    """Validate one completed keyed request configuration."""
    if (
        context.iteration_items is not None
        or context.before_iteration_hooks
        or context.after_iteration_hooks
        or context.llm_request_contexts
    ):
        raise ValueError("LLM request plugins must not configure nested request or iteration runs")
    if len(context.before_llm_request_hooks) != len(context.after_llm_request_hooks):
        raise ValueError("BeforeLlmRequestPlugin and AfterLlmRequestPlugin must be paired")
    if context.llm_config is None:
        raise ValueError("LLM request requires LlmConfigPlugin")


class DynamicOutputPathsPlugin(FileAgentPlugin):
    """Use the currently registered input paths as writable output paths."""

    def on_registered(self, context: AgentContext) -> None:
        """Copy input paths once the core agent context is available."""
        context.output_paths = list(context.input_paths)


def _resolve_agent_file_paths(context: AgentContext, paths: list[Path]) -> list[Path]:
    return [resolve_agent_file_path(context.cwd, path) for path in paths]


class ExistingPathGuardPlugin(FileAgentPlugin):
    """Abort a run before any configured output path is overwritten."""

    def on_registered(self, context: AgentContext) -> None:
        """Disable replacing existing output files for this agent."""
        context.allow_overwrite_existing_paths = False


@dataclass
class FileCleanupPlugin(FileAgentPlugin):
    """Register workspace files that the core runtime removes before each run."""

    paths: list[Path]

    def on_registered(self, context: AgentContext) -> None:
        """Resolve and append cleanup paths to the shared runtime context."""
        context.clean_up_paths.extend(_resolve_agent_file_paths(context, self.paths))


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(resolve_agent_root(root)).as_posix()
    except ValueError:
        return str(path)


@dataclass
class StaticSystemPromptPlugin(FileAgentPlugin):
    """Replace the system prompt with static text or context-derived text."""

    prompt: _SystemPromptProvider

    def on_registered(self, context: AgentContext) -> None:
        """Register this plugin as a system prompt hook."""
        context.system_prompt_hooks.append(self.build_system_prompt)

    def build_system_prompt(
        self,
        context: AgentContext,
        task_prompt: str,
        current: str,
    ) -> str:
        """Return the configured system prompt for this run."""
        _ = task_prompt, current
        prompt = self.prompt(context) if callable(self.prompt) else self.prompt
        return prompt.rstrip() + "\n"


@dataclass
class TemplateSystemPromptPlugin(FileAgentPlugin):
    """Render a template-backed system prompt using the registered agent context."""

    template_name: str
    template_root: Path = field(kw_only=True)
    template_vars: _TemplateVars | None = field(default=None, kw_only=True)

    def on_registered(self, context: AgentContext) -> None:
        """Register this template as a system prompt hook."""
        context.system_prompt_hooks.append(self.build_system_prompt)

    def build_system_prompt(
        self,
        context: AgentContext,
        task_prompt: str,
        current: str,
    ) -> str:
        """Replace the current system prompt with the rendered template contract."""
        _ = current
        variables = self._template_vars(context, task_prompt)
        return (
            _jinja_env(self.template_root).get_template(self.template_name).render(**variables)
        ).rstrip() + "\n"

    def _template_vars(self, context: AgentContext, task_prompt: str) -> Mapping[str, Any]:
        if self.template_vars is None:
            return {}
        if callable(self.template_vars):
            return self.template_vars(context, task_prompt)
        return self.template_vars


@dataclass
class TemplateUserPromptPlugin(FileAgentPlugin):
    """Render a template-backed user prompt using the registered agent context."""

    template_name: str
    template_root: Path = field(kw_only=True)
    template_vars: UserTemplateVars | None = field(default=None, kw_only=True)

    def on_registered(self, context: AgentContext) -> None:
        """Register this template as a user prompt hook."""
        context.user_prompt_hooks.append(self.build_user_prompt)

    def build_user_prompt(
        self,
        context: AgentContext,
        task_prompt: str,
        current: str,
        *,
        retry_errors: Sequence[str] | None = None,
    ) -> str:
        """Replace the current user prompt with the rendered template."""
        from tiny_coder.file_agent import agent_input_snapshots

        _ = task_prompt, current
        custom_vars = (
            self.template_vars(context)
            if callable(self.template_vars)
            else dict(self.template_vars or {})
        )
        variables = {
            **custom_vars,
            "input_files": agent_input_snapshots(context.cwd, context.input_paths),
            "retry_errors": list(retry_errors or []),
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
class FileTreeInputPathsPlugin(FileAgentPlugin):
    """Provide readable input paths from a directory tree."""

    root: Path
    first_paths: list[Path] = field(default_factory=list, kw_only=True)
    patterns: list[str] = field(default_factory=lambda: ["**/*"], kw_only=True)

    def on_registered(self, context: AgentContext) -> None:
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
class ResponseOutputTypePlugin(FileAgentPlugin):
    """Provide the Pydantic response model used for JSON Schema and parsing."""

    output_type: type[BaseModel]

    def __post_init__(self) -> None:
        self.output_type = require_output_model_type(self.output_type)

    def on_registered(self, context: AgentContext) -> None:
        """Register the response model on the shared agent context."""
        context.llm_response_output_type = self.output_type


@dataclass
class LlmConfigPlugin(FileAgentPlugin):
    """Provide the model configuration for the active LLM request group."""

    llm_config: BaseModel

    def on_registered(self, context: AgentContext) -> None:
        context.llm_config = self.llm_config


class SilentLlmCallPlugin(FileAgentPlugin):
    """Suppress streamed chunks while preserving the configured LLM call."""

    def on_registered(self, context: AgentContext) -> None:
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
class AgentRetryPolicyPlugin(FileAgentPlugin):
    """Configure the shared attempt limit for model calls and output handling."""

    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def on_registered(self, context: AgentContext) -> None:
        context.max_attempts = self.max_attempts


@dataclass(kw_only=True)
class JsonFieldStreamLlmCallPlugin(FileAgentPlugin):
    """Call the configured LLM while printing one JSON string field."""

    field_name: str
    raw_abort_threshold: int

    def on_registered(self, context: AgentContext) -> None:
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


class ApplyPatchWriterPlugin(FileAgentPlugin):
    """Apply an output model's generic patches to workspace text files."""

    def on_registered(self, context: AgentContext) -> None:
        """Register this writer after validating the configured response model."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError(
                "ApplyPatchWriterPlugin requires ResponseOutputTypePlugin to be registered first"
            )
        if not issubclass(output_type, ApplyPatchOutput):
            raise TypeError("response output type must inherit ApplyPatchOutput")
        context.output_writer = self.write_output

    def write_output(self, context: AgentContext, output: BaseModel) -> list[Path]:
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


class NoopOutputWriterPlugin(FileAgentPlugin):
    """Accept a validated response without writing workspace files."""

    def on_registered(self, context: AgentContext) -> None:
        context.output_writer = self.write_output

    def write_output(self, context: AgentContext, output: BaseModel) -> list[Path]:
        _ = context, output
        return []


__all__ = [
    "AgentRetryPolicyPlugin",
    "AfterIterationPlugin",
    "AfterLlmRequestPlugin",
    "AfterRunPlugin",
    "ApplyPatchWriterPlugin",
    "BeforeIterationPlugin",
    "BeforeLlmRequestPlugin",
    "BeforeRunPlugin",
    "ConditionalLlmRequestPlugin",
    "DynamicOutputPathsPlugin",
    "ExistingPathGuardPlugin",
    "FileAgentPlugin",
    "FileCleanupPlugin",
    "FileTreeInputPathsPlugin",
    "IterativeRunPlugin",
    "JsonFieldStreamLlmCallPlugin",
    "LlmConfigPlugin",
    "LlmCallJsonlRecorderPlugin",
    "LlmOutputResultPlugin",
    "LlmRequestGroupPlugin",
    "ResponseOutputTypePlugin",
    "require_output_model_type",
    "resolve_agent_file_path",
    "resolve_agent_root",
    "NoopOutputWriterPlugin",
    "SilentLlmCallPlugin",
    "StaticInputPathsPlugin",
    "StaticOutputPathsPlugin",
    "StaticSystemPromptPlugin",
    "TemplateSystemPromptPlugin",
    "TemplateUserPromptPlugin",
    "UserTemplateVars",
    "validate_llm_request_registration",
]
