from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from tiny_coder.json_utils import JsonStringFieldStreamer
from tiny_coder.text_replacement import (
    TextReplacement,
    TextReplacementFilePatch,
    apply_text_replacements,
)

if TYPE_CHECKING:
    from tiny_coder.file_agent import (
        AgentContext,
        LlmCallOutcome,
        SyncAgentResult,
        _LlmCall,
    )


def _agent_root(root: Path) -> Path:
    return Path(root).expanduser().resolve()


def resolve_agent_file_path(root: Path, path: Path) -> Path:
    """Resolve an agent-visible file path and keep it inside the workspace root."""
    root_path = _agent_root(root)
    raw = path.expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as e:
        raise ValueError(f"path is outside agent workspace: {path}") from e
    return candidate


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
_SessionTurnT = TypeVar("_SessionTurnT", bound=Mapping[str, Any])


class _FileAgentPlugin(Protocol):
    def on_registered(self, context: AgentContext) -> None: ...


@runtime_checkable
class _LabeledFileMapOutput(Protocol):
    """Structured output that maps agent-visible paths to custom file content."""

    def to_file_map(self, context: AgentContext) -> Mapping[Path, str]: ...


@runtime_checkable
class _TextReplacementFileOutput(Protocol):
    """Structured output that provides replacements and an optional report."""

    def to_text_replacement_file_patch(self) -> TextReplacementFilePatch: ...


@dataclass
class StaticInputPathsPlugin:
    """Provide a fixed set of readable input paths for the core agent lifecycle."""

    paths: list[Path]

    def on_registered(self, context: AgentContext) -> None:
        """Resolve configured paths once the core agent context is available."""
        context.input_paths = _resolve_agent_file_paths(context, self.paths)


@dataclass
class StaticOutputPathsPlugin:
    """Provide a fixed set of writable output paths for the core agent lifecycle."""

    paths: list[Path]

    def on_registered(self, context: AgentContext) -> None:
        """Resolve configured paths once the core agent context is available."""
        context.output_paths = _resolve_agent_file_paths(context, self.paths)


@dataclass(kw_only=True)
class LlmSessionTurnPlugin(Generic[_SessionTurnT]):
    """Append a caller-defined session turn after each successful LLM call."""

    path: Path
    turn_factory: Callable[[AgentContext, LlmCallOutcome], _SessionTurnT]
    target_path: Path = field(init=False)

    def on_registered(self, context: AgentContext) -> None:
        """Resolve the session path and register the turn recorder."""
        self.target_path = resolve_agent_file_path(context.cwd, self.path)
        context.after_llm_call_hooks.append(self._record_turn)

    async def _record_turn(self, context: AgentContext, outcome: LlmCallOutcome) -> None:
        self._append(self.turn_factory(context, outcome))

    def _append(self, turn: _SessionTurnT) -> None:
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        with self.target_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(dict(turn), ensure_ascii=False) + "\n")


@dataclass
class BeforeRunPlugin:
    """Register an asynchronous callback that prepares one agent run."""

    handler: _BeforeRunCallback

    def on_registered(self, context: AgentContext) -> None:
        """Append the callback to the shared run-preparation hooks."""
        context.before_run_hooks.append(self.handler)


@dataclass
class LlmOutputResultPlugin:
    """Inspect one parsed and written LLM output inside the retry loop."""

    handler: _LlmOutputResultCallback

    def on_registered(self, context: AgentContext) -> None:
        """Bind the context and register the output-result hook."""

        async def handle_output_result(result: SyncAgentResult) -> None:
            await self.handler(context, result)

        context.llm_output_result_hooks.append(handle_output_result)


@dataclass
class AfterRunPlugin:
    """Run a callback after the retrying agent run succeeds."""

    handler: _AfterRunCallback

    def on_registered(self, context: AgentContext) -> None:
        """Bind the context and register the post-run hook."""

        async def after_run(result: SyncAgentResult) -> None:
            await self.handler(context, result)

        context.after_run_hooks.append(after_run)


@dataclass
class IterativeRunPlugin:
    """Repeat the retrying agent run for context-owned iteration items."""

    items: list[object]

    def on_registered(self, context: AgentContext) -> None:
        """Register one iteration source on the shared agent context."""
        if context.iteration_items is not None:
            raise RuntimeError("only one IterativeRunPlugin can be registered")
        context.iteration_items = list(self.items)


@dataclass
class BeforeIterationPlugin:
    """Run a callback before each context-owned iteration."""

    handler: _BeforeIterationCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the shared iteration lifecycle."""
        context.before_iteration_hooks.append(self.handler)


@dataclass
class AfterIterationPlugin:
    """Run a callback after each context-owned iteration succeeds."""

    handler: _AfterIterationCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the shared iteration lifecycle."""
        context.after_iteration_hooks.append(self.handler)


@dataclass
class LlmRequestGroupPlugin:
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
        context.llm_request_contexts[self.key] = request_context


@dataclass
class BeforeLlmRequestPlugin:
    """Prepare the active keyed LLM request before each call."""

    handler: _BeforeLlmRequestCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the active request context."""
        if context.llm_request_key is None:
            raise RuntimeError("BeforeLlmRequestPlugin requires a preceding LlmRequestGroupPlugin")
        context.before_llm_request_hooks.append(self.handler)


@dataclass
class AfterLlmRequestPlugin:
    """Inspect the active keyed LLM request after it succeeds."""

    handler: _AfterLlmRequestCallback

    def on_registered(self, context: AgentContext) -> None:
        """Register the callback on the active request context."""
        if context.llm_request_key is None:
            raise RuntimeError("AfterLlmRequestPlugin requires a preceding LlmRequestGroupPlugin")
        context.after_llm_request_hooks.append(self.handler)


def _validate_llm_request_registration(context: AgentContext) -> None:
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


class DynamicOutputPathsPlugin:
    """Use the currently registered input paths as writable output paths."""

    def on_registered(self, context: AgentContext) -> None:
        """Copy input paths once the core agent context is available."""
        context.output_paths = list(context.input_paths)


def _resolve_agent_file_paths(context: AgentContext, paths: list[Path]) -> list[Path]:
    return [resolve_agent_file_path(context.cwd, path) for path in paths]


class ExistingPathGuardPlugin:
    """Abort a run before any configured output path is overwritten."""

    def on_registered(self, context: AgentContext) -> None:
        """Disable replacing existing output files for this agent."""
        context.allow_overwrite_existing_paths = False


@dataclass
class FileCleanupPlugin:
    """Register workspace files that the core runtime removes before each run."""

    paths: list[Path]

    def on_registered(self, context: AgentContext) -> None:
        """Resolve and append cleanup paths to the shared runtime context."""
        context.clean_up_paths.extend(_resolve_agent_file_paths(context, self.paths))


_FileContentT = TypeVar("_FileContentT")


def _resolve_labeled_file_map(
    root: Path,
    files: Mapping[Path, _FileContentT],
    *,
    allowed_paths: list[Path] | None = None,
) -> dict[Path, _FileContentT]:
    """Resolve an output file map to paths within the agent workspace."""
    allowed = {resolve_agent_file_path(root, path) for path in allowed_paths or []}
    resolved: dict[Path, _FileContentT] = {}
    for label, value in files.items():
        path = resolve_agent_file_path(root, label)
        if allowed and path not in allowed:
            allowed_labels = ", ".join(
                _relative_label(root, allowed_path) for allowed_path in sorted(allowed)
            )
            raise ValueError(
                f"output file is not allowed: {_relative_label(root, path)}; "
                f"allowed: {allowed_labels}"
            )
        if path in resolved:
            raise ValueError(f"duplicate output file: {_relative_label(root, path)}")
        resolved[path] = value
    return resolved


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(_agent_root(root)).as_posix()
    except ValueError:
        return str(path)


@dataclass
class StaticSystemPromptPlugin:
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
class TemplateSystemPromptPlugin:
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
class TemplateUserPromptPlugin:
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
        from tiny_coder.file_agent import _agent_input_snapshots

        _ = task_prompt, current
        custom_vars = (
            self.template_vars(context)
            if callable(self.template_vars)
            else dict(self.template_vars or {})
        )
        variables = {
            **custom_vars,
            "input_files": _agent_input_snapshots(context.cwd, context.input_paths),
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
class FileTreeInputPathsPlugin:
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


def _require_output_model_type(output_type: Any) -> type[BaseModel]:
    """Validate and return a BaseModel subclass."""
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise TypeError("output_type must be a BaseModel subclass")
    return output_type


@dataclass
class ResponseOutputTypePlugin:
    """Provide the Pydantic response model used for JSON Schema and parsing."""

    output_type: type[BaseModel]

    def __post_init__(self) -> None:
        self.output_type = _require_output_model_type(self.output_type)

    def on_registered(self, context: AgentContext) -> None:
        """Register the response model on the shared agent context."""
        context.llm_response_output_type = self.output_type


@dataclass
class LlmConfigPlugin:
    """Provide the model configuration for the active LLM request group."""

    llm_config: BaseModel

    def on_registered(self, context: AgentContext) -> None:
        context.llm_config = self.llm_config


class SilentLlmCallPlugin:
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
class AgentRetryPolicyPlugin:
    """Configure the shared attempt limit for model calls and output handling."""

    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def on_registered(self, context: AgentContext) -> None:
        context.max_attempts = self.max_attempts


@dataclass(kw_only=True)
class JsonFieldStreamLlmCallPlugin:
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
        llm_call: _LlmCall,
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


class LabeledFileMapWriterPlugin:
    """Resolve and write a structured output's labeled text file map."""

    def on_registered(self, context: AgentContext) -> None:
        """Bind shared context and require a compatible registered response model."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError(
                "LabeledFileMapWriterPlugin requires ResponseOutputTypePlugin to be registered first"
            )
        if not issubclass(output_type, _LabeledFileMapOutput):
            raise TypeError("response output type must implement to_file_map()")
        context.output_writer = self.write_output

    def write_output(self, context: AgentContext, output: BaseModel) -> list[Path]:
        """Resolve the output-provided text map and write the files."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError("LabeledFileMapWriterPlugin is not registered")
        if not isinstance(output, output_type) or not isinstance(output, _LabeledFileMapOutput):
            raise ValueError(
                "structured output type mismatch: "
                f"expected {output_type.__qualname__} with to_file_map(), "
                f"got {type(output).__qualname__}"
            )
        files = _resolve_labeled_file_map(
            context.cwd,
            cast(Mapping[Path, str], output.to_file_map(context)),
            allowed_paths=context.output_paths,
        )
        return [_write_text_file(path, content) for path, content in files.items()]


class NoopOutputWriterPlugin:
    """Accept a validated response without writing workspace files."""

    def on_registered(self, context: AgentContext) -> None:
        context.output_writer = self.write_output

    def write_output(self, context: AgentContext, output: BaseModel) -> list[Path]:
        _ = context, output
        return []


@dataclass(kw_only=True)
class TextReplacementFileWriterPlugin:
    """Apply model-provided replacements to one text file and write an optional report."""

    target_path: Path
    report_path: Path | None = None

    def on_registered(self, context: AgentContext) -> None:
        """Register this writer after validating the configured response model."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError(
                "TextReplacementFileWriterPlugin requires ResponseOutputTypePlugin "
                "to be registered first"
            )
        if not issubclass(output_type, _TextReplacementFileOutput):
            raise TypeError("response output type must implement to_text_replacement_file_patch()")
        context.output_writer = self.write_output

    def write_output(self, context: AgentContext, output: BaseModel) -> list[Path]:
        """Patch the target text file and write the output-provided report."""
        output_type = context.llm_response_output_type
        if output_type is None:
            raise RuntimeError("TextReplacementFileWriterPlugin is not registered")
        if type(output) is not output_type or not isinstance(output, _TextReplacementFileOutput):
            raise ValueError(
                "structured output type mismatch: "
                f"expected {output_type.__qualname__} with "
                "to_text_replacement_file_patch(), "
                f"got {type(output).__qualname__}"
            )

        patch = output.to_text_replacement_file_patch()
        report_content = patch.report
        if self.report_path is not None and report_content is None:
            raise ValueError("text replacement output did not provide report content")
        replacements: list[TextReplacement] = []
        for item in patch.replacements:
            from_text = _normalize_newlines(item.from_text)
            to_text = _normalize_newlines(item.to_text)
            if from_text and from_text != to_text:
                replacements.append(TextReplacement(from_text=from_text, to_text=to_text))
        target = resolve_agent_file_path(context.cwd, self.target_path)
        if replacements:
            updated = apply_text_replacements(_read_text_file(target), replacements)
            _write_text_file(target, updated)

        written = [target]
        if self.report_path is not None:
            report_target = resolve_agent_file_path(context.cwd, self.report_path)
            written.append(_write_text_file(report_target, cast(str, report_content)))
        return written


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as f:
        return f.read()


def _write_text_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return path


__all__ = [
    "AgentRetryPolicyPlugin",
    "AfterIterationPlugin",
    "AfterLlmRequestPlugin",
    "AfterRunPlugin",
    "BeforeIterationPlugin",
    "BeforeLlmRequestPlugin",
    "BeforeRunPlugin",
    "DynamicOutputPathsPlugin",
    "ExistingPathGuardPlugin",
    "FileCleanupPlugin",
    "FileTreeInputPathsPlugin",
    "IterativeRunPlugin",
    "JsonFieldStreamLlmCallPlugin",
    "LabeledFileMapWriterPlugin",
    "LlmConfigPlugin",
    "LlmOutputResultPlugin",
    "LlmRequestGroupPlugin",
    "ResponseOutputTypePlugin",
    "resolve_agent_file_path",
    "LlmSessionTurnPlugin",
    "NoopOutputWriterPlugin",
    "SilentLlmCallPlugin",
    "StaticInputPathsPlugin",
    "StaticOutputPathsPlugin",
    "StaticSystemPromptPlugin",
    "TemplateSystemPromptPlugin",
    "TemplateUserPromptPlugin",
    "TextReplacementFileWriterPlugin",
    "UserTemplateVars",
]
