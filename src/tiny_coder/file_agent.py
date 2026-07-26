from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict, TypeVar, cast

from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError

from tiny_coder.json_utils import JsonProtocolError, load_json_object
from tiny_coder.llm_format_stream import stream_llm_chat_format
from tiny_coder.plugins import (
    _agent_root,
    _FileAgentPlugin,
    _require_output_model_type,
)
from tiny_coder.plugins import (
    resolve_agent_file_path as _resolve_agent_file_path,
)
from tiny_coder.yaml_utils import dump_yaml_text

_OutputT = TypeVar("_OutputT", bound=BaseModel)
_SystemPromptCallback: TypeAlias = Callable[["AgentContext", str, str], str]
_UserPromptCallback: TypeAlias = Callable[..., str]
_OutputWriterCallback: TypeAlias = Callable[["AgentContext", BaseModel], list[Path]]

_CODE_FENCE_LANG_BY_SUFFIX = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonl": "jsonl",
    ".txt": "text",
}


class _SyncGraphState(TypedDict, total=False):
    system_prompt: str
    user_prompt: str
    raw_output: str
    prompt_eval_count: int
    eval_count: int
    summary: str
    written_paths: list[str]


@dataclass(frozen=True, slots=True)
class SyncAgentResult:
    """Result after the program validates structured LLM output and writes files."""

    summary: str
    written_paths: list[Path]


_BeforeRunCallback: TypeAlias = Callable[["AgentContext"], Awaitable[None]]
_LlmOutputResultHook: TypeAlias = Callable[[SyncAgentResult], Coroutine[Any, Any, None]]
_AfterRunHook: TypeAlias = Callable[[SyncAgentResult], Coroutine[Any, Any, None]]
_AfterLlmCallCallback: TypeAlias = Callable[
    ["AgentContext", "LlmCallOutcome"], Coroutine[Any, Any, None]
]


class _AgentInputSnapshot(TypedDict):
    label: str
    language: str
    content: str


def _default_llm_call() -> _LlmCall:
    return stream_llm_chat_format


@dataclass(slots=True)
class AgentContext:
    """Mutable runtime data shared with plugins through ``plugin.context``."""

    cwd: Path
    llm_config: BaseModel
    llm_call: _LlmCall = field(default_factory=_default_llm_call)
    max_attempts: int = 1
    llm_call_outcome: LlmCallOutcome | None = None
    llm_call_system_prompt: str = ""
    llm_call_user_prompt: str = ""
    input_paths: list[Path] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)
    before_run_hooks: list[_BeforeRunCallback] = field(default_factory=list)
    after_run_hooks: list[_AfterRunHook] = field(default_factory=list)
    system_prompt_hooks: list[_SystemPromptCallback] = field(default_factory=list)
    user_prompt_hooks: list[_UserPromptCallback] = field(default_factory=list)
    after_llm_call_hooks: list[_AfterLlmCallCallback] = field(default_factory=list)
    llm_response_output_type: type[BaseModel] | None = None
    llm_response_output: BaseModel | None = None
    output_writer: _OutputWriterCallback | None = None
    llm_output_result_hooks: list[_LlmOutputResultHook] = field(default_factory=list)
    allow_overwrite_existing_paths: bool = True
    clean_up_paths: list[Path] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


class LlmCallOutcome(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def prompt_eval_count(self) -> int: ...

    @property
    def eval_count(self) -> int: ...


class _LlmCall(Protocol):
    def __call__(
        self,
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        on_chunk: Callable[[str], None],
    ) -> Coroutine[Any, Any, LlmCallOutcome]: ...


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(_agent_root(root)).as_posix()
    except ValueError:
        return str(path)


def read_agent_file(root: Path, path: Path) -> str:
    """Read a UTF-8 text file from the agent workspace."""
    target = _resolve_agent_file_path(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"file not found: {_relative_label(root, target)}")
    with target.open("r", encoding="utf-8", newline="\n") as f:
        return f.read()


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as f:
        return f.read()


def _code_fence_language(path: Path) -> str:
    return _CODE_FENCE_LANG_BY_SUFFIX.get(path.suffix.lower(), "text")


def _agent_input_snapshots(root: Path, input_paths: list[Path]) -> list[_AgentInputSnapshot]:
    """Read agent input files and return template-ready snapshot objects."""
    return [
        {
            "label": _relative_label(root, path),
            "language": _code_fence_language(path),
            "content": _read_text_file(path).rstrip("\n"),
        }
        for path in input_paths
    ]


def _parse_structured_sync_output(raw_output: str, *, output_type: type[_OutputT]) -> _OutputT:
    """Parse the model JSON response into the plugin-selected output model."""
    try:
        payload = load_json_object(raw_output)
        model = _require_output_model_type(output_type)
        return cast(_OutputT, model.model_validate(payload))
    except (JsonProtocolError, ValidationError) as e:
        raise ValueError(f"invalid structured sync output: {e}") from e


def content_to_yaml_text(content: Any) -> str:
    """Serialize a validated content object to project YAML text."""
    data = _content_to_yaml_data(content)
    return dump_yaml_text(data, sort_keys=False).rstrip() + "\n"


def _content_to_yaml_data(content: Any) -> Any:
    """Convert Pydantic models nested in simple containers before YAML dumping."""
    if isinstance(content, BaseModel):
        return content.model_dump(mode="python", exclude_none=True)
    if isinstance(content, list):
        return [_content_to_yaml_data(item) for item in content]
    if isinstance(content, dict):
        return {key: _content_to_yaml_data(value) for key, value in content.items()}
    return content


def _print_llm_chunk(chunk: str) -> None:
    print(chunk, end="", flush=True)


@dataclass(kw_only=True)
class BasicFileAgent:
    """Core file-oriented LangGraph runner with plugin-only lifecycle hooks."""

    cwd: Path
    llm_config: BaseModel
    plugins: InitVar[Sequence[_FileAgentPlugin] | None] = None
    context: AgentContext = field(init=False)
    graph: Any = field(init=False, repr=False)

    def __post_init__(self, plugins: Sequence[_FileAgentPlugin] | None) -> None:
        self.cwd = _agent_root(self.cwd)
        self.context = AgentContext(
            cwd=self.cwd,
            llm_config=self.llm_config,
        )
        for plugin in plugins or []:
            plugin.on_registered(self.context)
        self.graph = self._build_graph()

    @property
    def input_paths(self) -> list[Path]:
        """Return the current readable input paths from shared agent context."""
        return self.context.input_paths

    def output_paths(self) -> list[Path]:
        """Return the files this sync run may write from registered plugins only."""
        return self.context.output_paths

    def response_output_type(self) -> type[BaseModel]:
        """Return the full structured output model from registered plugins only."""
        output_type = self.context.llm_response_output_type
        if output_type is None:
            raise NotImplementedError("llm_response_output_type must be provided by a plugin")
        return _require_output_model_type(output_type)

    def split_and_write_output(self, output: BaseModel) -> list[Path]:
        """Persist the validated output through registered plugins only."""
        if self.context.output_writer is None:
            raise NotImplementedError("output_writer must be provided by a plugin")
        return self.context.output_writer(self.context, output)

    def _guard_existing_output_paths(self) -> None:
        if self.context.allow_overwrite_existing_paths:
            return
        for path in self.output_paths():
            if path.exists():
                label = _relative_label(self.cwd, path)
                raise RuntimeError(f"refusing to overwrite existing path: {label}")

    def _clean_up_files(self) -> None:
        for path in self.context.clean_up_paths:
            if path.is_dir():
                label = _relative_label(self.cwd, path)
                raise RuntimeError(f"cleanup path is a directory: {label}")
            path.unlink(missing_ok=True)

    def build_system_prompt(self, task_prompt: str) -> str:
        """Build the system prompt from registered plugins."""
        if not self.context.system_prompt_hooks:
            raise RuntimeError("agent requires at least one system prompt hook")
        prompt = ""
        for hook in self.context.system_prompt_hooks:
            prompt = hook(self.context, task_prompt, prompt)
        return prompt

    def build_user_prompt(
        self,
        task_prompt: str,
        *,
        retry_errors: Sequence[str] | None = None,
    ) -> str:
        """Build the user prompt from registered plugins."""
        if not self.context.user_prompt_hooks:
            raise RuntimeError("agent requires at least one user prompt hook")
        prompt = ""
        for hook in self.context.user_prompt_hooks:
            prompt = hook(
                self.context,
                task_prompt,
                prompt,
                retry_errors=retry_errors,
            )
        return prompt

    def _build_graph(self) -> Any:
        graph = StateGraph(_SyncGraphState)
        graph.add_node("call_llm", self._call_llm)
        graph.add_node("apply_output", self._apply_output)
        graph.add_edge(START, "call_llm")
        graph.add_edge("call_llm", "apply_output")
        graph.add_edge("apply_output", END)
        return graph.compile()

    async def _call_llm(self, state: _SyncGraphState) -> _SyncGraphState:
        """Call LLM server with JSON Schema output; raises when the model call itself fails."""
        self.context.llm_call_system_prompt = state["system_prompt"]
        self.context.llm_call_user_prompt = state["user_prompt"]
        self.context.llm_call_outcome = None
        for attempt in range(1, self.context.max_attempts + 1):
            try:
                outcome = await self.context.llm_call(
                    llm_cfg=self.context.llm_config,
                    system=state["system_prompt"],
                    prompt=state["user_prompt"],
                    response_format=self.response_output_type().model_json_schema(),
                    on_chunk=_print_llm_chunk,
                )
                break
            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
                OSError,
            ) as error:
                if attempt == self.context.max_attempts:
                    raise RuntimeError(
                        f"LLM call failed after {self.context.max_attempts} attempt(s): {error}"
                    ) from error
        else:
            raise RuntimeError("LLM call retry loop unexpectedly exhausted")
        self.context.llm_call_outcome = outcome
        for hook in self.context.after_llm_call_hooks:
            await hook(self.context, outcome)
        if outcome.text:
            print()
        handled: _SyncGraphState = {
            "raw_output": outcome.text,
            "prompt_eval_count": outcome.prompt_eval_count,
            "eval_count": outcome.eval_count,
        }
        return handled

    def _apply_output(self, state: _SyncGraphState) -> _SyncGraphState:
        """Validate output and let plugins write it; raises on violations."""
        self.context.llm_response_output = None
        output_type = self.response_output_type()
        output = _parse_structured_sync_output(state["raw_output"], output_type=output_type)
        self.context.llm_response_output = output
        written = self.split_and_write_output(output)
        allowed = set(self.output_paths())
        actual = set(written)
        extra = actual - allowed
        if allowed and extra:
            labels = ", ".join(_relative_label(self.cwd, path) for path in sorted(extra))
            raise ValueError(f"sync writer returned unexpected paths: extra={labels}")
        summary = getattr(output, "summary", "")
        return {
            "summary": summary if isinstance(summary, str) else "",
            "written_paths": [_relative_label(self.cwd, target) for target in written],
        }

    async def run(self) -> SyncAgentResult:
        """Run this agent through the single no-argument entrypoint."""
        self._clean_up_files()
        for before_run_hook in self.context.before_run_hooks:
            await before_run_hook(self.context)
        self._guard_existing_output_paths()
        result = await self.call_llm_and_apply_output_with_retries()
        for after_run_hook in self.context.after_run_hooks:
            await after_run_hook(result)
        return result

    async def call_llm_and_apply_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> SyncAgentResult:
        """Call the LLM once, then validate and apply its structured output."""
        self._guard_existing_output_paths()
        return await self._call_llm_and_apply_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    async def _call_llm_and_apply_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> SyncAgentResult:
        final_state = cast(
            _SyncGraphState,
            await self.graph.ainvoke({"system_prompt": system_prompt, "user_prompt": user_prompt}),
        )
        written = [
            _resolve_agent_file_path(self.cwd, Path(path))
            for path in final_state.get("written_paths", [])
        ]
        return SyncAgentResult(summary=final_state.get("summary", ""), written_paths=written)

    async def call_llm_and_apply_output_with_retries(self) -> SyncAgentResult:
        """Call the LLM and apply output, retrying with prior validation errors."""
        max_attempts = self.context.max_attempts
        retry_errors: list[str] = []
        resolved_system_prompt = self.build_system_prompt("")
        for attempt in range(1, max_attempts + 1):
            user_prompt = self.build_user_prompt(
                "",
                retry_errors=retry_errors,
            )
            try:
                result = await self._call_llm_and_apply_output(
                    system_prompt=resolved_system_prompt,
                    user_prompt=user_prompt,
                )
                for output_result_hook in self.context.llm_output_result_hooks:
                    await output_result_hook(result)
            except (ValueError, OSError) as e:
                error_text = str(e)
                retry_errors.append(error_text)
                print(
                    self.format_attempt_error_message(
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=error_text,
                    )
                )
                if attempt >= max_attempts:
                    raise RuntimeError(
                        "LangGraph structured sync output failed "
                        f"after {max_attempts} attempt(s): {error_text}"
                    ) from e
                continue
            if result.summary:
                print(result.summary.strip())
            return result

        raise RuntimeError("LangGraph sync unexpectedly exhausted retries.")

    def format_attempt_error_message(
        self,
        *,
        attempt: int,
        max_attempts: int,
        error: str,
    ) -> str:
        """Return the user-visible retry message for one validation/write failure."""
        return f"\nStructured output validation failed (attempt {attempt}/{max_attempts}): {error}"


__all__ = [
    "BasicFileAgent",
    "AgentContext",
    "LlmCallOutcome",
    "SyncAgentResult",
    "content_to_yaml_text",
    "read_agent_file",
]
