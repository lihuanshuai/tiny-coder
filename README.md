# tiny-coder

`tiny-coder` is a small Python runtime for local LLM coding workflows. It provides a
file-oriented agent core that reads workspace files, asks an LLM for structured JSON, validates
the response with Pydantic, and lets plugins write the resulting files.

## Installation

```powershell
uv sync
uv run pre-commit install
```

The package requires Python 3.10 or newer.

## Basic Usage

`BasicFileAgent` uses the bundled OpenAI-compatible streaming client. Plugins configure prompts,
paths, output handling, and optional stream display behavior.

```python
from pathlib import Path

from pydantic import BaseModel

from tiny_coder.apply_patch import ApplyPatch, ApplyPatchOutput
from tiny_coder.file_agent import AgentContext, BasicFileAgent
from tiny_coder.plugins import (
    AgentRetryPolicyPlugin,
    ApplyPatchWriterPlugin,
    LlmConfigPlugin,
    LlmRequestGroupPlugin,
    ResponseOutputTypePlugin,
    StaticInputPathsPlugin,
    StaticOutputPathsPlugin,
)
from tiny_coder.text_replacement import TextReplacement


class FileOutput(BaseModel, ApplyPatchOutput):
    summary: str
    files: dict[str, str]

    def to_apply_patches(self, context: AgentContext) -> list[ApplyPatch]:
        _ = context
        return [
            ApplyPatch(
                path=Path(label),
                replacements=[TextReplacement(to_text=content)],
            )
            for label, content in self.files.items()
        ]


class ExampleLlmConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    llm_model: str = "local-model"
    num_ctx: int = 8192
    temperature: float = 0.2
    repeat_penalty: float = 1.1
    think: bool = False
    timeout: float = 600.0


agent = BasicFileAgent(
    cwd=Path.cwd(),
    plugins=[
        LlmConfigPlugin(ExampleLlmConfig()),
        LlmRequestGroupPlugin(key="sync"),
        StaticInputPathsPlugin([Path("input.md")]),
        StaticOutputPathsPlugin([Path("output.md")]),
        ResponseOutputTypePlugin(FileOutput),
        AgentRetryPolicyPlugin(3),
        ApplyPatchWriterPlugin(),
    ],
)
```

`ApplyPatch` is the shared file-mutation contract. A writable response model must inherit
`ApplyPatchOutput` and implement `to_apply_patches(context)`. Every patch contains one ordered
`replacements` list: `TextReplacement(to_text=...)` replaces the whole file, while a non-`None`
`from_text` replaces one matching text span.
`ApplyPatchWriterPlugin` resolves paths, checks configured output permissions, validates every
patch in memory, and writes only changed files.

Prefer importing from `tiny_coder.file_agent`, `tiny_coder.json_utils`, or
`tiny_coder.yaml_utils` instead of relying on package-level re-exports.

## Iterative Runs

`IterativeRunPlugin` stores the iteration list on `AgentContext`. Register preparation and
completion callbacks separately so each plugin has one lifecycle responsibility.

```python
from tiny_coder.file_agent import AgentContext, SyncAgentResult
from tiny_coder.plugins import (
    AfterIterationPlugin,
    BeforeIterationPlugin,
    IterativeRunPlugin,
)


async def prepare_iteration(context: AgentContext, item: object) -> None:
    context.extras["current_item"] = item


async def finish_iteration(
    context: AgentContext,
    item: object,
    result: SyncAgentResult,
) -> None:
    print(context.iteration_index, item, result.written_paths)


iteration_plugins = [
    IterativeRunPlugin(items=["first", "second"]),
    BeforeIterationPlugin(prepare_iteration),
    AfterIterationPlugin(finish_iteration),
]
```

## Serial LLM Requests

`LlmRequestGroupPlugin` starts a keyed request block in the flat agent plugin list. Following
plugins configure that request's templates, response model, paths, writer, retry policy, and
LLM-call behavior until the next `LlmRequestGroupPlugin`. Request blocks run in declaration order
inside each iteration. Every `BasicFileAgent` must declare at least one request block; construction
fails when `LlmRequestGroupPlugin` is omitted. Put agent-level lifecycle plugins before the first
request block. Configure the default model with `LlmConfigPlugin` before the first request block,
or register it inside a request block when that request needs an override.

Later requests can consume earlier parsed outputs from `context.llm_response_outputs`; call
outcomes and final request results are available from `context.llm_call_outcomes` and
`context.llm_request_results`. These keyed mappings are reset at the start of each iteration, so
iteration hooks see only the current iteration's requests.

```python
from tiny_coder.plugins import (
    ApplyPatchWriterPlugin,
    ConditionalLlmRequestPlugin,
    LlmCallJsonlRecorderPlugin,
    LlmConfigPlugin,
    LlmRequestGroupPlugin,
    ResponseOutputTypePlugin,
    TemplateSystemPromptPlugin,
    TemplateUserPromptPlugin,
)


template_root = Path(__file__).parent / "templates"


request_plugins = [
    LlmConfigPlugin(ExampleLlmConfig()),
    LlmCallJsonlRecorderPlugin(
        path=Path("logs/llm-calls.jsonl"),
        extra_handler=lambda context, _outcome: {
            "iteration_index": context.iteration_index,
        },
    ),
    LlmRequestGroupPlugin(key="plan"),
    ResponseOutputTypePlugin(FileOutput),
    TemplateSystemPromptPlugin(
        "plan-system.jinja",
        template_root=template_root,
    ),
    TemplateUserPromptPlugin(
        "plan-user.jinja",
        template_root=template_root,
    ),
    ApplyPatchWriterPlugin(),
    LlmRequestGroupPlugin(key="review"),
    ConditionalLlmRequestPlugin(
        should_run=lambda context, _key: "plan" in context.llm_response_outputs,
    ),
    ResponseOutputTypePlugin(FileOutput),
    TemplateSystemPromptPlugin(
        "review-system.jinja",
        template_root=template_root,
    ),
    TemplateUserPromptPlugin(
        "review-user.jinja",
        template_root=template_root,
        template_vars=lambda context: {
            "plan": context.llm_response_outputs["plan"],
        },
    ),
    ApplyPatchWriterPlugin(),
]
```

`BeforeLlmRequestPlugin` and `AfterLlmRequestPlugin` are optional. When used in a request block,
they must be paired and receive that request's context for dynamic per-iteration preparation and
result handling.

`ConditionalLlmRequestPlugin` evaluates its synchronous predicate immediately before its request
block would run. A false result skips the model call, cleanup, hooks, writer, and keyed result while
later request blocks continue normally. The predicate can inspect prior keyed results and shared
`context.extras`, so a fixed request sequence can express bounded conditional stages without
constructing another agent.

`LlmCallJsonlRecorderPlugin` records raw `request_key`, `system`, `user`, `output`, and token
`stats` for every completed LLM call. Register it before the first request group to cover every
keyed request in the agent, including calls whose structured output later fails validation and is
retried. `extra_handler` may add JSON-serializable caller metadata without changing the fixed call
record contract.

## Development

Useful local checks:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tiny_coder tests
uv run pre-commit run --all-files
```

When adding file I/O, use `pathlib.Path` and explicitly pass `encoding="utf-8"` and
`newline="\n"` for text reads and writes.
