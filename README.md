# tiny-coder

`tiny-coder` is a small runtime for file-oriented LLM conversations. An agent run is an ordered
list of `Conversation` objects. Each conversation owns its plugins, runs once, and publishes one
validated `ConversationResult` for later conversations to consume.

```text
Conversation("draft") -> ConversationResult
                             |
                             v
Conversation("review") -> ConversationResult
                             |
                             v
                      AgentResult
```

There is one composition mechanism:

- `BasicFileAgent` runs conversations in declaration order.
- `Conversation` is one complete model interaction and output-handling contract.
- `ConversationPlugin` configures only the conversation that contains it.
- `AgentContext.conversation_results` integrates completed conversation context.

## One Conversation

```python
from pathlib import Path

from pydantic import BaseModel

from tiny_coder.apply_patch import ApplyPatch, ApplyPatchOutput
from tiny_coder.file_agent import BasicFileAgent, Conversation, ConversationContext
from tiny_coder.llm_format_stream import LlmConfig
from tiny_coder.plugins import (
    ApplyPatchWriterPlugin,
    ConversationRetryPlugin,
    LlmConfigPlugin,
    ResponseOutputTypePlugin,
    StaticInputPathsPlugin,
    StaticOutputPathsPlugin,
    StaticSystemPromptPlugin,
    StaticUserPromptPlugin,
)
from tiny_coder.text_replacement import TextReplacement


class LocalLlmConfig(LlmConfig):
    base_url: str = "http://localhost:11434/v1"
    llm_model: str = "local-model"
    num_ctx: int = 8192
    temperature: float = 0.1
    repeat_penalty: float = 1.1
    think: bool = False
    timeout: float = 600.0
    max_output_tokens: int | None = 8192


class FileOutput(BaseModel, ApplyPatchOutput):
    summary: str
    path: str
    content: str

    def to_apply_patches(self, context: ConversationContext) -> list[ApplyPatch]:
        _ = context
        return [
            ApplyPatch(
                path=Path(self.path),
                replacements=[TextReplacement(to_text=self.content)],
            )
        ]


agent = BasicFileAgent(
    cwd=Path.cwd(),
    conversations=[
        Conversation(
            key="sync",
            plugins=[
                LlmConfigPlugin(LocalLlmConfig()),
                ConversationRetryPlugin(3),
                StaticInputPathsPlugin([Path("input.md")]),
                StaticOutputPathsPlugin([Path("output.md")]),
                StaticSystemPromptPlugin("Synchronize the output file."),
                StaticUserPromptPlugin("Read the input and return the complete output."),
                ResponseOutputTypePlugin(FileOutput),
                ApplyPatchWriterPlugin(),
            ],
        )
    ],
)

result = await agent.run()
print(result.conversations["sync"].summary)
```

Every conversation explicitly declares its model, prompts, response model, and output handler.
Construction fails early when one of those required parts is missing.

## Multiple Conversations

Later conversations read earlier validated outputs through `context.previous`. Failed or skipped
conversations never appear there.

```python
from collections.abc import Sequence
from typing import cast

from tiny_coder.file_agent import ConversationContext
from tiny_coder.plugins import NoopOutputWriterPlugin, UserPromptProvider


class PlanOutput(BaseModel):
    summary: str
    plan: list[str]


class ReviewOutput(BaseModel):
    summary: str
    approved: bool


class ReviewPromptProvider(UserPromptProvider):
    def __call__(
        self,
        context: ConversationContext,
        retry_errors: Sequence[str],
        /,
    ) -> str:
        plan = cast(PlanOutput, context.previous["plan"].output)
        retry_note = "\n".join(retry_errors)
        return f"Review this plan: {plan.plan}\nPrevious errors: {retry_note}"


conversations = [
    Conversation(
        key="plan",
        plugins=[
            LlmConfigPlugin(LocalLlmConfig()),
            StaticSystemPromptPlugin("Create a concise implementation plan."),
            StaticUserPromptPlugin("Plan the requested change."),
            ResponseOutputTypePlugin(PlanOutput),
            NoopOutputWriterPlugin(),
        ],
    ),
    Conversation(
        key="review",
        plugins=[
            LlmConfigPlugin(LocalLlmConfig()),
            StaticSystemPromptPlugin("Review the proposed plan."),
            StaticUserPromptPlugin(ReviewPromptProvider()),
            ResponseOutputTypePlugin(ReviewOutput),
            NoopOutputWriterPlugin(),
        ],
    ),
]
```

Use `ConditionalConversationPlugin` for a bounded optional turn. Its `ConversationConditionHandler`
subclass receives the same `ConversationContext` and can inspect `context.previous` and
`context.extras`.

For repeated or open-ended work, pass an `Iterable[Conversation]` or
`AsyncIterable[Conversation]`. `BasicFileAgent` requests the next conversation only after the
current one completes, so even a very large sequence is not materialized up front. Async sources
can also prepare the next conversation from files or results produced by the preceding one:

```python
from collections.abc import Iterator
from itertools import count


def generated_conversations() -> Iterator[Conversation]:
    for turn in count(1):
        conversation = build_conversation(key=f"turn-{turn}")
        yield conversation
        if should_stop(conversation.context):
            return


agent = BasicFileAgent(cwd=Path.cwd(), conversations=generated_conversations())
result = await agent.run()
```

An unbounded synchronous or asynchronous generator keeps the agent running until it returns,
raises, or the task is cancelled. One-shot iterators support one `run()`; use a re-iterable source
when the same agent must run again. Every conversation is registered only when it is consumed.
Completed results remain available through `context.previous`, so generation, retries, and
cross-turn context stay on the same conversation mechanism.

## Conversation Plugins

Common plugins include:

- model and retry: `LlmConfigPlugin`, `ConversationRetryPlugin`;
- prompts: `StaticSystemPromptPlugin`, `StaticUserPromptPlugin`, and Jinja template variants;
- files: static/tree input paths, static/dynamic output paths, cleanup, and overwrite protection;
- output: `ApplyPatchWriterPlugin`, `NoopOutputWriterPlugin`, and `ExecutableScriptPlugin`;
- lifecycle: `BeforeConversationPlugin` and `AfterConversationPlugin`;
- presentation and audit: token statistics, JSONL call recording, silent calls, and JSON-field
  streaming.

`BeforeConversationPlugin` runs once before the conversation's retry loop.
`AfterConversationPlugin` runs after parsed output handling and participates in retries when it
raises `ValueError` or `OSError`. A retry replays only the current conversation.

## Generated Scripts

`ExecutableScriptPlugin` validates structured script content, optionally asks for confirmation, and
passes the script to an interpreter through standard input without creating a script file. The
execution result is available as `result.conversations[key].script_execution`.

Generated code runs with the current process user's permissions and inherited environment. Treat
this plugin as an explicit trusted-code boundary and make retryable scripts idempotent.

## Development

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tiny_coder tests
uv run pre-commit run --all-files
```

Use `pathlib.Path` for filesystem APIs. Text reads and writes must explicitly use
`encoding="utf-8"` and `newline="\n"`.
