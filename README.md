# tiny-coder

`tiny-coder` is a small graph runtime for local LLM coding agents. A run is a directed graph of
async node functions over a mutable `dict` state. The same engine provides three things:

- **Graph execution** — ordered nodes, direct edges, and conditional routing.
- **Tool calling** — `stream_llm_chat` speaks the OpenAI tools protocol, and `ToolNode` executes the
  assistant's tool calls against registered `Tool` objects.
- **Checkpointing** — after every node, `JsonCheckpointer` writes the full state so a run can resume
  exactly where it stopped.

```text
        ┌─────────────────────────────┐
        │           Graph             │
        │  llm ──tool_calls──> tools  │
        │   ▲                     │   │
        │   └─────────────────────┘   │
        └──────────────┬──────────────┘
                       │ after each node
                       v
                 Checkpointer
```

## Installation

```powershell
uv sync
```

## Quick Start

Define an LLM node, register a `ToolNode`, and route between them with a conditional edge.

```python
import asyncio
from pathlib import Path

from tiny_coder.checkpoint import JsonCheckpointer
from tiny_coder.file_tools import default_file_tools
from tiny_coder.graph import END, Graph
from tiny_coder.llm import LlmConfig, stream_llm_chat
from tiny_coder.state import State
from tiny_coder.tool_node import ToolNode, has_tool_calls


class LocalLlmConfig(LlmConfig):
    base_url: str = "http://localhost:11434/v1"
    llm_model: str = "local-model"
    num_ctx: int = 8192
    temperature: float = 0.1
    repeat_penalty: float = 1.1
    think: bool = False
    timeout: float = 600.0
    max_output_tokens: int = 8192


SYSTEM = "You are a coding agent. Use tools to inspect and change the workspace."

config = LocalLlmConfig()
tools = default_file_tools(Path.cwd())


async def llm_node(state: State) -> dict[str, object]:
    async def stream(chunk: str) -> None:
        print(chunk, end="", flush=True)

    outcome = await stream_llm_chat(
        llm_cfg=config,
        messages=[{"role": "system", "content": SYSTEM}, *state["messages"]},
        tools=tools,
        on_chunk=stream,
    )
    print()
    return {"messages": [outcome.message]}


graph = Graph()
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge("tools", "llm")
graph.add_conditional_edges(
    "llm",
    lambda state: "tools" if has_tool_calls(state) else END,
    {"tools": "tools", END: END},
)

checkpointer = JsonCheckpointer(Path(".tiny-coder/checkpoint.json"))
final = asyncio.run(
    graph.run(
        {"messages": [{"role": "user", "content": "Add a docstring to app.py."}]},
        checkpointer=checkpointer,
    )
)
```

The first node added becomes the entry point; use `graph.set_entry("llm")` to be explicit. The
default reducer appends new items for the `messages` key and replaces every other key.

## Tools

A `Tool` is a Pydantic argument model plus an async handler. The model's JSON schema is what the LLM
sees, and the handler receives the validated keyword arguments.

```python
from pydantic import BaseModel, Field

from tiny_coder.state import State
from tiny_coder.tool_node import Tool


class SearchArgs(BaseModel):
    """Search the web for one query."""

    query: str = Field(description="What to search for.")


async def search(state: State, query: str) -> str:
    return await my_search(query)


search_tool = Tool.from_model(SearchArgs, search)
```

`ToolNode(tools)` reads the last message, executes each `tool_calls` entry by name, and returns
`{"messages": [...]}` tool result messages. Unknown tools, invalid arguments, and handler exceptions
become `error: ...` tool results instead of aborting the run.

Built-in workspace tools live in `tiny_coder.file_tools`:

- `read_file`, `write_file`, `list_dir`, `run_command`.

`default_file_tools(root)` binds all four to a workspace root and blocks paths that escape it.

## Structured Agent (Factory)

`create_structured_agent` builds a reusable agent. Call `invoke` for each task:

```python
from pydantic import BaseModel

from tiny_coder.structured_agent import StructuredInput, create_structured_agent


class Plan(BaseModel):
    summary: str
    steps: list[str]


agent = create_structured_agent(
    llm_config=LocalLlmConfig(),
    response_model=Plan,
    system_prompt="You are a planning assistant.",
    max_steps=3,
)


async def main():
    result = await agent.invoke(StructuredInput(prompt="Plan the refactoring."))
    plan = result.model(Plan)
    if plan is not None:
        print(plan.summary)
    revision = await agent.invoke(
        StructuredInput(prompt="Revise the plan.", llm_config=LocalLlmConfig(temperature=0.5))
    )
    print(revision.text)


asyncio.run(main())
```

All agents share `invoke(input)` and `send(input)`. Implement `_invoke(input)` in custom
agents; the base class owns the public entrypoints and lifecycle checks. `StructuredAgent`
accepts `StructuredInput`, which holds the prompt, messages, optional LLM settings, and context.
Both entrypoints accept the same input object.

Each invocation starts fresh messages and attempt counters and reuses the same graph.
Omitted configuration, schema, and system prompt inherit the agent's defaults.
Overrides are local to that invocation. Register the execution budget (`max_steps`), tools,
callbacks, and extensions when creating the agent. Each invocation gets a
fresh step counter under that budget. Use `agent.spawn` and `child.send` to dispatch child tasks.

Tools can be combined with structured output. The agent routes `tool_calls` through a `ToolNode`,
then returns the final structured response:

```python
from tiny_coder.file_tools import default_file_tools

agent = create_structured_agent(
    llm_config=LocalLlmConfig(),
    response_model=Plan,
    tools=default_file_tools(Path.cwd()),
    max_steps=20,
)
```

When tools are given, `tools` and `response_format` are sent in the same request; not every
OpenAI-compatible server supports both together. If a server rejects the combination, use a
structured-output-only agent or a manual `Graph` where tool rounds and the final structured call
are separate nodes.

Register async subscribers with `agent.events.subscribe(event, handler)`. Typed event keys
`BEFORE_CALL`, `AFTER_CALL`, `CALL_FAILED`, and `CHUNK_RECEIVED` carry an `AgentCall`.
Subscribers run in registration order and are awaited. Before-call subscribers may update
`messages`, `system_prompt`, and `llm_config`. After-call subscribers inspect `outcome`;
failure subscribers receive `error` and any completed outcome. Set `call.retry` to request
another LLM call. Errors propagate unless a failure subscriber requests a retry.

```python
from tiny_coder.structured_agent import AFTER_CALL, CALL_FAILED, AgentCall


async def validate(call: AgentCall) -> None:
    if call.outcome is not None and not call.outcome.tool_calls:
        if call.outcome.model(Plan) is None:
            raise ValueError("Expected a JSON object.")


async def retry_validation(call: AgentCall) -> None:
    if isinstance(call.error, ValueError) and call.attempt < 3:
        call.messages.append({"role": "user", "content": str(call.error)})
        call.retry = True


agent.events.subscribe(AFTER_CALL, validate)
agent.events.subscribe(CALL_FAILED, retry_validation)
```

A before-call subscriber can supply messages for `agent.invoke(StructuredInput())` without a prompt.
Call contexts and attempt counters are local to each invocation. Custom values placed
in `call.state` must be JSON-compatible when checkpointing.

`max_steps` (default `100`) bounds both LLM calls and tool-node executions, including subscriber-requested
retries, and raises `GraphError` when exceeded.

`CALL_STARTED` runs before preparation. Register attempt resources with
`call.resources.enter_context(...)` or `call.resources.callback(...)`; they are released
before failure dispatch, including on cancellation. Subscribe to `EXCHANGE_RECEIVED`
for raw `LlmExchange` records. Stream and exchange callbacks on the LLM API are unchanged.

## Agent Communication

An `Agent` owns its children directly. `spawn(name, factory)` creates a child once per
name and returns the same instance on subsequent calls. Only the main agent may spawn;
children cannot spawn further agents, including through a main-agent reference while
processing a scheduled task.

`child.send(input)` schedules `child.invoke(input)` and returns an ordinary `asyncio.Task`.
Await it to receive the result or error. Tasks sent to one child run serially; different
children can run concurrently. Use `asyncio.shield(task)` when cancelling a wait should
leave the scheduled task running.

```python
from tiny_coder.agent import Agent


class Worker(Agent[str, str]):
    async def _invoke(self, input: str) -> str:
        return input.upper()


async def main():
    async with Agent[str, str]() as main:
        worker = main.spawn("worker", Worker)
        print(await worker.send("task input"))
```

`StructuredAgent` inherits this capability and can spawn other structured agents. A business
agent can inherit `Agent` and reuse a structured agent in its `_invoke` method. Each invocation
retains its own configuration, schema, and retry state. Keep the main agent open across tasks
to reuse its children. Closing it cancels and joins outstanding work in itself and its children.
Each agent owns an `EventBus`. `invoke` dispatches through that bus, then publishes
`agent.invoked` with an `Invocation` containing `input` and `output`. Subscribers can await
another agent's work. `send` uses the same dispatch with serial scheduling.

## Composable Capabilities

Pass `extensions=(...)` to `create_structured_agent` to register capabilities once, or compose
them during setup with `agent.use(*extensions)`. Subscribers run in registration order. Each extension must inherit
`AgentExtension` and implement `register(events: EventBus)` to subscribe its async handlers.
Registration rejects objects that do not inherit `AgentExtension` with `TypeError`.

The capabilities in `tiny_coder.agent_extensions` can be used independently:

- `JinjaPrompt`: render system/user templates from a mapping or a callback receiving `AgentCall`.
- `StreamOutput`: display raw text or a JSON string field; optionally limit raw response size.
- `StructuredOutput`: parse the final response and optionally validate or persist it through
  an async `accept(call, output)` callback. The accepted model is available as `result.output`.
- `RetryPolicy`: bound retries for transient API errors and `ValueError`/`OSError`. Validation
  failures add `call.feedback` and select `validation_llm_config` when supplied.

```python
from jinja2 import Environment
from tiny_coder.agent_extensions import JinjaPrompt, RetryPolicy, StreamOutput, StructuredOutput
from tiny_coder.structured_agent import AgentCall, StructuredInput, create_structured_agent

env = Environment()

def prompt_variables(call: AgentCall) -> dict[str, object]:
    return {"topic": call.context}

agent = create_structured_agent(
    llm_config=LocalLlmConfig(),
    response_model=Plan,
    extensions=(
        JinjaPrompt(
            system="You are a planner.",
            user=env.from_string("Plan {{ topic }}. {{ retry_errors | join('; ') }}"),
            variables=prompt_variables,
        ),
        StreamOutput(field="summary"),
        RetryPolicy(max_attempts=3),
        StructuredOutput(),
    ),
)

async def generate(topic: str):
    return await agent.invoke(StructuredInput(context=topic))
```

`JinjaPrompt` refreshes variables on every attempt and supplies `retry_errors` from feedback.
Use `read_file_snapshots(cwd, paths)` in the variables callback for fresh file contents.
Completed tool exchanges are retained when rendering the next prompt.

For task-dependent options, supply callbacks receiving `AgentCall`: `JinjaPrompt.system/user`,
`StreamOutput.field/max_chars/enabled`, and `RetryPolicy.max_attempts/validation_llm_config`.
The registered extensions are reused; resolved stream settings remain local to each call.

Pass application data through `StructuredInput(context=...)`; hooks access it through `call.context`.
Extensions own their private state. Context, feedback, and streaming state are isolated between
concurrent invocations and excluded from checkpoints.
An extension can register a context manager with `hooks.add_scope(scope)` to manage state for
each LLM attempt. Scopes surround `before_call`, streaming, and `after_call`, and unwind before
`on_error`, including on cancellation. `StreamOutput` uses a scoped `ContextVar` so nested calls
restore the outer stream automatically.

## Structured Output

Any node can request a JSON-schema response instead of tool calls:

```python
from pydantic import BaseModel

from tiny_coder.llm import schema_response_format, stream_llm_chat


class Plan(BaseModel):
    summary: str
    steps: list[str]


outcome = await stream_llm_chat(
    llm_cfg=config,
    messages=[{"role": "user", "content": "Plan the change."}],
    response_format=schema_response_format(Plan),
)
plan = outcome.model(Plan)
```

## Checkpointing

`JsonCheckpointer` stores one checkpoint after each node: the step counter, the next node name, and
the full JSON state. Resume a crashed or interrupted run with:

```python
final = asyncio.run(graph.run(resume=True, checkpointer=checkpointer))
```

`Graph.run` takes `max_steps` (default `100`) as a loop guard and raises `GraphError` when exceeded.
Any object satisfying the `Checkpointer` protocol can replace `JsonCheckpointer`.

## Custom State and Reducers

State is a plain `dict[str, Any]`. Node functions return a partial update; the graph merges it using
a per-key reducer (replace by default, append for `messages`).

```python
graph.set_reducer("findings", lambda current, update: [*(current or []), *update])
```

## Development

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tiny_coder tests
uv run pre-commit run --all-files
```

Text file reads and writes must explicitly use `encoding="utf-8"` and `newline="\n"`.
