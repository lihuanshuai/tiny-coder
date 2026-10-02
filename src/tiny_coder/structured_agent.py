"""A ready-to-run agent that returns one structured response, optionally with tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from tiny_coder.agent import Agent
from tiny_coder.checkpoint import Checkpointer
from tiny_coder.graph import END, Graph
from tiny_coder.llm import (
    LlmChatOutcome,
    LlmConfig,
    LlmExchange,
    parse_structured,
    schema_response_format,
    stream_llm_chat,
)
from tiny_coder.state import State
from tiny_coder.tool_node import Tool, ToolNode, has_tool_calls

_ModelT = TypeVar("_ModelT", bound=BaseModel)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_LLM_CONFIG_KEY = "_llm_config"
_ATTEMPT_KEY = "_attempt"
_RETRY_KEY = "_retry"
_RESPONSE_FORMAT_KEY = "_response_format"
_SYSTEM_PROMPT_KEY = "_system_prompt"

HookEvent: TypeAlias = Literal["before_call", "after_call", "on_error", "on_chunk"]


@dataclass
class AgentCall:
    """One call's mutable request and loop decision, local to an invocation.

    Hooks may replace messages/config before a call; afterward, messages include
    the assistant response. Set retry=True to request another call. State is shared
    with tool handlers;
    keep custom state values JSON-compatible when using a checkpointer.
    """

    state: State
    messages: list[dict[str, Any]]
    llm_config: LlmConfig
    system_prompt: str
    attempt: int
    response_format: dict[str, Any] | None = None
    outcome: LlmChatOutcome | None = None
    error: Exception | None = None
    retry: bool = False
    response_model: type[BaseModel] | None = None
    output: BaseModel | None = None
    context: object | None = None
    previous_outcome: LlmChatOutcome | None = None
    feedback: list[str] = field(default_factory=list)
    chunk: str = ""


AgentHook: TypeAlias = Callable[[AgentCall], Awaitable[None]]
AgentScope: TypeAlias = Callable[[AgentCall], AbstractContextManager[None]]


@dataclass(frozen=True)
class StructuredResult:
    """Final state of one structured agent run."""

    state: State
    outcome: LlmChatOutcome | None = None
    output: BaseModel | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        messages = self.state.get("messages")
        return [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []

    @property
    def text(self) -> str:
        """Content of the last non-empty assistant message."""
        for message in reversed(self.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
        return ""

    def model(self, model_type: type[_ModelT]) -> _ModelT | None:
        """Parse the last non-empty assistant message as the structured model."""
        if isinstance(self.output, model_type):
            return self.output
        text = self.text
        return parse_structured(text, model_type) if text else None


class AgentExtension(ABC):
    """Base class for composable agent capabilities."""

    @abstractmethod
    def register(self, hooks: AgentHooks) -> None:
        """Register this capability's hooks."""


class AgentHooks:
    """Ordered hook registration shared by agents and their extensions."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[AgentHook]] = {
            "before_call": [],
            "after_call": [],
            "on_error": [],
            "on_chunk": [],
        }
        self._scopes: list[AgentScope] = []

    def add_hook(self, event: HookEvent, hook: AgentHook) -> None:
        if event not in self._hooks:
            raise ValueError(f"unknown hook event: {event!r}")
        self._hooks[event].append(hook)

    def add_scope(self, scope: AgentScope) -> None:
        """Manage extension state for each LLM attempt, including cancellation."""
        self._scopes.append(scope)

    @contextmanager
    def scope(self, call: AgentCall) -> Iterator[None]:
        with ExitStack() as stack:
            for scope in self._scopes:
                stack.enter_context(scope(call))
            yield

    def use(self, *extensions: AgentExtension) -> None:
        for extension in extensions:
            if not isinstance(extension, AgentExtension):
                raise TypeError("extensions must be AgentExtension instances")
            extension.register(self)

    def copy(self) -> AgentHooks:
        copied = AgentHooks()
        copied._hooks = {event: list(hooks) for event, hooks in self._hooks.items()}
        copied._scopes = list(self._scopes)
        return copied

    def has(self, event: HookEvent) -> bool:
        return bool(self._hooks[event])

    async def emit(self, event: HookEvent, call: AgentCall) -> None:
        for hook in tuple(self._hooks[event]):
            await hook(call)


@dataclass
class StructuredInput:
    """Input and optional LLM settings accepted by both invoke and send."""

    prompt: str | None = None
    messages: Sequence[Mapping[str, Any]] = ()
    llm_config: LlmConfig | None = None
    response_model: type[BaseModel] | None = None
    system_prompt: str | None = None
    context: object | None = None


@dataclass
class _Invocation:
    hooks: AgentHooks
    response_model: type[BaseModel] | None
    context: object | None
    feedback: list[str] = field(default_factory=list)
    call: AgentCall | None = None


@dataclass
class StructuredAgent(Agent[StructuredInput, StructuredResult]):
    """An agent that returns one structured model response, using tools when configured."""

    graph: Graph
    llm_config: LlmConfig
    response_model: type[BaseModel] | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: list[Tool] = field(default_factory=list)
    checkpointer: Checkpointer | None = None
    on_chunk: Callable[[str], Awaitable[None]] | None = None
    on_exchange: Callable[[LlmExchange], Awaitable[None]] | None = None
    _hooks: AgentHooks = field(default_factory=AgentHooks, init=False, repr=False)
    _invocation: ContextVar[_Invocation | None] = field(
        default_factory=lambda: ContextVar("structured_agent_invocation", default=None),
        init=False,
        repr=False,
    )

    def add_hook(self, event: HookEvent, hook: AgentHook) -> None:
        """Register an async hook in execution order."""
        self._hooks.add_hook(event, hook)

    def use(self, *extensions: AgentExtension) -> StructuredAgent:
        """Compose capabilities for all future invocations."""
        self._hooks.use(*extensions)
        return self

    async def _emit_chunk(self, chunk: str) -> None:
        invocation = self._invocation.get()
        assert invocation is not None and invocation.call is not None
        call = invocation.call
        call.chunk = chunk
        await invocation.hooks.emit("on_chunk", call)
        if self.on_chunk is not None:
            await self.on_chunk(chunk)

    async def _invoke(self, input: StructuredInput) -> StructuredResult:
        """Invoke this agent with fresh messages and retry state.

        Omitted options inherit the agent's defaults. Repeated invocations reuse
        the graph. Use spawn/send for child agents and scheduled invocations.
        """
        hooks = self._hooks.copy()
        history = [dict(message) for message in input.messages]
        if input.prompt:
            history.append({"role": "user", "content": input.prompt})
        if not history and not hooks.has("before_call"):
            raise ValueError("prompt or messages is required")
        config = input.llm_config or self.llm_config
        model = input.response_model or self.response_model
        invocation = _Invocation(hooks=hooks, response_model=model, context=input.context)
        token = self._invocation.set(invocation)
        try:
            state = await self.graph.run(
                {
                    "messages": history,
                    _LLM_CONFIG_KEY: config.model_dump(),
                    _RESPONSE_FORMAT_KEY: schema_response_format(model)
                    if model is not None
                    else None,
                    _SYSTEM_PROMPT_KEY: self.system_prompt
                    if input.system_prompt is None
                    else input.system_prompt,
                },
                checkpointer=self.checkpointer,
            )
            call = invocation.call
            return StructuredResult(
                state=_public_state(state),
                outcome=call.outcome if call is not None else None,
                output=call.output if call is not None else None,
            )
        finally:
            self._invocation.reset(token)

    async def _call(self, state: State) -> dict[str, Any]:
        invocation = self._invocation.get()
        assert invocation is not None
        previous_outcome = invocation.call.outcome if invocation.call is not None else None
        messages = state.get("messages")
        history = messages if isinstance(messages, list) else []
        call = AgentCall(
            state=state,
            messages=[dict(message) for message in history],
            llm_config=_config_from_state(state, self.llm_config),
            system_prompt=state.get(_SYSTEM_PROMPT_KEY, self.system_prompt),
            attempt=state.get(_ATTEMPT_KEY, 0) + 1,
            response_format=state.get(_RESPONSE_FORMAT_KEY),
            response_model=invocation.response_model,
            context=invocation.context,
            previous_outcome=previous_outcome,
            feedback=invocation.feedback,
        )
        invocation.call = call
        try:
            with invocation.hooks.scope(call):
                await invocation.hooks.emit("before_call", call)
                if not call.messages:
                    raise ValueError("prompt or messages is required")
                call.outcome = await stream_llm_chat(
                    llm_cfg=call.llm_config,
                    messages=[{"role": "system", "content": call.system_prompt}, *call.messages],
                    tools=self.tools or None,
                    response_format=call.response_format,
                    on_chunk=self._emit_chunk
                    if self.on_chunk is not None or invocation.hooks.has("on_chunk")
                    else None,
                    on_exchange=self.on_exchange,
                )
                call.messages.append(call.outcome.message)
                await invocation.hooks.emit("after_call", call)
        except Exception as error:
            call.error = error
            await invocation.hooks.emit("on_error", call)
            if not call.retry:
                raise
        state["messages"] = call.messages
        return {
            _LLM_CONFIG_KEY: call.llm_config.model_dump(),
            _ATTEMPT_KEY: call.attempt,
            _RETRY_KEY: call.retry,
            _RESPONSE_FORMAT_KEY: call.response_format,
            _SYSTEM_PROMPT_KEY: call.system_prompt,
        }

    def _route(self, state: State) -> str:
        if state.get(_RETRY_KEY):
            return "llm"
        return "tools" if self.tools and has_tool_calls(state) else END


def create_structured_agent(
    *,
    llm_config: LlmConfig,
    response_model: type[BaseModel] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Sequence[Tool] | None = None,
    checkpointer: Checkpointer | None = None,
    max_steps: int = 100,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    on_exchange: Callable[[LlmExchange], Awaitable[None]] | None = None,
    extensions: Sequence[AgentExtension] = (),
) -> StructuredAgent:
    """Build an agent that returns one structured response, or routes tool calls when tools are given.

    Set ``response_model`` to force JSON schema output. When tools are configured and no
    response model is set, the run is a pure tool loop whose results arrive through the
    tool handlers, so the server is not forced into structured output on every turn.
    Register hooks with ``agent.add_hook``. A hook's ``call.retry`` requests another
    LLM call; ``max_steps`` bounds both these calls and tool-node executions.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    resolved_tools = list(tools) if tools is not None else []
    graph = Graph(max_steps=max_steps)
    agent = StructuredAgent(
        graph=graph,
        llm_config=llm_config,
        response_model=response_model,
        system_prompt=system_prompt,
        tools=resolved_tools,
        checkpointer=checkpointer,
        on_chunk=on_chunk,
        on_exchange=on_exchange,
    )
    graph.add_node("llm", agent._call)
    if resolved_tools:
        graph.add_node("tools", ToolNode(resolved_tools))
        graph.add_edge("tools", "llm")
    graph.add_conditional_edges(
        "llm",
        agent._route,
        {"llm": "llm", END: END, **({"tools": "tools"} if resolved_tools else {})},
    )
    return agent.use(*extensions)


def _config_from_state(state: State, default: LlmConfig) -> LlmConfig:
    payload = state.get(_LLM_CONFIG_KEY)
    if isinstance(payload, dict):
        return LlmConfig.model_validate(payload)
    return default


def _public_state(state: State) -> State:
    return {
        key: value
        for key, value in state.items()
        if key
        not in {_LLM_CONFIG_KEY, _ATTEMPT_KEY, _RETRY_KEY, _RESPONSE_FORMAT_KEY, _SYSTEM_PROMPT_KEY}
    }


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentCall",
    "AgentExtension",
    "AgentHooks",
    "AgentHook",
    "AgentScope",
    "HookEvent",
    "StructuredAgent",
    "StructuredInput",
    "StructuredResult",
    "create_structured_agent",
]
