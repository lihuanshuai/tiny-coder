"""A ready-to-run agent that returns one structured response, optionally with tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from tiny_coder.agent import Agent
from tiny_coder.checkpoint import Checkpointer
from tiny_coder.eventbus import Event, EventBus
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


@dataclass
class AgentCall:
    """One call's mutable request and loop decision, local to an invocation.

    Subscribers may replace messages/config before a call; afterward, messages include
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
    resources: ExitStack = field(default_factory=ExitStack, repr=False)


CALL_STARTED = Event[AgentCall]("llm.call_started")
BEFORE_CALL = Event[AgentCall]("llm.before_call")
AFTER_CALL = Event[AgentCall]("llm.after_call")
CALL_FAILED = Event[AgentCall]("llm.call_failed")
CHUNK_RECEIVED = Event[AgentCall]("llm.chunk_received")
EXCHANGE_RECEIVED = Event[LlmExchange]("llm.exchange_received")


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
    def register(self, events: EventBus) -> None:
        """Subscribe this capability to agent events."""


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
    _invocation: ContextVar[_Invocation | None] = field(
        default_factory=lambda: ContextVar("structured_agent_invocation", default=None),
        init=False,
        repr=False,
    )

    def use(self, *extensions: AgentExtension) -> StructuredAgent:
        """Compose capabilities for all future invocations."""
        for extension in extensions:
            if not isinstance(extension, AgentExtension):
                raise TypeError("extensions must be AgentExtension instances")
            extension.register(self.events)
        return self

    async def _emit_chunk(self, chunk: str) -> None:
        invocation = self._invocation.get()
        assert invocation is not None and invocation.call is not None
        call = invocation.call
        call.chunk = chunk
        await self.events.emit(CHUNK_RECEIVED, call)

    async def _emit_exchange(self, exchange: LlmExchange) -> None:
        await self.events.emit(EXCHANGE_RECEIVED, exchange)

    async def _invoke(self, input: StructuredInput) -> StructuredResult:
        """Invoke this agent with fresh messages and retry state.

        Omitted options inherit the agent's defaults. Repeated invocations reuse
        the graph. Use spawn/send for child agents and scheduled invocations.
        """
        model = input.response_model or self.response_model
        initial = self._initial_state(input, model)
        invocation = _Invocation(response_model=model, context=input.context)
        token = self._invocation.set(invocation)
        try:
            state = await self.graph.run(
                initial,
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

    def _initial_state(self, input: StructuredInput, model: type[BaseModel] | None) -> State:
        history = [dict(message) for message in input.messages]
        if input.prompt:
            history.append({"role": "user", "content": input.prompt})
        if not history and not self.events.has_subscribers(BEFORE_CALL):
            raise ValueError("prompt or messages is required")
        config = input.llm_config or self.llm_config
        return {
            "messages": history,
            _LLM_CONFIG_KEY: config.model_dump(),
            _RESPONSE_FORMAT_KEY: schema_response_format(model) if model is not None else None,
            _SYSTEM_PROMPT_KEY: self.system_prompt
            if input.system_prompt is None
            else input.system_prompt,
        }

    async def _call(self, state: State) -> dict[str, Any]:
        call = self._create_call(state)
        await self._run_call(call)
        state["messages"] = call.messages
        return {
            _LLM_CONFIG_KEY: call.llm_config.model_dump(),
            _ATTEMPT_KEY: call.attempt,
            _RETRY_KEY: call.retry,
            _RESPONSE_FORMAT_KEY: call.response_format,
            _SYSTEM_PROMPT_KEY: call.system_prompt,
        }

    def _create_call(self, state: State) -> AgentCall:
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
        return call

    async def _run_call(self, call: AgentCall) -> None:
        """Run the ordered lifecycle; release resources before dispatching failures."""
        try:
            with call.resources:
                await self.events.emit(CALL_STARTED, call)
                await self.events.emit(BEFORE_CALL, call)
                call.outcome = await self._complete(call)
                call.messages.append(call.outcome.message)
                await self.events.emit(AFTER_CALL, call)
        except Exception as error:
            call.error = error
            await self.events.emit(CALL_FAILED, call)
            if not call.retry:
                raise

    async def _complete(self, call: AgentCall) -> LlmChatOutcome:
        """Translate a prepared call to the LLM transport and publish streamed data."""
        if not call.messages:
            raise ValueError("prompt or messages is required")
        return await stream_llm_chat(
            llm_cfg=call.llm_config,
            messages=[{"role": "system", "content": call.system_prompt}, *call.messages],
            tools=self.tools or None,
            response_format=call.response_format,
            on_chunk=self._emit_chunk if self.events.has_subscribers(CHUNK_RECEIVED) else None,
            on_exchange=self._emit_exchange
            if self.events.has_subscribers(EXCHANGE_RECEIVED)
            else None,
        )

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
    extensions: Sequence[AgentExtension] = (),
) -> StructuredAgent:
    """Build an agent that returns one structured response, or routes tool calls when tools are given.

    Set ``response_model`` to force JSON schema output. When tools are configured and no
    response model is set, the run is a pure tool loop whose results arrive through the
    tool handlers, so the server is not forced into structured output on every turn.
    Subscribe through ``agent.events``. A subscriber's ``call.retry`` requests another
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
    "CALL_STARTED",
    "BEFORE_CALL",
    "AFTER_CALL",
    "CALL_FAILED",
    "CHUNK_RECEIVED",
    "EXCHANGE_RECEIVED",
    "StructuredAgent",
    "StructuredInput",
    "StructuredResult",
    "create_structured_agent",
]
