"""A bounded conversation loop with tools, structured acceptance and reusable sessions."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import ExitStack
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai.types.chat import ChatCompletionToolChoiceOptionParam
from pydantic import BaseModel

from tiny_coder.agent import Agent
from tiny_coder.checkpoint import Checkpointer
from tiny_coder.eventbus import Event, EventBus
from tiny_coder.graph import END, Graph
from tiny_coder.llm import (
    LlmChatOutcome,
    LlmConfig,
    LlmExchange,
    ToolCall,
    parse_structured,
    schema_response_format,
    stream_llm_chat,
)
from tiny_coder.state import State
from tiny_coder.steering import current_instructions
from tiny_coder.tool_node import Tool, ToolNode, has_tool_calls

_ModelT = TypeVar("_ModelT", bound=BaseModel)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_LLM_CONFIG_KEY = "_llm_config"
_ATTEMPT_KEY = "_attempt"
_RETRY_KEY = "_retry"
_RESPONSE_FORMAT_KEY = "_response_format"
_SYSTEM_PROMPT_KEY = "_system_prompt"
SUBMIT_RESULT = "submit_result"


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
    previous_error: Exception | None = None
    feedback: list[str] = field(default_factory=list)
    chunk: str = ""
    steering: tuple[str, ...] = ()
    resources: ExitStack = field(default_factory=ExitStack, repr=False)
    tools: list[Tool] = field(default_factory=list)
    turn: int = 1
    prepared_extensions: set[int] = field(default_factory=set, repr=False)
    submission_enabled: bool = True
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None

    @property
    def submission(self) -> ToolCall | None:
        if self.outcome is not None and self.response_model is not None and self.tools:
            return next(
                (tc for tc in self.outcome.tool_calls if tc.function.get("name") == SUBMIT_RESULT),
                None,
            )
        return None

    @property
    def response_text(self) -> str:
        """The final JSON, whether returned as text or submit_result arguments."""
        submission = self.submission
        if submission is not None:
            return submission.function.get("arguments", "")
        return self.outcome.text if self.outcome is not None else ""


CALL_STARTED = Event[AgentCall]("llm.call_started")
BEFORE_CALL = Event[AgentCall]("llm.before_call")
RESPONSE_RECEIVED = Event[AgentCall]("llm.response_received")
BEFORE_OUTPUT = Event[AgentCall]("llm.before_output")
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
        if self.output is not None and self.outcome is not None and self.outcome.tool_calls:
            return self.output.model_dump_json()
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
    tools: Sequence[Tool] | None = None
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _Invocation:
    response_model: type[BaseModel] | None
    context: object | None
    feedback: list[str] = field(default_factory=list)
    call: AgentCall | None = None
    tools: Sequence[Tool] = ()
    prepared_extensions: set[int] = field(default_factory=set)


@dataclass
class StructuredAgent(Agent[StructuredInput, StructuredResult]):
    """Run a task through tool calls and validation until a final response is accepted."""

    graph: Graph
    llm_config: LlmConfig
    response_model: type[BaseModel] | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: list[Tool] = field(default_factory=list)
    checkpointer: Checkpointer | None = None
    accept: Callable[[AgentCall, BaseModel], Awaitable[None]] | None = None
    _invocation: ContextVar[_Invocation | None] = field(
        default_factory=lambda: ContextVar("structured_agent_invocation", default=None),
        init=False,
        repr=False,
    )

    def use(self, *extensions: AgentExtension) -> StructuredAgent:
        """Compose capabilities for all future invocations."""
        if self.accept is not None:
            # Output extensions depend on this module's lifecycle types.
            from tiny_coder.agent_extensions import StructuredOutput

            if any(isinstance(extension, StructuredOutput) for extension in extensions):
                raise ValueError("configure either accept or a StructuredOutput extension")
        for extension in extensions:
            if not isinstance(extension, AgentExtension):
                raise TypeError("extensions must be AgentExtension instances")
            extension.register(self.events)
        return self

    def conversation(self) -> AgentConversation:
        """Open an isolated, serial conversation using this agent's loop."""
        return AgentConversation(self)

    async def _emit_chunk(self, chunk: str) -> None:
        invocation = self._invocation.get()
        assert invocation is not None and invocation.call is not None
        call = invocation.call
        if call.tools and call.response_model is not None and chunk.strip():
            from tiny_coder.agent_extensions import ResponseValidationError

            raise ResponseValidationError(
                "structured tool response emitted assistant text; stream stopped before completion",
                code="submission_required",
                suggestion=(
                    "Use the API tool_calls field, not assistant text or textual tool-call tags. "
                    "Read missing context with the context tools, then call submit_result "
                    "with the complete result as arguments. No submission has been accepted."
                ),
            )
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
        invocation = _Invocation(
            response_model=model,
            context=input.context,
            tools=self.tools if input.tools is None else input.tools,
        )
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
        history = deepcopy([dict(message) for message in input.messages])
        if input.prompt:
            history.append({"role": "user", "content": input.prompt})
        if not history and not self.events.has_subscribers(BEFORE_CALL):
            raise ValueError("prompt or messages is required")
        config = input.llm_config or self.llm_config
        return {
            **deepcopy(dict(input.state)),
            "messages": history,
            _ATTEMPT_KEY: 0,
            _RETRY_KEY: False,
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
        previous_error = invocation.call.error if invocation.call is not None else None
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
            previous_error=previous_error,
            feedback=invocation.feedback,
            tools=list(invocation.tools),
            turn=invocation.call.turn + 1 if invocation.call is not None else 1,
            prepared_extensions=invocation.prepared_extensions,
        )
        if (
            invocation.call is not None
            and invocation.call.outcome is not None
            and invocation.call.outcome.tool_calls
            and not invocation.call.retry
        ):
            call.attempt = invocation.call.attempt
        invocation.call = call
        return call

    async def _run_call(self, call: AgentCall) -> None:
        """Run the ordered lifecycle; release resources before dispatching failures."""
        # Extensions import the event types above; load the default parser after module setup.
        from tiny_coder.agent_extensions import ResponseValidationError, StructuredOutput

        try:
            with call.resources:
                await self.events.emit(CALL_STARTED, call)
                await self.events.emit(BEFORE_CALL, call)
                call.outcome = await self._complete(call)
                call.messages.append(call.outcome.message)
                await self.events.emit(RESPONSE_RECEIVED, call)
                if call.submission is not None and len(call.outcome.tool_calls) != 1:
                    raise ResponseValidationError(
                        "submit_result must be the only tool call in its turn",
                        suggestion="Complete context tool calls first, then call submit_result once in a separate turn.",
                    )
                if not call.outcome.tool_calls or call.submission is not None:
                    await self.events.emit(BEFORE_OUTPUT, call)
                await self.events.emit(AFTER_CALL, call)
                if call.response_model is not None and call.output is None:
                    await StructuredOutput(self.accept).parse(call)
                if call.submission is not None:
                    self._submission_reply(call, '{"accepted":true}')
        except Exception as error:
            call.error = error
            await self.events.emit(CALL_FAILED, call)
            if not call.retry:
                raise

    @staticmethod
    def _submission_reply(call: AgentCall, content: str) -> None:
        assert call.outcome is not None
        for tool_call in call.outcome.tool_calls:
            call.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.get("name", ""),
                    "content": content,
                }
            )

    async def _tools(self, state: State) -> dict[str, Any]:
        invocation = self._invocation.get()
        assert invocation is not None and invocation.call is not None
        return dict(await ToolNode(invocation.call.tools)(state))

    async def _complete(self, call: AgentCall) -> LlmChatOutcome:
        """Translate a prepared call to the LLM transport and publish streamed data."""
        if not call.messages:
            raise ValueError("prompt or messages is required")
        # Apply after prompt extensions render. Keep supplementary instructions
        # out of graph history so retries and tool loops cannot duplicate them.
        call.steering = current_instructions()
        tools = list(call.tools)
        if tools and call.response_model is not None:
            if any(tool.name == SUBMIT_RESULT for tool in tools):
                raise ValueError(f"reserved tool name: {SUBMIT_RESULT}")

        if tools and call.response_model is not None and call.submission_enabled:

            async def submit(state: State, **kwargs: Any) -> str:
                raise RuntimeError("submit_result is handled by the agent loop")

            tools.append(
                Tool(
                    name=SUBMIT_RESULT,
                    description="Submit the complete final result after reading context and completing the task. Call this alone. Validation errors are returned for correction; a rejected submission is not accepted.",
                    model=call.response_model,
                    handler=submit,
                    parameters_schema=call.response_format["json_schema"]["schema"]
                    if call.response_format is not None
                    else None,
                )
            )
        ToolNode(tools)  # Reject duplicate names before sending any request.
        return await stream_llm_chat(
            llm_cfg=call.llm_config,
            messages=[
                {"role": "system", "content": call.system_prompt},
                *call.messages,
                *({"role": "user", "content": text} for text in call.steering),
            ],
            tools=tools or None,
            tool_choice=call.tool_choice,
            response_format=None if tools else call.response_format,
            on_chunk=self._emit_chunk
            if self.events.has_subscribers(CHUNK_RECEIVED)
            or (tools and call.response_model is not None)
            else None,
            on_exchange=self._emit_exchange
            if self.events.has_subscribers(EXCHANGE_RECEIVED)
            else None,
        )

    def _route(self, state: State) -> str:
        if state.get(_RETRY_KEY):
            return "llm"
        return "tools" if has_tool_calls(state) else END


@dataclass
class AgentConversation:
    """A transcript shared across successful turns, isolated from other conversations.

    Failed invocations do not replace the last successful transcript. Tool side effects
    remain the caller's responsibility; this object is not a filesystem transaction.
    """

    agent: StructuredAgent
    _state: State = field(default_factory=dict, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return deepcopy(self._state.get("messages", []))

    async def send(self, input: StructuredInput) -> StructuredResult:
        async with self._lock:
            request = StructuredInput(
                prompt=input.prompt,
                messages=[*self.messages, *input.messages],
                llm_config=input.llm_config,
                response_model=input.response_model,
                system_prompt=input.system_prompt,
                context=input.context,
                tools=input.tools,
                state={**deepcopy(self._state), **input.state},
            )
            result = await self.agent.send(request)
            self._state = deepcopy(result.state)
            return result


def create_structured_agent(
    *,
    llm_config: LlmConfig,
    response_model: type[BaseModel] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Sequence[Tool] | None = None,
    checkpointer: Checkpointer | None = None,
    max_steps: int = 100,
    extensions: Sequence[AgentExtension] = (),
    max_attempts: int | None = None,
    accept: Callable[[AgentCall, BaseModel], Awaitable[None]] | None = None,
) -> StructuredAgent:
    """Build a bounded task loop; no prompt or output extension is required.

    With tools and a response model, expose submit_result with that model's schema
    and validate its arguments locally; exploration is not constrained to JSON output.
    Without tools, request JSON schema output directly. ``accept`` may reject a result
    before it is returned. ``max_attempts`` enables failure feedback and retries;
    successful tool rounds do not consume that budget. ``max_steps`` bounds all LLM
    and tool-node executions, including loops without a final submission.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    resolved_tools = list(tools) if tools is not None else []
    ToolNode(resolved_tools)
    graph = Graph(max_steps=max_steps)
    agent = StructuredAgent(
        graph=graph,
        llm_config=llm_config,
        response_model=response_model,
        system_prompt=system_prompt,
        tools=resolved_tools,
        checkpointer=checkpointer,
        accept=accept,
    )
    graph.add_node("llm", agent._call)
    graph.add_node("tools", agent._tools)
    graph.add_edge("tools", "llm")
    graph.add_conditional_edges(
        "llm",
        agent._route,
        {"llm": "llm", END: END, "tools": "tools"},
    )
    if max_attempts is not None:
        # RetryPolicy depends on the lifecycle types defined in this module.
        from tiny_coder.agent_extensions import RetryPolicy

        if any(isinstance(extension, RetryPolicy) for extension in extensions):
            raise ValueError("configure either max_attempts or a RetryPolicy extension")
        agent.use(RetryPolicy(max_attempts))
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
    "RESPONSE_RECEIVED",
    "BEFORE_OUTPUT",
    "AFTER_CALL",
    "CALL_FAILED",
    "CHUNK_RECEIVED",
    "EXCHANGE_RECEIVED",
    "StructuredAgent",
    "AgentConversation",
    "StructuredInput",
    "StructuredResult",
    "create_structured_agent",
]
