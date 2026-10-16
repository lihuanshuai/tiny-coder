from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest
from jinja2 import Environment
from pydantic import BaseModel, Field
from test_structured_agent import SampleLlmConfig, _outcome

from tiny_coder.agent_extensions import JinjaPrompt, RetryPolicy, StructuredOutput
from tiny_coder.context import (
    BudgetReading,
    ContextBudget,
    estimate_tokens,
)
from tiny_coder.llm import LlmChatOutcome, LlmConfig
from tiny_coder.steering import Steering
from tiny_coder.structured_agent import (
    BEFORE_CALL,
    AgentCall,
    StructuredInput,
    create_structured_agent,
)


class Output(BaseModel):
    summary: str = Field(description="必须返回完整核对结论。" * 100)


def test_token_estimate_does_not_treat_chinese_as_ascii() -> None:
    assert estimate_tokens("汉" * 100) > estimate_tokens("a" * 100) * 4
    assert estimate_tokens("") == 0


@pytest.mark.parametrize("valid_first", [False, True])
def test_budget_calibrates_input_usage_and_keeps_models_separate(
    monkeypatch: pytest.MonkeyPatch, valid_first: bool
) -> None:
    env = Environment()
    readings: list[BudgetReading] = []
    systems: list[str] = []
    calls = 0

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal calls
        calls += 1
        systems.append(kwargs["messages"][0]["content"])
        assert systems[-1].count("BUDGET NOTICE") == 1
        outcome = _outcome('invalid' if calls == 1 and not valid_first else '{"summary":"ok"}')
        return replace(outcome, prompt_eval_count=8000)

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    budget = ContextBudget(
        env.from_string("BUDGET NOTICE {{ budget.estimated_input_tokens }}"), readings.append
    )
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        extensions=(
            JinjaPrompt("system", "user"),
            budget,
            RetryPolicy(2),
            StructuredOutput(),
        ),
    )

    async def run() -> None:
        async with agent:
            await agent.invoke(StructuredInput())
            await agent.invoke(StructuredInput())
            await agent.invoke(StructuredInput(llm_config=SampleLlmConfig(llm_model="other")))

    asyncio.run(run())
    assert readings[0].observed_input_tokens is None
    assert readings[1].observed_input_tokens == 8000
    assert readings[1].estimated_input_tokens > readings[0].estimated_input_tokens * 2
    assert readings[-1].observed_input_tokens is None
    assert readings[-1].estimated_input_tokens == readings[0].estimated_input_tokens
    assert readings[0].input_budget == 3996


def test_budget_includes_schema_steering_and_reports_unknown_output_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = Environment()
    readings: list[BudgetReading] = []
    notice = env.from_string(
        "预算 {{ budget.estimated_input_tokens }}，预留 {{ budget.output_reserve }}"
    )

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        assert kwargs["messages"][-1]["content"] == "补充要求" * 100
        payload = {"messages": kwargs["messages"], "response_format": kwargs["response_format"]}
        estimated = estimate_tokens(json.dumps(payload, ensure_ascii=False))
        assert abs(readings[-1].estimated_input_tokens - estimated) <= 1
        return _outcome('{"summary":"ok"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=LlmConfig.model_validate(
            {**SampleLlmConfig().model_dump(), "max_output_tokens": None}
        ),
        extensions=(ContextBudget(notice, readings.append),),
    )

    async def run() -> None:
        steering = Steering()
        with steering.activate():
            steering.add("补充要求" * 100)
            await agent.invoke(StructuredInput(prompt="hello"))
            await agent.invoke(StructuredInput(prompt="hello", response_model=Output))

    asyncio.run(run())
    assert readings[0].estimated_input_tokens > estimate_tokens("补充要求" * 100)
    assert readings[1].estimated_input_tokens > readings[0].estimated_input_tokens
    assert readings[0].output_reserve is None
    assert readings[0].input_budget is None
    assert not readings[0].over_budget


def test_budget_notice_does_not_accumulate_without_prompt_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = Environment()
    systems: list[str] = []

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        systems.append(kwargs["messages"][0]["content"])
        return _outcome('invalid' if len(systems) == 1 else '{"summary":"ok"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    readings: list[BudgetReading] = []
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Output,
        extensions=(
            ContextBudget(env.from_string("BUDGET NOTICE"), readings.append),
            RetryPolicy(2),
            StructuredOutput(),
        ),
    )
    asyncio.run(agent.invoke(StructuredInput(prompt="hello")))
    assert len(systems) == 2
    assert all(system.count("BUDGET NOTICE") == 1 for system in systems)


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_budget_cleans_up_failed_calls_and_reuses_the_agent(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    readings: list[BudgetReading] = []
    calls: list[AgentCall] = []

    async def capture(call: AgentCall) -> None:
        calls.append(call)

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        assert kwargs["messages"][0]["content"].count("BUDGET NOTICE") == 1
        if len(calls) == 1:
            raise error_type("interrupted")
        return replace(_outcome('{"summary":"ok"}'), prompt_eval_count=8000)

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        system_prompt="base system",
        extensions=(ContextBudget(Environment().from_string("BUDGET NOTICE"), readings.append),),
    )
    agent.events.subscribe(BEFORE_CALL, capture)

    async def run() -> None:
        async with agent:
            with pytest.raises(error_type, match="interrupted"):
                await agent.invoke(StructuredInput(prompt="hello"))
            await agent.invoke(StructuredInput(prompt="hello"))
            await agent.invoke(StructuredInput(prompt="hello"))

    asyncio.run(run())
    assert all(call.system_prompt == "base system" for call in calls)
    assert [reading.observed_input_tokens for reading in readings] == [None, None, 8000]
