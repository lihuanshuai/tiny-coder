from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Template
from pydantic import BaseModel, ConfigDict

from tiny_coder.eventbus import EventBus
from tiny_coder.steering import current_instructions
from tiny_coder.structured_agent import BEFORE_CALL, AgentCall, AgentExtension


class BudgetReading(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimated_input_tokens: int
    context_window: int
    output_reserve: int | None
    observed_input_tokens: int | None

    @property
    def input_budget(self) -> int | None:
        if self.output_reserve is None:
            return None
        return max(0, self.context_window - self.output_reserve)

    @property
    def over_budget(self) -> bool:
        return self.input_budget is not None and self.estimated_input_tokens > self.input_budget


def estimate_tokens(text: str) -> int:
    """Estimate mixed-language text; non-ASCII uses UTF-8 byte weight, not chars/4."""
    ascii_chars = sum(character.isascii() for character in text)
    other_bytes = len(text.encode("utf-8")) - ascii_chars
    return math.ceil(ascii_chars / 4 + other_bytes / 2)


@dataclass
class ContextBudget(AgentExtension):
    """Measure structured prompts after rendering and calibrate from input usage.

    Register after prompt rendering. Notices are advisory;
    this extension never truncates source text or changes output limits.
    """

    notice: Template
    report: Callable[[BudgetReading], None]
    _usage: dict[tuple[str, str], tuple[int, float]] = field(default_factory=dict, init=False)

    def register(self, events: EventBus) -> None:
        events.subscribe(BEFORE_CALL, self.prepare)

    async def prepare(self, call: AgentCall) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": call.system_prompt},
                *call.messages,
                *({"role": "user", "content": text} for text in current_instructions()),
            ],
            "response_format": call.response_format,
        }
        raw = estimate_tokens(json.dumps(payload, ensure_ascii=False))
        key = (call.llm_config.base_url, call.llm_config.llm_model)
        observed, factor = self._usage.get(key, (None, 1.0))
        reading = BudgetReading(
            estimated_input_tokens=math.ceil(raw * factor),
            context_window=call.llm_config.num_ctx,
            output_reserve=call.llm_config.max_output_tokens,
            observed_input_tokens=observed,
        )
        # Render once. The notice describes the payload measured before adding itself.
        notice = "\n\n" + self.notice.render(budget=reading).strip()
        call.system_prompt += notice
        raw += estimate_tokens(json.dumps(notice, ensure_ascii=False)[1:-1])
        call.resources.callback(self.finish, call, raw, notice)
        self.report(reading.model_copy(update={"estimated_input_tokens": math.ceil(raw * factor)}))

    def finish(self, call: AgentCall, raw: int, notice: str) -> None:
        # Keep retry prompts clean even on failed validation, transport errors or cancellation.
        call.system_prompt = call.system_prompt.removesuffix(notice)
        if call.outcome is None or call.outcome.prompt_eval_count is None:
            return
        observed = call.outcome.prompt_eval_count
        if observed <= 0 or raw <= 0:
            return
        key = (call.llm_config.base_url, call.llm_config.llm_model)
        sample = max(1.0, observed / raw)
        previous = self._usage.get(key)
        factor = sample if previous is None else (previous[1] + sample) / 2
        self._usage[key] = (observed, factor)
