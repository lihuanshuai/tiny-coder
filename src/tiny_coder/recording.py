"""JSONL recording of LLM exchanges for inspection and prompt tuning."""

from __future__ import annotations

import json
from pathlib import Path

from tiny_coder.llm import LlmExchange


class JsonlExchangeRecorder:
    """Append every LLM exchange to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    async def __call__(self, exchange: LlmExchange) -> None:
        record = {
            "messages": exchange.messages,
            "response": {
                "started_at": exchange.response.started_at,
                "completed_at": exchange.response.completed_at,
                "client_wall_time_ms": exchange.response.client_wall_time_ms,
                "prompt_eval_count": exchange.response.prompt_eval_count,
                "eval_count": exchange.response.eval_count,
                "text": exchange.response.text,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "name": call.function.get("name", ""),
                        "arguments": call.function.get("arguments", ""),
                    }
                    for call in exchange.response.tool_calls
                ],
                "llm": exchange.response.llm,
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["JsonlExchangeRecorder"]
