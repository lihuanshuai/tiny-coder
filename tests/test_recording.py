from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tiny_coder.llm import LlmChatOutcome, LlmExchange, ToolCall
from tiny_coder.recording import JsonlExchangeRecorder


def _outcome(text: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> LlmChatOutcome:
    return LlmChatOutcome(
        text=text,
        tool_calls=tool_calls,
        prompt_eval_count=3,
        eval_count=4,
        client_wall_time_ms=10.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={"provider": "openai-compatible"},
    )


def test_jsonl_exchange_recorder_appends_records(tmp_path: Path) -> None:
    recorder = JsonlExchangeRecorder(tmp_path / "nested" / "llm.jsonl")
    exchange = LlmExchange(
        messages=[{"role": "user", "content": "你好"}],
        response=_outcome(
            text="ok",
            tool_calls=(
                ToolCall(
                    id="c1",
                    type="function",
                    function={"name": "record_correction", "arguments": "{}"},
                ),
            ),
        ),
    )

    asyncio.run(recorder(exchange))
    asyncio.run(recorder(exchange))

    lines = (tmp_path / "nested" / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["messages"] == [{"role": "user", "content": "你好"}]
    assert payload["response"]["text"] == "ok"
    assert payload["response"]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "name": "record_correction",
            "arguments": "{}",
        }
    ]
    assert payload["response"]["llm"] == {"provider": "openai-compatible"}
    assert "你好" in lines[0]
