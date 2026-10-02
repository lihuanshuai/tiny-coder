from __future__ import annotations

import asyncio
from pathlib import Path

from tiny_coder.checkpoint import Checkpoint, JsonCheckpointer


def test_json_checkpointer_round_trips(tmp_path: Path) -> None:
    checkpointer = JsonCheckpointer(tmp_path / "checkpoint.json")
    checkpoint = Checkpoint(
        step=2,
        next_node="review",
        state={"messages": [{"role": "user", "content": "你好"}]},
    )

    asyncio.run(checkpointer.save(checkpoint))
    loaded = asyncio.run(checkpointer.load())

    assert loaded == checkpoint
    raw = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "你好" in raw


def test_json_checkpointer_returns_none_when_file_is_missing(tmp_path: Path) -> None:
    assert asyncio.run(JsonCheckpointer(tmp_path / "missing.json").load()) is None


def test_checkpoint_from_json_rejects_invalid_payload() -> None:
    import pytest

    with pytest.raises(ValueError, match="step"):
        Checkpoint.from_json({"step": "x", "next_node": "a", "state": {}})
    with pytest.raises(ValueError, match="next_node"):
        Checkpoint.from_json({"step": 1, "next_node": "", "state": {}})
    with pytest.raises(ValueError, match="state"):
        Checkpoint.from_json({"step": 1, "next_node": "a", "state": []})
    with pytest.raises(ValueError, match="JSON object"):
        Checkpoint.from_json([])
