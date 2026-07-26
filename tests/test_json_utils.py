from __future__ import annotations

import pytest

from tiny_coder.json_utils import JsonProtocolError, JsonStringFieldStreamer, load_json_object


def test_load_json_object_accepts_fenced_json() -> None:
    assert load_json_object('```json\n{"summary": "ok", "count": 2}\n```') == {
        "summary": "ok",
        "count": 2,
    }


def test_load_json_object_repairs_embedded_object() -> None:
    assert load_json_object('prefix {"message": "hello\nworld"} suffix') == {
        "message": "hello\nworld"
    }


def test_load_json_object_rejects_non_object_root() -> None:
    with pytest.raises(JsonProtocolError, match="根节点必须是 JSON 对象"):
        load_json_object("[1, 2, 3]")


def test_json_string_field_streamer_decodes_chunked_value() -> None:
    streamer = JsonStringFieldStreamer("summary")

    chunks = ['{"summary": "hello', r'\nwor', r'ld\u0021", "other": 1}']
    printed = "".join(streamer.feed(chunk) for chunk in chunks)

    assert printed == "hello\nworld!"
    assert streamer.streamed_text == "hello\nworld!"
