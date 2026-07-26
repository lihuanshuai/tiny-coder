from __future__ import annotations

from tiny_coder.yaml_utils import dump_yaml_text


def test_dump_yaml_text_preserves_unicode_and_key_order() -> None:
    data = {"title": "标题", "items": ["一", "二"]}

    assert dump_yaml_text(data, sort_keys=False) == "title: 标题\nitems:\n- 一\n- 二\n"


def test_dump_yaml_text_uses_literal_block_for_multiline_strings() -> None:
    assert dump_yaml_text({"body": "line 1\nline 2\n"}) == "body: |-\n  line 1\n  line 2\n"
