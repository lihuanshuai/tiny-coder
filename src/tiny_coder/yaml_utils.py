"""Stable YAML dumping for generated project content."""

from __future__ import annotations

import re
from typing import Any

import yaml

# 不含换行、超过该长度的字符串，需要输出为更易读的多行形式。
_YAML_STRING_FOLD_LENGTH = 72
_SENTENCE_BREAK_RE = re.compile(r"([。？！；，、!?;,])\s*")


def _break_lines(text: str, max_len: int = _YAML_STRING_FOLD_LENGTH) -> str:
    """在句末分界处将超长单行文本折为多行，每行尽量不超过 max_len。"""
    segments = _SENTENCE_BREAK_RE.split(text)
    lines: list[str] = []
    buf = ""
    index = 0
    while index < len(segments):
        part = segments[index] + (segments[index + 1] if index + 1 < len(segments) else "")
        part = part.rstrip()
        index += 2
        if not part:
            continue
        candidate = buf + part if buf else part
        if len(candidate) <= max_len:
            buf = candidate
            continue
        if buf:
            lines.append(buf)
        buf = part
    if buf:
        lines.append(buf)
    return "\n".join(lines) if lines else text


class _GameYamlDumper(yaml.SafeDumper):
    def write_folded(self, text: str) -> None:
        """输出 folded block 时不为预拆分的软换行插入空白行。"""
        hints = self.determine_block_hints(text)
        self.write_indicator(">" + hints, True)
        if hints[-1:] == "+":
            self.open_ended = True
        self.write_line_break()
        for line in text.split("\n"):
            self.write_indent()
            self._write_block_line(line)
            self.write_line_break()

    def _write_block_line(self, data: str) -> None:
        self.column += len(data)
        data_bytes = b""
        if self.encoding:
            data_bytes = data.encode(self.encoding)
        self.stream.write(data_bytes or data)


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.nodes.ScalarNode:
    if "\n" in data:
        data = data.rstrip("\n")
        style = "|"
    elif len(data) > _YAML_STRING_FOLD_LENGTH:
        data = _break_lines(data)
        style = ">"
    else:
        style = None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_GameYamlDumper.add_representer(str, _represent_str)


def dump_yaml_text(data: Any, *, sort_keys: bool = False) -> str:
    """将数据序列化为 YAML 文本（含换行用 ``|``，超长单行用 ``>``）。"""
    return yaml.dump(
        data,
        Dumper=_GameYamlDumper,
        allow_unicode=True,
        sort_keys=sort_keys,
        default_flow_style=False,
    )
