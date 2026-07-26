from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# 公共 JSON 字符串流式解析常量（供本模块及其他解析器复用）。
JSON_VALUE_WHITESPACE = {" ", "\t", "\r", "\n"}
JSON_ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class JsonProtocolError(ValueError):
    """JSON 协议结构不符合预期时抛出。"""


_JSON_CODE_FENCE_RE = re.compile(r"\A```[^\r\n]*\r?\n(?P<body>.*?)(?:\r?\n)?```\s*\Z", re.DOTALL)


def _strip_json_code_fence(raw: str) -> str:
    """兼容模型将 JSON 包在 Markdown fenced code block 内的输出。"""
    text = str(raw or "").strip()
    match = _JSON_CODE_FENCE_RE.match(text)
    if match is None:
        return text
    return match.group("body").strip()


def _json_object_candidate(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    stack: list[str] = []
    candidate: list[str] = []
    in_string = False
    in_escape = False

    for ch in text[start:]:
        if in_string:
            if in_escape:
                candidate.append(ch)
                in_escape = False
                continue
            if ch == "\\":
                candidate.append(ch)
                in_escape = True
                continue
            if ch == '"':
                candidate.append(ch)
                in_string = False
                continue
            if ch == "\n":
                candidate.append("\\n")
                continue
            if ch == "\r":
                candidate.append("\\r")
                continue
            if ch == "\t":
                candidate.append("\\t")
                continue
            if ord(ch) < 0x20:
                candidate.append(json.dumps(ch)[1:-1])
                continue
            candidate.append(ch)
            continue

        candidate.append(ch)
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch not in "}]":
            continue
        if not stack:
            return None
        opener = stack.pop()
        if (opener, ch) not in {("{", "}"), ("[", "]")}:
            return None
        if not stack:
            return "".join(candidate)

    if not stack:
        return "".join(candidate)
    if in_escape:
        candidate.append("\\")
    if in_string:
        candidate.append('"')
    for opener in reversed(stack):
        candidate.append("}" if opener == "{" else "]")
    return "".join(candidate)


def _loads_json_object_lenient(text: str) -> dict[str, Any]:
    candidate = _json_object_candidate(text)
    if candidate is None:
        raise JsonProtocolError("未找到 JSON 对象")
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise JsonProtocolError("根节点必须是 JSON 对象")
    return payload


def load_json_object(raw: str, *, legacy_markers: tuple[str, ...] = ()) -> dict[str, Any]:
    """将文本解析为 JSON 对象并做基础协议校验。"""
    text = _strip_json_code_fence(raw)
    if not text:
        raise JsonProtocolError("输出为空，须返回 JSON 对象")
    if legacy_markers and any(marker in text for marker in legacy_markers):
        raise JsonProtocolError("检测到不受支持的旧协议标记")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        try:
            return _loads_json_object_lenient(text)
        except (JsonProtocolError, json.JSONDecodeError) as repair_error:
            raise JsonProtocolError(f"JSON 解析失败: {e.msg}; 宽容修复失败: {repair_error}") from e
    if not isinstance(payload, dict):
        raise JsonProtocolError("根节点必须是 JSON 对象")
    return payload


@dataclass
class JsonStringFieldStreamer:
    """增量提取 JSON 对象中某个字符串字段的内容。"""

    field_key: str
    _buffer: str = field(init=False, default="")
    _search_start: int = field(init=False, default=0)
    _started: bool = field(init=False, default=False)
    _finished: bool = field(init=False, default=False)
    _in_escape: bool = field(init=False, default=False)
    _streamed: str = field(init=False, default="")

    @property
    def streamed_text(self) -> str:
        return self._streamed

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        printed = ""
        while True:
            if not self._started:
                key_idx = self._buffer.find(f'"{self.field_key}"', self._search_start)
                if key_idx < 0:
                    if len(self._buffer) > 256:
                        self._buffer = self._buffer[-256:]
                    self._search_start = max(0, len(self._buffer) - 32)
                    break
                colon_idx = self._buffer.find(":", key_idx + len(self.field_key) + 2)
                if colon_idx < 0:
                    self._search_start = key_idx
                    break
                value_quote_idx = colon_idx + 1
                while (
                    value_quote_idx < len(self._buffer)
                    and self._buffer[value_quote_idx] in JSON_VALUE_WHITESPACE
                ):
                    value_quote_idx += 1
                if value_quote_idx >= len(self._buffer):
                    break
                if self._buffer[value_quote_idx] != '"':
                    self._finished = True
                    break
                self._started = True
                self._search_start = value_quote_idx + 1
                continue

            if self._finished:
                break

            i = self._search_start
            while i < len(self._buffer):
                ch = self._buffer[i]
                if self._in_escape:
                    self._in_escape = False
                    if ch == "u":
                        if i + 4 >= len(self._buffer):
                            self._search_start = i - 1
                            return printed
                        hex_digits = self._buffer[i + 1 : i + 5]
                        try:
                            decoded = chr(int(hex_digits, 16))
                        except ValueError:
                            decoded = ""
                        if decoded:
                            printed += decoded
                            self._streamed += decoded
                        i += 5
                        continue
                    decoded = JSON_ESCAPE_MAP.get(ch, ch)
                    printed += decoded
                    self._streamed += decoded
                    i += 1
                    continue
                if ch == "\\":
                    self._in_escape = True
                    i += 1
                    continue
                if ch == '"':
                    self._finished = True
                    self._search_start = i + 1
                    return printed
                printed += ch
                self._streamed += ch
                i += 1
            self._search_start = i
            break
        return printed
