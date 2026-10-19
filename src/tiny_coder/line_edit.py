"""Snapshot-relative, inclusive line replacements without source-text matching."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class LineReplacement(BaseModel):
    """Replace start_line..end_line; end_line=start_line-1 inserts before start_line."""

    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(ge=1, strict=True, description="First original line, 1-based.")
    end_line: int = Field(
        ge=0, strict=True, description="Last original line, inclusive; start_line-1 inserts."
    )
    content: str = Field(description="Complete replacement lines, without line-number prefixes.")


def text_revision(content: str) -> str:
    """Fingerprint the exact source, including its newline convention."""
    return sha256(content.encode("utf-8")).hexdigest()


def text_lines(content: str) -> list[str]:
    """Split on LF/CRLF/CR, preserving blank lines and the final newline."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    parts = normalized.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def numbered_text(content: str, *, start_line: int = 1, end_line: int | None = None) -> str:
    """Render real source line numbers, including blank lines, for reads and prompts."""
    lines = text_lines(content)
    end = len(lines) if end_line is None else end_line
    if start_line < 1 or end < start_line - 1 or end > len(lines):
        raise ValueError(f"invalid read range {start_line}..{end}; file has {len(lines)} lines")
    return "\n".join(
        f"{index + 1}: {lines[index].removesuffix(chr(10))}" for index in range(start_line - 1, end)
    )


def grep_lines(content: str, pattern: str | re.Pattern[str]) -> Iterator[tuple[int, str]]:
    """Yield regex-matching lines with the same 1-based numbers as numbered_text."""
    query = re.compile(pattern)
    for number, line in enumerate(text_lines(content), 1):
        if query.search(line):
            yield number, line.removesuffix("\n")


def apply_line_replacements(original: str, replacements: Sequence[LineReplacement]) -> str:
    """Validate one batch against original line numbers, then splice from bottom to top."""
    if not replacements:
        raise ValueError("line replacements must not be empty")
    lines = text_lines(original)
    edits: list[tuple[int, int, str]] = []
    for index, item in enumerate(replacements, 1):
        start, end = item.start_line - 1, item.end_line
        if start > len(lines) or end < start or end > len(lines):
            raise ValueError(
                f"replacement {index}: invalid line range {item.start_line}..{item.end_line}; "
                f"file has {len(lines)} lines; use end_line=start_line-1 only for insertion"
            )
        content = item.content.replace("\r\n", "\n").replace("\r", "\n")
        if (
            content
            and not content.endswith("\n")
            and (end < len(lines) or (end > start and lines[end - 1].endswith("\n")))
        ):
            content += "\n"
        for other_start, other_end, _ in edits:
            if (start, end) == (other_start, other_end) or (
                start < other_end and other_start < end
            ):
                raise ValueError(f"replacement {index}: duplicate or overlapping line ranges")
        edits.append((start, end, content))

    for start, end, content in sorted(edits, reverse=True):
        lines[start:end] = text_lines(content)
    return "".join(
        line if line.endswith("\n") or index == len(lines) - 1 else line + "\n"
        for index, line in enumerate(lines)
    )
