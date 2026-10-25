"""Versioned line edits with optional adaptive source-fragment positioning."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence
from difflib import SequenceMatcher
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class _SourceNotFoundError(ValueError):
    """No sufficiently close source exists; an exact result may already be present."""


class LineReplacement(BaseModel):
    """Replace old_count original lines at old_start; zero inserts before that line."""

    model_config = ConfigDict(extra="forbid")

    old_start: int = Field(ge=1, strict=True, description="First original line, 1-based.")
    old_count: int = Field(
        ge=0, strict=True, description="Number of original lines; zero inserts before old_start."
    )
    new_text: str = Field(
        description=(
            "Replacement text, without line-number prefixes. Without old_text, supply complete "
            "lines; with old_text, replace only that fragment and preserve surrounding source. "
            "Omit replacements where old_text and new_text are identical."
        )
    )
    old_text: str | None = Field(
        default=None,
        description=(
            "Source fragment for adaptive positioning. Coordinates are hints for unique matches; "
            "a repeated full-line fragment requires a matching start and line count. "
            "small character differences require a unique match. A matching full-line range may "
            "ignore whitespace or equivalent punctuation. Include context when needed. "
            "For insertion, include an existing anchor in both old_text and new_text."
        ),
    )


class AdaptiveLineReplacement(LineReplacement):
    """Source-grounded edit; matching full-line coordinates can disambiguate repeats."""

    old_start: int = Field(
        ge=1, strict=True, description="Original first line; must be accurate for repeated source."
    )
    old_count: int = Field(
        ge=0, strict=True, description="Original line count; must be accurate for repeated source."
    )
    old_text: str = Field(
        min_length=1,
        description=(
            "Required source fragment copied from the original snapshot. Coordinates are "
            "hints for unique fragments. Repeated full lines need matching coordinates and source; "
            "inline fragments must be unique. Surrounding source is preserved. Preserve context in "
            "new_text; only actual changes become line replacements."
        ),
    )


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


def _comparable(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans("“”‘’", "\"\"''"))
    return re.sub(r"\s+", "", text)


def _matching_line_hint(
    lines: list[str], old: str, hint: int, count: int
) -> tuple[int, int] | None:
    start, end = hint - 1, hint - 1 + count
    old_lines = text_lines(old)
    if count <= 0 or end > len(lines) or count != len(old_lines):
        return None
    if all(
        _comparable(actual) == _comparable(expected)
        for actual, expected in zip(lines[start:end], old_lines, strict=True)
    ):
        return start, end
    return None


def _find_fragment(lines: list[str], old: str, hint: int, count: int) -> tuple[int, int]:
    old_lines = text_lines(old)
    size = sum(bool(line.strip()) for line in old_lines)
    if not size:
        raise ValueError("old_text needs nonblank source; include an existing anchor for insertion")
    visible = [index for index, line in enumerate(lines) if line.strip()]
    candidates = [
        (visible[index], visible[index + size - 1] + 1) for index in range(len(visible) - size + 1)
    ]
    candidates.sort(key=lambda bounds: abs(bounds[0] + 1 - hint))
    fragments = {
        bounds: "".join(lines[bounds[0] : bounds[1]]).rstrip("\n") for bounds in candidates
    }
    wanted = _comparable(old)
    for exact in (True, False):
        matches = [
            bounds
            for bounds, text in fragments.items()
            if (text == old if exact else _comparable(text) == wanted)
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            hinted = _matching_line_hint(lines, old, hint, count)
            if hinted is not None and hinted in matches:
                return hinted
            positions = ", ".join(f"{start + 1}..{end}" for start, end in matches)
            raise ValueError(
                f"old_text matches multiple locations ({positions}); provide a matching full-line "
                "range and source, or include more context"
            )

    scores: list[tuple[float, tuple[int, int]]] = []
    if len(wanted) >= 24:
        wanted_lines = [_comparable(line) for line in old_lines if line.strip()]
        for bounds, text in fragments.items():
            matcher = SequenceMatcher(None, wanted, _comparable(text), autojunk=False)
            if matcher.quick_ratio() < 0.9:
                continue
            score = matcher.ratio()
            if score < 0.9:
                continue
            # Long lines must not hide shifted headings or different section boundaries.
            actual_lines = [_comparable(line) for line in text_lines(text) if line.strip()]
            line_score = (
                sum(
                    SequenceMatcher(None, left, right, autojunk=False).ratio()
                    for left, right in zip(wanted_lines, actual_lines, strict=True)
                )
                / size
            )
            score = min(score, line_score)
            if score < 0.9:
                continue
            scores.append((score, bounds))
    scores.sort(key=lambda entry: entry[0], reverse=True)
    if scores:
        score, bounds = scores[0]
        if score >= 0.94:
            if len(scores) == 1 or score - scores[1][0] >= 0.04:
                return bounds
        options = "\n".join(
            f"lines {start + 1}..{end} ({score:.3f}):\n{fragments[start, end]}"
            for score, (start, end) in scores[:3]
        )
        if score < 0.94:
            raise _SourceNotFoundError(
                "old_text differs too much from source (minimum similarity 0.940); "
                f"copy exact source from:\n{options}"
            )
        raise ValueError(
            "old_text has no unambiguous close match (multiple similarly close locations); "
            f"copy exact source from:\n{options}"
        )
    raise _SourceNotFoundError(
        "old_text not found; copy a nonempty, unique source fragment from the snapshot"
    )


def _adapt_content(old: str, actual: str, content: str) -> str:
    """Apply only old-to-new changes, preserving source differences in unchanged context."""
    if not content:
        return ""
    old_terminated = old.endswith("\n")
    old = old.removesuffix("\n")
    source = actual.removesuffix("\n")
    new = content.removesuffix("\n")
    alignment = SequenceMatcher(None, old, source, autojunk=False).get_opcodes()

    def boundary(position: int, *, start: bool) -> int:
        choices: list[int] = []
        for tag, left, right, source_left, source_right in alignment:
            if tag == "equal" and left <= position <= right:
                choices.append(source_left + position - left)
            elif position == left:
                choices.append(source_left)
            elif position == right:
                choices.append(source_right)
        if not choices:
            raise ValueError(
                "cannot align requested change safely; copy exact old_text from the snapshot"
            )
        return max(choices) if start else min(choices)

    edits: list[tuple[int, int, str]] = []
    for tag, left, right, new_left, new_right in SequenceMatcher(
        None, old, new, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        start = boundary(left, start=True)
        end = start if left == right else boundary(right, start=False)
        if start > end:
            raise ValueError(
                "cannot align requested change safely; copy exact old_text from the snapshot"
            )
        edits.append((start, end, new[new_left:new_right]))
    for start, end, text in reversed(edits):
        source = source[:start] + text + source[end:]
    if actual.endswith("\n") or (content.endswith("\n") and not old_terminated):
        source += "\n"
    return source


def resolve_line_replacements(
    original: str, replacements: Sequence[LineReplacement]
) -> list[LineReplacement]:
    """Resolve adaptive fragments to minimal ranges in the same original snapshot."""
    lines = text_lines(original)
    source = "".join(lines)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    resolved: list[LineReplacement] = []
    edits: list[tuple[int, int, str]] = []
    for index, item in enumerate(replacements, 1):
        if item.old_text == item.new_text:
            continue
        if item.old_text is None:
            resolved.append(item)
            continue
        old = "".join(text_lines(item.old_text))
        content = "".join(text_lines(item.new_text))
        try:
            fragment = old.strip("\n")
            start = source.find(fragment) if fragment.strip() else -1
            if start >= 0 and source.find(fragment, start + 1) >= 0:
                hinted = _matching_line_hint(lines, fragment, item.old_start, item.old_count)
                if hinted is None:
                    raise ValueError(
                        "old_text matches multiple locations; provide a matching full-line range "
                        "and source, or include more context"
                    )
                start, end = offsets[hinted[0]], offsets[hinted[1]]
                updated = _adapt_content(old, source[start:end], content)
            elif start >= 0:
                end = start + len(fragment)
                whole_lines = (start == 0 or source[start - 1] == "\n") and (
                    end == len(source) or source[end] == "\n"
                )
                if whole_lines:
                    end += int(end < len(source))
                    updated = _adapt_content(old, source[start:end], content)
                else:
                    # A mirrored terminal LF describes the fragment, not the untouched suffix.
                    updated = content.removesuffix("\n") if old.endswith("\n") else content
            else:
                try:
                    first, last = _find_fragment(lines, fragment, item.old_start, item.old_count)
                except _SourceNotFoundError:
                    first = item.old_start - 1
                    last = first + item.old_count
                    result = content.removesuffix("\n")
                    if (
                        result.strip()
                        and item.old_count > 0
                        and last <= len(lines)
                        and "".join(lines[first:last]).removesuffix("\n") == result
                        and source.count(result) == 1
                    ):
                        continue
                    raise
                start, end = offsets[first], offsets[last]
                updated = _adapt_content(old, source[start:end], content)
            for tag, left, right, new_left, new_right in SequenceMatcher(
                None, source[start:end], updated, autojunk=False
            ).get_opcodes():
                if tag == "equal":
                    continue
                edit_start, edit_end = start + left, start + right
                for other_start, other_end, _ in edits:
                    if (edit_start, edit_end) == (other_start, other_end) or (
                        edit_start < other_end and other_start < edit_end
                    ):
                        raise ValueError("duplicate or overlapping source changes")
                edits.append((edit_start, edit_end, updated[new_left:new_right]))
        except ValueError as error:
            raise ValueError(f"replacement {index}: {error}") from error

    updated_source = source
    for start, end, content in sorted(edits, reverse=True):
        updated_source = updated_source[:start] + content + updated_source[end:]
    replacement_lines = text_lines(updated_source)
    for tag, left, right, new_left, new_right in SequenceMatcher(
        None, lines, replacement_lines, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            resolved.append(
                LineReplacement(
                    old_start=left + 1,
                    old_count=right - left,
                    new_text="".join(replacement_lines[new_left:new_right]),
                )
            )
    return resolved


def apply_line_replacements(original: str, replacements: Sequence[LineReplacement]) -> str:
    """Validate one batch against original line numbers, then splice from bottom to top."""
    if not replacements:
        raise ValueError("line replacements must not be empty")
    lines = text_lines(original)
    edits: list[tuple[int, int, str]] = []
    for index, item in enumerate(resolve_line_replacements(original, replacements), 1):
        start = item.old_start - 1
        end = start + item.old_count
        if start > len(lines) or end < start or end > len(lines):
            raise ValueError(
                f"replacement {index}: invalid old range start={item.old_start}, "
                f"count={item.old_count}; file has {len(lines)} lines; "
                "use old_count=0 for insertion"
            )
        content = item.new_text.replace("\r\n", "\n").replace("\r", "\n")
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
