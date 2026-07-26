from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

_FUZZY_MIN_RATIO = 0.82
_FUZZY_MIN_MARGIN = 0.05
_FUZZY_MIN_CHARS = 12


class _TextReplacementLike(Protocol):
    @property
    def from_text(self) -> str: ...

    @property
    def to_text(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TextReplacement:
    from_text: str
    to_text: str


@dataclass(frozen=True, slots=True)
class TextReplacementFilePatch:
    replacements: Sequence[TextReplacement]
    report: str | None = None


class TextReplacementApplyError(ValueError):
    """A model-provided text replacement cannot be safely applied."""


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canonical_for_fuzzy_match(text: str) -> str:
    return " ".join(_normalize_newlines(text).strip().split())


def _line_windows(text: str, from_text: str) -> Iterator[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    from_line_count = max(1, len(_normalize_newlines(from_text).splitlines()))
    window_lengths = {
        length
        for length in (from_line_count - 1, from_line_count, from_line_count + 1)
        if length > 0
    }
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    offsets.append(offset)

    for window_length in sorted(window_lengths):
        if window_length > len(lines):
            continue
        for start_line in range(0, len(lines) - window_length + 1):
            end_line = start_line + window_length
            start = offsets[start_line]
            end = offsets[end_line]
            yield start, end, text[start:end]


def _best_fuzzy_span(text: str, from_text: str, index: int) -> tuple[int, int]:
    query = _canonical_for_fuzzy_match(from_text)
    if len(query) < _FUZZY_MIN_CHARS:
        raise TextReplacementApplyError(
            f"replacement {index} from_text was not found; fuzzy fallback requires at least "
            f"{_FUZZY_MIN_CHARS} non-whitespace characters"
        )

    scored: list[tuple[float, int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, end, candidate in _line_windows(text, from_text):
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        candidate_text = _canonical_for_fuzzy_match(candidate)
        if not candidate_text:
            continue
        ratio = SequenceMatcher(None, query, candidate_text).ratio()
        if ratio >= _FUZZY_MIN_RATIO:
            scored.append((ratio, start, end))

    if not scored:
        raise TextReplacementApplyError(
            f"replacement {index} from_text was not found; no safe fuzzy match reached "
            f"{_FUZZY_MIN_RATIO:.2f} similarity"
        )

    scored.sort(reverse=True)
    best_ratio, best_start, best_end = scored[0]
    if len(scored) > 1 and best_ratio - scored[1][0] < _FUZZY_MIN_MARGIN:
        raise TextReplacementApplyError(
            f"replacement {index} fuzzy from_text matched multiple similar locations; "
            "refusing to apply"
        )
    return best_start, best_end


def apply_text_replacements(
    original: str,
    replacements: Sequence[_TextReplacementLike],
) -> str:
    """Apply single-hit text replacements in order and return updated LF text."""
    if not replacements:
        raise TextReplacementApplyError("replacement list must not be empty")
    updated = _normalize_newlines(original)
    for index, replacement in enumerate(replacements, start=1):
        from_text = _normalize_newlines(replacement.from_text)
        to_text = _normalize_newlines(replacement.to_text)
        if not from_text:
            raise TextReplacementApplyError(f"replacement {index} from_text must not be empty")
        if from_text == to_text:
            raise TextReplacementApplyError(f"replacement {index} does not change text")
        count = updated.count(from_text)
        if count == 0:
            start, end = _best_fuzzy_span(updated, from_text, index)
            replacement_text = to_text
            if updated[start:end].endswith("\n") and not replacement_text.endswith("\n"):
                replacement_text += "\n"
            updated = updated[:start] + replacement_text + updated[end:]
            continue
        if count > 1:
            raise TextReplacementApplyError(
                f"replacement {index} from_text matched {count} locations; refusing to apply"
            )
        updated = updated.replace(from_text, to_text, 1)
    return updated.rstrip("\n") + "\n"


__all__ = [
    "TextReplacement",
    "TextReplacementApplyError",
    "TextReplacementFilePatch",
    "apply_text_replacements",
]
