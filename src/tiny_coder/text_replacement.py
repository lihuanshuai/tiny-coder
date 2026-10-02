"""Ordered text replacements with fuzzy matching."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from unicodedata import category, normalize

_FUZZY_MIN_RATIO = 0.82
_FUZZY_MIN_MARGIN = 0.05
_FUZZY_MIN_CHARS = 12


@dataclass(frozen=True, slots=True)
class TextReplacement:
    to_text: str
    from_text: str | None = None


class TextReplacementApplyError(ValueError):
    """A model-provided text replacement cannot be safely applied."""


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canonical_for_fuzzy_match(
    text: str,
    *,
    ignore_punctuation_and_symbols: bool,
) -> str:
    normalized = normalize("NFKC", _normalize_newlines(text))
    if not ignore_punctuation_and_symbols:
        return " ".join(normalized.strip().split())
    return "".join(
        char for char in normalized if not char.isspace() and category(char)[0] not in {"P", "S"}
    )


def _unique_canonical_span(
    spans: list[tuple[int, int]],
    *,
    index: int,
) -> tuple[int, int]:
    minimal_spans = [
        span
        for span in spans
        if not any(other != span and span[0] <= other[0] and other[1] <= span[1] for other in spans)
    ]
    if len(minimal_spans) > 1:
        raise TextReplacementApplyError(
            f"replacement {index} normalized from_text matched multiple locations; "
            "refusing to apply"
        )
    return minimal_spans[0]


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


def _best_fuzzy_span(
    text: str,
    from_text: str,
    index: int,
    *,
    ignore_punctuation_and_symbols: bool,
) -> tuple[int, int]:
    query = _canonical_for_fuzzy_match(
        from_text,
        ignore_punctuation_and_symbols=ignore_punctuation_and_symbols,
    )
    if len(query) < _FUZZY_MIN_CHARS:
        raise TextReplacementApplyError(
            f"replacement {index} from_text was not found; fuzzy fallback requires at least "
            f"{_FUZZY_MIN_CHARS} comparable characters"
        )

    candidates: list[tuple[int, int, str]] = []
    canonical_spans: list[tuple[int, int]] = []
    scored: list[tuple[float, int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, end, candidate in _line_windows(text, from_text):
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        candidate_text = _canonical_for_fuzzy_match(
            candidate,
            ignore_punctuation_and_symbols=ignore_punctuation_and_symbols,
        )
        if not candidate_text:
            continue
        candidates.append((start, end, candidate_text))
        if candidate_text == query:
            canonical_spans.append(span)

    if canonical_spans:
        return _unique_canonical_span(canonical_spans, index=index)

    for start, end, candidate_text in candidates:
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


def _first_content_indent(text: str) -> int:
    for line in _normalize_newlines(text).splitlines():
        if line.strip():
            return len(line) - len(line.lstrip(" "))
    return 0


def _inherit_fuzzy_match_indentation(
    replacement_text: str,
    *,
    from_text: str,
    matched_text: str,
) -> str:
    """Keep a fuzzy replacement at the indentation level of its matched source."""
    from_indent = _first_content_indent(from_text)
    if _first_content_indent(replacement_text) != from_indent:
        return replacement_text
    indent_delta = _first_content_indent(matched_text) - from_indent
    if not indent_delta:
        return replacement_text

    lines = replacement_text.splitlines(keepends=True)
    if indent_delta > 0:
        return "".join(" " * indent_delta + line if line.strip() else line for line in lines)
    return "".join(
        line[min(-indent_delta, len(line) - len(line.lstrip(" "))) :] if line.strip() else line
        for line in lines
    )


def resolve_text_replacement(
    original: str,
    replacement: TextReplacement,
    index: int,
    *,
    ignore_punctuation_and_symbols: bool = False,
) -> tuple[int, int, str]:
    """Locate a replacement in LF-normalized text and adjust its indentation."""
    original = _normalize_newlines(original)
    to_text = _normalize_newlines(replacement.to_text)
    if replacement.from_text is None:
        return 0, len(original), to_text

    from_text = _normalize_newlines(replacement.from_text)
    if not from_text:
        raise TextReplacementApplyError(f"replacement {index} from_text must not be empty")
    if from_text == to_text:
        raise TextReplacementApplyError(f"replacement {index} does not change text")
    count = original.count(from_text)
    if count == 0:
        start, end = _best_fuzzy_span(
            original,
            from_text,
            index,
            ignore_punctuation_and_symbols=ignore_punctuation_and_symbols,
        )
        to_text = _inherit_fuzzy_match_indentation(
            to_text,
            from_text=from_text,
            matched_text=original[start:end],
        )
        if original[start:end].endswith("\n") and not to_text.endswith("\n"):
            to_text += "\n"
    elif count > 1:
        raise TextReplacementApplyError(
            f"replacement {index} from_text matched {count} locations; refusing to apply"
        )
    else:
        start = original.index(from_text)
        if original.find(from_text, start + 1) != -1:
            raise TextReplacementApplyError(
                f"replacement {index} from_text matched multiple locations; refusing to apply"
            )
        end = start + len(from_text)
    return start, end, to_text


def apply_text_replacements(
    original: str,
    replacements: Sequence[TextReplacement],
    *,
    ignore_punctuation_and_symbols: bool = False,
) -> str:
    """Apply single-hit replacements, optionally ignoring formatting during fallback matching."""
    if not replacements:
        raise TextReplacementApplyError("replacement list must not be empty")
    updated = _normalize_newlines(original)
    for index, replacement in enumerate(replacements, start=1):
        if replacement.from_text is None:
            updated = _normalize_newlines(replacement.to_text)
            continue
        start, end, to_text = resolve_text_replacement(
            updated,
            replacement,
            index,
            ignore_punctuation_and_symbols=ignore_punctuation_and_symbols,
        )
        updated = updated[:start] + to_text + updated[end:]
        updated = updated.rstrip("\n") + "\n" if updated else ""
    return updated


__all__ = [
    "TextReplacement",
    "TextReplacementApplyError",
    "apply_text_replacements",
    "resolve_text_replacement",
]
