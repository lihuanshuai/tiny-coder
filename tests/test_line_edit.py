from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from tiny_coder.line_edit import (
    LineReplacement,
    apply_line_replacements,
    grep_lines,
    numbered_text,
    resolve_line_replacements,
)


def _adaptive(old: str, new: str, *, hint: int = 999) -> LineReplacement:
    return LineReplacement(old_start=hint, old_count=11, old_text=old, new_text=new)


def test_adaptive_edit_uses_source_instead_of_incorrect_line_numbers() -> None:
    source = "# Header\n\noriginal detail\n\n# Footer\n"
    edits = [_adaptive("# Header\n\noriginal detail", "# Header\n\nupdated detail")]
    resolved = resolve_line_replacements(source, edits)
    assert [(item.old_start, item.old_count, item.new_text) for item in resolved] == [
        (3, 1, "updated detail\n")
    ]
    assert apply_line_replacements(source, edits) == source.replace("original", "updated")


def test_adaptive_formatting_errors_do_not_rewrite_unchanged_source() -> None:
    source = '- 项目:待确认 "线索"\n'
    old = '  - 项目：待确认 “线索”'
    new = old.replace("待确认", "已确认")
    assert apply_line_replacements(source, [_adaptive(old, new)]) == source.replace(
        "待确认", "已确认"
    )


def test_adaptive_typo_in_locator_preserves_actual_unchanged_text() -> None:
    source = "The investigator checked the original archive before leaving.\n"
    old = source.replace("original", "orignal").rstrip("\n")
    new = old.replace("before leaving", "and recorded the result")
    assert apply_line_replacements(source, [_adaptive(old, new)]) == source.replace(
        "before leaving", "and recorded the result"
    )
    assert apply_line_replacements(source, [_adaptive(old, old)]) == source


def test_adaptive_unique_close_match_allows_more_than_two_missing_characters() -> None:
    source = "The investigator checked the original city archive before leaving.\n"
    old = source.replace("city ", "").rstrip("\n")
    new = old.replace("before leaving", "and recorded the result")
    assert apply_line_replacements(source, [_adaptive(old, new)]) == source.replace(
        "before leaving", "and recorded the result"
    )


def test_adaptive_low_similarity_reports_difference_and_verbatim_candidate() -> None:
    source = "The investigator checked archive A before leaving the building.\n"
    old = source.replace("archive A", "archives ABCDEFGH").rstrip("\n")
    with pytest.raises(ValueError, match="differs too much") as error:
        apply_line_replacements(source, [_adaptive(old, "updated")])
    assert "lines 1..1" in str(error.value)
    assert source.rstrip("\n") in str(error.value)


@pytest.mark.parametrize("source,old", [("detail", "detail\n"), ("one\n\ntwo\n", "one\ntwo")])
def test_adaptive_unchanged_context_keeps_actual_line_endings_and_blank_lines(
    source: str, old: str
) -> None:
    assert apply_line_replacements(source, [_adaptive(old, old)]) == source


def test_identical_old_and_new_are_ignored_before_source_matching() -> None:
    source = "same\nsame\noriginal detail\n"
    unchanged = _adaptive("same", "same")
    assert resolve_line_replacements(source, [unchanged]) == []
    assert (
        apply_line_replacements(source, [unchanged, _adaptive("original detail", "updated detail")])
        == "same\nsame\nupdated detail\n"
    )


@pytest.mark.parametrize(
    "source,old",
    [
        ("TODO: expand\nTODO: expand\n", "TODO: expand"),
        ("- 项目：待核对\n- 项目:待核对\n", " - 项目： 待核对 "),
        (
            "The investigator checked archive A before leaving the building.\n"
            "The investigator checked archive B before leaving the building.\n",
            "The investigator checked archive C before leaving the building.",
        ),
        (
            "The investigator checked the original city archive before leaving.\n"
            "The investigator checked the original town archive before leaving.\n",
            "The investigator checked the original archive before leaving.",
        ),
    ],
)
def test_adaptive_edit_refuses_ambiguous_sources_even_with_close_line_hint(
    source: str, old: str
) -> None:
    with pytest.raises(ValueError, match="multiple locations|unambiguous"):
        apply_line_replacements(source, [_adaptive(old, "updated", hint=1)])


@pytest.mark.parametrize("old", ["", " \n ", "TODX", "completely unrelated source fragment"])
def test_adaptive_edit_refuses_insufficient_or_unmatched_source(old: str) -> None:
    with pytest.raises(ValueError, match="source|not found"):
        apply_line_replacements("TODO\n", [_adaptive(old, "updated")])


def test_adaptive_insert_delete_and_disjoint_edits_share_original_snapshot() -> None:
    source = "heading\nold\n\ntail\n"
    edits = [
        _adaptive("heading\nold", "heading\nnew"),
        _adaptive("tail", "inserted\ntail"),
    ]
    assert apply_line_replacements(source, edits) == "heading\nnew\n\ninserted\ntail\n"
    assert apply_line_replacements(source, [_adaptive("old", "")]) == "heading\n\ntail\n"


def test_adaptive_edits_with_shared_context_resolve_to_disjoint_minimal_ranges() -> None:
    source = "first\n\nsecond\n"
    edits = [
        _adaptive(source, source.replace("first", "A")),
        _adaptive(source, source.replace("second", "B")),
    ]
    assert apply_line_replacements(source, edits) == "A\n\nB\n"
    with pytest.raises(ValueError, match="overlapping"):
        apply_line_replacements(source, [edits[0], _adaptive("first", "other")])


def test_edits_use_original_lines_despite_length_changes_and_duplicate_text() -> None:
    source = "heading\nsame\n\nsame\ntail\n"
    updated = apply_line_replacements(
        source,
        [
            LineReplacement(old_start=2, old_count=1, new_text="first\nextra"),
            LineReplacement(old_start=4, old_count=1, new_text="second"),
        ],
    )
    assert updated == "heading\nfirst\nextra\n\nsecond\ntail\n"


@pytest.mark.parametrize(
    ("source", "start", "count", "content", "expected"),
    [
        ("a\nb\nc\n", 2, 1, "", "a\nc\n"),
        ("a\nb", 2, 0, "new", "a\nnew\nb"),
        ("a", 2, 0, "new", "a\nnew"),
        ("a\n", 2, 0, "new\n", "a\nnew\n"),
        ("", 1, 0, "new", "new"),
        ("a", 1, 1, "new", "new"),
        ("a\n", 1, 1, "new", "new\n"),
        ("a\nb", 1, 2, "", ""),
        ("a\r\n\r\nb\r\n", 3, 1, "c", "a\n\nc\n"),
        ("a\n  - old\nend\n", 2, 1, "  - new\n    - child", "a\n  - new\n    - child\nend\n"),
    ],
)
def test_line_boundaries(source: str, start: int, count: int, content: str, expected: str) -> None:
    assert (
        apply_line_replacements(
            source, [LineReplacement(old_start=start, old_count=count, new_text=content)]
        )
        == expected
    )


def test_append_and_replace_last_line_without_final_newline() -> None:
    assert (
        apply_line_replacements(
            "old",
            [
                LineReplacement(old_start=2, old_count=0, new_text="appended"),
                LineReplacement(old_start=1, old_count=1, new_text="updated"),
            ],
        )
        == "updated\nappended"
    )


@pytest.mark.parametrize("start,count", [(5, 1), (1, 5), (3, 2)])
def test_out_of_bounds_ranges_are_rejected(start: int, count: int) -> None:
    with pytest.raises(ValueError, match="file has 3 lines"):
        apply_line_replacements(
            "a\nb\nc\n", [LineReplacement(old_start=start, old_count=count, new_text="x")]
        )


@pytest.mark.parametrize("count", [-1, 1.5, True, "1"])
def test_old_count_requires_a_nonnegative_integer(count: int | float | str) -> None:
    with pytest.raises(ValidationError, match="old_count"):
        LineReplacement.model_validate({"old_start": 1, "old_count": count, "new_text": "x"})


def test_legacy_end_line_is_not_interpreted_as_a_count() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LineReplacement.model_validate({"start_line": 2, "end_line": 3, "content": "x"})


@pytest.mark.parametrize("start,count", [(1, 2), (2, 2), (2, 0)])
def test_duplicate_overlapping_and_interior_insertions_are_rejected(start: int, count: int) -> None:
    with pytest.raises(ValueError, match="overlapping"):
        apply_line_replacements(
            "a\nb\nc\n",
            [
                LineReplacement(old_start=1, old_count=2, new_text="first"),
                LineReplacement(old_start=start, old_count=count, new_text="second"),
            ],
        )


def test_numbered_reads_preserve_blank_lines_and_unicode_line_separators() -> None:
    assert numbered_text("a\r\n\r\nx\u2028y\n", start_line=2, end_line=3) == "2: \n3: x\u2028y"
    assert numbered_text("") == ""
    with pytest.raises(ValueError, match="file has 1 lines"):
        numbered_text("a", start_line=3)


@pytest.mark.parametrize("pattern", [r"(?i)^## |\bTODO\b", re.compile(r"^## |\bTODO\b", re.I)])
def test_grep_lines_locates_keywords_with_original_numbers(pattern: str | re.Pattern[str]) -> None:
    source = "# Title\r\n\r\n## Details\r\n- todo: expand\r\n\r\n## Details\r\n"
    assert list(grep_lines(source, pattern)) == [
        (3, "## Details"),
        (4, "- todo: expand"),
        (6, "## Details"),
    ]
    assert numbered_text(source, start_line=4, end_line=4) == "4: - todo: expand"
    assert list(grep_lines(source, "missing")) == []
    assert list(grep_lines("", pattern)) == []
