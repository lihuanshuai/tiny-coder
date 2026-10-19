from __future__ import annotations

import re

import pytest

from tiny_coder.line_edit import LineReplacement, apply_line_replacements, grep_lines, numbered_text


def test_edits_use_original_lines_despite_length_changes_and_duplicate_text() -> None:
    source = "heading\nsame\n\nsame\ntail\n"
    updated = apply_line_replacements(
        source,
        [
            LineReplacement(start_line=2, end_line=2, content="first\nextra"),
            LineReplacement(start_line=4, end_line=4, content="second"),
        ],
    )
    assert updated == "heading\nfirst\nextra\n\nsecond\ntail\n"


@pytest.mark.parametrize(
    ("source", "start", "end", "content", "expected"),
    [
        ("a\nb\nc\n", 2, 2, "", "a\nc\n"),
        ("a\nb", 2, 1, "new", "a\nnew\nb"),
        ("a", 2, 1, "new", "a\nnew"),
        ("a\n", 2, 1, "new\n", "a\nnew\n"),
        ("", 1, 0, "new", "new"),
        ("a", 1, 1, "new", "new"),
        ("a\n", 1, 1, "new", "new\n"),
        ("a\nb", 1, 2, "", ""),
        ("a\r\n\r\nb\r\n", 3, 3, "c", "a\n\nc\n"),
        ("a\n  - old\nend\n", 2, 2, "  - new\n    - child", "a\n  - new\n    - child\nend\n"),
    ],
)
def test_line_boundaries(source: str, start: int, end: int, content: str, expected: str) -> None:
    assert (
        apply_line_replacements(
            source, [LineReplacement(start_line=start, end_line=end, content=content)]
        )
        == expected
    )


def test_append_and_replace_last_line_without_final_newline() -> None:
    assert (
        apply_line_replacements(
            "old",
            [
                LineReplacement(start_line=2, end_line=1, content="appended"),
                LineReplacement(start_line=1, end_line=1, content="updated"),
            ],
        )
        == "updated\nappended"
    )


@pytest.mark.parametrize("start,end", [(5, 5), (1, 5), (3, 1)])
def test_out_of_bounds_ranges_are_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="file has 3 lines"):
        apply_line_replacements(
            "a\nb\nc\n", [LineReplacement(start_line=start, end_line=end, content="x")]
        )


@pytest.mark.parametrize("start,end", [(1, 2), (2, 3), (2, 1)])
def test_duplicate_overlapping_and_interior_insertions_are_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="overlapping"):
        apply_line_replacements(
            "a\nb\nc\n",
            [
                LineReplacement(start_line=1, end_line=2, content="first"),
                LineReplacement(start_line=start, end_line=end, content="second"),
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
