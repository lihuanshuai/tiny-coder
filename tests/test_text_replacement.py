from __future__ import annotations

import pytest

from tiny_coder.text_replacement import (
    TextReplacement,
    TextReplacementApplyError,
    apply_text_replacements,
)


def test_apply_text_replacements_updates_one_file() -> None:
    original = "# Plan\n\n- old goal\n- keep\n"
    replacements = [TextReplacement(from_text="- old goal", to_text="- new goal")]

    assert apply_text_replacements(original, replacements) == "# Plan\n\n- new goal\n- keep\n"


def test_apply_text_replacements_applies_operations_in_order() -> None:
    original = "# Plan\n- A\n- B\n"
    replacements = [
        TextReplacement(from_text="- A", to_text="- A1"),
        TextReplacement(from_text="- B", to_text="- B1"),
    ]

    assert apply_text_replacements(original, replacements) == "# Plan\n- A1\n- B1\n"


def test_apply_text_replacements_supports_insertion_by_anchor_replacement() -> None:
    original = "# Plan\n- first\n- second\n"
    replacements = [
        TextReplacement(
            from_text="- first\n- second",
            to_text="- first\n- inserted\n- second",
        )
    ]

    assert (
        apply_text_replacements(original, replacements) == "# Plan\n- first\n- inserted\n- second\n"
    )


def test_apply_text_replacements_uses_fuzzy_fallback_for_minor_typo() -> None:
    original = "# Plan\n- old goal for chapter one\n- keep\n"
    replacements = [
        TextReplacement(
            from_text="- old goal for chpater one",
            to_text="- new goal for chapter one",
        )
    ]

    assert (
        apply_text_replacements(original, replacements)
        == "# Plan\n- new goal for chapter one\n- keep\n"
    )


def test_apply_text_replacements_uses_fuzzy_fallback_for_whitespace_drift() -> None:
    original = "# Plan\n  - old goal for chapter one\n- keep\n"
    replacements = [
        TextReplacement(
            from_text="- old   goal for chapter one",
            to_text="- new goal for chapter one",
        )
    ]

    assert (
        apply_text_replacements(original, replacements)
        == "# Plan\n- new goal for chapter one\n- keep\n"
    )


def test_apply_text_replacements_ignores_formatting_for_normalized_match() -> None:
    original = (
        "# Configuration\n"
        "- **Output formats**: JSON, YAML, TOML\n"
        "\n"
        "## Validation\n"
        "Keep this section.\n"
    )
    replacements = [
        TextReplacement(
            from_text="Output formats: JSON / YAML / TOML\n# Validation",
            to_text="- **Output formats**: JSON, YAML, TOML, XML\n\n## Validation",
        )
    ]

    assert apply_text_replacements(
        original,
        replacements,
        ignore_punctuation_and_symbols=True,
    ) == (
        "# Configuration\n"
        "- **Output formats**: JSON, YAML, TOML, XML\n"
        "\n"
        "## Validation\n"
        "Keep this section.\n"
    )


def test_apply_text_replacements_prefers_normalized_exact_match() -> None:
    original = (
        "- **Output formats**: JSON, YAML, TOML\n"
        "## Validation\n"
        "First entry.\n"
        "- **Output formats**: JSON, YAML, TOML-extra\n"
        "## Validation\n"
        "Second entry.\n"
    )
    replacements = [
        TextReplacement(
            from_text="Output formats: JSON / YAML / TOML\n# Validation",
            to_text="- **Output formats**: updated\n## Validation",
        )
    ]

    assert apply_text_replacements(
        original,
        replacements,
        ignore_punctuation_and_symbols=True,
    ) == (
        "- **Output formats**: updated\n"
        "## Validation\n"
        "First entry.\n"
        "- **Output formats**: JSON, YAML, TOML-extra\n"
        "## Validation\n"
        "Second entry.\n"
    )


def test_apply_text_replacements_rejects_ambiguous_normalized_match() -> None:
    replacements = [
        TextReplacement(
            from_text="Output formats JSON YAML TOML",
            to_text="Output formats: updated",
        )
    ]

    with pytest.raises(TextReplacementApplyError, match="normalized.*multiple locations"):
        apply_text_replacements(
            "- **Output formats**: JSON, YAML, TOML\n- Output formats: JSON / YAML / TOML\n",
            replacements,
            ignore_punctuation_and_symbols=True,
        )


def test_normalized_match_rejects_duplicates_across_symbol_only_line() -> None:
    replacements = [
        TextReplacement(
            from_text="Output formats JSON YAML TOML",
            to_text="Output formats: updated",
        )
    ]

    with pytest.raises(TextReplacementApplyError, match="normalized.*multiple locations"):
        apply_text_replacements(
            "- **Output formats**: JSON, YAML, TOML\n---\n- Output formats: JSON / YAML / TOML\n",
            replacements,
            ignore_punctuation_and_symbols=True,
        )


def test_default_matching_keeps_punctuation_and_symbols_significant() -> None:
    replacements = [
        TextReplacement(
            from_text="value = alpha ++++++ beta",
            to_text="value = total",
        )
    ]

    with pytest.raises(TextReplacementApplyError, match="no safe fuzzy match"):
        apply_text_replacements("value = alpha ------ beta\n", replacements)


def test_apply_text_replacements_rejects_missing_from_text_without_safe_fuzzy_match() -> None:
    replacements = [TextReplacement(from_text="- missing", to_text="- new goal")]

    with pytest.raises(TextReplacementApplyError, match="fuzzy fallback requires"):
        apply_text_replacements("# Plan\n- old goal\n", replacements)


def test_apply_text_replacements_rejects_ambiguous_from_text() -> None:
    replacements = [TextReplacement(from_text="- duplicate", to_text="- new goal")]

    with pytest.raises(TextReplacementApplyError, match="matched 2 locations"):
        apply_text_replacements("# Plan\n- duplicate\n- duplicate\n", replacements)


def test_apply_text_replacements_rejects_ambiguous_fuzzy_match() -> None:
    replacements = [
        TextReplacement(
            from_text="- old goal for chaptr three",
            to_text="- new goal for chapter three",
        )
    ]

    with pytest.raises(TextReplacementApplyError, match="multiple similar locations"):
        apply_text_replacements(
            "# Plan\n- old goal for chapter one\n- old goal for chapter two\n",
            replacements,
        )
