import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from tiny_coder.apply_patch import (
    ApplyPatch,
    ApplyPatchOutput,
    apply_patch_output,
    apply_patches,
    preview_apply_patches,
)
from tiny_coder.text_replacement import TextReplacement, TextReplacementApplyError


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    root = Path(".test-tmp") / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_apply_patch_requires_replacements() -> None:
    with pytest.raises(ValueError, match="at least one replacement"):
        ApplyPatch(path=Path("notes.txt"), replacements=[])


def test_apply_patch_applies_ordered_whole_file_and_matched_replacements() -> None:
    patch = ApplyPatch(
        path=Path("notes.txt"),
        replacements=[
            TextReplacement(from_text="discarded", to_text="intermediate"),
            TextReplacement(to_text="first\nsecond\n"),
            TextReplacement(from_text="second", to_text="finished"),
            TextReplacement(to_text="final"),
        ],
    )

    assert patch.apply("discarded\n") == "final"


def test_apply_patch_matches_all_independent_edits_in_original_snapshot() -> None:
    patch = ApplyPatch(
        path=Path("notes.txt"),
        match_original=True,
        replacements=[
            TextReplacement(from_text="first", to_text="second"),
            TextReplacement(from_text="second", to_text="finished with a longer line"),
        ],
    )

    assert patch.apply("first\r\nsecond\r\n") == "second\nfinished with a longer line\n"


@pytest.mark.parametrize(
    ("original", "replacements", "reason"),
    [
        (
            "first second\n",
            [
                TextReplacement(from_text="first second", to_text="updated"),
                TextReplacement(from_text="second", to_text="finished"),
            ],
            "overlaps replacement",
        ),
        (
            "first\n",
            [
                TextReplacement(from_text="first", to_text="draft"),
                TextReplacement(from_text="draft", to_text="final"),
            ],
            "from_text was not found",
        ),
        (
            "- old goal for chapter one\n",
            [
                TextReplacement(from_text="- old goal for chpater one", to_text="- new goal A"),
                TextReplacement(from_text="- old goal for chaptor one", to_text="- new goal B"),
            ],
            "overlaps replacement",
        ),
        (
            "first first\n",
            [TextReplacement(from_text="first", to_text="updated")],
            "matched .*locations",
        ),
        (
            "aaaa\n",
            [TextReplacement(from_text="aaa", to_text="updated")],
            "matched multiple locations",
        ),
        (
            "first\n",
            [TextReplacement(to_text="draft"), TextReplacement(from_text="first", to_text="final")],
            "whole-file replacement cannot",
        ),
    ],
)
def test_snapshot_edits_are_rejected_before_any_file_is_written(
    workspace_tmp_path: Path,
    original: str,
    replacements: list[TextReplacement],
    reason: str,
) -> None:
    target = workspace_tmp_path / "notes.txt"
    target.write_text(original, encoding="utf-8", newline="\n")
    created = workspace_tmp_path / "created.txt"
    patches = [
        ApplyPatch(path=created, replacements=[TextReplacement(to_text="new file")]),
        ApplyPatch(path=target, replacements=replacements, match_original=True),
    ]

    with pytest.raises(TextReplacementApplyError, match=reason):
        apply_patches(workspace_tmp_path, patches)

    assert target.read_text(encoding="utf-8") == original
    assert not created.exists()


def test_snapshot_edits_allow_adjacent_deletions_and_whole_file_creation() -> None:
    patch = ApplyPatch(
        path=Path("notes.txt"),
        match_original=True,
        replacements=[
            TextReplacement(from_text="first", to_text=""),
            TextReplacement(from_text="second", to_text=""),
        ],
    )
    assert patch.apply("firstsecond") == ""
    assert (
        ApplyPatch(
            path=Path("new.txt"),
            match_original=True,
            replacements=[TextReplacement(to_text="whole file")],
        ).apply(None)
        == "whole file"
    )


def test_preview_apply_patches_combines_operations_without_writing(
    workspace_tmp_path: Path,
) -> None:
    target = workspace_tmp_path / "notes.txt"
    target.write_text("first\nsecond\n", encoding="utf-8", newline="\n")
    patches = [
        ApplyPatch(
            path=Path("notes.txt"),
            replacements=[TextReplacement(from_text="first", to_text="updated")],
        ),
        ApplyPatch(
            path=Path("notes.txt"),
            replacements=[TextReplacement(from_text="second", to_text="finished")],
        ),
    ]

    assert preview_apply_patches(workspace_tmp_path, patches) == {
        target.resolve(): "updated\nfinished\n"
    }
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


def test_apply_patches_validates_every_target_before_writing(
    workspace_tmp_path: Path,
) -> None:
    allowed = workspace_tmp_path / "allowed.txt"
    patches = [
        ApplyPatch(
            path=allowed,
            replacements=[TextReplacement(to_text="written only after validation\n")],
        ),
        ApplyPatch(
            path=Path("blocked.txt"),
            replacements=[TextReplacement(to_text="blocked\n")],
        ),
    ]

    with pytest.raises(ValueError, match="output file is not allowed"):
        apply_patches(workspace_tmp_path, patches, allowed_paths=[allowed])

    assert not allowed.exists()


def test_apply_patches_creates_file_from_whole_file_replacement(
    workspace_tmp_path: Path,
) -> None:
    target = workspace_tmp_path / "created.md"

    written = apply_patches(
        workspace_tmp_path,
        [
            ApplyPatch(
                path=Path("created.md"),
                replacements=[TextReplacement(to_text="# Created\n")],
            )
        ],
    )

    assert written == [target.resolve()]
    assert target.read_text(encoding="utf-8") == "# Created\n"


def test_apply_patches_skips_equivalent_replacement(workspace_tmp_path: Path) -> None:
    target = workspace_tmp_path / "notes.txt"
    target.write_text("same\r\ntext\r\n", encoding="utf-8", newline="")

    written = apply_patches(
        workspace_tmp_path,
        [
            ApplyPatch(
                path=target,
                replacements=[TextReplacement(from_text="same\r\ntext", to_text="same\ntext")],
            )
        ],
    )

    assert written == []
    assert target.read_bytes() == b"same\r\ntext\r\n"


class _PatchOutput(BaseModel, ApplyPatchOutput):
    content: str

    def to_apply_patches(self) -> list[ApplyPatch]:
        return [
            ApplyPatch(
                path=Path("note.md"),
                replacements=[TextReplacement(to_text=self.content)],
            )
        ]


def test_apply_patch_output_writes_converted_patches(workspace_tmp_path: Path) -> None:
    target = workspace_tmp_path / "note.md"

    written = apply_patch_output(workspace_tmp_path, _PatchOutput(content="# Note\n"))

    assert written == [target.resolve()]
    assert target.read_text(encoding="utf-8") == "# Note\n"
