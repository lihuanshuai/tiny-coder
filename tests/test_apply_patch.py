from __future__ import annotations

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
from tiny_coder.line_edit import LineReplacement, text_revision


def test_preview_and_write_line_patch(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("same\nsame\n", encoding="utf-8", newline="\n")
    patch = ApplyPatch(
        Path("notes.md"),
        replacements=[LineReplacement(old_start=2, old_count=1, new_text="updated")],
        expected_revision=text_revision("same\nsame\n"),
    )
    assert preview_apply_patches(tmp_path, [patch]) == {target: "same\nupdated\n"}
    assert target.read_text(encoding="utf-8") == "same\nsame\n"
    assert apply_patches(tmp_path, [patch]) == [target]
    assert target.read_text(encoding="utf-8") == "same\nupdated\n"


def test_stale_second_patch_prevents_all_writes(tmp_path: Path) -> None:
    target = tmp_path / "existing.md"
    target.write_text("external change\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="source revision changed"):
        apply_patches(
            tmp_path,
            [
                ApplyPatch(Path("new.md"), content="new\n"),
                ApplyPatch(
                    Path("existing.md"),
                    replacements=[LineReplacement(old_start=1, old_count=1, new_text="updated")],
                    expected_revision=text_revision("old\n"),
                ),
            ],
        )
    assert not (tmp_path / "new.md").exists()
    assert target.read_text(encoding="utf-8") == "external change\n"


@pytest.mark.parametrize("paths", [None, []])
def test_workspace_and_explicit_empty_allowlist(tmp_path: Path, paths: list[Path] | None) -> None:
    with pytest.raises(ValueError, match="outside|not allowed"):
        apply_patches(
            tmp_path, [ApplyPatch(Path("../escape.md"), content="x")], allowed_paths=paths
        )
    if paths == []:
        with pytest.raises(ValueError, match="not allowed"):
            apply_patches(tmp_path, [ApplyPatch(Path("inside.md"), content="x")], allowed_paths=[])


def test_ambiguous_adaptive_second_file_prevents_entire_batch(tmp_path: Path) -> None:
    target = tmp_path / "existing.md"
    source = "same\nsame\n"
    target.write_text(source, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="multiple locations"):
        apply_patches(
            tmp_path,
            [
                ApplyPatch(Path("new.md"), content="new\n"),
                ApplyPatch(
                    Path("existing.md"),
                    replacements=[
                        LineReplacement(
                            old_start=1, old_count=1, old_text="same", new_text="updated"
                        )
                    ],
                    expected_revision=text_revision(source),
                ),
            ],
        )
    assert not (tmp_path / "new.md").exists()
    assert target.read_text(encoding="utf-8") == source


def test_duplicate_file_batches_are_rejected_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate patch path"):
        apply_patches(
            tmp_path,
            [ApplyPatch(Path("a.md"), content="x"), ApplyPatch(Path("./a.md"), content="y")],
        )
    assert not (tmp_path / "a.md").exists()


def test_patch_requires_one_operation_and_line_source_version() -> None:
    with pytest.raises(ValueError, match="either content"):
        ApplyPatch(Path("a.md"))
    with pytest.raises(ValueError, match="expected_revision"):
        ApplyPatch(
            Path("a.md"), replacements=[LineReplacement(old_start=1, old_count=1, new_text="x")]
        )


class _Output(BaseModel, ApplyPatchOutput):
    content: str

    def to_apply_patches(self) -> list[ApplyPatch]:
        return [ApplyPatch(Path("nested/note.md"), content=self.content)]


def test_whole_file_output_and_unchanged_file(tmp_path: Path) -> None:
    output = _Output(content="# Note\r\n")
    target = tmp_path / "nested/note.md"
    assert apply_patch_output(tmp_path, output) == [target]
    assert target.read_bytes() == b"# Note\n"
    assert apply_patch_output(tmp_path, output) == []
