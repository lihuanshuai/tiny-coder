"""Whole-file writes and versioned line edits, preflighted as one batch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from tiny_coder.line_edit import LineReplacement, apply_line_replacements, text_revision


@dataclass
class ApplyPatch:
    """Write content or edit disjoint line ranges against one source revision."""

    path: Path
    content: str | None = None
    replacements: list[LineReplacement] = field(default_factory=list)
    expected_revision: str | None = None

    def __post_init__(self) -> None:
        if (self.content is not None) == bool(self.replacements):
            raise ValueError("provide either content or nonempty line replacements")
        if self.replacements and self.expected_revision is None:
            raise ValueError("line edits require expected_revision from the source snapshot")

    def apply(self, original: str | None) -> str:
        if self.expected_revision is not None and (
            original is None or text_revision(original) != self.expected_revision
        ):
            raise ValueError("source revision changed; read the file again and rebuild line ranges")
        if self.content is not None:
            return self.content.replace("\r\n", "\n").replace("\r", "\n")
        return apply_line_replacements(original or "", self.replacements)


class ApplyPatchOutput(ABC):
    """Structured output that converts itself to generic workspace patches."""

    @abstractmethod
    def to_apply_patches(self) -> list[ApplyPatch]: ...


def apply_patch_output(
    root: Path, output: ApplyPatchOutput, *, allowed_paths: list[Path] | None = None
) -> list[Path]:
    return apply_patches(root, output.to_apply_patches(), allowed_paths=allowed_paths)


def resolve_agent_root(root: Path) -> Path:
    return root.expanduser().resolve()


def resolve_agent_file_path(root: Path, path: Path) -> Path:
    """Resolve a file path and keep it inside the workspace root."""
    root_path = resolve_agent_root(root)
    raw = path.expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    if not candidate.is_relative_to(root_path):
        raise ValueError(f"path is outside agent workspace: {path}")
    return candidate


def preview_apply_patches(
    root: Path, patches: list[ApplyPatch], *, allowed_paths: list[Path] | None = None
) -> dict[Path, str]:
    """Check paths, versions and ranges for every file before any writes."""
    allowed = (
        {resolve_agent_file_path(root, path) for path in allowed_paths}
        if allowed_paths is not None
        else None
    )
    seen: set[Path] = set()
    changed: dict[Path, str] = {}
    for patch in patches:
        target = resolve_agent_file_path(root, patch.path)
        if allowed is not None and target not in allowed:
            raise ValueError(f"output file is not allowed: {target}")
        if target in seen:
            raise ValueError(f"duplicate patch path: {target}; merge edits into one batch")
        seen.add(target)
        original = None
        if target.exists():
            with target.open("r", encoding="utf-8", newline="\n") as source:
                original = source.read()
        try:
            updated = patch.apply(original)
        except ValueError as error:
            raise ValueError(
                f"{target.relative_to(resolve_agent_root(root)).as_posix()}: {error}"
            ) from error
        if original != updated:
            changed[target] = updated
    return changed


def apply_patches(
    root: Path, patches: list[ApplyPatch], *, allowed_paths: list[Path] | None = None
) -> list[Path]:
    """Preflight the entire batch, then write its changed files."""
    changed = preview_apply_patches(root, patches, allowed_paths=allowed_paths)
    for path, content in changed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as target:
            target.write(content)
    return list(changed)


__all__ = [
    "ApplyPatch",
    "ApplyPatchOutput",
    "apply_patch_output",
    "apply_patches",
    "preview_apply_patches",
    "resolve_agent_file_path",
    "resolve_agent_root",
]
