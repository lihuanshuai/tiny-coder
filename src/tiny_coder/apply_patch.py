from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tiny_coder.text_replacement import (
    TextReplacement,
    TextReplacementApplyError,
    apply_text_replacements,
)

if TYPE_CHECKING:
    from tiny_coder.file_agent import AgentContext


@dataclass
class ApplyPatch:
    """Describe one ordered sequence of replacements for a workspace text file."""

    path: Path
    replacements: list[TextReplacement]
    ignore_punctuation_and_symbols: bool = False

    def __post_init__(self) -> None:
        if not self.replacements:
            raise ValueError("ApplyPatch requires at least one replacement")

    def apply(self, original: str | None) -> str:
        """Return the patched text without mutating the target file."""
        replacements = [
            TextReplacement(
                to_text=_normalize_newlines(item.to_text),
                from_text=(
                    _normalize_newlines(item.from_text) if item.from_text is not None else None
                ),
            )
            for item in self.replacements
            if item.from_text is None
            or _normalize_newlines(item.from_text) != _normalize_newlines(item.to_text)
        ]
        if not replacements:
            return original or ""

        updated = original or ""
        for replacement in replacements:
            if replacement.from_text is None:
                updated = replacement.to_text
                continue
            try:
                updated = apply_text_replacements(
                    updated,
                    [replacement],
                    ignore_punctuation_and_symbols=self.ignore_punctuation_and_symbols,
                )
            except TextReplacementApplyError as error:
                snippet = _compact_text_snippet(replacement.from_text)
                raise TextReplacementApplyError(f"{error}; from_text={snippet!r}") from error
        return updated


class ApplyPatchOutput(ABC):
    """Structured output that converts itself to generic workspace patches."""

    @abstractmethod
    def to_apply_patches(self, context: AgentContext) -> list[ApplyPatch]: ...


def resolve_agent_root(root: Path) -> Path:
    return root.expanduser().resolve()


def resolve_agent_file_path(root: Path, path: Path) -> Path:
    """Resolve an agent-visible file path and keep it inside the workspace root."""
    root_path = resolve_agent_root(root)
    raw = path.expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise ValueError(f"path is outside agent workspace: {path}") from error
    return candidate


def preview_apply_patches(
    root: Path,
    patches: list[ApplyPatch],
    *,
    allowed_paths: list[Path] | None = None,
) -> dict[Path, str]:
    """Resolve and apply patches in memory, returning only changed file contents."""
    originals, updated = _prepare_apply_patches(root, patches, allowed_paths=allowed_paths)
    return {
        path: content
        for path, content in updated.items()
        if originals[path] is None or content != originals[path]
    }


def apply_patches(
    root: Path,
    patches: list[ApplyPatch],
    *,
    allowed_paths: list[Path] | None = None,
) -> list[Path]:
    """Validate all patches, then write their changed contents to the workspace."""
    changed = preview_apply_patches(root, patches, allowed_paths=allowed_paths)
    for path, content in changed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(content)
    return list(changed)


def _prepare_apply_patches(
    root: Path,
    patches: list[ApplyPatch],
    *,
    allowed_paths: list[Path] | None,
) -> tuple[dict[Path, str | None], dict[Path, str]]:
    root_path = resolve_agent_root(root)
    allowed = {resolve_agent_file_path(root_path, path) for path in allowed_paths or []}
    originals: dict[Path, str | None] = {}
    updated: dict[Path, str] = {}

    for patch in patches:
        if not isinstance(patch, ApplyPatch):
            raise TypeError(f"expected ApplyPatch, got {type(patch).__qualname__}")
        target = resolve_agent_file_path(root_path, patch.path)
        if allowed and target not in allowed:
            raise ValueError(f"output file is not allowed: {target}")
        if target not in originals:
            originals[target] = _read_text_file(target) if target.is_file() else None
            updated[target] = originals[target] or ""
        try:
            updated[target] = patch.apply(updated[target])
        except TextReplacementApplyError as error:
            label = target.relative_to(root_path).as_posix()
            raise TextReplacementApplyError(f"failed to apply {label}: {error}") from error
    return originals, updated


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _compact_text_snippet(text: str, *, limit: int = 160) -> str:
    lines = [line.strip() for line in _normalize_newlines(text).strip().splitlines()]
    snippet = " / ".join(line for line in lines if line)
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as file:
        return file.read()


__all__ = [
    "ApplyPatch",
    "ApplyPatchOutput",
    "apply_patches",
    "preview_apply_patches",
    "resolve_agent_file_path",
    "resolve_agent_root",
]
