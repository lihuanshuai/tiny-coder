from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Annotated

from pydantic import Field

from tiny_coder.context_tools import acknowledge_context_write
from tiny_coder.line_edit import numbered_text, text_lines, text_revision
from tiny_coder.state import State

FilePath = Annotated[str, Field(min_length=1)]
FileContent = Annotated[str, Field(min_length=1)]
FileRevision = Annotated[str, Field(min_length=1)]
InsertionLine = Annotated[int, Field(ge=1, strict=True)]
ReplacementStart = Annotated[
    int,
    Field(ge=1, strict=True, description="First line to replace, 1-based in the latest revision."),
]
ReplacementCount = Annotated[
    int,
    Field(
        ge=1,
        strict=True,
        description="Number of complete lines to replace, including blank lines; not the ending line number.",
    ),
]


class IneffectiveReplacementError(ValueError):
    """A range replacement that leaves the working copy unchanged."""


class _FileContents(Mapping[str, str]):
    def __init__(self, paths: Mapping[str, Path]) -> None:
        self.paths = dict(paths)

    def __getitem__(self, path: str) -> str:
        with self.paths[path].open(encoding="utf-8", newline="\n") as source:
            return source.read()

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


class TemporaryFileTools:
    """Edit allowlisted working copies; the caller validates and publishes changes.

    Each successful operation changes the temporary file immediately. Subsequent
    reads and edits use its new content and revision. No edit queue is retained.
    Use as a context manager to clean up on success, failure or cancellation.
    String values create owned copies; Path values bind existing caller-owned
    temporary files directly. Closing removes only owned files and snapshots.
    """

    def __init__(
        self,
        files: Mapping[str, str | Path],
        *,
        directory: Path = Path(".cache"),
        validate: Callable[[State, str, str], Awaitable[None]] | None = None,
    ) -> None:
        self.validate = validate
        self.written: set[str] = set()
        self.recovery_path: str | None = None
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = TemporaryDirectory(prefix="tiny-coder-", dir=directory)
        self.root = Path(self._directory.name)
        # Logical paths may be virtual. Never use them as filesystem paths.
        self._paths = {
            path: content if isinstance(content, Path) else self.root / f"{i}.txt"
            for i, (path, content) in enumerate(files.items())
        }
        originals = {path: self.root / f"{i}.original" for i, path in enumerate(files)}
        try:
            for path, content in files.items():
                if not isinstance(content, Path):
                    self._write(path, content)
                copyfile(self._paths[path], originals[path])
            self.originals: Mapping[str, str] = _FileContents(originals)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> TemporaryFileTools:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._directory.cleanup()

    def read(self, path: str) -> str:
        with self._paths[path].open(encoding="utf-8", newline="\n") as source:
            return source.read()

    def contents(self) -> dict[str, str]:
        return {path: self.read(path) for path in self._paths}

    def changes(self) -> dict[str, str]:
        return {
            path: content
            for path in self.originals
            if path in self.written
            if (content := self.read(path)) != self.originals[path]
        }

    def _write(self, path: str, content: str) -> None:
        with self._paths[path].open("w", encoding="utf-8", newline="\n") as target:
            target.write(content)

    async def _accept(
        self, state: State, path: str, content: str, *, start: int = 1, removed: int | None = None
    ) -> str:
        source = self.read(path)
        if self.validate is not None:
            await self.validate(state, path, content)
        self._write(path, content)
        self.written.add(path)
        self.recovery_path = None
        # An edit of a fully read source is known to the model from its arguments.
        acknowledge_context_write(state, path, source, content)
        return self._result(
            path,
            changed=source != content,
            start=start,
            removed=len(text_lines(source)) if removed is None else removed,
            line_delta=len(text_lines(content)) - len(text_lines(source)),
        )

    def _result(
        self, path: str, *, changed: bool, start: int, removed: int, line_delta: int
    ) -> str:
        content = self.read(path)
        total_lines = len(text_lines(content))
        added = removed + line_delta
        context_start = max(1, start + added - 2)
        context_end = min(total_lines, start + added + 1)
        result = {
            "path": path,
            "applied": True,
            "temporary": True,
            "changed": changed,
            "revision": text_revision(content),
            "total_lines": total_lines,
            "chars": len(content),
            "edit": {
                "start_line": start,
                "removed_lines": removed,
                "added_lines": added,
                "line_delta": line_delta,
                "following_old_line": start + removed,
                "following_new_line": start + added,
            },
            "context": {
                "start_line": context_start,
                "end_line": context_end,
                "content": numbered_text(content, start_line=context_start, end_line=context_end),
            },
        }
        return json.dumps(
            result,
            ensure_ascii=False,
        )

    async def replace_range(
        self,
        state: State,
        path: FilePath,
        old_start: ReplacementStart,
        old_count: ReplacementCount,
        new_text: str,
        expected_revision: FileRevision,
    ) -> str:
        """Replace complete lines at an exact revision and return bounded updated context.

        Use line numbers and revision from the latest read or successful edit.
        The result maps the following old line to its new position via line_delta.
        Stale revisions and out-of-bounds ranges never change the working copy.
        """
        source = self.read(path)
        lines = text_lines(source)
        if expected_revision != text_revision(source):
            self.recovery_path = path
            raise ValueError(
                f"replace_range {path}: expected_revision is stale. Read {path} near "
                f"line {old_start} and use its latest revision and line numbers. "
                "No change applied; earlier successful edits remain."
            )
        if old_start < 1 or old_count < 1 or old_start + old_count - 1 > len(lines):
            self.recovery_path = path
            raise ValueError(
                f"replace_range {path}: requested start={old_start}, count={old_count}; "
                f"use a complete range within lines 1..{len(lines)}. Read {path} near "
                "the intended section and correct the range. No change applied."
            )
        replacement = new_text
        if replacement and lines[old_start + old_count - 2].endswith("\n"):
            replacement = replacement.rstrip("\n") + "\n"
        updated = (
            "".join(lines[: old_start - 1])
            + replacement
            + "".join(lines[old_start + old_count - 1 :])
        )
        if updated == source:
            raise IneffectiveReplacementError(
                f"replace_range {path}: lines {old_start}..{old_start + old_count - 1} "
                "already contain new_text. Omit this edit; continue pending edits or "
                "submit_result if finished. No change applied."
            )
        return await self._accept(state, path, updated, start=old_start, removed=old_count)

    async def insert_text(
        self,
        state: State,
        path: FilePath,
        before_line: InsertionLine,
        content: FileContent,
        expected_revision: FileRevision,
    ) -> str:
        """Insert new complete lines before before_line in the latest temporary copy.

        Use total_lines + 1 to append. Pass only new text, not existing source.
        expected_revision must match this file's current revision.
        """
        source = self.read(path)
        if expected_revision != text_revision(source):
            raise ValueError(
                f"insert_text {path}: expected_revision is stale. Read the latest working "
                "copy and use its revision and line numbers; no change applied."
            )
        lines = text_lines(source)
        if not 1 <= before_line <= len(lines) + 1:
            raise ValueError(
                f"insert_text {path}: before_line must be between 1 and {len(lines) + 1}; "
                "use total_lines + 1 to append. No change applied."
            )
        if not content.strip():
            raise ValueError("insert_text: content must contain new text, not only whitespace")
        prefix = "".join(lines[: before_line - 1])
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        inserted = content if content.endswith("\n") else content + "\n"
        return await self._accept(
            state,
            path,
            prefix + inserted + "".join(lines[before_line - 1 :]),
            start=before_line,
            removed=0,
        )

    async def write_file(
        self, state: State, path: FilePath, content: FileContent, expected_revision: FileRevision
    ) -> str:
        """Write complete content to a temporary working copy.

        expected_revision must match its latest read revision. Validation precedes
        writing; original files remain unchanged.
        """
        current = text_revision(self.read(path))
        if expected_revision != current:
            raise ValueError(
                f"write_file {path}: expected_revision is stale. Read this "
                f"working copy again and use revision {current}; no change applied."
            )
        return await self._accept(state, path, content)
