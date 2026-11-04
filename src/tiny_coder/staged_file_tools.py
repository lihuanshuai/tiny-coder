from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field

from tiny_coder.context_tools import acknowledge_context_write
from tiny_coder.line_edit import (
    AdaptiveLineReplacement,
    apply_line_replacements,
    text_lines,
    text_revision,
)
from tiny_coder.state import State
from tiny_coder.tool_node import Tool


class FileReplacement(AdaptiveLineReplacement):
    path: str = Field(min_length=1)


class FileWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    content: str = Field(min_length=1)
    expected_revision: str = Field(min_length=1)


class FileInsertion(FileWrite):
    before_line: int = Field(ge=1, strict=True)


class IneffectiveReplacementError(ValueError):
    """An unchanged or already acknowledged edit that needs no source reread."""


class TemporaryFileTools:
    """Edit allowlisted working copies; the caller validates and publishes changes.

    Each successful operation changes the temporary file immediately. Subsequent
    reads and edits use its new content and revision. No edit queue is retained.
    Use as a context manager to clean up on success, failure or cancellation.
    """

    def __init__(
        self,
        files: Mapping[str, str],
        *,
        directory: Path = Path(".cache"),
        validate: Callable[[State, str, str], Awaitable[None]] | None = None,
        replacement_error: Callable[[FileReplacement, ValueError], str] | None = None,
    ) -> None:
        self.originals = dict(files)
        self.validate = validate
        self.replacement_error = replacement_error
        self.written: set[str] = set()
        self._last_replacements: dict[str, tuple[str, str, bool]] = {}
        self.recovery_path: str | None = None
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = TemporaryDirectory(prefix="tiny-coder-", dir=directory)
        self.root = Path(self._directory.name)
        # Logical paths may be virtual. Never use them as filesystem paths.
        self._paths = {path: self.root / f"{i}.txt" for i, path in enumerate(files)}
        try:
            for path, content in files.items():
                self._write(path, content)
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

    async def _accept(self, state: State, path: str, content: str) -> str:
        source = self.read(path)
        if self.validate is not None:
            await self.validate(state, path, content)
        self._write(path, content)
        self._last_replacements.pop(path, None)
        self.written.add(path)
        self.recovery_path = None
        # An edit of a fully read source is known to the model from its arguments.
        acknowledge_context_write(state, path, source, content)
        return self._result(path, changed=source != content)

    def _result(self, path: str, *, changed: bool, already_applied: bool = False) -> str:
        content = self.read(path)
        result = {
            "path": path,
            "applied": True,
            "temporary": True,
            "changed": changed,
            "revision": text_revision(content),
            "total_lines": len(text_lines(content)),
            "chars": len(content),
        }
        if already_applied:
            result["already_applied"] = True
            result["next_action"] = (
                "This exact operation already succeeded on the current working copy. "
                "Do not repeat it. Use the latest content and line numbers for remaining "
                "changes, or submit_result if the task is complete."
            )
        return json.dumps(
            result,
            ensure_ascii=False,
        )

    def replace_lines_tool(self, *, description: str = "") -> Tool:
        schema = FileReplacement.model_json_schema()
        schema["properties"]["path"] = {"type": "string", "enum": list(self._paths)}
        schema["properties"]["old_text"]["description"] = (
            "Exact source fragment from this file's latest temporary working copy. "
            "After a successful edit, use new content and line numbers for further edits."
        )

        async def handler(
            state: State, path: str, old_start: int, old_count: int, old_text: str, new_text: str
        ) -> str:
            proposal = FileReplacement(
                path=path,
                old_start=old_start,
                old_count=old_count,
                old_text=old_text,
                new_text=new_text,
            )
            source = self.read(path)
            signature = text_revision(proposal.model_dump_json())
            try:
                previous = self._last_replacements.get(path)
                if previous is not None and previous[:2] == (signature, text_revision(source)):
                    if previous[2]:
                        raise IneffectiveReplacementError(
                            "this exact operation already succeeded and was acknowledged; "
                            "repeating it does not change the working copy"
                        )
                    self._last_replacements[path] = (*previous[:2], True)
                    return self._result(path, changed=False, already_applied=True)
                updated = apply_line_replacements(source, [proposal])
                if updated == source:
                    raise IneffectiveReplacementError(
                        "replacement leaves the working copy unchanged; "
                        "old_text and new_text must produce an actual change"
                    )
            except ValueError as error:
                if not isinstance(error, IneffectiveReplacementError):
                    self.recovery_path = path
                if self.replacement_error is not None:
                    return "error: " + self.replacement_error(proposal, error)
                if isinstance(error, IneffectiveReplacementError):
                    return (
                        f"error: replace_lines {path}: {error}. No temporary content changed. "
                        "Do not resend this operation or reread the unchanged source. "
                        "Rewrite new_text to implement the remaining task, choose another "
                        "pending edit, or call submit_result if the task is complete."
                    )
                return (
                    f"error: replace_lines {path}: {error}. No temporary content changed. "
                    "Read this file's latest working copy and correct the source and line "
                    "range; if the intended content is already present, omit this edit."
                )
            result = await self._accept(state, path, updated)
            self._last_replacements[path] = (signature, text_revision(updated), False)
            return result

        return Tool(
            name="replace_lines",
            model=FileReplacement,
            handler=handler,
            description=description
            or (
                "Replace one fragment in the latest temporary working copy. "
                "Success changes that copy immediately; subsequent edits use its new "
                "content and line numbers. Original files remain unchanged."
            ),
            parameters_schema=schema,
        )

    def insert_text_tool(self, *, description: str = "") -> Tool:
        schema = FileInsertion.model_json_schema()
        schema["properties"]["path"] = {"type": "string", "enum": list(self._paths)}
        schema["properties"]["expected_revision"] = {
            "type": "string",
            "enum": list(dict.fromkeys(text_revision(self.read(path)) for path in self._paths)),
        }

        async def handler(
            state: State, path: str, before_line: int, content: str, expected_revision: str
        ) -> str:
            source = self.read(path)
            if expected_revision != text_revision(source):
                raise ValueError(
                    f"insert_text {path}: expected_revision is stale. Read the latest working "
                    "copy and use its revision and line numbers; no change applied."
                )
            lines = text_lines(source)
            if before_line > len(lines) + 1:
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
                state, path, prefix + inserted + "".join(lines[before_line - 1 :])
            )

        return Tool(
            name="insert_text",
            model=FileInsertion,
            handler=handler,
            description=description
            or (
                "Insert new complete lines before before_line in the latest temporary copy; "
                "use total_lines + 1 to append. Pass only new text, not existing source. "
                "expected_revision must match this file's current revision."
            ),
            parameters_schema=schema,
        )

    def write_file_tool(self, *, description: str = "") -> Tool:
        schema = FileWrite.model_json_schema()
        schema["properties"]["path"] = {"type": "string", "enum": list(self._paths)}
        schema["properties"]["expected_revision"] = {
            "type": "string",
            "enum": list(dict.fromkeys(text_revision(self.read(path)) for path in self._paths)),
        }

        async def handler(state: State, path: str, content: str, expected_revision: str) -> str:
            current = text_revision(self.read(path))
            if expected_revision != current:
                raise ValueError(
                    f"write_file {path}: expected_revision is stale. Read this "
                    f"working copy again and use revision {current}; no change applied."
                )
            return await self._accept(state, path, content)

        return Tool(
            name="write_file",
            model=FileWrite,
            handler=handler,
            description=description
            or (
                "Write complete content to a temporary working copy. "
                "expected_revision must match its latest read revision. "
                "Validation precedes writing; original files remain unchanged."
            ),
            parameters_schema=schema,
        )
