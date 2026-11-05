from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import PurePosixPath
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from tiny_coder.agent_extensions import ResponseValidationError
from tiny_coder.eventbus import EventBus
from tiny_coder.line_edit import grep_lines, numbered_text, text_lines, text_revision
from tiny_coder.state import State
from tiny_coder.structured_agent import (
    BEFORE_CALL,
    BEFORE_OUTPUT,
    SUBMIT_RESULT,
    AgentCall,
    AgentExtension,
)
from tiny_coder.tool_node import Tool, tool


class ContextFile(BaseModel):
    """An explicitly exposed snapshot; tools never access the filesystem themselves."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    description: str = ""
    writable: bool = False
    required: bool = False


class _ReadRecord(TypedDict):
    revision: str
    ranges: list[list[int]]


def _path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value or not path.parts:
        raise ValueError(f"invalid context path: {value!r}; use a path from list_files")
    return path.as_posix()


def _merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for first, last in sorted(ranges):
        if last < first:
            continue
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(last, merged[-1][1])
        else:
            merged.append([first, last])
    return merged


def _unread_ranges(ranges: list[list[int]], first: int, last: int) -> list[list[int]]:
    missing = []
    cursor = first
    for start, end in _merge_ranges(ranges):
        if start > last:
            break
        if start > cursor:
            missing.append([cursor, start - 1])
        cursor = max(cursor, end + 1)
    if cursor <= last:
        missing.append([cursor, last])
    return missing


def _read_complete(record: _ReadRecord | None, revision: str, total_lines: int) -> bool:
    return (
        record is not None
        and record["revision"] == revision
        and not _unread_ranges(record["ranges"], 1, total_lines)
    )


def acknowledge_context_write(state: State, path: str, before: str, after: str) -> None:
    """Keep a fully read snapshot current after an acknowledged tool edit."""
    reads: dict[str, _ReadRecord] = state.get("context_reads", {})
    if _read_complete(reads.get(path), text_revision(before), len(text_lines(before))):
        total = len(text_lines(after))
        reads[path] = {"revision": text_revision(after), "ranges": [[1, total]] if total else []}


@dataclass
class ContextTools(AgentExtension):
    """Load allowlisted, invocation-local snapshots through actual tool messages.

    A provider may publish a newer in-memory draft after validation. Required files
    must be read at their current revision before final output can be accepted.
    Reading disjoint ranges is supported; grep never satisfies this requirement.
    """

    files: Sequence[ContextFile] | Callable[[AgentCall], Sequence[ContextFile]]

    def register(self, events: EventBus) -> None:
        events.subscribe(BEFORE_CALL, self.prepare)
        events.subscribe(BEFORE_OUTPUT, self.check_reads)

    def _files(self, call: AgentCall) -> dict[str, ContextFile]:
        files = self.files(call) if callable(self.files) else self.files
        result: dict[str, ContextFile] = {}
        for file in files:
            path = _path(file.path)
            if path in result:
                raise ValueError(f"duplicate context path: {path}")
            result[path] = file
        return result

    def _file(self, call: AgentCall, path: str) -> ContextFile:
        files = self._files(call)
        canonical = _path(path)
        if canonical not in files:
            raise ValueError(
                f"context file is not available: {path}; call list_files for available paths"
            )
        return files[canonical]

    async def prepare(self, call: AgentCall) -> None:
        # Validate the allowlist before the first request as well as on each read.
        call.submission_enabled = not self._missing(call)
        if not call.submission_enabled:
            call.tool_choice = {"type": "function", "name": "read_required"}
        elif (
            call.response_model is not None
            and isinstance(call.previous_error, ResponseValidationError)
            and call.previous_error.code == "submission_required"
        ):
            call.tool_choice = {"type": "function", "name": SUBMIT_RESULT}

        key = f"context_tools:{id(self)}"
        reader = call.invocation_data.get(key)
        if not isinstance(reader, _ContextReader):
            reader = _ContextReader(self, call)
            call.invocation_data[key] = reader
        else:
            reader.call = call

        if call.submission_enabled and call.response_model is not None and call.messages:
            last = call.messages[-1]
            if last.get("role") == "tool" and last.get("name") in {"list_files", "read_required"}:
                try:
                    reply = json.loads(last["content"])
                except (ValueError, TypeError):
                    reply = {}
                if (
                    isinstance(reply, dict)
                    and reply.get("unchanged") is True
                    and (
                        last["name"] == "read_required"
                        or reply.get("revision") == text_revision(reader.listing())
                    )
                ):
                    call.tool_choice = {"type": "function", "name": SUBMIT_RESULT}

        call.register_tools(*reader.tools, replace=True)

    def _missing(self, call: AgentCall) -> list[str]:
        reads: dict[str, _ReadRecord] = call.state.get("context_reads", {})
        missing: list[str] = []
        for path, file in self._files(call).items():
            if not file.required:
                continue
            if not _read_complete(
                reads.get(path), text_revision(file.content), len(text_lines(file.content))
            ):
                missing.append(path)
        return missing

    async def check_reads(self, call: AgentCall) -> None:
        assert call.outcome is not None
        if call.outcome.tool_calls and call.submission is None:
            return
        missing = self._missing(call)
        if missing:
            raise ResponseValidationError(
                f"required context not read at its current revision: {', '.join(missing)}",
                code="context_not_read",
                context={"paths": missing},
                suggestion="Call read_required once to read all listed paths at their current revisions, then submit the complete result. Alternatively use read_file to cover all their lines. grep hits do not provide full source. No submission has been accepted.",
            )


@dataclass
class _ContextReader:
    context: ContextTools
    call: AgentCall

    @cached_property
    def tools(self) -> list[Tool[..., str]]:
        return [
            tool(handler)
            for handler in (self.list_files, self.read_file, self.read_required, self.grep)
        ]

    def listing(self) -> str:
        reads: dict[str, _ReadRecord] = self.call.state.get("context_reads", {})
        files = []
        for path, file in self.context._files(self.call).items():
            revision = text_revision(file.content)
            total_lines = len(text_lines(file.content))
            record = reads.get(path)
            ranges = (
                _merge_ranges(record["ranges"])
                if record is not None and record["revision"] == revision
                else []
            )
            files.append(
                {
                    "path": path,
                    "description": file.description,
                    "writable": file.writable,
                    "required": file.required,
                    "total_lines": total_lines,
                    "revision": revision,
                    "read": _read_complete(record, revision, total_lines),
                    "read_ranges": ranges,
                    "missing_ranges": _unread_ranges(ranges, 1, total_lines),
                }
            )
        return json.dumps(
            {"files": files},
            ensure_ascii=False,
        )

    async def list_files(self, state: State) -> str:
        """List context paths, revisions, read/missing ranges and access rules; no contents.

        An unchanged catalog returns only its revision; reuse the earlier listing.
        """
        content = self.listing()
        revision = text_revision(content)
        if state.get("context_listing_revision") == revision:
            return json.dumps({"files": [], "unchanged": True, "revision": revision})
        state["context_listing_revision"] = revision
        return content

    def read_snapshot(
        self, state: State, path: str, start_line: int, end_line: int | None, *, unread_only: bool
    ) -> str:
        file = self.context._file(self.call, path)
        canonical = _path(path)
        lines = text_lines(file.content)
        end = len(lines) if end_line is None else end_line
        if end < start_line or start_line > len(lines) or end > len(lines):
            if lines or start_line != 1 or end_line is not None:
                raise ValueError(
                    f"invalid line range {start_line}..{end}; {canonical} has {len(lines)} lines. "
                    f"Set start_line and end_line within 1..{len(lines)} with end_line >= "
                    "start_line, or use end_line=null to read to the end."
                )
        revision = text_revision(file.content)
        reads: dict[str, _ReadRecord] = state.setdefault("context_reads", {})
        record = reads.get(canonical)
        already_read = (
            record is not None
            and record["revision"] == revision
            and not _unread_ranges(record["ranges"], start_line, end)
        )
        if record is None or record["revision"] != revision:
            record = _ReadRecord(revision=revision, ranges=[])
            reads[canonical] = record
        if unread_only:
            returned = _unread_ranges(record["ranges"], start_line, end)
        else:
            returned = [[start_line, end]] if lines else []
        record["ranges"] = _merge_ranges([*record["ranges"], *returned])
        return json.dumps(
            {
                "path": canonical,
                "revision": revision,
                "description": file.description,
                "writable": file.writable,
                "start_line": start_line,
                "end_line": end,
                "total_lines": len(lines),
                "returned_ranges": returned,
                "read_ranges": record["ranges"],
                "missing_ranges": _unread_ranges(record["ranges"], 1, len(lines)),
                "already_read": already_read,
                "content": "\n".join(
                    numbered_text(file.content, start_line=first, end_line=last)
                    for first, last in returned
                ),
            },
            ensure_ascii=False,
        )

    async def read_file(
        self,
        state: State,
        path: str,
        start_line: Annotated[int, Field(ge=1, strict=True)] = 1,
        end_line: Annotated[int | None, Field(ge=1, strict=True)] = None,
    ) -> str:
        """Read requested snapshot lines, including previously read lines.

        Line numbers are 1-based; end_line=null reads to the end. returned_ranges
        identifies returned content; read_ranges covers all lines at this revision.
        Prefer earlier tool content when sufficient; reread a small range to repair anchors.
        """
        return self.read_snapshot(state, path, start_line, end_line, unread_only=False)

    async def grep(
        self,
        pattern: Annotated[str, Field(min_length=1)],
        path: Annotated[
            str,
            Field(description="Context path from list_files; empty searches all snapshots."),
        ] = "",
        max_matches: Annotated[int, Field(ge=1, le=500, strict=True)] = 100,
    ) -> str:
        """Locate a regex in exposed snapshots. Hits do not count as reading a file."""
        try:
            regex = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"invalid grep regex: {error}") from error
        files = (
            self.context._files(self.call)
            if path == ""
            else {_path(path): self.context._file(self.call, path)}
        )
        matches: list[dict[str, str | int]] = []
        for name, file in files.items():
            revision = text_revision(file.content)
            for number, line in grep_lines(file.content, regex):
                if len(matches) == max_matches:
                    return json.dumps({"matches": matches, "truncated": True}, ensure_ascii=False)
                matches.append({"path": name, "line": number, "text": line, "revision": revision})
        return json.dumps({"matches": matches, "truncated": False}, ensure_ascii=False)

    async def read_required(self, state: State) -> str:
        """Read every required snapshot not fully read at its current revision in one call.

        Only missing line ranges are returned; fully read current snapshots are skipped.
        """
        files = [
            json.loads(self.read_snapshot(state, path, 1, None, unread_only=True))
            for path in self.context._missing(self.call)
        ]
        return json.dumps({"files": files, "unchanged": not files}, ensure_ascii=False)


__all__ = ["ContextFile", "ContextTools"]
