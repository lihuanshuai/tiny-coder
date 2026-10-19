"""Workspace search, numbered reads, versioned edits, writes, and shell tools."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from tiny_coder.apply_patch import ApplyPatch, apply_patches, resolve_agent_file_path
from tiny_coder.line_edit import (
    LineReplacement,
    grep_lines,
    numbered_text,
    text_lines,
    text_revision,
)
from tiny_coder.state import State
from tiny_coder.tool_node import Tool


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as file:
        return file.read()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


class ReadFileArgs(BaseModel):
    """Read numbered UTF-8 lines and the revision required by edit_file/write_file."""

    path: str = Field(description="Workspace-relative path of the file to read.")
    start_line: int = Field(default=1, ge=1, strict=True)
    end_line: int | None = Field(default=None, ge=0, strict=True)


class WriteFileArgs(BaseModel):
    """Write a complete UTF-8 file; overwriting requires its current read revision."""

    path: str = Field(description="Workspace-relative path of the file to write.")
    content: str = Field(description="Complete new file content.")
    expected_revision: str | None = Field(default=None)


class EditFileArgs(BaseModel):
    """Edit with old_start/old_count/new_text; add old_text for adaptive minimal edits."""

    path: str
    expected_revision: str = Field(description="Revision returned by read_file or search_files.")
    replacements: list[LineReplacement] = Field(min_length=1)


class SearchFilesArgs(BaseModel):
    """Search UTF-8 files, skipping hidden entries; return paths, line numbers and revisions."""

    pattern: str = Field(min_length=1)
    path: str = Field(default=".", description="Workspace-relative file or directory.")
    glob: str = Field(default="*", description="Filename glob within the directory.")
    regex: bool = False
    max_matches: int = Field(default=100, ge=1, le=1000, strict=True)


class ListDirArgs(BaseModel):
    """List entries inside one workspace directory."""

    path: str = Field(default=".", description="Workspace-relative directory path.")


class RunCommandArgs(BaseModel):
    """Run a shell command with the workspace as its working directory."""

    command: str = Field(description="Shell command to execute.")
    timeout: float = Field(default=60.0, gt=0, description="Timeout in seconds.")


def read_file_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(state: State, path: str, start_line: int, end_line: int | None) -> str:
        _ = state
        target = resolve_agent_file_path(workspace, Path(path))
        if not target.is_file():
            return f"error: file not found: {path}"
        content = await asyncio.to_thread(_read_text, target)
        return json.dumps(
            {
                "path": _display_path(workspace, target),
                "revision": text_revision(content),
                "total_lines": len(text_lines(content)),
                "content": numbered_text(content, start_line=start_line, end_line=end_line),
            },
            ensure_ascii=False,
        )

    return Tool.from_model(ReadFileArgs, handler, name="read_file")


def write_file_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(state: State, path: str, content: str, expected_revision: str | None) -> str:
        _ = state
        target = resolve_agent_file_path(workspace, Path(path))
        if target.exists() and expected_revision is None:
            return "error: overwriting requires expected_revision; read_file first"
        await asyncio.to_thread(
            apply_patches,
            workspace,
            [ApplyPatch(target, content=content, expected_revision=expected_revision)],
        )
        return f"wrote {_display_path(workspace, target)} ({len(content)} chars)"

    return Tool.from_model(WriteFileArgs, handler, name="write_file")


def edit_file_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(
        state: State, path: str, expected_revision: str, replacements: list[dict[str, object]]
    ) -> str:
        _ = state
        patch = ApplyPatch(
            Path(path),
            replacements=[LineReplacement.model_validate(item) for item in replacements],
            expected_revision=expected_revision,
        )
        await asyncio.to_thread(apply_patches, workspace, [patch])
        return f"edited {path}; read_file again for updated line numbers and revision"

    return Tool.from_model(EditFileArgs, handler, name="edit_file")


def search_files_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    def search(path: str, pattern: str, glob: str, regex: bool, max_matches: int) -> str:
        target = resolve_agent_file_path(workspace, Path(path))
        if not target.exists():
            raise FileNotFoundError(path)
        query = re.compile(pattern if regex else re.escape(pattern))
        matches: list[dict[str, str | int]] = []
        pending = [target]
        while pending:
            candidate = pending.pop()
            if candidate.is_dir():
                pending.extend(
                    entry
                    for entry in sorted(candidate.iterdir(), reverse=True)
                    if not entry.name.startswith(".") and not entry.is_symlink()
                )
                continue
            if not candidate.is_file() or not candidate.match(glob):
                continue
            # Directory traversal can encounter symlinks pointing outside the workspace.
            if not candidate.resolve().is_relative_to(workspace):
                continue
            try:
                content = _read_text(candidate)
            except UnicodeError:
                continue
            if "\0" in content:
                continue
            revision = text_revision(content)
            for number, line in grep_lines(content, query):
                if len(matches) == max_matches:
                    return json.dumps({"matches": matches, "truncated": True}, ensure_ascii=False)
                matches.append(
                    {
                        "path": _display_path(workspace, candidate),
                        "line": number,
                        "content": line,
                        "revision": revision,
                    }
                )
        return json.dumps({"matches": matches, "truncated": False}, ensure_ascii=False)

    async def handler(
        state: State, path: str, pattern: str, glob: str, regex: bool, max_matches: int
    ) -> str:
        _ = state
        return await asyncio.to_thread(search, path, pattern, glob, regex, max_matches)

    return Tool.from_model(SearchFilesArgs, handler, name="search_files")


def list_dir_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(state: State, path: str) -> str:
        _ = state
        target = resolve_agent_file_path(workspace, Path(path))
        if not target.is_dir():
            return f"error: directory not found: {path}"
        entries = sorted(
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in target.iterdir()
        )
        return "\n".join(entries) if entries else "(empty)"

    return Tool.from_model(ListDirArgs, handler, name="list_dir")


def run_command_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(state: State, command: str, timeout: float) -> str:
        _ = state
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            result: tuple[bytes, bytes] = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            stdout_bytes, _ = result
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"error: command timed out after {timeout:g}s"
        output = stdout_bytes.decode("utf-8", errors="replace")
        return f"exit_code={process.returncode}\n{output.rstrip()}"

    return Tool.from_model(RunCommandArgs, handler, name="run_command")


def default_file_tools(root: Path) -> list[Tool]:
    """Return the standard workspace tools bound to one root directory."""
    return [
        search_files_tool(root),
        read_file_tool(root),
        edit_file_tool(root),
        write_file_tool(root),
        list_dir_tool(root),
        run_command_tool(root),
    ]


__all__ = [
    "ListDirArgs",
    "EditFileArgs",
    "SearchFilesArgs",
    "ReadFileArgs",
    "RunCommandArgs",
    "WriteFileArgs",
    "default_file_tools",
    "list_dir_tool",
    "edit_file_tool",
    "search_files_tool",
    "read_file_tool",
    "run_command_tool",
    "write_file_tool",
]
