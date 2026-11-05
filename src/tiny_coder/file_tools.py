"""Workspace search, numbered reads, versioned edits, writes, and shell tools."""

from __future__ import annotations

import asyncio
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Annotated

from pydantic import Field

from tiny_coder.apply_patch import ApplyPatch, apply_patches, resolve_agent_file_path
from tiny_coder.line_edit import (
    LineReplacement,
    grep_lines,
    numbered_text,
    text_lines,
    text_revision,
)
from tiny_coder.tool_node import Tool, tool


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as file:
        return file.read()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


class FileTools:
    """Async filesystem operations bound to one workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @cached_property
    def tools(self) -> list[Tool[..., str]]:
        """Build the workspace tool definitions once; individual methods remain callable."""
        return [
            tool(handler)
            for handler in (
                self.search_files,
                self.read_file,
                self.edit_file,
                self.write_file,
                self.list_dir,
                self.run_command,
            )
        ]

    async def read_file(
        self,
        path: Annotated[str, Field(description="Workspace-relative path of the file to read.")],
        start_line: Annotated[int, Field(ge=1, strict=True)] = 1,
        end_line: Annotated[int | None, Field(ge=0, strict=True)] = None,
    ) -> str:
        """Read numbered UTF-8 lines and the revision required by edit_file/write_file."""
        target = resolve_agent_file_path(self.root, Path(path))
        if not target.is_file():
            return f"error: file not found: {path}"
        content = await asyncio.to_thread(_read_text, target)
        return json.dumps(
            {
                "path": _display_path(self.root, target),
                "revision": text_revision(content),
                "total_lines": len(text_lines(content)),
                "content": numbered_text(content, start_line=start_line, end_line=end_line),
            },
            ensure_ascii=False,
        )

    async def write_file(
        self,
        path: Annotated[str, Field(description="Workspace-relative path of the file to write.")],
        content: Annotated[str, Field(description="Complete new file content.")],
        expected_revision: str | None = None,
    ) -> str:
        """Write a complete UTF-8 file; overwriting requires its current read revision."""
        target = resolve_agent_file_path(self.root, Path(path))
        if target.exists() and expected_revision is None:
            return "error: overwriting requires expected_revision; read_file first"
        await asyncio.to_thread(
            apply_patches,
            self.root,
            [ApplyPatch(target, content=content, expected_revision=expected_revision)],
        )
        return f"wrote {_display_path(self.root, target)} ({len(content)} chars)"

    async def edit_file(
        self,
        path: str,
        expected_revision: Annotated[
            str, Field(description="Revision returned by read_file or search_files.")
        ],
        replacements: Annotated[list[LineReplacement], Field(min_length=1)],
    ) -> str:
        """Edit with old_start/old_count/new_text; add old_text for adaptive minimal edits."""
        patch = ApplyPatch(
            Path(path),
            replacements=replacements,
            expected_revision=expected_revision,
        )
        await asyncio.to_thread(apply_patches, self.root, [patch])
        return f"edited {path}; read_file again for updated line numbers and revision"

    def _search(self, path: str, pattern: str, glob: str, regex: bool, max_matches: int) -> str:
        target = resolve_agent_file_path(self.root, Path(path))
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
            if not candidate.resolve().is_relative_to(self.root):
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
                        "path": _display_path(self.root, candidate),
                        "line": number,
                        "content": line,
                        "revision": revision,
                    }
                )
        return json.dumps({"matches": matches, "truncated": False}, ensure_ascii=False)

    async def search_files(
        self,
        pattern: Annotated[str, Field(min_length=1)],
        path: Annotated[str, Field(description="Workspace-relative file or directory.")] = ".",
        glob: Annotated[str, Field(description="Filename glob within the directory.")] = "*",
        regex: bool = False,
        max_matches: Annotated[int, Field(ge=1, le=1000, strict=True)] = 100,
    ) -> str:
        """Search UTF-8 files, skipping hidden entries; return paths, line numbers and revisions."""
        return await asyncio.to_thread(self._search, path, pattern, glob, regex, max_matches)

    async def list_dir(
        self,
        path: Annotated[str, Field(description="Workspace-relative directory path.")] = ".",
    ) -> str:
        """List entries inside one workspace directory."""
        target = resolve_agent_file_path(self.root, Path(path))
        if not target.is_dir():
            return f"error: directory not found: {path}"
        entries = sorted(
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in target.iterdir()
        )
        return "\n".join(entries) if entries else "(empty)"

    async def run_command(
        self,
        command: Annotated[str, Field(description="Shell command to execute.")],
        timeout: Annotated[float, Field(gt=0, description="Timeout in seconds.")] = 60.0,
    ) -> str:
        """Run a shell command with the workspace as its working directory."""
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
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


__all__ = ["FileTools"]
