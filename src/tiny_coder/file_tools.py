"""Built-in workspace tools: read, write, list, and run commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from tiny_coder.apply_patch import resolve_agent_file_path
from tiny_coder.state import State
from tiny_coder.tool_node import Tool


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="\n") as file:
        return file.read()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


class ReadFileArgs(BaseModel):
    """Read a UTF-8 text file from the workspace."""

    path: str = Field(description="Workspace-relative path of the file to read.")


class WriteFileArgs(BaseModel):
    """Create or overwrite a UTF-8 text file in the workspace."""

    path: str = Field(description="Workspace-relative path of the file to write.")
    content: str = Field(description="Complete new file content.")


class ListDirArgs(BaseModel):
    """List entries inside one workspace directory."""

    path: str = Field(default=".", description="Workspace-relative directory path.")


class RunCommandArgs(BaseModel):
    """Run a shell command with the workspace as its working directory."""

    command: str = Field(description="Shell command to execute.")
    timeout: float = Field(default=60.0, gt=0, description="Timeout in seconds.")


def read_file_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(state: State, path: str) -> str:
        _ = state
        target = resolve_agent_file_path(workspace, Path(path))
        if not target.is_file():
            return f"error: file not found: {path}"
        return _read_text(target)

    return Tool.from_model(ReadFileArgs, handler, name="read_file")


def write_file_tool(root: Path) -> Tool:
    workspace = root.expanduser().resolve()

    async def handler(state: State, path: str, content: str) -> str:
        _ = state
        target = resolve_agent_file_path(workspace, Path(path))
        _write_text(target, content)
        return f"wrote {_display_path(workspace, target)} ({len(content)} chars)"

    return Tool.from_model(WriteFileArgs, handler, name="write_file")


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
        read_file_tool(root),
        write_file_tool(root),
        list_dir_tool(root),
        run_command_tool(root),
    ]


__all__ = [
    "ListDirArgs",
    "ReadFileArgs",
    "RunCommandArgs",
    "WriteFileArgs",
    "default_file_tools",
    "list_dir_tool",
    "read_file_tool",
    "run_command_tool",
    "write_file_tool",
]
