from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tiny_coder.file_tools import (
    default_file_tools,
    list_dir_tool,
    read_file_tool,
    run_command_tool,
    write_file_tool,
)
from tiny_coder.tool_node import Tool


def _invoke(tool: Tool, arguments: dict[str, object]) -> str:
    return asyncio.run(tool.invoke({}, json.dumps(arguments)))


def test_read_file_tool_reads_workspace_file(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8", newline="\n")

    result = _invoke(read_file_tool(tmp_path), {"path": "notes.md"})

    assert result == "hello\n"


def test_read_file_tool_reports_missing_file(tmp_path: Path) -> None:
    result = _invoke(read_file_tool(tmp_path), {"path": "missing.txt"})

    assert result.startswith("error: file not found")


def test_read_file_tool_blocks_path_escape(tmp_path: Path) -> None:
    result = _invoke(read_file_tool(tmp_path), {"path": "../outside.txt"})

    assert result.startswith("error:")


def test_write_file_tool_writes_content(tmp_path: Path) -> None:
    result = _invoke(write_file_tool(tmp_path), {"path": "created.md", "content": "# Hi\n"})

    assert result.startswith("wrote created.md")
    assert (tmp_path / "created.md").read_text(encoding="utf-8") == "# Hi\n"


def test_write_file_tool_creates_parent_directories(tmp_path: Path) -> None:
    _invoke(write_file_tool(tmp_path), {"path": "nested/dir/file.txt", "content": "x\n"})

    assert (tmp_path / "nested" / "dir" / "file.txt").read_text(encoding="utf-8") == "x\n"


def test_list_dir_tool_lists_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    result = _invoke(list_dir_tool(tmp_path), {"path": "."})
    lines = result.split("\n")

    assert "a.txt" in lines
    assert "sub/" in lines


def test_list_dir_tool_reports_missing_directory(tmp_path: Path) -> None:
    result = _invoke(list_dir_tool(tmp_path), {"path": "nope"})

    assert result.startswith("error: directory not found")


def test_run_command_tool_returns_exit_code_and_output(tmp_path: Path) -> None:
    tool = run_command_tool(tmp_path)
    command = (
        "python -c \"print('hello from tool')\""
        if __import__("sys").platform == "win32"
        else "python -c \"print('hello from tool')\""
    )

    result = asyncio.run(tool.invoke({}, json.dumps({"command": command})))

    assert "exit_code=0" in result
    assert "hello from tool" in result


def test_default_file_tools_binds_all_workspace_tools(tmp_path: Path) -> None:
    tools = default_file_tools(tmp_path)

    assert [tool.name for tool in tools] == ["read_file", "write_file", "list_dir", "run_command"]
