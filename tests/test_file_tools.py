from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tiny_coder.file_tools import FileTools
from tiny_coder.tool_node import Tool, tool


def test_bound_file_operations_keep_workspace_roots_separate(tmp_path: Path) -> None:
    left = FileTools(tmp_path / "left")
    right = FileTools(tmp_path / "right")
    writers = [tool(workspace.write_file) for workspace in (left, right)]

    async def run() -> None:
        replies = await asyncio.gather(
            *[
                writer.invoke({}, json.dumps({"path": "note.md", "content": content}))
                for writer, content in zip(writers, ("left source", "right source"))
            ]
        )
        assert all(reply.startswith("wrote note.md") for reply in replies)
        for workspace, expected in ((left, "left source"), (right, "right source")):
            reader = tool(workspace.read_file)
            assert set(reader.parameters["properties"]) == {"path", "start_line", "end_line"}
            reply = json.loads(await reader.invoke({}, '{"path":"note.md"}'))
            assert reply["content"] == f"1: {expected}"
            assert "error:" in await reader.invoke({}, '{"path":"../outside.md"}')

    asyncio.run(run())


def _invoke(tool: Tool[..., str], arguments: dict[str, object]) -> str:
    return asyncio.run(tool.invoke({}, json.dumps(arguments)))


def test_read_file_tool_reads_workspace_file(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8", newline="\n")

    result = _invoke(tool(workspace.read_file), {"path": "notes.md"})

    assert json.loads(result)["content"] == "1: hello"
    assert json.loads(result)["total_lines"] == 1


def test_read_file_tool_reports_missing_file(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    result = _invoke(tool(workspace.read_file), {"path": "missing.txt"})

    assert result.startswith("error: file not found")


def test_read_file_tool_blocks_path_escape(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    result = _invoke(tool(workspace.read_file), {"path": "../outside.txt"})

    assert result.startswith("error:")


def test_write_file_tool_writes_content(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    result = _invoke(tool(workspace.write_file), {"path": "created.md", "content": "# Hi\n"})

    assert result.startswith("wrote created.md")
    assert (tmp_path / "created.md").read_text(encoding="utf-8") == "# Hi\n"


def test_write_file_tool_creates_parent_directories(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    _invoke(tool(workspace.write_file), {"path": "nested/dir/file.txt", "content": "x\n"})

    assert (tmp_path / "nested" / "dir" / "file.txt").read_text(encoding="utf-8") == "x\n"


def test_list_dir_tool_lists_files_and_directories(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    result = _invoke(tool(workspace.list_dir), {"path": "."})
    lines = result.split("\n")

    assert "a.txt" in lines
    assert "sub/" in lines


def test_list_dir_tool_reports_missing_directory(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    result = _invoke(tool(workspace.list_dir), {"path": "nope"})

    assert result.startswith("error: directory not found")


def test_run_command_tool_returns_exit_code_and_output(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    definition = tool(workspace.run_command)
    command = (
        "python -c \"print('hello from tool')\""
        if __import__("sys").platform == "win32"
        else "python -c \"print('hello from tool')\""
    )

    result = asyncio.run(definition.invoke({}, json.dumps({"command": command})))

    assert "exit_code=0" in result
    assert "hello from tool" in result


def test_workspace_tools_are_cached_and_share_one_root(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    tools = workspace.tools
    assert workspace.tools is tools

    assert [tool.name for tool in tools] == [
        "search_files",
        "read_file",
        "edit_file",
        "write_file",
        "list_dir",
        "run_command",
    ]


def test_search_read_and_edit_duplicate_lines(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("# One\nsame\n\n# Two\nsame\n", encoding="utf-8", newline="\n")
    found = json.loads(_invoke(tool(workspace.search_files), {"pattern": "same", "glob": "*.md"}))
    assert [match["line"] for match in found["matches"]] == [2, 5]
    read = json.loads(
        _invoke(tool(workspace.read_file), {"path": "notes.md", "start_line": 4, "end_line": 5})
    )
    assert read["content"] == "4: # Two\n5: same"
    assert read["revision"] == found["matches"][1]["revision"]
    result = _invoke(
        tool(workspace.edit_file),
        {
            "path": "notes.md",
            "expected_revision": read["revision"],
            "replacements": [{"old_start": 5, "old_count": 1, "new_text": "updated"}],
        },
    )
    assert result.startswith("edited")
    assert target.read_bytes() == b"# One\nsame\n\n# Two\nupdated\n"


def test_stale_line_edit_leaves_external_change_intact(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("old\n", encoding="utf-8", newline="\n")
    read = json.loads(_invoke(tool(workspace.read_file), {"path": "notes.md"}))
    target.write_text("new header\nold\n", encoding="utf-8", newline="\n")
    result = _invoke(
        tool(workspace.edit_file),
        {
            "path": "notes.md",
            "expected_revision": read["revision"],
            "replacements": [{"old_start": 1, "old_count": 1, "new_text": "updated"}],
        },
    )
    assert "revision changed" in result
    assert target.read_bytes() == b"new header\nold\n"


def test_adaptive_tool_edit_corrects_hints_but_never_bypasses_revision(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("# Notes\n\noriginal detail\n", encoding="utf-8", newline="\n")
    read = json.loads(_invoke(tool(workspace.read_file), {"path": "notes.md"}))
    arguments = {
        "path": "notes.md",
        "expected_revision": read["revision"],
        "replacements": [
            {
                "old_start": 999,
                "old_count": 2,
                "old_text": "original detail",
                "new_text": "updated detail",
            }
        ],
    }
    assert _invoke(tool(workspace.edit_file), arguments).startswith("edited")
    assert target.read_text(encoding="utf-8") == "# Notes\n\nupdated detail\n"
    assert "revision changed" in _invoke(tool(workspace.edit_file), arguments)
    assert target.read_text(encoding="utf-8") == "# Notes\n\nupdated detail\n"


def test_write_existing_file_requires_current_revision(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8", newline="\n")
    definition = tool(workspace.write_file)
    assert "requires expected_revision" in _invoke(definition, {"path": "a.txt", "content": "x"})
    read = json.loads(_invoke(tool(workspace.read_file), {"path": "a.txt"}))
    assert _invoke(
        definition, {"path": "a.txt", "content": "x", "expected_revision": read["revision"]}
    ).startswith("wrote")
    assert target.read_bytes() == b"x"
    assert "revision changed" in _invoke(
        definition, {"path": "a.txt", "content": "y", "expected_revision": read["revision"]}
    )
    assert target.read_bytes() == b"x"


@pytest.mark.parametrize("regex,expected", [(False, [1]), (True, [1, 2])])
def test_search_literal_and_regex(tmp_path: Path, regex: bool, expected: list[int]) -> None:
    workspace = FileTools(tmp_path)
    (tmp_path / "a.txt").write_text("a.b\naxb\n", encoding="utf-8", newline="\n")
    result = json.loads(_invoke(tool(workspace.search_files), {"pattern": "a.b", "regex": regex}))
    assert [item["line"] for item in result["matches"]] == expected
    assert result["truncated"] is False


def test_search_reports_truncation_and_skips_non_utf8(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"\xff")
    (tmp_path / "b.txt").write_text("x\nx\n", encoding="utf-8", newline="\n")
    result = json.loads(_invoke(tool(workspace.search_files), {"pattern": "x", "max_matches": 1}))
    assert len(result["matches"]) == 1
    assert result["truncated"] is True
    assert "outside" in _invoke(tool(workspace.search_files), {"path": "../", "pattern": "x"})
    assert "error:" in _invoke(tool(workspace.search_files), {"pattern": "[", "regex": True})


def test_search_skips_hidden_directories_and_binary_files(tmp_path: Path) -> None:
    workspace = FileTools(tmp_path)
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "a.txt").write_text("match", encoding="utf-8", newline="\n")
    (tmp_path / "b.bin").write_bytes(b"match\x00")
    (tmp_path / "c.txt").write_text("match", encoding="utf-8", newline="\n")
    found = json.loads(_invoke(tool(workspace.search_files), {"pattern": "match"}))
    assert [item["path"] for item in found["matches"]] == ["c.txt"]
