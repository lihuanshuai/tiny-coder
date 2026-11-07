from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from test_structured_agent import SampleLlmConfig, _outcome

from tiny_coder.context_tools import acknowledge_context_write
from tiny_coder.line_edit import text_revision
from tiny_coder.llm import LlmChatOutcome, ToolCall
from tiny_coder.staged_file_tools import (
    TemporaryFileTools,
)
from tiny_coder.state import State
from tiny_coder.structured_agent import (
    BEFORE_CALL,
    AgentCall,
    StructuredInput,
    create_structured_agent,
)
from tiny_coder.tool_node import Tool, tool


def bind(workspace: TemporaryFileTools, operation: str) -> Tool[..., str]:
    """Register native methods with the same invocation-local constraints as consumers."""
    parameters: dict[str, dict[str, Any]] = {
        "path": {"type": "string", "enum": list(workspace.originals)}
    }
    if operation != "replace_range":
        parameters["expected_revision"] = {
            "type": "string",
            "enum": list(
                dict.fromkeys(text_revision(text) for text in workspace.contents().values())
            ),
        }
    return tool(getattr(workspace, operation), parameters=parameters)


def invoke(tool: Tool[..., str], state: State, **arguments: object) -> str:
    return asyncio.run(tool.invoke(state, json.dumps(arguments)))


def edit(old: str, new: str, *, path: str = "one.md", start: int = 1) -> dict[str, object]:
    return dict(
        path=path, old_start=start, old_count=1, expected_revision=text_revision(old), new_text=new
    )


def test_native_async_methods_edit_and_preserve_failed_write_progress(tmp_path: Path) -> None:
    async def run() -> None:
        with TemporaryFileTools({"one.md": "first\n"}, directory=tmp_path / ".cache") as workspace:
            await workspace.replace_range({}, "one.md", 1, 1, "FIRST", text_revision("first\n"))
            revision = text_revision(workspace.read("one.md"))
            await workspace.insert_text({}, "one.md", 2, "second", revision)
            assert workspace.read("one.md") == "FIRST\nsecond\n"
            with pytest.raises(ValueError, match="stale"):
                await workspace.write_file({}, "one.md", "invalid", revision)
            assert workspace.read("one.md") == "FIRST\nsecond\n"
            revision = text_revision(workspace.read("one.md"))
            with pytest.raises(ValueError, match="between 1 and 3"):
                await workspace.insert_text({}, "one.md", 0, "invalid", revision)
            await workspace.write_file({}, "one.md", "complete\n", revision)
            assert workspace.changes() == {"one.md": "complete\n"}
        assert not workspace.root.exists()

    asyncio.run(run())


def test_bound_paths_edit_existing_files_and_keep_original_snapshots_on_disk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "draft.md"
    path.write_text("first\n", encoding="utf-8", newline="\n")

    async def run() -> None:
        with TemporaryFileTools({"draft.md": path}, directory=tmp_path / ".cache") as workspace:
            assert workspace.originals["draft.md"] == "first\n"
            path.write_text("external\n", encoding="utf-8", newline="\n")
            assert workspace.read("draft.md") == "external\n"
            await workspace.replace_range(
                {}, "draft.md", 1, 1, "updated", text_revision("external\n")
            )
            assert path.read_text(encoding="utf-8") == "updated\n"
            assert workspace.originals["draft.md"] == "first\n"
            assert workspace.changes() == {"draft.md": "updated\n"}
            root = workspace.root
        assert not root.exists()
        assert path.read_text(encoding="utf-8") == "updated\n"

    asyncio.run(run())


def test_replace_range_uses_latest_revision_and_preserves_neighboring_lines(tmp_path: Path) -> None:
    async def run() -> None:
        source = "# Heading\n\nold\n\n## Next\nkeep\n"
        with TemporaryFileTools({"one.md": source}, directory=tmp_path) as workspace:
            result = json.loads(
                await workspace.replace_range(
                    {}, "one.md", 3, 1, "new\nextra", text_revision(source)
                )
            )
            assert workspace.read("one.md") == "# Heading\n\nnew\nextra\n\n## Next\nkeep\n"
            assert result["total_lines"] == 7
            await workspace.replace_range({}, "one.md", 4, 1, "revised", result["revision"])
            assert workspace.read("one.md").endswith("new\nrevised\n\n## Next\nkeep\n")
            assert workspace.originals["one.md"] == source

    asyncio.run(run())


@pytest.mark.parametrize("replacement", ["expanded\nextra\nthird", "", "revised"])
def test_edit_result_locates_following_target_after_line_shift(
    tmp_path: Path, replacement: str
) -> None:
    async def run() -> None:
        source = "heading\nfirst\n\nsecond\nkeep\n"
        with TemporaryFileTools({"one.md": source}, directory=tmp_path) as workspace:
            result = json.loads(
                await workspace.replace_range(
                    {}, "one.md", 2, 1, replacement, text_revision(source)
                )
            )
            delta = len(replacement.splitlines()) - 1
            assert result["edit"]["line_delta"] == delta
            assert result["edit"]["following_old_line"] == 3
            assert result["edit"]["following_new_line"] == 3 + delta
            target = 4 + result["edit"]["line_delta"]
            assert f"{target}: second" in result["context"]["content"]
            await workspace.replace_range({}, "one.md", target, 1, "SECOND", result["revision"])
            assert workspace.read("one.md").endswith("\nSECOND\nkeep\n")
            assert "second" not in workspace.read("one.md")

    asyncio.run(run())


def test_insert_result_and_empty_deletion_report_bounded_context(tmp_path: Path) -> None:
    async def run() -> None:
        with TemporaryFileTools({"one.md": "tail"}, directory=tmp_path) as workspace:
            result = json.loads(
                await workspace.insert_text(
                    {},
                    "one.md",
                    1,
                    "\n".join(f"new {i}" for i in range(100)),
                    text_revision("tail"),
                )
            )
            assert result["edit"]["added_lines"] == 100
            assert result["edit"]["line_delta"] == 100
            assert result["context"]["content"].endswith("101: tail")
            assert len(result["context"]["content"].splitlines()) <= 4
            result = json.loads(
                await workspace.replace_range({}, "one.md", 1, 101, "", result["revision"])
            )
            assert workspace.read("one.md") == ""
            assert result["edit"]["added_lines"] == 0
            assert result["context"]["content"] == ""

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["stale", "bounds", "unchanged", "validation"])
def test_rejected_range_edit_preserves_successful_progress(tmp_path: Path, failure: str) -> None:
    async def run() -> None:
        with TemporaryFileTools({"one.md": "first\nsecond\n"}, directory=tmp_path) as workspace:
            original_revision = text_revision(workspace.read("one.md"))
            result = json.loads(
                await workspace.replace_range({}, "one.md", 1, 1, "FIRST", original_revision)
            )
            source = workspace.read("one.md")

            async def reject(state: State, path: str, content: str) -> None:
                raise ValueError("domain rejected")

            if failure == "validation":
                workspace.validate = reject
            with pytest.raises(
                ValueError,
                match={
                    "stale": "latest revision",
                    "bounds": "within lines 1..2",
                    "unchanged": "Omit this edit",
                    "validation": "domain rejected",
                }[failure],
            ):
                await workspace.replace_range(
                    {},
                    "one.md",
                    2,
                    3 if failure == "bounds" else 1,
                    "second" if failure == "unchanged" else "SECOND",
                    original_revision if failure == "stale" else result["revision"],
                )
            assert workspace.read("one.md") == source
            assert workspace.changes() == {"one.md": source}
            assert workspace.recovery_path == ("one.md" if failure in {"stale", "bounds"} else None)

    asyncio.run(run())


@pytest.mark.parametrize("invalid", [{"old_count": 0}, {"old_start": True}, {"path": "../one.md"}])
def test_range_tool_rejects_invalid_arguments_before_editing(
    tmp_path: Path, invalid: dict[str, object]
) -> None:
    with TemporaryFileTools({"one.md": "keep\n"}, directory=tmp_path) as workspace:
        definition = bind(workspace, "replace_range")
        assert definition.parameters["properties"]["old_count"]["minimum"] == 1
        result = invoke(
            definition,
            {},
            **{
                "path": "one.md",
                "old_start": 1,
                "old_count": 1,
                "new_text": "changed",
                "expected_revision": text_revision(workspace.read("one.md")),
                **invalid,
            },
        )
        assert result.startswith("error:")
        assert workspace.read("one.md") == "keep\n" and not workspace.written


def test_sequential_replacements_use_latest_content_and_shifted_lines(tmp_path: Path) -> None:
    path = tmp_path / "one.md"
    path.write_text("first\nsecond\n", encoding="utf-8", newline="\n")
    with TemporaryFileTools({"one.md": "first\nsecond\n"}) as workspace:
        root = workspace.root
        state: State = {}
        for start, new in [(1, "FIRST\ninserted"), (3, "SECOND"), (2, "revised")]:
            result = json.loads(
                invoke(
                    bind(workspace, "replace_range"),
                    state,
                    **edit(workspace.read("one.md"), new, start=start),
                )
            )
            assert result["applied"] and result["temporary"] and result["total_lines"] == 3
        assert workspace.changes() == {"one.md": "FIRST\nrevised\nSECOND\n"}
        assert path.read_text(encoding="utf-8") == workspace.originals["one.md"]
        assert not state
    assert not root.exists()


def test_working_copy_uses_cache_in_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with TemporaryFileTools({"one.md": "first"}) as workspace:
        root = workspace.root.resolve()
        assert root.parent == tmp_path / ".cache"
        assert workspace.read("one.md") == "first"
    assert not root.exists()


@pytest.mark.parametrize(
    "source,line,expected",
    [
        ("first\nlast\n", 2, "first\nnew\nlast\n"),
        ("first", 2, "first\nnew\n"),
        ("", 1, "new\n"),
        ("last\n", 1, "new\nlast\n"),
    ],
)
def test_insert_text_uses_revision_and_preserves_existing_lines(
    tmp_path: Path,
    source: str,
    line: int,
    expected: str,
) -> None:
    with TemporaryFileTools({"one.md": source}, directory=tmp_path / ".cache") as workspace:
        tool = bind(workspace, "insert_text")
        arguments = dict(
            path="one.md", before_line=line, content="new", expected_revision=text_revision(source)
        )
        result = json.loads(invoke(tool, {}, **arguments))
        assert result["applied"] and result["changed"] and result["temporary"]
        assert workspace.read("one.md") == expected
        assert result["revision"] == text_revision(expected)
        assert "stale" in invoke(tool, {}, **arguments)
        assert workspace.read("one.md") == expected
    assert not workspace.root.exists()


def test_invalid_insertions_preserve_working_copy_and_run_domain_validation(tmp_path: Path) -> None:
    async def validate(state: State, path: str, content: str) -> None:
        if "forbidden" in content:
            raise ValueError("forbidden content")

    with TemporaryFileTools(
        {"one.md": "original\n"}, directory=tmp_path / ".cache", validate=validate
    ) as workspace:
        args = dict(
            path="one.md",
            before_line=1,
            content="new",
            expected_revision=text_revision("original\n"),
        )
        tool = bind(workspace, "insert_text")
        invalid: list[tuple[dict[str, object], str]] = [
            ({"path": "outside.md"}, "path"),
            ({"before_line": 3}, "between 1 and 2"),
            ({"before_line": 0}, "before_line"),
            ({"content": "  \n"}, "whitespace"),
            ({"content": "forbidden"}, "forbidden content"),
        ]
        for changes, detail in invalid:
            error = invoke(tool, {}, **{**args, **changes})
            assert error.startswith("error:") and detail in error
            assert workspace.read("one.md") == "original\n"
            assert not workspace.changes()


def test_custom_cache_cleanup_preserves_other_tasks_and_files(tmp_path: Path) -> None:
    cache = tmp_path / "story" / ".cache"
    with TemporaryFileTools({"one.md": "first"}, directory=cache) as first:
        unrelated = cache / "other.txt"
        unrelated.write_text("keep", encoding="utf-8", newline="\n")
        with TemporaryFileTools({"one.md": "second"}, directory=cache) as second:
            second_root = second.root
            assert first.root.parent == second.root.parent == cache
            assert first.root != second.root
        assert not second_root.exists()
        assert first.read("one.md") == "first"
    assert not first.root.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("new", ["first\ninserted", "", "FIRST"])
def test_repeated_range_edit_with_old_revision_is_rejected(new: str) -> None:
    source = "first\nsecond\n"
    with TemporaryFileTools({"one.md": source}) as workspace:
        proposal = edit(source, new)
        first = json.loads(invoke(bind(workspace, "replace_range"), {}, **proposal))
        saved = workspace.read("one.md")
        for _ in range(2):
            repeated = invoke(bind(workspace, "replace_range"), {}, **proposal)
            assert repeated.startswith("error:") and "stale" in repeated
            assert workspace.read("one.md") == saved
            assert first["revision"] == text_revision(saved)
            assert workspace.recovery_path == "one.md"


@pytest.mark.parametrize("new", ["first", "first\n"])
def test_unchanged_range_is_rejected_without_writing_or_forcing_reread(new: str) -> None:
    async def validate(state: State, path: str, content: str) -> None:
        pytest.fail("an unchanged replacement must not reach candidate validation")

    source = "first\nsecond\n"
    with TemporaryFileTools({"one.md": source}, validate=validate) as workspace:
        error = invoke(bind(workspace, "replace_range"), {}, **edit(source, new))
        assert error.startswith("error:") and "already contain new_text" in error
        assert "Omit this edit" in error and "submit_result" in error
        assert workspace.recovery_path is None
        assert not workspace.written and not workspace.changes()
        assert workspace.read("one.md") == source


def test_same_text_on_distinct_lines_requires_each_latest_revision() -> None:
    with TemporaryFileTools({"one.md": "first\nfirst\n"}) as workspace:
        for start in (1, 2):
            result = json.loads(
                invoke(
                    bind(workspace, "replace_range"),
                    {},
                    **edit(workspace.read("one.md"), "FIRST", start=start),
                )
            )
            assert result["changed"]
        assert workspace.read("one.md") == "FIRST\nFIRST\n"


def test_rejections_preserve_progress_and_consumer_validation_feedback() -> None:
    async def validate(state: State, path: str, content: str) -> None:
        if "forbidden" in content:
            raise ValueError("forbidden content: preserve existing words")

    with TemporaryFileTools({"one.md": "first\nsecond\n"}, validate=validate) as workspace:
        invoke(bind(workspace, "replace_range"), {}, **edit(workspace.read("one.md"), "FIRST"))
        earlier = workspace.read("one.md")
        error = invoke(bind(workspace, "replace_range"), {}, **edit("stale", "different"))
        assert error.startswith("error:") and "earlier successful edits remain" in error
        assert workspace.recovery_path == "one.md"
        assert "preserve existing words" in invoke(
            bind(workspace, "replace_range"), {}, **edit(earlier, "forbidden")
        )
        assert workspace.read("one.md") == earlier
        invoke(bind(workspace, "replace_range"), {}, **edit(earlier, "SECOND", start=2))
        assert workspace.recovery_path is None


def test_full_writes_require_latest_revision_and_validate_before_writing() -> None:
    async def validate(state: State, path: str, content: str) -> None:
        if content == "bad":
            raise ValueError("replace bad with complete valid content")

    with TemporaryFileTools(
        {"one.md": "first", "two.md": "second", "@output/summary.md": ""}, validate=validate
    ) as workspace:
        tool = bind(workspace, "write_file")
        arguments = dict(path="one.md", content="updated", expected_revision=text_revision("first"))
        assert json.loads(invoke(tool, {}, **arguments))["applied"]
        assert "expected_revision" in invoke(tool, {}, **arguments)
        assert "stale" in invoke(
            tool, {}, **{**arguments, "expected_revision": text_revision("second")}
        )
        refreshed = bind(workspace, "write_file")
        assert "complete valid content" in invoke(
            refreshed,
            {},
            **{**arguments, "content": "bad", "expected_revision": text_revision("updated")},
        )
        assert workspace.read("one.md") == "updated"
        invoke(
            refreshed,
            {},
            path="@output/summary.md",
            content="summary",
            expected_revision=text_revision(""),
        )
        invoke(
            refreshed,
            {},
            **{**arguments, "content": "revised", "expected_revision": text_revision("updated")},
        )
        assert workspace.changes() == {"one.md": "revised", "@output/summary.md": "summary"}
        assert workspace.originals["one.md"] == "first"
        assert not (workspace.root / "@output").exists()


@pytest.mark.parametrize("operation", ["write_file", "replace_range"])
@pytest.mark.parametrize("invalid", ["../outside.md", "/absolute.md", "unknown.md"])
def test_tools_reject_paths_outside_allowlist(operation: str, invalid: str) -> None:
    with TemporaryFileTools({"one.md": "first"}) as workspace:
        tool = bind(workspace, operation)
        arguments = (
            dict(path=invalid, content="updated", expected_revision=text_revision("first"))
            if operation == "write_file"
            else edit("first", "updated", path=invalid)
        )
        result = invoke(tool, {}, **arguments)
        assert result.startswith("error:") and "path" in result
        assert not workspace.changes()


def test_write_schema_enum_compatibility_and_extra_argument_rejection() -> None:
    with TemporaryFileTools({"one.md": "first"}) as workspace:
        tool = bind(workspace, "write_file")
        for name in ("path", "expected_revision"):
            assert set(tool.parameters["properties"][name]) == {"type", "enum"}
        assert tool.parameters["properties"]["content"]["minLength"] == 1
        result = invoke(
            tool,
            {},
            path="one.md",
            content="valid",
            expected_revision=text_revision("first"),
            unexpected=True,
        )
        assert "unexpected" in result and not workspace.written


@pytest.mark.parametrize(
    "invalid",
    [
        {"old_start": 0},
        {"old_start": True},
        {"old_count": -1},
        {"old_count": "1"},
        {"expected_revision": ""},
        {"old_text": "first"},
    ],
)
def test_inferred_replacement_constraints_reject_invalid_arguments(
    invalid: dict[str, object],
) -> None:
    with TemporaryFileTools({"one.md": "first"}) as workspace:
        definition = bind(workspace, "replace_range")
        result = invoke(definition, {}, **{**edit("first", "updated"), **invalid})
        feedback = json.loads(result.removeprefix("error: invalid arguments: "))
        assert feedback["executed"] is False
        assert feedback["errors"][0]["loc"] == list(invalid)
        assert not workspace.written and workspace.read("one.md") == "first"


@pytest.mark.parametrize("ranges", [[], [[1, 1]], [[1, 2]]])
def test_only_fully_read_source_is_acknowledged_after_edit(ranges: list[list[int]]) -> None:
    before, after = "first\nsecond\n", "FIRST\ninserted\nsecond\n"
    state: State = {
        "context_reads": {"one.md": {"revision": text_revision(before), "ranges": ranges}}
    }
    acknowledge_context_write(state, "one.md", before, after)
    if ranges == [[1, 2]]:
        assert state["context_reads"]["one.md"] == {
            "revision": text_revision(after),
            "ranges": [[1, 3]],
        }
    else:
        assert state["context_reads"]["one.md"]["revision"] == text_revision(before)
    unread: State = {}
    acknowledge_context_write(unread, "unread.md", before, after)
    assert not unread


@pytest.mark.parametrize("ending", ["success", "failure", "cancellation"])
def test_workspace_lifetime_spans_rounds_and_cleans_up_on_exit(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    roots: list[Path] = []
    calls: list[AgentCall] = []
    ready = asyncio.Event()

    async def prepare(call: AgentCall) -> None:
        calls.append(call)
        workspace = call.invocation_data.get("workspace")
        if not isinstance(workspace, TemporaryFileTools):
            workspace = call.invocation_resources.enter_context(
                TemporaryFileTools({"one.md": "first"})
            )
            call.invocation_data["workspace"] = workspace
            roots.append(workspace.root)
        call.tools.append(bind(workspace, "replace_range"))

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        call = calls[-1]
        workspace = call.invocation_data["workspace"]
        assert isinstance(workspace, TemporaryFileTools) and workspace.root.exists()
        if call.turn == 1:
            return _outcome(
                tool_calls=(
                    ToolCall(
                        id="edit",
                        type="function",
                        function={
                            "name": "replace_range",
                            "arguments": json.dumps(edit("first", "updated")),
                        },
                    ),
                )
            )
        assert workspace.read("one.md") == "updated"
        if ending == "failure":
            raise RuntimeError("transport failed")
        if ending == "cancellation":
            ready.set()
            await asyncio.Future[None]()
        return _outcome(text="done")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)

    async def run() -> None:
        agent = create_structured_agent(llm_config=SampleLlmConfig())
        agent.events.subscribe(BEFORE_CALL, prepare)
        task = asyncio.create_task(agent.invoke(StructuredInput(prompt="edit")))
        if ending == "cancellation":
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "failure":
            with pytest.raises(RuntimeError, match="transport failed"):
                await task
        else:
            assert (await task).text == "done"
            await agent.invoke(StructuredInput(prompt="again"))

    asyncio.run(run())
    assert all(not root.exists() for root in roots)
    assert calls[0].invocation_data is calls[1].invocation_data
    if ending == "success":
        assert len(roots) == 2 and calls[0].invocation_data is not calls[-1].invocation_data


def test_concurrent_invocations_have_independent_working_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspaces: list[TemporaryFileTools] = []
    both_started = asyncio.Event()

    async def prepare(call: AgentCall) -> None:
        workspace = call.invocation_resources.enter_context(
            TemporaryFileTools({"one.md": call.messages[-1]["content"]})
        )
        call.invocation_data["workspace"] = workspace
        workspaces.append(workspace)
        if len(workspaces) == 2:
            both_started.set()
        await both_started.wait()
        assert workspace.read("one.md") == call.messages[-1]["content"]

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        return _outcome(text="done")

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)

    async def run() -> None:
        agent = create_structured_agent(llm_config=SampleLlmConfig())
        agent.events.subscribe(BEFORE_CALL, prepare)
        await asyncio.gather(
            agent.invoke(StructuredInput(prompt="first")),
            agent.invoke(StructuredInput(prompt="second")),
        )

    asyncio.run(run())
    assert len({workspace.root for workspace in workspaces}) == 2
    assert all(not workspace.root.exists() for workspace in workspaces)
