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
    FileReplacement,
    IneffectiveReplacementError,
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
    if operation != "replace_lines":
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
    return dict(path=path, old_start=start, old_count=1, old_text=old, new_text=new)


def test_native_async_methods_edit_and_preserve_failed_write_progress(tmp_path: Path) -> None:
    async def run() -> None:
        with TemporaryFileTools({"one.md": "first\n"}, directory=tmp_path / ".cache") as workspace:
            await workspace.replace_lines({}, "one.md", 1, 1, "first", "FIRST")
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


def test_sequential_replacements_use_latest_content_and_shifted_lines(tmp_path: Path) -> None:
    path = tmp_path / "one.md"
    path.write_text("first\nsecond\n", encoding="utf-8", newline="\n")
    with TemporaryFileTools({"one.md": "first\nsecond\n"}) as workspace:
        root = workspace.root
        tool = bind(workspace, "replace_lines")
        state: State = {}
        first = json.loads(invoke(tool, state, **edit("first", "FIRST\ninserted")))
        assert first["applied"] and first["temporary"] and first["total_lines"] == 3
        invoke(tool, state, **edit("second", "SECOND", start=3))
        invoke(tool, state, **edit("inserted", "revised", start=2))
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
def test_identical_retry_acknowledges_success_without_reapplying(new: str) -> None:
    with TemporaryFileTools({"one.md": "first\nsecond\n"}) as workspace:
        proposal = edit("first", new)
        tool = bind(workspace, "replace_lines")
        first = json.loads(invoke(tool, {}, **proposal))
        saved = workspace.read("one.md")
        # Schemas and tool handlers are recreated every model turn.
        retry = json.loads(invoke(bind(workspace, "replace_lines"), {}, **proposal))
        assert retry["already_applied"] and not retry["changed"]
        assert retry["revision"] == first["revision"]
        assert "Do not repeat" in retry["next_action"]
        assert workspace.read("one.md") == saved
        for _ in range(2):
            repeated = invoke(bind(workspace, "replace_lines"), {}, **proposal)
            assert (
                repeated.startswith("error:")
                and "already succeeded and was acknowledged" in repeated
            )
            assert "submit_result" in repeated
            assert workspace.read("one.md") == saved
            assert workspace.recovery_path is None


@pytest.mark.parametrize("old,new", [("first", "first"), ("first", "first\n")])
def test_unchanged_replacement_is_rejected_without_writing_or_forcing_reread(
    old: str, new: str
) -> None:
    async def validate(state: State, path: str, content: str) -> None:
        pytest.fail("an unchanged replacement must not reach candidate validation")

    def feedback(proposal: FileReplacement, error: ValueError) -> str:
        assert isinstance(error, IneffectiveReplacementError)
        return f"{proposal.path}: no actual change; rewrite new_text or submit_result"

    with TemporaryFileTools({"one.md": "first\nsecond\n"}, validate=validate) as workspace:
        error = invoke(bind(workspace, "replace_lines"), {}, **edit(old, new))
        assert error.startswith("error:") and "unchanged" in error
        assert "Rewrite new_text" in error and "submit_result" in error
        workspace.replacement_error = feedback
        error = invoke(bind(workspace, "replace_lines"), {}, **edit(old, new))
        assert error == "error: one.md: no actual change; rewrite new_text or submit_result"
        assert workspace.recovery_path is None
        assert not workspace.written and not workspace.changes()
        assert workspace.read("one.md") == "first\nsecond\n"


def test_later_writes_and_distinct_ranges_are_not_mistaken_for_retries() -> None:
    with TemporaryFileTools({"one.md": "first\nfirst\n"}) as workspace:
        tool = bind(workspace, "replace_lines")
        proposal = edit("first", "FIRST")
        invoke(tool, {}, **proposal)
        other = json.loads(invoke(tool, {}, **{**proposal, "old_start": 2}))
        assert other["changed"] and not other.get("already_applied")
        assert workspace.read("one.md") == "FIRST\nFIRST\n"
        invoke(
            bind(workspace, "write_file"),
            {},
            path="one.md",
            content="first\nfirst\n",
            expected_revision=text_revision("FIRST\nFIRST\n"),
        )
        reapplied = json.loads(invoke(tool, {}, **proposal))
        assert reapplied["changed"] and not reapplied.get("already_applied")


def test_rejections_preserve_progress_and_consumer_feedback() -> None:
    async def validate(state: State, path: str, content: str) -> None:
        if "forbidden" in content:
            raise ValueError("forbidden content: preserve existing words")

    def feedback(proposal: FileReplacement, error: ValueError) -> str:
        return f"{proposal.path}:{proposal.old_start}: {error}; previous progress preserved"

    with TemporaryFileTools(
        {"one.md": "first\nsecond\n"}, validate=validate, replacement_error=feedback
    ) as workspace:
        tool = bind(workspace, "replace_lines")
        invoke(tool, {}, **edit("first", "FIRST"))
        earlier = workspace.read("one.md")
        error = invoke(tool, {}, **edit("missing", "different"))
        assert error.startswith("error: one.md:1:") and "progress preserved" in error
        assert workspace.recovery_path == "one.md"
        assert "preserve existing words" in invoke(tool, {}, **edit("FIRST", "forbidden"))
        assert workspace.read("one.md") == earlier
        invoke(tool, {}, **edit("second", "SECOND", start=2))
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
        assert len(list(workspace.root.iterdir())) == 3
        assert not (workspace.root / "@output").exists()


@pytest.mark.parametrize("operation", ["write_file", "replace_lines"])
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
        {"old_text": ""},
    ],
)
def test_inferred_replacement_constraints_reject_invalid_arguments(
    invalid: dict[str, object],
) -> None:
    with TemporaryFileTools({"one.md": "first"}) as workspace:
        definition = bind(workspace, "replace_lines")
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
        call.tools.append(bind(workspace, "replace_lines"))

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
                            "name": "replace_lines",
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
