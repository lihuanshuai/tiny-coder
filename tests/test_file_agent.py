from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from tiny_coder import file_agent
from tiny_coder.file_agent import (
    AgentContext,
    BasicFileAgent,
    LlmCallOutcome,
    SyncAgentResult,
    _agent_input_snapshots,
    _parse_structured_sync_output,
    content_to_yaml_text,
    read_agent_file,
)
from tiny_coder.plugins import (
    AfterRunPlugin,
    BeforeRunPlugin,
    DynamicOutputPathsPlugin,
    ExistingPathGuardPlugin,
    FileCleanupPlugin,
    FileTreeInputPathsPlugin,
    JsonFieldStreamLlmCallPlugin,
    LabeledFileMapWriterPlugin,
    LlmOutputResultPlugin,
    LlmSessionTurnPlugin,
    ResponseOutputTypePlugin,
    RetryPlugin,
    SilentLlmCallPlugin,
    StaticInputPathsPlugin,
    StaticOutputPathsPlugin,
    StaticSystemPromptPlugin,
    TemplateSystemPromptPlugin,
    TemplateUserPromptPlugin,
    TextReplacementFileWriterPlugin,
    resolve_agent_file_path,
)
from tiny_coder.text_replacement import TextReplacement, TextReplacementFilePatch


class SampleOutput(BaseModel):
    summary: str
    content: dict[str, Any]
    optional: str | None = None


class SampleFileMapOutput(BaseModel):
    summary: str = ""
    files: dict[str, str]

    def to_file_map(self, context: AgentContext) -> dict[Path, str]:
        return {
            resolve_agent_file_path(context.cwd, Path(label)): content
            for label, content in self.files.items()
        }


class OtherFileMapOutput(BaseModel):
    files: dict[str, str]

    def to_file_map(self, context: AgentContext) -> dict[Path, str]:
        return {
            resolve_agent_file_path(context.cwd, Path(label)): content
            for label, content in self.files.items()
        }


class SampleTextReplacementOutput(BaseModel):
    from_text: str
    to_text: str
    report: str | None = None

    def to_text_replacement_file_patch(self) -> TextReplacementFilePatch:
        return TextReplacementFilePatch(
            replacements=[TextReplacement(from_text=self.from_text, to_text=self.to_text)],
            report=self.report,
        )


class OtherTextReplacementOutput(SampleTextReplacementOutput):
    pass


class SampleLlmConfig(BaseModel):
    model: str = "test-model"


class SampleLlmOutcome(BaseModel):
    text: str
    prompt_eval_count: int
    eval_count: int


async def _unused_llm_call(
    *,
    llm_cfg: BaseModel,
    system: str,
    prompt: str,
    response_format: dict[str, Any],
    on_chunk: Callable[[str], None],
) -> LlmCallOutcome:
    _ = llm_cfg, system, prompt, response_format, on_chunk
    raise AssertionError("test should not call the LLM")


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    root = Path(".test-tmp") / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_test_file(root: Path, path: Path, content: str) -> Path:
    target = resolve_agent_file_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return target


def test_agent_file_helpers_stay_inside_root(workspace_tmp_path: Path) -> None:
    written = _write_test_file(workspace_tmp_path, Path("notes/output.txt"), "hello\n")

    assert written == (workspace_tmp_path / "notes" / "output.txt").resolve()
    assert read_agent_file(workspace_tmp_path, Path("notes/output.txt")) == "hello\n"

    with pytest.raises(ValueError, match="outside agent workspace"):
        resolve_agent_file_path(workspace_tmp_path, Path("../outside.txt"))


def test_parse_structured_sync_output_validates_model() -> None:
    output = _parse_structured_sync_output(
        '{"summary": "done", "content": {"title": "Tiny"}}',
        output_type=SampleOutput,
    )

    assert output.summary == "done"
    assert output.content == {"title": "Tiny"}


def test_content_to_yaml_text_converts_pydantic_models() -> None:
    output = SampleOutput(summary="done", content={"title": "Tiny"}, optional=None)

    assert content_to_yaml_text(output) == "summary: done\ncontent:\n  title: Tiny\n"


def test_agent_input_snapshots_include_labels_and_languages(workspace_tmp_path: Path) -> None:
    _write_test_file(workspace_tmp_path, Path("docs/readme.md"), "# Title\n")
    _write_test_file(workspace_tmp_path, Path("data/config.yaml"), "name: tiny\n")

    snapshots = _agent_input_snapshots(
        workspace_tmp_path,
        [
            resolve_agent_file_path(workspace_tmp_path, Path("docs/readme.md")),
            resolve_agent_file_path(workspace_tmp_path, Path("data/config.yaml")),
        ],
    )

    assert snapshots == [
        {"label": "docs/readme.md", "language": "markdown", "content": "# Title"},
        {"label": "data/config.yaml", "language": "yaml", "content": "name: tiny"},
    ]


def test_static_input_paths_plugin_resolves_workspace_paths(workspace_tmp_path: Path) -> None:
    plugin = StaticInputPathsPlugin(
        [
            Path("docs/readme.md"),
            workspace_tmp_path / "data" / "config.yaml",
        ]
    )

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[plugin],
    )

    expected = [
        (workspace_tmp_path / "docs" / "readme.md").resolve(),
        (workspace_tmp_path / "data" / "config.yaml").resolve(),
    ]
    assert agent.input_paths == expected
    assert agent.context.input_paths == expected


def test_static_output_paths_plugin_resolves_workspace_paths(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[StaticOutputPathsPlugin([Path("docs/output.md")])],
    )

    assert list(agent.output_paths()) == [(workspace_tmp_path / "docs" / "output.md").resolve()]


def test_dynamic_output_paths_plugin_copies_registered_input_paths(
    workspace_tmp_path: Path,
) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            StaticInputPathsPlugin([Path("docs/readme.md"), Path("data/config.yaml")]),
            DynamicOutputPathsPlugin(),
        ],
    )

    expected = [
        (workspace_tmp_path / "docs" / "readme.md").resolve(),
        (workspace_tmp_path / "data" / "config.yaml").resolve(),
    ]
    assert list(agent.output_paths()) == expected
    assert agent.context.output_paths is not agent.context.input_paths


def test_existing_path_guard_plugin_rejects_existing_static_path(
    workspace_tmp_path: Path,
) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[StaticOutputPathsPlugin([Path("blocked.txt")]), ExistingPathGuardPlugin()],
    )

    _write_test_file(workspace_tmp_path, Path("blocked.txt"), "keep\n")

    assert agent.context.allow_overwrite_existing_paths is False
    assert agent.context.output_paths == [
        resolve_agent_file_path(workspace_tmp_path, Path("blocked.txt"))
    ]
    with pytest.raises(RuntimeError, match="refusing to overwrite existing path: blocked.txt"):
        asyncio.run(agent.call_llm_and_apply_output(system_prompt="", user_prompt=""))


def test_file_cleanup_plugin_injects_and_removes_files(
    workspace_tmp_path: Path,
) -> None:
    stale_path = _write_test_file(workspace_tmp_path, Path("stale.jsonl"), "old\n")
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[FileCleanupPlugin([Path("stale.jsonl")])],
    )

    assert agent.context.clean_up_paths == [stale_path]
    with pytest.raises(RuntimeError, match="agent requires at least one system prompt hook"):
        asyncio.run(agent.run())
    assert not stale_path.exists()


def test_file_cleanup_plugin_rejects_directories(workspace_tmp_path: Path) -> None:
    directory = workspace_tmp_path / "cache"
    directory.mkdir()
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[FileCleanupPlugin([directory])],
    )

    with pytest.raises(RuntimeError, match="cleanup path is a directory: cache"):
        asyncio.run(agent.run())


def test_file_tree_input_paths_plugin_resolves_ordered_tree_paths(
    workspace_tmp_path: Path,
) -> None:
    _write_test_file(workspace_tmp_path, Path("plan/overview.md"), "# Overview\n")
    _write_test_file(workspace_tmp_path, Path("plan/002.md"), "two\n")
    _write_test_file(workspace_tmp_path, Path("plan/001.md"), "one\n")
    _write_test_file(workspace_tmp_path, Path("plan/notes.txt"), "ignored\n")

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            FileTreeInputPathsPlugin(
                Path("plan"),
                first_paths=[Path("overview.md")],
                patterns=["[0-9][0-9][0-9].md"],
            )
        ],
    )

    assert [path.name for path in agent.input_paths] == ["overview.md", "001.md", "002.md"]


def test_response_output_type_plugin_registers_context_model(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[ResponseOutputTypePlugin(SampleOutput)],
    )

    assert agent.context.llm_response_output_type is SampleOutput
    assert agent.response_output_type() is SampleOutput


def test_file_agent_records_parsed_response_output(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            ResponseOutputTypePlugin(SampleFileMapOutput),
            LabeledFileMapWriterPlugin(),
        ],
    )

    agent._apply_output(
        {"raw_output": '{"summary":"done","files":{"notes/output.txt":"hello\\n"}}'}
    )

    assert agent.context.llm_response_output == SampleFileMapOutput(
        summary="done",
        files={"notes/output.txt": "hello\n"},
    )


def test_file_agent_records_llm_call_outcome(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = SampleLlmOutcome(
        text='{"summary":"done","content":{}}',
        prompt_eval_count=3,
        eval_count=4,
    )

    async def llm_call(**_kwargs: Any) -> LlmCallOutcome:
        return outcome

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[ResponseOutputTypePlugin(SampleOutput)],
    )

    asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert agent.context.llm_call_outcome == outcome
    assert agent.context.llm_call_system_prompt == "system"
    assert agent.context.llm_call_user_prompt == "prompt"


def test_llm_session_turn_plugin_appends_call(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = SampleLlmOutcome(
        text="原始响应",
        prompt_eval_count=3,
        eval_count=4,
    )

    async def llm_call(**_kwargs: Any) -> LlmCallOutcome:
        return outcome

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    plugin = LlmSessionTurnPlugin[dict[str, Any]](
        path=Path("sessions/agent.jsonl"),
        turn_factory=lambda context, call_outcome: {
            "cwd": context.cwd.name,
            "text": call_outcome.text,
        },
    )
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[ResponseOutputTypePlugin(SampleOutput), plugin],
    )

    asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    target = (workspace_tmp_path / "sessions" / "agent.jsonl").resolve()
    assert [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()] == [
        {"cwd": workspace_tmp_path.name, "text": "原始响应"},
    ]


def test_llm_output_result_plugin_registers_context_callback(workspace_tmp_path: Path) -> None:
    received: list[tuple[AgentContext, SyncAgentResult]] = []

    async def handle_result(context: AgentContext, result: SyncAgentResult) -> None:
        received.append((context, result))

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[LlmOutputResultPlugin(handle_result)],
    )
    result = SyncAgentResult(summary="done", written_paths=[workspace_tmp_path / "output.txt"])

    asyncio.run(agent.context.llm_output_result_hooks[0](result))

    assert received == [(agent.context, result)]


def test_before_run_plugin_prepares_after_cleanup(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = workspace_tmp_path / "context.md"
    input_path.write_text("stale\n", encoding="utf-8", newline="\n")
    events: list[str] = []

    async def prepare(context: AgentContext) -> None:
        assert not input_path.exists()
        input_path.write_text(context.cwd.name, encoding="utf-8", newline="\n")
        events.append("prepare")

    async def finish(context: AgentContext, result: SyncAgentResult) -> None:
        assert context.cwd == workspace_tmp_path
        assert result.summary == "done"
        events.append("after")

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            FileCleanupPlugin([input_path]),
            BeforeRunPlugin(prepare),
            AfterRunPlugin(finish),
        ],
    )
    expected = SyncAgentResult(summary="done", written_paths=[])

    async def run_agent() -> SyncAgentResult:
        events.append("run")
        return expected

    monkeypatch.setattr(agent, "call_llm_and_apply_output_with_retries", run_agent)

    result = asyncio.run(agent.run())

    assert result is expected
    assert events == ["prepare", "run", "after"]
    assert input_path.read_text(encoding="utf-8") == workspace_tmp_path.name


def test_after_run_plugin_failure_does_not_retry_agent(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fail_after_run(_context: AgentContext, _result: SyncAgentResult) -> None:
        raise ValueError("post-run failure")

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[AfterRunPlugin(fail_after_run)],
    )

    async def run_agent() -> SyncAgentResult:
        nonlocal calls
        calls += 1
        return SyncAgentResult(summary="done", written_paths=[])

    monkeypatch.setattr(agent, "call_llm_and_apply_output_with_retries", run_agent)

    with pytest.raises(ValueError, match="post-run failure"):
        asyncio.run(agent.run())

    assert calls == 1


def test_json_field_stream_llm_call_registers_wrapped_call(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_agent, "stream_llm_chat_format", _unused_llm_call)

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            JsonFieldStreamLlmCallPlugin(
                field_name="summary",
                raw_abort_threshold=100,
            ),
        ],
    )

    assert agent.context.llm_call is not _unused_llm_call


def test_silent_llm_call_suppresses_chunks_and_preserves_call(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def llm_call(
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        on_chunk: Callable[[str], None],
    ) -> LlmCallOutcome:
        calls.append(
            {
                "llm_cfg": llm_cfg,
                "system": system,
                "prompt": prompt,
                "response_format": response_format,
            }
        )
        on_chunk("hidden")
        return SampleLlmOutcome(text='{"summary":"done"}', prompt_eval_count=3, eval_count=4)

    llm_config = SampleLlmConfig()
    response_format = {"type": "object"}
    streamed: list[str] = []
    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=llm_config,
        plugins=[SilentLlmCallPlugin()],
    )
    registered_llm_call = agent.context.llm_call
    assert registered_llm_call is not None

    outcome = asyncio.run(
        registered_llm_call(
            llm_cfg=llm_config,
            system="system",
            prompt="prompt",
            response_format=response_format,
            on_chunk=streamed.append,
        )
    )

    assert streamed == []
    assert calls == [
        {
            "llm_cfg": llm_config,
            "system": "system",
            "prompt": "prompt",
            "response_format": response_format,
        }
    ]
    assert outcome.text == '{"summary":"done"}'


def test_retry_llm_call_retries_transient_errors(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def llm_call(**_kwargs: Any) -> LlmCallOutcome:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("temporary failure")
        return SampleLlmOutcome(text="done", prompt_eval_count=1, eval_count=2)

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[ResponseOutputTypePlugin(SampleOutput), RetryPlugin()],
    )

    result = asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert calls == 3
    assert agent.context.max_attempts == 3
    assert result["raw_output"] == "done"


def test_retry_llm_call_does_not_retry_deterministic_errors(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def llm_call(**_kwargs: Any) -> LlmCallOutcome:
        nonlocal calls
        calls += 1
        raise ValueError("invalid request")

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[ResponseOutputTypePlugin(SampleOutput), RetryPlugin()],
    )

    with pytest.raises(ValueError, match="invalid request"):
        asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert calls == 1


def test_retry_llm_call_stops_after_configured_attempts(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def llm_call(**_kwargs: Any) -> LlmCallOutcome:
        nonlocal calls
        calls += 1
        raise OSError("still unavailable")

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            ResponseOutputTypePlugin(SampleOutput),
            RetryPlugin(max_attempts=2),
        ],
    )

    with pytest.raises(RuntimeError, match=r"LLM call failed after 2 attempt\(s\)"):
        asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert calls == 2


def test_retry_plugin_retries_invalid_structured_output(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_test_file(
        workspace_tmp_path,
        Path("user.jinja"),
        "{{ retry_errors | join(':') }}",
    )
    prompts: list[str] = []

    async def llm_call(**kwargs: Any) -> LlmCallOutcome:
        prompts.append(kwargs["prompt"])
        if len(prompts) == 1:
            return SampleLlmOutcome(text="not json", prompt_eval_count=1, eval_count=1)
        return SampleLlmOutcome(
            text='{"files":{"output.txt":"done\\n"}}',
            prompt_eval_count=1,
            eval_count=1,
        )

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            StaticOutputPathsPlugin([Path("output.txt")]),
            ResponseOutputTypePlugin(SampleFileMapOutput),
            RetryPlugin(max_attempts=2),
            StaticSystemPromptPlugin("system"),
            TemplateUserPromptPlugin(
                "user.jinja",
                template_root=workspace_tmp_path,
            ),
            LabeledFileMapWriterPlugin(),
        ],
    )

    result = asyncio.run(agent.run())

    assert result.written_paths == [(workspace_tmp_path / "output.txt").resolve()]
    assert read_agent_file(workspace_tmp_path, Path("output.txt")) == "done\n"
    assert prompts[0] == "\n"
    assert "invalid structured sync output" in prompts[1]


def test_json_field_stream_llm_call_streams_field_and_preserves_outcome(
    workspace_tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = ['{"summary":"hello', r"\nwor", 'ld","content":{}}']
    raw_output = "".join(chunks)
    calls: list[dict[str, Any]] = []

    async def llm_call(
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        on_chunk: Callable[[str], None],
    ) -> LlmCallOutcome:
        calls.append(
            {
                "llm_cfg": llm_cfg,
                "system": system,
                "prompt": prompt,
                "response_format": response_format,
            }
        )
        for chunk in chunks:
            on_chunk(chunk)
        return SampleLlmOutcome(
            text=raw_output,
            prompt_eval_count=3,
            eval_count=4,
        )

    plugin = JsonFieldStreamLlmCallPlugin(
        field_name="summary",
        raw_abort_threshold=1_000,
    )
    llm_config = SampleLlmConfig()
    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=llm_config,
        plugins=[
            ResponseOutputTypePlugin(SampleOutput),
            plugin,
        ],
    )
    outcome = asyncio.run(
        agent.context.llm_call(
            llm_cfg=llm_config,
            system="system",
            prompt="prompt",
            response_format=SampleOutput.model_json_schema(),
            on_chunk=lambda _chunk: None,
        )
    )

    assert outcome == SampleLlmOutcome(text=raw_output, prompt_eval_count=3, eval_count=4)
    assert capsys.readouterr().out == "hello\nworld\n"
    assert calls == [
        {
            "llm_cfg": llm_config,
            "system": "system",
            "prompt": "prompt",
            "response_format": SampleOutput.model_json_schema(),
        }
    ]
    assert outcome.prompt_eval_count == 3


def test_json_field_stream_llm_call_aborts_over_raw_threshold(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def llm_call(
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        on_chunk: Callable[[str], None],
    ) -> LlmCallOutcome:
        _ = llm_cfg, system, prompt, response_format
        on_chunk("12345")
        raise AssertionError("stream abort should stop the call")

    plugin = JsonFieldStreamLlmCallPlugin(
        field_name="summary",
        raw_abort_threshold=4,
    )
    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            ResponseOutputTypePlugin(SampleOutput),
            plugin,
        ],
    )
    with pytest.raises(ValueError, match=r"5 characters \(limit 4\)"):
        asyncio.run(
            agent.context.llm_call(
                llm_cfg=agent.context.llm_config,
                system="system",
                prompt="prompt",
                response_format=SampleOutput.model_json_schema(),
                on_chunk=lambda _chunk: None,
            )
        )


def test_labeled_file_map_writer_requires_compatible_output_type(
    workspace_tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match=r"must implement to_file_map\(\)"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                ResponseOutputTypePlugin(SampleOutput),
                LabeledFileMapWriterPlugin(),
            ],
        )


def test_labeled_file_map_writer_resolves_and_writes_output(
    workspace_tmp_path: Path,
) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            StaticOutputPathsPlugin([Path("notes/output.txt")]),
            ResponseOutputTypePlugin(SampleFileMapOutput),
            LabeledFileMapWriterPlugin(),
        ],
    )

    written = agent.split_and_write_output(
        SampleFileMapOutput(files={"notes/output.txt": "hello\n"})
    )

    assert written == [(workspace_tmp_path / "notes" / "output.txt").resolve()]
    assert read_agent_file(workspace_tmp_path, Path("notes/output.txt")) == "hello\n"

    with pytest.raises(ValueError, match="output file is not allowed"):
        agent.split_and_write_output(SampleFileMapOutput(files={"notes/other.txt": "no"}))

    with pytest.raises(ValueError, match="expected SampleFileMapOutput.*got OtherFileMapOutput"):
        agent.split_and_write_output(OtherFileMapOutput(files={"notes/output.txt": "wrong"}))


def test_file_agent_accepts_subset_of_allowed_output_paths(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            StaticOutputPathsPlugin([Path("notes/output.txt"), Path("notes/unchanged.txt")]),
            ResponseOutputTypePlugin(SampleFileMapOutput),
            LabeledFileMapWriterPlugin(),
        ],
    )

    state = agent._apply_output({"raw_output": '{"files":{"notes/output.txt":"hello\\n"}}'})

    assert state["written_paths"] == ["notes/output.txt"]
    assert read_agent_file(workspace_tmp_path, Path("notes/output.txt")) == "hello\n"
    assert not (workspace_tmp_path / "notes" / "unchanged.txt").exists()


def test_text_replacement_file_writer_requires_compatible_output_type(
    workspace_tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match=r"must implement to_text_replacement_file_patch\(\)",
    ):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                ResponseOutputTypePlugin(SampleOutput),
                TextReplacementFileWriterPlugin(target_path=Path("draft.txt")),
            ],
        )


def test_text_replacement_file_writer_patches_target_and_writes_report(
    workspace_tmp_path: Path,
) -> None:
    target = _write_test_file(workspace_tmp_path, Path("draft.txt"), "old text\n")
    report = workspace_tmp_path / "review.json"
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            StaticOutputPathsPlugin([target, report]),
            ResponseOutputTypePlugin(SampleTextReplacementOutput),
            TextReplacementFileWriterPlugin(
                target_path=target,
                report_path=report,
            ),
        ],
    )

    written = agent.split_and_write_output(
        SampleTextReplacementOutput(
            from_text="old text",
            to_text="new text",
            report='{"status":"pass"}\n',
        )
    )

    assert written == [target, report]
    assert target.read_text(encoding="utf-8") == "new text\n"
    assert report.read_text(encoding="utf-8") == '{"status":"pass"}\n'

    with pytest.raises(ValueError, match="did not provide report content"):
        agent.split_and_write_output(
            SampleTextReplacementOutput(
                from_text="new text",
                to_text="partially written text",
            )
        )
    assert target.read_text(encoding="utf-8") == "new text\n"

    with pytest.raises(
        ValueError,
        match="expected SampleTextReplacementOutput.*got OtherTextReplacementOutput",
    ):
        agent.split_and_write_output(
            OtherTextReplacementOutput(
                from_text="new text",
                to_text="other text",
                report="{}\n",
            )
        )


def test_text_replacement_file_writer_ignores_unchanged_replacement(
    workspace_tmp_path: Path,
) -> None:
    target = _write_test_file(workspace_tmp_path, Path("draft.txt"), "same text\n")
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            ResponseOutputTypePlugin(SampleTextReplacementOutput),
            TextReplacementFileWriterPlugin(target_path=target),
        ],
    )

    written = agent.split_and_write_output(
        SampleTextReplacementOutput(
            from_text="same\r\ntext",
            to_text="same\ntext",
        )
    )

    assert written == [target]
    assert target.read_text(encoding="utf-8") == "same text\n"


def test_static_system_prompt_plugin_uses_context_vars(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[StaticSystemPromptPlugin(lambda context: f"SYSTEM:{context.cwd.name}")],
    )

    assert agent.build_system_prompt("") == f"SYSTEM:{workspace_tmp_path.name}\n"


def test_file_agent_requires_system_prompt_hook(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
    )

    with pytest.raises(RuntimeError, match="system prompt hook"):
        agent.build_system_prompt("")


def test_template_system_prompt_plugin_uses_context_vars(
    workspace_tmp_path: Path,
) -> None:
    _write_test_file(workspace_tmp_path, Path("system.jinja"), "Contract for {{ cwd }}")
    plugin = TemplateSystemPromptPlugin(
        "system.jinja",
        template_root=workspace_tmp_path,
        template_vars=lambda context, _task_prompt: {"cwd": context.cwd.name},
    )
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[plugin],
    )

    assert agent.build_system_prompt("") == f"Contract for {workspace_tmp_path.name}\n"


def test_file_agent_requires_user_prompt_hook(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
    )

    with pytest.raises(RuntimeError, match="user prompt hook"):
        agent.build_user_prompt("")


def test_template_user_prompt_plugin_adds_context_and_common_variables(
    workspace_tmp_path: Path,
) -> None:
    _write_test_file(workspace_tmp_path, Path("docs/readme.md"), "# Readme\n")
    _write_test_file(workspace_tmp_path, Path("data/config.yaml"), "name: tiny\n")
    _write_test_file(
        workspace_tmp_path,
        Path("user.jinja"),
        "{{ file_count }}:{{ retry_errors[-1] if retry_errors else '' }}",
    )
    plugin = TemplateUserPromptPlugin(
        "user.jinja",
        template_root=workspace_tmp_path,
        template_vars=lambda context: {
            "file_count": len(context.input_paths),
        },
    )
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            StaticInputPathsPlugin([Path("docs/readme.md"), Path("data/config.yaml")]),
            plugin,
        ],
    )

    assert agent.build_user_prompt("", retry_errors=["bad json"]) == "2:bad json\n"


def test_template_user_prompt_plugin_accepts_static_variables(
    workspace_tmp_path: Path,
) -> None:
    _write_test_file(
        workspace_tmp_path,
        Path("user.jinja"),
        "{{ greeting }} {{ input_files | length }}",
    )
    plugin = TemplateUserPromptPlugin(
        "user.jinja",
        template_root=workspace_tmp_path,
        template_vars={"greeting": "hello"},
    )
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[plugin],
    )

    assert plugin.build_user_prompt(agent.context, "task", "current") == "hello 0\n"
