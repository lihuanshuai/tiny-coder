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
    LlmCallOutcome,
    SyncAgentResult,
    _agent_input_snapshots,
    _parse_structured_sync_output,
    content_to_yaml_text,
    read_agent_file,
)
from tiny_coder.file_agent import (
    BasicFileAgent as _BasicFileAgent,
)
from tiny_coder.plugins import (
    AfterIterationPlugin,
    AfterLlmRequestPlugin,
    AfterRunPlugin,
    AgentRetryPolicyPlugin,
    BeforeIterationPlugin,
    BeforeLlmRequestPlugin,
    BeforeRunPlugin,
    ConditionalLlmRequestPlugin,
    DynamicOutputPathsPlugin,
    ExistingPathGuardPlugin,
    FileCleanupPlugin,
    FileTreeInputPathsPlugin,
    IterativeRunPlugin,
    JsonFieldStreamLlmCallPlugin,
    LabeledFileMapWriterPlugin,
    LlmCallJsonlRecorderPlugin,
    LlmConfigPlugin,
    LlmOutputResultPlugin,
    LlmRequestGroupPlugin,
    ResponseOutputTypePlugin,
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


class SamplePlanOutput(BaseModel):
    plan: str

    def to_file_map(self, context: AgentContext) -> dict[Path, str]:
        _ = context
        return {}


class SampleReviewOutput(BaseModel):
    review: str

    def to_file_map(self, context: AgentContext) -> dict[Path, str]:
        _ = context
        return {}


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


def BasicFileAgent(
    *,
    cwd: Path,
    llm_config: BaseModel,
    plugins: list[Any] | None = None,
) -> _BasicFileAgent:
    """Build a test agent with the required plugin-only model configuration."""
    return _BasicFileAgent(
        cwd=cwd,
        plugins=[LlmConfigPlugin(llm_config), *(plugins or [])],
    )


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
        plugins=[LlmRequestGroupPlugin(key="paths"), plugin],
    )

    expected = [
        (workspace_tmp_path / "docs" / "readme.md").resolve(),
        (workspace_tmp_path / "data" / "config.yaml").resolve(),
    ]
    assert agent.input_paths == expected
    assert agent.context.llm_request_contexts["paths"].input_paths == expected


def test_static_output_paths_plugin_resolves_workspace_paths(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="paths"),
            StaticOutputPathsPlugin([Path("docs/output.md")]),
        ],
    )

    assert list(agent.output_paths()) == [(workspace_tmp_path / "docs" / "output.md").resolve()]


def test_dynamic_output_paths_plugin_copies_registered_input_paths(
    workspace_tmp_path: Path,
) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="paths"),
            StaticInputPathsPlugin([Path("docs/readme.md"), Path("data/config.yaml")]),
            DynamicOutputPathsPlugin(),
        ],
    )

    expected = [
        (workspace_tmp_path / "docs" / "readme.md").resolve(),
        (workspace_tmp_path / "data" / "config.yaml").resolve(),
    ]
    assert list(agent.output_paths()) == expected
    request_context = agent.context.llm_request_contexts["paths"]
    assert request_context.output_paths is not request_context.input_paths


def test_existing_path_guard_plugin_rejects_existing_static_path(
    workspace_tmp_path: Path,
) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="guard"),
            StaticOutputPathsPlugin([Path("blocked.txt")]),
            ExistingPathGuardPlugin(),
        ],
    )
    request_context = agent.context.llm_request_contexts["guard"]

    _write_test_file(workspace_tmp_path, Path("blocked.txt"), "keep\n")

    assert request_context.allow_overwrite_existing_paths is False
    assert request_context.output_paths == [
        resolve_agent_file_path(workspace_tmp_path, Path("blocked.txt"))
    ]
    with pytest.raises(RuntimeError, match="refusing to overwrite existing path: blocked.txt"):
        asyncio.run(agent.run())


def test_file_cleanup_plugin_injects_and_removes_files(
    workspace_tmp_path: Path,
) -> None:
    stale_path = _write_test_file(workspace_tmp_path, Path("stale.jsonl"), "old\n")
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            FileCleanupPlugin([Path("stale.jsonl")]),
            LlmRequestGroupPlugin(key="cleanup"),
        ],
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
        plugins=[
            FileCleanupPlugin([directory]),
            LlmRequestGroupPlugin(key="cleanup"),
        ],
    )

    with pytest.raises(RuntimeError, match="cleanup path is a directory: cache"):
        asyncio.run(agent.run())


def test_basic_file_agent_requires_llm_request_group(workspace_tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires at least one LlmRequestGroupPlugin"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
        )


def test_basic_file_agent_uses_llm_config_plugin(workspace_tmp_path: Path) -> None:
    llm_config = SampleLlmConfig()
    agent = _BasicFileAgent(
        cwd=workspace_tmp_path,
        plugins=[
            LlmConfigPlugin(llm_config),
            LlmRequestGroupPlugin(key="configured"),
        ],
    )

    assert not hasattr(agent, "llm_config")
    assert agent.context.llm_request_contexts["configured"].llm_config is llm_config


def test_llm_request_group_requires_llm_config_plugin(workspace_tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LLM request requires LlmConfigPlugin"):
        _BasicFileAgent(
            cwd=workspace_tmp_path,
            plugins=[LlmRequestGroupPlugin(key="missing-config")],
        )


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
            LlmRequestGroupPlugin(key="tree"),
            FileTreeInputPathsPlugin(
                Path("plan"),
                first_paths=[Path("overview.md")],
                patterns=["[0-9][0-9][0-9].md"],
            ),
        ],
    )

    assert [path.name for path in agent.input_paths] == ["overview.md", "001.md", "002.md"]


def test_response_output_type_plugin_registers_context_model(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="response"),
            ResponseOutputTypePlugin(SampleOutput),
        ],
    )

    assert agent.context.llm_request_contexts["response"].llm_response_output_type is SampleOutput
    assert agent.response_output_type() is SampleOutput


def test_file_agent_records_parsed_response_output(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="apply"),
            ResponseOutputTypePlugin(SampleFileMapOutput),
            LabeledFileMapWriterPlugin(),
        ],
    )
    request_context = agent.context.llm_request_contexts["apply"]
    agent.context = request_context

    agent._apply_output(
        {"raw_output": '{"summary":"done","files":{"notes/output.txt":"hello\\n"}}'}
    )

    assert request_context.llm_response_output == SampleFileMapOutput(
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
        plugins=[
            LlmRequestGroupPlugin(key="call"),
            ResponseOutputTypePlugin(SampleOutput),
        ],
    )
    request_context = agent.context.llm_request_contexts["call"]
    agent.context = request_context

    asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert request_context.llm_call_outcome == outcome
    assert request_context.llm_call_system_prompt == "system"
    assert request_context.llm_call_user_prompt == "prompt"


def test_llm_call_jsonl_recorder_records_every_call_with_custom_extra(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter(
        [
            SampleLlmOutcome(text="第一次响应", prompt_eval_count=3, eval_count=4),
            SampleLlmOutcome(text="第二次响应", prompt_eval_count=5, eval_count=6),
        ]
    )

    async def llm_call(**_kwargs: Any) -> LlmCallOutcome:
        return next(outcomes)

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    plugin = LlmCallJsonlRecorderPlugin(
        path=Path("sessions/agent.jsonl"),
        extra_handler=lambda context, _outcome: {
            "cwd": context.cwd.name,
        },
    )
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            plugin,
            LlmRequestGroupPlugin(key="session"),
            ResponseOutputTypePlugin(SampleOutput),
        ],
    )
    agent.context = agent.context.llm_request_contexts["session"]

    asyncio.run(agent._call_llm({"system_prompt": "system 1", "user_prompt": "prompt 1"}))
    asyncio.run(agent._call_llm({"system_prompt": "system 2", "user_prompt": "prompt 2"}))

    target = (workspace_tmp_path / "sessions" / "agent.jsonl").resolve()
    assert [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()] == [
        {
            "request_key": "session",
            "system": "system 1",
            "user": "prompt 1",
            "output": "第一次响应",
            "stats": {"prompt_eval_count": 3, "eval_count": 4},
            "extra": {"cwd": workspace_tmp_path.name},
        },
        {
            "request_key": "session",
            "system": "system 2",
            "user": "prompt 2",
            "output": "第二次响应",
            "stats": {"prompt_eval_count": 5, "eval_count": 6},
            "extra": {"cwd": workspace_tmp_path.name},
        },
    ]


def test_llm_output_result_plugin_registers_context_callback(workspace_tmp_path: Path) -> None:
    received: list[tuple[AgentContext, SyncAgentResult]] = []

    async def handle_result(context: AgentContext, result: SyncAgentResult) -> None:
        received.append((context, result))

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="result"),
            LlmOutputResultPlugin(handle_result),
        ],
    )
    request_context = agent.context.llm_request_contexts["result"]
    result = SyncAgentResult(summary="done", written_paths=[workspace_tmp_path / "output.txt"])

    asyncio.run(request_context.llm_output_result_hooks[0](result))

    assert received == [(request_context, result)]


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
            LlmRequestGroupPlugin(key="run"),
        ],
    )
    expected = SyncAgentResult(summary="done", written_paths=[])

    async def run_agent() -> SyncAgentResult:
        events.append("run")
        return expected

    monkeypatch.setattr(agent, "_run_llm_requests", run_agent)

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
        plugins=[
            AfterRunPlugin(fail_after_run),
            LlmRequestGroupPlugin(key="run"),
        ],
    )

    async def run_agent() -> SyncAgentResult:
        nonlocal calls
        calls += 1
        return SyncAgentResult(summary="done", written_paths=[])

    monkeypatch.setattr(agent, "_run_llm_requests", run_agent)

    with pytest.raises(ValueError, match="post-run failure"):
        asyncio.run(agent.run())

    assert calls == 1


def test_iterative_run_plugin_uses_context_items_and_aggregates_results(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int | None, object]] = []

    async def prepare(context: AgentContext, item: object) -> None:
        events.append(("before", context.iteration_index, item))

    async def finish(
        context: AgentContext,
        item: object,
        result: SyncAgentResult,
    ) -> None:
        assert result.summary == str(item)
        events.append(("after", context.iteration_index, item))

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            IterativeRunPlugin(items=["first", "second"]),
            BeforeIterationPlugin(prepare),
            AfterIterationPlugin(finish),
            LlmRequestGroupPlugin(key="iteration"),
        ],
    )

    async def run_iteration() -> SyncAgentResult:
        item = agent.context.iteration_item
        path = workspace_tmp_path / f"{item}.txt"
        return SyncAgentResult(summary=str(item), written_paths=[path])

    monkeypatch.setattr(agent, "_run_llm_requests", run_iteration)

    result = asyncio.run(agent.run())

    assert agent.context.iteration_items == ["first", "second"]
    assert agent.context.iteration_index == 1
    assert agent.context.iteration_item == "second"
    assert events == [
        ("before", 0, "first"),
        ("after", 0, "first"),
        ("before", 1, "second"),
        ("after", 1, "second"),
    ]
    assert result == SyncAgentResult(
        summary="",
        written_paths=[
            workspace_tmp_path / "first.txt",
            workspace_tmp_path / "second.txt",
        ],
    )


def test_iterative_run_plugin_rejects_multiple_iteration_sources(
    workspace_tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="only one IterativeRunPlugin"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                IterativeRunPlugin(items=[1]),
                IterativeRunPlugin(items=[2]),
                LlmRequestGroupPlugin(key="iteration"),
            ],
        )


def test_serial_llm_requests_run_in_order_and_record_keyed_results(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_test_file(workspace_tmp_path, Path("plan-system.jinja"), "plan system")
    _write_test_file(
        workspace_tmp_path,
        Path("plan-user.jinja"),
        "plan:{{ item }}:{{ previous_keys | join(',') }}",
    )
    _write_test_file(workspace_tmp_path, Path("review-system.jinja"), "review system")
    _write_test_file(
        workspace_tmp_path,
        Path("review-user.jinja"),
        "review:{{ item }}:{{ plan.plan }}",
    )
    calls: list[tuple[str, str, list[str]]] = []
    events: list[tuple[str, str, object]] = []
    iteration_outputs: list[tuple[object, dict[str, BaseModel]]] = []

    async def llm_call(**kwargs: Any) -> LlmCallOutcome:
        properties = list(kwargs["response_format"]["properties"])
        prompt = kwargs["prompt"].strip()
        calls.append((kwargs["system"].strip(), prompt, properties))
        parts = prompt.split(":")
        text = (
            json.dumps({"plan": f"{parts[1]} draft"})
            if "plan" in properties
            else json.dumps({"review": f"reviewed {parts[2]}"})
        )
        return SampleLlmOutcome(text=text, prompt_eval_count=1, eval_count=1)

    async def prepare_request(context: AgentContext, key: str) -> None:
        assert context.llm_request_key == key
        events.append(("before", key, context.iteration_item))

    async def finish_request(
        context: AgentContext,
        key: str,
        result: SyncAgentResult,
    ) -> None:
        assert context.llm_request_results[key] is result
        assert context.llm_request_key == key
        events.append(("after", key, context.iteration_item))

    async def finish_iteration(
        context: AgentContext,
        item: object,
        result: SyncAgentResult,
    ) -> None:
        _ = result
        iteration_outputs.append((item, dict(context.llm_response_outputs)))

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            IterativeRunPlugin(items=["first", "second"]),
            AfterIterationPlugin(finish_iteration),
            LlmRequestGroupPlugin(key="plan"),
            BeforeLlmRequestPlugin(handler=prepare_request),
            ResponseOutputTypePlugin(SamplePlanOutput),
            TemplateSystemPromptPlugin(
                "plan-system.jinja",
                template_root=workspace_tmp_path,
            ),
            TemplateUserPromptPlugin(
                "plan-user.jinja",
                template_root=workspace_tmp_path,
                template_vars=lambda context: {
                    "item": context.iteration_item,
                    "previous_keys": list(context.llm_response_outputs),
                },
            ),
            LabeledFileMapWriterPlugin(),
            AfterLlmRequestPlugin(handler=finish_request),
            LlmRequestGroupPlugin(key="review"),
            BeforeLlmRequestPlugin(handler=prepare_request),
            ResponseOutputTypePlugin(SampleReviewOutput),
            TemplateSystemPromptPlugin(
                "review-system.jinja",
                template_root=workspace_tmp_path,
            ),
            TemplateUserPromptPlugin(
                "review-user.jinja",
                template_root=workspace_tmp_path,
                template_vars=lambda context: {
                    "item": context.iteration_item,
                    "plan": context.llm_response_outputs["plan"],
                },
            ),
            LabeledFileMapWriterPlugin(),
            AfterLlmRequestPlugin(handler=finish_request),
        ],
    )

    result = asyncio.run(agent.run())

    assert calls == [
        ("plan system", "plan:first:", ["plan"]),
        ("review system", "review:first:first draft", ["review"]),
        ("plan system", "plan:second:", ["plan"]),
        ("review system", "review:second:second draft", ["review"]),
    ]
    assert events == [
        ("before", "plan", "first"),
        ("after", "plan", "first"),
        ("before", "review", "first"),
        ("after", "review", "first"),
        ("before", "plan", "second"),
        ("after", "plan", "second"),
        ("before", "review", "second"),
        ("after", "review", "second"),
    ]
    assert iteration_outputs == [
        (
            "first",
            {
                "plan": SamplePlanOutput(plan="first draft"),
                "review": SampleReviewOutput(review="reviewed first draft"),
            },
        ),
        (
            "second",
            {
                "plan": SamplePlanOutput(plan="second draft"),
                "review": SampleReviewOutput(review="reviewed second draft"),
            },
        ),
    ]
    assert list(agent.context.llm_call_outcomes) == ["plan", "review"]
    assert agent.context.llm_response_outputs == {
        "plan": SamplePlanOutput(plan="second draft"),
        "review": SampleReviewOutput(review="reviewed second draft"),
    }
    assert list(agent.context.llm_request_results) == ["plan", "review"]
    assert result == SyncAgentResult(summary="", written_paths=[])


def test_conditional_llm_requests_use_prior_results_and_skip_cleanly(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_test_file(workspace_tmp_path, Path("system.jinja"), "system")
    _write_test_file(workspace_tmp_path, Path("user.jinja"), "user")
    marker_path = _write_test_file(workspace_tmp_path, Path("keep.txt"), "keep")
    calls: list[str] = []
    skipped_events: list[str] = []

    async def llm_call(**kwargs: Any) -> LlmCallOutcome:
        properties = list(kwargs["response_format"]["properties"])
        key = "plan" if "plan" in properties else "review"
        calls.append(key)
        payload = {"plan": "draft"} if key == "plan" else {"review": "checked"}
        return SampleLlmOutcome(
            text=json.dumps(payload),
            prompt_eval_count=1,
            eval_count=1,
        )

    def request_plugins(key: str, output_type: type[BaseModel]) -> list[Any]:
        return [
            LlmRequestGroupPlugin(key=key),
            ResponseOutputTypePlugin(output_type),
            TemplateSystemPromptPlugin("system.jinja", template_root=workspace_tmp_path),
            TemplateUserPromptPlugin("user.jinja", template_root=workspace_tmp_path),
            LabeledFileMapWriterPlugin(),
        ]

    async def before_skipped(_context: AgentContext, _key: str) -> None:
        skipped_events.append("before")

    async def after_skipped(
        _context: AgentContext,
        _key: str,
        _result: SyncAgentResult,
    ) -> None:
        skipped_events.append("after")

    monkeypatch.setattr(file_agent, "stream_llm_chat_format", llm_call)
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            *request_plugins("plan", SamplePlanOutput),
            LlmRequestGroupPlugin(key="review"),
            ConditionalLlmRequestPlugin(
                should_run=lambda context, _key: (
                    context.llm_response_outputs.get("plan") == SamplePlanOutput(plan="draft")
                )
            ),
            ResponseOutputTypePlugin(SampleReviewOutput),
            TemplateSystemPromptPlugin("system.jinja", template_root=workspace_tmp_path),
            TemplateUserPromptPlugin("user.jinja", template_root=workspace_tmp_path),
            LabeledFileMapWriterPlugin(),
            LlmRequestGroupPlugin(key="skipped"),
            ConditionalLlmRequestPlugin(should_run=lambda _context, _key: False),
            BeforeLlmRequestPlugin(handler=before_skipped),
            FileCleanupPlugin([marker_path]),
            ResponseOutputTypePlugin(SampleReviewOutput),
            TemplateSystemPromptPlugin("system.jinja", template_root=workspace_tmp_path),
            TemplateUserPromptPlugin("user.jinja", template_root=workspace_tmp_path),
            LabeledFileMapWriterPlugin(),
            AfterLlmRequestPlugin(handler=after_skipped),
        ],
    )

    result = asyncio.run(agent.run())

    assert calls == ["plan", "review"]
    assert list(agent.context.llm_response_outputs) == ["plan", "review"]
    assert "skipped" not in agent.context.llm_call_outcomes
    assert "skipped" not in agent.context.llm_request_results
    assert marker_path.read_text(encoding="utf-8") == "keep"
    assert skipped_events == []
    assert result == SyncAgentResult(summary="", written_paths=[])


def test_conditional_llm_request_requires_request_group(workspace_tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires a preceding LlmRequestGroupPlugin"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                ConditionalLlmRequestPlugin(should_run=lambda _context, _key: True),
            ],
        )


def test_llm_request_group_plugin_rejects_blank_key() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        LlmRequestGroupPlugin(key="")


def test_llm_request_hooks_are_optional(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[LlmRequestGroupPlugin(key="plan")],
    )

    assert list(agent.context.llm_request_contexts) == ["plan"]


def test_llm_request_hooks_require_request_group(workspace_tmp_path: Path) -> None:
    async def before_request(_context: AgentContext, _key: str) -> None:
        pass

    async def after_request(
        _context: AgentContext,
        _key: str,
        _result: SyncAgentResult,
    ) -> None:
        pass

    with pytest.raises(RuntimeError, match="requires a preceding LlmRequestGroupPlugin"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[BeforeLlmRequestPlugin(handler=before_request)],
        )

    with pytest.raises(RuntimeError, match="requires a preceding LlmRequestGroupPlugin"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[AfterLlmRequestPlugin(handler=after_request)],
        )


def test_llm_request_hooks_must_be_paired(workspace_tmp_path: Path) -> None:
    async def before_request(_context: AgentContext, _key: str) -> None:
        pass

    async def after_request(
        _context: AgentContext,
        _key: str,
        _result: SyncAgentResult,
    ) -> None:
        pass

    with pytest.raises(ValueError, match="must be paired"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                LlmRequestGroupPlugin(key="plan"),
                BeforeLlmRequestPlugin(handler=before_request),
            ],
        )

    with pytest.raises(ValueError, match="must be paired"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                LlmRequestGroupPlugin(key="plan"),
                AfterLlmRequestPlugin(handler=after_request),
            ],
        )


def test_llm_request_plugins_reject_duplicate_keys(workspace_tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate LLM request key"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                LlmRequestGroupPlugin(key="plan"),
                LlmRequestGroupPlugin(key="plan"),
            ],
        )


def test_llm_request_plugins_reject_nested_iteration(workspace_tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not configure nested"):
        BasicFileAgent(
            cwd=workspace_tmp_path,
            llm_config=SampleLlmConfig(),
            plugins=[
                LlmRequestGroupPlugin(key="plan"),
                IterativeRunPlugin(items=["nested"]),
            ],
        )


def test_json_field_stream_llm_call_registers_wrapped_call(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_agent, "stream_llm_chat_format", _unused_llm_call)

    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[
            LlmRequestGroupPlugin(key="stream"),
            JsonFieldStreamLlmCallPlugin(
                field_name="summary",
                raw_abort_threshold=100,
            ),
        ],
    )

    assert agent.context.llm_request_contexts["stream"].llm_call is not _unused_llm_call


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
        plugins=[
            LlmRequestGroupPlugin(key="silent"),
            SilentLlmCallPlugin(),
        ],
    )
    registered_llm_call = agent.context.llm_request_contexts["silent"].llm_call
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
        plugins=[
            LlmRequestGroupPlugin(key="retry"),
            ResponseOutputTypePlugin(SampleOutput),
            AgentRetryPolicyPlugin(),
        ],
    )
    request_context = agent.context.llm_request_contexts["retry"]
    agent.context = request_context

    result = asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert calls == 3
    assert request_context.max_attempts == 3
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
        plugins=[
            LlmRequestGroupPlugin(key="retry"),
            ResponseOutputTypePlugin(SampleOutput),
            AgentRetryPolicyPlugin(),
        ],
    )
    agent.context = agent.context.llm_request_contexts["retry"]

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
            LlmRequestGroupPlugin(key="retry"),
            ResponseOutputTypePlugin(SampleOutput),
            AgentRetryPolicyPlugin(max_attempts=2),
        ],
    )
    agent.context = agent.context.llm_request_contexts["retry"]

    with pytest.raises(RuntimeError, match=r"LLM call failed after 2 attempt\(s\)"):
        asyncio.run(agent._call_llm({"system_prompt": "system", "user_prompt": "prompt"}))

    assert calls == 2


def test_agent_retry_policy_plugin_retries_invalid_structured_output(
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
            LlmRequestGroupPlugin(key="retry"),
            StaticOutputPathsPlugin([Path("output.txt")]),
            ResponseOutputTypePlugin(SampleFileMapOutput),
            AgentRetryPolicyPlugin(max_attempts=2),
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
            LlmRequestGroupPlugin(key="stream"),
            ResponseOutputTypePlugin(SampleOutput),
            plugin,
        ],
    )
    request_context = agent.context.llm_request_contexts["stream"]
    outcome = asyncio.run(
        request_context.llm_call(
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
            LlmRequestGroupPlugin(key="stream"),
            ResponseOutputTypePlugin(SampleOutput),
            plugin,
        ],
    )
    request_context = agent.context.llm_request_contexts["stream"]
    llm_config = request_context.llm_config
    assert llm_config is not None
    with pytest.raises(ValueError, match=r"5 characters \(limit 4\)"):
        asyncio.run(
            request_context.llm_call(
                llm_cfg=llm_config,
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
                LlmRequestGroupPlugin(key="write"),
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
            LlmRequestGroupPlugin(key="write"),
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
            LlmRequestGroupPlugin(key="write"),
            StaticOutputPathsPlugin([Path("notes/output.txt"), Path("notes/unchanged.txt")]),
            ResponseOutputTypePlugin(SampleFileMapOutput),
            LabeledFileMapWriterPlugin(),
        ],
    )
    agent.context = agent.context.llm_request_contexts["write"]

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
                LlmRequestGroupPlugin(key="replace"),
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
            LlmRequestGroupPlugin(key="replace"),
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
            LlmRequestGroupPlugin(key="replace"),
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
        plugins=[
            LlmRequestGroupPlugin(key="prompt"),
            StaticSystemPromptPlugin(lambda context: f"SYSTEM:{context.cwd.name}"),
        ],
    )

    assert agent.build_system_prompt("") == f"SYSTEM:{workspace_tmp_path.name}\n"


def test_file_agent_requires_system_prompt_hook(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[LlmRequestGroupPlugin(key="prompt")],
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
        plugins=[LlmRequestGroupPlugin(key="prompt"), plugin],
    )

    assert agent.build_system_prompt("") == f"Contract for {workspace_tmp_path.name}\n"


def test_file_agent_requires_user_prompt_hook(workspace_tmp_path: Path) -> None:
    agent = BasicFileAgent(
        cwd=workspace_tmp_path,
        llm_config=SampleLlmConfig(),
        plugins=[LlmRequestGroupPlugin(key="prompt")],
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
            LlmRequestGroupPlugin(key="prompt"),
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
        plugins=[LlmRequestGroupPlugin(key="prompt"), plugin],
    )

    request_context = agent.context.llm_request_contexts["prompt"]
    assert plugin.build_user_prompt(request_context, "task", "current") == "hello 0\n"
