from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from tiny_coder.apply_patch import ApplyPatch, ApplyPatchOutput
from tiny_coder.file_agent import (
    AgentResult,
    BasicFileAgent,
    Conversation,
    ConversationContext,
    ConversationResult,
    LlmCallOutcome,
    agent_input_snapshots,
    parse_structured_output,
    read_agent_file,
)
from tiny_coder.plugins import (
    AfterConversationHandler,
    AfterConversationPlugin,
    ApplyPatchWriterPlugin,
    BeforeConversationHandler,
    BeforeConversationPlugin,
    ConditionalConversationPlugin,
    ConversationConditionHandler,
    ConversationPlugin,
    ConversationRetryPlugin,
    DynamicOutputPathsPlugin,
    ExistingPathGuardPlugin,
    FileCleanupPlugin,
    FileTreeInputPathsPlugin,
    LlmCallJsonlRecorderExtraProvider,
    LlmCallJsonlRecorderPlugin,
    LlmCallTokenStats,
    LlmCallTokenStatsHandler,
    LlmCallTokenStatsPlugin,
    LlmConfigPlugin,
    NoopOutputWriterPlugin,
    ResponseOutputTypePlugin,
    SilentLlmCallPlugin,
    StaticInputPathsPlugin,
    StaticOutputPathsPlugin,
    StaticSystemPromptPlugin,
    StaticUserPromptPlugin,
    SystemPromptProvider,
    TemplateSystemPromptPlugin,
    TemplateUserPromptPlugin,
    TemplateVarsProvider,
    UserPromptProvider,
)
from tiny_coder.text_replacement import TextReplacement


class _Config(BaseModel):
    model: str = "test"


class _SummaryOutput(BaseModel):
    summary: str


class _PriorExtraProvider(LlmCallJsonlRecorderExtraProvider):
    def __call__(
        self,
        context: ConversationContext,
        _outcome: LlmCallOutcome,
        /,
    ) -> dict[str, int]:
        return {"prior": len(context.previous)}


class _ConversationKeySystemPromptProvider(SystemPromptProvider):
    def __call__(self, context: ConversationContext, /) -> str:
        return f"system:{context.key}"


class _MissingPreviousConversationConditionHandler(ConversationConditionHandler):
    def __call__(self, context: ConversationContext, /) -> bool:
        return "missing" in context.previous


class _SystemTemplateVarsProvider(TemplateVarsProvider):
    def __call__(
        self,
        context: ConversationContext,
        /,
    ) -> dict[str, str]:
        _ = context
        return {"role": "reviewer"}


class _UserTemplateVarsProvider(TemplateVarsProvider):
    def __call__(
        self,
        context: ConversationContext,
        /,
    ) -> dict[str, str]:
        return {"prior": context.previous["draft"].summary}


class _PatchOutput(BaseModel, ApplyPatchOutput):
    summary: str
    path: str
    content: str

    def to_apply_patches(self, context: ConversationContext) -> list[ApplyPatch]:
        _ = context
        return [
            ApplyPatch(
                path=Path(self.path),
                replacements=[TextReplacement(to_text=self.content)],
            )
        ]


@dataclass
class _Outcome:
    text: str
    prompt_eval_count: int = 3
    eval_count: int = 2
    llm: dict[str, float] = field(default_factory=dict)


@dataclass
class _QueuedLlmPlugin(ConversationPlugin):
    outputs: list[str | Exception]
    calls: list[tuple[str, str]] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)

    def on_registered(self, context: ConversationContext) -> None:
        context.llm_call = self.call

    async def call(
        self,
        *,
        llm_cfg: BaseModel,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        on_chunk: Callable[[str], None],
    ) -> _Outcome:
        _ = llm_cfg, response_format
        self.calls.append((system, prompt))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        for chunk in self.chunks:
            on_chunk(chunk)
        return _Outcome(text=output)


def _conversation(
    *,
    key: str = "main",
    llm: _QueuedLlmPlugin | None = None,
    plugins: Sequence[ConversationPlugin] = (),
    output_type: type[BaseModel] = _SummaryOutput,
    system_prompt: str | SystemPromptProvider = "system",
    user_prompt: str | UserPromptProvider = "user",
) -> Conversation:
    return Conversation(
        key=key,
        plugins=[
            LlmConfigPlugin(_Config()),
            llm or _QueuedLlmPlugin(['{"summary":"done"}']),
            StaticSystemPromptPlugin(system_prompt),
            StaticUserPromptPlugin(user_prompt),
            ResponseOutputTypePlugin(output_type),
            NoopOutputWriterPlugin(),
            *plugins,
        ],
    )


def _run(tmp_path: Path, *conversations: Conversation) -> AgentResult:
    return asyncio.run(BasicFileAgent(cwd=tmp_path, conversations=list(conversations)).run())


def test_agent_file_helpers_stay_inside_root(tmp_path: Path) -> None:
    source = tmp_path / "input.md"
    source.write_text("hello\n", encoding="utf-8", newline="\n")

    assert read_agent_file(tmp_path, Path("input.md")) == "hello\n"
    with pytest.raises(ValueError, match="outside agent workspace"):
        read_agent_file(tmp_path, Path("../outside.md"))


def test_parse_structured_output_validates_model() -> None:
    assert parse_structured_output('{"summary":"ok"}', output_type=_SummaryOutput).summary == "ok"
    with pytest.raises(ValueError, match="invalid structured output"):
        parse_structured_output('{"missing":true}', output_type=_SummaryOutput)


def test_agent_input_snapshots_include_labels_and_languages(tmp_path: Path) -> None:
    markdown = tmp_path / "notes.md"
    config = tmp_path / "config.yaml"
    markdown.write_text("# Notes\n", encoding="utf-8", newline="\n")
    config.write_text("enabled: true\n", encoding="utf-8", newline="\n")

    assert agent_input_snapshots(tmp_path, [markdown, config]) == [
        {"label": "notes.md", "language": "markdown", "content": "# Notes"},
        {"label": "config.yaml", "language": "yaml", "content": "enabled: true"},
    ]


def test_agent_requires_explicit_conversations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one Conversation"):
        BasicFileAgent(cwd=tmp_path, conversations=[])
    with pytest.raises(ValueError, match="must not be blank"):
        Conversation(key=" ", plugins=[])


def test_agent_rejects_duplicate_conversation_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate conversation key"):
        BasicFileAgent(
            cwd=tmp_path,
            conversations=[_conversation(key="same"), _conversation(key="same")],
        )


def test_conversation_requires_conversation_plugins(tmp_path: Path) -> None:
    class _StructuralPlugin:
        def on_registered(self, context: ConversationContext) -> None:
            _ = context

    with pytest.raises(TypeError, match="ConversationPlugin"):
        BasicFileAgent(
            cwd=tmp_path,
            conversations=[
                Conversation(
                    key="invalid",
                    plugins=[cast(ConversationPlugin, _StructuralPlugin())],
                )
            ],
        )


@pytest.mark.parametrize(
    ("plugins", "message"),
    [
        ([], "LlmConfigPlugin"),
        ([LlmConfigPlugin(_Config())], "ResponseOutputTypePlugin"),
        (
            [LlmConfigPlugin(_Config()), ResponseOutputTypePlugin(_SummaryOutput)],
            "output writer",
        ),
        (
            [
                LlmConfigPlugin(_Config()),
                ResponseOutputTypePlugin(_SummaryOutput),
                NoopOutputWriterPlugin(),
            ],
            "system prompt",
        ),
        (
            [
                LlmConfigPlugin(_Config()),
                ResponseOutputTypePlugin(_SummaryOutput),
                NoopOutputWriterPlugin(),
                StaticSystemPromptPlugin("system"),
            ],
            "user prompt",
        ),
    ],
)
def test_conversation_registration_requires_complete_contract(
    tmp_path: Path,
    plugins: list[ConversationPlugin],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BasicFileAgent(
            cwd=tmp_path,
            conversations=[Conversation(key="incomplete", plugins=plugins)],
        )


def test_single_conversation_returns_validated_result(tmp_path: Path) -> None:
    llm = _QueuedLlmPlugin(['{"summary":"finished"}'])

    result = _run(
        tmp_path,
        _conversation(
            key="sync",
            llm=llm,
            system_prompt=_ConversationKeySystemPromptProvider(),
        ),
    )

    assert list(result.conversations) == ["sync"]
    assert result.conversations["sync"].summary == "finished"
    assert result.conversations["sync"].output == _SummaryOutput(summary="finished")
    assert result.written_paths == []
    assert llm.calls == [("system:sync\n", "user\n")]


def test_multiple_conversations_share_only_completed_results(tmp_path: Path) -> None:
    seen_previous: list[list[str]] = []

    class _ReviewPromptProvider(UserPromptProvider):
        def __call__(
            self,
            context: ConversationContext,
            _errors: Sequence[str],
            /,
        ) -> str:
            seen_previous.append(list(context.previous))
            plan = cast(_SummaryOutput, context.previous["plan"].output)
            return f"review: {plan.summary}"

    plan = _conversation(
        key="plan",
        llm=_QueuedLlmPlugin(['{"summary":"the plan"}']),
    )
    review_llm = _QueuedLlmPlugin(['{"summary":"approved"}'])
    review = Conversation(
        key="review",
        plugins=[
            LlmConfigPlugin(_Config()),
            review_llm,
            StaticSystemPromptPlugin("review system"),
            StaticUserPromptPlugin(_ReviewPromptProvider()),
            ResponseOutputTypePlugin(_SummaryOutput),
            NoopOutputWriterPlugin(),
        ],
    )

    result = _run(tmp_path, plan, review)

    assert list(result.conversations) == ["plan", "review"]
    assert seen_previous == [["plan"]]
    assert review_llm.calls == [("review system\n", "review: the plan\n")]


def test_conditional_conversation_skips_without_side_effects(tmp_path: Path) -> None:
    skipped_llm = _QueuedLlmPlugin(['{"summary":"unexpected"}'])
    skipped = _conversation(
        key="skip",
        llm=skipped_llm,
        plugins=[ConditionalConversationPlugin(_MissingPreviousConversationConditionHandler())],
    )

    result = _run(tmp_path, _conversation(key="first"), skipped)

    assert list(result.conversations) == ["first"]
    assert skipped_llm.calls == []


def test_conversation_callbacks_wrap_one_conversation_not_the_flow(tmp_path: Path) -> None:
    events: list[str] = []

    class _PrepareHandler(BeforeConversationHandler):
        async def __call__(self, context: ConversationContext, /) -> None:
            events.append(f"before:{context.key}:{list(context.previous)}")

    class _ValidateHandler(AfterConversationHandler):
        async def __call__(
            self,
            context: ConversationContext,
            result: ConversationResult,
            /,
        ) -> None:
            events.append(f"after:{context.key}:{result.summary}")

    conversation = _conversation(
        key="turn",
        plugins=[
            BeforeConversationPlugin(_PrepareHandler()),
            AfterConversationPlugin(_ValidateHandler()),
        ],
    )

    _run(tmp_path, conversation)

    assert events == ["before:turn:[]", "after:turn:done"]


def test_conversation_plugins_require_handler_subclasses() -> None:
    with pytest.raises(TypeError, match="handler must inherit BeforeConversationHandler"):
        BeforeConversationPlugin(cast(Any, lambda _context: None))
    with pytest.raises(TypeError, match="handler must inherit AfterConversationHandler"):
        AfterConversationPlugin(cast(Any, lambda _context, _result: None))


def test_prompt_plugins_require_provider_subclasses_for_dynamic_prompts() -> None:
    with pytest.raises(TypeError, match="prompt must be a string or inherit SystemPromptProvider"):
        StaticSystemPromptPlugin(cast(Any, lambda _context: "system"))
    with pytest.raises(TypeError, match="prompt must be a string or inherit UserPromptProvider"):
        StaticUserPromptPlugin(cast(Any, lambda _context, _errors: "user"))


def test_conditional_conversation_plugin_requires_condition_subclass() -> None:
    with pytest.raises(TypeError, match="handler must inherit ConversationConditionHandler"):
        ConditionalConversationPlugin(cast(Any, lambda _context: True))


def test_conversation_retries_with_validation_errors_in_next_prompt(tmp_path: Path) -> None:
    llm = _QueuedLlmPlugin(["not json", '{"summary":"fixed"}'])
    before_calls: list[str] = []

    class _PrepareHandler(BeforeConversationHandler):
        async def __call__(self, context: ConversationContext, /) -> None:
            before_calls.append(context.key)

    class _RetryPromptProvider(UserPromptProvider):
        def __call__(
            self,
            _context: ConversationContext,
            errors: Sequence[str],
            /,
        ) -> str:
            return "retry: " + " | ".join(errors) if errors else "first attempt"

    conversation = Conversation(
        key="retry",
        plugins=[
            LlmConfigPlugin(_Config()),
            llm,
            ConversationRetryPlugin(2),
            BeforeConversationPlugin(_PrepareHandler()),
            StaticSystemPromptPlugin("system"),
            StaticUserPromptPlugin(_RetryPromptProvider()),
            ResponseOutputTypePlugin(_SummaryOutput),
            NoopOutputWriterPlugin(),
        ],
    )

    result = _run(tmp_path, conversation)

    assert result.conversations["retry"].summary == "fixed"
    assert before_calls == ["retry"]
    assert llm.calls[0][1] == "first attempt\n"
    assert "invalid structured output" in llm.calls[1][1]


def test_after_conversation_validation_participates_in_retry(tmp_path: Path) -> None:
    llm = _QueuedLlmPlugin(['{"summary":"first"}', '{"summary":"second"}'])
    calls = 0

    class _RejectOnceHandler(AfterConversationHandler):
        async def __call__(
            self,
            _context: ConversationContext,
            _result: ConversationResult,
            /,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("domain validation failed")

    class _ValidationPromptProvider(UserPromptProvider):
        def __call__(
            self,
            _context: ConversationContext,
            errors: Sequence[str],
            /,
        ) -> str:
            return " | ".join(errors) if errors else "user"

    result = _run(
        tmp_path,
        _conversation(
            key="validate",
            llm=llm,
            user_prompt=_ValidationPromptProvider(),
            plugins=[
                ConversationRetryPlugin(2),
                AfterConversationPlugin(_RejectOnceHandler()),
            ],
        ),
    )

    assert result.conversations["validate"].summary == "second"
    assert "domain validation failed" in llm.calls[1][1]


def test_conversation_failure_does_not_rerun_previous_conversation(tmp_path: Path) -> None:
    first_llm = _QueuedLlmPlugin(['{"summary":"first"}'])
    second_llm = _QueuedLlmPlugin(["bad", '{"summary":"second"}'])

    _run(
        tmp_path,
        _conversation(key="first", llm=first_llm),
        _conversation(
            key="second",
            llm=second_llm,
            plugins=[ConversationRetryPlugin(2)],
        ),
    )

    assert len(first_llm.calls) == 1
    assert len(second_llm.calls) == 2


def test_path_plugins_are_scoped_to_their_conversation(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8", newline="\n")
    conversation = _conversation(
        plugins=[
            StaticInputPathsPlugin([Path("source.md")]),
            DynamicOutputPathsPlugin(),
        ]
    )
    agent = BasicFileAgent(cwd=tmp_path, conversations=[conversation])

    assert conversation.context.input_paths == [source]
    assert conversation.context.output_paths == [source]
    assert agent.context.cwd == tmp_path.resolve()


def test_file_tree_input_paths_preserve_first_path_and_deduplicate(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    first = root / "b.md"
    second = root / "a.md"
    first.write_text("b", encoding="utf-8", newline="\n")
    second.write_text("a", encoding="utf-8", newline="\n")
    conversation = _conversation(
        plugins=[
            FileTreeInputPathsPlugin(
                Path("docs"),
                first_paths=[Path("b.md")],
                patterns=["*.md"],
            )
        ]
    )

    BasicFileAgent(cwd=tmp_path, conversations=[conversation])

    assert conversation.context.input_paths == [first, second]


def test_cleanup_runs_before_prepare_and_existing_path_guard(tmp_path: Path) -> None:
    stale = tmp_path / "stale.txt"
    stale.write_text("old", encoding="utf-8", newline="\n")
    observed: list[bool] = []

    class _PrepareHandler(BeforeConversationHandler):
        async def __call__(self, _context: ConversationContext, /) -> None:
            observed.append(stale.exists())

    conversation = _conversation(
        plugins=[
            StaticOutputPathsPlugin([Path("stale.txt")]),
            FileCleanupPlugin([Path("stale.txt")]),
            ExistingPathGuardPlugin(),
            BeforeConversationPlugin(_PrepareHandler()),
        ]
    )

    _run(tmp_path, conversation)

    assert observed == [False]


def test_existing_path_guard_rejects_without_calling_model(tmp_path: Path) -> None:
    target = tmp_path / "output.txt"
    target.write_text("existing", encoding="utf-8", newline="\n")
    llm = _QueuedLlmPlugin(['{"summary":"unused"}'])
    conversation = _conversation(
        llm=llm,
        plugins=[
            StaticOutputPathsPlugin([Path("output.txt")]),
            ExistingPathGuardPlugin(),
        ],
    )

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _run(tmp_path, conversation)
    assert llm.calls == []


@pytest.mark.parametrize(
    "system_template_vars",
    [{"role": "reviewer"}, _SystemTemplateVarsProvider()],
)
@pytest.mark.parametrize(
    "user_template_vars",
    [{"prior": "done"}, _UserTemplateVarsProvider()],
)
def test_template_plugins_render_files_and_previous_conversation(
    tmp_path: Path,
    system_template_vars: dict[str, str] | TemplateVarsProvider,
    user_template_vars: dict[str, str] | TemplateVarsProvider,
) -> None:
    template_root = tmp_path / "templates"
    template_root.mkdir()
    (template_root / "system.jinja").write_text(
        "role={{ role }}",
        encoding="utf-8",
        newline="\n",
    )
    (template_root / "user.jinja").write_text(
        "{{ prior }}|{{ input_files[0].content }}|{{ retry_errors|length }}",
        encoding="utf-8",
        newline="\n",
    )
    source = tmp_path / "input.md"
    source.write_text("content\n", encoding="utf-8", newline="\n")
    review_llm = _QueuedLlmPlugin(['{"summary":"reviewed"}'])
    review = Conversation(
        key="review",
        plugins=[
            LlmConfigPlugin(_Config()),
            review_llm,
            StaticInputPathsPlugin([Path("input.md")]),
            TemplateSystemPromptPlugin(
                "system.jinja",
                template_root=template_root,
                template_vars=system_template_vars,
            ),
            TemplateUserPromptPlugin(
                "user.jinja",
                template_root=template_root,
                template_vars=user_template_vars,
            ),
            ResponseOutputTypePlugin(_SummaryOutput),
            NoopOutputWriterPlugin(),
        ],
    )

    _run(tmp_path, _conversation(key="draft"), review)

    assert review_llm.calls == [("role=reviewer\n", "done|content|0\n")]


@pytest.mark.parametrize(
    "plugin_type",
    [TemplateSystemPromptPlugin, TemplateUserPromptPlugin],
)
def test_template_prompt_requires_template_vars_provider(
    tmp_path: Path,
    plugin_type: type[TemplateSystemPromptPlugin] | type[TemplateUserPromptPlugin],
) -> None:
    with pytest.raises(
        TypeError,
        match="template_vars must be a mapping or inherit TemplateVarsProvider",
    ):
        plugin_type(
            "system.jinja",
            template_root=tmp_path,
            template_vars=cast(Any, lambda _context: {}),
        )


def test_llm_call_recorder_uses_conversation_vocabulary(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.jsonl"
    conversation = _conversation(
        key="recorded",
        plugins=[
            LlmCallJsonlRecorderPlugin(
                path=Path("calls.jsonl"),
                extra_provider=_PriorExtraProvider(),
            )
        ],
    )

    _run(tmp_path, conversation)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["conversation_key"] == "recorded"
    assert record["system"] == "system\n"
    assert record["user"] == "user\n"
    assert record["extra"] == {"prior": 0}


def test_llm_call_recorder_requires_extra_provider_subclass() -> None:
    with pytest.raises(
        TypeError,
        match="extra_provider must inherit LlmCallJsonlRecorderExtraProvider",
    ):
        LlmCallJsonlRecorderPlugin(
            path=Path("calls.jsonl"),
            extra_provider=cast(Any, lambda _context, _outcome: {}),
        )


def test_token_stats_handler_receives_conversation_context(tmp_path: Path) -> None:
    received: list[tuple[str, LlmCallTokenStats]] = []

    class _RecordTokenStatsHandler(LlmCallTokenStatsHandler):
        async def __call__(
            self,
            context: ConversationContext,
            stats: LlmCallTokenStats,
            /,
        ) -> None:
            received.append((context.key, stats))

    conversation = _conversation(
        key="stats",
        plugins=[LlmCallTokenStatsPlugin(handler=_RecordTokenStatsHandler())],
    )

    _run(tmp_path, conversation)

    assert received == [
        (
            "stats",
            LlmCallTokenStats(
                prompt_eval_count=3,
                eval_count=2,
                prompt_tokens_per_second=None,
                eval_tokens_per_second=None,
            ),
        )
    ]


def test_token_stats_wait_until_conversation_validation_succeeds(tmp_path: Path) -> None:
    llm = _QueuedLlmPlugin(['{"summary":"first"}', '{"summary":"second"}'])
    validated = 0
    stats_calls: list[str] = []

    class _ValidateHandler(AfterConversationHandler):
        async def __call__(
            self,
            _context: ConversationContext,
            _result: ConversationResult,
            /,
        ) -> None:
            nonlocal validated
            validated += 1
            if validated == 1:
                raise ValueError("try again")

    class _RecordTokenStatsHandler(LlmCallTokenStatsHandler):
        async def __call__(
            self,
            context: ConversationContext,
            _stats: LlmCallTokenStats,
            /,
        ) -> None:
            stats_calls.append(context.key)

    conversation = _conversation(
        key="stats",
        llm=llm,
        plugins=[
            ConversationRetryPlugin(2),
            LlmCallTokenStatsPlugin(handler=_RecordTokenStatsHandler()),
            AfterConversationPlugin(_ValidateHandler()),
        ],
    )

    _run(tmp_path, conversation)

    assert stats_calls == ["stats"]


def test_token_stats_plugin_requires_handler_subclass() -> None:
    with pytest.raises(TypeError, match="handler must inherit LlmCallTokenStatsHandler"):
        LlmCallTokenStatsPlugin(handler=cast(Any, lambda _context, _stats: None))


def test_silent_llm_call_suppresses_chunks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = _QueuedLlmPlugin(['{"summary":"quiet"}'], chunks=["raw chunk"])
    conversation = _conversation(llm=llm, plugins=[SilentLlmCallPlugin()])

    _run(tmp_path, conversation)

    assert "raw chunk" not in capsys.readouterr().out


def test_apply_patch_writer_writes_allowed_output(tmp_path: Path) -> None:
    llm = _QueuedLlmPlugin(['{"summary":"written","path":"output.md","content":"new content\\n"}'])
    conversation = Conversation(
        key="write",
        plugins=[
            LlmConfigPlugin(_Config()),
            llm,
            StaticSystemPromptPlugin("system"),
            StaticUserPromptPlugin("user"),
            StaticOutputPathsPlugin([Path("output.md")]),
            ResponseOutputTypePlugin(_PatchOutput),
            ApplyPatchWriterPlugin(),
        ],
    )

    result = _run(tmp_path, conversation)

    output = tmp_path / "output.md"
    assert output.read_text(encoding="utf-8") == "new content\n"
    assert result.written_paths == [output]


def test_agent_context_is_reset_between_runs_but_extras_persist(tmp_path: Path) -> None:
    llm = _QueuedLlmPlugin(['{"summary":"one"}', '{"summary":"two"}'])
    agent = BasicFileAgent(
        cwd=tmp_path,
        conversations=[_conversation(key="turn", llm=llm)],
    )
    agent.context.extras["stable"] = True

    first = asyncio.run(agent.run())
    second = asyncio.run(agent.run())

    assert first.conversations["turn"].summary == "one"
    assert second.conversations["turn"].summary == "two"
    assert agent.context.extras == {"stable": True}
