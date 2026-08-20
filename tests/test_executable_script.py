from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from tiny_coder import file_agent
from tiny_coder.executable_script import GeneratedScriptOutput
from tiny_coder.file_agent import LlmCall, LlmCallOutcome
from tiny_coder.plugins import (
    AgentRetryPolicyPlugin,
    ExecutableScriptPlugin,
    LlmConfigPlugin,
    LlmRequestGroupPlugin,
    ResponseOutputTypePlugin,
    StaticSystemPromptPlugin,
    TemplateUserPromptPlugin,
)


class SampleLlmConfig(BaseModel):
    model: str = "test-model"


class SampleLlmOutcome(BaseModel):
    text: str
    prompt_eval_count: int = 1
    eval_count: int = 1


class NonScriptOutput(BaseModel):
    script: str


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    root = Path(".test-tmp") / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_test_file(root: Path, path: Path, content: str) -> Path:
    target = (root / path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)
    return target


def _build_agent(
    root: Path,
    llm_call: LlmCall,
    *,
    script_plugin: ExecutableScriptPlugin,
    max_attempts: int = 1,
) -> file_agent.BasicFileAgent:
    _write_test_file(root, Path("script-user.jinja"), "{{ retry_errors | join(' | ') }}")
    agent = file_agent.BasicFileAgent(
        cwd=root,
        plugins=[
            LlmConfigPlugin(SampleLlmConfig()),
            LlmRequestGroupPlugin(key="script"),
            AgentRetryPolicyPlugin(max_attempts),
            ResponseOutputTypePlugin(GeneratedScriptOutput),
            StaticSystemPromptPlugin("Generate one executable Python script."),
            TemplateUserPromptPlugin(
                "script-user.jinja",
                template_root=root,
            ),
            script_plugin,
        ],
    )
    agent.context.llm_request_contexts["script"].llm_call = llm_call
    return agent


def test_executable_script_plugin_validates_then_executes_content(
    workspace_tmp_path: Path,
) -> None:
    async def llm_call(**kwargs: object) -> LlmCallOutcome:
        _ = kwargs
        return SampleLlmOutcome(
            text=json.dumps(
                {
                    "summary": "generated",
                    "script": "import sys\r\nprint('hello')\r\nprint(sys.argv[1])",
                }
            )
        )

    agent = _build_agent(
        workspace_tmp_path,
        llm_call,
        script_plugin=ExecutableScriptPlugin(arguments=["argument"]),
    )

    result = asyncio.run(agent.run())

    assert result.summary == "generated"
    assert result.written_paths == []
    assert list(workspace_tmp_path.iterdir()) == [workspace_tmp_path / "script-user.jinja"]
    execution = agent.context.script_execution_results["script"]
    assert execution.command == (sys.executable, "-", "argument")
    assert execution.returncode == 0
    assert execution.stdout.splitlines() == ["hello", "argument"]
    assert execution.stderr == ""
    assert not execution.timed_out


def test_executable_script_failure_is_available_to_generation_retry(
    workspace_tmp_path: Path,
) -> None:
    prompts: list[str] = []

    async def llm_call(**kwargs: object) -> LlmCallOutcome:
        prompt = str(kwargs["prompt"])
        prompts.append(prompt)
        code = (
            "import sys\nsys.stderr.write('broken\\n')\nraise SystemExit(7)"
            if len(prompts) == 1
            else "print('fixed')"
        )
        return SampleLlmOutcome(text=json.dumps({"script": code}))

    agent = _build_agent(
        workspace_tmp_path,
        llm_call,
        script_plugin=ExecutableScriptPlugin(),
        max_attempts=2,
    )

    asyncio.run(agent.run())

    assert prompts[0].strip() == ""
    assert "generated script failed with exit code 7" in prompts[1]
    assert "stderr:" in prompts[1]
    assert "broken" in prompts[1]
    execution = agent.context.script_execution_results["script"]
    assert execution.returncode == 0
    assert execution.stdout.strip() == "fixed"


def test_executable_script_timeout_is_captured(
    workspace_tmp_path: Path,
) -> None:
    async def llm_call(**kwargs: object) -> LlmCallOutcome:
        _ = kwargs
        return SampleLlmOutcome(text=json.dumps({"script": "import time\ntime.sleep(5)"}))

    agent = _build_agent(
        workspace_tmp_path,
        llm_call,
        script_plugin=ExecutableScriptPlugin(
            timeout_seconds=0.1,
        ),
    )

    with pytest.raises(RuntimeError, match="generated script timed out after 0.1 seconds"):
        asyncio.run(agent.run())

    execution = agent.context.script_execution_results["script"]
    assert execution.timed_out
    assert execution.returncode != 0


def test_executable_script_plugin_requires_explicit_output_contract(
    workspace_tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="must inherit ExecutableScriptOutput"):
        file_agent.BasicFileAgent(
            cwd=workspace_tmp_path,
            plugins=[
                LlmConfigPlugin(SampleLlmConfig()),
                LlmRequestGroupPlugin(key="script"),
                ResponseOutputTypePlugin(NonScriptOutput),
                ExecutableScriptPlugin(),
            ],
        )


def test_invalid_script_output_is_not_executed(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def llm_call(**kwargs: object) -> LlmCallOutcome:
        _ = kwargs
        return SampleLlmOutcome(text=json.dumps({"script": "   "}))

    async def unexpected_process(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("invalid script must not be executed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_process)
    agent = _build_agent(
        workspace_tmp_path,
        llm_call,
        script_plugin=ExecutableScriptPlugin(),
    )

    with pytest.raises(RuntimeError, match="generated script must not be blank"):
        asyncio.run(agent.run())


def test_executable_script_plugin_rejects_unsafe_configuration() -> None:

    with pytest.raises(ValueError, match="command must contain non-blank"):
        ExecutableScriptPlugin(command=[])

    with pytest.raises(ValueError, match="must be greater than zero"):
        ExecutableScriptPlugin(timeout_seconds=0)
