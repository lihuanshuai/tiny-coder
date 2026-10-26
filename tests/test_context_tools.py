from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from test_structured_agent import SampleLlmConfig, _outcome

from tiny_coder.agent_extensions import CallRecording, ResponseValidationError, StructuredOutput
from tiny_coder.checkpoint import JsonCheckpointer
from tiny_coder.context_tools import ContextFile, ContextTools
from tiny_coder.graph import GraphError
from tiny_coder.llm import LlmChatOutcome, ToolCall
from tiny_coder.structured_agent import (
    BEFORE_CALL,
    AgentCall,
    StructuredInput,
    create_structured_agent,
)


class Answer(BaseModel):
    answer: str


def tool_call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(
        id=name,
        type="function",
        function={"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    )


def test_task_loop_reads_context_and_repairs_rejected_submission_without_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[list[dict[str, Any]]] = []
    attempts: list[tuple[int, int]] = []
    accepted: list[BaseModel] = []
    responses = iter(
        [
            tool_call("list_files"),
            tool_call("grep", path="brief.md", pattern="TODO|^#|<!--|-->"),
            tool_call("read_file", path="brief.md", start_line=1, end_line=2),
            tool_call("read_file", path="brief.md", start_line=3),
            tool_call("submit_result", answer="wrong"),
            tool_call("submit_result", answer="C1 checked the record"),
        ]
    )

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        requests.append(deepcopy(kwargs["messages"]))
        assert kwargs["response_format"] is None
        assert [tool.name for tool in kwargs["tools"]] == [
            "list_files",
            "read_file",
            "read_required",
            "grep",
        ] + (["submit_result"] if len(requests) >= 5 else [])
        if len(requests) == 2:
            listing = json.loads(requests[-1][-1]["content"])
            assert "content" not in listing["files"][0]
        if len(requests) == 3:
            matches = json.loads(requests[-1][-1]["content"])["matches"]
            assert [item["line"] for item in matches] == [1, 2, 3]
        if len(requests) == 6:
            feedback = json.loads(requests[-1][-1]["content"])
            assert requests[-1][-1]["role"] == "tool"
            assert (
                feedback["response_suggestion"]
                == "Use C1 as the actor and describe the completed check."
            )
        return _outcome(tool_calls=(next(responses),))

    async def accept(call: AgentCall, output: BaseModel) -> None:
        assert isinstance(output, Answer)
        if output.answer == "wrong":
            raise ResponseValidationError(
                "actor and action missing",
                suggestion="Use C1 as the actor and describe the completed check.",
            )
        accepted.append(output)

    async def observe(call: AgentCall) -> None:
        attempts.append((call.turn, call.attempt))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        system_prompt="Complete the task using the context tools, then submit_result.",
        max_attempts=2,
        accept=accept,
        extensions=(
            ContextTools(
                [
                    ContextFile(
                        path="brief.md",
                        content="# Record\n<!-- Keep this note -->\nTODO: Check the record.\n",
                        required=True,
                    )
                ]
            ),
        ),
    )
    agent.events.subscribe(BEFORE_CALL, observe)
    result = asyncio.run(agent.invoke(StructuredInput(prompt="Complete the record check.")))
    assert result.model(Answer) == Answer(answer="C1 checked the record")
    assert json.loads(result.text) == {"answer": "C1 checked the record"}
    assert accepted == [result.output]
    assert attempts == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2)]
    for request in requests:
        assert request[:2] == requests[0][:2]
    assert [message["role"] for message in result.messages] == [
        "user",
        *[role for _ in range(6) for role in ("assistant", "tool")],
    ]
    assert json.loads(result.messages[-1]["content"]) == {"accepted": True}


@pytest.mark.parametrize("read_mode", ["none", "grep", "partial", "stale"])
def test_final_output_requires_complete_current_context(
    read_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = ContextFile(path="note.md", content="first\nsecond\n", required=True)
    calls = 0
    accepted: list[BaseModel] = []

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal calls
        calls += 1
        if calls == 1 and read_mode == "grep":
            return _outcome(tool_calls=(tool_call("grep", pattern=".*"),))
        if calls == 1 and read_mode in {"partial", "stale"}:
            return _outcome(
                tool_calls=(
                    tool_call(
                        "read_file", path="note.md", end_line=1 if read_mode == "partial" else 2
                    ),
                )
            )
        if read_mode == "stale":
            file.content = "updated\nsecond\n"
        return _outcome(tool_calls=(tool_call("submit_result", answer="done"),))

    async def accept(call: AgentCall, output: BaseModel) -> None:
        accepted.append(output)

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        accept=accept,
        extensions=(ContextTools([file]),),
    )
    with pytest.raises(ResponseValidationError, match="note.md") as error:
        asyncio.run(agent.invoke(StructuredInput(prompt="Check the note")))
    assert error.value.code == "context_not_read"
    assert not accepted


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "../secret.md"},
        {"path": "C:/secret.md"},
        {"path": "/secret.md"},
        {"path": "unknown.md"},
        {"path": "note.md", "start_line": 3},
        {"path": "note.md", "start_line": 2, "end_line": 1},
        {"path": "note.md", "end_line": 3},
        {"path": "note.md", "start_line": True},
    ],
)
def test_context_read_errors_are_tool_results_and_can_be_corrected(
    arguments: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _outcome(tool_calls=(tool_call("read_file", **arguments),))
        if calls == 2:
            assert kwargs["messages"][-1]["content"].startswith("error:")
            return _outcome(tool_calls=(tool_call("read_file", path="note.md"),))
        data = json.loads(kwargs["messages"][-1]["content"])
        assert data["content"] == "1: first\n2: second"
        return _outcome(tool_calls=(tool_call("submit_result", answer="corrected"),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        max_attempts=1,
        extensions=(
            ContextTools([ContextFile(path="note.md", content="first\nsecond", required=True)]),
        ),
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="Read the note")))
    assert result.model(Answer) == Answer(answer="corrected")


def test_context_and_tools_are_isolated_between_concurrent_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def llm(**kwargs: Any) -> LlmChatOutcome:
        messages = kwargs["messages"]
        task = messages[1]["content"]
        await asyncio.sleep(0)
        if messages[-1]["role"] == "user":
            return _outcome(tool_calls=(tool_call("read_file", path="note.md"),))
        assert json.loads(messages[-1]["content"])["content"] == f"1: {task}"
        return _outcome(tool_calls=(tool_call("submit_result", answer=task),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        extensions=(
            ContextTools(
                lambda call: [ContextFile(path="note.md", content=str(call.context), required=True)]
            ),
        ),
    )

    async def run() -> None:
        results = await asyncio.gather(
            *[
                agent.invoke(StructuredInput(prompt=name, context=name))
                for name in ("left", "right")
            ]
        )
        assert [result.model(Answer) for result in results] == [
            Answer(answer="left"),
            Answer(answer="right"),
        ]

    asyncio.run(run())
    assert agent.tools == []


def test_tool_loop_is_bounded_even_when_no_submission_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def llm(**kwargs: Any) -> LlmChatOutcome:
        return _outcome(tool_calls=(tool_call("list_files"),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        max_steps=4,
        extensions=(ContextTools([]),),
    )
    with pytest.raises(GraphError, match="max_steps=4"):
        asyncio.run(agent.invoke(StructuredInput(prompt="Loop")))


def test_conversations_keep_successful_history_without_leaking_to_other_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[dict[str, Any]]] = []

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        calls.append(deepcopy(kwargs["messages"]))
        if calls[-1][-1]["content"] == "fail":
            raise RuntimeError("transport unavailable")
        return _outcome('{"answer":"accepted"}')

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(llm_config=SampleLlmConfig(), response_model=Answer)

    async def run() -> None:
        left, right = agent.conversation(), agent.conversation()
        await left.send(StructuredInput(prompt="first"))
        with pytest.raises(RuntimeError, match="transport unavailable"):
            await left.send(StructuredInput(prompt="fail"))
        assert len(left.messages) == 2
        copy = left.messages
        copy[0]["content"] = "changed externally"
        await left.send(StructuredInput(prompt="next"))
        await right.send(StructuredInput(prompt="other"))
        assert [m["content"] for m in calls[2][1:]] == ["first", '{"answer":"accepted"}', "next"]
        assert len(calls[3]) == 2
        assert len(left.messages) == 4
        assert len(right.messages) == 2

    asyncio.run(run())


def test_context_preflight_precedes_acceptance_regardless_of_extension_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted: list[BaseModel] = []

    async def accept(call: AgentCall, output: BaseModel) -> None:
        accepted.append(output)

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        return _outcome(tool_calls=(tool_call("submit_result", answer="guessed"),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        extensions=(
            StructuredOutput(accept),
            ContextTools(
                [
                    ContextFile(path="required.md", content="Read me", required=True),
                ]
            ),
        ),
    )
    with pytest.raises(ResponseValidationError, match="required.md"):
        asyncio.run(agent.invoke(StructuredInput(prompt="Read first")))
    assert not accepted


def test_mixed_submission_batch_is_rejected_without_executing_other_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _outcome(
                tool_calls=(
                    tool_call("submit_result", answer="done"),
                    tool_call("read_file", path="note.md"),
                )
            )
        feedback = [json.loads(message["content"]) for message in kwargs["messages"][-2:]]
        assert all(
            "submit_result must be the only tool call" in item["validation_errors"][0]["msg"]
            for item in feedback
        )
        return _outcome(tool_calls=(tool_call("submit_result", answer="corrected"),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        max_attempts=2,
        extensions=(ContextTools([ContextFile(path="note.md", content="note")]),),
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="Submit")))
    assert result.model(Answer) == Answer(answer="corrected")
    assert "context_reads" not in result.state


def test_native_accept_and_output_extension_cannot_silently_override_each_other() -> None:
    async def accept(call: AgentCall, output: BaseModel) -> None:
        pass

    with pytest.raises(ValueError, match="either accept"):
        create_structured_agent(
            llm_config=SampleLlmConfig(), accept=accept, extensions=(StructuredOutput(),)
        )


def test_conversation_reuses_current_reads_but_requires_changed_sources_to_be_read_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = ContextFile(path="note.md", content="first", required=True)
    reads = 0

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal reads
        messages = kwargs["messages"]
        if messages[-1]["role"] == "user":
            return _outcome(tool_calls=(tool_call("list_files"),))
        data = json.loads(messages[-1]["content"])
        if "files" in data:
            current = data["files"][0]
            if not current["read"]:
                reads += 1
                return _outcome(tool_calls=(tool_call("read_file", path="note.md"),))
        return _outcome(tool_calls=(tool_call("submit_result", answer=file.content),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        extensions=(ContextTools([file]),),
    )

    async def run() -> None:
        conversation = agent.conversation()
        await conversation.send(StructuredInput(prompt="Read"))
        await conversation.send(StructuredInput(prompt="Use the same source"))
        assert reads == 1
        file.content = "second"
        result = await conversation.send(StructuredInput(prompt="Use the changed source"))
        assert result.model(Answer) == Answer(answer="second")
        assert reads == 2

    asyncio.run(run())


def test_submissions_and_rejected_context_are_recorded_and_checkpointed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            tool_call("submit_result", answer="not read"),
            tool_call("read_file", path="empty.md"),
            tool_call("submit_result", answer="empty"),
        ]
    )

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        return _outcome(tool_calls=(next(responses),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    record_path = tmp_path / "calls.jsonl"
    checkpointer = JsonCheckpointer(tmp_path / "checkpoint.json")
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        max_attempts=2,
        checkpointer=checkpointer,
        extensions=(
            ContextTools([ContextFile(path="empty.md", content="", required=True)]),
            CallRecording(record_path),
        ),
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="Check the empty file")))
    with record_path.open(encoding="utf-8", newline="\n") as source:
        records = [json.loads(line) for line in source]
    assert json.loads(records[0]["output"]) == {"answer": "not read"}
    assert json.loads(records[2]["output"]) == {"answer": "empty"}
    read_message = records[2]["messages"][-2]
    assert read_message["name"] == "read_file"
    assert json.loads(read_message["content"])["content"] == ""
    checkpoint = asyncio.run(checkpointer.load())
    assert checkpoint is not None
    assert checkpoint.state["messages"] == result.messages
    assert checkpoint.state["context_reads"]["empty.md"]["ranges"] == [[1, 0]]


def test_required_batch_gates_submission_and_only_rereads_changed_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = [
        ContextFile(path=f"note-{index}.md", content=f"line {index}", required=True)
        for index in range(50)
    ]
    files.append(ContextFile(path="optional.md", content="not required"))
    turns = 0
    attempts = []

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal turns
        turns += 1
        names = {tool.name for tool in kwargs["tools"]}
        if turns == 1:
            assert "submit_result" not in names
            return _outcome(tool_calls=(tool_call("read_required"),))
        assert "submit_result" in names
        read = json.loads(kwargs["messages"][-1]["content"])["files"]
        if turns == 2:
            assert len(read) == 50
            assert read[0]["content"] == "1: line 0"
            assert all(item["path"] != "optional.md" for item in read)
            files[0].content = "updated"
            return _outcome(tool_calls=(tool_call("read_required"),))
        if turns == 3:
            assert [item["path"] for item in read] == ["note-0.md"]
            assert read[0]["content"] == "1: updated"
            return _outcome(tool_calls=(tool_call("read_required"),))
        assert read == []
        return _outcome(tool_calls=(tool_call("submit_result", answer="done"),))

    async def observe(call: AgentCall) -> None:
        attempts.append(call.attempt)

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Answer,
        max_attempts=1,
        extensions=(ContextTools(files),),
    )
    agent.events.subscribe(BEFORE_CALL, observe)
    result = asyncio.run(agent.invoke(StructuredInput(prompt="Read all required snapshots")))
    assert result.model(Answer) == Answer(answer="done")
    assert attempts == [1, 1, 1, 1]
    assert "optional.md" not in result.state["context_reads"]


def test_stringified_array_and_incomplete_json_receive_specific_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result(BaseModel):
        files: list[dict[str, str]]

    turns = 0

    async def llm(**kwargs: Any) -> LlmChatOutcome:
        nonlocal turns
        turns += 1
        if turns == 1:
            return _outcome(tool_calls=(tool_call("read_required"),))
        if turns == 2:
            return _outcome(tool_calls=(tool_call("submit_result", files='[{"path":"note.md"}]'),))
        feedback = json.loads(kwargs["messages"][-1]["content"])
        if turns == 3:
            assert feedback["validation_errors"][0]["loc"] == ["files"]
            assert feedback["validation_errors"][0]["type"] == "list_type"
            return _outcome(
                tool_calls=(
                    ToolCall(
                        id="broken",
                        type="function",
                        function={"name": "submit_result", "arguments": '{"files":'},
                    ),
                )
            )
        error = feedback["validation_errors"][0]
        assert error["type"] == "invalid_submission_json"
        assert error["ctx"] == {"line": 1, "column": 10, "offset": 9}
        assert "column 10" in feedback["response_suggestion"]
        return _outcome(tool_calls=(tool_call("submit_result", files=[{"path": "note.md"}]),))

    monkeypatch.setattr("tiny_coder.structured_agent.stream_llm_chat", llm)
    agent = create_structured_agent(
        llm_config=SampleLlmConfig(),
        response_model=Result,
        max_attempts=3,
        extensions=(ContextTools([ContextFile(path="note.md", content="source", required=True)]),),
    )
    result = asyncio.run(agent.invoke(StructuredInput(prompt="Return file objects")))
    assert result.model(Result) == Result(files=[{"path": "note.md"}])
