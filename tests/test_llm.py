from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import BaseModel, TypeAdapter, ValidationError

from tiny_coder import llm as llm_module
from tiny_coder.agent_extensions import StructuredOutput
from tiny_coder.json_utils import JsonProtocolError
from tiny_coder.llm import (
    LlmChatOutcome,
    LlmConfig,
    LlmExchange,
    nonnegative_int_from_llm_field,
    responses_input,
    schema_response_format,
    stream_llm_chat,
    tool_openai_spec,
)
from tiny_coder.state import State
from tiny_coder.structured_agent import (
    CHUNK_RECEIVED,
    AgentCall,
    StructuredInput,
    create_structured_agent,
)
from tiny_coder.tool_node import Tool


def test_tool_wire_schema_exposes_nested_arrays_and_preserves_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {"files": {"type": "array", "items": {"$ref": "#/$defs/File"}}},
        "required": ["files"],
        "$defs": {
            "File": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "enum": ["note.md"]},
                    "replacements": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/Edit"},
                    },
                    "default": {"$ref": "#/$defs/Edit"},
                },
                "required": ["path", "replacements"],
            },
            "Edit": {
                "type": "object",
                "properties": {"old_count": {"type": "integer", "minimum": 1}},
            },
        },
    }
    original = deepcopy(schema)
    wire = tool_openai_spec(SimpleNamespace(name="submit_result", parameters=schema))["parameters"]
    file = wire["properties"]["files"]["items"]
    assert file["type"] == "object"
    assert file["properties"]["path"]["enum"] == ["note.md"]
    assert file["additionalProperties"] is False
    assert file["properties"]["replacements"]["minItems"] == 1
    assert file["properties"]["replacements"]["items"]["properties"]["old_count"]["minimum"] == 1
    assert file["properties"]["default"]["type"] == "object"
    assert "$defs" not in wire and "$ref" not in str(wire)
    assert schema == original
    file["properties"]["path"]["enum"].append("other.md")
    assert schema == original


def test_recursive_tool_schema_keeps_resolvable_definitions() -> None:
    schema: dict[str, Any] = {
        "$ref": "#/$defs/Node",
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    }
    wire = tool_openai_spec(SimpleNamespace(name="tree", parameters=schema))["parameters"]
    assert wire['$ref'] == schema['$ref']
    assert wire['$defs']['Node']['properties'] == schema['$defs']['Node']['properties']
    assert wire['$defs']['Node']['additionalProperties'] is False
    assert wire['$defs']['Node']['required'] == ['child']
    assert wire is not schema


def test_tool_ref_sibling_constraints_remain_intersected_and_examples_are_data() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Value", "enum": ["b", "c"]}},
        "$defs": {"Value": {"type": "string", "enum": ["a", "b"]}},
        "examples": [{"$ref": "#/$defs/Value"}],
    }
    wire = tool_openai_spec(SimpleNamespace(name="choose", parameters=schema))["parameters"]
    assert wire["properties"]["value"] == {
        "allOf": [{"type": "string", "enum": ["a", "b"]}, {"enum": ["b", "c"]}]
    }
    assert wire["examples"] == schema["examples"]


class SampleOpenAIConfig(LlmConfig):
    base_url: str = "http://localhost:8080/v1"
    llm_model: str = "sample-model"
    num_ctx: int = 4096
    temperature: float = 0.5
    repeat_penalty: float = 1.1
    think: bool = False
    timeout: float = 37.0
    max_output_tokens: int | None = 8192


class _SummaryOutput(BaseModel):
    summary: str


class PlainPydanticConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"


class FakeStream:
    def __init__(self, packets: Sequence[ResponseStreamEvent]) -> None:
        self._packets = iter(packets)
        self.closed = False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        try:
            return next(self._packets)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def close(self) -> None:
        self.closed = True


class FakeResponses:
    kwargs: dict[str, Any] = {}
    packets: list[ResponseStreamEvent] = []
    stream: FakeStream

    @classmethod
    async def create(cls, **kwargs: Any) -> FakeStream:
        cls.kwargs = kwargs
        cls.stream = FakeStream(cls.packets)
        return cls.stream


class FakeAsyncOpenAI:
    closed = False
    timeout = 0.0

    def __init__(self, **kwargs: Any) -> None:
        self.base_url = kwargs.get('base_url', '')
        type(self).timeout = kwargs.get('timeout', 0.0)
        self.responses = FakeResponses()

    async def close(self) -> None:
        type(self).closed = True


def _response(text: str = 'hello world', **overrides: Any) -> Response:
    raw = {
        'id': 'resp_1',
        'created_at': 0,
        'model': 'sample-model',
        'object': 'response',
        'status': 'completed',
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'output': [
            {
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
            }
        ],
        'usage': {
            'input_tokens': 3,
            'output_tokens': 4,
            'total_tokens': 7,
            'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 0},
        },
        **overrides,
    }
    return Response.model_validate(raw)


def _event(kind: str, **values: Any) -> ResponseStreamEvent:
    return TypeAdapter(ResponseStreamEvent).validate_python(
        {'type': kind, 'sequence_number': 0, **values}
    )


def _delta(text: str) -> ResponseStreamEvent:
    return _event(
        'response.output_text.delta',
        delta=text,
        content_index=0,
        item_id='msg_1',
        output_index=0,
        logprobs=[],
    )


def _terminal(response: Response | None = None) -> ResponseStreamEvent:
    response = response or _response()
    return _event(f'response.{response.status}', response=response)


def _patch(monkeypatch: pytest.MonkeyPatch, packets: Sequence[ResponseStreamEvent]) -> None:
    FakeResponses.packets = list(packets)
    FakeResponses.kwargs = {}
    FakeAsyncOpenAI.closed = False
    monkeypatch.setattr(llm_module, 'AsyncOpenAI', FakeAsyncOpenAI)


def _run(**options: Any) -> LlmChatOutcome:
    return asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(), messages=[{'role': 'user', 'content': 'hi'}], **options
        )
    )


def test_llm_config_declares_required_fields() -> None:
    assert set(LlmConfig.model_fields) == {
        "base_url",
        "llm_model",
        "num_ctx",
        "temperature",
        "repeat_penalty",
        "think",
        "timeout",
        "max_output_tokens",
    }


def test_nonnegative_int_from_llm_field_accepts_only_nonnegative_integral_values() -> None:
    assert nonnegative_int_from_llm_field(3) == 3
    assert nonnegative_int_from_llm_field(3.0) == 3
    assert nonnegative_int_from_llm_field(True) is None
    assert nonnegative_int_from_llm_field(-1) is None
    assert nonnegative_int_from_llm_field(3.5) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"summary": "ok"}', {"summary": "ok"}),
        ("not json", None),
    ],
)
def test_llm_chat_outcome_json(text: str, expected: dict[str, object] | None) -> None:
    outcome = LlmChatOutcome(
        text=text,
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    assert outcome.json() == expected


def test_llm_chat_outcome_model_returns_requested_type() -> None:
    outcome = LlmChatOutcome(
        text='{"summary":"ok"}',
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    model = outcome.model(_SummaryOutput)
    assert model == _SummaryOutput(summary="ok")


def test_llm_chat_outcome_model_preserves_failures() -> None:
    outcome = LlmChatOutcome(
        text="not json",
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    assert outcome.model(_SummaryOutput) is None

    outcome2 = LlmChatOutcome(
        text='{"missing":true}',
        tool_calls=(),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        llm={},
    )
    with pytest.raises(ValidationError, match="summary"):
        outcome2.model(_SummaryOutput)


def test_llm_chat_outcome_message_appends_tool_calls() -> None:
    outcome = LlmChatOutcome(
        text="",
        tool_calls=(
            llm_module.ToolCall(
                id="c1", type="function", function={"name": "add", "arguments": "{}"}
            ),
        ),
        prompt_eval_count=1,
        eval_count=1,
        client_wall_time_ms=1.0,
        started_at="",
        completed_at="",
        llm={},
    )
    msg = outcome.message
    assert msg["role"] == "assistant"
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["id"] == "c1"


def test_responses_request_streaming_metrics_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        [
            _delta('hello'),
            _delta(' world'),
            _terminal(_response(prompt_eval_duration=1_500_000_000, eval_duration=2_000_000_000)),
        ],
    )
    streamed: list[str] = []
    exchanges: list[LlmExchange] = []

    async def chunk(text: str) -> None:
        streamed.append(text)

    async def exchange(value: LlmExchange) -> None:
        exchanges.append(value)

    result = _run(on_chunk=chunk, on_exchange=exchange)
    assert streamed == ['hello', ' world']
    assert result.text == 'hello world'
    assert result.prompt_eval_count == 3 and result.eval_count == 4
    assert result.llm['provider'] == 'openai-responses'
    assert result.llm['response_id'] == 'resp_1'
    assert result.llm['prompt_tokens_per_second'] == result.llm['eval_tokens_per_second'] == 2
    assert result.llm['prompt_eval_duration'] == 1_500_000_000
    assert result.llm['eval_duration'] == 2_000_000_000
    assert FakeResponses.kwargs['input'] == [{'role': 'user', 'content': 'hi'}]
    assert FakeResponses.kwargs['store'] is False
    assert FakeResponses.kwargs['stream'] is True
    assert FakeResponses.kwargs['max_output_tokens'] == 8192
    assert FakeResponses.kwargs['reasoning'] == {'effort': 'none'}
    assert 'think' not in FakeResponses.kwargs['extra_body']
    assert result.llm['reasoning_tokens'] == 0
    assert (
        not {'messages', 'max_tokens', 'stream_options', 'response_format'}
        & FakeResponses.kwargs.keys()
    )
    assert FakeAsyncOpenAI.closed and FakeResponses.stream.closed
    assert FakeAsyncOpenAI.timeout == 37
    assert exchanges[0].response is result


@pytest.mark.parametrize('kind', ['text', 'tool', 'reasoning', 'buffered'])
def test_first_output_timings_only_use_observed_nonempty_deltas(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    packets = [_event('response.in_progress', response=_response()), _delta('')]
    if kind == 'text':
        packets.append(_delta('hello'))
    elif kind == 'tool':
        packets.append(
            _event(
                'response.function_call_arguments.delta',
                delta='{"path":',
                item_id='fc_1',
                output_index=0,
            )
        )
    elif kind == 'reasoning':
        packets.append(
            _event(
                'response.reasoning_text.delta',
                delta='check',
                item_id='rs_1',
                output_index=0,
                content_index=0,
            )
        )
    packets.append(_terminal())
    _patch(monkeypatch, packets)
    clock = iter([0.0, 2.0] if kind == 'buffered' else [0.0, 0.5, 2.0])
    monkeypatch.setattr(llm_module.time, 'perf_counter', lambda: next(clock))
    result = _run()
    assert result.client_wall_time_ms == 2000.0
    if kind == 'buffered':
        assert 'time_to_first_output_ms' not in result.llm
        assert 'stream_output_ms' not in result.llm
    else:
        assert result.llm['time_to_first_output_ms'] == 500.0
        assert result.llm['stream_output_ms'] == 1500.0


def test_text_format_and_native_named_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [_terminal()])
    schema = schema_response_format(_SummaryOutput)
    tool = SimpleNamespace(name='add', description='Add', parameters={'type': 'object'})
    _run(response_format=schema, tools=[tool], tool_choice={'type': 'function', 'name': 'add'})
    assert FakeResponses.kwargs['text'] == {'format': schema}
    assert FakeResponses.kwargs['tool_choice'] == {'type': 'function', 'name': 'add'}
    assert FakeResponses.kwargs['tools'] == [tool_openai_spec(tool)]
    assert FakeResponses.kwargs['tools'][0]['name'] == 'add'
    assert 'function' not in FakeResponses.kwargs['tools'][0]


def test_native_tool_calls_use_call_id_and_preserve_parallel_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [
        {
            'type': 'function_call',
            'id': 'fc_2',
            'call_id': 'call_2',
            'name': 'second',
            'arguments': '{"text":"原文"}',
        },
        {
            'type': 'function_call',
            'id': 'fc_1',
            'call_id': 'call_1',
            'name': 'first',
            'arguments': '{}',
        },
    ]
    _patch(
        monkeypatch,
        [
            _event(
                'response.function_call_arguments.delta',
                item_id='fc_2',
                output_index=0,
                delta='{"text":',
            ),
            _event(
                'response.function_call_arguments.done',
                item_id='fc_2',
                output_index=0,
                name='second',
                arguments='{"text":"原文"}',
            ),
            _terminal(_response(output=calls)),
        ],
    )
    result = _run()
    assert result.text == ''
    assert [call.id for call in result.tool_calls] == ['call_2', 'call_1']
    assert result.tool_calls[0].function == {'name': 'second', 'arguments': '{"text":"原文"}'}
    assert result.llm['done_reason'] == 'tool_calls'
    inputs = responses_input(
        [
            result.message,
            {'role': 'tool', 'tool_call_id': 'call_2', 'name': 'second', 'content': 'done'},
        ]
    )
    assert inputs[0] == {
        'type': 'function_call',
        'call_id': 'call_2',
        'name': 'second',
        'arguments': '{"text":"原文"}',
    }
    assert inputs[-1] == {'type': 'function_call_output', 'call_id': 'call_2', 'output': 'done'}


def test_reasoning_replayed_but_rejected_arguments_follow_edited_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'}
    function = {
        'type': 'function_call',
        'id': 'fc_1',
        'call_id': 'call_1',
        'name': 'submit_result',
        'arguments': '{"files":"broken"}',
    }
    _patch(monkeypatch, [_terminal(_response(output=[reasoning, function]))])
    outcome = _run()
    message = outcome.message
    message['tool_calls'][0]['function']['arguments'] = '{"summary":"fixed"}'
    inputs = responses_input([message])
    assert inputs[0] == reasoning
    assert inputs[1]['arguments'] == '{"summary":"fixed"}'
    assert 'broken' not in json.dumps(inputs)
    assert outcome.tool_calls[0].function['arguments'] == '{"files":"broken"}'


def test_buffered_response_uses_whole_request_time(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response(
        usage={
            'input_tokens': 100,
            'output_tokens': 1849,
            'total_tokens': 1949,
            'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 0},
        }
    )
    _patch(monkeypatch, [_terminal(response)])
    clock = iter([0.0, 50.0011])
    monkeypatch.setattr(llm_module.time, 'perf_counter', lambda: next(clock))
    result = _run()
    assert result.llm['request_tokens_per_second'] == pytest.approx(1849 / 50.0011)
    assert 'eval_tokens_per_second' not in result.llm


@pytest.mark.parametrize('on_terminal', [False, True])
def test_server_timings_may_arrive_before_terminal(
    monkeypatch: pytest.MonkeyPatch, on_terminal: bool
) -> None:
    values = {
        'prompt_per_second': 1200.0,
        'predicted_per_second': 42.5,
        'prompt_ms': 150.0,
        'predicted_ms': 200.0,
    }
    _patch(
        monkeypatch,
        [
            _event(
                'response.in_progress', response=_response(timings=None if on_terminal else values)
            ),
            _terminal(_response(timings=values if on_terminal else None)),
        ],
    )
    result = _run()
    assert result.llm['prompt_tokens_per_second'] == 1200
    assert result.llm['eval_tokens_per_second'] == 42.5
    assert result.llm['prompt_ms'] == 150.0
    assert result.llm['predicted_ms'] == 200.0


@pytest.mark.parametrize('invalid', [float('inf'), float('nan'), -1.0, True])
def test_invalid_server_speeds_use_request_average(
    monkeypatch: pytest.MonkeyPatch, invalid: float | bool
) -> None:
    _patch(monkeypatch, [_terminal(_response(timings={'predicted_per_second': invalid}))])
    result = _run()
    assert 'eval_tokens_per_second' not in result.llm
    assert result.llm['request_tokens_per_second'] > 0


@pytest.mark.parametrize('failure', ['error', 'failed', 'incomplete', 'disconnected'])
def test_failed_or_incomplete_stream_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    terminal = {
        'error': _event('error', code='bad', message='stream failed', param=None),
        'failed': _event(
            'response.failed',
            response=_response(
                status='failed', error={'code': 'server_error', 'message': 'stream failed'}
            ),
        ),
        'incomplete': _terminal(
            _response(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
        ),
    }
    packets = [_delta('partial')]
    if failure != 'disconnected':
        packets.append(terminal[failure])
    _patch(monkeypatch, packets)
    with pytest.raises((RuntimeError, JsonProtocolError)):
        _run()
    assert FakeAsyncOpenAI.closed and FakeResponses.stream.closed


@pytest.mark.parametrize('failed', [False, True])
def test_caller_owned_client_is_never_closed(monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    _patch(monkeypatch, [] if failed else [_terminal()])
    client = FakeAsyncOpenAI()
    if failed:
        with pytest.raises(JsonProtocolError):
            _run(client=client)
    else:
        assert _run(client=client).text == _run(client=client).text == 'hello world'
    assert not client.closed
    assert FakeResponses.stream.closed


def test_truncated_function_call_is_rejected_as_retryable_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = {
        'type': 'function_call',
        'id': 'fc_1',
        'call_id': 'call_1',
        'name': 'replace_lines',
        'arguments': '{"path":',
    }
    _patch(
        monkeypatch,
        [
            _terminal(
                _response(
                    status='incomplete',
                    incomplete_details={'reason': 'max_output_tokens'},
                    output=[function],
                )
            )
        ],
    )
    with pytest.raises(JsonProtocolError, match='max_output_tokens.*tool calls rejected'):
        _run()


def test_missing_usage_remains_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [_terminal(_response(usage=None))])
    result = _run()
    assert result.prompt_eval_count == result.eval_count == 0


def test_observed_reasoning_tokens_are_recorded_even_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response()
    assert response.usage is not None and response.usage.output_tokens_details is not None
    response.usage.output_tokens_details.reasoning_tokens = 37
    _patch(monkeypatch, [_terminal(response)])
    result = _run()
    assert FakeResponses.kwargs['reasoning'] == {'effort': 'none'}
    assert result.llm['reasoning_tokens'] == 37


def test_thinking_and_output_limit_options(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [_terminal()])
    asyncio.run(
        stream_llm_chat(
            llm_cfg=SampleOpenAIConfig(think=True, max_output_tokens=None),
            messages=[{'role': 'user', 'content': 'hi'}],
        )
    )
    assert FakeResponses.kwargs['reasoning'] == {'effort': 'medium'}
    assert 'think' not in FakeResponses.kwargs['extra_body']
    assert 'max_output_tokens' not in FakeResponses.kwargs


def test_request_validation() -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        asyncio.run(stream_llm_chat(llm_cfg=SampleOpenAIConfig(), messages=[]))
    with pytest.raises(ValueError, match='tool_choice requires tools'):
        _run(tool_choice='required')
    with pytest.raises(TypeError, match='LlmConfig'):
        asyncio.run(
            stream_llm_chat(
                llm_cfg=PlainPydanticConfig(),  # type: ignore[arg-type]
                messages=[{'role': 'user', 'content': 'hi'}],
            )
        )
    with pytest.raises(TypeError, match='BaseModel'):
        schema_response_format(object())  # type: ignore[arg-type]


@pytest.mark.parametrize('stream_preamble', [False, True])
def test_agent_loop_uses_native_responses_over_sdk_transport(
    monkeypatch: pytest.MonkeyPatch,
    stream_preamble: bool,
) -> None:
    class Inspect(BaseModel):
        key: str

    requests: list[dict[str, Any]] = []
    chunks: list[str] = []
    reasoning = {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'}

    async def inspect(state: State, key: str) -> str:
        assert key == 'A-13'
        return 'record-13'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == '/v1/responses'
        body = json.loads(request.content)
        requests.append(body)
        assert body['store'] is False
        assert body['tool_choice'] == 'required'
        assert body['reasoning'] == {'effort': 'none'}
        assert 'think' not in body
        assert 'messages' not in body and 'response_format' not in body
        assert {tool['name'] for tool in body['tools']} == {'inspect', 'submit_result'}
        submit = next(tool for tool in body['tools'] if tool['name'] == 'submit_result')
        assert submit['strict'] is True
        assert submit['parameters']['additionalProperties'] is False
        assert submit['parameters']['required'] == ['summary']
        inspect_tool = next(tool for tool in body['tools'] if tool['name'] == 'inspect')
        assert inspect_tool['strict'] is True
        if len(requests) == 1:
            items = [
                reasoning,
                {
                    'type': 'function_call',
                    'id': 'fc_1',
                    'call_id': 'call_1',
                    'name': 'inspect',
                    'arguments': '{"key":"A-13"}',
                },
            ]
        else:
            assert len(requests) == 2
            assert reasoning in body['input']
            assert {
                'type': 'function_call_output',
                'call_id': 'call_1',
                'output': 'record-13',
            } in body['input']
            items = [
                {
                    'type': 'function_call',
                    'id': 'fc_2',
                    'call_id': 'call_2',
                    'name': 'submit_result',
                    'arguments': '{"summary":"record-13"}',
                }
            ]
        items.insert(0, _response('Calling the tool.').output[0].model_dump(mode='json'))
        event = _terminal(_response(output=items)).model_dump(mode='json')
        delta = {
            'type': 'response.output_text.delta',
            'sequence_number': 0,
            'item_id': 'msg_1',
            'output_index': 0,
            'content_index': 0,
            'delta': 'Calling the tool.',
            'logprobs': [],
        }
        return httpx.Response(
            200,
            headers={'Content-Type': 'text/event-stream'},
            text=(
                'event: response.output_text.delta\ndata: ' + json.dumps(delta) + '\n\n'
                if stream_preamble
                else ''
            )
            + 'event: response.completed\ndata: '
            + json.dumps(event)
            + '\n\n',
        )

    def client(**kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(
            **kwargs,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(llm_module, 'AsyncOpenAI', client)
    agent = create_structured_agent(
        llm_config=SampleOpenAIConfig(),
        response_model=_SummaryOutput,
        tools=[Tool.from_model(Inspect, inspect, name='inspect')],
        extensions=(StructuredOutput(),),
    )

    async def receive_chunk(call: AgentCall) -> None:
        chunks.append(call.chunk)

    agent.events.subscribe(CHUNK_RECEIVED, receive_chunk)
    result = asyncio.run(agent.invoke(StructuredInput(prompt='Inspect A-13.')))
    assert result.output == _SummaryOutput(summary='record-13')
    assert len(requests) == 2
    assert chunks == ['Calling the tool.', 'Calling the tool.']
