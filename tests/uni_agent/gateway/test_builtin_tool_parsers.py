from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.session.builtin_tool_parsers import (
    _process_tool_calls_hermes,
    extract_tool_calls_with_builtin_fallback,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


def test_hermes_parser_extracts_call_and_strips_envelope_from_content():
    content, calls = _process_tool_calls_hermes(
        'I will search.\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>',
        TOOLS,
    )

    assert content == "I will search."
    assert [(call.name, call.arguments) for call in calls] == [("search", {"query": "x"})]


def test_hermes_parser_drops_calls_to_tools_that_were_not_offered():
    content, calls = _process_tool_calls_hermes(
        '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>',
        TOOLS,
    )

    assert content == '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>'
    assert calls == []


def test_hermes_parser_skips_malformed_json_payloads():
    content, calls = _process_tool_calls_hermes("<tool_call>{not json}</tool_call>", TOOLS)

    assert content == "<tool_call>{not json}</tool_call>"
    assert calls == []


def test_hermes_parser_returns_text_unchanged_when_there_is_no_envelope():
    content, calls = _process_tool_calls_hermes("just a plain reply", TOOLS)

    assert content == "just a plain reply"
    assert calls == []


def test_fallback_prefers_the_engine_extractor_when_it_finds_calls():
    def engine_extractor(text, tools, parser_name, tokenizer):
        return "visible", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    content, calls = extract_tool_calls_with_builtin_fallback(
        "raw", TOOLS, "hermes", FakeTokenizer(), engine_extractor
    )

    assert content == "visible"
    assert calls[0].name == "search"


def test_fallback_runs_the_builtin_parser_when_the_engine_extractor_finds_nothing():
    """This is the AWS Neuron / mini-verl case: neither vLLM nor SGLang is importable, so
    codec.py's engine extractor always returns ``(text, [])``. Without this fallback, an
    unparsed tool call stays in ``content`` and the agent, seeing no tool calls, ends the
    episode after one turn -- a silent single-turn degradation, not an error."""

    def engine_extractor_with_no_inference_backend(text, tools, parser_name, tokenizer):
        return text, []

    content, calls = extract_tool_calls_with_builtin_fallback(
        'I will search.\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>',
        TOOLS,
        "hermes",
        FakeTokenizer(),
        engine_extractor_with_no_inference_backend,
    )

    assert content == "I will search."
    assert [(call.name, call.arguments) for call in calls] == [("search", {"query": "x"})]


def test_fallback_has_no_builtin_for_an_unknown_format():
    def engine_extractor_with_no_inference_backend(text, tools, parser_name, tokenizer):
        return text, []

    content, calls = extract_tool_calls_with_builtin_fallback(
        '<tool_call>{"name": "search", "arguments": {}}</tool_call>',
        TOOLS,
        "qwen3_xml",
        FakeTokenizer(),
        engine_extractor_with_no_inference_backend,
    )

    assert content == '<tool_call>{"name": "search", "arguments": {}}</tool_call>'
    assert calls == []


@pytest.mark.asyncio
async def test_decode_response_falls_back_to_builtin_hermes_parser(monkeypatch):
    """End-to-end through MessageCodec.decode_response with the real dispatch call site."""
    import uni_agent.gateway.session.codec as codec_mod
    from uni_agent.gateway.session.codec import MessageCodec

    def missing_backend(*args, **kwargs):
        raise ModuleNotFoundError("no inference engine")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_backend, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", missing_backend, raising=False)

    codec = MessageCodec(FakeTokenizer(), tool_parser_name="hermes")
    message, finish_reason = await codec.decode_response(
        [ord(char) for char in '<tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>'],
        tools=TOOLS,
        stop_reason="stop",
    )

    assert finish_reason == "tool_calls"
    assert message["tool_calls"][0]["function"] == {"name": "search", "arguments": {"query": "x"}}
