from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeTokenizer

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


def test_tool_call_dispatch_prefers_sglang(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def fake_sglang(text, tools, parser_name):
        seen["sglang"] = (text, tools, parser_name)
        return "visible", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    def fail_vllm(*args, **kwargs):
        raise AssertionError("vLLM should not run when SGLang succeeds")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", fake_sglang, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", fail_vllm, raising=False)

    content, calls = codec_mod._extract_tool_calls("raw", TOOLS, "hermes", FakeTokenizer())

    assert content == "visible"
    assert calls[0].name == "search"
    assert seen["sglang"] == ("raw", TOOLS, "hermes")


def test_tool_call_dispatch_falls_back_to_vllm_with_name_mapping(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def missing_sglang(*args, **kwargs):
        raise ModuleNotFoundError("sglang")

    def fake_vllm(text, tools, parser_name, tokenizer):
        seen["vllm"] = (text, tools, parser_name, tokenizer)
        return "", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_sglang, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", fake_vllm, raising=False)

    tokenizer = FakeTokenizer()
    content, calls = codec_mod._extract_tool_calls("raw", TOOLS, "qwen25", tokenizer)

    assert content == ""
    assert calls[0].arguments == '{"query":"x"}'
    assert seen["vllm"] == ("raw", TOOLS, "qwen3_xml", tokenizer)


@pytest.fixture
def no_inference_engine(monkeypatch):
    """Neither SGLang nor vLLM importable — the situation on an accelerator that has neither."""
    import uni_agent.gateway.session.codec as codec_mod

    def missing_backend(*args, **kwargs):
        raise ModuleNotFoundError("tool parser backend")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_backend, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", missing_backend, raising=False)
    return codec_mod


def test_tool_call_dispatch_returns_text_when_there_is_nothing_to_parse(no_inference_engine):
    content, calls = no_inference_engine._extract_tool_calls("plain text", TOOLS, "hermes", FakeTokenizer())

    assert content == "plain text"
    assert calls == []


def test_tool_call_dispatch_falls_back_to_the_builtin_hermes_parser(no_inference_engine):
    """Without this fallback an unparsed tool call stays in ``content`` and the agent, seeing no tool
    calls, ends the episode after one turn — a silent single-turn degradation, not an error."""
    content, calls = no_inference_engine._extract_tool_calls(
        'I will search.\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>',
        TOOLS,
        "hermes",
        FakeTokenizer(),
    )

    assert content == "I will search."
    assert [(call.name, call.arguments) for call in calls] == [("search", {"query": "x"})]


def test_tool_call_dispatch_has_no_builtin_for_an_unknown_format(no_inference_engine):
    content, calls = no_inference_engine._extract_tool_calls(
        '<tool_call>{"name": "search", "arguments": {}}</tool_call>',
        TOOLS,
        "qwen3_xml",
        FakeTokenizer(),
    )

    assert calls == []
    assert content == '<tool_call>{"name": "search", "arguments": {}}</tool_call>'


def test_builtin_hermes_parser_drops_a_call_to_a_tool_that_was_not_offered(no_inference_engine):
    content, calls = no_inference_engine._extract_tool_calls(
        '<tool_call>{"name": "rm_rf", "arguments": {"path": "/"}}</tool_call>',
        TOOLS,
        "hermes",
        FakeTokenizer(),
    )

    assert calls == []
    assert "rm_rf" in content  # left in the message rather than silently executed


def test_builtin_hermes_parser_surfaces_a_malformed_payload_instead_of_dropping_it(no_inference_engine):
    """An early-training policy that garbles its own JSON (bad string escaping here) should get an
    error it can learn from, not a silently-ended episode."""
    raw = '<tool_call>\n{"name": "search", "arguments": {"query": "a\\ b"}}\n</tool_call>'
    content, calls = no_inference_engine._extract_tool_calls(raw, TOOLS, "hermes", FakeTokenizer())

    assert content == ""  # the (still-broken) envelope is treated as an attempted call, not prose
    assert len(calls) == 1
    assert calls[0].name == "search"  # recovered even though the JSON as a whole doesn't parse
    assert "\\ b" in calls[0].arguments  # raw, still-broken text is forwarded as-is


def test_builtin_hermes_parser_forwards_an_unrecoverable_name_as_empty(no_inference_engine):
    content, calls = no_inference_engine._extract_tool_calls(
        "<tool_call>\nnot even close to json\n</tool_call>",
        TOOLS,
        "hermes",
        FakeTokenizer(),
    )

    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == ""
    assert calls[0].arguments == "not even close to json"


def test_malformed_hermes_call_reaches_toolbox_as_an_actionable_format_error():
    """End-to-end: the malformed call flows through the real ``Toolbox.call`` path (as the ReAct
    loop would use it) and comes back as the same corrective observation a valid call with bad
    arguments would get -- not a silent drop."""
    import asyncio

    from uni_agent.tools import Tool, Toolbox, ToolResult

    class _Search(Tool):
        name = "search"

        async def run(self, args, *, timeout=None):
            return ToolResult(text=f"found: {args}")

    import uni_agent.gateway.session.codec as codec_mod

    content, calls = codec_mod._process_tool_calls_hermes(
        '<tool_call>\n{"name": "search", "arguments": {"query": "a\\ b"}}\n</tool_call>',
        TOOLS,
    )
    assert len(calls) == 1

    toolbox = Toolbox([_Search(sandbox=object())])
    result = asyncio.run(toolbox.call(calls[0].name, calls[0].arguments))

    assert result.status == "format_error"
    assert "Invalid action: could not parse arguments for 'search' as JSON" in result.text


@pytest.mark.asyncio
async def test_decode_response_uses_gateway_dispatcher_for_tool_calls(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    def fake_dispatch(text, tools, parser_name, tokenizer):
        seen["dispatch"] = (text, tools, parser_name, tokenizer)
        return "", [SimpleNamespace(name="search", arguments='{"query":"weather"}')]

    monkeypatch.setattr(codec_mod, "_extract_tool_calls", fake_dispatch, raising=False)

    tokenizer = FakeTokenizer()
    codec = MessageCodec(tokenizer, tool_parser_name="qwen3_xml")
    message, finish_reason = await codec.decode_response(
        [ord(char) for char in "<tool_call>ignored</tool_call>"],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search docs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"anyOf": [{"const": "file"}, {"type": "string"}]},
                        },
                    },
                },
            }
        ],
        stop_reason="stop",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] == ""
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"] == {"name": "search", "arguments": '{"query":"weather"}'}
    assert seen["dispatch"][2] == "qwen3_xml"
