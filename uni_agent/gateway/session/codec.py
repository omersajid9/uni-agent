"""Model-scoped codec for tokenizer, processor, tool-parser, and decode paths.

This layer stays within the model boundary: it applies chat templates, handles
processor-backed multimodal inputs, parses tools, and decodes backend outputs.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template as _apply_chat_template
from verl.utils.tokenizer.chat_template import initialize_system_prompt

logger = logging.getLogger("gateway")

# Map backend stop_reason values into the gateway's internal finish_reason vocabulary.
_FINISH_REASON_MAP = {
    "completed": "stop",
    "stop": "stop",
    "matched_stop": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "aborted": "stop",
    "abort": "stop",
}

_SGLANG_TOOL_PARSER_ALIASES = {
    "qwen3_xml": "qwen3_coder",
}

_VLLM_TOOL_PARSER_ALIASES = {
    "qwen": "qwen3_xml",
    "qwen25": "qwen3_xml",
    "qwen3": "qwen3_xml",
}


def _canonicalize_tool_arguments_for_comparison(arguments: Any) -> tuple[str, Any]:
    if isinstance(arguments, dict | list):
        return ("json", arguments)
    if isinstance(arguments, str):
        try:
            return ("json", json.loads(arguments))
        except json.JSONDecodeError:
            return ("raw", arguments)
    return ("raw", arguments)


def _process_tool_calls_sglang(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
) -> tuple[str, list[Any]]:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    sglang_tools = [SglTool(type=tool["type"], function=SglFunction(**tool["function"])) for tool in tools]
    parser = FunctionCallParser(sglang_tools, parser_name)
    if not parser.has_tool_call(text):
        return text, []

    content, calls = parser.parse_non_stream(text)
    return content, [SimpleNamespace(name=call.name, arguments=call.parameters) for call in calls]


def _process_tool_calls_vllm(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
    tokenizer,
) -> tuple[str, list[Any]]:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.tool_parsers import ToolParserManager

    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = SimpleNamespace(
        tools=[ChatCompletionToolsParam(**tool) if isinstance(tool, dict) else tool for tool in tools],
        tool_choice="auto",
        skip_special_tokens=True,
    )
    parsed = parser.extract_tool_calls(text, request)
    if not parsed.tools_called:
        return text, []
    return parsed.content or "", [tool_call.function for tool_call in parsed.tool_calls]


_HERMES_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _extract_hermes_call_name(raw_payload: str) -> str:
    """Best-effort ``"name"`` recovery from a ``<tool_call>`` body that failed to parse as JSON.

    The body as a whole is broken (bad escaping, a stray brace, a hallucinated extra field, ...)
    but the ``"name"`` field is almost always a clean quoted string near the front, so a small
    regex recovers enough of it to route the call into :meth:`Toolbox.call`'s existing
    malformed-argument handling rather than dropping the call outright.
    """
    match = _HERMES_NAME_RE.search(raw_payload)
    if match is None:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _process_tool_calls_hermes(text: str, tools: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Parse the Hermes ``<tool_call>{json}</tool_call>`` envelope with no inference engine.

    The only parser here that has no third-party dependency, which is what makes it the fallback
    rather than an alternative: neither vLLM nor SGLang installs on every accelerator (AWS Neuron
    has neither), and without a parser the gateway hands the agent an assistant message whose tool
    call is still sitting in ``content`` -- the loop then reads "no tool calls" and ends the episode
    after one turn instead of failing, which is the kind of silence that looks like a bad policy.

    Deliberately narrow: exactly the Qwen chat template's envelope. A call naming a tool that was
    not offered is dropped rather than passed on. A call whose JSON body fails to parse (a policy
    still learning the format -- bad string escaping, an extra hallucinated field, mismatched
    braces) is *not* dropped: it is forwarded with its raw, still-broken text as ``arguments`` so
    :meth:`Toolbox.call` re-parses it and turns the same error into an "Invalid action: ..."
    observation. That keeps the episode going and gives the policy a training signal to correct
    its formatting instead of the turn silently reading as "no tool call" and ending the episode.
    """
    offered = {
        (tool.get("function") or {}).get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        for tool in tools
    }
    calls = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        raw_payload = match.group(1)
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            logger.warning("hermes tool-call payload is not valid JSON; surfacing the parse error to the model")
            calls.append(SimpleNamespace(name=_extract_hermes_call_name(raw_payload), arguments=raw_payload))
            continue
        name = payload.get("name")
        if name not in offered:
            logger.warning("model called tool %r which was not offered; skipping it", name)
            continue
        calls.append(SimpleNamespace(name=name, arguments=payload.get("arguments", {})))
    if not calls:
        return text, []
    # Whatever the model wrote outside the envelopes is the assistant's prose (its reasoning).
    content = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", text, flags=re.DOTALL).strip()
    return content, calls


#: Parsers that need no inference engine, keyed by the format name the config asks for. Consulted
#: only after SGLang and vLLM have both turned out to be unimportable.
_BUILTIN_TOOL_PARSERS = {"hermes": _process_tool_calls_hermes}


def _extract_tool_calls(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
    tokenizer,
) -> tuple[str, list[Any]]:
    sglang_name = _SGLANG_TOOL_PARSER_ALIASES.get(parser_name, parser_name)
    try:
        return _process_tool_calls_sglang(text, tools, sglang_name)
    except ModuleNotFoundError:
        pass
    except Exception:
        logger.warning("SGLang tool-call parsing failed; trying vLLM", exc_info=True)

    vllm_name = _VLLM_TOOL_PARSER_ALIASES.get(parser_name, parser_name)
    try:
        return _process_tool_calls_vllm(text, tools, vllm_name, tokenizer)
    except ModuleNotFoundError:
        pass
    except Exception:
        logger.warning("vLLM tool-call parsing failed; trying the built-in parser", exc_info=True)

    builtin = _BUILTIN_TOOL_PARSERS.get(parser_name)
    if builtin is None:
        logger.warning(
            "no tool parser available for format %r: neither SGLang nor vLLM is importable and "
            "there is no built-in parser for it, so tool calls will stay in the message content",
            parser_name,
        )
        return text, []
    return builtin(text, tools)


class MessageCodec:
    """Model-scoped request codec used by gateway sessions.

    ``_GatewayActor`` owns one codec per actor and injects it into
    ``GatewaySession`` instances. The codec renders chat templates, handles
    multimodal processor inputs, and decodes backend token outputs without
    reading session state.
    """

    def __init__(
        self,
        tokenizer,
        *,
        processor=None,
        vision_info_extractor=None,
        vision_info_extractor_kwargs: dict[str, Any] | None = None,
        tool_parser_name: str | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self._vision_info_extractor = vision_info_extractor or self._default_vision_info_extractor
        self._vision_info_extractor_kwargs = dict(vision_info_extractor_kwargs or {})
        self._apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self._system_prompt = initialize_system_prompt(
            self._processor if self._processor is not None else tokenizer,
            **self._apply_chat_template_kwargs,
        )
        self._tool_parser_name = tool_parser_name

    async def _default_vision_info_extractor(
        self,
        messages: list[dict[str, Any]],
        *,
        image_patch_size: int,
        **_extra: Any,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        # Lazy import so callers without multi-modal needs do not load
        # qwen_vl_utils. ``_extra`` absorbs ``vision_info_extractor_kwargs`` that
        # ``extract_multi_modal_data`` forwards for custom extractors; the
        # default path needs nothing beyond ``messages`` and patch size.
        from qwen_vl_utils import process_vision_info

        return process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )

    async def extract_multi_modal_data(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any] | None, list[Any] | None]:
        """Extract image and video inputs when a processor-backed request needs them."""
        if self._processor is None:
            return None, None

        has_multi_modal_blocks = False
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url", "video", "video_url"}:
                    has_multi_modal_blocks = True
                    break
            if has_multi_modal_blocks:
                break

        if not has_multi_modal_blocks:
            return None, None

        return await self._vision_info_extractor(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            **self._vision_info_extractor_kwargs,
        )

    def encode_full(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode a full chat history into prompt token IDs."""
        if self._processor is not None:
            raw_prompt = _apply_chat_template(
                self._processor,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **self._apply_chat_template_kwargs,
            )
            videos = video_data
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos, strict=False)
                videos, video_metadata = list(videos), list(video_metadata)
            model_inputs = self._processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            )
            return normalize_token_ids(model_inputs["input_ids"])

        return normalize_token_ids(
            _apply_chat_template(
                self._tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                **self._apply_chat_template_kwargs,
            )
        )

    # TODO: check if delta tokenization is better than remove_system_prompt
    def encode_incremental(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode continuation messages without the cached system prompt prefix."""
        if self._processor is not None:
            raw_prompt = _apply_chat_template(
                self._processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self._apply_chat_template_kwargs,
            )
            videos = video_data
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos, strict=False)
                videos, video_metadata = list(videos), list(video_metadata)
            model_inputs = self._processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            )
            ids = normalize_token_ids(model_inputs["input_ids"])
        else:
            ids = normalize_token_ids(
                _apply_chat_template(
                    self._tokenizer,
                    messages,
                    add_generation_prompt=True,
                    **self._apply_chat_template_kwargs,
                )
            )
        return ids[len(self._system_prompt) :]

    async def decode_response(
        self,
        response_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop_reason: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Decode model output tokens into an assistant message and finish reason."""
        if self._tool_parser_name and tools:
            response_text = self._tokenizer.decode(response_ids, skip_special_tokens=False)
            content, function_calls = _extract_tool_calls(
                response_text,
                tools,
                self._tool_parser_name,
                self._tokenizer,
            )
            if function_calls:
                tool_calls = [
                    {
                        "id": f"call_{uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": fc.name, "arguments": fc.arguments},
                    }
                    for fc in function_calls
                ]
                message = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
                return message, "tool_calls"
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, stop_reason) if stop_reason else "stop"
        return {"role": "assistant", "content": response_text}, finish_reason

    def canonicalize_message_for_prefix_comparison(self, message: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize one message before session prefix comparison."""
        normalized = dict(message)
        normalized.pop("tool_call_id", None)
        tool_calls = normalized.get("tool_calls")
        if not isinstance(tool_calls, list):
            return normalized

        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            normalized_tool_call = dict(tool_call)
            normalized_tool_call.pop("id", None)
            function = normalized_tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                normalized_function = dict(function)
                normalized_function["arguments"] = _canonicalize_tool_arguments_for_comparison(function["arguments"])
                normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)
        normalized["tool_calls"] = normalized_tool_calls
        return normalized
