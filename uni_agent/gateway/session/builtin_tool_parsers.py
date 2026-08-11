"""Tool-call parsers that need no inference-engine dependency (no vLLM, no SGLang).

Kept in its own module, separate from ``codec.py``, so that accelerators without vLLM/SGLang
(e.g. AWS Neuron via mini-verl) can gain a fallback parser without editing the SGLang/vLLM
dispatch code that upstream actively maintains. The only integration point with ``codec.py``
is :func:`extract_tool_calls_with_builtin_fallback`, which wraps whatever engine-backed
extractor codec.py already has -- so future upstream changes to the SGLang/vLLM dispatch
internals do not need to touch this file, and changes here do not need to touch codec.py
beyond the one call site.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Any, Callable

logger = logging.getLogger("gateway")

EngineExtractor = Callable[[str, list[dict[str, Any]], str, Any], tuple[str, list[Any]]]

_TOOL_CALL_ENVELOPE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _process_tool_calls_hermes(text: str, tools: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Parse the Hermes ``<tool_call>{json}</tool_call>`` envelope with no inference engine.

    The only parser here that has no third-party dependency, which is what makes it the fallback
    rather than an alternative: neither vLLM nor SGLang installs on every accelerator (AWS Neuron
    has neither), and without a parser the gateway hands the agent an assistant message whose tool
    call is still sitting in ``content`` -- the loop then reads "no tool calls" and ends the episode
    after one turn instead of failing, which is the kind of silence that looks like a bad policy.

    Deliberately narrow: exactly the Qwen chat template's envelope, and a call naming a tool that
    was not offered is dropped rather than passed on.
    """
    offered = {
        (tool.get("function") or {}).get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        for tool in tools
    }
    calls = []
    for match in _TOOL_CALL_ENVELOPE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("hermes tool-call payload is not valid JSON; skipping it")
            continue
        name = payload.get("name")
        if name not in offered:
            logger.warning("model called tool %r which was not offered; skipping it", name)
            continue
        calls.append(SimpleNamespace(name=name, arguments=payload.get("arguments", {})))
    if not calls:
        return text, []
    # Whatever the model wrote outside the envelopes is the assistant's prose (its reasoning).
    content = _TOOL_CALL_ENVELOPE.sub("", text).strip()
    return content, calls


#: Parsers keyed by the tool-parser format name, consulted only after the engine-backed
#: extractor (SGLang/vLLM) has come back with no tool calls -- e.g. because neither engine
#: is importable in this process.
BUILTIN_TOOL_PARSERS: dict[str, Callable[[str, list[dict[str, Any]]], tuple[str, list[Any]]]] = {
    "hermes": _process_tool_calls_hermes,
}


def extract_tool_calls_with_builtin_fallback(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
    tokenizer: Any,
    engine_extractor: EngineExtractor,
) -> tuple[str, list[Any]]:
    """Run ``engine_extractor`` first; fall back to a builtin parser if it found nothing.

    ``engine_extractor`` is codec.py's own SGLang/vLLM dispatcher, passed in rather than
    imported, so this module has no import-time dependency on ``codec.py`` and codec.py's
    internals can keep changing without this file (or its tests) needing to change too.
    """
    content, calls = engine_extractor(text, tools, parser_name, tokenizer)
    if calls:
        return content, calls

    builtin = BUILTIN_TOOL_PARSERS.get(parser_name)
    if builtin is None:
        return content, calls
    return builtin(text, tools)
