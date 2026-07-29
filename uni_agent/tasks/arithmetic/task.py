"""Arithmetic task: solve one arithmetic problem with the ``calculator`` tool.

The point of this task is to be the smallest thing that still exercises every
property the SWE tasks depend on -- multiple model turns, a tool observation folded
back into the context, a trajectory-level reward -- while needing no container
image, no Modal, and no long context. It is what the first end-to-end Trainium run
trains on before the same path is pointed at a real SWE sample.

The sample lives in :attr:`~uni_agent.tasks.base.TaskConfig.metadata` as
``{"expression": "17 * 23", "answer": 391}``; the reward compares the answer the
policy passes to ``finish`` against ``answer``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

#: Tools whose ``answer`` argument counts as the episode's final answer.
_FINISH_TOOLS = ("finish", "submit")


class ArithmeticTaskConfig(TaskConfig):
    name: str = "arithmetic"
    calculator_reward: float = Field(
        default=0.2,
        description="Partial credit when the policy called the calculator and it returned the correct "
        "value but the final answer was wrong or missing. Gives the first training runs a gradient "
        "signal before a small model reliably formats its final answer.",
    )
    answer_tolerance: float = Field(
        default=1e-6,
        description="Absolute tolerance when comparing the policy's answer to the expected one.",
    )


@register_task("arithmetic")
class ArithmeticTask(Task):
    name = "arithmetic"
    config_model = ArithmeticTaskConfig

    async def run(self) -> TaskResult:
        cfg: ArithmeticTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata if isinstance(cfg.metadata, dict) else {}
        logger.info(
            "starting arithmetic task (expression=%r answer=%r)",
            sample.get("expression"),
            sample.get("answer"),
        )
        # The calculator and finish tools are pure host-side functions, so the sandbox is
        # only here to satisfy the agent contract -- `local` starts and stops as a no-op.
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            result = await agent.run(sandbox=sandbox, messages=cfg.prompt)
        return await self.score(result.transcript, result.info)

    async def score(self, transcript: list[dict[str, Any]], info: dict[str, Any]) -> TaskResult:
        """Score a finished episode's ``transcript``. Split out from :meth:`run` so the
        reward can be tested without an endpoint or a sandbox."""
        cfg: ArithmeticTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata if isinstance(cfg.metadata, dict) else {}
        expected = sample.get("answer")
        if expected is None:
            raise ValueError("arithmetic task requires metadata['answer'] (the expected result)")

        answer = extract_final_answer(transcript)
        answered_correctly = _matches(answer, expected, cfg.answer_tolerance)
        calculator_correct = _calculator_produced(transcript, expected, cfg.answer_tolerance)

        if answered_correctly:
            reward = 1.0
        elif calculator_correct:
            reward = float(cfg.calculator_reward)
        else:
            reward = 0.0

        result_info: dict[str, Any] = {
            "expression": sample.get("expression"),
            "expected_answer": expected,
            "final_answer": answer,
            "answer_correct": answered_correctly,
            "calculator_correct": calculator_correct,
            "exit_reason": info.get("exit_reason"),
            "num_tool_calls": info.get("num_tool_calls"),
        }
        logger.info("task done: reward=%s info=%s", reward, result_info)
        return TaskResult(reward=reward, accuracy=float(answered_correctly), info=result_info)


def extract_final_answer(transcript: list[dict[str, Any]]) -> str | None:
    """Return the ``answer`` from the last finish/submit call, else the last assistant text.

    Falling back to the trailing assistant message matters early in training: a small
    policy often writes the answer as plain text instead of calling ``finish``, and
    scoring that as zero would leave the whole GRPO group flat.
    """
    for message in reversed(transcript):
        if message.get("role") != "assistant":
            continue
        for tool_call in reversed(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            if function.get("name") in _FINISH_TOOLS:
                arguments = _parse_arguments(function.get("arguments"))
                answer = arguments.get("answer")
                if answer is not None:
                    return str(answer)
        content = message.get("content")
        if content:
            return str(content)
    return None


def _calculator_produced(
    transcript: list[dict[str, Any]], expected: Any, tolerance: float
) -> bool:
    """Whether some ``calculator`` observation in the transcript holds the expected value."""
    calculator_call_ids = set()
    for message in transcript:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if (tool_call.get("function") or {}).get("name") == "calculator":
                    calculator_call_ids.add(tool_call.get("id"))
        elif message.get("role") == "tool" and (
            message.get("name") == "calculator" or message.get("tool_call_id") in calculator_call_ids
        ):
            if _matches(message.get("content"), expected, tolerance):
                return True
    return False


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _matches(text: Any, expected: Any, tolerance: float) -> bool:
    """Whether ``text`` contains the expected number (last number wins)."""
    if text is None:
        return False
    try:
        expected_value = float(expected)
    except (TypeError, ValueError):
        return str(expected).strip() == str(text).strip()
    for candidate in reversed(_numbers_in(str(text))):
        if math.isclose(candidate, expected_value, abs_tol=tolerance, rel_tol=0.0):
            return True
    return False


def _numbers_in(text: str) -> list[float]:
    """Every number in ``text``, ignoring thousands separators."""
    numbers = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return numbers
