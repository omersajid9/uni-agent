"""Tests for the arithmetic task's scoring -- the reward the GRPO run actually sees.

Two behaviours are worth pinning down. First, the answer is read from the ``finish``
call when there is one and from the trailing assistant text when there isn't, because a
small policy early in training often just writes the number. Second, a run that
computed the right value with the calculator but fluffed the final answer gets partial
credit, so a GRPO group is not uniformly zero (which would give no advantage at all).

Scoring is pure transcript inspection, so these run with no sandbox and no endpoint.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from uni_agent.tasks.arithmetic.task import ArithmeticTask, ArithmeticTaskConfig, extract_final_answer


def _assistant(content, *, tool=None, arguments=None, call_id="call_0"):
    message = {"role": "assistant", "content": content}
    if tool is not None:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments or {})},
            }
        ]
    return message


def _observation(text, *, name="calculator", call_id="call_0"):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": f"Observation:\n{text}"}


def _score(transcript, *, answer=391, **config_kwargs):
    """Run the task's scoring over a canned ``transcript`` (no agent, no sandbox)."""
    task = ArithmeticTask(
        ArithmeticTaskConfig(
            metadata={"expression": "17 * 23", "answer": answer},
            sandbox={"provider": "local"},
            **config_kwargs,
        )
    )
    return asyncio.run(task.score(transcript, info={"exit_reason": "finished", "num_tool_calls": 2}))


_SOLVED = [
    {"role": "user", "content": "What is 17 * 23?"},
    _assistant("Let me compute it.", tool="calculator", arguments={"expression": "17 * 23"}),
    _observation("391"),
    _assistant("Done.", tool="finish", arguments={"answer": "391"}, call_id="call_1"),
]


def test_correct_answer_scores_one():
    result = _score(_SOLVED)
    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.info["final_answer"] == "391"


def test_wrong_answer_after_a_correct_calculator_call_gets_partial_credit():
    transcript = _SOLVED[:-1] + [
        _assistant("Done.", tool="finish", arguments={"answer": "319"}, call_id="call_1")
    ]
    result = _score(transcript)
    assert result.reward == pytest.approx(0.2)
    assert result.accuracy == 0.0
    assert result.info["calculator_correct"] is True


def test_wrong_throughout_scores_zero():
    transcript = [
        {"role": "user", "content": "What is 17 * 23?"},
        _assistant("It is 100.", tool="finish", arguments={"answer": "100"}),
    ]
    result = _score(transcript)
    assert result.reward == 0.0
    assert result.info["calculator_correct"] is False


def test_plain_text_answer_still_counts():
    transcript = [
        {"role": "user", "content": "What is 17 * 23?"},
        _assistant("The answer is 391."),
    ]
    assert _score(transcript).reward == 1.0


def test_answer_read_from_the_last_finish_call():
    transcript = _SOLVED + [
        _assistant("Actually...", tool="finish", arguments={"answer": "392"}, call_id="call_2")
    ]
    assert extract_final_answer(transcript) == "392"


def test_thousands_separators_and_float_tails_match():
    assert _score([_assistant("The answer is 1,024.")], answer=1024).reward == 1.0
    assert _score([_assistant("It is 391.0")], answer=391).reward == 1.0


def test_empty_transcript_scores_zero():
    result = _score([])
    assert result.reward == 0.0
    assert result.info["final_answer"] is None


def test_missing_expected_answer_is_a_config_error():
    task = ArithmeticTask(
        ArithmeticTaskConfig(metadata={"expression": "17 * 23"}, sandbox={"provider": "local"})
    )
    with pytest.raises(ValueError, match="metadata\\['answer'\\]"):
        asyncio.run(task.score([], info={}))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))