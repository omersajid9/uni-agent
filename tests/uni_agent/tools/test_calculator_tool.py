"""Tests for the ``calculator`` tool: what it evaluates and what it refuses.

The refusals matter more than the arithmetic. The tool runs host-side, in the training
process, with a string the policy wrote, so it must never reach anything outside the
arithmetic node types -- ``__import__`` and friends have to come back as a
:class:`ToolError` observation, not execute.
"""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.tools import ToolError
from uni_agent.tools.calculator import CalculatorTool, evaluate_expression, format_number


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("17 * 23", 391),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-7 + 10", 3),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("1 / 4", 0.25),
        ("  42  ", 42),
    ],
)
def test_evaluates_arithmetic(expression, expected):
    assert evaluate_expression(expression) == expected


@pytest.mark.parametrize(
    "expression, needle",
    [
        ("", "empty"),
        ("__import__('os').system('ls')", "unsupported expression element"),
        ("open('/etc/passwd').read()", "unsupported expression element"),
        ("x + 1", "unsupported expression element"),
        ("[1, 2]", "unsupported expression element"),
        ("'a' * 3", "only numbers are allowed"),
        ("True + 1", "only numbers are allowed"),
        ("1 / 0", "division by zero"),
        ("2 ** 10000", "exceeds the limit"),
        ("17 *", "could not parse"),
    ],
)
def test_rejects_non_arithmetic(expression, needle):
    with pytest.raises(ToolError, match=needle):
        evaluate_expression(expression)


@pytest.mark.parametrize(
    "value, rendered",
    [(391, "391"), (4.0, "4"), (0.25, "0.25"), (1 / 3, "0.333333")],
)
def test_format_number_drops_integral_decimal_tails(value, rendered):
    assert format_number(value) == rendered


def test_tool_returns_the_value_as_an_observation():
    tool = CalculatorTool(sandbox=object())  # arithmetic never touches the sandbox
    result = asyncio.run(tool.run({"expression": "17 * 23"}))
    assert result.status == "ok"
    assert result.text == "391"
    assert result.to_observation() == "Observation:\n391"


def test_tool_requires_a_string_expression():
    tool = CalculatorTool(sandbox=object())
    with pytest.raises(ToolError, match="requires a string 'expression'"):
        asyncio.run(tool.run({"expression": 17}))


def test_tool_is_registered_under_its_name():
    from uni_agent.tools.base import TOOL_REGISTRY

    assert TOOL_REGISTRY["calculator"] is CalculatorTool


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
