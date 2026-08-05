"""``calculator``: evaluate one arithmetic expression, host-side.

Unlike the shell / editor tools this one never touches the sandbox -- it is a pure
function of its arguments, which is exactly what makes it usable as the smallest
possible multi-turn RL domain: an episode is ``calculator`` then ``finish``, with no
container to provision. The expression grammar is restricted to arithmetic over
numeric literals (no names, calls, or attribute access), so evaluating it is safe.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

from .base import Tool, ToolError, ToolResult, register_tool

DESCRIPTION = """
Evaluate an arithmetic expression and return its value.
Supports + - * / // % ** and parentheses over numbers, e.g. "17 * 23".
""".strip()

#: Cap on ``**`` so a single call cannot burn the rollout on a huge integer power.
_MAX_EXPONENT = 64

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorArguments(BaseModel):
    expression: str = Field(description="Arithmetic expression to evaluate, e.g. '17 * 23'.")


def evaluate_expression(expression: str) -> float | int:
    """Evaluate an arithmetic ``expression``, raising :class:`ToolError` if it is not one.

    Parses with :mod:`ast` and walks only the arithmetic node types, so nothing in the
    expression can name, call, or reach anything in the host process.
    """
    text = expression.strip()
    if not text:
        raise ToolError("expression is empty")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse {text!r} as an arithmetic expression: {exc.msg}") from None
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolError(f"unsupported literal {node.value!r}; only numbers are allowed")
        return node.value
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported unary operator {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported operator {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError(f"exponent {right} exceeds the limit of {_MAX_EXPONENT}")
        try:
            return op(left, right)
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
    raise ToolError(f"unsupported expression element {type(node).__name__}")


def format_number(value: float | int) -> str:
    """Render a result the way the reward compares it: integers without a decimal tail."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError(f"result is not finite ({value})")
        if value.is_integer():
            return str(int(value))
        return repr(round(value, 6))
    return str(value)


@register_tool("calculator")
class CalculatorTool(Tool):
    name = "calculator"
    description = DESCRIPTION
    args_model = CalculatorArguments

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        expression = args.get("expression")
        if not isinstance(expression, str):
            raise ToolError("calculator requires a string 'expression' argument")
        return ToolResult(text=format_number(evaluate_expression(expression)))