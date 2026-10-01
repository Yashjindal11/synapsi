"""A safe arithmetic/statistics evaluator. Never uses ``eval``."""

from __future__ import annotations

import ast
import math
import operator
import statistics
from collections.abc import Callable
from typing import Any

from synapsi.evidence.models import SourceKind
from synapsi.tools.base import Tool

_MAX_EXPRESSION = 500
_MAX_EXPONENT = 1000
_MAX_SEQUENCE = 10_000

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "mean": statistics.fmean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "pstdev": statistics.pstdev,
    "variance": statistics.variance,
    "comb": math.comb,
    "factorial": math.factorial,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


def evaluate(expression: str) -> float | int:
    if len(expression) > _MAX_EXPRESSION:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc.msg}") from exc
    result = _eval(tree.body)
    if not isinstance(result, int | float):
        raise CalculatorError("expression must evaluate to a number")
    return result


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise CalculatorError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.List | ast.Tuple):
        if len(node.elts) > _MAX_SEQUENCE:
            raise CalculatorError("sequence too long")
        return [_eval(e) for e in node.elts]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > 1000:
            raise CalculatorError("factorial argument too large")
        return _FUNCTIONS[node.func.id](*args)
    raise CalculatorError(f"unsupported syntax: {ast.dump(node)[:60]}")


def calculate(expression: str) -> str:
    return f"{expression} = {evaluate(expression)!r}"


def calculator_tool() -> Tool:
    names = ", ".join(sorted(_FUNCTIONS))
    return Tool(
        "calculator",
        f"Evaluate an arithmetic expression. Functions: {names}; lists like mean([1,2,3]).",
        calculate,
        parameters={"expression": "str"},
        evidence_kind=SourceKind.CALCULATION,
    )
