"""Restricted execution of model-written schedule-search strategies.

The model writes a function ``strategy(api)`` in a small subset of Python. The
code is parsed and checked before it runs:
- No imports, attribute access to private/dunder names, ``exec``/``eval``,
  file, network or process access, ``global``/``nonlocal``, classes,
  lambdas over names outside the allowed builtins, or ``while`` loops.
- It runs with a whitelisted builtins table and a step budget enforced by a
  trace function, plus a wall-clock limit.

The only object the strategy can touch is ``api`` (a ``StrategyAPI``), which
exposes read-only facts about the current instance and two tools:
``api.predict(config)`` (fast continuous estimate, no build) and
``api.measure(config)`` (full build plus published-formula Replay estimate).
The strategy returns a list of configurations; the agent re-checks each one
independently and the final answer is always fully validated.
"""

from __future__ import annotations

import ast
import math
import sys
from time import perf_counter
from typing import Any, Callable


class StrategyError(Exception):
    """Raised when strategy code is rejected or fails while running."""


_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
    ast.AnnAssign, ast.For, ast.If, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Call, ast.keyword,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.USub, ast.UAdd, ast.Not, ast.Invert, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Lambda, ast.Starred,
)

_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max",
        "min", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip", "isinstance",
        "str", "print",
    )
}
_SAFE_BUILTINS["math_log"] = math.log
_SAFE_BUILTINS["math_sqrt"] = math.sqrt
_SAFE_BUILTINS["True"] = True
_SAFE_BUILTINS["False"] = False
_SAFE_BUILTINS["None"] = None

_API_ATTRIBUTES = {
    "operator", "ranks", "splits", "max_rounds", "max_width", "algorithms", "steps", "dims",
    "bytes_per_rank", "predict", "measure", "best", "log", "time_left", "size_weight", "sweep",
}


def check_source(source: str) -> ast.Module:
    """Parse and statically check strategy code; return the module AST."""

    if len(source) > 6000:
        raise StrategyError("strategy code is longer than 6000 characters")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise StrategyError(f"syntax error line {exc.lineno}: {exc.msg}") from None
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != len(functions) or not any(node.name == "strategy" for node in functions):
        raise StrategyError("code must contain only function definitions, including strategy(api)")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise StrategyError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.returns is not None):
            raise StrategyError(f"line {node.lineno}: decorators and return annotations are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise StrategyError(f"line {node.lineno}: name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise StrategyError(f"line {node.lineno}: private attribute {node.attr!r} is not allowed")
            if isinstance(node.value, ast.Name) and node.value.id == "api" and node.attr not in _API_ATTRIBUTES:
                raise StrategyError(f"line {node.lineno}: api.{node.attr} does not exist")
            if node.attr not in _API_ATTRIBUTES and node.attr not in {
                "append", "extend", "get", "items", "keys", "values", "copy", "update", "pop", "sort", "index", "count",
                "bit_length",
            }:
                raise StrategyError(f"line {node.lineno}: attribute .{node.attr} is not allowed")
    _check_names(tree)
    return tree


_BLOCKED_NAMES = {"exec", "eval", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals",
                  "vars", "dir", "type", "object", "input", "breakpoint", "memoryview", "help", "id", "hash"}


def _check_names(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise StrategyError(f"line {node.lineno}: name {node.id!r} is not allowed")
        if isinstance(node, (ast.arg,)) and node.arg.startswith("_"):
            raise StrategyError(f"line {node.lineno}: argument {node.arg!r} is not allowed")


def run_strategy(source: str, api: Any, *, max_steps: int = 2_000_000, max_seconds: float = 3.0) -> Any:
    """Run ``strategy(api)`` from checked source under step and time limits."""

    tree = check_source(source)
    code = compile(tree, "<strategy>", "exec")
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(code, namespace)  # noqa: S102 - code is AST-checked and runs with whitelisted builtins only
    function: Callable[[Any], Any] = namespace["strategy"]
    steps = 0
    deadline = perf_counter() + max_seconds
    previous = sys.gettrace()

    def tracer(frame, event, _arg):
        nonlocal steps
        if frame.f_code.co_filename != "<strategy>":
            return None
        steps += 1
        if steps > max_steps:
            raise StrategyError(f"strategy exceeded {max_steps} steps")
        if steps % 2000 == 0 and perf_counter() > deadline:
            raise StrategyError(f"strategy exceeded {max_seconds} s")
        return tracer

    sys.settrace(tracer)
    try:
        return function(api)
    except StrategyError:
        raise
    except Exception as exc:  # report the model's runtime error back to it
        line = None
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == "<strategy>":
                line = tb.tb_lineno
            tb = tb.tb_next
        raise StrategyError(f"{type(exc).__name__} at line {line}: {exc}") from None
    finally:
        sys.settrace(previous)
