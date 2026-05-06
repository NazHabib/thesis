from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List

import pandas as pd


class RuleParseError(ValueError):
    pass


@dataclass(frozen=True)
class AtomicCondition:
    feature: str
    operator: str
    threshold: Any


@dataclass(frozen=True)
class CompiledExpression:
    """
    Safe, compiled rule condition.
    """
    source: str
    ast_dump: str
    feature_names: List[str]
    atomic_conditions: List[AtomicCondition]

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return _eval_node(self._tree, df)  # type: ignore[attr-defined]


_ALLOWED_FUNCTIONS = {"isna", "notna", "between"}
_ALLOWED_COMPARE_OPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)
_ALLOWED_BOOL_OPS = (ast.And, ast.Or)
_ALLOWED_UNARY_OPS = (ast.Not,)
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Call,
)


def compile_condition(expr: str) -> CompiledExpression:
    """
    Compile a safe DSL expression into a reusable object.

    Supported:
    - and / or / not
    - == != < <= > >=
    - in / not in
    - isna(x)
    - notna(x)
    - between(x, lo, hi)   inclusive
    """
    if not isinstance(expr, str) or not expr.strip():
        raise RuleParseError("Condition must be a non-empty string")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleParseError(f"Invalid expression syntax: {expr}") from e

    feature_names: List[str] = []
    _validate_node(tree, feature_names)
    atomic_conditions = _extract_atomic_conditions(tree.body)

    compiled = CompiledExpression(
        source=expr,
        ast_dump=ast.dump(tree, include_attributes=False),
        feature_names=sorted(set(feature_names)),
        atomic_conditions=atomic_conditions,
    )
    object.__setattr__(compiled, "_tree", tree.body)
    return compiled


def evaluate_condition(expr: str, df: pd.DataFrame) -> pd.Series:
    compiled = compile_condition(expr)
    return compiled.evaluate(df)


def apply_missing_policy(mask: pd.Series, policy: str, expr: str = "") -> pd.Series:
    """
    mask may contain True / False / NA.
    Policies:
    - na_false: missing becomes False
    - na_true: missing becomes True
    - na_error: raise if any missing
    """
    if policy == "na_false":
        return mask.fillna(False).astype(bool)
    if policy == "na_true":
        return mask.fillna(True).astype(bool)
    if policy == "na_error":
        if mask.isna().any():
            raise RuleParseError(f"Expression produced NA under na_error policy: {expr}")
        return mask.astype(bool)
    raise RuleParseError(f"Unknown missing_policy: {policy}")


def _validate_node(node: ast.AST, feature_names: List[str]) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise RuleParseError(f"Disallowed syntax: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        _validate_node(node.body, feature_names)
        return

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOL_OPS):
            raise RuleParseError(f"Disallowed boolean operator: {type(node.op).__name__}")
        for v in node.values:
            _validate_node(v, feature_names)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            raise RuleParseError(f"Disallowed unary operator: {type(node.op).__name__}")
        _validate_node(node.operand, feature_names)
        return

    if isinstance(node, ast.Compare):
        _validate_node(node.left, feature_names)
        for op in node.ops:
            if not isinstance(op, _ALLOWED_COMPARE_OPS):
                raise RuleParseError(f"Disallowed comparison operator: {type(op).__name__}")
        for c in node.comparators:
            _validate_node(c, feature_names)
        return

    if isinstance(node, ast.Name):
        feature_names.append(node.id)
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool, type(None))):
            return
        raise RuleParseError(f"Disallowed constant type: {type(node.value).__name__}")

    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            _validate_node(elt, feature_names)
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise RuleParseError("Only direct function calls are allowed")
        if node.func.id not in _ALLOWED_FUNCTIONS:
            raise RuleParseError(f"Function not allowed: {node.func.id}")

        if node.func.id in feature_names:
            feature_names[:] = [f for f in feature_names if f != node.func.id]

        for arg in node.args:
            _validate_node(arg, feature_names)
        if node.keywords:
            raise RuleParseError("Keyword arguments are not allowed")
        _validate_call_arity(node)
        return


def _validate_call_arity(node: ast.Call) -> None:
    fn = node.func.id  # type: ignore[attr-defined]
    argc = len(node.args)
    if fn in {"isna", "notna"} and argc != 1:
        raise RuleParseError(f"{fn} expects exactly 1 argument")
    if fn == "between" and argc != 3:
        raise RuleParseError("between expects exactly 3 arguments")


def _operator_to_text(op: ast.AST) -> str:
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.In):
        return "in"
    if isinstance(op, ast.NotIn):
        return "not in"
    raise RuleParseError(f"Unsupported operator for evidence extraction: {type(op).__name__}")


def _literal_from_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_node(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_from_node(elt) for elt in node.elts)
    return None


def _extract_atomic_conditions(node: ast.AST) -> List[AtomicCondition]:
    """
    Extract simple atomic comparisons for explainability.
    """
    out: List[AtomicCondition] = []

    if isinstance(node, ast.BoolOp):
        for v in node.values:
            out.extend(_extract_atomic_conditions(v))
        return out

    if isinstance(node, ast.UnaryOp):
        out.extend(_extract_atomic_conditions(node.operand))
        return out

    if isinstance(node, ast.Compare):
        if isinstance(node.left, ast.Name) and len(node.ops) == 1 and len(node.comparators) == 1:
            feature = node.left.id
            op = _operator_to_text(node.ops[0])
            threshold = _literal_from_node(node.comparators[0])
            out.append(AtomicCondition(feature=feature, operator=op, threshold=threshold))
        return out

    if isinstance(node, ast.Call):
        return out

    return out


def _series_or_scalar(value: Any, df: pd.DataFrame) -> Any:
    if isinstance(value, pd.Series):
        return value
    return value


def _get_column(name: str, df: pd.DataFrame) -> pd.Series:
    if name not in df.columns:
        raise RuleParseError(f"Column not found in DataFrame: {name}")
    return df[name]


def _eval_node(node: ast.AST, df: pd.DataFrame) -> pd.Series | Any:
    if isinstance(node, ast.Name):
        return _get_column(node.id, df)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_eval_node(elt, df) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, df) for elt in node.elts)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, df)
        if isinstance(node.op, ast.Not):
            if isinstance(operand, pd.Series):
                return ~operand.astype("boolean")
            return not bool(operand)
        raise RuleParseError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BoolOp):
        values = [_ensure_boolean_series(_eval_node(v, df), df) for v in node.values]
        if isinstance(node.op, ast.And):
            out = values[0]
            for v in values[1:]:
                out = out & v
            return out
        if isinstance(node.op, ast.Or):
            out = values[0]
            for v in values[1:]:
                out = out | v
            return out
        raise RuleParseError(f"Unsupported BoolOp: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, df)
        result = None

        current_left = left
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, df)
            partial = _eval_compare(current_left, op, right, df)
            result = partial if result is None else (result & partial)
            current_left = right
        return result

    if isinstance(node, ast.Call):
        fn = node.func.id  # type: ignore[attr-defined]
        args = [_eval_node(a, df) for a in node.args]

        if fn == "isna":
            arg = args[0]
            if isinstance(arg, pd.Series):
                return arg.isna().astype("boolean")
            return pd.Series([pd.isna(arg)] * len(df), index=df.index, dtype="boolean")

        if fn == "notna":
            arg = args[0]
            if isinstance(arg, pd.Series):
                return arg.notna().astype("boolean")
            return pd.Series([pd.notna(arg)] * len(df), index=df.index, dtype="boolean")

        if fn == "between":
            x, lo, hi = args
            x = _ensure_series(x, df)
            return ((x >= lo) & (x <= hi)).astype("boolean")

        raise RuleParseError(f"Unsupported function: {fn}")

    raise RuleParseError(f"Unsupported node during evaluation: {type(node).__name__}")


def _ensure_series(value: Any, df: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(df), index=df.index)


def _ensure_boolean_series(value: Any, df: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype("boolean")
    return pd.Series([bool(value)] * len(df), index=df.index, dtype="boolean")


def _eval_compare(left: Any, op: ast.AST, right: Any, df: pd.DataFrame) -> pd.Series:
    left_is_series = isinstance(left, pd.Series)
    right_is_series = isinstance(right, pd.Series)

    if isinstance(op, ast.In):
        if right_is_series:
            raise RuleParseError("'in' right-hand side cannot be a column/Series")
        left_s = _ensure_series(left, df)
        if not isinstance(right, (list, tuple, set)):
            raise RuleParseError("'in' requires a list/tuple/set on the right-hand side")
        return left_s.isin(list(right)).astype("boolean")

    if isinstance(op, ast.NotIn):
        if right_is_series:
            raise RuleParseError("'not in' right-hand side cannot be a column/Series")
        left_s = _ensure_series(left, df)
        if not isinstance(right, (list, tuple, set)):
            raise RuleParseError("'not in' requires a list/tuple/set on the right-hand side")
        return (~left_s.isin(list(right))).astype("boolean")

    left_v = left
    right_v = right

    if isinstance(op, ast.Eq):
        res = left_v == right_v
    elif isinstance(op, ast.NotEq):
        res = left_v != right_v
    elif isinstance(op, ast.Lt):
        res = left_v < right_v
    elif isinstance(op, ast.LtE):
        res = left_v <= right_v
    elif isinstance(op, ast.Gt):
        res = left_v > right_v
    elif isinstance(op, ast.GtE):
        res = left_v >= right_v
    else:
        raise RuleParseError(f"Unsupported comparison operator: {type(op).__name__}")

    if isinstance(res, pd.Series):
        return res.astype("boolean")
    return pd.Series([bool(res)] * len(df), index=df.index, dtype="boolean")
