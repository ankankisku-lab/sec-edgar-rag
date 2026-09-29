"""Numerical safeguards for generated answers.

1. <calc> -- the LLM never does arithmetic. It writes the calculation, e.g.
   <calc>(28202 - 25352) / 25352 * 100</calc>, and Python evaluates it (a small
   AST evaluator: numbers and + - * / ( ) only, no eval()).
2. verify -- every number left in the answer must appear in the sources, be a
   <calc> result, or be a millions->billions rounding of a source number;
   anything else is reported as unverified (a numerical-hallucination signal).
"""
from __future__ import annotations

import ast
import operator
import re

CALC_RE = re.compile(r"<calc>(.*?)</calc>", re.S)
NUM_RE = re.compile(r"(?<![A-Za-z\d])\$?\(?-?\d[\d,]*(?:\.\d+)?\)?")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        raise ValueError(f"unsupported expression: {expr!r}")
    # The prompt asks for plain numbers with negatives as -171 (not accounting "(171)",
    # which would silently parse as a positive parenthesised 171).
    cleaned = expr.replace(",", "").replace("$", "").replace("%", "").strip()
    return ev(ast.parse(cleaned, mode="eval"))


def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".") if abs(x) >= 1 else f"{x:.4f}".rstrip("0")


def apply_calcs(answer: str) -> tuple[str, list[dict]]:
    """Replace each <calc>expr</calc> with its value; return the log of calculations."""
    log: list[dict] = []

    def sub(m: re.Match) -> str:
        expr = m.group(1)
        try:
            value = safe_eval(expr)
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            log.append({"expression": expr, "error": str(e)})
            return f"[calculation failed: {expr}]"
        text = format_number(value)
        log.append({"expression": expr, "value": value, "text": text})
        return text

    return CALC_RE.sub(sub, answer), log


def _value(token: str) -> float | None:
    t = token.replace("$", "").replace(",", "")
    neg = t.startswith("(") and t.endswith(")")
    try:
        v = float(t.strip("()"))
    except ValueError:
        return None
    return -v if neg else v


def number_spans(answer: str, keep_percent: bool = False) -> list[tuple[int, int, str, float]]:
    """(start, end, token, absolute value) of the numbers that verification checks: citation
    markers, period labels, years and day-of-month numbers are not claims about the filing's figures.

    The day-of-month rule also skips whole percentages up to 31 ("11%"); keep_percent=True keeps
    them (the Phase 17 audit and the citation view use it)."""
    blank = lambda m: " " * len(m.group())  # noqa: E731 -- keep offsets aligned with the answer
    text = re.sub(r"\[\d+\](?:\[\d+\])*", blank, answer)          # citation markers
    text = re.sub(r"\b(?:Q[1-4]|FY)\s?\d{2,4}\b", blank, text)     # period labels
    out = []
    for m in NUM_RE.finditer(text):
        token, v = m.group(), _value(m.group())
        if v is None:
            continue
        v = abs(v)
        percent = keep_percent and text[m.end():m.end() + 1] == "%"
        # Years ("2025") and day-of-month numbers are skipped; "$2,002" or "2,002 million" is an
        # amount, not a year, so a $ sign or thousands comma always keeps the number.
        amount = "$" in token or re.search(r"\d,\d", token) is not None   # not "June 28," / "2025,"
        if v.is_integer() and not amount and (1900 <= v <= 2100 or v <= 31) and not percent:
            continue
        out.append((m.start(), m.end(), token, v))
    return out


def number_tokens(answer: str, keep_percent: bool = False) -> list[tuple[str, float]]:
    """number_spans() without the positions."""
    return [(token, v) for _, _, token, v in number_spans(answer, keep_percent)]


def verify_numbers(answer: str, sources: str, question: str = "", calcs: list[dict] | None = None) -> list[str]:
    """Numbers in the answer that no source (or calculation) supports."""
    allowed = {abs(v) for v in (_value(t) for t in NUM_RE.findall(sources + " " + question)) if v is not None}
    allowed |= {abs(c["value"]) for c in (calcs or []) if "value" in c}
    unsupported = []
    for token, v in number_tokens(answer):
        if v in allowed:
            continue
        # "$28.2 billion" from a table in millions (28,202); allow 1-2 decimal rounding
        if any(round(a / 1000, d) == v for a in allowed if a >= 1000 for d in (1, 2)):
            continue
        # rounded calculation results, e.g. 11.24 -> "11.2%"
        if any(round(a, d) == v for a in allowed for d in (0, 1, 2)):
            continue
        unsupported.append(token)
    return unsupported
