"""Verbatim Gherkin text of one scenario (with its tags, tables, doc strings and Examples) and of its Background."""
from __future__ import annotations

import re

_SCENARIO = re.compile(r"^\s*(Scenario Outline|Scenario Template|Scenario|Example)\s*:")
_HEADER = re.compile(r"^\s*(Scenario Outline|Scenario Template|Scenario|Example|Rule|Background|Feature)\s*:")
_RULE = re.compile(r"^\s*Rule\s*:")
_BACKGROUND = re.compile(r"^\s*Background\s*:")
_FENCE = re.compile(r'^\s*("""|`{3})')  # `{3}: a literal triple backtick would end the plan's code fence


def _dedent(lines: list[str]) -> str:
    lines = list(lines)
    while lines and not lines[-1].strip():
        lines.pop()
    width = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[width:].rstrip() for l in lines)


def _block(lines: list[str], start: int) -> list[str]:
    """Lines from `start` (0-based) up to the next header, ignoring header-like lines inside doc strings. Trailing
    tag, comment and blank lines belong to the NEXT block and are dropped."""
    out, fence = [lines[start]], None
    for line in lines[start + 1:]:
        m = _FENCE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        elif fence is None and _HEADER.match(line):
            break
        out.append(line)
    while len(out) > 1 and (not out[-1].strip() or out[-1].strip()[0] in "@#"):
        out.pop()
    return out


def _background_between(lines: list[str], start: int, stop: int) -> str:
    """The Background header between `start` and `stop` that precedes every scenario in that range, if any."""
    for j in range(start, stop):
        if _BACKGROUND.match(lines[j]):
            return _dedent(_block(lines, j))
        if _SCENARIO.match(lines[j]) or (j > start and _RULE.match(lines[j])):
            return ""
    return ""


def scenario_block(text: str, line: int) -> dict:
    """`line` is the 1-based line of the scenario keyword (Phase 1 `scenario_line`). Returns
    {"background": str, "scenario": str}. `scenario` starts with the scenario's own tag lines. `background` joins
    the feature-level Background and the enclosing Rule's Background (Gherkin runs both, in that order); it is ""
    when there is none."""
    lines = text.splitlines()
    idx = line - 1
    if not (0 <= idx < len(lines)) or not _SCENARIO.match(lines[idx]):
        raise ValueError(f"line {line} is not a scenario header")
    first = idx
    while first > 0 and lines[first - 1].strip().startswith("@"):
        first -= 1
    rule = next((j for j in range(idx - 1, -1, -1) if _RULE.match(lines[j])), None)
    parts = [_background_between(lines, 0, idx if rule is None else rule)]
    if rule is not None:
        parts.append(_background_between(lines, rule, idx))
    return {"background": "\n\n".join(p for p in parts if p), "scenario": _dedent(lines[first:idx] + _block(lines, idx))}
