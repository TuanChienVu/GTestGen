"""Bind each scenario step to the step definitions whose pattern matches it."""
from __future__ import annotations

import re
from collections import defaultdict

from .patterns import compile_pattern

_FIRST_WORD = re.compile(r"^[A-Za-z]+")
_LEADING_WORD = re.compile(r"^([A-Za-z]+)(?=\s|$)")


def _index_key(stepdef) -> str | None:
    """Plain first word every matching step must start with, used to prune candidates; None means 'check against every step'."""
    pattern = stepdef.pattern
    if stepdef.kind == "regex" and not pattern.startswith("^"):
        return None  # searched, not fullmatched: may match anywhere in the step text
    if stepdef.kind in ("regex", "regex_full"):
        if "|" in pattern:
            return None  # an alternation can let another branch start with a different word
        pattern = pattern.removeprefix("^")
    m = _LEADING_WORD.match(pattern)  # a word followed by '(', '/', '?', '\\' ... is not a plain literal
    return m.group(1).lower() if m else None


class Library:
    def __init__(self, defs):
        self.uncompilable = 0
        self._by_word = defaultdict(list)
        self._wild = []
        for d in defs:
            compiled = compile_pattern(d.kind, d.pattern)
            if compiled is None:
                self.uncompilable += 1
                continue
            key = _index_key(d)
            (self._by_word[key] if key else self._wild).append((d, *compiled))

    def matches(self, text: str):
        word = _FIRST_WORD.match(text)
        candidates = self._wild + (self._by_word.get(word.group(0).lower(), []) if word else [])
        return [d for d, rx, mode in candidates if (rx.fullmatch(text) if mode == "fullmatch" else rx.search(text))]


def bind_scenario(scenario, library: Library) -> dict:
    targets, unbound, ambiguous = set(), [], 0
    for step in scenario.steps:
        found = library.matches(step.text)
        if not found:
            unbound.append(step.text)
        ambiguous += len(found) > 1
        targets.update(f"{d.path}:{d.line}" for d in found)
    status = "UNBOUND" if len(unbound) == len(scenario.steps) else ("PARTIAL" if unbound else "BOUND")
    return {"status": status, "ambiguous_steps": ambiguous, "target_ids": sorted(targets), "unbound_steps": unbound}
