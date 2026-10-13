"""Step typing for ER / ER-trace: the effective keyword of every step and the index sets the runner needs.

Indices count Background steps first, then Scenario steps, from 0, in the order Behat executes them; this is also the
order of `record["steps"]` and `record["step_kinds"]` (Phase 2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STEP_LINE = re.compile(r"\s*(Given|When|Then|And|But|\*)\s+\S")
DOC_FENCE = re.compile(r'\s*("""|`{3})')


def effective_keywords(background: str | None, scenario: str) -> list[str]:
    """Given/When/Then for each step; And/But/* inherit from the previous step of the same block. Doc strings skipped."""
    out: list[str] = []
    for block in (background or "", scenario):
        last = None
        in_doc = False
        for line in block.split("\n"):
            if DOC_FENCE.match(line):
                in_doc = not in_doc
                continue
            if in_doc:
                continue
            m = STEP_LINE.match(line)
            if not m:
                continue
            kw = m.group(1)
            if kw in ("And", "But", "*"):
                if last is None:
                    raise ValueError(f"step without a leading Given/When/Then: {line.strip()!r}")
                kw = last
            last = kw
            out.append(kw)
    return out


@dataclass(frozen=True)
class StepPlan:
    types: tuple[str, ...]        # effective keyword per step
    act_create: tuple[int, ...]   # CREATE steps that act (Given/When): no-op'd in the M1 mutation
    then_create: tuple[int, ...]  # CREATE Then steps: checked for sensitivity
    actions: tuple[int, ...]      # every non-Then step: the scenario trace is taken over these


def step_plan(record: dict) -> StepPlan:
    types = effective_keywords(record.get("background"), record["scenario"])
    kinds = record["step_kinds"]
    if not (len(types) == len(kinds) == len(record["steps"])):
        raise ValueError(f"{record['id']}: {len(types)} Gherkin steps, {len(kinds)} step_kinds, {len(record['steps'])} steps")
    return StepPlan(
        types=tuple(types),
        act_create=tuple(i for i, (t, k) in enumerate(zip(types, kinds)) if k == "create" and t != "Then"),
        then_create=tuple(i for i, (t, k) in enumerate(zip(types, kinds)) if k == "create" and t == "Then"),
        actions=tuple(i for i, t in enumerate(types) if t != "Then"),
    )
