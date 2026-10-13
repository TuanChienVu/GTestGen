"""Minimal, deterministic Gherkin reader: scenarios and their steps, English keywords only."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEAD = re.compile(r"^\s*(Feature|Background|Scenario Outline|Scenario Template|Scenario|Example|Examples|Scenarios|Rule)\s*:\s*(.*)$")
_STEP = re.compile(r"^\s*(Given|When|Then|And|But|\*)\s+(.*\S)\s*$")
_PARAM = re.compile(r"<([^<>]+)>")


@dataclass(frozen=True)
class Step:
    keyword: str
    text: str
    line: int
    from_background: bool = False


@dataclass
class Scenario:
    name: str
    line: int
    steps: list[Step] = field(default_factory=list)
    is_outline: bool = False
    expanded: bool = False
    tags: list[str] = field(default_factory=list)
    examples_tags: list[list[str]] = field(default_factory=list)


def parse_feature(text: str) -> list[Scenario]:
    background: list[Step] = []
    scenarios: list[Scenario] = []
    section: str | None = None
    current: Scenario | None = None
    last_keyword = "Given"
    header: list[str] | None = None
    fence: str | None = None
    pending_tags: list[str] = []
    feature_tags: list[str] = []
    rule_tags: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if fence:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith('"""') or stripped.startswith("`" * 3):
            fence = stripped[:3]
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("@"):
            for token in stripped.split():
                if token.startswith("#"):
                    break  # '#' starts a same-line comment: it and everything after it is not a tag
                if token.startswith("@"):
                    pending_tags.append(token)
            continue
        head = _HEAD.match(raw)
        if head:
            kind = head.group(1)
            if kind == "Feature":
                feature_tags, pending_tags = pending_tags, []
                section, current = None, None
            elif kind == "Background":
                pending_tags = []
                section, current = "background", None
            elif kind in ("Scenario", "Scenario Outline", "Scenario Template", "Example"):
                own_tags, pending_tags = pending_tags, []
                current = Scenario(name=head.group(2).strip(), line=number, steps=list(background),
                                   is_outline=kind in ("Scenario Outline", "Scenario Template"),
                                   tags=feature_tags + rule_tags + own_tags)
                scenarios.append(current)
                section, header = "scenario", None
            elif kind in ("Examples", "Scenarios"):
                own_tags, pending_tags = pending_tags, []
                if current is not None:
                    current.examples_tags.append(own_tags)
                    if len(current.examples_tags) == 1:
                        current.tags = current.tags + own_tags
                section, header = "examples", None
            elif kind == "Rule":
                rule_tags, pending_tags = pending_tags, []
                section, current = None, None
            else:
                pending_tags = []
                section, current = None, None
            last_keyword = "Given"
            continue
        if stripped.startswith("|"):
            if section == "examples" and current is not None:
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                if header is None:
                    header = cells
                elif current.is_outline and not current.expanded:
                    row = dict(zip(header, cells))
                    current.steps = [
                        Step(s.keyword, _PARAM.sub(lambda m: row.get(m.group(1), m.group(0)), s.text), s.line, s.from_background)
                        for s in current.steps
                    ]
                    current.expanded = True
            continue
        step = _STEP.match(raw)
        if step and section in ("background", "scenario"):
            keyword = step.group(1)
            keyword = last_keyword if keyword in ("And", "But", "*") else keyword
            last_keyword = keyword
            if section == "background":
                background.append(Step(keyword, step.group(2), number, True))
            elif current is not None:
                current.steps.append(Step(keyword, step.group(2), number, False))
    return scenarios
