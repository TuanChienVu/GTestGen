"""ER-trace scoring (descriptive metric, declared before any output): scenario action trace at level base_status,
multiset, mutation M1 (no-op CREATE action steps), Then sensitivity "any".

A trace is the JSON written by the harness Recorder: {"steps": [{i, type, text, line, status, error?}],
"events": [{kind, step, ...}]}. Events outside steps (hooks) have step == -1 and are ignored. http events carry the
matched `route` and the values of token-named route parameters (`token_values`), recorded at kernel.response.

Known limit: token numbering ({tokK}, by first appearance) and {n} for numeric ids drop entity identity, so a run that
swaps two carts (or two ids) gives the same trace. On the measured records an action trace holds at most one
token.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .steps import StepPlan

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
STEP_STATUSES = {"passed", "failed", "undefined", "pending", "skipped", "noop"}


class TraceError(ValueError):
    pass


def _is(v, t) -> bool:
    return isinstance(v, t) and not (t is int and isinstance(v, bool))


def validate_trace(trace) -> None:
    """Raise TraceError unless `trace` has the Recorder's shape (it is written from inside the container)."""
    if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list) or not isinstance(trace.get("events"), list):
        raise TraceError("trace is not {steps: [], events: []}")
    for pos, s in enumerate(trace["steps"]):
        if not (isinstance(s, dict) and _is(s.get("i"), int) and s["i"] == pos and _is(s.get("text"), str)
                and _is(s.get("type"), str) and s.get("status") in STEP_STATUSES):
            raise TraceError(f"bad step {str(s)[:80]}")
    for e in trace["events"]:
        if not (isinstance(e, dict) and _is(e.get("kind"), str) and _is(e.get("step"), int)):
            raise TraceError(f"bad event {str(e)[:80]}")
        if e["kind"] == "http":
            ok = (_is(e.get("method"), str) and _is(e.get("path"), str) and isinstance(e.get("body_keys"), list)
                  and all(_is(k, str) for k in e["body_keys"]) and (e.get("status") is None or _is(e["status"], int))
                  and (e.get("route") is None or _is(e["route"], str))
                  and ("token_values" not in e  # absent when the request never reached kernel.response
                       or (isinstance(e["token_values"], list) and all(_is(v, str) for v in e["token_values"]))))
        elif e["kind"] == "db":
            ok = _is(e.get("op"), str) and _is(e.get("entity"), str)
        else:
            ok = True
        if not ok:
            raise TraceError(f"bad {e['kind']} event {str(e)[:80]}")


def norm_path(path: str, tokens: dict[str, str], token_values: set[str], routed: bool) -> str:
    """Token route parameters -> {tokK} (numbered by first appearance); ids -> {n}; UUIDs -> {uuid}. Codes and slugs
    are kept. A request that matched no route falls back to: the segment after "orders" is a token."""
    out, prev = [], ""
    for seg in path.split("/"):
        orig = seg
        if seg and (seg in token_values or (not routed and prev.lower() == "orders")):
            seg = tokens.setdefault(seg, f"{{tok{len(tokens) + 1}}}")
        elif seg.isdigit():
            seg = "{n}"
        elif UUID.match(seg):
            seg = "{uuid}"
        out.append(seg)
        prev = orig
    return "/".join(out)


def norm_event(event: dict, tokens: dict[str, str], token_values: set[str]):
    """Level base_status: http (method, path, top-level body keys, status); db (op, entity class). Others ignored."""
    if event["kind"] == "http":
        path = norm_path(event["path"], tokens, token_values, event.get("route") is not None)
        return ("http", event["method"], path, tuple(sorted(event["body_keys"])), event.get("status"))
    if event["kind"] == "db":
        return ("db", event["op"], event["entity"])
    return None


def action_trace(trace: dict, actions: tuple[int, ...]) -> Counter:
    """Multiset of normalised events emitted while an action (non-Then) step ran; tokens numbered over these only."""
    keep = set(actions)
    kept = [e for e in trace["events"] if e["step"] in keep]
    token_values = {v for e in kept if e["kind"] == "http" for v in e.get("token_values") or []}
    tokens: dict[str, str] = {}
    out: Counter = Counter()
    for event in kept:
        norm = norm_event(event, tokens, token_values)
        if norm is not None:
            out[norm] += 1
    return out


def statuses(trace: dict) -> list[str]:
    return [s.get("status", "missing") for s in trace["steps"]]


def status_from_steps(steps: list[str]) -> str:
    """Strict-Behat status of a soft run: the first non-passed step decides; undefined anywhere wins over pending."""
    first = next((s for s in steps if s != "passed"), None)
    if first is None:
        return "PASS"
    if first == "failed":
        return "FAIL"
    return "UNDEFINED" if "undefined" in steps else "FAIL"


def aligned(trace: dict, record: dict, plan: StepPlan) -> bool:
    """The run executed exactly the record's steps, in order, with the same effective keywords."""
    got = [(" ".join(s["text"].split()), s["type"]) for s in trace["steps"]]
    want = [(" ".join(s.split()), t) for s, t in zip(record["steps"], plan.types)]
    return got == want


@dataclass(frozen=True)
class GoldRef:
    action_trace: Counter
    act_evaluable: bool                 # the CREATE action steps change the scenario trace
    then_evaluable: tuple[int, ...]     # CREATE Then steps that pass on gold and fail on gold under M1
    gold_pass: bool
    m1_action_trace: Counter | None = None  # action trace of the gold M1 run, when there is one


def gold_reference(plan: StepPlan, gold: dict, gold_m1: dict | None) -> GoldRef:
    """gold_m1 is None when the record has no CREATE action step (M1 would be identical to the normal run)."""
    g = statuses(gold)
    ref_trace = action_trace(gold, plan.actions)
    if gold_m1 is None:
        return GoldRef(ref_trace, False, (), all(s == "passed" for s in g))
    gm = statuses(gold_m1)
    m1_trace = action_trace(gold_m1, plan.actions)
    return GoldRef(
        action_trace=ref_trace,
        act_evaluable=ref_trace != m1_trace,
        then_evaluable=tuple(i for i in plan.then_create if g[i] == "passed" and gm[i] == "failed"),
        gold_pass=all(s == "passed" for s in g),
        m1_action_trace=m1_trace,
    )


def evaluability(ref: GoldRef) -> dict:
    return {"act_evaluable": ref.act_evaluable, "n_then_evaluable": len(ref.then_evaluable)}


def score_missing(ref: GoldRef) -> dict:
    """An output without a usable trace (ERROR, TIMEOUT, OOM, SKIP): 0 on every part the record can evaluate."""
    ta = 0 if ref.act_evaluable else None
    tt = 0 if ref.then_evaluable else None
    return {"trace_action": ta, "trace_then": tt, "er_trace": None if ta is None and tt is None else 0,
            **evaluability(ref)}


def score_output(plan: StepPlan, ref: GoldRef, normal: dict, mutated: dict | None) -> dict:
    """trace_action / trace_then / er_trace in {1, 0, None}; None = not evaluable for this record.
    mutated=None with evaluable Then steps means the output's M1 run did not complete: trace_then = 0."""
    st = statuses(normal)
    passed = all(s == "passed" for s in st)
    trace_action = None
    if ref.act_evaluable:
        trace_action = int(action_trace(normal, plan.actions) == ref.action_trace)
    trace_then = None
    if ref.then_evaluable:
        if mutated is None:
            trace_then = 0
        else:
            mt = statuses(mutated)
            trace_then = int(all(st[i] == "passed" and mt[i] == "failed" for i in ref.then_evaluable))
    if trace_action is None and trace_then is None:
        er_trace = None
    else:
        er_trace = int(passed and trace_action in (1, None) and trace_then in (1, None))
    return {"trace_action": trace_action, "trace_then": trace_then, "er_trace": er_trace, **evaluability(ref)}
