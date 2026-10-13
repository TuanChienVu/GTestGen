from collections import Counter

import pytest

from er.steps import StepPlan
from er.trace import (GoldRef, TraceError, action_trace, aligned, gold_reference, norm_path, score_missing,
                      score_output, status_from_steps, validate_trace)


def trace(statuses, events, types=("Given", "When", "When", "Then")):
    return {"steps": [{"i": i, "type": types[i], "text": f"s{i}", "status": s} for i, s in enumerate(statuses)],
            "events": events}


def http(step, path, keys=(), status=200, method="POST", route="r", tokens=()):
    return {"kind": "http", "step": step, "method": method, "path": path, "body_keys": list(keys), "status": status,
            "route": route, "token_values": list(tokens)}


def db(step, op="insert", entity="Zone"):
    return {"kind": "db", "step": step, "op": op, "entity": entity}


# steps: 0 Given reuse, 1 When create (builder), 2 When reuse (sends), 3 Then create
PLAN = StepPlan(types=("Given", "When", "When", "Then"), act_create=(1,), then_create=(3,), actions=(0, 1, 2))


def test_route_tokens_are_replaced_but_codes_and_slugs_are_kept():
    tokens = {}
    got = norm_path("/api/v2/shop/orders/qXwNMO6kkG/items/12", tokens, {"qXwNMO6kkG"}, routed=True)
    assert got == "/api/v2/shop/orders/{tok1}/items/{n}"
    assert norm_path("/api/v2/admin/channels/Mixed_Code_2", tokens, set(), routed=True) == "/api/v2/admin/channels/Mixed_Code_2"
    assert norm_path("/api/v2/admin/products/php-t-shirt", tokens, set(), routed=True) == "/api/v2/admin/products/php-t-shirt"


def test_unrouted_request_falls_back_to_the_segment_after_orders_in_any_case():
    tokens = {}
    assert norm_path("/api/v2/shop/orders/abcdefghij", tokens, set(), routed=False) == "/api/v2/shop/orders/{tok1}"
    assert norm_path("/api/v2/shop/ORDERS/zzz", tokens, set(), routed=False) == "/api/v2/shop/ORDERS/{tok2}"


def test_tokens_are_numbered_over_kept_action_events_only():
    then_first = trace(["passed"] * 4, [http(3, "/o/AAAA1111bb", method="GET", tokens=["AAAA1111bb"]),
                                        http(2, "/o/BBBB2222cc", ["x"], tokens=["BBBB2222cc"])])
    plain = trace(["passed"] * 4, [http(2, "/o/CCCC3333dd", ["x"], tokens=["CCCC3333dd"])])
    assert action_trace(then_first, PLAN.actions) == action_trace(plain, PLAN.actions)
    assert ("http", "POST", "/o/{tok1}", ("x",), 200) in action_trace(plain, PLAN.actions)


def test_action_trace_ignores_then_hooks_and_storage():
    t = trace(["passed"] * 4, [db(-1, "delete"), db(0), {"kind": "storage", "step": 0, "op": "set", "key": "k"},
                               http(2, "/z", ["code"]), http(3, "/z", method="GET")])
    assert action_trace(t, PLAN.actions) == {("db", "insert", "Zone"): 1, ("http", "POST", "/z", ("code",), 200): 1}


def test_two_runs_with_different_order_tokens_have_equal_traces():
    a = trace(["passed"] * 4, [http(2, "/api/v2/shop/orders/qXwNMO6kkG", ["x"], tokens=["qXwNMO6kkG"])])
    b = trace(["passed"] * 4, [http(2, "/api/v2/shop/orders/Rb4J~FZXZ7", ["x"], tokens=["Rb4J~FZXZ7"])])
    assert action_trace(a, PLAN.actions) == action_trace(b, PLAN.actions)


def test_status_from_steps():
    assert status_from_steps(["passed", "passed"]) == "PASS"
    assert status_from_steps(["passed", "undefined", "failed"]) == "UNDEFINED"
    assert status_from_steps(["passed", "failed", "undefined"]) == "FAIL"
    assert status_from_steps(["pending", "passed"]) == "FAIL"


def test_validate_trace_rejects_malformed_shapes():
    validate_trace(trace(["passed"] * 4, [db(0), http(2, "/z", ["k"])]))
    for bad in ([], {"steps": [{"i": 0, "type": "Given", "text": "a", "status": "won"}], "events": []},
                {"steps": [], "events": [{"kind": "http", "step": 0, "method": "GET", "path": 5, "body_keys": []}]},
                {"steps": [], "events": [{"kind": "db", "step": "0", "op": "insert", "entity": "Zone"}]}):
        with pytest.raises(TraceError):
            validate_trace(bad)


def test_validate_trace_token_values_must_be_a_list_of_str_when_present():
    no_response = http(2, "/z", ["k"])
    del no_response["token_values"]  # the request never reached kernel.response
    validate_trace(trace(["passed"] * 4, [no_response, http(2, "/o/AB12", tokens=["AB12"])]))
    for tokens in ("AB12", {"tokenValue": "AB12"}, None, [1], ["AB12", None]):
        e = http(2, "/o/AB12")
        e["token_values"] = tokens
        with pytest.raises(TraceError, match="bad http event"):
            validate_trace(trace(["passed"] * 4, [e]))


def test_validate_trace_step_index_must_be_its_position():
    for ids in ((1, 0, 2, 3), (0, 0, 2, 3), (1, 2, 3, 4), (0, 1, 2, 4)):
        t = trace(["passed"] * 4, [])
        for s, i in zip(t["steps"], ids):
            s["i"] = i
        with pytest.raises(TraceError, match="bad step"):
            validate_trace(t)


GOLD = trace(["passed"] * 4, [db(0), http(2, "/z", ["code", "members"], 422)])
GOLD_M1 = trace(["passed", "noop", "passed", "failed"], [db(0), http(2, "/z", ["code"], 201)])


def test_gold_reference_evaluability():
    ref = gold_reference(PLAN, GOLD, GOLD_M1)
    assert ref.act_evaluable and ref.then_evaluable == (3,) and ref.gold_pass


def test_gold_reference_keeps_the_gold_m1_action_trace():
    ref = gold_reference(PLAN, GOLD, GOLD_M1)
    assert ref.m1_action_trace == action_trace(GOLD_M1, PLAN.actions) == {("db", "insert", "Zone"): 1,
                                                                          ("http", "POST", "/z", ("code",), 201): 1}
    assert ref.m1_action_trace != ref.action_trace
    assert gold_reference(PLAN, GOLD, None).m1_action_trace is None
    assert GoldRef(Counter(), False, (), True).m1_action_trace is None  # last field, default None


def test_score_gold_stub_and_half_gold():
    ref = gold_reference(PLAN, GOLD, GOLD_M1)
    assert score_output(PLAN, ref, GOLD, GOLD_M1) == {"trace_action": 1, "trace_then": 1, "er_trace": 1,
                                                      "act_evaluable": True, "n_then_evaluable": 1}
    stub = trace(["passed"] * 4, [db(0), http(2, "/z", ["code"], 201)])
    stub_m1 = trace(["passed", "noop", "passed", "passed"], [db(0), http(2, "/z", ["code"], 201)])
    s = score_output(PLAN, ref, stub, stub_m1)
    assert (s["trace_action"], s["trace_then"], s["er_trace"]) == (0, 0, 0)
    half = trace(["passed"] * 4, GOLD["events"])  # correct actions, Then that never fails
    h = score_output(PLAN, ref, half, stub_m1)
    assert (h["trace_action"], h["trace_then"], h["er_trace"]) == (1, 0, 0)


def test_failed_m1_and_missing_trace_score_zero_not_none():
    ref = gold_reference(PLAN, GOLD, GOLD_M1)
    s = score_output(PLAN, ref, GOLD, None)  # the output's M1 run did not complete
    assert (s["trace_action"], s["trace_then"], s["er_trace"]) == (1, 0, 0)
    assert score_missing(ref) == {"trace_action": 0, "trace_then": 0, "er_trace": 0, "act_evaluable": True,
                                  "n_then_evaluable": 1}


def test_not_evaluable_is_none():
    plan = StepPlan(types=("Given", "Then"), act_create=(), then_create=(1,), actions=(0,))
    ref = gold_reference(plan, trace(["passed"] * 2, [], ("Given", "Then")), None)
    assert score_output(plan, ref, trace(["passed"] * 2, [], ("Given", "Then")), None)["er_trace"] is None
    assert score_missing(ref)["er_trace"] is None


def test_aligned_compares_step_texts_and_types():
    rec = {"steps": ["s0", "s1", "s2", "s3"]}
    assert aligned(trace(["passed"] * 4, []), rec, PLAN)
    assert not aligned(trace(["passed"] * 3, []), rec, PLAN)
    assert not aligned(trace(["passed"] * 4, [], ("Given", "When", "Then", "Then")), rec, PLAN)
