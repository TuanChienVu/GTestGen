"""Design §2.3b, before the SAP lock: hand-check 30 cases that PASS ER but fail the trace, on the dev-round outputs, to
measure ER-trace's false negatives. This tool draws the cases by a rule fixed before any new output exists (06/10)
and writes a worksheet with each case's evidence; the verdicts are filled by hand. It reads out/ and writes only --out (a
new directory). It never imports the scoring code.

The rule (see tools/ER_TRACE_FN_SAMPLE.md):
- Population P = the union, over every dev ER run named with --source (the old dev_pilot_v0, dev_pilot_v0_0482a76 and the
  grid's <label>_0482a76 runs that exist before the lock), of the rows of a model condition (not neg/gold/stub) with
  er == 1, act_evaluable true and trace_action == 0 or trace_then == 0. A null trace is "not evaluable": counted apart,
  never drawn. A row whose tamper verdict is "cheat" is out (its ER is 0).
- |P| <= N (30): every case is checked. Otherwise the N cases with the smallest
  sha256("2.3b|<run>|<record_id>|<condition>").
- Kind of a case: "action" (trace_action 0) or "then" (trace_action 1, trace_then 0).
- No ER run is added only to reach N.

usage: er_trace_fn_sample.py --controls-er <out/phase3/er/controls> --out NEWDIR [--n 30]
                             --source RUN <out/phase3/er/RUN> <out/phase3/overlays/RUN/manifest.jsonl>  (repeat)
       er_trace_fn_sample.py --selftest"""
import argparse
import collections
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ER_CONTROLS = ("neg", "gold", "stub")


def rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x]


def trace(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def ev_key(e: dict) -> str:
    return json.dumps({k: v for k, v in e.items() if k != "step"}, sort_keys=True, ensure_ascii=False)


def multiset_diff(a, b) -> dict:
    """Events of trace b missing from a and extra in b (multisets, step index ignored), as the harness compares."""
    if a is None or b is None:
        return {"note": "trace missing"}
    ca, cb = collections.Counter(map(ev_key, a["events"])), collections.Counter(map(ev_key, b["events"]))
    return {"missing": sorted((ca - cb).elements())[:8], "extra": sorted((cb - ca).elements())[:8],
            "n_missing": sum((ca - cb).values()), "n_extra": sum((cb - ca).values())}


def then_status(t) -> list[str]:
    return [] if t is None else [f"{s['i']}:{s['status']}" for s in t["steps"] if s["type"] == "Then"]


def in_population(r: dict) -> bool:
    return (r["condition"] not in ER_CONTROLS and r.get("er") == 1 and bool(r.get("act_evaluable"))
            and r.get("tamper_verdict") != "cheat" and (r.get("trace_action") == 0 or r.get("trace_then") == 0))


def sort_key(run: str, r: dict) -> str:
    return hashlib.sha256(f"2.3b|{run}|{r['record_id']}|{r['condition']}".encode()).hexdigest()


def draw(sources: list[tuple[str, Path, Path]], controls_er: Path, n: int, out: Path) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists: give a new --out")
    if len({s[0] for s in sources}) != len(sources):
        raise SystemExit("a run is named twice")
    pop, not_eval, slugs = [], collections.Counter(), {}
    for run, er_dir, manifest in sources:
        slugs[run] = {r["record_id"]: r["base"].split("/")[1] for r in rows(manifest)}  # a skipped row has dir None
        for r in rows(er_dir / "er_results.jsonl"):
            if r["condition"] in ER_CONTROLS:
                continue
            if in_population(r):
                pop.append((run, er_dir, r))
            elif r.get("er") == 1 and r.get("trace_action") is None:
                not_eval[run] += 1
    sample = sorted(pop, key=lambda x: sort_key(x[0], x[2]))[:n]
    out.mkdir(parents=True)
    cases = []
    for i, (run, er_dir, r) in enumerate(sample, 1):
        s, c = slugs[run][r["record_id"]], r["condition"]
        g = {m: controls_er / "er_runs" / s / "gold" / m / "trace.json" for m in ("normal", "m1")}
        o = {m: er_dir / "er_runs" / s / c / m / "trace.json" for m in ("normal", "m1")}
        cases.append({"case": i, "run": run, "record_id": r["record_id"], "condition": c, "slug": s,
                      "kind": "action" if r.get("trace_action") == 0 else "then",
                      "trace_action": r.get("trace_action"), "trace_then": r.get("trace_then"),
                      "tamper_flags": r.get("tamper_flags"), "sort_key": sort_key(run, r),
                      "gold_trace": {m: str(p) for m, p in g.items()}, "output_trace": {m: str(p) for m, p in o.items()},
                      "behat_log": str(er_dir / "er_runs" / s / c / "normal" / "behat.log"),
                      "diff_normal": multiset_diff(trace(g["normal"]), trace(o["normal"])),
                      "then_m1_gold": then_status(trace(g["m1"])), "then_m1_output": then_status(trace(o["m1"]))})
    (out / "sample.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cases), encoding="utf-8")
    kind = lambda r: "action" if r.get("trace_action") == 0 else "then"
    meta = {"rule": "§2.3b, 06/10 (tools/ER_TRACE_FN_SAMPLE.md)", "n": n, "runs": [s[0] for s in sources],
            "population": len(pop), "checked": len(cases), "census": len(pop) <= n,
            "er_pass_trace_not_evaluable": dict(not_eval),
            "population_by_run_condition_kind":
                dict(sorted(collections.Counter(f"{run}|{r['condition']}|{kind(r)}" for run, _, r in pop).items())),
            "population_records": len({(run, r["record_id"]) for run, _, r in pop})}
    (out / "SAMPLE_META.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    md = ["# Kiểm tay §2.3b: PASS ER nhưng trượt vết", "",
          f"Lượt: {', '.join(meta['runs'])}. Quần thể {len(pop)} ca trên {meta['population_records']} (lượt, bản ghi); "
          f"kiểm {len(cases)} ({'toàn bộ' if meta['census'] else 'mẫu theo sha256'}). PASS ER mà vết không đánh giá được "
          f"(báo riêng, không lấy mẫu): {sum(not_eval.values())}.", "",
          "Phán quyết mỗi ca (điền tay; mỗi ca được phán quyết hai lần, và lần thứ hai được chốt trước khi xem kết quả "
          "của lần thứ nhất):",
          "- `FN`: output đúng về hành vi (cùng tác dụng với gold, hay Then kiểm đúng điều kịch bản đòi), vết trượt vì lý do "
          "không thuộc hành vi (thứ tự, request đọc thêm, chuẩn hoá, dữ liệu có sẵn…): âm tính giả của ER-trace;",
          "- `TP`: vết trượt phản ánh khác biệt thật (thiếu/khác tác dụng; Then không bắt được khi hành động bị bỏ);",
          "- `UNCLEAR`: không quyết được từ bằng chứng; ghi lý do.", "",
          "Tổng: tỷ lệ FN = FN / (số ca − UNCLEAR), KTC 95 % Wilson; báo theo lượt, điều kiện, loại (action/then) và theo "
          "bản ghi (các ca dồn cụm).", "",
          "| ca | lượt | điều kiện | loại | bản ghi | phán quyết | lý do | lần phán quyết | thời điểm |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {x['case']} | {x['run']} | {x['condition'].split('.')[2] if x['condition'].count('.') >= 2 else x['condition']} "
           f"| {x['kind']} | `{x['slug']}` |  |  |  |  |" for x in cases]
    for x in cases:
        md += ["", f"## Ca {x['case']}: `{x['slug']}` · {x['run']} · {x['condition']} · {x['kind']}", "",
               f"- record_id: `{x['record_id']}`; trace_action {x['trace_action']}, trace_then {x['trace_then']}; "
               f"tamper_flags {x['tamper_flags']}",
               f"- vết gold: `{x['gold_trace']['normal']}`, M1: `{x['gold_trace']['m1']}`",
               f"- vết output: `{x['output_trace']['normal']}`, M1: `{x['output_trace']['m1']}`; log: `{x['behat_log']}`",
               f"- khác multiset (lượt thường, gold → output): thiếu {x['diff_normal'].get('n_missing')}, "
               f"thừa {x['diff_normal'].get('n_extra')}",
               f"- Then ở M1: gold {x['then_m1_gold']}, output {x['then_m1_output']}",
               "- phán quyết: … · lý do: … · lần phán quyết: … · thời điểm: …"]
    (out / "WORKSHEET.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return meta


def selftest() -> int:
    """Synthetic runs: the rule picks exactly the PASS-but-trace-failing rows of model conditions, counts null traces
    apart, drops controls and "cheat", takes every case when |P| <= n and the n smallest keys otherwise."""
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)

        def run(name, specs):
            er = t / "er" / name
            (er).mkdir(parents=True)
            res, man = [], []
            for i, (cond, e, ta, tt, extra) in enumerate(specs):
                rid = f"Sylius/Sylius|f{i}.feature|{i}|api"
                res.append({"record_id": rid, "condition": cond, "er": e, "act_evaluable": ta is not None,
                            "trace_action": ta, "trace_then": tt, "tamper_flags": [], **extra})
                man.append({"record_id": rid, "base": f"{name}/s{i}/base", "dir": None})
            (er / "er_results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res))
            (t / f"{name}.manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in man))
            return name, er, t / f"{name}.manifest.jsonl"

        a = run("a", [("m.base.zero_shot.a", 1, 0, None, {}), ("m.s42.full_aware.a", 1, 1, 0, {}),
                      ("m.s42.ft_only.a", 1, 1, 1, {}), ("gold", 1, 0, 0, {}), ("m.s42.full_neutral.a", 1, None, None, {}),
                      ("m.s42.ft_only.a", 0, 0, 0, {}), ("m.s42.full_aware.a", 1, 0, 0, {"tamper_verdict": "cheat"})])
        b = run("b", [("m.s42.full_aware.b", 1, 1, 0, {}), ("m.s42.full_aware.b", 1, 0, 1, {})])
        meta = draw([a, b], t / "controls", 30, t / "o1")
        assert (meta["population"], meta["checked"], meta["census"]) == (4, 4, True), meta
        assert meta["er_pass_trace_not_evaluable"] == {"a": 1}, meta
        got = [json.loads(x) for x in (t / "o1" / "sample.jsonl").read_text().split("\n") if x]
        assert sorted((x["run"], x["kind"]) for x in got) == [("a", "action"), ("a", "then"), ("b", "action"), ("b", "then")]
        meta2 = draw([a, b], t / "controls", 2, t / "o2")
        got2 = [json.loads(x) for x in (t / "o2" / "sample.jsonl").read_text().split("\n") if x]
        keys = sorted(x["sort_key"] for x in got)
        assert meta2["census"] is False and [x["sort_key"] for x in got2] == keys[:2], (got2, keys)
        try:
            draw([a], t / "controls", 30, t / "o1")
        except SystemExit:
            pass
        else:
            raise AssertionError("an existing --out must be refused")
    print("selftest OK")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls-er", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--source", nargs=3, action="append", required=True, metavar=("RUN", "ER_DIR", "MANIFEST"))
    args = ap.parse_args()
    meta = draw([(r, Path(e), Path(m)) for r, e, m in args.source], Path(args.controls_er), args.n, Path(args.out))
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
