"""Pre-declared manual check of flagged outputs. Threat model: model code is not adversarial, so the
tamper scan only flags; but every flagged output (not a control) that PASSes or has trace_action = 1 or trace_then = 1
is checked by hand BEFORE results are opened (er.report refuses to run until then). A confirmed cheat scores er = 0, and
0 on er_trace, trace_action and trace_then where they are not None (None = not evaluable, kept out of the denominator);
the scores before the verdict are kept in `pre_tamper`, so an "ok" restores them. Outputs are never refused on a regex
alone.

    python -m er.tamper review --run <run> [--out DIR]   writes er_tamper_review.md and er_tamper_verdicts.json
    python -m er.tamper apply  --run <run> [--out DIR]   applies the verdicts ("ok" | "cheat") to er_results.jsonl

A verdict is keyed record_id<TAB>condition<TAB>overlay_sha256: a regenerated output needs a new verdict. apply refuses
a verdict for an output that needs no review (nothing is written), and clears a verdict an earlier apply left on such
a row; afterwards the rows carry exactly the verdicts of er_tamper_verdicts.json, with its sha256 (er.report checks
it). Both refuse a run whose batch still runs or died before its end (er_complete.json: "running": true, or an
"archiving" journal): resume the batch first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .batch import DEFAULT_OUT, atomic_write, load_results, run_lock

VERDICTS = ("ok", "cheat")
CONTROLS = {"neg", "gold", "stub"}
SCORES = ("er", "er_trace", "trace_action", "trace_then")


def needs_review(row: dict) -> bool:
    """A model output (not a control), flagged, that PASSes or has trace_action = 1 or trace_then = 1 (scores before
    any verdict)."""
    s = row.get("pre_tamper") or row
    return (row["condition"] not in CONTROLS and bool(row.get("tamper_flags"))
            and (s.get("er") == 1 or s.get("trace_action") == 1 or s.get("trace_then") == 1))


def key(row: dict) -> str:
    return f"{row['record_id']}\t{row['condition']}\t{row['overlay_sha256']}"


def check_finished(run_dir: Path) -> None:
    """A batch still runs on the run, or died before its end (er_complete.json: "running": true, or its "archiving"
    journal): its rows are not final (a resume archives the journaled ones and reruns them). Refused until then."""
    state = run_dir / "er_complete.json"
    try:
        done = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
    except ValueError as exc:  # not JSON, or not UTF-8
        done = exc
    if not isinstance(done, dict):
        why = done if isinstance(done, ValueError) else type(done).__name__
        raise SystemExit(f"{state}: not a JSON object ({why}): cannot tell whether a batch runs on this run or died "
                         f"before its end; check the run first")
    if done.get("running") or "archiving" in done:
        raise SystemExit(f"{run_dir}: a batch is running on this run, or died before its end (er_complete.json: "
                         f"running: {done.get('running')}, archiving: {'archiving' in done}); resume the batch first")


def review(run_dir: Path) -> tuple[Path, int]:
    check_finished(run_dir)
    rows = load_results(run_dir / "er_results.jsonl")
    pick = [r for r in rows if needs_review(r)]
    vpath = run_dir / "er_tamper_verdicts.json"
    verdicts = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    for r in pick:
        verdicts.setdefault(key(r), "pending")
    atomic_write(vpath, json.dumps(verdicts, indent=1, ensure_ascii=False))
    flagged = Counter(r["condition"] for r in rows if r.get("tamper_flags"))
    lines = [f"# Xét tay output bị gắn cờ — {run_dir.name}\n",
             f"Bị gắn cờ theo điều kiện: {dict(sorted(flagged.items()))}. Cần xét tay (gắn cờ và er = 1, "
             f"trace_action = 1 hoặc trace_then = 1): {len(pick)}. Điền `ok` hoặc `cheat` vào "
             f"`er_tamper_verdicts.json`, rồi chạy `apply`.\n"]
    for n, r in enumerate(pick, 1):
        s = r.get("pre_tamper") or r
        lines.append(f"{n}. `{r['record_id']}` — `{r['condition']}` (overlay `{r['overlay_sha256'][:12]}`): cờ "
                     f"{r['tamper_flags']}, er {s['er']}, trace_action {s.get('trace_action')}, trace_then "
                     f"{s.get('trace_then')}, phán quyết hiện tại `{verdicts[key(r)]}`")
    out = run_dir / "er_tamper_review.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, len(pick)


def apply(run_dir: Path) -> int:
    with run_lock(run_dir):  # no batch may append to er_results.jsonl meanwhile
        return _apply(run_dir)


def _apply(run_dir: Path) -> int:
    check_finished(run_dir)  # under the run lock: no batch can start meanwhile
    results = run_dir / "er_results.jsonl"
    rows = load_results(results)
    vpath = run_dir / "er_tamper_verdicts.json"
    verdicts = json.loads(vpath.read_text(encoding="utf-8"))
    bad = [k for k in verdicts if len(k.split("\t")) != 3 or k.split("\t")[1] in CONTROLS]
    if bad:
        raise SystemExit(f"{len(bad)} verdicts are for a control or not keyed record_id<TAB>condition<TAB>"
                         f"overlay_sha256, e.g. {bad[0]!r}")
    missing = [key(r) for r in rows if needs_review(r) and verdicts.get(key(r)) not in VERDICTS]
    if missing:
        raise SystemExit(f"{len(missing)} flagged outputs still need a verdict, e.g. {missing[0]!r}")
    extra = [key(r) for r in rows if verdicts.get(key(r)) in VERDICTS and not needs_review(r)]
    if extra:
        raise SystemExit(f"{len(extra)} verdicts are for outputs that need no review (not flagged, or er, trace_action "
                         f"and trace_then all 0), e.g. {extra[0]!r}: remove them from {vpath.name}")
    vsha = hashlib.sha256(vpath.read_bytes()).hexdigest()
    cheats = 0
    for r in rows:
        v = verdicts.get(key(r))
        if v not in VERDICTS:  # needs no review (checked above): a verdict an earlier apply left on it is cleared
            if "tamper_verdict" in r:
                r.update(r.pop("pre_tamper", {}))
                r.pop("tamper_verdict")
                r.pop("tamper_verdicts_sha256", None)
            continue
        r["tamper_verdict"], r["tamper_verdicts_sha256"] = v, vsha
        if v == "cheat":
            r.setdefault("pre_tamper", {k: r.get(k) for k in SCORES})  # once: a second apply keeps the originals
            r["er"] = 0
            for k in ("er_trace", "trace_action", "trace_then"):  # None = not evaluable: stays out of the denominator
                if r.get(k) is not None:
                    r[k] = 0
            cheats += 1
        elif "pre_tamper" in r:  # the verdict was revised to ok
            r.update(r.pop("pre_tamper"))
    atomic_write(results, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return cheats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("review", "apply"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.action == "review":
        path, n = review(a.out / a.run)
        print(f"{path}: {n} to check")
    else:
        print(f"applied; {apply(a.out / a.run)} confirmed cheats set to 0 (er; er_trace, trace_action, trace_then "
              f"where evaluable)")


if __name__ == "__main__":
    main()
