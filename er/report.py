"""Summarise <out>/<run>/er_results.jsonl as Markdown: python -m er.report --run <run> [--out DIR]

Read-only. Refuses while a flagged output has no tamper verdict, and when the rows do not carry exactly the verdicts
of the current er_tamper_verdicts.json with its sha256 (edited after tamper apply, or applied from another file: run
tamper apply again); a run where nothing was reviewed (no such file, no row with a verdict) is reported. The first line
warns when rows are missing from the WHOLE manifest (er_complete.json of the last batch; also when there is no such file
or it says a batch is running).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .batch import DEFAULT_OUT, load_results
from .tamper import VERDICTS, key, needs_review


def frac(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(vals)}/{len(vals)}"


def verdicts_mismatch(run_dir: Path, rows: list[dict]) -> str | None:
    """Why the rows do not carry exactly the verdicts of the current er_tamper_verdicts.json (tamper apply writes each
    row's verdict and the file's sha256); None when they do, or when there is no such file and no row has a verdict."""
    vpath = run_dir / "er_tamper_verdicts.json"
    sha, verdicts = None, {}
    if vpath.exists():
        data = vpath.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        try:
            verdicts = json.loads(data)
        except ValueError as exc:
            return f"{vpath}: not valid JSON ({exc})"
        if not isinstance(verdicts, dict):
            return f"{vpath}: not a JSON object"
    bad = [r for r in rows if (r.get("tamper_verdict"), r.get("tamper_verdicts_sha256")) !=
           ((verdicts[key(r)], sha) if verdicts.get(key(r)) in VERDICTS else (None, None))]
    if not bad:
        return None
    return (f"{len(bad)} rows do not carry the verdicts of the current {vpath} ("
            f"{'the file is missing' if sha is None else 'edited after tamper apply, or applied from another file'}; "
            f"e.g. {key(bad[0])!r}): run python -m er.tamper apply again")


def completeness(rows: list[dict], complete: dict | None) -> str | None:
    """Warning line when the rows are not the complete set the last batch recorded."""
    if complete is None:
        return "**THIẾU: không có er_complete.json** — batch chưa chạy xong lần nào; số liệu dưới đây có thể thiếu dòng."
    if complete.get("running"):
        return ("**THIẾU: er_complete.json ghi batch đang chạy** (hoặc bị giết giữa chừng) — số liệu dưới đây có thể "
                "thiếu dòng.")
    out = []
    if complete["missing"]:
        rid, cond = complete["missing"][0]
        out.append(f"**THIẾU {len(complete['missing'])} dòng** so với toàn bộ manifest (vd. `{rid}` — `{cond}`)")
    if complete.get("duplicates"):
        out.append(f"**{len(complete['duplicates'])} (record, condition) có hai dòng**")
    if len(rows) != complete["rows"]:
        out.append(f"**er_results.jsonl có {len(rows)} dòng, er_complete.json ghi {complete['rows']}**: tệp đã đổi "
                   f"sau lần chạy batch cuối")
    return "; ".join(out) or None


def report(rows: list[dict], complete: dict | None = None) -> str:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    warn = completeness(rows, complete)
    out = [warn, ""] if warn else []
    out += ["| điều kiện | n | trạng thái ER | ER | ER-trace | trace_action | trace_then | không đánh giá được | tamper |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cond in sorted(by_cond):
        rs = by_cond[cond]
        st = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["er_status"] for r in rs).items()))
        none = sum(r["er_trace"] is None for r in rs)
        tam = sum(bool(r["tamper_flags"]) for r in rs)
        out.append(f"| {cond} | {len(rs)} | {st} | {frac(rs, 'er')} | {frac(rs, 'er_trace')} | "
                   f"{frac(rs, 'trace_action')} | {frac(rs, 'trace_then')} | {none} | {tam} |")
    status = {(r["record_id"], r["condition"]): r["er_status"] for r in rows}
    rids = sorted({r["record_id"] for r in rows})
    if {"neg", "gold", "stub"} <= set(by_cond):
        ok = [rid for rid in rids if status.get((rid, "gold")) == "PASS" and status.get((rid, "neg")) != "PASS"
              and status.get((rid, "stub")) != "PASS"]
        out.append(f"\nER, qua đủ ba đối chứng (gold PASS, neg và stub không PASS): **{len(ok)}/{len(rids)}**")
    flagged = Counter(r["condition"] for r in rows if r.get("tamper_flags"))
    review = sum(needs_review(r) for r in rows)
    out.append(f"\nGắn cờ tamper theo điều kiện: {dict(sorted(flagged.items())) or '—'}; cần xét tay "
               f"(gắn cờ và er = 1, trace_action = 1 hoặc trace_then = 1): {review}")
    verdicts = {v: Counter(r["condition"] for r in rows if r.get("tamper_verdict") == v) for v in VERDICTS}
    out.append("\nKết quả xét tay tamper theo điều kiện: "
               + "; ".join(f"{v} {dict(sorted(n.items())) or '—'}" for v, n in verdicts.items()))
    errs = Counter(r["condition"] for r in rows if r.get("score_error"))
    out.append(f"\nLỗi khi chấm vết (score_error, ER giữ nguyên): {dict(sorted(errs.items())) or '—'}")
    refs = Counter(r.get("ref_error") for r in rows if r.get("ref_error"))
    if refs:
        out.append(f"\nLỗi tham chiếu gold: {dict(refs)}")
    secs = sorted(r["seconds"] for r in rows if r.get("seconds") is not None)
    if secs:
        out.append(f"\nLượt thường: {len(secs)} lượt, trung bình {sum(secs) / len(secs):.2f} s, "
                   f"p95 {secs[int(0.95 * (len(secs) - 1))]:.2f} s, max {secs[-1]:.2f} s; "
                   f"lạnh {sum(bool(r.get('cold')) for r in rows)}; RAM đỉnh "
                   f"{max((r.get('mem_peak_bytes') or 0) for r in rows) / 2**20:.0f} MB")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    run_dir = a.out / a.run
    rows = load_results(run_dir / "er_results.jsonl")
    pending = [r for r in rows if needs_review(r) and r.get("tamper_verdict") not in VERDICTS]
    if pending:
        raise SystemExit(f"{len(pending)} flagged outputs not reviewed: run python -m er.tamper review/apply first")
    why = verdicts_mismatch(run_dir, rows)
    if why:
        raise SystemExit(why)
    cpath = run_dir / "er_complete.json"
    print(report(rows, json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else None))


if __name__ == "__main__":
    main()
