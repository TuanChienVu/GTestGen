"""fp32 8192-token speed probe before the dev-pilot's real job (added 28/09; fit rule fixed 28/09).

The measured fp32 tok/s used to plan a job's max_hours (v2gpu.plan_time.FP32_A6000_TOK_PER_S = 858.0) was measured
at length 2048 only: fp32 cannot use the flash kernel, so its cost at the 8191-token tail, where
the dev pilot's own histogram has real mass, is unmeasured. Before the dev pilot's real job starts, this runs a FEW
steps of train_job in canary mode at max_seq 8192 with --upcast-fp32 on the real inputs, reads the measured
tokens_per_second from its receipt, and compares the job's PROJECTED real runtime at that measured speed to the
budget already fixed before the pod exists (20.6 h since 28/09, ops/TRAIN_RUNSHEET.md §1/§3) -- this probe only checks,
with a REAL measurement, that the upper bound was not itself too optimistic.

**The planning margin is applied ONCE, not twice.** budget_hours (--budget-hours) IS
max_hours, already computed via v2gpu.plan_time.train_max_hours(tokens, 858.0, margin=1.5, overhead_h=0.5, ...) BEFORE
the pod existed -- it already carries the 1.5x safety margin. The old code recomputed max_hours a SECOND time from
the measured speed (train_max_hours(tokens, measured, margin, overhead_h), margin applied AGAIN) and compared THAT
to budget_hours, so a measured speed only 0.3% under the planning constant (858 tok/s) already refused a job that
would in fact finish with hours to spare (a reproduction: 850 tok/s -> "recomputed" 20.1 h > 19.9 h budget,
REFUSED, although the job would actually run only about 13.0 h). The fit decision now compares the RAW projected
runtime -- tokens / measured_tok_per_s / 3600 + overhead_h, no margin -- to what remains of the budget (budget_hours
- hours_already_elapsed; hours_already_elapsed is 0.0 for a probe that runs before the job has spent any of its own
budget, the only case so far -- see probe_verdict()). recomputed_max_hours (the OLD, doubly-margined quantity) is
still computed and reported, for the human record, but no longer decides fits.

**The dev passes are in the projection.** The epoch-end dev passes are work
tokens_per_second never counts, so the projection is (tokens + dev_tokens x 1/3) / measured_tok_per_s / 3600 +
overhead_h, still with no margin; --dev-tokens is required (ops/TRAIN_RUNSHEET.md §1) and overhead_h covers only model load and
checkpoint I/O. max_hours counts the same dev passes (v2gpu.plan_time.train_max_hours, dev_tokens): the dev pilot's
budget moves from 19.9 to 20.6 h and its cutoff from 570.35 to 581.92 tok/s (probe_verdict() has the arithmetic).

If the projected runtime would not fit what remains of the budget, this refuses instead of letting a doomed run
start: it still writes train_job's own receipt-train.json (the few steps it ran are real, if short) plus its own
probe_decision.json beside it, in the SAME workdir. v2gpu.jobqueue's own queue-running rule
now lets a trailing copy-window-* job run to completion (its own max_hours, or the overall deadline, whichever is
first) even after an EARLIER job's non-DONE verdict -- success (ALL_DONE), a genuine FAILED job, a TIMEOUT, or this
probe's own controlled REFUSED_WOULD_NOT_FIT are all "not DONE" the same way to the queue, and in every one of them
the queue now skips the remaining TRAINING jobs but still runs the copy window before self_stop_cmd fires. So, as
long as this probe is followed by a copy-window-* job in the SAME queue (ops/TRAIN_RUNSHEET.md §6), both
receipt-train.json and probe_decision.json ARE reliably copied home by the usual §7 copy loop of ops/TRAIN_RUNSHEET.md (which
polls each job's own status file directly, not only the queue's aggregate outcome.json) WHILE the pod is still
provably up for the copy window's own duration -- not racing a pod that may already have stopped, which is what
an earlier version of this docstring got wrong.

python -m v2gpu.train_probe --tokens TOKENS --dev-tokens DEV_TOKENS --budget-hours HOURS --workdir DIR \
    [--margin 1.5] [--overhead-h 0.5] [--hours-already-elapsed 0.0] -- --model deepseek --seed 42 --subset all --mode canary --max-steps N \
       --upcast-fp32 --max-seq 8192 --records ... --dev-records ... --catalog ... --forbidden ... \
       --train-prompts ... --dev-prompts ... --prompts-sums ... --workdir DIR
       (train_job's own args, after --; --workdir must match the one given before --)

Exit 0, PROBE_OK: the projected runtime fits what remains of the budget -- the real job follows.
Exit 5, REFUSED_WOULD_NOT_FIT: it does not fit -- this job's own nonzero exit is exactly what v2gpu.jobqueue already
treats as a non-DONE job (the queue then skips the real training job but still runs the trailing copy window).
Exit 2, PROBE_ERROR: train_job itself did not reach MAX_STEPS (any other status means something is already wrong
with the setup, independent of speed) or wrote no tokens_per_second, or the receipt shows this did not actually
measure what it claims to: a mismatched --workdir before/after --, mode != "canary",
train_config.upcast_fp32 != True or train_config.max_seq != 8192 -- a bf16 or non-8192 probe would measure the wrong
speed entirely, silently, without any of these checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import train_job
from .plan_time import DEV_FORWARD_COST, train_max_hours

REFUSED_WOULD_NOT_FIT = 5
PROBE_ERROR = 2


def probe_verdict(tokens: int, measured_tok_per_s: float, budget_hours: float, overhead_h: float = 0.5,
                  hours_already_elapsed: float = 0.0, margin: float = 1.5, *, dev_tokens: int,
                  dev_forward_cost: float = DEV_FORWARD_COST) -> dict:
    """Whether the job's PROJECTED real runtime at the just-measured tok/s --
    tokens / measured_tok_per_s / 3600 + overhead_h, no safety margin -- fits what remains of budget_hours
    (budget_hours - hours_already_elapsed). budget_hours is max_hours, already computed from the conservative
    upper-bound tok/s via v2gpu.plan_time.train_max_hours(..., margin, ...) BEFORE the pod existed: it already
    carries the margin once. Applying margin again here (the old rule: comparing train_max_hours(tokens, measured,
    margin, overhead_h) to budget_hours) double-counts it, refusing jobs that would in fact finish comfortably (a
    reproduction: 850 tok/s, only 1% under the 858 tok/s planning constant, "recomputed" to 20.1 h > a 19.9 h
    budget under the old rule, although the job would actually run about 13.0 h + 0.5 h overhead = 13.5 h).

    hours_already_elapsed accounts for any of the SAME job's own budget already spent before this probe measurement
    -- 0.0 for a probe that runs before the real job has started at all, the only case so far; the parameter exists
    so a probe re-run after a resume, partway through the budget, can be judged against what is actually left of it.

    recomputed_max_hours (the OLD, doubly-margined quantity, via v2gpu.plan_time.train_max_hours with `margin`) is
    still computed and reported for the human record (it is what a FRESH plan would use if replanning from this
    measured speed), but it no longer decides `fits`.

    The epoch-end dev passes are real work that tokens_per_second never
    counts (train_loop.dev_loss, one forward per example under no_grad). dev_tokens is the job's dev tokens over all
    its passes (tokens of every --dev-prompts example x the maximum number of epochs); each counts as
    dev_forward_cost (v2gpu.plan_time.DEV_FORWARD_COST, 1/3) of a training token, with no margin, like the training
    tokens: projected = (tokens + dev_tokens x dev_forward_cost) / measured / 3600 + overhead_h. overhead_h now
    covers only model load and checkpoint I/O. Without the dev passes, the dev pilot's cutoff was 570.35 tok/s and
    jobs between it and about 585 tok/s passed the probe but would TIMEOUT at 19.9 h. budget_hours now counts the
    same dev passes too (v2gpu.plan_time.train_max_hours: 20.6 h for the dev pilot), and the cutoff is
    (39,833,334 + 6,823,705 / 3) / ((20.6 - 0.5) x 3600) = 581.92 tok/s. dev_tokens has no default, so no caller
    can leave the dev passes out by accident. cutoff_tok_per_s (the slowest measured speed that still fits; None
    when the remaining budget does not even cover overhead_h) is reported for the human record."""
    if dev_tokens < 0:
        raise ValueError(f"probe_verdict: dev_tokens must be >= 0, got {dev_tokens}")
    work_tokens = tokens + dev_tokens * dev_forward_cost
    projected_hours = work_tokens / measured_tok_per_s / 3600 + overhead_h
    remaining_budget_hours = budget_hours - hours_already_elapsed
    cutoff_tok_per_s = (work_tokens / ((remaining_budget_hours - overhead_h) * 3600)
                        if remaining_budget_hours > overhead_h else None)
    recomputed_max_hours = train_max_hours(tokens, measured_tok_per_s, margin, overhead_h, dev_tokens=dev_tokens,
                                           dev_forward_cost=dev_forward_cost)
    return {"measured_tok_per_s": measured_tok_per_s, "tokens": tokens, "dev_tokens": dev_tokens,
           "dev_forward_cost": dev_forward_cost, "overhead_h": overhead_h,
           "hours_already_elapsed": hours_already_elapsed, "projected_hours": projected_hours,
           "budget_hours": budget_hours, "remaining_budget_hours": remaining_budget_hours, "margin": margin,
           "cutoff_tok_per_s": cutoff_tok_per_s, "recomputed_max_hours": recomputed_max_hours,
           "fits": projected_hours <= remaining_budget_hours}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m v2gpu.train_probe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, required=True, help="the REAL job's own token estimate (ops/TRAIN_RUNSHEET.md §1)")
    ap.add_argument("--dev-tokens", type=int, required=True,
                    help="the REAL job's dev tokens over all its dev passes (ops/TRAIN_RUNSHEET.md §1)")
    ap.add_argument("--budget-hours", type=float, required=True,
                    help="max_hours already given to the queue for the real job (the conservative upper bound)")
    ap.add_argument("--margin", type=float, default=1.5, help="only for the reported recomputed_max_hours; "
                                                               "NOT applied again to decide fits")
    ap.add_argument("--overhead-h", type=float, default=0.5)
    ap.add_argument("--hours-already-elapsed", type=float, default=0.0,
                    help="hours of budget_hours already spent by this job before this probe measurement (0.0 "
                        "unless this probe runs mid-resume)")
    ap.add_argument("--workdir", type=Path, required=True, help="the probe's OWN workdir; must equal the --workdir "
                                                                 "also given after -- to train_job")
    ap.add_argument("train_job_argv", nargs=argparse.REMAINDER,
                    help="train_job's own args, after --; must include --mode canary, --max-steps, --upcast-fp32 "
                        "and --max-seq 8192")
    a = ap.parse_args(argv)
    a.workdir.mkdir(parents=True, exist_ok=True)   # so decision_path is writable even if train_job refuses first
    inner = [x for x in a.train_job_argv if x != "--"]
    receipt_path = a.workdir / "receipt-train.json"
    decision_path = a.workdir / "probe_decision.json"

    def refuse(error: str) -> int:
        decision_path.write_text(json.dumps({"status": "PROBE_ERROR", "error": error}, indent=2, sort_keys=True))
        print(f"PROBE_ERROR: {error}", file=sys.stderr)
        return PROBE_ERROR

    # A mismatched --workdir before/after -- would have train_job write its receipt somewhere this
    # probe never reads (or, worse, read an UNRELATED prior receipt sitting at a.workdir by coincidence) -- caught
    # before train_job even runs, with a clear reason, rather than a confusing "no readable receipt-train.json".
    if "--workdir" in inner:
        inner_workdir = Path(inner[inner.index("--workdir") + 1])
        if inner_workdir != a.workdir:
            return refuse(f"--workdir after -- ({inner_workdir}) does not match this probe's own --workdir "
                          f"({a.workdir})")
    code = train_job.main(inner)
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as e:
        return refuse(f"no readable receipt-train.json ({e!r}); train_job exited {code}")
    if receipt.get("status") != "MAX_STEPS":
        return refuse(f"train_job status {receipt.get('status')!r}, not MAX_STEPS (exit {code})")
    tok_per_s = (receipt.get("result") or {}).get("tokens_per_second")
    if not tok_per_s:
        return refuse("receipt has no tokens_per_second")
    # This probe's whole point is measuring the REAL job's own fp32-upcast speed at its own max_seq;
    # a bf16 or shorter-max_seq run would silently measure the wrong thing and still look like a normal MAX_STEPS
    # receipt. Enforce the docstring's own requirements instead of trusting the caller's argv -- checked LAST, after
    # the more fundamental status/tokens_per_second checks above, so those keep their own specific PROBE_ERROR text.
    if receipt.get("mode") != "canary":
        return refuse(f"train_job mode {receipt.get('mode')!r}, not 'canary' -- this probe must run in canary mode")
    train_config = receipt.get("train_config") or {}
    if train_config.get("upcast_fp32") is not True:
        return refuse(f"train_config.upcast_fp32 = {train_config.get('upcast_fp32')!r}, not True -- this probe must "
                      f"measure the real job's own fp32-upcast speed, never bf16")
    if train_config.get("max_seq") != 8192:
        return refuse(f"train_config.max_seq = {train_config.get('max_seq')!r}, not 8192 -- this probe must measure "
                      f"at the real job's own max_seq")
    verdict = probe_verdict(a.tokens, tok_per_s, a.budget_hours, overhead_h=a.overhead_h,
                            hours_already_elapsed=a.hours_already_elapsed, margin=a.margin, dev_tokens=a.dev_tokens)
    verdict["status"] = "PROBE_OK" if verdict["fits"] else "REFUSED_WOULD_NOT_FIT"
    decision_path.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if not verdict["fits"]:
        print(f"REFUSED_WOULD_NOT_FIT: projected {verdict['projected_hours']:.2f} h > remaining budget "
             f"{verdict['remaining_budget_hours']:.2f} h (the margin is not applied twice) -- not starting a run "
             f"that would be cut off", file=sys.stderr)
        return REFUSED_WOULD_NOT_FIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
