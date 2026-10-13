"""The fp32 8192-token speed probe (added 28/09) before the dev pilot's real job.
Arithmetic (probe_verdict), the refusal branch (would not fit -> controlled stop, exit 5, no real job started) and
the receipt (train_job's own receipt-train.json plus probe_decision.json, written even on refusal); the
boundary tests of the fit rule (the margin must not be applied twice) and a test that a REAL
v2gpu.jobqueue queue still runs a trailing copy-window-* job after a job that exits with this probe's refusal code (a
stand-in script, not v2gpu.train_probe itself); and the dev passes in the projection and in
max_hours (20.6 h for the dev pilot, cutoff 581.92 tok/s), with the boundary tests at the new cutoff."""
import datetime as dt
import json
import sys

import pytest

from tests_gpu.test_train_job import files
from v2gpu.jobqueue import run_queue
from v2gpu.train_probe import PROBE_ERROR, REFUSED_WOULD_NOT_FIT, main, probe_verdict


def decision(tmp_path):
    return json.loads((tmp_path / "run" / "probe_decision.json").read_text())


def receipt(tmp_path):
    return json.loads((tmp_path / "run" / "receipt-train.json").read_text())


def probe(tmp_path, tokenizers, tokens=100_000, budget_hours=100.0, max_steps="2", dev_tokens=0, **extra):
    """--upcast-fp32 and --max-seq 8192 (this probe's OWN requirements)
    on top of files()'s own --max-seq 512 -- argparse keeps the LAST value of a repeated option, so 8192 wins."""
    canary = [a if a != "test" else "canary" for a in files(tmp_path, tokenizers)]
    args = ["--tokens", str(tokens), "--dev-tokens", str(dev_tokens), "--budget-hours", str(budget_hours),
           "--workdir", str(tmp_path / "run"), "--"] + canary + ["--subset", "all", "--max-steps", max_steps, "--upcast-fp32", "--max-seq", "8192",
           "--workdir", str(tmp_path / "run")]
    return main(args + [f"--{k.replace('_', '-')}={v}" for k, v in extra.items()])


# arithmetic (the margin already inside budget_hours must not be applied a second time;
# the epoch-end dev passes are counted in the projection, forward-only, margin-free)

DEV_PILOT_TOKENS = 39_833_334          # the dev pilot's own 5-epoch upper-bound token estimate (TRAIN_RUNSHEET §1)
DEV_PILOT_DEV_TOKENS = 6_823_705       # 5 dev passes x 1,364,741 tokens (612 v0 dev examples, DeepSeek tokenizer)
DEV_PILOT_BUDGET_H = 20.6              # HOURS, fixed at TRAIN_RUNSHEET §1/§3 BEFORE the pod exists (already margined)
# (39,833,334 + 6,823,705 / 3) / ((20.6 - 0.5) * 3600) = 42,107,902.33 / 72,360 = 581.92 tok/s
DEV_PILOT_CUTOFF = (DEV_PILOT_TOKENS + DEV_PILOT_DEV_TOKENS / 3) / ((DEV_PILOT_BUDGET_H - 0.5) * 3600)


def v_expected(tokens, tok_per_s, margin=1.5, overhead_h=0.5, dev_tokens=0):
    from v2gpu.plan_time import train_max_hours

    return train_max_hours(tokens, tok_per_s, margin, overhead_h, dev_tokens=dev_tokens)


def dev_pilot(tok_per_s):
    return probe_verdict(DEV_PILOT_TOKENS, tok_per_s, DEV_PILOT_BUDGET_H, dev_tokens=DEV_PILOT_DEV_TOKENS)


def test_probe_verdict_compares_the_raw_projected_runtime_not_a_re_margined_one():
    """The old code compared train_max_hours(tokens, measured, margin,
    overhead_h) -- the margin applied AGAIN -- to budget_hours (which already carries the SAME margin, set once
    before the pod exists). fits must instead compare tokens/measured/3600 + overhead_h (no margin) to what remains
    of the budget."""
    v = probe_verdict(tokens=1_000_000, measured_tok_per_s=100.0, budget_hours=10.0, dev_tokens=0)
    assert v["projected_hours"] == pytest.approx(1_000_000 / 100.0 / 3600 + 0.5)
    assert v["remaining_budget_hours"] == 10.0            # hours_already_elapsed defaults to 0.0
    assert v["fits"] is (v["projected_hours"] <= v["remaining_budget_hours"])
    # recomputed_max_hours (the OLD, doubly-margined quantity) is still reported, but must not decide fits:
    assert v["recomputed_max_hours"] == v_expected(1_000_000, 100.0, dev_tokens=0)


def test_the_dev_passes_are_in_the_projection_forward_only_and_margin_free():
    """The epoch-end dev passes (train_loop.dev_loss, one forward per example under no_grad) are real work
    tokens_per_second never counts. A training token costs about three forwards (forward, the checkpoint recompute,
    the backward through the frozen base), so each dev token counts as 1/3 of a training token (DEV_FORWARD_COST);
    no margin on either term."""
    from v2gpu.plan_time import DEV_FORWARD_COST

    assert DEV_FORWARD_COST == pytest.approx(1 / 3)
    v = probe_verdict(tokens=1_000_000, measured_tok_per_s=100.0, budget_hours=10.0, dev_tokens=300_000)
    assert v["projected_hours"] == pytest.approx((1_000_000 + 300_000 / 3) / 100.0 / 3600 + 0.5)
    assert v["dev_tokens"] == 300_000 and v["dev_forward_cost"] == pytest.approx(1 / 3)


def test_dev_tokens_is_required():
    """No silent default of 0: a caller that forgets the dev passes would leave them out again unnoticed."""
    with pytest.raises(TypeError):
        probe_verdict(1000, 1000.0, 10.0)


def test_the_verdict_reports_the_cutoff_speed():
    """cutoff_tok_per_s: the slowest measured speed that still fits, for the report and probe_decision.json. For
    the dev pilot: (39,833,334 + 6,823,705 / 3) / ((20.6 - 0.5) * 3600) = 581.92 tok/s (the earlier cutoff, without
    the dev passes and with max_hours 19.9, was 570.35)."""
    v = dev_pilot(800.0)
    assert v["cutoff_tok_per_s"] == pytest.approx(DEV_PILOT_CUTOFF)
    assert v["cutoff_tok_per_s"] == pytest.approx(581.92, abs=0.01)


def test_no_cutoff_when_the_remaining_budget_does_not_even_cover_the_overhead():
    v = probe_verdict(1000, 1000.0, budget_hours=0.5, dev_tokens=0)             # 0.5 h left, 0.5 h overhead
    assert v["cutoff_tok_per_s"] is None and v["fits"] is False


def test_negative_dev_tokens_are_refused():
    with pytest.raises(ValueError, match="dev_tokens"):
        probe_verdict(1000, 1000.0, 10.0, dev_tokens=-1)


def test_probe_verdict_the_850_tok_per_s_case_the_old_rule_refused_but_should_not_have():
    """A reproduction: at 850 tok/s (1% under the 858 tok/s planning constant) the OLD rule
    "recomputed" max_hours from the measured speed (with the dev passes: 20.8 h), ABOVE the 20.6 h
    dev-pilot budget -- REFUSED a job that would in fact run about (39,833,334 + 6,823,705 / 3) / 850 / 3600 =
    13.76 h + 0.5 h overhead = 14.26 h, inside 20.6 h."""
    v = dev_pilot(850.0)
    assert v["fits"] is True                                                     # NEW rule: PASSES
    old_rule_recomputed = v_expected(DEV_PILOT_TOKENS, 850.0, dev_tokens=DEV_PILOT_DEV_TOKENS)
    assert old_rule_recomputed > DEV_PILOT_BUDGET_H                              # OLD rule would have REFUSED this
    assert v["recomputed_max_hours"] == old_rule_recomputed                      # still reported, just not decisive


def test_800_tok_per_s_must_pass_the_dev_pilot_budget():
    """(39,833,334 + 6,823,705 / 3) / 800 / 3600 = 14.62 h + 0.5 h overhead = 15.12 h, inside the 20.6 h budget.
    This MUST PASS."""
    v = dev_pilot(800.0)
    assert v["fits"] is True
    assert v["projected_hours"] == pytest.approx((DEV_PILOT_TOKENS + DEV_PILOT_DEV_TOKENS / 3) / 800.0 / 3600 + 0.5)


def test_855_tok_per_s_passes_comfortably():
    """855 tok/s (just under the 858 tok/s planning constant) was REFUSED under the old double-margin rule; it must
    pass easily: (39,833,334 + 6,823,705 / 3) / 855 / 3600 = 13.68 h + 0.5 h = 14.18 h <= 20.6 h."""
    assert dev_pilot(855.0)["fits"] is True


@pytest.mark.parametrize("tok_per_s", [571.0, 575.0, 580.0])
def test_the_old_570_to_585_band_is_now_refused(tok_per_s):
    """In 570.35-585 tok/s the earlier rule (no dev passes, max_hours 19.9) PASSED jobs that would TIMEOUT.
    580 tok/s: that rule projected 39,833,334 / 580 / 3600 + 0.5 = 19.58 h <= 19.9 h (passed), but the dev passes alone
    add 0.73 h even at a cheaper 1/4 forward cost: training + dev = 19.89 h before any model load.
    Now: (39,833,334 + 6,823,705 / 3) / 580 / 3600 + 0.5 = 20.67 h > 20.6 h -- REFUSED. (582-585 tok/s now pass:
    they fit in the 20.6 h that includes the dev passes; see the boundary test below.)"""
    old_rule = DEV_PILOT_TOKENS / tok_per_s / 3600 + 0.5
    assert old_rule <= 19.9                                                      # the earlier rule let it through
    v = dev_pilot(tok_per_s)
    assert v["fits"] is False
    assert v["projected_hours"] > DEV_PILOT_BUDGET_H


def test_just_below_the_new_cutoff_is_refused_just_above_it_passes():
    """The new cutoff is 581.92 tok/s. 581 tok/s: (39,833,334 + 6,823,705 / 3) / 581 / 3600 + 0.5 = 20.632 h >
    20.6 h -- REFUSED. 582 tok/s: 20.597 h <= 20.6 h -- PASSES."""
    below, above = dev_pilot(581.0), dev_pilot(582.0)
    assert below["fits"] is False and below["projected_hours"] == pytest.approx(20.632, abs=0.001)
    assert above["fits"] is True and above["projected_hours"] == pytest.approx(20.597, abs=0.001)


def test_the_probe_budget_is_the_plan_time_max_hours():
    """The budget the probe is judged against is the plan's own max_hours, with the same dev passes."""
    assert v_expected(DEV_PILOT_TOKENS, 858.0, dev_tokens=DEV_PILOT_DEV_TOKENS) == DEV_PILOT_BUDGET_H


def test_probe_verdict_fits_exactly_at_the_boundary():
    """A clean, exact boundary (no reproduction numbers): fits iff the projected runtime is <= budget_hours -
    hours_already_elapsed, with NO margin applied to the projection."""
    tokens, dev_tokens, tok_per_s, overhead_h = 1000, 300, 1000.0, 0.5
    exact = (tokens + dev_tokens / 3) / tok_per_s / 3600 + overhead_h
    v = probe_verdict(tokens, tok_per_s, budget_hours=exact, overhead_h=overhead_h, dev_tokens=dev_tokens)
    assert v["fits"] is True
    v2 = probe_verdict(tokens, tok_per_s, budget_hours=exact - 0.0001, overhead_h=overhead_h, dev_tokens=dev_tokens)
    assert v2["fits"] is False


def test_probe_verdict_hours_already_elapsed_shrinks_the_remaining_budget():
    """hours_already_elapsed (0.0 unless this probe runs mid-resume) is subtracted from budget_hours before the fit
    check -- a job resumed with only a little budget left must be judged against what is actually left of it, not
    the whole original budget again."""
    tokens, tok_per_s, overhead_h = 1000, 1000.0, 0.5
    exact = tokens / tok_per_s / 3600 + overhead_h
    v = probe_verdict(tokens, tok_per_s, budget_hours=exact + 1.0, overhead_h=overhead_h, hours_already_elapsed=1.0,
                      dev_tokens=0)
    assert v["remaining_budget_hours"] == pytest.approx(exact)
    assert v["fits"] is True
    v2 = probe_verdict(tokens, tok_per_s, budget_hours=exact + 1.0, overhead_h=overhead_h,
                       hours_already_elapsed=1.0001, dev_tokens=0)
    assert v2["fits"] is False


# the receipt and the fitting (proceed) branch


def test_probe_ok_when_the_measured_speed_easily_fits_the_budget(tmp_path, tokenizers):
    assert probe(tmp_path, tokenizers, tokens=1000, budget_hours=1000.0) == 0
    d = decision(tmp_path)
    assert d["status"] == "PROBE_OK" and d["fits"] is True and d["measured_tok_per_s"] > 0
    r = receipt(tmp_path)
    assert r["status"] == "MAX_STEPS" and r["result"]["tokens_per_second"] > 0


def test_the_decision_file_records_the_dev_passes_and_the_cutoff(tmp_path, tokenizers):
    assert probe(tmp_path, tokenizers, tokens=1000, dev_tokens=3000, budget_hours=1000.0) == 0
    d = decision(tmp_path)
    assert d["dev_tokens"] == 3000 and d["dev_forward_cost"] == pytest.approx(1 / 3)
    assert d["projected_hours"] == pytest.approx((1000 + 3000 / 3) / d["measured_tok_per_s"] / 3600 + 0.5)
    assert d["cutoff_tok_per_s"] == pytest.approx((1000 + 3000 / 3) / ((1000.0 - 0.5) * 3600))


def test_the_cli_refuses_to_run_without_dev_tokens(tmp_path, capsys):
    """--dev-tokens is required: argparse exits 2 before train_job runs, so no probe measures without the dev passes
    in its projection."""
    with pytest.raises(SystemExit) as e:
        main(["--tokens", "1000", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--", "ignored"])
    assert e.value.code == 2 and "--dev-tokens" in capsys.readouterr().err
    assert not (tmp_path / "run" / "receipt-train.json").exists()


# the refusal branch: a controlled stop, not a crash -- exit code, receipt and decision file all present


def test_probe_refuses_and_stops_a_run_that_would_be_cut_off(tmp_path, tokenizers):
    assert probe(tmp_path, tokenizers, tokens=10**12, budget_hours=0.001) == REFUSED_WOULD_NOT_FIT
    d = decision(tmp_path)
    assert d["status"] == "REFUSED_WOULD_NOT_FIT" and d["fits"] is False
    assert d["recomputed_max_hours"] > d["budget_hours"]
    r = receipt(tmp_path)                          # the receipt is still written: this is a controlled stop
    assert r["status"] == "MAX_STEPS" and r["result"]["tokens_per_second"] > 0


def test_probe_error_when_train_job_never_reaches_max_steps(tmp_path, tokenizers):
    """A refused preflight (here the canary is missing --max-steps, which train_job itself refuses before writing
    any receipt at all) is a setup problem, independent of speed: PROBE_ERROR, not a would-not-fit refusal."""
    canary = [a if a != "test" else "canary" for a in files(tmp_path, tokenizers)]
    args = ["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--"] + canary + [
        "--subset", "all", "--workdir", str(tmp_path / "run")]                 # no --max-steps: canary mode refuses
    assert main(args) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and "no readable receipt" in d["error"]


def test_probe_error_when_train_job_finishes_done_instead_of_max_steps(tmp_path, tokenizers):
    """A receipt DOES exist, but its status is not MAX_STEPS (here: a full --mode test run reaches DONE, since it was
    never given --max-steps at all) -- still a setup problem, not a speed verdict."""
    args = files(tmp_path, tokenizers)      # --mode test (unmodified): a tiny full training run, status DONE
    assert main(["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--"] + args + [
        "--subset", "all", "--workdir", str(tmp_path / "run")]) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and "DONE" in d["error"] and "MAX_STEPS" in d["error"]
    assert receipt(tmp_path)["status"] == "DONE"


def test_probe_error_when_the_receipt_has_no_tokens_per_second(tmp_path, tokenizers, monkeypatch):
    """A defensive branch: if train_job ever wrote a MAX_STEPS receipt with no tokens_per_second, the probe must
    refuse for that reason too, rather than dividing by nothing or silently treating it as fitting."""
    import v2gpu.train_probe as tp

    def fake_main(argv):
        (tmp_path / "run").mkdir(parents=True, exist_ok=True)
        (tmp_path / "run" / "receipt-train.json").write_text(
            json.dumps({"status": "MAX_STEPS", "result": {"tokens_per_second": None}}))
        return 4

    monkeypatch.setattr(tp.train_job, "main", fake_main)
    assert tp.main(["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--",
                    "ignored"]) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and "tokens_per_second" in d["error"]


# A bf16 or non-8192 probe would measure the wrong speed entirely, silently -- this
# enforces the docstring's own requirements (--mode canary, --upcast-fp32, --max-seq 8192, matching --workdir)
# instead of trusting the caller's argv.


def fake_receipt(**train_config_overrides):
    def fake_main(tmp_path):
        def _fake(argv):
            (tmp_path / "run").mkdir(parents=True, exist_ok=True)
            cfg = {"upcast_fp32": True, "max_seq": 8192, **train_config_overrides}
            (tmp_path / "run" / "receipt-train.json").write_text(json.dumps(
                {"status": "MAX_STEPS", "mode": "canary", "train_config": cfg,
                 "result": {"tokens_per_second": 100.0}}))
            return 0
        return _fake
    return fake_main


def test_probe_error_when_train_job_did_not_run_in_canary_mode(tmp_path, monkeypatch):
    import v2gpu.train_probe as tp

    def fake_main(argv):
        (tmp_path / "run").mkdir(parents=True, exist_ok=True)
        (tmp_path / "run" / "receipt-train.json").write_text(json.dumps(
            {"status": "MAX_STEPS", "mode": "job", "train_config": {"upcast_fp32": True, "max_seq": 8192},
             "result": {"tokens_per_second": 100.0}}))
        return 0

    monkeypatch.setattr(tp.train_job, "main", fake_main)
    assert tp.main(["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--",
                    "ignored"]) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and "canary" in d["error"] and "'job'" in d["error"]


def test_probe_error_when_the_receipt_shows_no_upcast_fp32(tmp_path, monkeypatch):
    """A bf16 probe would measure the WRONG speed (28% faster than fp32, measured 27/09) and silently pass a job
    that would in fact be cut off -- must never be mistaken for a valid measurement."""
    import v2gpu.train_probe as tp

    monkeypatch.setattr(tp.train_job, "main", fake_receipt(upcast_fp32=False)(tmp_path))
    assert tp.main(["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--",
                    "ignored"]) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and "upcast_fp32" in d["error"]


def test_probe_error_when_the_receipt_was_not_measured_at_max_seq_8192(tmp_path, monkeypatch):
    import v2gpu.train_probe as tp

    monkeypatch.setattr(tp.train_job, "main", fake_receipt(max_seq=2048)(tmp_path))
    assert tp.main(["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--",
                    "ignored"]) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and "max_seq" in d["error"] and "8192" in d["error"]


def test_probe_error_when_workdir_after_the_double_dash_does_not_match_the_probes_own(tmp_path):
    """A mismatched --workdir before/after -- would have train_job write its receipt somewhere this probe never
    reads: caught before train_job even runs, with a clear reason (never a confusing "no readable receipt")."""
    other = tmp_path / "elsewhere"
    assert main(["--tokens", "1000", "--dev-tokens", "0", "--budget-hours", "1000.0", "--workdir", str(tmp_path / "run"), "--", "ignored",
                "--workdir", str(other)]) == PROBE_ERROR
    d = decision(tmp_path)
    assert d["status"] == "PROBE_ERROR" and str(other) in d["error"] and str(tmp_path / "run") in d["error"]
    assert not other.exists()   # train_job never even ran


# A REAL v2gpu.jobqueue queue with this probe as job 1 must not self-stop before
# a trailing copy-window-* job (job 2) runs, so a refusal's own files are actually copyable home while the pod is
# still up (not racing a pod that may already have stopped -- an earlier gap). Proved in two parts: (a)
# above, this probe run for real (in-process, tokenizers fixture) writes REFUSED_WOULD_NOT_FIT + a real receipt
# BEFORE its nonzero exit (test_probe_refuses_and_stops_a_run_that_would_be_cut_off); (b) here, a REAL
# v2gpu.jobqueue.run_queue, given a job that exits with this probe's OWN REFUSED_WOULD_NOT_FIT code, still runs a
# trailing copy-window-* job (a fast, real subprocess is used here instead of a real train_job/model run, which
# tests_gpu/test_jobqueue.py's own copy-window tests already cover generically for FAILED/TIMEOUT/success).


def test_a_real_jobqueue_still_runs_the_copy_window_after_a_probe_refusal(tmp_path):
    probe_wd = tmp_path / "runs" / "fp32-probe-8192"
    copy_wd = tmp_path / "runs" / "copy-window-x"
    probe_cmd = [sys.executable, "-c",
                f"import pathlib, sys; p = pathlib.Path({str(probe_wd)!r}); p.mkdir(parents=True, exist_ok=True); "
                f"(p / 'probe_decision.json').write_text('{{\"status\": \"REFUSED_WOULD_NOT_FIT\"}}'); "
                f"sys.exit({REFUSED_WOULD_NOT_FIT})"]
    copy_cmd = [sys.executable, "-c",
               f"import pathlib; p = pathlib.Path({str(copy_wd)!r}); p.mkdir(parents=True, exist_ok=True); "
               f"(p / 'out.txt').write_text('ok')"]
    queue_dir = tmp_path / "queue"
    spec_path = tmp_path / "queue.json"
    spec_path.write_text(json.dumps({
        "queue_dir": str(queue_dir), "lock_path": str(tmp_path / "pod.lock"), "self_stop_cmd": None,
        "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)).isoformat(),
        "jobs": [{"name": "fp32-probe-8192", "workdir": str(probe_wd), "max_hours": 0.05, "cmd": probe_cmd},
                {"name": "copy-window-x", "workdir": str(copy_wd), "max_hours": 0.05, "cmd": copy_cmd}]}))
    assert run_queue(spec_path, poll=0.05) == 1                               # not ALL_DONE: the probe refused
    outcome = json.loads((queue_dir / "outcome.json").read_text())
    assert outcome["outcome"] == "FAILED_fp32-probe-8192"       # jobqueue sees a controlled refusal as a plain exit
    assert (queue_dir / "copy-window-x.status").read_text() == "DONE\n"       # still ran, unlike before
    assert (copy_wd / "out.txt").exists()
    assert json.loads((probe_wd / "probe_decision.json").read_text())["status"] == "REFUSED_WOULD_NOT_FIT"
