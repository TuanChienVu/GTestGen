# tests_gpu/test_plan_time.py
import subprocess
import sys
from pathlib import Path

import pytest

from v2gpu.plan_time import FP32_A6000_TOK_PER_S, train_max_hours

ROOT = Path(__file__).resolve().parents[1]


def test_the_measured_constant_is_the_decided_a6000_fp32_speed():
    """A6000 fp32, pre-canary 27/09, measured 858 tok/s (bf16 1190, -28%; fp32
    upcast is now mandatory for every real job, so only fp32's own speed plans a real job's max_hours)."""
    assert FP32_A6000_TOK_PER_S == 858.0


def test_train_max_hours_arithmetic_matches_the_formula_by_hand():
    """measured_h = tokens / tok_per_s / 3600; result = ceil((measured_h * margin + overhead_h) * 10) / 10."""
    tokens, tok_per_s, margin, overhead_h = 858 * 3600, 858.0, 1.5, 0.5   # exactly 1 measured hour
    assert train_max_hours(tokens, tok_per_s, margin, overhead_h, dev_tokens=0) == 2.0        # 1*1.5 + 0.5 = 2.0 exactly
    assert train_max_hours(0, tok_per_s, margin, overhead_h, dev_tokens=0) == 0.5             # no tokens: only the overhead
    assert train_max_hours(tokens, tok_per_s, margin, 0.0, dev_tokens=0) == 1.5


def test_train_max_hours_rounds_up_to_the_next_tenth_of_an_hour():
    """A job TIMEOUT a few seconds short of finishing wastes the whole run: the fuse must never be tighter than the
    arithmetic behind it, so any remainder rounds UP, never down or to the nearest tenth."""
    # 1 token at 858 tok/s * 1.5 margin + 0 overhead is a hair above 0.0; must round up to 0.1, never to 0.0
    assert train_max_hours(1, 858.0, 1.5, 0.0, dev_tokens=0) == 0.1
    # a value that lands just over a tenth boundary rounds up to the NEXT tenth, not down to it
    exact_tenth_tokens = round(0.1 / 1.5 * 858.0 * 3600)         # margin-adjusted hours land exactly on 0.1
    assert train_max_hours(exact_tenth_tokens + 1, 858.0, 1.5, 0.0, dev_tokens=0) > 0.1


def test_train_max_hours_defaults_are_the_documented_ones():
    """The function's own defaults (tok_per_s, margin, overhead_h) are the ones the docstring documents; a caller
    that only passes tokens gets them."""
    tokens = 500_000
    assert train_max_hours(tokens, dev_tokens=900) == train_max_hours(tokens, 858.0, 1.5, 0.5, dev_tokens=900,
                                                                  dev_forward_cost=1 / 3)


@pytest.mark.parametrize("kwargs, bad", [
    ({"tokens": -1}, "tokens"), ({"tokens": 1, "tok_per_s": 0}, "tok_per_s"),
    ({"tokens": 1, "tok_per_s": -1}, "tok_per_s"), ({"tokens": 1, "margin": 0}, "margin"),
    ({"tokens": 1, "margin": -1}, "margin"), ({"tokens": 1, "overhead_h": -1}, "overhead_h"),
    ({"tokens": 1, "dev_tokens": -1}, "dev_tokens"),
])
def test_train_max_hours_refuses_non_positive_inputs(kwargs, bad):
    with pytest.raises(ValueError, match=bad):
        train_max_hours(**{"dev_tokens": 0, **kwargs})


def test_train_max_hours_accepts_zero_overhead_and_zero_tokens():
    assert train_max_hours(0, overhead_h=0.0, dev_tokens=0) == 0.0


def test_cli_prints_the_same_number_the_function_returns():
    out = subprocess.run([sys.executable, "-m", "v2gpu.plan_time", "3000000", "--dev-tokens", "600000"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert float(out.stdout.strip()) == train_max_hours(3_000_000, dev_tokens=600_000)


def test_cli_accepts_the_override_flags():
    out = subprocess.run([sys.executable, "-m", "v2gpu.plan_time", "1000", "--tok-per-s", "100", "--margin", "2",
                          "--overhead-h", "1", "--dev-tokens", "300"], cwd=ROOT, capture_output=True, text=True,
                         check=True)
    assert float(out.stdout.strip()) == train_max_hours(1000, 100.0, 2.0, 1.0, dev_tokens=300)


# The epoch-end dev passes are planned work too


def test_dev_passes_are_added_forward_only_and_margin_free():
    """decision 28/09: the dev passes enter max_hours at DEV_FORWARD_COST (1/3) of a training token each, with NO
    margin; the margin stays on the training tokens only. 858*3600 tokens = 1 measured hour: 1 h x 1.5 = 1.5 h;
    858*3600*3 dev tokens x 1/3 = 1 h, unmargined; + 0.5 h overhead = 3.0 h."""
    one_hour = 858 * 3600
    assert train_max_hours(one_hour, 858.0, 1.5, 0.5, dev_tokens=3 * one_hour) == 3.0
    assert train_max_hours(one_hour, 858.0, 1.5, 0.5, dev_tokens=0) == 2.0


def test_the_dev_pilot_max_hours_is_20_6():
    """(39,833,334 / 858 / 3600) x 1.5 = 19.343 h; 6,823,705 / 3 / 858 / 3600 = 0.736 h (5 dev passes of 612 v0
    examples, 1,364,741 tokens each); + 0.5 h = 20.580 h, rounded UP to 20.6 (19.9 before the dev passes were
    counted)."""
    assert train_max_hours(39_833_334, dev_tokens=6_823_705) == 20.6
    assert train_max_hours(23_900_000, dev_tokens=3 * 1_364_741) == 12.6          # the 3-epoch expectation


def test_dev_tokens_is_required():
    """No silent default of 0: a plan that forgets the dev passes would leave them out again unnoticed."""
    with pytest.raises(TypeError):
        train_max_hours(1000)


def test_the_cli_requires_dev_tokens():
    out = subprocess.run([sys.executable, "-m", "v2gpu.plan_time", "1000"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 2 and "--dev-tokens" in out.stderr


def test_the_cli_prints_the_dev_pilot_max_hours():
    out = subprocess.run([sys.executable, "-m", "v2gpu.plan_time", "39833334", "--dev-tokens", "6823705"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "20.6"
