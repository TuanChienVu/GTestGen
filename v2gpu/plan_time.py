"""max_hours arithmetic for a training job's queue entry.

python -m v2gpu.plan_time TOKENS --dev-tokens DEV_TOKENS [--tok-per-s 858.0] [--margin 1.5] [--overhead-h 0.5]
    -> prints max_hours

The training procedure (ops/TRAIN_RUNSHEET.md) uses train_max_hours() to write each job's `max_hours` (the job
queue's per-job spending fuse, v2gpu/jobqueue.py: a job still running after max_hours is TIMEOUT-terminated).
"""
from __future__ import annotations

import argparse
import math

# The measured constant: A6000 fp32, pre-canary 27/09. bf16 measured 1190 tok/s on the same run, but fp32 upcast is
# now mandatory for every real job, so
# fp32's own speed is the only one a real job's max_hours may be planned from. After the first A100 job, recompute
# this constant from A100's own measured speed (ops/TRAIN_RUNSHEET.md notes this too).
FP32_A6000_TOK_PER_S = 858.0

# The cost of one dev-pass token (train_loop.dev_loss: one forward per
# example under no_grad, at the end of every epoch) as a fraction of one training token. A training token costs
# about three forwards -- the forward, the gradient-checkpoint recompute, and the backward through the frozen
# quantized base (activation gradients only, so about one forward) -- and tokens_per_second counts only training
# tokens. An estimate, not a measurement (the canaries ran MAX_STEPS and never reached an epoch end); 1/4 is
# another estimate, and 1/3 is the more conservative of the two. train_max_hours and v2gpu.train_probe both use it.
DEV_FORWARD_COST = 1 / 3


def train_max_hours(tokens: int, tok_per_s: float = FP32_A6000_TOK_PER_S, margin: float = 1.5,
                    overhead_h: float = 0.5, *, dev_tokens: int, dev_forward_cost: float = DEV_FORWARD_COST) -> float:
    """The `max_hours` a training job's queue entry declares: the measured duration of processing `tokens` tokens at
    `tok_per_s`, times a safety `margin`, plus a fixed per-job `overhead_h` -- model load and the checkpoint I/O
    (train_loop.LoopCfg.save_every steps, and one at every epoch end), which tok_per_s does not count (it times the
    training forward/backward over every input_ids token, prompt and answer, train_loop.py). Rounded UP to the next
    0.1 hour, so the declared fuse is never tighter than the arithmetic behind it (a job TIMEOUT a few seconds short
    of finishing wastes the whole run, v2gpu/jobqueue.py).

    Since 28/09, the epoch-end dev passes are no longer hidden in
    overhead_h but a term of their own: dev_tokens (the job's dev tokens over ALL its passes: tokens of every
    --dev-prompts example x the maximum number of epochs) x dev_forward_cost / tok_per_s, with NO margin -- the
    margin stays on the training tokens only. v2gpu.train_probe projects the same terms, margin-free, from the
    measured speed. dev_tokens has no default, so no plan can leave the dev passes out by accident. For the dev
    pilot: 39,833,334 / 858 / 3600 x 1.5 = 19.343 h, + 6,823,705 / 3 / 858 / 3600 = 0.736 h, + 0.5 h = 20.580 h,
    rounded up to 20.6 h (19.9 h before the dev passes were counted).

    tok_per_s default: FP32_A6000_TOK_PER_S (858.0), the one measured constant (see above). margin
    (1.5) and overhead_h (0.5h = 30 min) are estimates, not measured: margin follows the same
    "measured x margin" fuse convention genqueue's --max-hours/--recheck-max-hours already use; overhead_h is a
    conservative guess at fixed per-job cost for loading a QLoRA 7B model and a LoRA adapter's checkpoint I/O.
    Both are parameters precisely so they can be revised once the dev-pilot job's own receipt gives a real number
    (ops/TRAIN_RUNSHEET.md notes it too: recompute after the first A100 job with A100's measured speed)."""
    if tokens < 0:
        raise ValueError(f"train_max_hours: tokens must be >= 0, got {tokens}")
    if tok_per_s <= 0:
        raise ValueError(f"train_max_hours: tok_per_s must be > 0, got {tok_per_s}")
    if margin <= 0:
        raise ValueError(f"train_max_hours: margin must be > 0, got {margin}")
    if overhead_h < 0:
        raise ValueError(f"train_max_hours: overhead_h must be >= 0, got {overhead_h}")
    if dev_tokens < 0:
        raise ValueError(f"train_max_hours: dev_tokens must be >= 0, got {dev_tokens}")
    measured_h = tokens / tok_per_s / 3600
    dev_h = dev_tokens * dev_forward_cost / tok_per_s / 3600
    return math.ceil((measured_h * margin + dev_h + overhead_h) * 10) / 10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m v2gpu.plan_time", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokens", type=int, help="total training tokens the job is planned to process (all epochs)")
    ap.add_argument("--dev-tokens", type=int, required=True,
                    help="the job's dev tokens over all its dev passes (every --dev-prompts example x max epochs)")
    ap.add_argument("--tok-per-s", type=float, default=FP32_A6000_TOK_PER_S)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--overhead-h", type=float, default=0.5)
    a = ap.parse_args(argv)
    print(train_max_hours(a.tokens, a.tok_per_s, a.margin, a.overhead_h, dev_tokens=a.dev_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
