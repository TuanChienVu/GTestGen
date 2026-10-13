"""The QLoRA training loop of scenario A, written out instead of hidden in a Trainer so every rule is visible.

Pre-declared rules:
- micro-batch = one example (no padding); the TRAIN loss of a step is the mean over its `grad_accum` examples of each
  example's mean answer-token loss (every example weighs the same, whatever its length);
- constant LR after linear warm-up (the QLoRA recipe), so the pre-declared extra epochs need no new schedule; the
  warm-up is `warmup_ratio` of the planned optimizer steps, counted on the items rule D left in epoch 1;
- after each epoch: save `epoch-<e>/` and compute the DEV loss = total answer-token NLL / total answer tokens over every
  dev record in F0, F1 and F2 (token-weighted, all three formats); the per-format dev losses are recorded beside it;
- epochs 1..`epochs` always run; epoch e+1 (up to `epochs + max_extra_epochs`) runs only if the dev loss of epoch e is
  at least `extra_epoch_min_gain` (relative) below epoch e-1; the selected epoch is the argmin of dev loss. The rule
  is checked at the top of the loop, so resuming a finished run trains nothing more;
- state (adapter, optimizer, scheduler, RNG, position) is saved every `save_every` steps and at every epoch end by an
  atomic swap (write and fsync state.tmp, rename state -> state.old, rename state.tmp -> state, fsync the workdir);
  resume reads state, else a complete state.tmp, else state.old, and drops metric lines logged after the saved position;
  since 27/09, every snapshot also carries its own identity.json,
  written into state.tmp before pos.json (so it travels with the atomic swap, and with a rename to state.old too); a
  resume compares the CURRENT identity against the snapshot _load_state actually selects, before touching anything
  else in it, and refuses (IdentityMismatch) on a mismatch or a snapshot with no identity.json at all -- this is the
  primary defence against a resume silently carrying on under another seed, LR, prompt set or training code, for the
  case the receipt compare could not reach: a kill (SIGTERM/SIGKILL, a pod loss) that left state/ but no receipt;
- a (record, format) missing from `examples` was excluded by rule D (train_job): the rotation keeps its order, the item
  is skipped and counted in `skipped_item_epochs` (once each time the rotation reaches it, so 3 epochs count an
  excluded pair once and 5 epochs up to twice; train_job's `preflight.excluded_rule_d` counts the pairs themselves).
On CPU a resumed run is bit-identical to an uninterrupted one (tested); CUDA kernels are not bit-deterministic, so on
the card resume is checked functionally (it reaches the planned step count), not bit-for-bit.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .schedule import epoch_items
from .serialize import Encoded, encode


@dataclass(frozen=True)
class LoopCfg:
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    grad_accum: int = 16
    epochs: int = 3
    max_extra_epochs: int = 2
    extra_epoch_min_gain: float = 0.01
    max_seq: int = 8192
    save_every: int = 50
    grad_clip: float = 1.0
    max_steps: int | None = None  # canary only: stop after this many optimizer steps


def continue_after(dev: list[float], cfg: LoopCfg) -> bool:
    """Whether another epoch runs after the len(dev) epochs already done (pre-declared rule)."""
    done = len(dev)
    if done < cfg.epochs:
        return True
    if done >= cfg.epochs + cfg.max_extra_epochs:
        return False
    return dev[-1] < dev[-2] * (1 - cfg.extra_epoch_min_gain)


def select_epoch(dev: list[float]) -> int:
    """1-based epoch with the lowest dev loss; the earliest one on a tie."""
    return min(range(len(dev)), key=lambda i: (dev[i], i)) + 1


def dev_loss(model, dev_encs: dict[str, list[Encoded]], device) -> dict[str, float]:
    """{"all": token-weighted loss over every format, "F0": ..., "F1": ..., "F2": ...}."""
    import torch

    model.eval()
    sums = {}
    with torch.no_grad():
        for fmt, encs in sorted(dev_encs.items()):
            total, n = 0.0, 0
            for enc in encs:
                out = model(input_ids=torch.tensor([enc.input_ids], device=device),
                            labels=torch.tensor([enc.labels], device=device))
                total += out.loss.item() * enc.n_answer
                n += enc.n_answer
            sums[fmt] = (total, n)
    model.train()
    res = {fmt: t / n for fmt, (t, n) in sums.items()}
    res["all"] = sum(t for t, _ in sums.values()) / sum(n for _, n in sums.values())
    return res


def _rng_state(cuda: bool):
    import torch

    state = {"cpu": torch.get_rng_state()}
    if cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state, cuda: bool):
    import torch

    torch.set_rng_state(state["cpu"])
    if cuda and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    for p in sorted(path.rglob("*")):
        if p.is_file():
            with p.open("rb") as f:
                os.fsync(f.fileno())
    for d in sorted({p.parent for p in path.rglob("*")} | {path}):
        _fsync_dir(d)


def _save_state(workdir: Path, model, opt, sched, pos: dict, identity: dict, cuda: bool = False) -> None:
    import torch

    tmp, final, old = workdir / "state.tmp", workdir / "state", workdir / "state.old"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    # Identity travels with the snapshot, written before pos.json (which still marks it
    # complete); the whole tmp/ -> state swap below is what makes both of them atomic together.
    (tmp / "identity.json").write_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    model.save_pretrained(tmp / "adapter")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "rng": _rng_state(cuda)}, tmp / "trainer.pt")
    (tmp / "pos.json").write_text(json.dumps(pos, sort_keys=True))  # written last: marks tmp complete
    _fsync_tree(tmp)
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    _fsync_dir(workdir)
    shutil.rmtree(old, ignore_errors=True)


def state_dir(workdir: Path) -> Path | None:
    for name in ("state", "state.tmp", "state.old"):
        d = Path(workdir) / name
        if (d / "pos.json").exists():
            return d
    return None


class IdentityMismatch(Exception):
    """The snapshot _load_state selected has no identity.json, or its identity differs
    from the resume's own. `fields` is None for "no identity.json in the state" and otherwise a list of the dotted
    field names that differ (see identity_mismatches); train_job.py turns this into REFUSED_RESUME_IDENTITY."""

    def __init__(self, fields: list[str] | None):
        self.fields = fields
        super().__init__("no identity.json in the state" if fields is None else
                         "identity mismatch: " + ", ".join(fields))


def identity_mismatches(saved, current: dict) -> list[str]:
    """Dotted field names where `current` (model, seed, subset, gpu_name, data, train_config, train_code_sha256; the
    same identity the receipt compare always used, plus gpu_name) differs from `saved`: model,
    seed, subset, gpu_name and train_code_sha256 as whole values, data and train_config key by key.
    gpu_name (receipt.environment()["device"]["name"], None with no CUDA device, e.g. a CPU test) joins the
    resume identity so a resume on a different card class is refused like any other identity field
    (REFUSED_RESUME_IDENTITY) -- card class is chosen per seed (decision 27/09) and must never silently change
    mid-training."""
    saved = saved or {}
    out = [k for k in ("model", "seed", "subset", "gpu_name") if saved.get(k) != current.get(k)]
    for group in ("data", "train_config"):
        prev_group = saved.get(group) or {}
        out += [f"{group}.{k}" for k, v in (current.get(group) or {}).items() if prev_group.get(k) != v]
    if saved.get("train_code_sha256") != current.get("train_code_sha256"):
        out.append("train_code_sha256")
    return out


def _load_state(workdir: Path, model, opt, sched, identity: dict, cuda: bool = False) -> dict:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = state_dir(workdir)
    if state is None:
        raise FileNotFoundError(f"no complete state in {workdir}")
    # The selected snapshot's OWN identity.json is checked before anything else in it is
    # touched, so a mismatch (or a snapshot saved before this fix, with no identity.json at all) never loads any
    # adapter, optimizer or RNG state -- state/ stays byte-identical.
    try:
        saved = json.loads((state / "identity.json").read_text())
    except (OSError, ValueError):
        saved = None
    if not isinstance(saved, dict):
        raise IdentityMismatch(None)
    mismatches = identity_mismatches(saved, identity)
    if mismatches:
        raise IdentityMismatch(mismatches)
    set_peft_model_state_dict(model, load_file(state / "adapter" / "adapter_model.safetensors"))
    blob = torch.load(state / "trainer.pt", weights_only=True)
    opt.load_state_dict(blob["optimizer"])
    sched.load_state_dict(blob["scheduler"])
    _set_rng_state(blob["rng"], cuda)
    return json.loads((state / "pos.json").read_text())


def _truncate_metrics(path: Path, pos: dict) -> None:
    """Keep only metric lines up to the saved position (steps re-run after a crash are logged again).

    metrics.jsonl is flushed, not fsynced (the state is), so a host crash can tear its LAST line: an unparsable final
    line (cut mid-JSON, inside a multibyte character, or zero-filled) is dropped, not a crash of the resume run.
    A metric line is informational (epoch selection reads the state's dev losses),
    and its step is re-run and re-logged when it lies past the saved position. An unparsable line BEFORE the last is
    not a torn tail and still raises, naming the line. Lines split on "\n" only, as jsonl_lines does, each decoded as
    strict UTF-8; kept lines are written back byte for byte."""
    if not path.exists():
        return
    lines = path.read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    keep = []
    for n, raw in enumerate(lines, start=1):
        try:
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("not a JSON object")
        except ValueError as e:                    # UnicodeDecodeError is a ValueError
            if n == len(lines):
                break                              # the torn tail of a crash
            raise ValueError(f"{path}:{n}: unreadable metrics line before the last ({e})") from e
        if ("step" in row and row["step"] <= pos["global_step"]) or \
                ("epoch_end" in row and row["epoch_end"] < pos["epoch"]):
            keep.append(raw + b"\n")
    path.write_bytes(b"".join(keep))


def warmup_steps(n_items: int, cfg: LoopCfg) -> int:
    return max(1, round(cfg.warmup_ratio * math.ceil(n_items / cfg.grad_accum) * cfg.epochs))


def run(model, tok, cfg: LoopCfg, record_ids: list[str], seed: int, examples: dict,
        dev_encs: dict[str, list[Encoded]], workdir: Path, device, identity: dict, resume: bool = False) -> dict:
    """`identity` is saved into every snapshot and, on a resume, checked against the snapshot
    _load_state selects, before anything else in it is touched (see identity_mismatches, IdentityMismatch)."""
    import torch

    workdir = Path(workdir)
    cuda = torch.device(device).type == "cuda"   # the training device, not whether the host has a card
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    warmup = warmup_steps(sum((it.record_id, it.fmt) in examples for it in epoch_items(record_ids, seed, 1)), cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    pos = {"epoch": 1, "group": 0, "global_step": 0, "dev": [], "dev_by_format": [], "tokens": 0,
           "answer_tokens": 0, "train_seconds": 0.0, "skipped_item_epochs": 0}
    metrics_path = workdir / "metrics.jsonl"
    if resume:
        pos = _load_state(workdir, model, opt, sched, identity, cuda)
        _truncate_metrics(metrics_path, pos)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    status = "DONE"
    with metrics_path.open("a") as log:
        while True:
            if pos["group"] == 0 and pos["dev"] and not continue_after(pos["dev"], cfg):
                break
            e = pos["epoch"]
            rotation = epoch_items(record_ids, seed, e)
            items = [it for it in rotation if (it.record_id, it.fmt) in examples]
            if pos["group"] == 0:
                pos["skipped_item_epochs"] += len(rotation) - len(items)
            groups = [items[i:i + cfg.grad_accum] for i in range(0, len(items), cfg.grad_accum)]
            for gi in range(pos["group"], len(groups)):
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                step_loss = 0.0
                for it in groups[gi]:
                    ex = examples[(it.record_id, it.fmt)]
                    enc = encode(tok, ex.messages, ex.answer, cfg.max_seq)
                    loss = model(input_ids=torch.tensor([enc.input_ids], device=device),
                                 labels=torch.tensor([enc.labels], device=device)).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at epoch {e} record {it.record_id} {it.fmt}")
                    (loss / len(groups[gi])).backward()
                    step_loss += loss.item() / len(groups[gi])
                    pos["tokens"] += len(enc.input_ids)
                    pos["answer_tokens"] += enc.n_answer
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                pos["global_step"] += 1
                pos["group"] = gi + 1
                pos["train_seconds"] += time.time() - t0
                log.write(json.dumps({"step": pos["global_step"], "epoch": e, "loss": step_loss,
                                      "formats": dict(Counter(it.fmt for it in groups[gi])),
                                      "lr": sched.get_last_lr()[0], "tokens": pos["tokens"],
                                      "train_seconds": round(pos["train_seconds"], 3)}) + "\n")
                log.flush()
                if pos["global_step"] % cfg.save_every == 0:
                    _save_state(workdir, model, opt, sched, pos, identity, cuda)
                if cfg.max_steps is not None and pos["global_step"] >= cfg.max_steps:
                    _save_state(workdir, model, opt, sched, pos, identity, cuda)
                    status = "MAX_STEPS"
                    break
            if status == "MAX_STEPS":
                break
            model.save_pretrained(workdir / f"epoch-{e}")
            d = dev_loss(model, dev_encs, device)
            pos["dev"].append(d["all"])
            pos["dev_by_format"].append(d)
            log.write(json.dumps({"epoch_end": e, "dev_loss": d["all"], "dev_loss_by_format": d}) + "\n")
            log.flush()
            pos["epoch"], pos["group"] = e + 1, 0
            _save_state(workdir, model, opt, sched, pos, identity, cuda)
    secs = pos["train_seconds"]
    return {"status": status, "cfg": asdict(cfg), "global_step": pos["global_step"], "dev_loss": pos["dev"],
            "dev_loss_by_format": pos["dev_by_format"],
            "selected_epoch": select_epoch(pos["dev"]) if pos["dev"] else None, "tokens": pos["tokens"],
            "answer_tokens": pos["answer_tokens"], "skipped_item_epochs": pos["skipped_item_epochs"],
            "train_seconds": round(secs, 3),
            "tokens_per_second": round(pos["tokens"] / secs, 2) if secs else None,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if cuda else None}
