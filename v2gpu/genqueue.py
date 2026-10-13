"""Build the QUEUE.json of one generation pod, on the laptop, from finished and checked inputs:

python -m v2gpu.genqueue --variant main --model deepseek --weights base s42 s11 s23 s7 s99 --partitions test aux_test
    --prompts P/deepseek.all.test.F0.jsonl ... --prompts-sums P/PROMPTS_SHA256SUMS
    --leakcheck-receipt receipts/leakcheck-receipt.deepseek.json
    --train-receipts results/deepseek-s42-all/receipt-train.json ...   (one per adapter weights, in --weights order)
    --adapters results/deepseek-s42-all/epoch-3 ...                     (the selected adapter directories, same order)
    --pod-root /workspace --gpu-name "NVIDIA RTX A4000" --deadline-utc 2026-10-08T12:00:00+00:00 --max-hours 9.5
    --recheck-max-hours 1.5 --recheck-sample 5 --canary-verdict results/gencanary-deepseek-<a12>/verdict.json
    --decision results/decision.main.json [--canary-train-receipt R] [--copy-window-minutes 30]
    [--prepend-queue ship/QUEUE.xpod-....json] --out ship/QUEUE.gen-main-deepseek.json
python -m v2gpu.genqueue ... --recheck-only --originals results/gen-main-deepseek-base ...  (another pod's workdirs)

- --variant: any label runs.variant_spec resolves (main, dev_pilot_v0, dev_pilot_v1, the dev grid labels), never gen_canary.
- Every job's inputs go through genprep.preflight here first (the checks the pod repeats, including the leak-audit
  receipt's code_sha256 against this checkout); any problem refuses and writes nothing. A missing or unreadable train
  receipt or adapter is such a problem, not a traceback; the refusal shows the count and the first 20.
- A train receipt must be DONE, job mode, this model, this seed and pool "all"; its selected_adapter.tensor_sha256 must
  be the digest of the adapter directory that is shipped. That digest and the leak-audit receipt's sha256 go into the
  job commands, so the pod verifies exactly what the laptop checked (QUEUE.json is the jobs' input manifest).
- Each generation job is followed by a recheck job: a fresh process regenerates the pre-declared recheck pairs (N
  round-robin records, the longest prompt, up to two capped pairs; generate.select_pairs) and compares the lines byte
  for byte (--recheck-of, --sample N). --recheck-max-hours is required, derived from the canary speed like
  --max-hours (a recheck TIMEOUT blocks every later job of the pod).
- Cross-pod recheck (decision 24/09): --recheck-only --originals DIR ... queues ONLY recheck jobs
  (recheck-<variant>-<model>-<weights>-xpod) for workdirs finished on another pod; each original (DONE, outputs
  matching its receipt, this model, weights and variant) is shipped to runs/<its name> first. For every (model,
  variant main) one such recheck runs on a different pod from the original job; the dev pilot's is queued on the
  first main-run pod, before its own jobs. A mismatch means stop and report.
  The originals are keyed by the weights of their identity and shipped to
  runs/<their own name>; the identity core each recheck command will have on the pod (generate.build_parser and
  generate._identity over this laptop's preflight and prompt ids, then generate._core) must equal the original's
  job.json core, else the queue is refused here and names the differing keys (a regenerated leak receipt, another
  memory step or other generation code would be REFUSED_RECHECK on a paid pod). A normal queue carries such rechecks
  with --prepend-queue XPOD.json (a --recheck-only queue built for this pod root): its recheck jobs run first, then this
  queue's jobs, ONE copy window and ONE self-stop; its ship map is merged (ship_put). --originals alone belongs to
  --recheck-only (the originals of another model or variant have other prompts, adapters and verdicts).
  Every prepended command must name this queue's pin set and card (a pod has one
  venv-gen and one card; another pin set fails REFUSED_ENVIRONMENT at the head of the queue); otherwise the rechecks
  run as their own --recheck-only QUEUE on a pod built for them.
- Every adapter command carries --adapter-config-sha256, the sha256 of the adapter_config.json shipped
  (rank, alpha and targets decide the numbers with the tensors).
- --run-tag with --recheck-only is refused: the -xpod recheck workdirs keep their names; a lost pod's
  half-done recheck workdir is deleted before its volume is re-attached.
- Lost-pod recovery: --run-tag needs --results DIR, the workdirs copied home; a weights whose gen
  workdir (any tag) is DONE there is refused, since the renamed job would regenerate it and coverage would count its
  pairs twice; a DONE gen whose recheck was lost goes through --recheck-only --originals.
- One decision per variant: --decision, required, is the variant's decision file
  (python -m v2gpu.gencanary decide over a passing verdict of EVERY model the variant plans, all with one decision):
  passing, of this variant and its planned models, and equal to this model's trained verdict's decision; the trained
  verdict must also carry the decision's gen_code_sha256, declared engine and engine fingerprint (gencanary.BINDING,
  non-null) and, when the queue holds s42, that adapter's adapter_config.json sha256.
  The decision (schema v2) must also be bound to THIS checkout: its gen_code_sha256 and its
  declared engine (engines.declared_engine of generate.job_engine_settings at the decision's memory step: pinned
  kwargs, batch_invariant, env, tuned shrink files) are recomputed here and must be equal, so a route flipped after the
  canary is refused; every command carries --decision-sha256 (the decision file's sha256, part of the job identity).
- The canary decides, once for the whole run (decision 24/09): --canary-verdict, required for every real variant, is
  this model's passing gencanary verdict (every gate, hf_agreement included) of a TRAINED adapter,
  on this card; its decision (pin set, gpu_memory_utilization, enforce_eager) goes into every command (--pin-set,
  --gpu-memory-utilization, --enforce-eager), and gencoverage then refuses workdirs that differ. The canary's adapter
  digest must be this run's seed-42 adapter: the s42 train receipt of the queue, or
  --canary-train-receipt when the queue has no s42 (the dev pilot's own s42 for the pilot, the official s42 for main);
  a --canary-train-receipt selecting another digest than the queue's own s42 is refused. The
  memory step is one of engines.MEMORY_LADDER. For main, a canary probe must have read this prompt directory (verdict
  probed_sums holds its PROMPTS_SHA256SUMS digest): the test/aux probe.
- --partitions are exactly the partitions where the variant plans runs of the model (runs.variant_spec; DeepSeek
  main: test and aux_test), and --recheck-sample is at least 1.
- Prompt length (24/09): the longest prompt of the queue's prompt files (this model's tokenizer, on the
  laptop) must be no longer than the longest prompt the canary probed (criteria.probe.probed_prompt). The probe filled
  the window from that prompt, so every shorter prompt fits in memory; a longer directory (e.g. a dev grid) needs a new
  --probe-longest run on it and a new verdict.
- Re-decode (24/09): a pin set whose transformers/tokenizers differ from training's (fallback-tf,
  fallback-tf-bnb) needs criteria.redecode of the verdict to PASS for this A (of_outputs_sha256 = a_outputs_sha256),
  made with the training versions of transformers and tokenizers (gencanary on the laptop, training venv).
- Every command carries --ids-sha256, this laptop's (training venv's) digest of the job's prompt ids
  (genprep.tokenize): a generation venv that tokenizes differently is refused before any engine loads.
- The queue ends with a copy window (copy-window-<variant>-<model>, --copy-window-minutes, default 30): the pod stays
  up that long after the last job, so the operator's per-job copy loop (GEN_RUNSHEET.md section 7) fetches the last workdirs
  before the queue's self-stop (decision 24/09: stop is not delete; copy each workdir as soon as its job is DONE).
- Every command starts with `env HF_HUB_OFFLINE=1 HF_HOME=<venv root>/hf` (the bootstrap
  downloads the models into its HF_HOME, and an offline job looking in the user's default Hugging Face cache crashes before its
  workdir; since 05/10 that HF_HOME is $V2GPU_LOCAL/hf, beside the venvs, so it follows --venv-root). --run-tag TAG (lost-pod recovery on a re-attached volume) appends -TAG to every job
  name and workdir, so the lost pod's workdir and its <queue_dir>/<job>.status do not block the new queue. The prompt
  files must sit beside PROMPTS_SHA256SUMS:
  the pod checks them by their path relative to the sums, and they ship flat into one directory.
- Pod layout under --pod-root, content-addressed (a dev canary, the dev-pilot cross-pod recheck and the
  main queue meet on one pod, and flat targets collided): venv-gen/bin/python (ops/pod_bootstrap.sh with KIND=gen;
  under --venv-root when given: the container's own disk /opt/v2, never the pod volume),
  prompts/<sha256(PROMPTS_SHA256SUMS)[:16]>/<file>, receipts/<sha256(receipt)[:16]>/<file>,
  adapters/<model>-<weights>-<tensor digest[:12]>/, runs/<job>/, queue/. The queue runs from the repo checkout
  (python -m v2gpu.jobqueue QUEUE.json); "ship" lists what to copy where before it starts: the job's prompt files,
  PROMPTS_SHA256SUMS, PROMPTS_META.json, excluded.jsonl and pools.json (nothing else of the prompt directory), the
  leak-audit receipt, the adapters, and the originals of cross-pod rechecks. A ship map that would send two different
  sources to one target is refused.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from . import engines, generate
from .engines import (GEN_PINS, MEMORY_LADDER, PIN_SETS, card_problems, declared_engine, memory_step, names_engine,
                      tokenizes_like_training)
from .gencanary import BINDING, DECISION_SCHEMA, longest_prompt, planned_models
from .genprep import adapter_compose_problems, dev_pilot_compose, preflight, tokenize
from .models import MODELS, load_tokenizer
from .receipt import sha256_file, tensor_digest
from .runs import variant_arg, variant_spec

LOCK = "/root/v2gpu-queue.lock"
SELF_STOP = ["runpodctl", "stop", "pod", "$RUNPOD_POD_ID"]
SHIPPED_BESIDE = ("PROMPTS_META.json", "excluded.jsonl", "pools.json")   # what genprep reads beside the sums


def refusal_message(problems: list[str], shown: int = 20) -> str:
    """The count and the first `shown` problems: a long list stays readable, and nothing is hidden."""
    more = len(problems) - shown
    return (f"REFUSED ({len(problems)} problem{'' if len(problems) == 1 else 's'}): " + "; ".join(problems[:shown]) +
            (f" (+{more} more)" if more > 0 else ""))


def train_receipt_problems(path: Path, model: str, weights: str,
                           adapter_dir: Path) -> tuple[list[str], str | None, str | None, str | None]:
    """A missing or unreadable receipt or adapter is a problem, as in canary_digest. It
    also returns train_code_sha256 and train_cfg_sha256 (as a canonical json string, for hashing wholesale),
    so the caller can require one value of each across every adapter of a variant."""
    try:
        r = json.loads(Path(path).read_text())
        if not isinstance(r, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        return [f"{path}: no readable train receipt ({e})"], None, None, None
    out = []
    if r.get("status") != "DONE" or r.get("mode") != "job":
        out.append(f"{path}: status {r.get('status')!r} in mode {r.get('mode')!r}, a DONE job is needed")
    if (r.get("model") or {}).get("key") != model or f"s{r.get('seed')}" != weights or r.get("subset") != "all":
        out.append(f"{path}: model {(r.get('model') or {}).get('key')!r} seed {r.get('seed')!r} subset "
                   f"{r.get('subset')!r} is not {model} {weights} all")
    digest = (r.get("selected_adapter") or {}).get("tensor_sha256")
    if digest is None:
        out.append(f"{path}: no selected_adapter.tensor_sha256")
    else:
        try:
            got = tensor_digest(Path(adapter_dir) / "adapter_model.safetensors")
        except Exception as e:  # noqa: BLE001 - a missing or unreadable adapter is a refusal, not a traceback
            out.append(f"{adapter_dir}: no readable adapter_model.safetensors ({e})")
        else:
            if got != digest:
                out.append(f"{adapter_dir}: tensor digest differs from the selected adapter of {path}")
    train_code = r.get("train_code_sha256")
    if not isinstance(train_code, dict) or not train_code:
        out.append(f"{path}: no train_code_sha256")
        train_code_key = None
    else:
        train_code_key = json.dumps(train_code, sort_keys=True)
    train_cfg = r.get("train_cfg_sha256")
    if not isinstance(train_cfg, str) or not train_cfg:
        out.append(f"{path}: no train_cfg_sha256")
        train_cfg = None
    # Decision 27/09: fp32 upcast is mandatory for every real training job; an adapter
    # whose train receipt says otherwise (or omits train_config) never ships, whatever else about it checks out.
    if (r.get("train_config") or {}).get("upcast_fp32") is not True:
        out.append(f"{path}: train_config.upcast_fp32 is not true: every real adapter must "
                   f"be trained with --upcast-fp32")
    return out, digest, train_code_key, train_cfg


def train_code_consistency_problems(codes: dict[str, str | None], cfgs: dict[str, str | None]) -> list[str]:
    """Every adapter of a variant (one genqueue call, one --variant) must
    share one train_code_sha256 and one train_cfg_sha256; a record that lacks either was already refused by
    train_receipt_problems, so here only the cross-adapter agreement is checked."""
    out = []
    for label, values in (("train_code_sha256", codes), ("train_cfg_sha256", cfgs)):
        present = {w: v for w, v in values.items() if v is not None}
        if len(set(present.values())) > 1:
            out.append(f"{label} differs across adapters of this variant: " +
                      "; ".join(f"{w}={v}" for w, v in sorted(present.items())))
    return out


def train_compose(path: Path):
    """The compose the adapter was trained on: data.prompts.meta.compose of its train receipt (None if unreadable)."""
    try:
        return (((_receipt(path).get("data") or {}).get("prompts") or {}).get("meta") or {}).get("compose")
    except (OSError, ValueError, AttributeError):
        return None


def ship_put(ship: dict, section: str, target: str, source) -> list[str]:
    """Add target <- source to ship[section]; a second, different file for one target is a problem, the same file
    again (another spelling of its path) is not."""
    old = ship[section].get(target)
    if old is not None and Path(old).resolve() != Path(source).resolve():
        return [f"ship target {target} would receive two sources: {old} and {source}"]
    ship[section].setdefault(target, str(source))
    return []


def _receipt(path: Path) -> dict:
    r = json.loads(Path(path).read_text())
    if not isinstance(r, dict):
        raise ValueError("not a JSON object")
    return r


def canary_digest(path: Path | None, trained: dict, model: str) -> tuple[list[str], str | None]:
    """The seed-42 adapter digest the canary must have run: the queue's own s42 when it has one (an explicit
    --canary-train-receipt must then select the same digest), else --canary-train-receipt."""
    own = trained.get("s42")
    if path is None:
        if own is None:
            return ["--canary-train-receipt is required: the queue has no s42 adapter, and the canary ran the model's "
                    "seed-42 adapter"], None
        path = own[0]
    try:
        r = _receipt(path)
    except (OSError, ValueError) as e:
        return [f"{path}: no readable train receipt ({e})"], None
    if r.get("status") != "DONE" or r.get("mode") != "job" or (r.get("model") or {}).get("key") != model or \
            r.get("seed") != 42 or r.get("subset") != "all":
        return [f"{path}: not a DONE job-mode {model} seed-42 train receipt of pool all"], None
    digest = (r.get("selected_adapter") or {}).get("tensor_sha256")
    if own is not None and Path(path).resolve() != Path(own[0]).resolve():
        try:
            mine = (_receipt(own[0]).get("selected_adapter") or {}).get("tensor_sha256")
        except (OSError, ValueError):
            mine = None
        if mine is not None and mine != digest:     # (none: train_receipt_problems refuses the s42 receipt itself)
            return [f"--canary-train-receipt {path} selects adapter {str(digest)[:12]}, this queue's s42 "
                    f"({own[0]}) {str(mine)[:12]}: the canary must have run the queue's own s42"], None
    return [], digest


def verdict_problems(path: Path, model: str, gpu_name: str, variant: str, sums_sha256: str,
                     adapter_sha256: str | None, longest: int | None) -> tuple[list[str], dict]:
    """The canary verdict the queue's settings come from: passing (hf_agreement too), trained adapter of this run's
    seed 42, this model and card; for main, a probe that read this prompt directory; a probed prompt at least as long
    as the queue's longest; for a tf fallback set, a passing re-decode in the training venv."""
    try:
        v = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return [f"{path}: no readable canary verdict ({e})"], {}
    d = v.get("decision") if isinstance(v, dict) else None
    if not isinstance(d, dict):
        return [f"{path}: no canary decision in the verdict (rerun v2gpu.gencanary)"], {}
    out = []
    if v.get("pass") is not True:
        out.append(f"{path}: not a passing canary verdict")
    if v.get("trained") is not True:
        out.append(f"{path}: the canary ran without --trained; rerun it with the trained adapter before real jobs")
    if v.get("model") != model:
        out.append(f"{path}: canary of {v.get('model')!r}, not {model}")
    if d.get("gpu_name") != gpu_name:
        out.append(f"{path}: canary gpu_name {d.get('gpu_name')!r}, not {gpu_name!r}")
    if d.get("pin_set") not in PIN_SETS:
        out.append(f"{path}: pin set {d.get('pin_set')!r} is not one of {sorted(PIN_SETS)}")
    if memory_step(d) is None:
        out.append(f"{path}: gpu_memory_utilization {d.get('gpu_memory_utilization')!r} / enforce_eager "
                   f"{d.get('enforce_eager')!r} are not a pre-declared memory step of {MEMORY_LADDER}")
    elif engines.BATCH_INVARIANT and MEMORY_LADDER[memory_step(d)] != engines.DETERMINISM_STEP:
        out.append(f"{path}: memory step {MEMORY_LADDER[memory_step(d)]} refused: decision 05/10, every job runs at "
                   f"{engines.DETERMINISM_STEP} (gpu_memory_utilization, enforce_eager) while batch invariance is on")
    if ((v.get("criteria") or {}).get("hf_agreement") or {}).get("pass") is not True:
        out.append(f"{path}: the canary's hf_agreement gate did not pass (required on every model)")
    if adapter_sha256 is not None and v.get("adapter_tensor_sha256") != adapter_sha256:
        out.append(f"{path}: the canary ran adapter {str(v.get('adapter_tensor_sha256'))[:12]}, not this run's seed-42 "
                   f"adapter {adapter_sha256[:12]}")
    if variant == "main" and sums_sha256 not in (v.get("probed_sums") or []):
        out.append(f"{path}: no canary probe read this prompt directory (PROMPTS_SHA256SUMS {sums_sha256[:12]}): "
                   f"main needs the test/aux probe")
    probed = ((v.get("criteria") or {}).get("probe") or {}).get("probed_prompt")
    if longest is not None and (not isinstance(probed, int) or longest > probed):
        out.append(f"{path}: the longest prompt of this queue has {longest} tokens, the canary probed {probed}: run a "
                   f"new --probe-longest on this prompt directory and a new verdict first")
    if d.get("pin_set") in PIN_SETS and not tokenizes_like_training(d["pin_set"]):
        rd = (v.get("criteria") or {}).get("redecode") or {}
        training = {k: GEN_PINS[k] for k in ("transformers", "tokenizers")}
        if rd.get("pass") is not True or rd.get("of_outputs_sha256") != v.get("a_outputs_sha256") or \
                rd.get("versions") != training:
            out.append(f"{path}: pin set {d['pin_set']} tokenizes with other versions than training: the verdict needs "
                       f"a passing re-decode of this canary run A made with {training} (gencanary on the laptop)")
    return out, d


def trained_binding_problems(verdict_path: Path, decision_path: Path, config_sha: str | None) -> list[str]:
    """The TRAINED verdict ran the code and the engine the variant's decision binds (the decision comes from untrained
    canaries, the trained one runs later): every key of gencanary.BINDING equal and
    non-null, the engine fingerprint naming its engine (engines.names_engine); and it ran the
    adapter config of this queue's s42 (config_sha, when the queue holds s42)."""
    try:
        v, d = _receipt(verdict_path), _receipt(decision_path)
    except (OSError, ValueError):
        return []                          # verdict_problems and decision_file_problems report the unreadable file
    out = [f"{verdict_path}: the trained canary's {k} is not the decision's ({decision_path}): rerun the trained "
           f"canary with this checkout and the variant's engine" for k in BINDING if v.get(k) is None or
           v.get(k) != d.get(k) or (k == "engine_fingerprint" and not names_engine(v.get(k)))]
    if config_sha is not None and v.get("adapter_config_sha256") != config_sha:
        out.append(f"{verdict_path}: the canary ran adapter_config.json {str(v.get('adapter_config_sha256'))[:12]}, "
                   f"not this queue's s42 {config_sha[:12]}")
    return out


def checkout_binding(model: str, decision: dict) -> dict:
    """What a decision binds, as this checkout would run it: the generation code and the declared
    engine at the decision's memory step ({} for a decision off the ladder: verdict_problems refuses it)."""
    if memory_step(decision or {}) is None:
        return {}
    return {"gen_code_sha256": generate.gen_code_sha256(),
            "engine_settings": declared_engine(generate.job_engine_settings(
                model, None, MODELS[model].window, decision["gpu_memory_utilization"], decision["enforce_eager"],
                "vllm"))}


def decision_file_problems(path: Path, variant: str, model: str, verdict_decision: dict) -> list[str]:
    """The variant's one decision (gencanary decide): passing, of this variant, taken on the
    verdicts of every model it plans, and equal to this model's trained canary decision; made by this checkout's
    generation code and declared engine (a route flipped after the canary is refused)."""
    try:
        d = _receipt(path)
    except (OSError, ValueError) as e:
        return [f"{path}: no readable decision file ({e}); run python -m v2gpu.gencanary decide"]
    if d.get("schema") != DECISION_SCHEMA or d.get("pass") is not True or not isinstance(d.get("decision"), dict):
        return [f"{path}: not a passing {DECISION_SCHEMA} decision (python -m v2gpu.gencanary decide)"]
    out = []
    if d.get("variant") != variant:
        out.append(f"{path}: the decision of variant {d.get('variant')!r}, not {variant}")
    got, planned = sorted(d.get("verdicts") or {}), planned_models(variant)
    if got != planned:
        out.append(f"{path}: decided on the verdicts of {got}; {variant} plans {planned}")
    if verdict_decision and verdict_decision != d["decision"]:
        out.append(f"{path}: the variant's decision {json.dumps(d['decision'], sort_keys=True)} is not this model's "
                   f"trained canary decision {json.dumps(verdict_decision, sort_keys=True)}: rerun the trained canary "
                   f"at the variant's decision")
    for k, here in checkout_binding(model, d["decision"]).items():
        theirs = d.get(k) if isinstance(d.get(k), dict) else {}
        if d.get(k) != here:
            keys = sorted(x for x in set(here) | set(theirs) if theirs.get(x) != here.get(x))
            out.append(f"{path}: the canaries ran another {k} than this checkout ({', '.join(keys)}): the decision "
                       f"binds the code and the engine; rerun the canaries (and decide) with this checkout")
    return out


def prepend_problems(path: Path, queue_dir: str) -> tuple[list[str], dict]:
    """A --recheck-only QUEUE.json of cross-pod rechecks for this pod: its recheck jobs run first,
    then this queue's jobs, one copy window and one self-stop."""
    try:
        q = _receipt(path)
    except (OSError, ValueError) as e:
        return [f"--prepend-queue {path}: no readable QUEUE.json ({e})"], {}
    jobs = q.get("jobs") if isinstance(q.get("jobs"), list) else []
    names = [j.get("name") if isinstance(j, dict) else None for j in jobs]
    if len(names) < 2 or not all(isinstance(n, str) and n.endswith("-xpod") for n in names) or \
            not all(n.startswith("recheck-") for n in names[:-1]) or not names[-1].startswith("copy-window-"):
        return [f"--prepend-queue {path} is not a --recheck-only queue (recheck-...-xpod jobs, then its copy "
                f"window)"], {}
    if (q.get("queue_dir"), q.get("lock_path"), q.get("self_stop_cmd")) != (queue_dir, LOCK, SELF_STOP):
        return [f"--prepend-queue {path} was built for another pod root, lock or self-stop"], {}
    if not isinstance(q.get("ship"), dict) or not all(isinstance(v, dict) for v in q["ship"].values()):
        return [f"--prepend-queue {path} has no ship map"], {}
    return [], q


def prepend_settings_problems(q: dict, pin_set: str, gpu_name: str) -> list[str]:
    """The prepended rechecks run on this pod's one venv-gen and card: every command
    must name this queue's pin set and card, or its first job fails REFUSED_ENVIRONMENT and blocks the queue."""
    out = []
    for j in q["jobs"]:
        cmd = j.get("cmd") or []
        for flag, want in (("--pin-set", pin_set), ("--gpu-name", gpu_name)):
            if flag in cmd and cmd[cmd.index(flag) + 1] != want:
                out.append(f"--prepend-queue job {j.get('name')}: {flag} {cmd[cmd.index(flag) + 1]}, this queue runs "
                           f"{want}: run it as its own --recheck-only QUEUE on a pod built for it")
    return out


def done_at_home(results: Path, variant: str, model: str, weights) -> list[str]:
    """Weights whose gen workdir (any run tag) is DONE among the workdirs copied home: a run-tag
    queue would regenerate them under a new name, and coverage would count their pairs twice."""
    out = []
    for w in weights:
        stem = f"gen-{variant}-{model}-{w}"
        for d in sorted(Path(results).glob(f"{stem}*")) if Path(results).is_dir() else []:
            if not (d.name == stem or re.fullmatch(re.escape(stem) + r"-[a-z0-9]{1,8}", d.name)):
                continue
            try:
                done = json.loads((d / "receipt-generate.json").read_text()).get("status") == "DONE"
            except (OSError, ValueError, AttributeError):
                done = False
            if done:
                out.append(f"{w}: {d} is DONE at home: leave it out of the recovery queue (a DONE gen whose recheck "
                           f"was lost goes through --recheck-only --originals)")
    return out


def recheck_core_problems(original: Path, cmd: list[str], prep, ids_sha256: str) -> list[str]:
    """The identity core the recheck command will have on the pod (generate's own parser and _identity, with this
    laptop's preflight and prompt ids) against the original's job.json: a difference is refused
    here, not by REFUSED_RECHECK on a paid pod."""
    ns = generate.build_parser().parse_args(cmd[cmd.index("v2gpu.generate") + 1:])
    engine = ns.engine or ("hf" if ns.mode == "test" else "vllm")
    mine = generate._core(generate._identity(ns, engine, prep, ids_sha256,
                                             generate._window(MODELS[ns.model], ns.model_path)))
    try:
        theirs = generate._core(_receipt(Path(original) / "job.json"))
    except (OSError, ValueError) as e:
        return [f"{Path(original).name}: no readable job.json ({e})"]
    differ = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    return [f"{Path(original).name}: the recheck's job identity differs from the original's job.json in {differ}; "
            f"the pod would refuse it (REFUSED_RECHECK)"] if differ else []


def original_problems(workdir: Path, model: str, variant: str) -> tuple[list[str], str | None]:
    """A workdir finished on another pod, to be rechecked here: DONE, outputs matching its receipt, this job; and the
    weights its identity names."""
    w = Path(workdir)
    try:
        rc = _receipt(w / "receipt-generate.json")
    except (OSError, ValueError):
        return [f"{w}: no readable receipt-generate.json"], None
    ident = rc.get("identity") if isinstance(rc.get("identity"), dict) else {}
    out = []
    if rc.get("status") != "DONE" or not (w / "outputs.jsonl").exists() or \
            rc.get("outputs_sha256") != sha256_file(w / "outputs.jsonl"):
        out.append(f"{w.name}: not DONE with outputs matching its receipt")
    if ident.get("model") != model or ident.get("variant") != variant or ident.get("recheck_of"):
        out.append(f"{w.name}: not a {variant} gen workdir of {model}")
    if not (w / "job.json").exists():
        out.append(f"{w.name}: no job.json")
    return out, ident.get("weights")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m v2gpu.genqueue", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, type=variant_arg, help="a label runs.variant_spec resolves, "
                                                                        "not gen_canary")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--partitions", nargs="+", required=True)
    ap.add_argument("--prompts", type=Path, nargs="+", required=True)
    ap.add_argument("--prompts-sums", type=Path, required=True)
    ap.add_argument("--leakcheck-receipt", type=Path, required=True)
    ap.add_argument("--train-receipts", type=Path, nargs="*", default=[])
    ap.add_argument("--adapters", type=Path, nargs="*", default=[])
    ap.add_argument("--pod-root", required=True)
    ap.add_argument("--venv-root", help="where venv-gen lives on the pod (default: --pod-root); the "
                                        "container's own disk, /opt/v2, since a MooseFS /workspace imports torch in "
                                        "about a minute")
    ap.add_argument("--gpu-name", required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--max-hours", type=float, required=True, help="each generation job (measured x margin)")
    ap.add_argument("--recheck-max-hours", type=float, required=True, help="each recheck job (canary speed x margin)")
    ap.add_argument("--recheck-sample", type=int, default=5, help="N round-robin records of each recheck (>= 1)")
    ap.add_argument("--recheck-only", action="store_true", help="only recheck jobs, of --originals from another pod")
    ap.add_argument("--originals", type=Path, nargs="*", default=[], help="finished gen workdirs copied home")
    ap.add_argument("--prepend-queue", type=Path, help="a --recheck-only QUEUE.json whose cross-pod rechecks run "
                                                       "first on this pod (one copy window, one self-stop)")
    ap.add_argument("--decision", type=Path, required=True, help="the variant's decision (gencanary decide)")
    ap.add_argument("--canary-verdict", type=Path, required=True, help="the model's passing, trained canary verdict")
    ap.add_argument("--canary-train-receipt", type=Path, help="train receipt of the s42 adapter the canary ran "
                                                              "(default: the queue's s42)")
    ap.add_argument("--dev-pilot-meta", type=Path, help="a second-round dev grid variant only: the "
                                                        "round's OWN dev_pilot_vN PROMPTS_META.json, to check that "
                                                        "the adapter was trained on it (compose, every key)")
    ap.add_argument("--copy-window-minutes", type=int, default=30, help="pod time after the last job for the copy loop")
    ap.add_argument("--run-tag", help="lost-pod recovery: -TAG on every job name and workdir (1-8 of a-z0-9)")
    ap.add_argument("--results", type=Path, help="with --run-tag (required then): the workdirs copied home; weights "
                                                 "whose gen workdir is DONE there are refused")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    problems = [] if a.variant != "gen_canary" else ["gen_canary is the canary's variant: no queue of real jobs"]
    root = a.pod_root.rstrip("/")
    venv_root = (a.venv_root or a.pod_root).rstrip("/")
    if dt.datetime.fromisoformat(a.deadline_utc).tzinfo is None:
        problems.append(f"--deadline-utc {a.deadline_utc} has no timezone")
    if a.recheck_sample < 1:
        problems.append(f"--recheck-sample {a.recheck_sample}: at least 1 (the recheck's round-robin records)")
    problems += [f"--gpu-name: {p}" for p in card_problems(a.gpu_name)]
    if a.run_tag is not None and not re.fullmatch(r"[a-z0-9]{1,8}", a.run_tag):
        problems.append(f"--run-tag {a.run_tag!r}: 1 to 8 characters of a-z and 0-9")
    tag = f"-{a.run_tag}" if a.run_tag else ""
    if a.run_tag is not None and a.recheck_only:        # the -xpod recheck workdirs are never tagged
        problems.append("--run-tag is never used with --recheck-only: a lost pod's half-done -xpod recheck workdir is "
                        "deleted before its volume is re-attached, then the same recheck-only QUEUE is rebuilt")
    if a.run_tag is not None and a.results is None:
        problems.append("--run-tag needs --results DIR (the workdirs copied home): a recovery queue lists only the "
                        "weights not DONE there")
    elif a.run_tag is not None:
        problems += done_at_home(a.results, a.variant, a.model, a.weights)
    planned_parts = sorted(p for p, (models, _) in variant_spec(a.variant).items() if a.model in models)
    if sorted(a.partitions) != planned_parts:
        problems.append(f"--partitions {' '.join(a.partitions)}: {a.variant} plans runs of {a.model} on exactly "
                        f"{' '.join(planned_parts)}")
    if a.originals and not a.recheck_only:
        problems.append("--originals needs --recheck-only; to run cross-pod rechecks on this pod before its own jobs, "
                        "build their --recheck-only QUEUE.json and pass it here as --prepend-queue")
    prepended = {}
    if a.prepend_queue is not None:
        if a.recheck_only:
            problems.append("--prepend-queue goes on a normal queue, not on a --recheck-only one")
        found, prepended = prepend_problems(a.prepend_queue, f"{root}/queue")
        problems += found
    adapter_weights = [w for w in a.weights if w != "base"]
    counted = len(adapter_weights) == len(a.train_receipts) == len(a.adapters)
    if not counted:
        problems.append(f"{len(adapter_weights)} adapter weights need as many --train-receipts and --adapters "
                        f"(got {len(a.train_receipts)} and {len(a.adapters)})")
    sums_dir = a.prompts_sums.resolve().parent
    problems += [f"{p}: not beside {a.prompts_sums.name} (the pod checks the flat layout)" for p in a.prompts
                 if p.resolve().parent != sums_dir]
    originals, ship = {}, {"prompts": {}, "receipts": {}, "adapters": {}, "originals": {}}
    for o in a.originals:                 # keyed by the weights of their identity, shipped to runs/<their name>
        found, w = original_problems(o, a.model, a.variant)
        problems += found + ship_put(ship, "originals", f"{root}/runs/{o.name}", o)
        if w in originals and originals[w].resolve() != o.resolve():
            problems.append(f"two --originals of weights {w}: {originals[w]} and {o}")
        originals.setdefault(w, o)
    if a.recheck_only and sorted(originals, key=str) != sorted(a.weights):
        problems.append(f"--recheck-only needs one finished --originals workdir of {a.variant} {a.model} per --weights "
                        f"{' '.join(a.weights)}; got weights {sorted(map(str, originals))}")
    trained = dict(zip(adapter_weights, zip(a.train_receipts, a.adapters, strict=True), strict=True)) if counted else {}
    found, canary_sha = canary_digest(a.canary_train_receipt, trained, a.model)
    problems += found
    try:
        longest = longest_prompt(a.prompts, a.model)[0]
    except (OSError, ValueError) as e:
        longest = None
        problems.append(f"prompt files: {e}")
    found, decision = verdict_problems(a.canary_verdict, a.model, a.gpu_name, a.variant, sha256_file(a.prompts_sums),
                                       canary_sha, longest)
    problems += found + decision_file_problems(a.decision, a.variant, a.model, decision)
    s42_config = None if "s42" not in trained or not (Path(trained["s42"][1]) / "adapter_config.json").is_file() \
        else sha256_file(Path(trained["s42"][1]) / "adapter_config.json")
    problems += trained_binding_problems(a.canary_verdict, a.decision, s42_config)
    if prepended and isinstance(decision, dict):
        problems += prepend_settings_problems(prepended, decision.get("pin_set"), a.gpu_name)
    if a.copy_window_minutes < 1:
        problems.append("--copy-window-minutes must be at least 1")
    if problems:
        print(refusal_message(problems), file=sys.stderr)
        return 2
    receipt_sha = sha256_file(a.leakcheck_receipt)
    prompt_dir = f"{root}/prompts/{sha256_file(a.prompts_sums)[:16]}"
    receipt = f"{root}/receipts/{receipt_sha[:16]}/{a.leakcheck_receipt.name}"
    pod_prompts = [f"{prompt_dir}/{p.name}" for p in a.prompts]
    tok = load_tokenizer(MODELS[a.model])
    suffix = "-xpod" if a.recheck_only else ""
    settings = ["--pin-set", decision["pin_set"], "--gpu-memory-utilization", str(decision["gpu_memory_utilization"]),
                *(["--enforce-eager"] if decision["enforce_eager"] else [])]
    beside = [a.prompts_sums.parent / n for n in SHIPPED_BESIDE]
    jobs = []
    for p in [*a.prompts, a.prompts_sums, *beside]:
        problems += ship_put(ship, "prompts", f"{prompt_dir}/{p.name}", p)
    problems += ship_put(ship, "receipts", receipt, a.leakcheck_receipt)
    pilot_compose = None                       # the round's OWN dev_pilot_vN META compose
    if a.dev_pilot_meta is not None:
        pilot_compose = dev_pilot_compose(a.dev_pilot_meta)
        if pilot_compose is None:
            problems.append(f"--dev-pilot-meta {a.dev_pilot_meta}: no readable PROMPTS_META.json compose")
    train_codes, train_cfgs = {}, {}   # one train_code_sha256, one train_cfg_sha256 per variant
    for w in a.weights:
        digest = adapter = None
        if w != "base":
            found, digest, train_codes[w], train_cfgs[w] = train_receipt_problems(trained[w][0], a.model, w,
                                                                                  trained[w][1])
            problems += found
            adapter = trained[w][1]
        config_sha = None if adapter is None or not (Path(adapter) / "adapter_config.json").is_file() else \
            sha256_file(Path(adapter) / "adapter_config.json")
        found, prep = preflight(a.model, w, a.variant, a.partitions, a.prompts, a.prompts_sums, "all",
                                a.leakcheck_receipt, receipt_sha, adapter, digest, config_sha)
        token_problems, _, ids_sha256 = tokenize(tok, prep.pairs, MODELS[a.model].window)
        problems += [f"{w}: {p}" for p in found + token_problems]
        if w != "base":                   # the adapter's compose; base runs need only (a)
            problems += [f"{w}: {trained[w][0]}: {p}" for p in adapter_compose_problems(
                a.variant, prep.prompts_meta["meta"], train_compose(trained[w][0]), pilot_compose)]
        name = f"gen-{a.variant}-{a.model}-{w}{tag}"
        workdir = f"{root}/runs/{originals[w].name if a.recheck_only else name}"
        cmd = ["env", "HF_HUB_OFFLINE=1", f"HF_HOME={venv_root}/hf", f"{venv_root}/venv-gen/bin/python", "-m",
               "v2gpu.generate", "--model", a.model,
               "--weights", w,
               "--variant", a.variant, "--partitions", *a.partitions, "--prompts", *pod_prompts,
               "--prompts-sums", f"{prompt_dir}/{a.prompts_sums.name}", "--leakcheck-receipt", receipt,
               "--leakcheck-receipt-sha256", receipt_sha, "--gpu-name", a.gpu_name, *settings,
               "--decision-sha256", sha256_file(a.decision),
               "--ids-sha256", ids_sha256]
        if w != "base":
            target = f"{root}/adapters/{a.model}-{w}-{str(digest)[:12]}"
            cmd += ["--adapter", target, "--adapter-sha256", digest,
                    *(["--adapter-config-sha256", config_sha] if config_sha else [])]
            problems += ship_put(ship, "adapters", target, adapter)
        if not a.recheck_only:
            jobs.append({"name": name, "workdir": workdir, "max_hours": a.max_hours,
                         "cmd": cmd + ["--workdir", workdir]})
        check = f"{root}/runs/recheck-{a.variant}-{a.model}-{w}{tag}{suffix}"
        recheck = cmd + ["--workdir", check, "--recheck-of", workdir, "--sample", str(a.recheck_sample)]
        if a.recheck_only:
            problems += recheck_core_problems(originals[w], recheck, prep, ids_sha256)
        jobs.append({"name": f"recheck-{a.variant}-{a.model}-{w}{tag}{suffix}", "workdir": check,
                     "max_hours": a.recheck_max_hours, "cmd": recheck})
    problems += train_code_consistency_problems(train_codes, train_cfgs)
    window = f"{root}/runs/copy-window-{a.variant}-{a.model}{tag}{suffix}"
    jobs.append({"name": f"copy-window-{a.variant}-{a.model}{tag}{suffix}", "workdir": window,
                 "max_hours": round(a.copy_window_minutes / 60 + 0.1, 3),
                 "cmd": ["bash", "-c", f"mkdir -p {window} && sleep {a.copy_window_minutes * 60}"]})
    if prepended:                         # its rechecks first; its copy window dropped: one window, one self-stop
        jobs = prepended["jobs"][:-1] + jobs
        for section, items in sorted(prepended["ship"].items()):
            ship.setdefault(section, {})
            for target, source in sorted(items.items()):
                problems += ship_put(ship, section, target, source)
        names = [j["name"] for j in jobs]
        problems += [f"job name {n} twice in the queue" for n in sorted(set(names)) if names.count(n) > 1]
        theirs = {j["workdir"] for j in prepended["jobs"]} | set(prepended["ship"].get("originals", {}))
        problems += [f"workdir {w} is both a prepended recheck's (or its original's) and a job's of this queue"
                     for w in sorted(theirs & {j["workdir"] for j in jobs[len(prepended["jobs"]) - 1:]})]
    if problems:
        print(refusal_message(problems), file=sys.stderr)
        return 2
    queue = {"queue_dir": f"{root}/queue", "deadline_utc": a.deadline_utc, "lock_path": LOCK,
             "self_stop_cmd": SELF_STOP, "jobs": jobs, "ship": ship}
    a.out.write_text(json.dumps(queue, indent=2, sort_keys=True))
    print(json.dumps({"jobs": [j["name"] for j in jobs], "leakcheck_receipt_sha256": receipt_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
