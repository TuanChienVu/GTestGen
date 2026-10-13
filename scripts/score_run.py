"""Score one partition. Three modes:
--controls   score gold, stub-v1/v2/v3/v3c/v3m/v3all, swap, the helper controls and the D11 controls
             (stub-v3names, stub-v3top4) for every record; write controls.jsonl (with n_calls and call_precision,
             D11), control_outputs.jsonl (the
             control texts, for score_independent), exclusions.jsonl (the mechanical rule: gold fails BV, its HEAD
             target file does not
             parse, or CA is undefined; with excl_bv and excl_ca), ca_ref.jsonl (the CA of stub-v3m, stub-v3all and
             stub-v3names per record; v3names None for Ruby), CONTROLS.json (per repo; D11: ca_prec@τ),
             CA_PREC_EXCLUDED.json (D11: gold or gold-helpers rows with CA 1 and CA_prec@0.25 = 0) and
             OPEN_RESOLUTION.json.
--outputs    score generation JSONL rows (the generation run's format: record_id, model, seed, condition, variant, text,
             prompt_tokens, new_tokens, max_new_tokens, stopped_at_eos, prompts_sha256, prompt_sha256, ...) into
             scores.jsonl, one row per (record, run), and OPEN_RESOLUTION.json; needs --controls-dir
             (--self-contained and --subsets have the sealed files as defaults); ER columns come from --er-results, and every (record, run) of an
             --er-manifest must have its er_results row; an ER run is read only when the er_complete.json next to its
             er_results.jsonl has complete true, is not running and names a given manifest (manifest_sha256), every
             er_results row carries its manifest line's overlay_sha256, and a flagged model output that passes has a
             tamper verdict (the er_results contract 0c42865). --outputs needs --prompts-meta: the 3A
             files that decide V(r) here (state.py, compose.py) must be those of the prompts' v2tools_commit, read with
             git in --a3-repo, symcache.py one of the versions of A3_ALLOWED, and PROMPTS_META symtimes_sha256 the
             sha256 of --symtimes. The generation contract (fixed 26/09/2026; `read_plan`):
             --manifest and --coverage are required unless every row is a probe (model "probe"); each scored row must
             be an `ok` manifest line (key, raw-line sha256, sha fields), every `ok` line of a (model, variant) in the
             outputs and of a record of the partition must have its row, and `missing` lines need coverage
             status_except_missing PASS and --accepted-missing (+ --accepted-missing-commit) listing exactly them, each
             with evidence (EVIDENCE_STATUSES). A row of a known record of the partition's OTHER test/aux_test
             partner (skipped_other_partition; one generation workdir holds both) is
             skipped and counted; a row of an unknown record, or of a KNOWN record of any OTHER partition (a train or
             dev record scored under test/aux_test, or the reverse: a test/aux_test record scored under dev), is
             refused, never skipped (TEST_AUX_PARTITIONS). Score rows copy decision_sha256, gen_code_sha256 and
             line_sha256 from the manifest (None for a probe row, which has no manifest
             line); SCORES_META gains scores_sha256, the sha256 of the scores.jsonl this run wrote.
--dose-table write dose.jsonl, one row {record_id, model, dose} per (model, record) of the partition: describe.dose of
             the record, prepared as every mode prepares it, and the model's F2 prompt (messages[0].content of its
             --prompts-f2 line), i.e. the `dose` describe.row writes in that model's scores rows. Run once the official
             prompts are built and before the SAP lock, which checks every score row's dose against it.
             Needs --prompts-f2 and --compose-excluded; a record of the partition with no F2 line for a
             model, unless compose's excluded.jsonl drops it for that model, stops the run (see `dose_rows`).
Fail closed: a missing input file, field or er_results row, an output of a record outside the partition, or the same
(record, run) twice stops the run with exit 3. So do a malformed input (not JSON, a JSONL line that is not an object,
an output field of the wrong type, a record with no
label or source), a run with no mode, an --out that already holds a run (SCORES_META.json), --prompts-f2 lines that
are not this partition's F2 lines of that model, and with --require-clean a tree that is not a clean git commit. SCORES_META.json records the sha256 of every input, the tool
commit, whether v2tools/ or scripts/ had uncommitted changes, a digest of the scoring code (code_sha256: an outputs run
refuses a --controls-dir scored with other code, in or out of git) and the sha256 of the 3A files; every
file written is listed in <out>/SHA256SUMS."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2tools import symcache  # noqa: E402
from v2tools.compose import COMMENT, FENCE, Composer  # noqa: E402
from v2tools.controls import controls, dev_prior, donors, shown4  # noqa: E402
from v2tools.describe import (CA_PREC_TAUS, InputError, ca_prec, ca_prec_excluded, call_overlap,  # noqa: E402
                               dose, need, open_resolution_rates, row)
from v2tools.evaluate import CONDITIONS, Repo, ca, future_methods, prepare, run_name, score_output, suite_index  # noqa: E402
from v2tools.extract import extract  # noqa: E402
from v2tools.ertree import parser, phantoms  # noqa: E402
from v2tools.state import RepoContext  # noqa: E402


OUTPUT_FIELDS = ("record_id", "model", "seed", "condition", "variant", "text", "prompt_tokens", "new_tokens",
                 "max_new_tokens", "stopped_at_eos", "prompts_sha256", "prompt_sha256")
OUTPUT_TYPES = {"text": str, "prompt_tokens": int, "new_tokens": int, "max_new_tokens": int,
                "stopped_at_eos": bool}  # what the scoring computes with
BV_EXCLUDED = ("gold_fails_bv", "head_unparsable")  # §2.2: the reasons that also leave a record out of BV
A3_FILES = ("v2tools/state.py", "v2tools/compose.py", "v2tools/records.py", "v2tools/outline.py", "v2tools/inherit.py",
            "v2tools/symcache.py", "v2tools/symtime.py")  # the 3A code that decides V(r) on the scoring side
# The official v0 prompts (PROMPTS_SHA256SUMS 3d306db2…, bound to the
# pod) were built at 37ed1a2, before two later 3A commits (8d232db, c6b16c6) added symcache.jsonl_rows and made
# load call it, the file's only change (the six other 3A files are equal at both commits). symcache.py
# is checked against these versions by sha256, never "any commit"; the data it reads is tied by
# PROMPTS_META.symtimes_sha256.
A3_ALLOWED = {"v2tools/symcache.py": frozenset({
    "c21ee44837a0479aef92914663536020f7013d71dc07a8dc5b9935808b3e35ed",  # 37ed1a2 (3A: the v0 prompts)
    "aa30b8cdba11ad0cfca2e340dab0cc5ebdbb6469782e51299b2ede68f6950dc7",  # c6b16c6 (the last 3A commit)
    "f1c41842ac05c37fb17e55d18ecfac14823c045fc77eab743ad3e62470e54402"})}  # public tree (c6b16c6, docstrings only)
ER_CONTROLS = ("neg", "gold", "stub")
PARTITIONS = ("train", "dev", "test", "aux_test")
TEST_AUX_PARTITIONS = frozenset({"test", "aux_test"})  # the only two partitions that
# ever share a generation workdir (the generation run's genqueue); a row of any other partition pairing is refused, not
# skipped
CA_REF_CONTROLS = {"ca_ref_v3m": "stub-v3m", "ca_ref_v3all": "stub-v3all", "ca_ref_v3names": "stub-v3names",
                   "ca_ref_top4": "stub-v3top4"}  # ca_ref.jsonl (describe.CA_REFS): None when the record lacks it
# the generation run's manifest (contract fixed 26/09/2026)
PROBE_MODEL = "probe"  # the pre-registered probes (scripts/probe_outputs.py): no manifest needed, never in the SAP
PLAN_KEY = ("model", "condition", "seed", "variant", "record_id", "fmt")
PLAN_SHAS = ("prompts_sha256", "prompt_sha256", "adapter_tensor_sha256", "decision_sha256", "gen_code_sha256",
             "leakcheck_receipt_sha256")
PLAN_FIELDS = PLAN_KEY + PLAN_SHAS + ("workdir", "receipt_sha256", "outputs_sha256", "line_sha256", "status",
                                     "attempts")
CONDITION_FMT = {"zero_shot": "F0", "ft_only": "F0", "rag_only": "F1", "full_neutral": "F1", "full_aware": "F2"}
EVIDENCE_STATUSES = frozenset({"CRASHED", "INCOMPLETE"})  # 26/09/2026: the last receipt statuses that are evidence
EVIDENCE_MIN_ATTEMPTS = 2  # ... over at least this many attempts, each with its receipt's sha256


def load(p: Path) -> list[dict]:
    if not p.exists():
        raise InputError(f"missing input {p}")
    # 3A's JSONL reader: "\n" only, from the bytes, as the generation run's jsonl_lines; U+2028, U+0085
    # or a lone "\r" of an output text (ensure_ascii=False) stays in it; a blank line raises
    try:
        rows = symcache.jsonl_rows(p)
    except ValueError as e:
        raise InputError(f"{p}: not JSONL ({e})") from e
    bad = next((i for i, x in enumerate(rows, 1) if not isinstance(x, dict)), None)  # a refusal
    if bad is not None:
        raise InputError(f"{p}: line {bad} is not a JSON object")
    return rows


def read_json(p: Path):
    if not p.exists():
        raise InputError(f"missing input {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:  # not JSON (or not UTF-8) is a refusal naming the file
        raise InputError(f"{p}: not JSON ({e})") from e


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_sha256() -> str:
    """Digest of the scoring code (every v2tools/*.py and scripts/*.py under ROOT, by relative path), so that a controls
    run and an outputs run are tied to the same code even outside a git checkout."""
    h = hashlib.sha256()
    for p in sorted([*ROOT.glob("v2tools/*.py"), *ROOT.glob("scripts/*.py")]):
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def tool_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "v2tools", "scripts"], cwd=ROOT, capture_output=True, text=True)
    return {"tool_commit": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip()) if head.returncode == 0 else None,
            "code_sha256": code_sha256(), "a3_sha256": {f: sha256(ROOT / f) for f in A3_FILES}}


def a3_errors(repo: Path, commit) -> list[str]:
    """Why the scoring side would not read V(r) as the prompts did: the 3A files here must be byte for byte
    those of the commit that built the prompts (PROMPTS_META.json `v2tools_commit`), read with git in `repo` (a checkout
    of that history, by default ROOT); symcache.py, there and here, must each be a version A3_ALLOWED names. No commit,
    or a file git cannot show, is an error too."""
    if not isinstance(commit, str) or not commit:
        return ["PROMPTS_META.json has no v2tools_commit"]
    out = []
    for f in A3_FILES:
        shown = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{f}"], capture_output=True)
        if shown.returncode != 0:
            out.append(f"git cannot show {f} at {commit} in {repo}")
        elif f in A3_ALLOWED:  # the prompts' version and this one, each an allowed version
            out += [f"{f} {where} is not an allowed version (A3_ALLOWED)" for where, digest in
                    ((f"at {commit}", hashlib.sha256(shown.stdout).hexdigest()), ("here", sha256(ROOT / f)))
                    if digest not in A3_ALLOWED[f]]
        elif hashlib.sha256(shown.stdout).hexdigest() != sha256(ROOT / f):
            out.append(f"{f} differs from the one at {commit} that built the prompts")
    return out


def unreviewed(x: dict) -> bool:
    """The er.report rule (0c42865): a flagged model output (not a control) that PASSes or
    whose trace matches must carry a manual tamper verdict ("ok" or "cheat") before its results are read."""
    s = x.get("pre_tamper") or x
    return (x["condition"] not in ER_CONTROLS and bool(x.get("tamper_flags"))
            and (s.get("er") == 1 or s.get("trace_action") == 1 or s.get("trace_then") == 1)
            and x.get("tamper_verdict") not in ("ok", "cheat"))


TAMPER_VERDICTS = ("ok", "cheat")  # er.tamper's manual verdicts


def verdicts_mismatch(f: Path, rows: list[dict]) -> str | None:
    """As er.report.verdicts_mismatch: the rows of an ER run carry
    exactly the verdicts of the CURRENT er_tamper_verdicts.json of their run (next to er_results.jsonl): a row whose key
    (record_id\tcondition\toverlay_sha256) the file judges "ok" or "cheat" has that tamper_verdict and the file's
    sha256 as tamper_verdicts_sha256; any other row has neither. None when they do (or no file and no verdict)."""
    vpath = f.parent / "er_tamper_verdicts.json"
    sha, verdicts = None, {}
    if vpath.exists():
        data = vpath.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        try:
            verdicts = json.loads(data)
        except ValueError as e:
            return f"{vpath}: not valid JSON ({e})"
        if not isinstance(verdicts, dict):
            return f"{vpath}: not a JSON object"
    k = lambda r: f"{r['record_id']}\t{r['condition']}\t{r.get('overlay_sha256')}"
    bad = [r for r in rows if (r.get("tamper_verdict"), r.get("tamper_verdicts_sha256")) !=
           ((verdicts[k(r)], sha) if verdicts.get(k(r)) in TAMPER_VERDICTS else (None, None))]
    if not bad:
        return None
    return (f"{len(bad)} er_results rows do not carry the tamper verdicts of the current {vpath} "
            f"({'the file is missing' if sha is None else 'edited after tamper apply, or applied from another file'}; "
            f"e.g. {k(bad[0])!r})")


def future_rows(preps: dict) -> list[dict]:
    """The FUTURE methods (§2.2b) of each record's target files ("origin:name", sorted), a property of the
    record (V(r) at t_vis, the full t_view before any budget cut), never of a format, model or condition. The controls
    run writes them; an outputs run recomputes them and must find the same sets."""
    return [{"record_id": rid, "path": path, "future": sorted({f"{f.origin}:{f.name}" for f in fut.values()})}
            for rid, p in preps.items() for path, fut in future_methods(p).items()]


def ca_refs(res: dict, ca_of) -> dict:
    """The ca_ref columns of one record (ca_ref.jsonl, describe.CA_REFS): each the CA (`ca_of(score)`) of its own
    control in `res` (controls(): name -> (score, fallback, blocks)), None when the record lacks it (Ruby)."""
    return {col: ca_of(res[name][0]) if name in res else None for col, name in CA_REF_CONTROLS.items()}


def controls_summary(rows: list[dict]) -> dict:
    """CONTROLS.json: per repo and ALL, per control, the counts n, bv, ca, ca_defined, fallback (records with a
    fallback function) and, D11 (c), ca_prec@τ for each τ of CA_PREC_TAUS: the sum of CA × [call_precision ≥ τ] (a
    record without CA adds 0, like .ca)."""
    summary: dict = defaultdict(Counter)
    for x in rows:
        name = x["control"]
        for rep in (x["repo"], "ALL"):
            c = summary[rep]
            c[f"{name}.n"] += 1
            c[f"{name}.bv"] += x["bv"]
            c[f"{name}.ca"] += x["ca"]
            c[f"{name}.ca_defined"] += int(x["ca_defined"])
            c[f"{name}.fallback"] += x["fallback"] > 0
            for tau in CA_PREC_TAUS:
                c[f"{name}.ca_prec@{tau}"] += ca_prec(x["ca"], x["call_precision"], tau)
    return summary


def plan_key(x: dict) -> tuple:
    return tuple(x[k] for k in PLAN_KEY)


def evidence(line: dict) -> bool:
    """The evidence rule (fixed 26/09/2026) for an accepted `missing` manifest line: >= EVIDENCE_MIN_ATTEMPTS attempts
    (the generation run's chronological `attempts`), each with a sha256, at least that many with a status (a receipt
    that parsed), the last one's status in EVIDENCE_STATUSES. REFUSED_*, RECHECK_MISMATCH and an absent job
    (attempts []) never count."""
    att = line.get("attempts")
    return (isinstance(att, list) and len(att) >= EVIDENCE_MIN_ATTEMPTS
            # as the generation run's evidence_eligible, attempts that read as a
            # receipt count
            and sum(isinstance(a, dict) and isinstance(a.get("status"), str) for a in att) >= EVIDENCE_MIN_ATTEMPTS
            and all(isinstance(a, dict) and isinstance(a.get("sha256"), str) and a["sha256"] for a in att)
            and att[-1].get("status") in EVIDENCE_STATUSES)


def line_hashes(path: Path) -> list[str]:
    """sha256 of the exact raw bytes of each line of a JSONL file: its bytes split on b"\\n", no trailing "\\n" (the
    manifest's line_sha256)."""
    parts = path.read_bytes().split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return [hashlib.sha256(x).hexdigest() for x in parts]


def read_plan(args, inputs: list) -> tuple[dict | None, dict]:
    """(key -> manifest line, or None without --manifest; SCORES_META fields) of the generation run's
    manifest (contract fixed 26/09/2026). Refused (InputError): --manifest without --coverage or the reverse;
    --accepted-missing without --accepted-missing-commit or the reverse, or without --manifest; a line without a field
    of PLAN_FIELDS (plus `reason` when excluded, `receipt_status` when missing); a key twice; lines not sorted by
    (model, condition, seed, record_id, fmt); a condition whose fmt is not CONDITION_FMT's; a status not
    ok/excluded/missing; an `ok` line without line_sha256 or another line with one; attempts not null on ok/excluded,
    not a list on missing; a missing line whose receipt_status/receipt_sha256 are not its last attempt's ("absent"/null
    when there is none); an evidence_eligible that disagrees with `evidence`; coverage.manifest_sha256 not the
    manifest's sha256; coverage status PASS with missing lines; status not PASS and status_except_missing not PASS;
    missing lines without --accepted-missing; an accepted entry (PLAN_KEY, reason, receipt_sha256) twice, of a line that
    is not missing, or with another receipt_sha256; a missing line no entry lists, or that is not `evidence`."""
    meta = {"manifest_sha256": None, "coverage_sha256": None, "accepted_missing_sha256": None,
            "accepted_missing_commit": args.accepted_missing_commit}
    if (args.manifest is None) != (args.coverage is None):
        raise InputError("--manifest and --coverage go together (manifest.<variant>.jsonl, coverage.<variant>.json)")
    if (args.accepted_missing is None) != (args.accepted_missing_commit is None):
        raise InputError("--accepted-missing and --accepted-missing-commit go together")
    if args.manifest is None:
        if args.accepted_missing is not None:
            raise InputError("--accepted-missing needs --manifest")
        return None, meta
    lines, cov = load(args.manifest), read_json(args.coverage)
    inputs += [args.manifest, args.coverage]
    meta.update(manifest_sha256=sha256(args.manifest), coverage_sha256=sha256(args.coverage))
    if not isinstance(cov, dict) or cov.get("manifest_sha256") != meta["manifest_sha256"]:
        raise InputError(f"{args.coverage}: manifest_sha256 is not the sha256 of {args.manifest}")
    plan, order = {}, []
    for x in lines:
        need(x, PLAN_FIELDS, f"manifest line in {args.manifest}")
        k, st = plan_key(x), x["status"]
        if k in plan:
            raise InputError(f"{args.manifest}: key {k} twice")
        if CONDITION_FMT.get(x["condition"]) != x["fmt"]:
            raise InputError(f"{args.manifest}: {x['condition']} with fmt {x['fmt']} "
                             f"(want {CONDITION_FMT.get(x['condition'])})")
        if st not in ("ok", "excluded", "missing"):
            raise InputError(f"{args.manifest}: {k} has status {st!r}")
        if (st == "ok") != (isinstance(x["line_sha256"], str) and bool(x["line_sha256"])) or (
                st != "ok" and x["line_sha256"] is not None):
            raise InputError(f"{args.manifest}: {k} ({st}) has line_sha256 {x['line_sha256']!r}")
        if st == "excluded":
            need(x, ("reason",), f"excluded manifest line {k}")
        if st != "missing" and x["attempts"] is not None:
            raise InputError(f"{args.manifest}: {k} ({st}) has attempts")
        if st == "missing":
            need(x, ("receipt_status",), f"missing manifest line {k}")
            att = x["attempts"]
            if not isinstance(att, list) or not all(isinstance(a, dict) for a in att):
                raise InputError(f"{args.manifest}: {k} (missing) has no attempts list")
            last = (att[-1].get("status"), att[-1].get("sha256")) if att else ("absent", None)
            if (x["receipt_status"], x["receipt_sha256"]) != last:
                raise InputError(f"{args.manifest}: {k}: receipt_status/receipt_sha256 are not the last attempt's")
        if "evidence_eligible" in x and x["evidence_eligible"] != (st == "missing" and evidence(x)):
            raise InputError(f"{args.manifest}: {k}: evidence_eligible {x['evidence_eligible']} disagrees with "
                             "the attempts")
        plan[k] = x
        order.append((x["model"], x["condition"], x["seed"], x["record_id"], x["fmt"]))
    if order != sorted(order):
        raise InputError(f"{args.manifest}: lines not sorted by (model, condition, seed, record_id, fmt)")
    missing = {k for k, x in plan.items() if x["status"] == "missing"}
    if cov.get("status") == "PASS":
        if missing:
            raise InputError(f"{args.coverage}: status PASS, but the manifest has {len(missing)} missing lines")
    elif cov.get("status_except_missing") != "PASS":
        raise InputError(f"{args.coverage}: status {cov.get('status')} and status_except_missing "
                         f"{cov.get('status_except_missing')}: the generation is not complete")
    elif args.accepted_missing is None:
        raise InputError(f"{args.manifest}: {len(missing)} missing lines and no --accepted-missing")
    if args.accepted_missing is not None:
        got = {}
        for e in load(args.accepted_missing):
            need(e, PLAN_KEY + ("reason", "receipt_sha256"), f"entry of {args.accepted_missing}")
            k = plan_key(e)
            if k in got:
                raise InputError(f"{args.accepted_missing}: {k} twice")
            if k not in missing:
                raise InputError(f"{args.accepted_missing}: {k} is not missing in {args.manifest}")
            if e["receipt_sha256"] != plan[k]["receipt_sha256"]:
                raise InputError(f"{args.accepted_missing}: {k}: receipt_sha256 is not the manifest line's")
            got[k] = e
        if set(got) != missing:
            raise InputError(f"{args.accepted_missing}: {len(missing - set(got))} missing lines are not listed, "
                             f"e.g. {sorted(missing - set(got))[0]}")
        weak = sorted(k for k in missing if not evidence(plan[k]))
        if weak:
            raise InputError(f"{args.manifest}: {len(weak)} missing lines have no evidence (status in "
                             f"{sorted(EVIDENCE_STATUSES)}, >= {EVIDENCE_MIN_ATTEMPTS} attempts with sha256), "
                             f"e.g. {weak[0]}: {plan[weak[0]]['receipt_status']}")
        inputs.append(args.accepted_missing)
        meta["accepted_missing_sha256"] = sha256(args.accepted_missing)
    return plan, meta


def key(x: dict) -> tuple:
    return x["repo"], x["feature_path"], x["scenario_line"], x["suite"]


def as_text(language: str, blocks: dict[str, str]) -> str:
    ticks = "`" * 3
    return "\n\n".join(f"{ticks}{FENCE[language]}\n{COMMENT[language]} FILE: {p}\n{c}\n{ticks}" for p, c in blocks.items())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def clean_errors(state: dict) -> list[str]:
    """--require-clean: a scoring run of record is made by a committed tree, tool_commit set and nothing uncommitted
    under v2tools/ or scripts/ (tool_state)."""
    if not state.get("tool_commit"):
        return ["--require-clean: the scoring tree is not a git checkout (no tool commit)"]
    return ["--require-clean: v2tools/ or scripts/ has uncommitted changes"] if state.get("dirty") is not False else []


def f2_prompts(args, spec: str, preps: dict) -> tuple[str, Path, dict[str, str]]:
    """(model, file, {record_id: F2 prompt text}) of one `--prompts-f2 <model>=<file>`, checked as the dose table
    checks it, in both modes: every line
    has record_id, partition, fmt, model and messages, is this partition's F2 line of that model, of a record of the
    partition, once."""
    model, sep, name = spec.partition("=")
    if not sep or not model or not name:
        raise InputError(f"--prompts-f2 {spec!r}: need <model>=<file>, each model once")
    path, texts = Path(name), {}
    for x in load(path):
        need(x, ("record_id", "partition", "fmt", "model", "messages"), f"F2 prompt line in {path}")
        rid = x["record_id"]
        if (x["partition"], x["fmt"], x["model"]) != (args.partition, "F2", model):
            raise InputError(f"{path}: {rid} is a {x['partition']}/{x['fmt']}/{x['model']} line, not "
                             f"{args.partition}/F2/{model}")
        if rid not in preps:
            raise InputError(f"{path}: record {rid} is not in partition {args.partition}")
        if rid in texts:
            raise InputError(f"{path}: record {rid} twice")
        texts[rid] = x["messages"][0]["content"]
    return model, path, texts


def dose_rows(args, preps: dict, inputs: list) -> list[dict]:
    """--dose-table: {record_id, model, dose} for every record of the partition and every
    `--prompts-f2 <model>=<file>`, with describe.dose(p, messages[0].content), as describe.row computes a scores row's
    dose. Refused (InputError): no --prompts-f2 or --compose-excluded; a model given twice; a line without record_id,
    partition, fmt, model or messages; a line whose partition, fmt ("F2") or model is not this run's; a line of a record
    outside the partition, or of a record compose's excluded.jsonl drops for that model (model null or that model, fmt
    null or "F2"); the same record twice; a record of the partition with no line and not dropped."""
    if not args.prompts_f2:
        raise InputError("--dose-table needs --prompts-f2 <model>=<F2 prompt JSONL of this partition>")
    if args.compose_excluded is None:
        raise InputError("--dose-table needs --compose-excluded (compose's excluded.jsonl of those prompts)")
    dropped = load(args.compose_excluded)
    inputs.append(args.compose_excluded)
    out, models = [], set()
    for spec in args.prompts_f2:
        model, path, texts = f2_prompts(args, spec, preps)
        if model in models:
            raise InputError(f"--prompts-f2 {spec!r}: need <model>=<file>, each model once")
        models.add(model)
        gone = {x["record_id"] for x in dropped if x.get("model") in (None, model) and x.get("fmt") in (None, "F2")}
        for rid in texts:
            if rid in gone:
                raise InputError(f"{path}: record {rid} has a line, but compose's excluded.jsonl drops it for {model}")
        missing = [rid for rid in preps if rid not in texts and rid not in gone]
        if missing:
            raise InputError(f"{path}: {len(missing)} records of {args.partition} have no F2 line, e.g. {missing[0]}")
        inputs.append(path)
        out += [{"record_id": rid, "model": model, "dose": dose(p, texts[rid])}
                for rid, p in preps.items() if rid in texts]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--full-root", type=Path, default=Path("repos-full"))
    ap.add_argument("--symtimes", type=Path, default=Path("out/phase3/symtimes.jsonl"))
    ap.add_argument("--self-contained", type=Path, default=Path("out/phase3/self_contained.jsonl"))
    ap.add_argument("--subsets", type=Path, default=Path("out/phase2/sensitivity_subsets.json"))
    ap.add_argument("--partition", required=True, choices=("test", "dev", "aux_test"))
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--outputs", type=Path, nargs="*", default=[])
    ap.add_argument("--controls-dir", type=Path, help="the --controls output of the same partition")
    ap.add_argument("--er-results", type=Path, nargs="*", default=[])
    ap.add_argument("--prompts-f2", nargs="*", default=[], help="<model>=<F2 prompt JSONL of this partition>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--er-manifest", type=Path, nargs="*", default=[], help="manifest.jsonl of the ER runs (write_overlays)")
    ap.add_argument("--prompts-meta", type=Path, help="PROMPTS_META.json of the prompts that made the outputs (--outputs)")
    ap.add_argument("--a3-repo", type=Path, default=ROOT, help="git checkout holding the prompts' v2tools_commit")
    ap.add_argument("--dose-table", action="store_true", help="write dose.jsonl from --prompts-f2 (before the lock)")
    ap.add_argument("--compose-excluded", type=Path, help="compose's excluded.jsonl of the --prompts-f2 files")
    ap.add_argument("--manifest", type=Path, help="the generation run's manifest.<variant>.jsonl of the outputs")
    ap.add_argument("--coverage", type=Path, help="the generation run's coverage.<variant>.json of that manifest")
    ap.add_argument("--accepted-missing", type=Path, help="the committed accepted-missing JSONL")
    ap.add_argument("--accepted-missing-commit", help="the git commit of --accepted-missing (recorded; sap_run checks)")
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse a tree that is not a clean git commit")
    args = ap.parse_args()
    try:
        return run(args)
    except (InputError, json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:  # a refusal, not a traceback
        print(f"refusing: {e if isinstance(e, InputError) else f'{type(e).__name__}: {e}'}", file=sys.stderr)
        return 3


def run(args) -> int:
    if not (args.controls or args.outputs or args.dose_table):  # never an empty META and exit 0
        raise InputError("nothing to do: give --controls, --outputs <files> or --dose-table")
    if (args.out / "SCORES_META.json").exists():  # a sealed run is never written over
        raise InputError(f"{args.out} already holds a scoring run (SCORES_META.json): give a new --out")
    if args.require_clean and clean_errors(tool_state()):
        raise InputError("; ".join(clean_errors(tool_state())))
    inputs = [args.phase2 / "records" / f"{args.partition}.jsonl", args.phase2 / "labels.jsonl", args.phase2 / "catalog.jsonl",
              args.phase2 / "forbidden.json", args.pairs, args.config, args.symtimes]
    records = [r for r in load(args.phase2 / "records" / f"{args.partition}.jsonl")]
    labels = {key(l): l for l in load(args.phase2 / "labels.jsonl")}
    pairs = {key(p): p for p in load(args.pairs)}
    cat = defaultdict(dict)
    for f in load(args.phase2 / "catalog.jsonl"):
        cat[f["repo"]][f["fn_id"]] = f
    forbidden = read_json(args.phase2 / "forbidden.json")
    sources = {s["repo"]: s for s in read_json(args.config)}
    if not args.symtimes.exists():
        raise InputError(f"missing input {args.symtimes}")
    sym = symcache.load(args.symtimes)
    suites = suite_index(list(labels.values()), pairs)
    repos = {}
    for r in records:
        if r["repo"] not in repos:
            if r["repo"] not in sources:  # a refusal, not a KeyError
                raise InputError(f"{r['id']}: its repo {r['repo']} is not in {args.config}")
            s = sources[r["repo"]]
            ctx = RepoContext(r["repo"], s["language"], args.full_root / f"{s['dir']}.git", r["commit"], cat[r["repo"]],
                              symtimes=sym.get(r["repo"], {}))
            try:
                ghost = phantoms(s["language"], cat[r["repo"]], ctx.text)
            except ValueError as e:  # the catalog disagrees with the HEAD files, or a row has no annotation
                msg = str(e).replace(f" (repo {r['repo']})", "")  # the repo named once
                raise InputError(f"{r['repo']}: {msg}") from e
            repos[r["repo"]] = Repo(r["repo"], s["language"], s["styles"], cat[r["repo"]], ctx, ghost,
                                    {su: v for (rp, su), v in suites.items() if rp == r["repo"]}, forbidden)
    # what 3A's composer keeps out of ④ for each record: its leak rules and the masked test-answer infrastructure,
    # which depends on the held-out records of the repo, so the composer holds the repo's records of every partition
    every = [args.phase2 / "records" / f"{part}.jsonl" for part in PARTITIONS]
    inputs += [f for f in every if f not in inputs]
    comp = Composer({rp: rep.ctx for rp, rep in repos.items()}, {rp: cat[rp] for rp in repos}, forbidden,
                    {x["id"]: x for f in every for x in load(f) if x["repo"] in repos})

    def timed(xs: list[dict]) -> list[dict]:  # 3A's composer (shown4) puts each record at its t_vis, as ertree.removal
        for x in xs:                          # does: without one, a clean exit 3 naming the id once, not a traceback
            if not isinstance(x.get("t_vis"), int):
                raise InputError(f"{x['id']}: no t_vis")
        return xs
    shown = {r["id"]: shown4(comp, r) for r in timed(records)}

    def prep(x: dict, masked):  # a record the ER tree cannot be built for: exit 3, not a traceback
        if key(x) not in labels:  # a refusal, not a KeyError
            raise InputError(f"{x['id']}: no row in {args.phase2 / 'labels.jsonl'}")
        try:
            return prepare(x, repos[x["repo"]], labels[key(x)], masked)
        except ValueError as e:
            raise InputError(f"{x['id']}: {e}") from e
    preps = {r["id"]: prep(r, shown[r["id"]].masked) for r in records}
    if args.dose_table and (args.controls or args.outputs):
        raise InputError("--dose-table is a mode of its own: not with --controls or --outputs")
    if not args.outputs and (args.manifest or args.coverage or args.accepted_missing or args.accepted_missing_commit):
        raise InputError("--manifest, --coverage and --accepted-missing belong to --outputs")
    table = dose_rows(args, preps, inputs) if args.dose_table else None  # refused before anything is written
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if table is not None:
        write_jsonl(args.out / "dose.jsonl", table)
        written.append("dose.jsonl")
    if args.controls:
        pool = donors(list(preps.values()))
        # D11 (d): stub-v3top4's prior, from the DEV partition's records prepared once as for scoring dev (the record's
        # own gold is left out of its prior by DevPrior itself); only the repos this partition scores
        if args.partition == "dev":
            dev_preps = list(preps.values())
        else:
            dev_preps = [prep(x, shown4(comp, x).masked)
                         for x in timed([x for x in load(args.phase2 / "records" / "dev.jsonl") if x["repo"] in repos])]
        priors = dev_prior(dev_preps)
        rows, texts, excl, refs = [], [], [], []
        for rid, p in preps.items():
            res = controls(p, pool, shown[rid], priors)
            gold = res["gold"][0]
            defined, _ = ca(p, gold)
            if not gold.bv or not defined:
                lang, text = p.repo.language, p.repo.ctx.text
                head_bad = any(parser(lang).parse(text(t).encode()).root_node.has_error for t in p.targets if text(t))
                reason = ("head_unparsable" if head_bad else "gold_fails_bv") if not gold.bv else "ca_undefined"
                excl.append({"record_id": rid, "repo": p.record["repo"], "reason": reason, "excl_bv": reason in BV_EXCLUDED,
                             "excl_ca": True, "failed": [c for c in CONDITIONS if not gold.conds[c]]})
            for name, (s, fallback, blocks) in res.items():
                d, v = ca(p, s)
                ov = call_overlap(p, s)  # D11 (b): n_calls, call_precision, after the expansion of helper calls
                rows.append({"record_id": rid, "repo": p.record["repo"], "control": name, "bv": s.bv, "ca": v,
                             "ca_defined": d, "fallback": fallback, "open_resolution": bool(s.detail.get("open_resolution")),
                             **ov, **{f"bv_{c}": int(x) for c, x in s.conds.items()}})
                texts.append({"record_id": rid, "model": "control", "seed": "-", "condition": name, "variant": "control",
                              "text": as_text(p.repo.language, blocks)})
            refs.append({"record_id": rid, **ca_refs(res, lambda s: ca(p, s)[1])})
        summary = controls_summary(rows)
        for name, data in (("controls.jsonl", rows), ("control_outputs.jsonl", texts), ("exclusions.jsonl", excl), ("ca_ref.jsonl", refs),
                           ("future_methods.jsonl", future_rows(preps))):
            write_jsonl(args.out / name, data)
            written.append(name)
        (args.out / "CONTROLS.json").write_text(json.dumps({k: dict(sorted(v.items())) for k, v in sorted(summary.items())},
                                                           indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append("CONTROLS.json")
        (args.out / "OPEN_RESOLUTION.json").write_text(json.dumps(open_resolution_rates(rows, ("repo", "control")), indent=1) + "\n",
                                                     encoding="utf-8")
        written.append("OPEN_RESOLUTION.json")
        dropped = ca_prec_excluded(rows)  # D11: listed for a check before the lock (expected none)
        for x in dropped:
            print(f"CA_prec@0.25 = 0 with CA 1: {json.dumps(x)}", file=sys.stderr)
        (args.out / "CA_PREC_EXCLUDED.json").write_text(json.dumps(dropped, indent=1) + "\n", encoding="utf-8")
        written.append("CA_PREC_EXCLUDED.json")
    runs: Counter = Counter()
    if args.outputs:
        if args.controls_dir is None:
            raise InputError("--outputs needs --controls-dir")
        if len(args.er_results) != len(args.er_manifest):  # one ER run per manifest, in the same order
            raise InputError("--er-results and --er-manifest go together, one manifest per ER run, in the same order")
        if args.prompts_meta is None:
            raise InputError("--outputs needs --prompts-meta (PROMPTS_META.json of the prompts that made the outputs)")
        pm = read_json(args.prompts_meta)
        bad = a3_errors(args.a3_repo, pm.get("v2tools_commit") if isinstance(pm, dict) else None)
        st = pm.get("symtimes_sha256") if isinstance(pm, dict) else None
        if st != sha256(args.symtimes):  # the data symcache.load reads is the prompts' (required)
            bad.append(f"PROMPTS_META.json symtimes_sha256 {st!r} is not the sha256 of {args.symtimes}")
        if bad:
            raise InputError("; ".join(bad) + " (the scoring side must read V(r) with the prompts' own 3A code)")
        plan, plan_meta = read_plan(args, inputs)  # the generation run's manifest
        # id -> its own partition, over every partition's records
        record_partition = {x["id"]: part for part in PARTITIONS
                            for x in load(args.phase2 / "records" / f"{part}.jsonl")}
        inputs.append(args.prompts_meta)
        manifests = {}  # (record_id, condition) -> overlay_sha256 of its manifest line (ER contract)
        for f in args.er_manifest:
            if not f.is_file():  # a missing or malformed ER manifest is a refusal
                raise InputError(f"missing input {f}")
            for line in f.read_text(encoding="utf-8").split("\n"):  # as er.manifest.load_manifest: U+2028 survives
                if line.strip():
                    try:
                        x = json.loads(line)
                    except ValueError as e:
                        raise InputError(f"{f}: a line is not JSON ({e})") from e
                    if not isinstance(x, dict):
                        raise InputError(f"{f}: a line is not a JSON object")
                    need(x, ("record_id", "condition"), f"line of {f}")
                    if (x["record_id"], x["condition"]) in manifests:
                        raise InputError(f"{f}: {(x['record_id'], x['condition'])} twice in the ER manifests")
                    manifests[(x["record_id"], x["condition"])] = hashlib.sha256(line.encode()).hexdigest()
        for f, m in zip(args.er_results, args.er_manifest):  # contract 0c42865: read a run once er_complete.json says so
            done = f.parent / "er_complete.json"
            if not done.exists():
                raise InputError(f"missing input {done}")
            try:
                state = json.loads(done.read_text(encoding="utf-8"))
            except ValueError:
                state = None
            if not isinstance(state, dict):
                raise InputError(f"{done}: not a JSON object")
            if state.get("complete") is not True or state.get("running"):
                raise InputError(f"{done}: the ER run is not complete (complete {state.get('complete')}, "
                                 f"running {state.get('running')})")
            if state.get("manifest_sha256") != sha256(m):  # the run was made from its own manifest
                raise InputError(f"{done}: manifest_sha256 {state.get('manifest_sha256')} is not that of {m}")
            inputs.append(done)
        meta = read_json(args.controls_dir / "SCORES_META.json")
        if not isinstance(meta, dict) or not isinstance(meta.get("inputs"), dict):  # a refusal
            raise InputError(f"{args.controls_dir / 'SCORES_META.json'}: not a scoring META (no inputs)")
        now = tool_state()
        if meta.get("code_sha256") != now["code_sha256"]:  # holds outside git too
            raise InputError(f"{args.controls_dir} was scored with other code (code_sha256 {meta.get('code_sha256')})")
        if meta.get("tool_commit") != now["tool_commit"]:
            raise InputError(f"{args.controls_dir} was scored by tool commit {meta.get('tool_commit')}, not the current one")
        made = set(meta["inputs"].values())
        for f in inputs[:7]:  # the same Phase 2 records, labels and catalog, and the
            # same forbidden.json, pairs, source config and symtimes: what every scoring reads
            if sha256(f) not in made:
                raise InputError(f"{args.controls_dir} was not built from this {f.name}")
        if load(args.controls_dir / "future_methods.jsonl") != future_rows(preps):  # one set per record
            raise InputError(f"the future methods of the records differ from {args.controls_dir}/future_methods.jsonl")
        ref = {x["record_id"]: x for x in load(args.controls_dir / "ca_ref.jsonl")}
        excl = {x["record_id"]: x for x in load(args.controls_dir / "exclusions.jsonl")}
        sc = {}
        for x in load(args.self_contained):
            need(x, ("record_id", "self_contained"), f"row of {args.self_contained}")  # a refusal, not a KeyError
            sc[x["record_id"]] = x["self_contained"]
        sealed = read_json(args.subsets)
        er = {}
        for f in args.er_results:
            rows_f = load(f)
            why = verdicts_mismatch(f, rows_f)
            if why:
                raise InputError(why)
            for x in rows_f:
                need(x, ("record_id", "condition"), f"er_results row in {f}")
                k = (x["record_id"], x["condition"])
                if k in er:
                    raise InputError(f"{f}: er_results row {k} twice")
                if k in manifests and x.get("overlay_sha256") != manifests[k]:  # its own manifest line's
                    raise InputError(f"{f}: er_results row {k} has an overlay_sha256 that is not its manifest line's")
                if unreviewed(x):
                    raise InputError(f"{f}: er_results row {k} is flagged (tamper_flags) and has no tamper verdict")
                er[k] = x
        in_manifest = set(manifests)
        lost, stray = sorted(in_manifest - set(er)), sorted(set(er) - in_manifest)
        if lost:
            raise InputError(f"{len(lost)} (record, run) of the ER manifests have no er_results row, e.g. {lost[0]}")
        if stray:
            raise InputError(f"{len(stray)} er_results rows are in no ER manifest, e.g. {stray[0]}")
        prompts, models = {}, set()
        for spec in args.prompts_f2:
            model, path, texts = f2_prompts(args, spec, preps)
            if model in models:
                raise InputError(f"--prompts-f2 {spec!r}: need <model>=<file>, each model once")
            models.add(model)
            prompts.update({(model, rid): t for rid, t in texts.items()})
            inputs.append(path)
        inputs += [args.controls_dir / "ca_ref.jsonl", args.controls_dir / "exclusions.jsonl",
                   args.controls_dir / "future_methods.jsonl", args.self_contained,
                   args.subsets, *args.er_results, *args.er_manifest, *args.outputs]
        rows, seen, planned, skipped = [], set(), set(), 0
        for f in args.outputs:
            found, hashes = load(f), line_hashes(f)
            if len(found) != len(hashes):
                raise InputError(f"{f}: {len(found)} rows but {len(hashes)} lines")
            for o, h in zip(found, hashes):
                need(o, OUTPUT_FIELDS, f"output row in {f}")
                wrong = [k for k, t in OUTPUT_TYPES.items() if not isinstance(o[k], t) or t is int and isinstance(o[k], bool)]
                if wrong:  # a refusal, not a traceback or (probes) a silent cap_hit
                    raise InputError(f"output row in {f}: {', '.join(wrong)} of the wrong type")
                k = (o["record_id"], run_name(o))
                if o["record_id"] not in preps:
                    other = record_partition.get(o["record_id"])
                    if other is None:
                        raise InputError(f"{f}: record {o['record_id']} is not in partition {args.partition}")
                    # only a test/aux_test pair ever shares a generation workdir; a train
                    # or dev row scored under test/aux_test, or a test/aux_test row scored under dev, is refused
                    if {other, args.partition} != TEST_AUX_PARTITIONS:
                        raise InputError(f"{f}: record {o['record_id']} is a {other} record, not {args.partition} "
                                         "(only test and aux_test share a generation workdir)")
                    # the partition's test/aux_test partner (the generation run's genqueue: one workdir holds both)
                    skipped += 1
                    continue
                if k in seen:
                    raise InputError(f"{f}: (record, run) {k} twice")
                seen.add(k)
                if plan is None and o["model"] != PROBE_MODEL:
                    raise InputError(f"{f}: a {o['model']} row needs --manifest and --coverage (probes go without)")
                m = None
                if plan is not None:
                    need(o, ("fmt",), f"output row in {f}")
                    m = plan.get(plan_key(o))
                    if m is None or m["status"] != "ok":
                        raise InputError(f"{f}: row {plan_key(o)} is not an ok line of {args.manifest}")
                    if h != m["line_sha256"]:
                        raise InputError(f"{f}: the raw bytes of row {plan_key(o)} do not hash to its line_sha256")
                    for x in PLAN_SHAS:
                        if x in o and o[x] != m[x]:
                            raise InputError(f"{f}: row {plan_key(o)}: {x} differs from its manifest line's")
                    planned.add(plan_key(o))
                runs[run_name(o)] += 1
                p = preps[o["record_id"]]
                s = score_output(p, o["text"])
                scored = {**o, "cap_hit": (not o["stopped_at_eos"]) and o["new_tokens"] >= o["max_new_tokens"]}
                if m is not None:  # provenance of the generation: the manifest's decision, code and own line
                    # (line_sha256, checked above against the output row's own raw bytes)
                    scored.update({x: m[x] for x in ("decision_sha256", "gen_code_sha256", "line_sha256")})
                blocks = extract(o["text"], p.targets).by_path()
                x = excl.get(o["record_id"])
                flags = {"excl_bv": bool(x) and x["reason"] in BV_EXCLUDED, "excl_ca": bool(x)}
                if o["record_id"] not in ref:
                    raise InputError(f"no ca_ref row for {o['record_id']} in {args.controls_dir}")
                if o["record_id"] not in sc:
                    raise InputError(f"{o['record_id']} is not in {args.self_contained}")
                rows.append(row(p, scored, s, blocks, ref[o["record_id"]], flags, sealed, sc[o["record_id"]],
                                er.get(k) if k in in_manifest else None, prompts.get((o["model"], o["record_id"])),
                                shown[o["record_id"]].masked))
        if plan is not None:  # every ok pair of a (model, variant) in the outputs, of this partition, has its row
            present = {(k[0], k[3]) for k in planned}
            lost = sorted(k for k, x in plan.items() if x["status"] == "ok" and (k[0], k[3]) in present
                          and x["record_id"] in preps and k not in planned)
            if lost:
                raise InputError(f"{args.manifest}: {len(lost)} ok lines have no output row, e.g. {lost[0]}")
        write_jsonl(args.out / "scores.jsonl", rows)
        written.append("scores.jsonl")
        by = ("repo", "model", "seed", "condition", "variant")
        (args.out / "OPEN_RESOLUTION.json").write_text(json.dumps(open_resolution_rates(rows, by), indent=1) + "\n",
                                                     encoding="utf-8")
        written.append("OPEN_RESOLUTION.json")
    mode = "dose" if args.dose_table else "controls" if args.controls else "outputs"
    meta = {"partition": args.partition, "mode": mode, **tool_state(),
            "prompts_v2tools_commit": read_json(args.prompts_meta).get("v2tools_commit") if args.outputs else None,
            "inputs": {str(x): sha256(x) for x in inputs}, "runs": dict(sorted(runs.items()))}
    if args.outputs:  # what sap_run checks (it verifies --accepted-missing-commit with git)
        meta.update(plan_meta, skipped_other_partition=skipped,
                    scored_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    # the sha256 of the scores.jsonl this run wrote
                    scores_sha256=sha256(args.out / "scores.jsonl"))
    (args.out / "SCORES_META.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    written.append("SCORES_META.json")
    sums = "".join(f"{hashlib.sha256((args.out / n).read_bytes()).hexdigest()}  {n}\n" for n in sorted(written))
    (args.out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(json.dumps({"partition": args.partition, "records": len(records), "written": sorted(written)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
