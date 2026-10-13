"""The Phase 3 statistical analysis plan (§5 of the design specification at 40bc09a), run on the score rows of 3C. Every
choice is read from config/sap.json, which is locked (with the 3D analysis code and every input that decides a
denominator, a subset, the dev choice or an MDE, by sha256) before any test output.

Unit: the scenario (`scenario_key`); cluster: its .feature file within its repo (`repo|feature_path`). Within one run a
scenario's value is the mean over its records (suites). Two conditions are paired SEED BY SEED first (an adapter
condition against the one base run, or two adapter conditions on their common seeds) and the per-seed differences are
then averaged. Confirmatory tests (anchor model, variant "main"): RQ1 = Full-aware vs Zero-shot, RAG-only and FT-only;
RQ2 = Full-aware vs Full-neutral; on CA. A comparison is "supported" when BOTH tests are significant after Holm in the
expected direction, each test with its own Holm family per RQ: (1) the cluster sign-flip on the seed means, (2) the
two-level bootstrap (seeds, then clusters). BV, ER, ER-trace and the per-seed estimates are descriptive."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable

from .stats import Comparison, compare, holm, kappa, mde, percentile_ci, two_level, two_level_p

EXTRACT_STATUSES = ("ok", "multi_file_fence", "unclosed", "malformed", "none")  # 3C's extractor; per_run
RunKey = tuple[str, str, str]  # (model, condition, variant)
PerSeed = dict[str, dict[str, tuple[str, float]]]  # seed -> scenario -> (cluster, d)
ER_METRICS = {"er": "er", "er_trace": "trace", "trace_action": "trace", "trace_then": "trace"}


class SapInputError(Exception):
    """An input breaks a pre-declared rule (completeness, lock, dev variant, subsets): the analysis stops."""


def jsonl_rows(path: Path) -> list[dict]:
    """The rows of a JSONL file, for every 3D reader, read exactly as 3A's symcache.jsonl_rows
    and the generation run's jsonl_lines read them (one copy here: the 3D code is locked alone and imports no
    3A module): the UTF-8 bytes decoded and split on "\\n" only, since str.splitlines() also breaks at
    U+2028, U+2029 and U+0085, which json.dumps(ensure_ascii=False) leaves raw inside a string; only a final "" (the
    last newline) is dropped, and every other line must be a JSON row, so a blank line inside the file, or rows split by
    a lone "\\r", stop the analysis. Where 3A raises json's error (or the decoder's), 3D raises
    SapInputError naming the line, which each CLI turns into its input refusal; a UTF-8 BOM is not skipped (line 1 is
    then not JSON, as in 3A), and a line that is JSON but not an object (a list, a number, a string, null)
    is refused too, where a reader would crash on it."""
    return jsonl_bytes(path.read_bytes(), path)


def jsonl_bytes(data: bytes, path) -> list[dict]:
    """jsonl_rows of bytes already read: a file whose sha256 is checked is read ONCE, and the rows come from
    the very bytes that were hashed; `path` names the file in the messages."""
    try:
        lines = data.decode("utf-8").split("\n")
    except UnicodeDecodeError as e:
        raise SapInputError(f"{path} is not UTF-8: {e}") from e
    if lines[-1] == "":
        lines.pop()
    rows = []
    for i, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            raise SapInputError(f"{path} line {i} is not a JSON row: {e}") from e
        if not isinstance(rows[-1], dict):
            raise SapInputError(f"{path} line {i} is not a JSON object")
    return rows


def pre_t_flags(path: Path) -> dict[str, set[str]]:
    """The frozen pre-T list (evidence/pret_flags/FLAGGED_PRE_T.tsv;
    config/sap.json names its sha256): no header, one "<partition>\\t<record_id>" per line, split on "\\n" only as
    jsonl_rows does (a final "" dropped), read as partition -> record ids. A line that is not two non-empty
    tab-separated fields without surrounding whitespace (a blank line inside the file, or a "\\r\\n" ending, included)
    stops the analysis."""
    try:
        lines = path.read_bytes().decode("utf-8").split("\n")
    except UnicodeDecodeError as e:
        raise SapInputError(f"{path} is not UTF-8: {e}") from e
    if lines[-1] == "":
        lines.pop()
    out: dict[str, set[str]] = {}
    for i, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 2 or not all(f and f == f.strip() for f in fields):
            raise SapInputError(f"{path} line {i} is not <partition>\\t<record_id>")
        out.setdefault(fields[0], set()).add(fields[1])
    return out


PRE_T_PARTITIONS = ("test", "dev", "aux_test")


def pre_t_errors(cfg: dict, path: Path, digest: str | None, flags: dict[str, set[str]],
                 known: dict[str, set[str]]) -> list[str]:
    """Why the pre-T list at `path` (read by pre_t_flags; `digest` its sha256) cannot be used: its
    sha256 is not the one config/sap.json names (cfg["pre_t_flags"]["sha256"]); it lists a partition outside test, dev
    and aux_test; or a record it lists for a partition of `known` (test and aux_test: the ids of the locked records)
    is not one of that partition's records, which the robustness view would not drop (the list would then be a no-op
    unseen)."""
    named = (cfg.get("pre_t_flags") or {}).get("sha256")
    out = [] if digest is not None and digest == named else [
        f"{path} (sha256 {digest}) is not the sha256 config/sap.json names for the pre-T flags ({named})"]
    other = sorted(set(flags) - set(PRE_T_PARTITIONS))
    if other:
        out.append(f"{path} lists partitions outside test, dev and aux_test: {other}")
    for part, ids in known.items():
        stray = sorted(flags.get(part, set()) - ids)
        if stray:
            out.append(f"{path} lists {len(stray)} {part} records that are not in the locked {part} records, "
                       f"e.g. {stray[0]}")
    return out


def run_key(row: dict) -> RunKey:
    return row["model"], row["condition"], row.get("variant") or "main"


def nan_none(x):
    return None if x is None or x != x else x


def as_dict(c: Comparison) -> dict:
    """JSON-safe: an empty comparison (no scenario) gives None, never NaN."""
    return {"n_scenarios": c.n_scenarios, "n_clusters": c.n_clusters, "estimate": nan_none(c.estimate),
            "ci95": [nan_none(x) for x in c.ci], "p": nan_none(c.p)}


# ----- eligibility and denominators ------------------------------------------------------------------------------------

def eligibility(er_controls: list[dict], self_contained: dict[str, bool],
                expected: set[str]) -> dict[str, dict[str, bool]]:
    """Records whose ER / ER-trace can be scored (§2.3 and §2.3b of the design specification): ER needs gold PASS and
    stub not PASS; ER-trace needs an evaluable trace where gold's trace matches (trace_action 1) and stub's does not
    (0). Each also in a self_contained version. Rows are the ER harness's er_results of the controls run (fields
    er_status, trace_action), one per (record_id, condition). The run covers exactly `expected` (the exec_core records
    of records/test.jsonl): a duplicated (record_id, condition) row (never last-wins), an expected record without its
    gold row or its stub row, or with no row at all (a partial ER control file), or a row of a record not expected (a
    stray row) stops the analysis. Callers pass the rows of the test partition only (er_scope), so a stray row is one of
    a test record that is not exec_core."""
    ctl: dict = defaultdict(dict)
    dup = set()
    for x in er_controls:
        missing = [k for k in ("record_id", "condition", "er_status", "trace_action") if k not in x]
        if missing:
            raise SapInputError(f"er_results row lacks {missing}")
        if x["condition"] in ctl[x["record_id"]]:
            dup.add((x["record_id"], x["condition"]))
        ctl[x["record_id"]][x["condition"]] = x
    if dup:
        raise SapInputError(f"{len(dup)} duplicated (record_id, condition) rows in the ER control results, "
                            f"e.g. {min(dup)}")
    partial = sorted(rid for rid in expected if not {"gold", "stub"} <= set(ctl.get(rid, {})))
    if partial:
        raise SapInputError(f"partial ER control file: {len(partial)} exec_core test records lack a gold row or a stub "
                            f"row (or both), e.g. {partial[0]}")
    stray = sorted(set(ctl) - expected)
    if stray:
        raise SapInputError(f"{len(stray)} records in the ER control results are not exec_core test records, "
                            f"e.g. {stray[0]}")
    er = {rid: c.get("gold", {}).get("er_status") == "PASS" and c.get("stub", {}).get("er_status") != "PASS" for rid, c in ctl.items()}
    tr = {rid: c.get("gold", {}).get("trace_action") == 1 and c.get("stub", {}).get("trace_action") == 0 for rid, c in ctl.items()}
    sc = lambda d: {rid: ok and bool(self_contained.get(rid)) for rid, ok in d.items()}
    return {"er": er, "er@self_contained": sc(er), "trace": tr, "trace@self_contained": sc(tr)}


def er_scope(er_rows: list[dict], records: list[dict]) -> tuple[list[dict], set[str]]:
    """(the ER control rows the SAP reads, the records they must cover) for eligibility, from the rows of
    records/test.jsonl (`records`). The ER controls run is not test-only: write_overlays' default ids are the test
    exec_core records plus the dev Sylius api* records, whose rows serve the ER pilot, not the SAP, and are left out
    here (every row of a record outside records/test.jsonl). Every row of a test record stays, so eligibility still
    refuses a row of a test record that is not exec_core (a stray row); so does a row without record_id, so that
    eligibility reports the missing field. The records to cover: the ids of the exec_core test records."""
    test = {r["id"] for r in records}
    return ([x for x in er_rows if "record_id" not in x or x["record_id"] in test],
            {r["id"] for r in records if r["exec_core"]})


def er_complete_errors(er_files: list[Path], manifest_sha256: str | None) -> list[str]:
    """Why the ER control results given cannot be read (the ER harness's er_results contract, 0c42865): every ER run
    directory holds er_complete.json next to er_results.jsonl (expected, rows, missing, extra,
    duplicates, suspect, scope, scope_missing, complete, manifest_sha256; running true while a batch is still going),
    and a reader refuses the results unless complete is true. For the directory of each file given (once per
    directory): its er_complete.json is missing, is not a JSON object, has complete not exactly true, says running, or
    has a manifest_sha256 that is not `manifest_sha256` (the sha256 of the --er-manifest given; None, a
    missing manifest, never matches). Only the files given are read, never the rows moved to *.superseded.jsonl."""
    out = []
    for d in dict.fromkeys(p.parent for p in er_files):
        path = d / "er_complete.json"
        if not path.is_file():
            out.append(f"{path} is missing")
            continue
        try:
            done = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            done = None
        if not isinstance(done, dict):
            out.append(f"{path} is not a JSON object")
            continue
        if done.get("complete") is not True:
            out.append(f"{path} complete is not true")
        if done.get("running"):
            out.append(f"{path} says the batch is still running")
        if manifest_sha256 is None or done.get("manifest_sha256") != manifest_sha256:
            out.append(f"{path} manifest_sha256 is not the sha256 of the ER manifest given here")
    return out


def manifest_overlays(path: Path) -> dict[tuple[str, str], str]:
    """(record_id, condition) -> overlay_sha256 of the lines of an ER manifest.jsonl, computed as the ER harness does
    (the ER harness's er/manifest.py, load_manifest and Entry.overlay_sha256): the sha256 of the raw line, the file
    split on "\\n" only (not splitlines: U+2028, U+2029 and U+0085 stay inside JSON strings), blank lines skipped. It is
    on purpose not a jsonl_rows reader: it keeps load_manifest's semantics, blank lines included,
    so that its hashes are the harness's. A pair listed twice stops the analysis, as it stops the harness; a line that
    is not a JSON object with record_id and condition is refused (SapInputError), not a crash."""
    out: dict = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            o = None
        if not isinstance(o, dict) or "record_id" not in o or "condition" not in o:
            raise SapInputError(f"ER manifest {path} line {i} is not a JSON object with record_id and condition")
        key = (o["record_id"], o["condition"])
        if key in out:
            raise SapInputError(f"ER manifest {path} lists {key} twice")
        out[key] = hashlib.sha256(line.encode("utf-8")).hexdigest()
    return out


def overlay_errors(er_rows: list[dict], overlays: dict[tuple[str, str], str]) -> list[str]:
    """ER provenance against the lock: every ER control row given (sap_lock and sap_run pass the rows of
    the test records, er_scope) is the result of the manifest line of its (record_id, condition), so its overlay_sha256
    is that line's (`overlays`, from manifest_overlays). The rows of a pair the manifest does not list, then the rows
    with another overlay_sha256 (or none), are reported, each with the count and one example."""
    key = lambda x: (x.get("record_id"), x.get("condition"))
    unlisted = sorted(key(x) for x in er_rows if key(x) not in overlays)
    stale = sorted(key(x) for x in er_rows if key(x) in overlays and x.get("overlay_sha256") != overlays[key(x)])
    out = []
    if unlisted:
        out.append(f"{len(unlisted)} ER control rows are not in the ER manifest given here, e.g. {unlisted[0]}")
    if stale:
        out.append(f"{len(stale)} ER control rows have an overlay_sha256 that is not the sha256 of their manifest "
                   f"line, e.g. {stale[0]}")
    return out


def compose_dropped(compose_excluded: list[dict], model: str) -> set[str]:
    """Records without anchor outputs: compose's excluded.jsonl entries that hold for every model (model null) or for the
    anchor model (any format)."""
    return {x["record_id"] for x in compose_excluded if x.get("model") in (None, model)}


def usable_records(controls: list[dict], exclusions: list[dict], compose_excluded: list[dict], model: str) -> set[str]:
    """Completeness (§5 of the design specification): the records of the controls run minus the mechanical exclusions
    minus compose_dropped."""
    drop = {x["record_id"] for x in exclusions} | compose_dropped(compose_excluded, model)
    return {x["record_id"] for x in controls if x.get("control", "gold") == "gold"} - drop


def design_records(controls: list[dict], compose_excluded: list[dict], model: str) -> set[str]:
    """The records every design run must cover, one row each: the records of the
    controls run (a gold row) minus compose_dropped. The mechanical exclusions stay: they are out of CA only, and still
    count in BV (those not excl_bv) and ER, whose denominators must not move either."""
    gold = {x["record_id"] for x in controls if x.get("control", "gold") == "gold"}
    return gold - compose_dropped(compose_excluded, model)


def locked_flags(controls: list[dict], exclusions: list[dict]) -> dict[str, dict[str, bool]]:
    """record_id -> the flags 3C gives every score row of the record from its controls run (§2.2 of the design
    specification):
    ca_defined of its gold row in controls.jsonl (it depends on the gold answer alone), excl_bv of its exclusions.jsonl
    entry and excl_ca = listed there (a record not listed: False, False). Records with a gold row only."""
    excl = {x["record_id"]: x for x in exclusions}
    return {x["record_id"]: {"ca_defined": bool(x.get("ca_defined")),
                             "excl_bv": bool(excl.get(x["record_id"], {}).get("excl_bv")), "excl_ca": x["record_id"] in excl}
            for x in controls if x.get("control", "gold") == "gold"}


def usable(metric: str, excluded: set[str], elig: dict[str, dict[str, bool]] | None = None,
           dropped: set[str] = frozenset()) -> Callable[[dict], bool]:
    """Rows in the denominator of `metric`. Every metric drops the compose-dropped records (`dropped`: compose_dropped
    for the anchor model, so that every model has one denominator). CA also drops the mechanical exclusions (`excluded`,
    any reason; sap_run passes every record outside usable_records) and needs ca_defined, and so does CA_prec (D11 (c):
    the same denominator as CA). BV drops only the rows 3C flags excl_bv (gold fails BV, or its HEAD does not parse;
    §2.2 of the design specification), beyond the compose-dropped ones. ER metrics use their eligibility table (`elig`;
    "<metric>@self_contained" uses the self_contained one) and need a value, with no mechanical exclusion."""
    base, _, sub = metric.partition("@")

    def keep(r: dict) -> bool:
        if r["record_id"] in dropped:
            return False
        if base == "bv":
            return not r["excl_bv"] and r.get("bv") is not None
        if base in ER_METRICS:
            table = (elig or {}).get(ER_METRICS[base] + ("@self_contained" if sub else ""), {})
            return bool(table.get(r["record_id"])) and r.get(base) is not None
        if r["record_id"] in excluded:
            return False
        if base in ("ca", "ca_prec"):
            return bool(r.get("ca_defined"))
        return r.get(base) is not None
    return keep


def seed_values(rows: list[dict], metric: str, keep: Callable[[dict], bool]) -> dict[RunKey, PerSeed]:
    """run -> seed -> scenario -> (cluster, mean over the scenario's records)."""
    base = metric.partition("@")[0]
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cluster = {}
    for r in rows:
        if keep(r):
            acc[run_key(r)][r["seed"]][r["scenario_key"]].append(float(r[base]))
            cluster[r["scenario_key"]] = f"{r['repo']}|{r['feature_path']}"
    return {rk: {sd: {s: (cluster[s], sum(v) / len(v)) for s, v in sc.items()} for sd, sc in by.items()} for rk, by in acc.items()}


def scenario_values(rows: list[dict], metric: str, keep: Callable[[dict], bool]) -> dict[RunKey, dict[str, tuple[str, float]]]:
    """run -> scenario -> (cluster, mean over the seeds that scored it)."""
    out = {}
    for rk, by in seed_values(rows, metric, keep).items():
        acc: dict = defaultdict(list)
        cl = {}
        for sc in by.values():
            for s, (c, v) in sc.items():
                acc[s].append(v)
                cl[s] = c
        out[rk] = {s: (cl[s], sum(v) / len(v)) for s, v in acc.items()}
    return out


def per_seed_pairs(sv: dict[RunKey, PerSeed], ka: RunKey, kb: RunKey, common_only: bool = False) -> PerSeed:
    """Seed -> scenario -> (cluster, a - b): b's single run (a base condition) serves every seed of a; two adapter
    conditions are paired on their common seeds. With `common_only` (the P26 contrasts) b is never taken
    for a base run, so a b whose values survive at one seed only is paired at that seed alone."""
    a, b = sv.get(ka, {}), sv.get(kb, {})
    if not a or not b:
        return {}
    single = not common_only and len(b) == 1 and set(b) != set(a)
    out = {}
    for s in sorted(a):
        bb = next(iter(b.values())) if single else b.get(s, {})
        d = {x: (a[s][x][0], a[s][x][1] - bb[x][1]) for x in sorted(set(a[s]) & set(bb))}
        if d:
            out[s] = d
    return out


def pooled(ps: PerSeed) -> list[tuple[str, float]]:
    """(cluster, mean over the seeds of the per-seed difference) per scenario, in scenario order."""
    acc: dict = defaultdict(list)
    for d in ps.values():
        for x, cd in d.items():
            acc[x].append(cd)
    return [(v[0][0], sum(d for _, d in v) / len(v)) for _, v in sorted(acc.items())]


# ----- completeness ------------------------------------------------------------------------------------------------------

def design(cfg: dict) -> list[tuple[RunKey, str]]:
    """Every run the report uses, as (run, seed): for each model of cfg["seeds"], the
    base conditions once (seed "base") and the adapter conditions × that model's seeds, variant main; the anchor's
    serve the confirmatory tests, the others RQ3 and the descriptive tables."""
    return [((m, c, "main"), "base") for m in cfg["seeds"] for c in cfg["base_conditions"]] + \
           [((m, c, "main"), s) for m in cfg["seeds"] for c in cfg["adapter_conditions"] for s in cfg["seeds"][m]]


def aux_design(cfg: dict) -> dict:
    """The config of the aux_test design (added 27/09/2026), which --descriptive-only checks
    with check_complete: the anchor model's runs only (the generation plan, fixed 24/09/2026, runs aux_test with
    DeepSeek alone, the two base conditions once and the three adapter ones × its 5 seeds, 17 runs); `cfg` is not
    changed."""
    return {**cfg, "seeds": {cfg["anchor_model"]: cfg["seeds"][cfg["anchor_model"]]}}


def deferred_runs(cfg: dict) -> set[RunKey]:
    """The anchor's runs of the deferred analyses (§0b of the design specification), outside the design: the
    learning-curve variants of the adapter conditions and the ablation variants of Full-aware, allowed only with
    --with-deferred."""
    m = cfg["anchor_model"]
    return ({(m, c, v) for v in cfg.get("learning_curve", []) for c in cfg["adapter_conditions"]}
            | {(m, "full_aware", v) for v in cfg.get("ablations", [])})


FILL_ZERO = {"ca": 0, "bv": 0, **{f"bv_{c}": 0 for c in ("syntax", "binding", "no_ambiguity", "arity", "grounded", "resolved")}}
RECORD_KEYS = ("scenario_key", "feature_path", "repo", "language", "partition", "suite", "preexisting", "self_contained",
               "exec_core", "len_bucket", "ca_defined", "excl_bv", "excl_ca")


def check_complete(rows: list[dict], ids: dict[str, set[str]], cfg: dict,
                   accepted_missing: set[tuple[str, str, str, str]] = frozenset(),
                   flags: dict[str, dict[str, bool]] | None = None, deferred: bool = False) -> tuple[list[dict], dict]:
    """Every run of the design (every model of cfg["seeds"]) must have exactly one row per record
    of `ids[model]` (sap_run passes design_records per model: every record of the controls run but those compose
    dropped for that model, so that the BV and ER denominators are checked as well as CA's). A row of a run
    outside the design (another model or variant; the deferred runs are allowed with `deferred`, --with-deferred), or
    of a design run under a seed the design does not have ("base" for a base condition, one of the model's config seeds
    for an adapter one: an extra adapter seed, or a seeded copy of a base run, would enter the pairing)
    stops the analysis. So does a duplicate, and a missing row, unless it is listed in `accepted_missing` as
    (record_id, model, condition, seed) after a rerun failed: it then scores CA = 0 and BV = 0, counting in CA and BV
    where its record does and never in ER or ER-trace (it has no ER value), keeps the record's own attributes, takes its
    flags (ca_defined, excl_bv, excl_ca) from `flags` (sap_run: locked_flags of the locked controls run) over those of a
    template row, and is counted in `filled_missing`."""
    runs = design(cfg)
    names = {rk for rk, _ in runs}
    allowed = names | (deferred_runs(cfg) if deferred else set())
    outside = sorted((r["record_id"], run_key(r), r["seed"]) for r in rows if run_key(r) not in allowed)
    if outside:
        raise SapInputError(f"{len(outside)} rows of runs outside the design (model, condition, variant), e.g. "
                            f"{outside[0]}")
    stray = sorted((r["record_id"], run_key(r), r["seed"]) for r in rows
                   if run_key(r) in names and (run_key(r), r["seed"]) not in set(runs))
    if stray:
        raise SapInputError(f"{len(stray)} rows of a design run under a seed the design does not have, e.g. {stray[0]}")
    seen: dict = defaultdict(int)
    by_record: dict = defaultdict(list)
    for r in rows:
        if r["record_id"] in ids.get(r["model"], ()):
            seen[(r["record_id"], run_key(r), r["seed"])] += 1
            by_record[(r["model"], r["record_id"])].append(r)
    dup = sorted(k for k, n in seen.items() if n > 1)
    if dup:
        raise SapInputError(f"{len(dup)} duplicated (record, run, seed) rows, e.g. {dup[0]}")
    missing = [(rid, rk, s) for rk, s in runs for rid in sorted(ids.get(rk[0], ())) if (rid, rk, s) not in seen]
    unaccepted = [x for x in missing if (x[0], x[1][0], x[1][1], x[2]) not in accepted_missing]
    if unaccepted:
        raise SapInputError(f"{len(unaccepted)} missing rows, e.g. {unaccepted[0]}: rerun them; list them as accepted "
                            "missing only after the rerun failed")
    filled = []
    for rid, (model, cond, variant), s in missing:
        same = [r for r in by_record[(model, rid)] if r["condition"] == cond]
        tpl = (same or by_record[(model, rid)] or [{}])[0]
        key = rid.rsplit("|", 1)[0]
        row = {"scenario_key": key, "feature_path": key.split("|")[1], "repo": key.split("|")[0], "ca_defined": True,
               "excl_bv": False, "excl_ca": False,
               **{k: tpl[k] for k in RECORD_KEYS if k in tpl}, **{k: v for k, v in tpl.items() if k.startswith("sub_")},
               **(flags or {}).get(rid, {})}
        if same and "dose" in tpl:
            row["dose"] = tpl["dose"]
        filled.append({**row, "record_id": rid, "model": model, "seed": s, "condition": cond, "variant": variant,
                       **FILL_ZERO, "filled_missing": True})
    return rows + filled, {"records_covered": {m: len(v) for m, v in ids.items()}, "missing_filled": len(filled)}


SELF_CONTAINED = "self_contained"  # subset (iv), the self-contained records: read from the locked self_contained.jsonl


def subset_flags(rows: list[dict], sealed: dict, names: list[str],
                 self_contained: dict[str, bool] | None = None) -> list[dict]:
    """The sensitivity subsets `names` (config "subsets"), from locked files, not from the rows. Subsets (i)-(iii) come
    from sensitivity_subsets.json (`sealed`: subset -> record ids, possibly split by partition): a subset missing from
    the file, or a 3C row whose sub_<name> is not exactly its membership in the file (an absent column included),
    stops the analysis. Subset (iv), SELF_CONTAINED, is the records flagged True in the locked self_contained.jsonl
    (`self_contained`: record_id -> flag), whatever sensitivity_subsets.json says: a 3C row whose self_contained
    column is not its record's flag in that file (an absent column, or a record the file lacks, included) stops the
    analysis. The rows returned carry every membership as sub_<name>; rows filled by check_complete take it without
    comparison."""
    phase2 = [s for s in names if s != SELF_CONTAINED]
    missing = [s for s in phase2 if s not in sealed]
    if missing:
        raise SapInputError(f"sensitivity_subsets.json lacks the subsets {missing}")
    ids = {s: set(sealed[s]) if isinstance(sealed[s], list) else {i for x in sealed[s].values() for i in x}
           for s in phase2}
    if SELF_CONTAINED in names:
        if self_contained is None:
            raise SapInputError("the self_contained subset needs the locked self_contained.jsonl")
        ids[SELF_CONTAINED] = {rid for rid, flag in self_contained.items() if flag is True}
        odd = [r["record_id"] for r in rows if not r.get("filled_missing") and
               (r["record_id"] not in self_contained or r.get("self_contained") is not self_contained[r["record_id"]])]
        if odd:
            raise SapInputError(f"{len(odd)} rows have a self_contained that differs from self_contained.jsonl (or a "
                                f"record it lacks), e.g. {odd[0]}")
    bad = [(r["record_id"], s) for r in rows if not r.get("filled_missing") for s in phase2
           if r.get(f"sub_{s}") is not (r["record_id"] in ids[s])]
    if bad:
        raise SapInputError(f"{len(bad)} (row, subset) flags differ from sensitivity_subsets.json, e.g. {bad[0]}")
    return [{**r, **{f"sub_{s}": r["record_id"] in ids[s] for s in names}} for r in rows]


def call_errors(row: dict) -> bool:
    """Whether a score row breaks the D11 column contract of call precision: n_calls a count, call_precision present,
    None exactly when n_calls is 0, else in [0, 1]."""
    nc, cp = row.get("n_calls"), row.get("call_precision")
    if "call_precision" not in row or not isinstance(nc, int) or isinstance(nc, bool) or nc < 0:
        return True
    return (cp is None) != (nc == 0) or (cp is not None and not 0 <= cp <= 1)


# design specification §2.3 (added 26/09/2026): the two step forms of a PHP output
STEP_COUNTS = ("php_docblock_steps", "php_attribute_steps")
# 3C's reader counts (design specification §2.2 at c6999dc)
READER_COUNTS = ("unread_step_annotations", "cuke_escaped_patterns", "cuke_custom_types")


def column_errors(row: dict) -> bool:
    """Whether a score row breaks 3C's column contract of the lazy references, the PHP tag and the PHP step forms (D11
    (d), the design specification at 4d8bf2a; §1.4 and §2.3, added 26/09/2026): each of LAZY_REFS present, 0, 1 or None
    (no such control for the record, e.g. Ruby); php_tag_stripped present, a bool or None (no PHP text); each of
    STEP_COUNTS present, an int >= 0 or None (no PHP text); under §2.2 of the design specification, each of
    READER_COUNTS present, an int >= 0 or None (no text)."""
    refs_ok = all(k in row and (row[k] is None or row[k] in (0, 1) and not isinstance(row[k], (str, float)))
                  for k in LAZY_REFS)
    tag = row.get("php_tag_stripped", "absent")
    steps_ok = all(k in row and (row[k] is None or type(row[k]) is int and row[k] >= 0) for k in STEP_COUNTS + READER_COUNTS)
    return not refs_ok or not (tag is None or isinstance(tag, bool)) or not steps_ok


def descriptive_errors(row: dict) -> bool:
    """P26 (added 06/10/2026): whether a score row breaks 3C's contract of the descriptive columns P26
    reads (describe.row), each checked when present and not None: members_used and hallucinated ints >= 0 with
    hallucinated <= members_used, duplicate_reuse and exact bools, conv_* in [0, 1], len_bucket one of LEN_BUCKETS. A
    malformed value would be left out of a rate, or counted, without a word."""
    v = {k: row.get(k) for k in ("members_used", "hallucinated", "duplicate_reuse", "exact", "len_bucket",
                                 "conv_style", "conv_pattern", "conv_naming")}
    count = lambda x: x is None or type(x) is int and x >= 0
    if not (count(v["members_used"]) and count(v["hallucinated"])):
        return True
    if v["members_used"] is not None and v["hallucinated"] is not None and v["hallucinated"] > v["members_used"]:
        return True
    if any(v[k] is not None and type(v[k]) is not bool for k in ("duplicate_reuse", "exact")):
        return True
    if any(v[k] is not None and (type(v[k]) not in (int, float) or not 0 <= v[k] <= 1)
           for k in ("conv_style", "conv_pattern", "conv_naming")):
        return True
    return v["len_bucket"] is not None and v["len_bucket"] not in LEN_BUCKETS


ER_VALUE_FIELDS = ("er", "er_trace", "trace_action", "trace_then")  # the ER fields the SAP reads as numbers


def er_value_errors(row: dict) -> bool:
    """Added 07/10/2026: whether one of ER_VALUE_FIELDS is neither the int 0 or 1 nor None (absent is None). The real ER
    results (dev_pilot_v0_0482a76, controls) carry ints 0/1 and None only, and 3C copies them into the score rows
    (describe.ER_FIELDS); the SAP averages them and agreement_ca_trace reads int(trace_action), so a "1", a 2 or a 0.5
    would enter a level or the kappa table unseen. A bool or a float is refused too: the real data has none."""
    return any(row.get(k) is not None and not (type(row[k]) is int and row[k] in (0, 1)) for k in ER_VALUE_FIELDS)


SCORE_ROW_RULES = {
    "carried": "score rows already carry filled_missing (only check_complete adds it)",
    "unknown": "score rows are of records with no gold row in the locked controls.jsonl",
    "key": "score rows have a scenario_key, repo or feature_path that is not that of their record_id "
           "(repo|feature_path|scenario_line|suite)",
    "bv": "score rows have an excl_bv that is not the one of the locked exclusions.jsonl",
    "ca": "score rows have a ca_defined that is not the one of their record's gold row in the locked controls.jsonl",
    "value": "score rows have no ca value although ca_defined, or no bv value although not excl_bv",
    "status": "score rows have an extract_status that is not one of EXTRACT_STATUSES",
    "calls": "score rows lack call_precision or n_calls, or have a call_precision that does not fit n_calls (None "
             "exactly when n_calls is 0, else in [0, 1])",
    "columns": "score rows lack ca_ref_v3m, ca_ref_v3all, ca_ref_v3names, ca_ref_top4, php_tag_stripped, "
               "unread_step_annotations, cuke_escaped_patterns, cuke_custom_types, "
               "php_docblock_steps or php_attribute_steps, or have a lazy reference that is not 0, 1 or None, a "
               "php_tag_stripped that is not a bool or None, or a step count that is not an int >= 0 or None",
    "descriptive": "score rows have a members_used or hallucinated that is not an int >= 0 (hallucinated <= "
                   "members_used), a duplicate_reuse or exact that is not a bool, a conv_* outside [0, 1], or a "
                   "len_bucket that is not le8192 or gt8192 (P26; None or absent are allowed)",
    "record": "score rows have a language, suite, exec_core, partition or preexisting that is not their record's in "
              "the locked records",
    "dose": "score rows have a dose that is not the one of the locked dose table for their (model, record), or no "
            "entry there",
    "er": "score rows of ER-eligible records (anchor model, main) have no er value",
    "er_values": "score rows have an er, er_trace, trace_action or trace_then that is not the int 0 or 1, or None",
    "manifest": "score rows are not an ok line of the manifests (record_id, model, condition, variant, seed)",
    "shas": "score rows lack decision_sha256 (64 lowercase hex) or gen_code_sha256 (a non-empty {relative path: 64 "
            "lowercase hex}), or carry a sha256 (prompts, prompt, adapter, "
            "decision, gen code, leak-check receipt) or a line_sha256 that is not their manifest line's"}
RECORD_FIELDS = ("language", "suite", "exec_core", "partition")  # row fields copied from the record (describe.row)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
CODE_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*")  # relative POSIX, no ".", ".."


def provenance_ok(row: dict) -> bool:
    """The two sha fields a score row takes from its manifest line are
    in the generation run's shapes (fixed 04/10/2026; the real writer, ee6bc8c): decision_sha256 is 64 lowercase hex,
    gen_code_sha256 a non-empty object {relative POSIX path: 64 lowercase hex} (generate.gen_code_sha256: one sha256 per
    v2gpu module). The old string form of gen_code_sha256 is refused: the writer never wrote it, and the fixtures that
    did hid the real shape (every real row was refused before). Equality with the line stays a rule of its own
    (MANIFEST_SHAS)."""
    code, decision = row.get("gen_code_sha256"), row.get("decision_sha256")
    return (isinstance(decision, str) and SHA256_HEX.fullmatch(decision) is not None
            and isinstance(code, dict) and bool(code)
            and all(isinstance(p, str) and CODE_PATH.fullmatch(p) and isinstance(s, str) and SHA256_HEX.fullmatch(s)
                    for p, s in code.items()))


def record_errors(row: dict, record: dict | None) -> bool:
    """Whether a score row's record fields are not its locked record's: RECORD_FIELDS as is, and preexisting = any
    create entry preexisting, as 3C's describe.row writes them; they decide the strata and groups."""
    if record is None:
        return True
    return (any(row.get(k) != record.get(k) for k in RECORD_FIELDS)
            or row.get("preexisting") != any(c.get("preexisting") for c in record.get("create", [])))


def score_row_errors(rows: list[dict], controls: list[dict], exclusions: list[dict], elig: dict[str, dict[str, bool]],
                     cfg: dict, full: bool, records: dict[str, dict] | None = None,
                     dose: dict[tuple[str, str], int | None] | None = None,
                     manifest: dict[tuple, dict] | None = None) -> list[str]:
    """The --scores rows, before check_complete fills any, against the locked controls run, so that no denominator moves
    unseen. Every row must: carry no filled_missing key (only check_complete adds it; a row carrying it would pass for a
    filled one); be of a record with a gold row in `controls` (controls.jsonl), the records 3C scores; have the
    scenario_key, repo and feature_path of its record_id, repo|feature_path|scenario_line|suite (they make its scenario
    and its cluster); have the excl_bv and ca_defined of its record in the controls run (locked_flags: 3C copies excl_bv
    from `exclusions`, exclusions.jsonl, and ca_defined depends on the gold answer alone, calls.agreement); have a ca
    when ca_defined and a bv when not excl_bv (3C always writes both; a None would crash the analysis or leave the row
    out); have an extract_status of EXTRACT_STATUSES (per_run counts those only); have n_calls and call_precision (D11:
    CA_prec reads the latter), call_precision None exactly when n_calls is 0 (3C's call_overlap), else in [0, 1]
    (call_errors); carry the four lazy references, php_tag_stripped and the PHP step counts within their contract
    (column_errors; D11 (d), §1.4 and §2.3 of the design specification), and the descriptive columns of P26 within
    theirs (descriptive_errors); with `records` (the locked records by id), have its record's language, suite,
    exec_core, partition and preexisting (record_errors); with `dose` (the locked 3C dose table, (model, record_id) ->
    dose; added 25/09/2026), have the table's dose for its (model, record), which the table must list. In full mode
    (`full`), every row of the anchor model, variant "main", of an ER-eligible record (elig["er"]) must have an er value
    (None means the run was not in the ER manifest). Every row has its er, er_trace, trace_action and trace_then the int
    0 or 1, or None (er_value_errors; added 07/10/2026). With `manifest` (read_manifests: pair -> line; fixed
    26/09/2026) every row is the output of an `ok` line of its pair, carries decision_sha256 and gen_code_sha256
    (score_run copies them from the line) in the generation run's shapes (provenance_ok), every sha256 of MANIFEST_SHAS
    equals the line's, None included (a base run has no adapter tensor on either side), and its line_sha256 (also copied
    from the line by score_run; None for a row with no manifest tie) equals the line's own line_sha256 (this ties the
    row to the exact ok line it was scored from, not merely a same-key line a tampered manifest could substitute). One
    message per rule broken (SCORE_ROW_RULES), with the count and one example (record_id, model, condition, variant,
    seed)."""
    flags = locked_flags(controls, exclusions)
    eligible = elig.get("er", {})
    key = lambda r: (r["record_id"], r["model"], r["condition"], r.get("variant") or "main", r["seed"])
    broken: dict = {k: [] for k in SCORE_ROW_RULES}
    for r in rows:
        rid, f = r["record_id"], flags.get(r["record_id"])
        parts = rid.split("|")
        rules = {"carried": "filled_missing" in r, "unknown": f is None,
                 "key": len(parts) != 4 or (r.get("scenario_key"), r.get("repo"), r.get("feature_path"))
                 != ("|".join(parts[:3]), parts[0], parts[1]),
                 "bv": f is not None and r.get("excl_bv") != f["excl_bv"],
                 "ca": f is not None and r.get("ca_defined") != f["ca_defined"],
                 "value": (bool(r.get("ca_defined")) and r.get("ca") is None
                           or not r.get("excl_bv") and r.get("bv") is None),
                 "status": r.get("extract_status") not in EXTRACT_STATUSES,
                 "calls": call_errors(r),
                 "columns": column_errors(r),
                 "descriptive": descriptive_errors(r),
                 "record": records is not None and record_errors(r, records.get(rid)),
                 "dose": dose is not None and ((r["model"], rid) not in dose
                                              or r.get("dose") != dose[(r["model"], rid)]),
                 "er": full and bool(eligible.get(rid)) and r["model"] == cfg["anchor_model"]
                 and (r.get("variant") or "main") == "main" and r.get("er") is None,
                 "er_values": er_value_errors(r)}
        line = None if manifest is None else manifest.get(pair(r))
        rules["manifest"] = manifest is not None and (line is None or line.get("status") != "ok")
        rules["shas"] = line is not None and line.get("status") == "ok" and (
            not provenance_ok(r)  # gen_code_sha256 is an object {path: sha256}, never "any truthy"
            or any(r.get(k) != line.get(k) for k in MANIFEST_SHAS)  # strict, None included (fixed 27/09/2026)
            or r.get("line_sha256") != line.get("line_sha256"))  # the row is tied to the
        # exact ok line it was scored from, not just a same-key line a tampered manifest could substitute
        for name, bad in rules.items():
            if bad:
                broken[name].append(key(r))
    return [f"{len(xs)} {SCORE_ROW_RULES[name]}, e.g. {min(xs)}" for name, xs in broken.items() if xs]


# ----- confirmatory ------------------------------------------------------------------------------------------------------

def verdict_of(estimate, p1_holm, p2_holm, alpha: float) -> str:
    """The verdict of one comparison (§5 of the design specification at 40bc09a): "supported" / "contrary" need both
    tests significant after Holm (estimate > 0 / < 0); test (1) alone gives "conditional_only_positive" /
    "conditional_only_negative"."""
    if p1_holm is None:
        return "not_run"
    s1, s2 = p1_holm < alpha, p2_holm is not None and p2_holm < alpha
    if s1 and s2:
        return "supported" if estimate > 0 else "contrary"
    if s1:
        return "conditional_only_positive" if estimate > 0 else "conditional_only_negative"
    return "no_difference"


def rq_verdict(verdicts: list[str]) -> str:
    """RQ verdict: "full" (every comparison supported), "partial" (some supported, none contrary), "not_supported"
    (none supported, none contrary), "contrary" (at least one contrary). conditional_only counts as neither."""
    if "contrary" in verdicts:
        return "contrary"
    n = verdicts.count("supported")
    return "full" if n == len(verdicts) else ("partial" if n else "not_supported")


# the direction labels of the presentation rule (§2.4 of the design specification, fixed 06/10/2026)
DIRECTIONS = ("theo chiều kỳ vọng", "ngược chiều", "không rõ chiều")


def direction(ci95: list, sign: int, n_clusters: int | None = None) -> str | None:
    """The presentation rule of §2.4 of the design specification (fixed 06/10/2026, before any test output): the label
    of a descriptive measure whose expected direction is `sign` (+1: A − B > 0; −1: < 0): "không rõ chiều" when its 95%
    CI holds 0 (an end at 0 included) or rests on fewer than 2 clusters (`n_clusters`, added 06/10/2026: a bootstrap
    over one cluster has no spread), else "theo chiều kỳ vọng" or "ngược chiều" by the side of the CI; None without a CI
    (no scenario). A presentation rule, not a test (P26)."""
    lo, hi = ci95
    if lo is None or hi is None:
        return None
    if lo <= 0 <= hi or n_clusters is not None and n_clusters < 2:
        return DIRECTIONS[2]
    return DIRECTIONS[0] if (lo > 0) == (sign > 0) else DIRECTIONS[1]


def comparison(sv: dict, ka: RunKey, kb: RunKey, cfg: dict, seed: int) -> dict:
    """a - b: pooled estimate, cluster-bootstrap interval and sign-flip p (test 1); two-level interval and p (test 2) on
    the scenarios every seed scored; per-seed estimates and intervals (descriptive)."""
    ps = per_seed_pairs(sv, ka, kb)
    out = as_dict(compare(pooled(ps), cfg["reps_bootstrap"], cfg["reps_permutation"], seed))
    out["per_seed"] = {s: as_dict(compare(list(d.values()), cfg["reps_bootstrap"], 0, seed + 1 + i, with_p=False))
                       for i, (s, d) in enumerate(sorted(ps.items()))}
    out["seeds_positive"] = sum((v["estimate"] or 0) > 0 for v in out["per_seed"].values())
    common = sorted(set.intersection(*(set(d) for d in ps.values()))) if ps else []
    if common:
        dist = two_level({s: [d[x] for x in common] for s, d in ps.items()}, cfg["reps_bootstrap"], seed + 99)
        out["two_level"] = {"n_seeds": len(ps), "n_scenarios": len(common), "ci95": list(percentile_ci(dist)), "p": two_level_p(dist)}
    else:
        out["two_level"] = {"n_seeds": len(ps), "n_scenarios": 0, "ci95": [None, None], "p": None}
    return out


def family(sv: dict, cfg: dict, pairs: list, seed: int) -> dict:
    """The comparisons of one Holm family with both tests Holm-adjusted separately (a comparison with no data keeps its
    place with p = 1) and their verdicts."""
    model = cfg["anchor_model"]
    res = {f"{a}_vs_{b}": comparison(sv, (model, a, "main"), (model, b, "main"), cfg, seed + 100 * i) for i, (a, b) in enumerate(pairs)}
    adj1 = holm({k: 1.0 if v["p"] is None else v["p"] for k, v in res.items()})
    adj2 = holm({k: 1.0 if v["two_level"]["p"] is None else v["two_level"]["p"] for k, v in res.items()})
    for k, v in res.items():
        v["p_holm"] = None if v["p"] is None else adj1[k]
        v["two_level"]["p_holm"] = None if v["two_level"]["p"] is None else adj2[k]
        v["verdict"] = verdict_of(v["estimate"], v["p_holm"], v["two_level"]["p_holm"], cfg["alpha"])
    return res


def confirmatory(rows: list[dict], cfg: dict, excluded: set[str], elig: dict, dropped: set[str] = frozenset(),
                 metric: str = "ca") -> dict:
    """RQ1 and RQ2 on CA (both tests, Holm per family and per test), the RQ-level verdict, and the descriptive metrics
    beside each comparison; all comparisons with data must share one n, and one two-level n (a seed that
    only some conditions have shrinks the latter alone). With `metric` "ca_prec" (D11 (c), the pre-registered
    sensitivity analysis; rows from with_ca_prec), the same comparisons, seeds, tests, Holm families, verdicts and n
    checks on CA_prec, reported under "ca_prec", without the descriptive metrics."""
    model, seed = cfg["anchor_model"], cfg["seed"]
    beside = [m for base in cfg["descriptive"]
              for m in ([base, base + "@self_contained"] if base in ER_METRICS else [base])]
    metrics = beside if metric == "ca" else []
    main = seed_values(rows, metric, usable(metric, excluded, dropped=dropped))
    sv = {m: seed_values(rows, m, usable(m, excluded, elig, dropped)) for m in metrics}
    out = {}
    for f, (fam, pairs) in enumerate(cfg["families"].items()):
        res = {k: {metric: v, "verdict": v.pop("verdict")}
               for k, v in family(main, cfg, pairs, seed + 10000 * f).items()}
        for i, (a, b) in enumerate(pairs if metric == "ca" else []):
            v = res[f"{a}_vs_{b}"]
            v["descriptive"] = {}
            for m in metrics:
                d = as_dict(compare(pooled(per_seed_pairs(sv[m], (model, a, "main"), (model, b, "main"))),
                                    cfg["reps_bootstrap"], 0, seed + 10000 * f + 100 * i + 50, with_p=False))
                # flagged only when the interval excludes 0 against the hypothesis, or the estimate goes against a
                # supported CA
                d["opposite"] = d["n_scenarios"] > 0 and ((d["ci95"][1] is not None and d["ci95"][1] < 0)
                                                         or (v["verdict"] == "supported" and (d["estimate"] or 0) < 0))
                # P26: the label of §2.4 of the design specification beside the P11 flag
                d["direction"] = direction(d["ci95"], 1, d["n_clusters"])
                v["descriptive"][m] = d
        out[fam] = {"comparisons": res, "rq_verdict": rq_verdict([v["verdict"] for v in res.values()])}
    ns = {v[metric]["n_scenarios"] for f in out.values() for v in f["comparisons"].values()
          if v[metric]["p"] is not None}
    if len(ns) > 1:
        raise SapInputError(f"confirmatory comparisons have different n: {sorted(ns)}")
    n2 = {v[metric]["two_level"]["n_scenarios"] for f in out.values() for v in f["comparisons"].values()
          if v[metric]["two_level"]["p"] is not None}
    if len(n2) > 1:
        raise SapInputError(f"confirmatory comparisons have different two-level n: {sorted(n2)}")
    return out


def ca_prec(row: dict, tau: float) -> int:
    """D11 (c) (the design specification at 3739dd6, fixed 25/09/2026): CA_prec = CA × [call_precision ≥ τ], τ =
    cfg["ca_prec_tau"] fixed before any output; call_precision None (an output without project calls, n_calls = 0)
    counts as 0. CA itself is unchanged: CA_prec only feeds the pre-registered sensitivity analysis."""
    cp = row.get("call_precision")
    return int(bool(row.get("ca")) and cp is not None and cp >= tau)


def ca_prec_denominator_errors(conf: dict, conf_prec: dict) -> list[str]:
    """CA_prec is read over CA's denominator (D11 (c)); every comparison whose CA_prec n
    (scenarios, clusters, two-level scenarios) is not CA's, as one message each (sap_run refuses them: a CA_prec run on
    other records, e.g. without the exclusions or the compose-dropped records, would pass unseen)."""
    n = lambda c: (c["n_scenarios"], c["n_clusters"], c["two_level"]["n_scenarios"])
    out = []
    for fam, f in conf.items():
        for name, v in f["comparisons"].items():
            p = conf_prec.get(fam, {}).get("comparisons", {}).get(name, {}).get("ca_prec")
            if p is None or n(p) != n(v["ca"]):
                out.append(f"{fam} · {name}: CA_prec n {None if p is None else n(p)} is not CA's {n(v['ca'])} "
                           "(scenarios, clusters, two-level)")
    return out


def with_ca_prec(rows: list[dict], tau: float) -> list[dict]:
    """The rows with their ca_prec (a filled row, CA = 0, scores 0)."""
    return [{**r, "ca_prec": ca_prec(r, tau)} for r in rows]


def ca_prec_differences(conf: dict, conf_prec: dict) -> list[str]:
    """D11 (c): every comparison, then every RQ, whose conclusion on CA_prec is not its conclusion on CA, whatever the
    direction (no selection), as one Vietnamese sentence each for SAP_REPORT; one sentence when there is none."""
    out = []
    for fam, f in conf.items():
        p = conf_prec.get(fam, {"comparisons": {}, "rq_verdict": None})
        for name, v in f["comparisons"].items():
            other = p["comparisons"].get(name, {}).get("verdict")
            if other != v["verdict"]:
                out.append(f"{fam} · {name}: kết luận khác nhau giữa CA ({v['verdict']}) và CA_prec ({other}).")
        if p["rq_verdict"] != f["rq_verdict"]:
            out.append(f"{fam}: kết luận mức RQ khác nhau giữa CA ({f['rq_verdict']}) và CA_prec ({p['rq_verdict']}).")
    return out or ["Kết luận theo CA_prec trùng với CA ở mọi phép so và mọi RQ."]


def controls_ca_prec(controls: list[dict], keep: set[str], cfg: dict) -> dict:
    """D11 (fixed 25/09/2026), descriptive: for each control of cfg["ca_prec_controls"] in controls.jsonl (which
    carries n_calls and call_precision, D11), over the usable records (`keep`) whose CA is defined: n, the CA rate and
    the CA_prec rate at each τ of cfg["ca_prec_control_taus"] (None when n = 0, a control absent from the run). A
    counted row without call_precision stops the analysis."""
    out = {}
    for name in cfg["ca_prec_controls"]:
        xs = [x for x in controls if x.get("control") == name and x["record_id"] in keep and x.get("ca_defined")]
        if any("call_precision" not in x for x in xs):
            raise SapInputError(f"controls.jsonl rows of {name} lack call_precision (D11)")
        rate = lambda f: sum(f(x) for x in xs) / len(xs) if xs else None
        out[name] = {"n": len(xs), "ca": rate(lambda x: bool(x.get("ca"))),
                     **{f"ca_prec@{t:g}": rate(lambda x, t=t: ca_prec(x, t)) for t in cfg["ca_prec_control_taus"]}}
    return out


# ----- descriptive -------------------------------------------------------------------------------------------------------

LEN_BUCKETS = ("le8192", "gt8192")  # 3C's len_bucket (describe.row): prompt_tokens <= 8192, > 8192


def len_strata(rows: list[dict], cfg: dict, dropped: set[str] = frozenset()) -> tuple[dict[str, str], int]:
    """The prompt length groups of §2.4 of the design specification (P26, 06/10/2026, before any test output; by the
    anchor model's Full-aware prompt, the F2 of the length rule and of dose_response): record_id -> the len_bucket of
    its real (not filled_missing) Full-aware rows of the anchor model, variant main; and how many records of `rows` not
    compose-dropped (`dropped`) have no such row with a len_bucket, so are in neither group. Every condition of a record
    falls in its record's group, so a comparison keeps its pairs. One prompt serves every seed: Full-aware rows of one
    record in two buckets, or in a bucket 3C does not write (LEN_BUCKETS), stop the analysis."""
    key = (cfg["anchor_model"], "full_aware", "main")
    got: dict = defaultdict(set)
    for r in rows:
        if run_key(r) == key and not r.get("filled_missing") and r.get("len_bucket") is not None:
            got[r["record_id"]].add(r["len_bucket"])
    bad = sorted(k for k, v in got.items() if len(v) > 1 or not v <= set(LEN_BUCKETS))
    if bad:
        raise SapInputError(f"{len(bad)} records have Full-aware rows in two len_bucket values or in one that is not "
                            f"{'/'.join(LEN_BUCKETS)}, e.g. {bad[0]}: {sorted(got[bad[0]])}")
    strata = {k: next(iter(v)) for k, v in got.items()}
    return strata, len({r["record_id"] for r in rows} - set(strata) - set(dropped))


def robustness(rows: list[dict], cfg: dict, excluded: set[str], dropped: set[str] = frozenset(),
               pre_t: set[str] = frozenset()) -> dict:
    """The exploratory robustness section ("độ vững"): every comparison of the families,
    without its family label, on each view of the rows: the subsets of cfg["subsets"] (sub_<name> from subset_flags),
    the strata by repo and by language, the preexisting groups, each repo left out, and the rows without the records
    `pre_t` (the frozen pre-T list for the partition the rows are from, pre_t_flags), then (P26, added 06/10/2026,
    before any test output, and appended last so the views before keep their seeds) the prompt length groups of
    len_strata, len_bucket=<b> for each of LEN_BUCKETS. Each gives n, the
    estimate and the 95% cluster-bootstrap CI only: no p, no Holm, no verdict; it never yields a conclusion (the four
    subsets, strata and groups left the Holm families here)."""
    views = {f"subset:{s}": [r for r in rows if r.get(f"sub_{s}") is True] for s in cfg["subsets"]}
    for f in ("repo", "language"):
        for g in sorted({r.get(f) for r in rows if r.get(f)}):
            views[f"{f}={g}"] = [r for r in rows if r.get(f) == g]
    for v in (True, False):
        views[f"preexisting={v}"] = [r for r in rows if bool(r.get("preexisting")) == v]
    for repo in sorted({r["repo"] for r in rows}):
        views[f"without_repo={repo}"] = [r for r in rows if r["repo"] != repo]
    views["without_pre_t"] = [r for r in rows if r["record_id"] not in pre_t]
    strata = len_strata(rows, cfg, dropped)[0]
    for b in LEN_BUCKETS:
        views[f"len_bucket={b}"] = [r for r in rows if strata.get(r["record_id"]) == b]
    model, pairs = cfg["anchor_model"], [p for ps in cfg["families"].values() for p in ps]
    out = {}
    for i, (name, rs) in enumerate(views.items()):
        sv = seed_values(rs, "ca", usable("ca", excluded, dropped=dropped))
        out[name] = {}
        for j, (a, b) in enumerate(pairs):
            c = as_dict(compare(pooled(per_seed_pairs(sv, (model, a, "main"), (model, b, "main"))),
                                cfg["reps_bootstrap"], 0, cfg["seed"] + 20000 + 100 * i + j, with_p=False))
            out[name][f"{a}_vs_{b}"] = {k: c[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")}
    return out


def level(values: dict[str, tuple[str, float]], cfg: dict, seed: int) -> dict:
    return as_dict(compare(list(values.values()), cfg["reps_bootstrap"], 0, seed, with_p=False))


def rq3_full_aware_by_model(rows: list[dict], cfg: dict, excluded: set[str], dropped: set[str] = frozenset()) -> dict:
    """RQ3 (P14): the CA level at Full-aware (variant main) of each model with one, the seeds pooled, as `descriptive`
    has always given it under this key (seed cfg["seed"] + 3000 + the model's rank among the models with a CA run)."""
    ca = scenario_values(rows, "ca", usable("ca", excluded, dropped=dropped))
    return {m: level(ca[(m, "full_aware", "main")], cfg, cfg["seed"] + 3000 + i)
            for i, m in enumerate(sorted({k[0] for k in ca})) if (m, "full_aware", "main") in ca}


RQ3_METRICS = ("ca", "bv", "er", "er@self_contained", "er_trace", "er_trace@self_contained")


def rq3_profile(rows: list[dict], cfg: dict, excluded: set[str], elig: dict,
                dropped: set[str] = frozenset(), *, ca_levels: dict) -> dict:
    """P14 addition (07/10/2026, before any test output), to match the original wording of RQ3 (a quality profile with
    the Executable Rate), descriptive: for each model of cfg["seeds"], the level at Full-aware (variant main) of each of
    RQ3_METRICS: the scenario means pooled over the model's seeds (scenario_values), with the 95% cluster-bootstrap CI,
    no p and no label (RQ3 and H3 carry no label: §5 of the design specification). CA is the very level of
    rq3_full_aware_by_model (`ca_levels`, the same estimate and CI; fixed 07/10/2026: one CI per number, no second
    bootstrap). Denominators as usable(): CA over the usable records with ca_defined, BV without the excl_bv rows, the
    ER metrics over their P11 eligibility (`elig`; @self_contained its own). score_row_errors asks an ER value of the
    anchor model only (a descriptive measure never stops the confirmatory analysis), so an ER metric is never a level
    over part of a model's eligible Full-aware rows, but for the pairs accepted as missing under P4, which
    check_complete fills and which never have an ER value (as in confirmatory). With some eligible rows lacking the
    metric's value it has n 0, a null estimate and CI and a `reason`: "incomplete", with the counts, when the ER run
    covers the model (an er or er_trace value on one of its real Full-aware rows; a row never says both), and "does not
    cover" when it does not. A model with no Full-aware row has that reason for every metric, and a metric with no row
    in its denominator "no row in the denominator". Never 0."""
    out = {}
    for i, model in enumerate(sorted(cfg["seeds"])):
        key = (model, "full_aware", "main")
        mine = [r for r in rows if run_key(r) == key and r["record_id"] not in dropped]
        real = [r for r in mine if not r.get("filled_missing")]  # an accepted-missing pair has no ER value (P4)
        covered = any(r.get(b) is not None for r in real for b in ("er", "er_trace"))
        out[model] = {}
        for j, metric in enumerate(RQ3_METRICS):
            base, _, sub = metric.partition("@")
            reason = None if mine else "no Full-aware row of this model"
            if reason is None and base in ER_METRICS:
                table = (elig or {}).get(ER_METRICS[base] + ("@self_contained" if sub else ""), {})
                eligible = [r for r in real if table.get(r["record_id"])]
                missing = sum(r.get(base) is None for r in eligible)
                if missing and not covered:
                    reason = "no ER value: the ER run does not cover this model"
                elif missing:
                    reason = (f"ER incomplete for this model: {missing} of {len(eligible)} eligible Full-aware rows "
                              "have no value")
            if reason is not None:
                d = None
            elif metric == "ca":
                d = ca_levels.get(model)
            else:
                values = scenario_values(mine, metric, usable(metric, excluded, elig, dropped)).get(key)
                d = values and level(values, cfg, cfg["seed"] + 9000 + 10 * i + j)
            if d:
                out[model][metric] = {**{k: d[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")},
                                      "reason": None}
            else:
                out[model][metric] = {"n_scenarios": 0, "n_clusters": 0, "estimate": None, "ci95": [None, None],
                                      "reason": reason or "no row in the denominator"}
    return out


def descriptive(rows: list[dict], cfg: dict, excluded: set[str], elig: dict, with_deferred: bool = False,
                dropped: set[str] = frozenset(), pre_t: set[str] = frozenset()) -> dict:
    """The pre-declared secondary analyses of §5 of the design specification (outside the Holm families). Those listed
    in cfg["deferred"] (§0b of the design specification: learning curve, ablations) run only with `with_deferred`.
    `by_repo` gives the CA and BV levels of every run per repo (the aux_test report of §0b, beside per_run's lazy
    references v3m / v3all). `robustness` is the exploratory "độ vững" section (the four subsets of §5 of the design
    specification at 4ae2737, strata, groups, each repo left out, without the pre-T records `pre_t`): estimates and CIs
    only, no test. `kappa_ca_trace_action` (P16) adds its attribute-only version (kappa_ca_trace) and `docblock_by_repo`
    the PHP docblock steps next to ER (§2.3 of the design specification, added 26/09/2026). P26 (added 06/10/2026):
    `contrasts` gives the CI of the expected directions of §2.4 of the design specification, `len_strata` the records of
    each prompt length group of `robustness` and how many are in neither. P14 (added 07/10/2026): `rq3_profile` gives
    each model's Full-aware level of CA, BV, ER and ER-trace (rq3_profile)."""
    model, seed, out = cfg["anchor_model"], cfg["seed"], {}
    run = lambda name: with_deferred or name not in cfg.get("deferred", [])
    ca = scenario_values(rows, "ca", usable("ca", excluded, dropped=dropped))
    sv = seed_values(rows, "ca", usable("ca", excluded, dropped=dropped))
    out["levels"] = {"|".join(k): level(v, cfg, seed + 2000 + i) for i, (k, v) in enumerate(sorted(ca.items()))}
    out["by_repo"] = {}
    for j, repo in enumerate(sorted({r["repo"] for r in rows})):
        sub = [r for r in rows if r["repo"] == repo]
        out["by_repo"][repo] = {m: {"|".join(k): level(v, cfg, seed + 2500 + 100 * j + i)
                                    for i, (k, v) in enumerate(sorted(scenario_values(sub, m, usable(m, excluded, dropped=dropped)).items()))}
                                for m in ("ca", "bv")}
    out["rq3_full_aware_by_model"] = rq3_full_aware_by_model(rows, cfg, excluded, dropped)
    out["rq3_profile"] = rq3_profile(rows, cfg, excluded, elig, dropped, ca_levels=out["rq3_full_aware_by_model"])
    out["robustness"] = robustness(rows, cfg, excluded, dropped, pre_t)
    if run("learning_curve"):
        lc = [(v, c) for v in cfg["learning_curve"] + ["main"] for c in cfg["adapter_conditions"]]
        out["learning_curve"] = {f"{c}@{v}": level(ca[(model, c, v)], cfg, seed + 4000 + i) for i, (v, c) in enumerate(lc) if (model, c, v) in ca}
    if run("ablations"):
        out["ablations"] = {v: as_dict(compare(pooled(per_seed_pairs(sv, (model, "full_aware", v), (model, "full_aware", "main"))),
                                               cfg["reps_bootstrap"], 0, seed + 5000 + i, with_p=False))
                            for i, v in enumerate(cfg["ablations"]) if (model, "full_aware", v) in sv}
    out["dose"] = dose_response(rows, cfg, excluded, dropped)
    out["contrasts"] = contrasts(rows, cfg, dropped)
    strata, unplaced = len_strata(rows, cfg, dropped)
    out["len_strata"] = {"records": {b: sum(v == b for v in strata.values()) for b in LEN_BUCKETS},
                         "unplaced": unplaced}
    out["per_run"] = per_run(rows, excluded, dropped)
    out["calls_by_repo"] = calls_by_repo(rows, excluded, dropped)
    out["lazy_by_repo"] = lazy_by_repo(rows, excluded, dropped)
    out["kappa_ca_trace_action"] = kappa_ca_trace(rows, excluded | dropped, elig, model)
    out["docblock_by_repo"] = docblock_by_repo(rows, dropped)
    return out


def bucket(dose: int | None, buckets: list[list]) -> str | None:
    if dose is None:
        return None
    for lo, hi in buckets:
        if dose >= lo and (hi is None or dose <= hi):
            return f"{lo}-{'' if hi is None else hi}"
    return None


def dose_response(rows: list[dict], cfg: dict, excluded: set[str], dropped: set[str] = frozenset()) -> dict:
    """Full-aware minus Full-neutral (anchor, main, paired by seed) by the dose of the record's Full-aware prompt, each
    bucket with the label of §2.4 of the design specification for its F2 − F1 (expected > 0; P26 (e), fixed
    06/10/2026). The rise with the dose is read from
    the bucket CIs, descriptively, with no label: the buckets are disjoint sets of records."""
    model = cfg["anchor_model"]
    dose = {r["record_id"]: r.get("dose") for r in rows if run_key(r) == (model, "full_aware", "main")}
    out = {}
    for i, (lo, hi) in enumerate(cfg["dose_buckets"]):
        name = bucket(lo, [[lo, hi]])
        sv = seed_values([r for r in rows if bucket(dose.get(r["record_id"]), cfg["dose_buckets"]) == name], "ca",
                         usable("ca", excluded, dropped=dropped))
        out[name] = as_dict(compare(pooled(per_seed_pairs(sv, (model, "full_aware", "main"), (model, "full_neutral", "main"))),
                                    cfg["reps_bootstrap"], 0, cfg["seed"] + 6000 + i, with_p=False))
        out[name]["direction"] = direction(out[name]["ci95"], 1, out[name]["n_clusters"])
    return out


# the expected directions of §2.4 of the design specification: F2 < F1, F2 > F1
CONTRASTS = {"hallucinated_rate": -1, "conv_style": 1, "conv_pattern": 1, "conv_naming": 1}


def hallucinated_rate(row: dict) -> float | None:
    """A record's share of made-up members (§2.4 of the design specification): hallucinated / members_used; None when it
    used none."""
    used = row.get("members_used")
    if type(used) is int and used > 0 and row.get("hallucinated") is not None:
        return row["hallucinated"] / used
    return None


def contrasts(rows: list[dict], cfg: dict, dropped: set[str] = frozenset()) -> dict:
    """P26 (06/10/2026, before any test output; §2.4 of the design specification, "chiều kỳ vọng khai trước"),
    descriptive: for each of
    CONTRASTS, Full-aware minus Full-neutral (F2 − F1 with an adapter; anchor model, variant main), paired by seed and
    pooled per scenario as dose_response, with its 95% cluster-bootstrap CI and no p, the expected sign ("< 0" for the
    made-up symbols, "> 0" for the conventions) and the label of `direction`. The rows are those per_run reads for the
    level of the measure: not compose-dropped (`dropped`), not excl_bv, with a value (a filled row has none); the
    made-up symbol rate is hallucinated_rate, so the rows with members_used > 0."""
    model = cfg["anchor_model"]
    a, b = (model, "full_aware", "main"), (model, "full_neutral", "main")
    rs = [{**r, "hallucinated_rate": hallucinated_rate(r)} for r in rows if run_key(r) in (a, b)]
    out = {}
    for i, (m, sign) in enumerate(CONTRASTS.items()):
        keep = lambda r, m=m: r["record_id"] not in dropped and not r["excl_bv"] and r.get(m) is not None
        d = as_dict(compare(pooled(per_seed_pairs(seed_values(rs, m, keep), a, b, True)), cfg["reps_bootstrap"], 0,
                            cfg["seed"] + 8000 + i, with_p=False))
        out[m] = {"expected": "< 0" if sign < 0 else "> 0",
                  **{k: d[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")},
                  "direction": direction(d["ci95"], sign, d["n_clusters"])}
    return out


LAZY_REFS = {"ca_ref_v3m": "ca_above_lazy", "ca_ref_v3all": "ca_above_v3all", "ca_ref_v3names": "ca_above_v3names",
             "ca_ref_top4": "ca_prec_above_top4"}  # D11 (d): the fourth lazy level, stub-v3top4
LAZY_METRIC = {"ca_ref_top4": "ca_prec"}  # "CA_prec vượt mức lười top4": CA_prec = 1 and ca_ref_top4 = 0; else CA


def above_lazy(rows: list[dict], ref: str) -> dict:
    """The rows whose lazy reference `ref` is known, as {n, rate} of the reference and {n, rate} of the metric of
    LAZY_METRIC (CA by default) at 1 with the reference at 0 ("vượt mức lười"; D11 (d) for top4). A row without
    ca_prec (sap.with_ca_prec) stops the analysis when the metric is CA_prec."""
    metric, known = LAZY_METRIC.get(ref, "ca"), [x for x in rows if x.get(ref) is not None]
    if any(metric not in x for x in known):
        raise SapInputError(f"rows without {metric} (sap.with_ca_prec) cannot give the rate above {ref}")
    rate = lambda f: {"n": len(known), "rate": (sum(bool(f(x)) for x in known) / len(known)) if known else None}
    return {"ref": rate(lambda x: x[ref]), "above": rate(lambda x: x[metric] and not x[ref])}


def above_top4(rows: list[dict], cfg: dict, excluded: set[str], dropped: set[str] = frozenset()) -> dict:
    """D11 (d) (the design specification at 4d8bf2a), descriptive: "CA_prec vượt mức lười top4" for the runs of the
    RQ1/RQ2 comparisons, per condition of the families in order (anchor model, variant main, the seeds pooled): {n,
    rate} of CA_prec = 1 and ca_ref_top4 = 0 over the usable CA rows (not excluded, not compose-dropped, CA defined)
    whose ca_ref_top4 is known (above_lazy). No test."""
    keep = usable("ca", excluded, dropped=dropped)
    conds = list(dict.fromkeys(c for ps in cfg["families"].values() for pair in ps for c in pair))
    return {c: above_lazy([r for r in rows if run_key(r) == (cfg["anchor_model"], c, "main") and keep(r)],
                          "ca_ref_top4")["above"] for c in conds}
# the PrestaShop note of D11 (b) (design specification §2.2b at 3739dd6)
REPO_NOTES = {"PrestaShop/PrestaShop": "Ở PrestaShop, CA dễ đạt chỉ bằng các lời gọi hạ tầng (D11): đọc CA ở repo này "
                                       "cùng call_precision và CA_prec."}  # printed beside SAP_REPORT's call table


def quantile(xs: list[float], q: float) -> float | None:
    """The q-quantile of `xs` by linear interpolation between order statistics (numpy's default); None when empty."""
    if not xs:
        return None
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = int(h)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (h - lo)


def call_stats(rows: list[dict]) -> dict:
    """D11 (b), descriptive: call_precision (median, Q1, Q3) and n_calls (median, p90) over `rows` with an output (not
    filled by check_complete) and a defined value (call_precision is None when the output has no project call)."""
    outs = [r for r in rows if not r.get("filled_missing")]
    cp = [r["call_precision"] for r in outs if r.get("call_precision") is not None]
    nc = [r["n_calls"] for r in outs if r.get("n_calls") is not None]
    return {"call_precision": {"n": len(cp), "median": quantile(cp, 0.5), "q1": quantile(cp, 0.25),
                               "q3": quantile(cp, 0.75)},
            "n_calls": {"n": len(nc), "median": quantile(nc, 0.5), "p90": quantile(nc, 0.9)}}


def calls_by_repo(rows: list[dict], excluded: set[str], dropped: set[str] = frozenset()) -> dict:
    """D11 (b): call_stats per repo and per condition (model|condition|variant, the seeds pooled) over the CA rows (not
    compose-dropped, not excluded, CA defined)."""
    keep = usable("ca", excluded, dropped=dropped)
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if keep(r):
            acc[r["repo"]]["|".join(run_key(r))].append(r)
    return {repo: {run: call_stats(rs) for run, rs in sorted(by.items())} for repo, by in sorted(acc.items())}


def lazy_by_repo(rows: list[dict], excluded: set[str], dropped: set[str] = frozenset()) -> dict:
    """The lazy references of the per-repo CA table: per repo and run (model|condition|variant,
    the seeds pooled), each reference of LAZY_REFS that the group's CA rows carry (not dropped, not excluded, CA
    defined), with its "vượt" rate (above_lazy)."""
    keep = usable("ca", excluded, dropped=dropped)
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if keep(r):
            acc[r["repo"]]["|".join(run_key(r))].append(r)
    out: dict = {}
    for repo, by in sorted(acc.items()):
        for run, rs in sorted(by.items()):
            refs = out.setdefault(repo, {}).setdefault(run, {})
            for ref, above in LAZY_REFS.items():
                if any(ref in x for x in rs):
                    lazy = above_lazy(rs, ref)
                    refs[ref], refs[above] = lazy["ref"], lazy["above"]
    return out


def per_run(rows: list[dict], excluded: set[str], dropped: set[str] = frozenset()) -> dict:
    """Rates per run and seed (record level, with denominators): CA and its lazy references (LAZY_REFS: stub-v3m, and
    stub-v3all, stub-v3names (D11 (a)) and stub-v3top4 (D11 (d)) when the score rows carry them), each over the CA rows
    whose reference is known (3C writes None where it has no control row, e.g. Ruby; a filled row has none), with its
    "CA above the lazy writer" rate (CA = 1 and the reference = 0; for top4, "CA_prec vượt mức lười top4": CA_prec = 1
    and ca_ref_top4 = 0, above_lazy); the spread of call_precision and n_calls over the CA rows (call_stats; D11 (b));
    BV, syntax, conditions 2-6 on parsed code (a 3C choice: always next to the syntax rate), extraction status (a count
    per status of EXTRACT_STATUSES over the rows with an output, whose number `n` they sum to), the rate of
    fence_truncated over the rows with a text (not None), the rate of php_tag_stripped over the PHP rows with a text
    (not None; §1.4 of the design specification, added 26/09/2026), and (§2.2 of the design specification, added
    27/09/2026) over the rows with a text the share with at least one of each of READER_COUNTS and their sum, and the
    rate of er_status "ERROR" (the output made the ER run fail to load, e.g. an attribute Behat rejects) over the rows
    with an er_status, cap hits, open resolution, hallucinated members, conventions, filled missing rows; and (P26,
    06/10/2026; §2.4 of the design specification) the rates of duplicate_reuse (the output rewrites a step REUSE already
    defines) and of exact (a verbatim match of the gold answer after whitespace normalisation) over the rows with an
    output and the column, excl_bv rows out as for `extract`. The `dropped` records (compose_dropped) are out of every
    rate."""
    groups: dict = defaultdict(list)
    for r in rows:
        if r["record_id"] not in dropped:
            groups["|".join(run_key(r)) + "|" + r["seed"]].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        ca_rows = [r for r in rs if r["record_id"] not in excluded and r.get("ca_defined")]
        rs = [r for r in rs if not r["excl_bv"]]  # BV and its conditions: design specification §2.2's denominator
        parsed = [r for r in rs if r.get("bv_syntax")]
        outputs = [r for r in rs if not r.get("filled_missing")]  # a filled row has no output
        rate = lambda xs, f: {"n": len(xs), "rate": (sum(bool(f(x)) for x in xs) / len(xs)) if xs else None}
        avg = lambda xs, k: (lambda v: sum(v) / len(v) if v else None)([x[k] for x in xs if x.get(k) is not None])
        refs = {}
        for ref, above in LAZY_REFS.items():
            if ref == "ca_ref_v3m" or any(ref in x for x in ca_rows):
                lazy = above_lazy(ca_rows, ref)
                refs[ref], refs[above] = lazy["ref"], lazy["above"]
        out[g] = {"ca": rate(ca_rows, lambda x: x["ca"]), **refs, **call_stats(ca_rows),
                  "bv": rate(rs, lambda x: x.get("bv")), "syntax": rate(rs, lambda x: x.get("bv_syntax")),
                  **{f"{c}_given_parsed": rate(parsed, lambda x, c=c: x.get(f"bv_{c}"))
                     for c in ("binding", "no_ambiguity", "arity", "grounded", "resolved")},
                  "extract": {**{s: sum(x.get("extract_status") == s for x in outputs) for s in EXTRACT_STATUSES},
                              "n": len(outputs)},
                  **{k: rate([x for x in outputs if x.get(k) is not None], lambda x, k=k: x[k])
                     for k in ("duplicate_reuse", "exact")},
                  "fence_truncated": rate([x for x in rs if x.get("fence_truncated") is not None],
                                          lambda x: x["fence_truncated"]),
                  "php_tag_stripped": rate([x for x in rs if x.get("language") == "php"
                                            and x.get("php_tag_stripped") is not None],
                                           lambda x: x["php_tag_stripped"]),
                  **{k: {**rate([x for x in rs if x.get(k) is not None], lambda x, k=k: x[k] > 0),
                         "sum": sum(x[k] for x in rs if x.get(k) is not None)} for k in READER_COUNTS},
                  "er_error": rate([x for x in rs if x.get("er_status") is not None], lambda x: x["er_status"] == "ERROR"),
                  "cap_hit": rate([x for x in rs if x.get("cap_hit") is not None], lambda x: x["cap_hit"]),
                  "open_resolution": rate(parsed, lambda x: x.get("open_resolution")),
                  "hallucinated_per_member": sum(x.get("hallucinated") or 0 for x in rs) / max(1, sum(x.get("members_used") or 0 for x in rs)),
                  "filled_missing": sum(bool(x.get("filled_missing")) for x in rs),
                  **{k: avg(rs, k) for k in ("conv_style", "conv_pattern", "conv_naming", "rouge_l")}}
    return out


def agreement_ca_trace(rows: list[dict], excluded: set[str], ok: dict[str, bool], anchor: str) -> dict:
    """Cohen's kappa of CA against ER-trace's action verdict over eligible records with both defined, per run, and
    pooled ("ALL") over the runs of the `anchor` model only (P16, fixed 07/10/2026: the ER run of the other models at
    Full-aware serves RQ3 and adds their own rows; it does not change what the registered pooled measure means)."""
    pairs: dict = defaultdict(list)
    for r in rows:
        if r["record_id"] in excluded or not r.get("ca_defined") or r.get("trace_action") is None or not ok.get(r["record_id"]):
            continue
        pairs["|".join(run_key(r)) + "|" + r["seed"]].append((int(r["ca"]), int(r["trace_action"])))
        if r["model"] == anchor:
            pairs["ALL"].append((int(r["ca"]), int(r["trace_action"])))
    return {k: kappa(v) for k, v in sorted(pairs.items())}


# outputs with php_docblock_steps == 0 (design specification §2.3: the ER image registers these)
ATTRIBUTE_ONLY = "attribute_only"


def kappa_ca_trace(rows: list[dict], excluded: set[str], elig: dict[str, dict[str, bool]], anchor: str) -> dict:
    """P16: agreement_ca_trace (pooled over the `anchor` model's runs) for each eligibility, trace and
    trace@self_contained, each followed by its version over the outputs with php_docblock_steps == 0 (k@attribute_only;
    added 26/09/2026, §2.3 of the design specification: the ER image registers only PHP 8 attribute steps, a docblock
    step is undefined there while BV and CA take both forms); an output with docblock steps, or without a count, is out
    of that version."""
    attr = [r for r in rows if type(r.get("php_docblock_steps")) is int and r["php_docblock_steps"] == 0]
    out = {}
    for k in ("trace", "trace@self_contained"):
        out[k] = agreement_ca_trace(rows, excluded, elig.get(k, {}), anchor)
        out[f"{k}@{ATTRIBUTE_ONLY}"] = agreement_ca_trace(attr, excluded, elig.get(k, {}), anchor)
    return out


def docblock_by_repo(rows: list[dict], dropped: set[str] = frozenset()) -> dict:
    """§2.3 of the design specification (added 26/09/2026), descriptive, next to ER: per repo and run
    (model|condition|variant, the seeds pooled), over the rows not compose-dropped (`dropped`) with a php_docblock_steps
    (PHP outputs with a text): n, the rate of outputs with >= 1 docblock step, the docblock and the attribute steps
    summed; and ER {n, rate} over the rows with an er value. Groups with neither are left out. No test."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["record_id"] not in dropped:
            acc[r["repo"]]["|".join(run_key(r))].append(r)
    out: dict = {}
    for repo, by in sorted(acc.items()):
        for run, rs in sorted(by.items()):
            php = [r for r in rs if r.get("php_docblock_steps") is not None]
            er = [r["er"] for r in rs if r.get("er") is not None]
            if php or er:
                out.setdefault(repo, {})[run] = {
                    "n": len(php), "docblock_rate": sum(r["php_docblock_steps"] >= 1 for r in php) / len(php)
                    if php else None, "docblock_steps": sum(r["php_docblock_steps"] for r in php),
                    "attribute_steps": sum(r.get("php_attribute_steps") or 0 for r in php),
                    "er": {"n": len(er), "rate": sum(er) / len(er) if er else None}}
    return out


# ----- dev: MDE and the configuration choice -----------------------------------------------------------------------------

def dev_rows_of(dev_rows: list[dict], cfg: dict) -> list[dict]:
    """The anchor model's rows of the locked dev variant (fixed 24/09/2026: each dev variant is scored
    into its own file, and the files of other variants may be read with it): rows of other variants are left out; no
    row of the locked variant is an error."""
    variant = cfg["dev"]["variant"]
    out = [r for r in dev_rows if r["model"] == cfg["anchor_model"] and (r.get("variant") or "main") == variant]
    if not out:
        found = sorted({r.get("variant") or "main" for r in dev_rows if r["model"] == cfg["anchor_model"]})
        raise SapInputError(f"no dev rows of the locked variant {variant!r} (anchor model), found {found}")
    return out


def mde_from_dev(dev_rows: list[dict], cfg: dict, excluded: set[str], clusters_target: int, scenarios_target: int,
                 calibration: dict | None = None) -> dict:
    """MDE of every confirmatory comparison from the dev pilot (anchor model, the locked dev variant, one seed), with
    the family's strictest Holm level. Test (1): stats.mde (one seed, so no seed averaging: conservative). Test (2):
    MDE(1) x r, with r of the comparison's type ("base" or "adapter") from sap_calibrate at tau = 0 and, for reference,
    tau = 0.04 (fixed 24/09/2026); `mde_k_seeds@...` scales MDE(1) to the K-seed mean the same way, to compare both
    tests on one footing. Published before any test output to honour the pre-registration only: it never decides whether
    the test runs."""
    rows = dev_rows_of(dev_rows, cfg)
    variant = cfg["dev"]["variant"]
    sv = seed_values(rows, "ca", usable("ca", excluded))
    model, out = cfg["anchor_model"], {}
    for fam, pairs in cfg["families"].items():
        for i, (a, b) in enumerate(pairs):
            d = pooled(per_seed_pairs(sv, (model, a, variant), (model, b, variant)))
            m = mde(d, clusters_target, scenarios_target, cfg["alpha"] / len(pairs), cfg["power"],
                    cfg["reps_bootstrap"], cfg["seed"] + 7000 + i) if d else None
            if m and calibration:
                kind = "base" if b in cfg["base_conditions"] else "adapter"
                m["type"] = kind
                for name, table in (("mde_two_level", calibration["r"]), ("mde_k_seeds", calibration["r_sign_flip"])):
                    m.update({f"{name}@{t}": m["mde"] * r for t, r in sorted(table[kind].items())})
            out[f"{fam}:{a}_vs_{b}"] = m
    return out


def select_dev_config(dev_rows: list[dict], cfg: dict, excluded: set[str]) -> dict:
    """The dev configuration choice: keep the baseline compose configuration unless another one scores at least `margin`
    scenarios more CA at Full-aware on dev (anchor model); a smaller difference is a tie and keeps the baseline; among
    several better ones the highest wins, ties by name."""
    rule = cfg["dev_selection"]
    rows = [r for r in dev_rows if r["model"] == cfg["anchor_model"] and r["condition"] == rule["condition"]]
    vals = scenario_values(rows, "ca", usable("ca", excluded))
    score = {k[2]: sum(v for _, v in sc.values()) for k, sc in vals.items()}
    if rule["baseline"] not in score:
        raise SapInputError(f"no dev rows for the baseline {rule['baseline']!r}")
    best = min((v for v in score if score[v] >= score[rule["baseline"]] + rule["margin_scenarios"]),
               key=lambda v: (-score[v], v), default=rule["baseline"])
    return {"scores": dict(sorted(score.items())), "baseline": rule["baseline"], "margin": rule["margin_scenarios"], "choice": best}


# ----- SemV and answers --------------------------------------------------------------------------------------------------

def semv_sample(rows: list[dict], cfg: dict, excluded: set[str],
                texts: dict[tuple[str, str, str], str] | None = None) -> tuple[list[dict], list[dict]]:
    """(blind sheet, key) for the SemV rubric: `n_records` distinct records with outputs in every listed condition
    (anchor model, main), ordered by sha256 of the record id; one seed per record chosen by that hash, the same for
    every condition; items sorted by their id. The sheet holds the output text (from `texts`, keyed by (record_id,
    condition, seed)) but never the condition, which only the key holds."""
    spec = cfg["semv"]
    model, seeds = cfg["anchor_model"], cfg["seeds"][cfg["anchor_model"]]
    have: dict = defaultdict(set)
    for r in rows:
        if r["model"] == model and (r.get("variant") or "main") == "main" and r["record_id"] not in excluded and not r.get("filled_missing"):
            have[r["record_id"]].add((r["condition"], r["seed"]))
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    records = sorted((rid for rid, got in have.items() if all(any(c == cond for c, _ in got) for cond in spec["conditions"])), key=h)
    blind, key = [], []
    for rid in records[:spec["n_records"]]:
        sd = seeds[int(h(rid), 16) % len(seeds)]
        for cond in spec["conditions"]:
            use = sd if (cond, sd) in have[rid] else sorted(s for c, s in have[rid] if c == cond)[0]
            item = h(f"{rid}|{cond}|{use}")[:12]
            blind.append({"item_id": item, "record_id": rid, "text": (texts or {}).get((rid, cond, use))})
            key.append({"item_id": item, "record_id": rid, "condition": cond, "seed": use})
    return sorted(blind, key=lambda x: x["item_id"]), sorted(key, key=lambda x: x["item_id"])


ANSWERS = {
    "supported": "{fam}: Full-aware cao hơn {other} về CA {est:+.3f} (KTC 95% [{lo:+.3f}, {hi:+.3f}]; p Holm đảo dấu = {p1:.4f}; KTC hai tầng [{lo2:+.3f}, {hi2:+.3f}], p Holm hai tầng = {p2:.4f}). Cả hai phép thử đều có ý nghĩa: giả thuyết được ủng hộ, và kết quả khái quát qua các lần huấn luyện.",
    "conditional_only_positive": "{fam}: với các adapter đã huấn luyện, Full-aware cao hơn {other} về CA {est:+.3f} (KTC 95% [{lo:+.3f}, {hi:+.3f}]; p Holm đảo dấu = {p1:.4f}). Nhưng bootstrap hai tầng không có ý nghĩa (KTC [{lo2:+.3f}, {hi2:+.3f}]; p Holm = {p2:.4f}): kết quả chưa khái quát qua các lần huấn luyện, nên KHÔNG được coi là ủng hộ.",
    "conditional_only_negative": "{fam}: với các adapter đã huấn luyện, Full-aware thấp hơn {other} về CA {est:+.3f} (KTC 95% [{lo:+.3f}, {hi:+.3f}]; p Holm đảo dấu = {p1:.4f}). Nhưng bootstrap hai tầng không có ý nghĩa (KTC [{lo2:+.3f}, {hi2:+.3f}]; p Holm = {p2:.4f}): kết quả chưa khái quát qua các lần huấn luyện, nên không tính là trái chiều, nhưng được nêu và bàn trong đề án.",
    "contrary": "{fam}: Full-aware thấp hơn {other} về CA {est:+.3f} (KTC 95% [{lo:+.3f}, {hi:+.3f}]; p Holm đảo dấu = {p1:.4f}; KTC hai tầng [{lo2:+.3f}, {hi2:+.3f}], p Holm hai tầng = {p2:.4f}). Cả hai phép thử đều có ý nghĩa theo chiều ngược: kết quả trái chiều giả thuyết.",
    "no_difference": "{fam}: không phát hiện khác biệt về CA giữa Full-aware và {other} (ước lượng {est:+.3f}, KTC 95% [{lo:+.3f}, {hi:+.3f}]; p Holm đảo dấu = {p1:.4f}; p Holm hai tầng = {p2:.4f}). Trên cùng mặt bằng 5 seed, MDE là {mde_k} với phép (1) và {mde2} với phép (2); đề án không loại trừ được một hiệu ứng nhỏ hơn các MDE đó.",
    "not_run": "{fam}: phép so Full-aware với {other} không có kịch bản nào có dữ liệu ở cả hai điều kiện; phải báo lượt chạy bị thiếu.",
}
RQ_ANSWERS = {
    "full": "{fam} được ủng hộ đầy đủ: {n}/{n} phép so supported.",
    "partial": "{fam} được ủng hộ một phần: {s}/{n} phép so supported, không phép so nào trái chiều.",
    "not_supported": "{fam} không được ủng hộ: không phép so nào supported.",
    "contrary": "{fam} trái chiều: ít nhất một phép so có ý nghĩa theo chiều ngược giả thuyết ở cả hai phép thử.",
}
RQ_NEGATIVE_NOTE = (" Lưu ý: với các adapter đã huấn luyện, Full-aware thấp hơn ở {names} (chỉ phép đảo dấu có ý nghĩa);"
                    " điều này chưa khái quát qua các lần huấn luyện và được nêu, bàn trong đề án.")


ROBUST_NOTE = " Độ vững (CA và CA_prec): {label}."  # P22 (27/09/2026): after every answer, with robust_verdicts


def answers(conf: dict, mdes: dict | None = None, robust: dict | None = None) -> list[str]:
    """The pre-written answers, comparison by comparison and then per RQ, filled with their numbers; with `robust`
    (robust_verdicts), each ends with its label (P22, fixed 27/09/2026: robust_label)."""
    note = lambda r: ROBUST_NOTE.format(label=robust_label(r)) if robust else ""
    shown = lambda m, k: "(chưa công bố)" if not m or m.get(k) is None else f"{m[k]:.3f}" + (" (suy biến)" if m.get("degenerate") else "")
    num = lambda x: float("nan") if x is None else x
    out = []
    for fam, f in conf.items():
        for name, v in f["comparisons"].items():
            other, c = name.split("_vs_")[1], v["ca"]
            r = robust["comparisons"][fam][name] if robust else None
            if v["verdict"] == "not_run":
                out.append(ANSWERS["not_run"].format(fam=fam, other=other) + note(r))
                continue
            m = (mdes or {}).get(f"{fam}:{name}")
            t = c["two_level"]
            out.append(ANSWERS[v["verdict"]].format(fam=fam, other=other, est=c["estimate"], lo=c["ci95"][0], hi=c["ci95"][1],
                                                    p1=c["p_holm"], lo2=num(t["ci95"][0]), hi2=num(t["ci95"][1]), p2=num(t["p_holm"]),
                                                    mde_k=shown(m, "mde_k_seeds@tau_0"),
                                                    mde2=shown(m, "mde_two_level@tau_0")) + note(r))
        vs = {name: v["verdict"] for name, v in f["comparisons"].items()}
        text = RQ_ANSWERS[f["rq_verdict"]].format(fam=fam, n=len(vs), s=list(vs.values()).count("supported"))
        neg = [name.split("_vs_")[1] for name, v in vs.items() if v == "conditional_only_negative"]
        out.append(text + (RQ_NEGATIVE_NOTE.format(names=", ".join(neg)) if neg else "")
                   + note(robust["rq"][fam] if robust else None))
    return out


# ----- the lock ----------------------------------------------------------------------------------------------------------

def sha256_of(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.exists() else None


def listed_sha256(path: Path) -> dict[str, str]:
    """The entries of a sums file (PHASE2_SHA256SUMS: one "<sha256>  <path>" per line, paths such as
    out/phase2/records/test.jsonl), as path -> lowercase sha256."""
    lines = [l.split(maxsplit=1) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {p.strip(): h.lower() for h, p in lines}


def listed_at(listed: dict[str, str], suffix: str) -> str | None:
    """The sha256 of the one listed path that ends with "/" + `suffix` ("records/test.jsonl" matches
    out/phase2/records/test.jsonl, not out/phase2/myrecords/test.jsonl); None when no path does, or more than one."""
    hits = [h for p, h in listed.items() if p.endswith("/" + suffix)]
    return hits[0] if len(hits) == 1 else None


def lock_inputs(a) -> dict[str, Path | list[Path]]:
    """The inputs the lock covers (§5 of the design specification, fixed 24/09/2026), by name, from the arguments
    sap_lock and sap_run share (an argparse namespace): the config, MDE.json, CALIBRATION.json, DEV_SELECTION.json, the
    dev score files and dev exclusions sap_mde read, the test controls, their exclusions and their future_methods.jsonl
    (3C's future methods per record and target path, which decide CA), compose's excluded.jsonl, Phase 2's
    records/test.jsonl, sensitivity_subsets.json, self_contained.jsonl, Phase 2's PHASE2_SHA256SUMS, the ER control
    results and the manifest.jsonl of their run (write_overlays), the aux_test controls (controls, exclusions, future
    methods) and records/aux_test.jsonl (the descriptive-only run reads them), the frozen pre-T list
    (FLAGGED_PRE_T.tsv), and, when given, 3C's dose table (--dose; added 25/09/2026)."""
    return {"config": a.config, "mde": a.mde, "calibration": a.calibration, "dev_selection": a.dev_selection,
            "dev_scores": list(a.dev_scores), "dev_exclusions": a.dev_controls_dir / "exclusions.jsonl",
            "controls": a.controls_dir / "controls.jsonl", "exclusions": a.controls_dir / "exclusions.jsonl",
            "future_methods": a.controls_dir / "future_methods.jsonl", "compose_excluded": a.compose_excluded,
            "records": a.records, "subsets": a.subsets, "self_contained": a.self_contained, "phase2_sums": a.phase2_sums,
            "er_controls": list(a.er_controls), "er_manifest": a.er_manifest,
            "aux_controls": a.aux_controls_dir / "controls.jsonl",
            "aux_exclusions": a.aux_controls_dir / "exclusions.jsonl",
            "aux_future_methods": a.aux_controls_dir / "future_methods.jsonl", "aux_records": a.aux_records,
            "pre_t_flags": a.pre_t_flags, **({"dose": a.dose} if getattr(a, "dose", None) is not None else {})}


def digest_inputs(inputs: dict[str, Path | list[Path] | None]) -> dict[str, str | None]:
    """sha256 of every locked input; a list is hashed file by file in order; a missing file is None."""
    out = {}
    for name, v in sorted(inputs.items()):
        if isinstance(v, list):
            for i, p in enumerate(v):
                out[f"{name}[{i}]"] = sha256_of(p)
        else:
            out[name] = sha256_of(v)
    return out


def provenance_errors(mde: dict, calibration: dict, digests: dict[str, str | None], selection: dict, cfg: dict,
                      dev_chosen: list[str], phase2_listed: dict[str, str]) -> list[str]:
    """Why the lock must wait (fixed 24/09/2026): MDE.json and CALIBRATION.json were not computed from the current
    config and calibration (a config edit after sap_calibrate or sap_mde means rerunning them); MDE.json's dev_inputs
    are not the dev files given to the lock, or its test_inputs not the test controls, exclusions and compose file;
    CALIBRATION.json's inputs are not those three test files; the records file given to the lock (`digests["records"]`)
    is not PHASE2_SHA256SUMS's records/test.jsonl (`phase2_listed`: path -> sha256, from listed_sha256; no such entry,
    or several, is an error), or CALIBRATION.json's records file is not that one (a calibration on records/dev.jsonl,
    listed too, does not pass); the sensitivity_subsets.json given to the lock is not the one PHASE2_SHA256SUMS lists;
    DEV_SELECTION.json's inputs do not hold every dev score file of the chosen variant given to the lock (`dev_chosen`:
    their names in `digests`, "dev_scores[i]"; none is an error) or the dev exclusions given to it; the variant
    DEV_SELECTION.json chose, MDE.json's and the config's are not the same. A missing digest or entry never matches."""
    out = [f"{src} {key} is not the current {name}" for src, d, key, name in
           (("MDE.json", mde, "config_sha256", "config"), ("MDE.json", mde, "calibration_sha256", "calibration"),
            ("CALIBRATION.json", calibration, "config_sha256", "config")) if d.get(key) != digests.get(name)]
    dev = sorted(v for k, v in digests.items() if k.startswith("dev_scores[") or k == "dev_exclusions")
    if sorted((mde.get("dev_inputs") or {}).values()) != dev:
        out.append("MDE.json dev_inputs are not the dev score files and dev exclusions given here")
    test = sorted(digests.get(k) for k in ("controls", "exclusions", "compose_excluded"))
    if sorted((mde.get("test_inputs") or {}).values()) != test:
        out.append("MDE.json test_inputs are not the test controls, exclusions and compose file given here")
    cal = calibration.get("inputs") or {}
    stale = [k for k in ("controls", "exclusions", "compose_excluded") if cal.get(k) != digests.get(k)]
    if stale:
        out.append(f"CALIBRATION.json inputs {stale} are not the test files given here")
    records = digests.get("records")
    if records is None or records != listed_at(phase2_listed, "records/test.jsonl"):
        out.append("the records file given here is not PHASE2_SHA256SUMS's records/test.jsonl")
    if records is None or cal.get("records") != records:
        out.append("CALIBRATION.json inputs records is not the records file given here")
    if digests.get("subsets") is None or digests.get("subsets") != listed_at(phase2_listed, "sensitivity_subsets.json"):
        out.append("the subsets file given here is not PHASE2_SHA256SUMS's sensitivity_subsets.json")
    seen = set((selection.get("inputs") or {}).values())
    if not dev_chosen or any(digests.get(k) not in seen for k in dev_chosen):
        out.append("DEV_SELECTION.json inputs do not hold the chosen variant's dev score files given here")
    if digests.get("dev_exclusions") not in seen:
        out.append("DEV_SELECTION.json inputs do not hold the dev exclusions given here")
    variants = {selection.get("choice"), mde.get("dev_variant"), cfg["dev"]["variant"]}
    if len(variants) != 1:
        out.append(f"DEV_SELECTION.json choice, MDE.json dev_variant and config dev.variant differ: {sorted(map(str, variants))}")
    return out


def git_run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Every git command of the anchor checks (added 28/09/2026): run in --repo with no GIT_*
    variable of the caller's environment (none may point git at another repository, object store, graft file or
    replace base) and with GIT_NO_REPLACE_OBJECTS=1, so that a local `git replace` (of a blob, or a graft of parents)
    never changes what a commit holds or what it descends from. Output as bytes."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env)


def history_errors(repo: Path) -> list[str]:
    """Added 28/09/2026: the history git reads in --repo must be the one its commits hold. A `git replace` can swap the
    committed META or lock blob for an edited one, and `git replace --graft` can make a META cut before the lock its
    descendant, all locally and with the commits the registry names (in a probe, RQ1 0.2711 -> 0.5482 at exit
    0). git_run already ignores replace objects; beyond that the run is refused when --repo has any refs/replace/* ref,
    an info/grafts file, or is a shallow clone (its cut history can hide a first write of a file), or when
    GIT_REPLACE_REF_BASE or GIT_GRAFT_FILE is set in the environment (fail closed)."""
    out = [f"{v} is set in the environment: a replace base or graft file is refused"
           for v in ("GIT_REPLACE_REF_BASE", "GIT_GRAFT_FILE") if v in os.environ]
    refs = git_run(repo, "for-each-ref", "--format=%(refname)", "refs/replace/").stdout.decode().split()
    if refs:
        out.append(f"--repo has replace refs ({', '.join(refs[:3])}{', …' if len(refs) > 3 else ''}): a local `git "
                   "replace` can swap what a commit holds or its parents")
    grafts = git_run(repo, "rev-parse", "--git-path", "info/grafts").stdout.decode().strip()
    if grafts and (Path(grafts) if Path(grafts).is_absolute() else repo / grafts).exists():
        out.append(f"--repo has a grafts file ({grafts}): it can fake a commit's parents")
    if git_run(repo, "rev-parse", "--is-shallow-repository").stdout.decode().strip() == "true":
        out.append("--repo is a shallow clone: its cut history can hide an earlier write of a file")
    return out


def committed_bytes(repo: Path, sha: str, rel: str) -> bytes | None:
    """The bytes of `rel` at commit `sha` (git show <sha>:<rel>, replace objects ignored: git_run), or None when the
    commit does not hold it. Since 28/09/2026, sap_run reads the CONTENT of the lock, of each
    SCORES_META and of the accepted_missing file from these bytes, never from the path it was given; the file on disk
    is only compared with them (committed_file_errors)."""
    shown = git_run(repo, "show", f"{sha}:{rel}")
    return shown.stdout if shown.returncode == 0 else None


# ----- the run from a fresh clone of origin, against the registry (added 28/09/2026) --------------------------

LEDGER_COLUMNS = ("LOCK_COMMIT", "sha256(SAP_LOCK.json)", "META_COMMIT", "sha256(SCORES_META.json)", "scores_sha256",
                  "ngày commit của META_COMMIT (+07)")  # after `phần`, in the order of §6 of PREREG_LEDGER.md
RESULTS_IGNORED = ("tool_commit", "generation.manifests[].manifest", "generation.manifests[].coverage",
                   "generation.scores_meta[].file", "accepted_missing.file")  # HEAD of the code and the path of each
# file differ between the working checkout and the fresh clone; every other field of SAP_RESULTS.json must be equal


def ledger_rows(text: str) -> dict[str, list[list[str]]]:
    """The rows of §6 (Neo kết quả SAP) of PREREG_LEDGER.md: `phần` (test, aux_test) -> the cells after it, of each
    Markdown table row between the heading holding "§6" and the next heading."""
    parts = re.split(r"(?m)^#+[^\n]*§6\b[^\n]*\n", text, maxsplit=1)
    table = re.split(r"(?m)^#", parts[1])[0] if len(parts) == 2 else ""
    rows: dict[str, list[list[str]]] = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if cells and cells[0] in ("test", "aux_test"):
            rows.setdefault(cells[0], []).append(cells[1:])
    return rows


def ledger_values(clone: Path, results: dict) -> list[str]:
    """The six values of the §6 row of a run (LEDGER_COLUMNS), from its SAP_RESULTS.json (the two commits, the
    scores_sha256 its META names) and from git in `clone`: the sha256 of the lock and of the META as committed
    (committed_bytes), and the commit date of META_COMMIT (%cI) in +07, ISO 8601 with its offset."""
    lock, meta = results["lock_commit"], results["generation"]["scores_meta_commit"]
    when = dt.datetime.fromisoformat(git_run(clone, "show", "-s", "--format=%cI", meta["sha"]).stdout.decode().strip())
    return [lock["sha"], hashlib.sha256(committed_bytes(clone, lock["sha"], lock["path"]) or b"").hexdigest(),
            meta["sha"], hashlib.sha256(committed_bytes(clone, meta["sha"], meta["paths"][0]) or b"").hexdigest(),
            results["generation"]["scores_meta"][0]["scores_sha256"],
            when.astimezone(dt.timezone(dt.timedelta(hours=7))).isoformat()]


def ledger_errors(rows: dict[str, list[list[str]]], part: str, values: list[str]) -> list[str]:
    """The §6 row of `part` must be exactly one, equal in all six columns to `values` (ledger_values)."""
    mine = rows.get(part, [])
    if len(mine) != 1:
        return [f"§6 of the ledger has {len(mine)} rows for {part}, not one"]
    cells = (mine[0] + [""] * len(LEDGER_COLUMNS))[:len(LEDGER_COLUMNS)]
    return [f"§6 {part}: {col} is {c!r} in the ledger but {v!r} in the run from the fresh clone"
            for col, c, v in zip(LEDGER_COLUMNS, cells, values) if c != v]


def results_differences(got: dict, ref: dict) -> list[str]:
    """The top-level fields where two SAP_RESULTS.json differ, every field compared except RESULTS_IGNORED (the code's
    HEAD and the paths of the files, which differ between the working checkout and the fresh clone)."""
    def strip(r: dict) -> dict:
        r = json.loads(json.dumps(r))
        r.pop("tool_commit", None)
        gen = r.get("generation") or {}
        for m in gen.get("manifests") or []:
            m.pop("manifest", None)
            m.pop("coverage", None)
        for m in gen.get("scores_meta") or []:
            m.pop("file", None)
        if isinstance(r.get("accepted_missing"), dict):
            r["accepted_missing"].pop("file", None)
        return r
    a, b = strip(got), strip(ref)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))

def repo_rel(path: Path, repo: Path) -> str | None:
    """Where `path` sits under --repo (added 28/09/2026), as a fixed
    string taken from the path AS GIVEN (os.path.abspath folds `.` and `..` but follows no symlink), against --repo as
    given or resolved (a --repo reached through a symlink ABOVE it, like macOS's /tmp, is the same checkout); None
    when `path` is not under --repo. The former `path.resolve().relative_to(repo.resolve())` followed a symlink at or
    below the canonical path to wherever it pointed, so a META written once in another folder passed as the
    canonical one."""
    p = Path(os.path.abspath(path))
    for base in (Path(os.path.abspath(repo)), repo.resolve()):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            pass
    return None


def link_errors(repo: Path, rel: str, what: str) -> list[str]:
    """Against a symlink bypass (added 28/09/2026): every step from --repo down to `rel`, each folder and the file,
    must be a real entry and not a symlink. A symlink at out/phase3/scores/test/SCORES_META.json, or one in place of
    the folder scores/test, can send the checks to a META written once elsewhere (in a probe, RQ1 0.2711 -> 0.5482).
    The file must also have a single link (st_nlink 1): a hard link opens no bypass, since the bytes at the fixed path
    are the ones compared and anchored (hard-link probes were refused as not byte-equal), but no
    legitimate run gives the anchored file a second name, so it is refused as well (fail closed)."""
    top = repo.resolve()
    cur = top
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink():
            return [f"{what} {rel}: {cur.relative_to(top).as_posix()} is a symlink under --repo; the file and every "
                    "folder above it must be real entries"]
    if cur.is_file() and cur.stat().st_nlink > 1:
        return [f"{what} {rel} has {cur.stat().st_nlink} hard links; the anchored file must have one"]
    return []


def top_level_errors(repo: Path) -> list[str]:
    """Against a subfolder bypass (added 28/09/2026): --repo must be the top level of its git checkout (git rev-parse
    --show-toplevel, resolved, equals --repo resolved). From a subfolder, `git show <sha>:<rel>` reads <rel> from the
    root while `git log -- <rel>` reads it from the subfolder, so the bytes and the written-once history could come
    from two different files: a META rewritten at the root would pass as written once."""
    top = git_run(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return [f"--repo {repo} is not a git checkout"]
    if Path(top.stdout.decode().strip()).resolve() != repo.resolve():
        return [f"--repo {repo} is not the top level of its git checkout ({top.stdout.decode().strip()}); paths are "
                "read from the top level"]
    return []


def committed_file_errors(path: Path, sha: str, repo: Path, what: str, flag: str,
                          remote: str | None = None, once: bool = False) -> tuple[list[str], dict | None]:
    """A file counts only as committed and published (the lock, accepted_missing.jsonl under design specification §5 at
    7af1673, each SCORES_META): `path` lies inside `repo` and equals, byte for byte, git show <sha>:<its path>; the
    commit is an ancestor of HEAD (the checkout that runs the analysis holds it) and is on a remote branch (git branch
    -r --contains: published, with its push as the independent time anchor), of `remote` when given ("origin" for the
    accepted_missing file). With `once` (added 27/09/2026; the accepted_missing file and each SCORES_META), that commit
    is also the only one of `remote` that GENUINELY writes the path (since 28/09/2026): a commit counts only when its
    blob of `rel` (absent counts as a value; a root commit counts when it has the file) differs from that of EVERY
    parent, so the file was written once and never edited, committed and pushed again. Before this fix, a plain `git log
    --full-history` count also caught a --no-ff merge that carries the file forward unchanged (TREESAME to the branch it
    merges: history simplification, which --full-history disables, is what normally hides such a merge from a path's
    log), so a branch's push and its later `git merge --no-ff` of it into main both counted as a "write" and a
    legitimate file was refused; genuine same-bytes recommits are excluded the same way. Rewriting the remote's history
    (a force push) is the remaining way round the once check, and the registry, written at each push, is its anchor.
    `what` names the file and `flag` the option of the commit in the messages. Since 28/09/2026, --repo must be the top
    level of its checkout (top_level_errors), the path's place under it is a fixed string taken without following
    symlinks (repo_rel), no step down to the file may be a symlink and the file has a single link (link_errors), and the
    file read and the path given to git are that one place (git log with `:(top)`). Since 28/09/2026, the path given
    must also lead to that very file (a `x/..` through a symlink `x` read another one), every git command runs with
    replace objects ignored (git_run), and --repo may hold no replace ref, grafts or shallow cut (history_errors); the
    callers read the file's CONTENT from the commit (committed_bytes). Returns the problems and, when there is none,
    {sha, path, fuller, committed} (git show -s --format=fuller: author and commit dates; %cI: the commit date) for
    SAP_RESULTS and SAP_REPORT."""
    git = lambda *a: git_run(repo, *a)  # replace objects ignored, no GIT_* of the caller's environment
    rel = repo_rel(path, repo)
    if rel is None:
        return [f"{what} {path} is not inside --repo {repo}"], None
    bad = top_level_errors(repo) or history_errors(repo) or link_errors(repo, rel, what)
    if bad:
        return bad, None
    target = repo.resolve() / rel  # the one place checked below; its content is read from the commit, never from disk
    if Path(os.path.realpath(path)) != target:
        return [f"{what} {path} does not lead to {rel} under --repo but to {os.path.realpath(path)}: the file checked and "
                "the file given must be one"], None
    if not target.is_file():
        return [f"{what} {path} does not exist"], None
    full = git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    if full.returncode != 0:
        return [f"{flag} {sha} is not a commit of {repo}"], None
    sha = full.stdout.decode().strip()
    shown, out = git("show", f"{sha}:{rel}"), []
    if shown.returncode != 0 or shown.stdout != target.read_bytes():
        out.append(f"{what} {rel} is not, byte for byte, the one committed at {sha}")
    if git("merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
        out.append(f"commit {sha} is not an ancestor of HEAD")
    branches = git("branch", "-r", "--contains", sha).stdout.decode().split()
    if remote is None and not branches:
        out.append(f"commit {sha} is on no remote branch")
    elif remote is not None and not any(b.startswith(f"{remote}/") for b in branches):
        out.append(f"commit {sha} is on no branch of {remote}")
    if not out and once:
        entries = [line.split() for line in git("log", "--full-history", "--format=%H %P", f"--remotes={remote}",
                                                 "--", f":(top){rel}").stdout.decode().splitlines()]
        blob = lambda c: git("rev-parse", "--verify", "--quiet", f"{c}:{rel}").stdout.decode().strip() or None

        def genuine(c: str, parents: list[str]) -> bool:
            """`c` really writes `rel` when its blob differs from every parent's (absent is a value
            too); a root commit (no parent) writes it when it has the file at all."""
            return blob(c) is not None if not parents else all(blob(c) != blob(p) for p in parents)

        writes = [c for c, *parents in entries if genuine(c, parents)]
        if writes != [sha]:
            out.append(f"{what} {rel} is written on {remote} by {', '.join(writes)}, not by {flag} {sha} alone: it is "
                       "written once and never edited")
    if out:
        return out, None
    return [], {"sha": sha, "path": rel, "fuller": git("show", "-s", "--format=fuller", sha).stdout.decode(),
                "committed": git("show", "-s", "--format=%cI", sha).stdout.decode().strip()}


def lock_commit_errors(lock: Path, sha: str, repo: Path) -> tuple[list[str], dict | None]:
    """SAP_LOCK.json counts only as committed: committed_file_errors of the lock, on a
    branch of origin (fixed 27/09/2026: a push to another remote, a fork, does not do)."""
    return committed_file_errors(lock, sha, repo, "the lock", "--lock-commit", "origin")


def lock_precedes_scoring_errors(metas: list[tuple[Path, dict]], lock_committed: str) -> list[str]:
    """The SAP lock's own commit date (`lock_committed`, committed_file_errors' %cI of
    --lock-commit) must be strictly before every scored_utc a given SCORES_META names, so the lock cannot have been
    adjusted after the scores it will be run on were already known; a lock committed at or after a scoring is refused.
    A meta with no readable scored_utc is left to scores_meta_errors (it already refuses that); this check only ever
    runs on metas that parsed as a dict. One message per meta that fails it (the meta files, not the manifest lines,
    since a SCORES_META's scored_utc is the one moment the lock is checked against, independent of --repo's
    local clock or state: the documented limit is that git itself still rests on the local checkout's clock and
    history, and the independent time anchor for the lock is its push to origin, read from the
    registry)."""
    lt = utc_time(lock_committed)
    out = []
    for p, m in metas:
        st = utc_time(m.get("scored_utc"))
        if st is not None and lt is not None and not (lt < st):
            out.append(f"{p}: scored_utc {m['scored_utc']} is not after the lock's commit date {lock_committed}")
    return out



def canonical_meta_errors(metas: list[Path], repo: Path, partition: str) -> list[str]:
    """Against a copied META (added 28/09/2026): an edited scores.jsonl copied into a NEW folder, with a new
    SCORES_META.json written there holding the edited file's scores_sha256, committed and pushed once, is anchored by
    scores_meta_commit_errors just as well as any other path, so nothing would stop a scores.jsonl edited after scoring
    from passing every check (in a probe, RQ1 went from 0.2711 to 0.5482, exit 0). So a SCORES_META.json is read only at
    the canonical path out/phase3/scores/<partition>/SCORES_META.json, relative to --repo ("test" for the confirmatory
    run, "aux_test" for --descriptive-only); any other path is refused here, before scores_meta_commit_errors (or any
    content check) ever runs, so a META anchored anywhere else cannot be read at all, whatever it is committed as. The
    canonical path holds one file, so it names exactly one META per partition: exactly one --scores file per run (no
    existing test passed --scores-meta or --scores more than once per run, so this narrows no case the suite already
    exercises). Called from sap_run.generation before the commit check. Since 28/09/2026, the path is compared as a
    fixed string under --repo (repo_rel), not after resolving it, and no step down to it may be a symlink nor the file a
    hard link (link_errors): a symlink at the canonical path, or in place of its folder, resolved to a META written once
    elsewhere and passed."""
    rel = f"out/phase3/scores/{partition}/SCORES_META.json"
    canon = repo / rel
    out = []
    if len(metas) > 1:
        out.append(f"exactly one SCORES_META.json per run: got {len(metas)} for partition {partition} (the canonical "
                   "path holds one META per partition, so one --scores file per run)")
    wrong = sorted({str(p) for p in metas if repo_rel(p, repo) != rel})
    if wrong:
        out.append(f"the scores meta must sit at the canonical path {canon} (relative to --repo), not at "
                   f"{', '.join(wrong)}: any other path can be bypassed by copying the scores and a fresh META into "
                   "it")
    elif metas:
        out += link_errors(repo, rel, "the scores meta")
        if not out:  # `x/../SCORES_META.json` through a symlink x lexically IS the canonical path
            out += [f"the scores meta {p} does not lead to {rel} under --repo but to {os.path.realpath(p)}: the file "
                    "checked and the file given must be one" for p in metas
                    if Path(os.path.realpath(p)) != repo.resolve() / rel]
    return out


def scores_meta_commit_errors(metas: list[tuple[Path, dict]], sha: str, repo: Path, lock_committed: str,
                              lock_commit: str) -> tuple[list[str], dict | None]:
    """Added 27/09/2026: a SCORES_META.json names its scores file's bytes (scores_sha256), so it is the anchor of the
    scores and must itself be anchored, as the accepted_missing file is: every SCORES_META is committed right after the
    scoring and pushed to origin, once. Each meta of `metas` must be committed_file_errors-valid at `sha`
    (--scores-meta-commit) on origin with `once` (byte for byte, the commit an ancestor of HEAD and the only commit of
    origin that GENUINELY writes the meta's path: a META edited, committed and pushed again is refused (since
    27/09/2026); a --no-ff merge that carries it forward unchanged no longer is), and that commit dated strictly after
    the lock's commit date (`lock_committed`) and after each meta's scored_utc (a meta whose scored_utc does not parse
    is left to scores_meta_errors). Otherwise a scores.jsonl edited together with its META's scores_sha256 would pass
    every check. Since 28/09/2026, the commit must also be a DESCENDANT of `lock_commit` (--lock-commit; git merge-base
    --is-ancestor), not merely dated after it, since a commit's date rests on the local clock alone and a META cut from
    a branch older than the lock, then given a late date, would otherwise pass the date checks while never actually
    building on the locked inputs. The documented limit: git dates rest on the local clock and origin's history can be
    rewritten (a force push); the independent anchor is the registry, which records LOCK_COMMIT, META_COMMIT and
    AUX_META_COMMIT with GitHub's push time at each push. Returns the problems (each once) and, when there is none,
    {sha, paths, fuller, committed} for SAP_RESULTS and SAP_REPORT."""
    out, info = [], None
    for p, _ in metas:
        problems, c = committed_file_errors(p, sha, repo, "the scores meta", "--scores-meta-commit", "origin", True)
        out += [x for x in problems if x not in out]
        if c is not None:
            info = info or {"sha": c["sha"], "paths": [], "fuller": c["fuller"], "committed": c["committed"]}
            info["paths"].append(c["path"])
    if out or info is None:
        return out, None
    ct, lt = utc_time(info["committed"]), utc_time(lock_committed)
    if not (ct is not None and lt is not None and lt < ct):
        out.append(f"--scores-meta-commit {info['sha']} (committed {info['committed']}) is not after the lock's commit "
                   f"date {lock_committed}")
    if git_run(repo, "merge-base", "--is-ancestor", lock_commit, info["sha"]).returncode != 0:
        out.append(f"--lock-commit {lock_commit} is not an ancestor of --scores-meta-commit {info['sha']}: the lock "
                   "must precede the scores meta in this repository's history, not merely by date")
    for p, m in metas:
        st = utc_time(m.get("scored_utc"))
        if st is not None and not (ct is not None and st < ct):
            out.append(f"{p}: --scores-meta-commit {info['sha']} (committed {info['committed']}) is not after its "
                       f"scored_utc {m['scored_utc']}")
    return (out, None) if out else ([], info)

# ----- generation manifest, SCORES_META, accepted missing pairs (fixed 26/09/2026; design specification §5 at 7af1673)

MANIFEST_KEY = ("model", "condition", "seed", "variant", "record_id", "fmt")  # generation manifest.<variant>.jsonl
MANIFEST_SHAS = ("prompts_sha256", "prompt_sha256", "adapter_tensor_sha256", "decision_sha256", "gen_code_sha256",
                 "leakcheck_receipt_sha256")  # prompts, prompt, adapter, canary decision, gen code, leak-check receipt
MANIFEST_FIELDS = MANIFEST_KEY + MANIFEST_SHAS + ("status", "line_sha256", "attempts")
# the evidence a missing pair needs (fixed 26/09/2026)
EVIDENCE_RULE = {"statuses": ("CRASHED", "INCOMPLETE"), "min_attempts": 2}
MANIFEST_RULES = {  # one message per rule broken by lines of a manifest (read_manifests)
    "fields": "lines lack a field of the manifest contract (the key, the six sha256, status, line_sha256, attempts; "
              "receipt_status and receipt_sha256 when missing, reason when excluded)",
    "status": "lines have a status that is not ok, excluded or missing",
    "line": "lines have a line_sha256 that does not fit their status (a non-empty one exactly when ok)",
    "attempts": "lines have attempts that do not fit their status (null when ok or excluded, a list of objects when "
                "missing)",
    "receipt": "missing lines have a receipt_status or receipt_sha256 that is not their last attempt's (absent and "
               "null without one)",
    "eligible": "lines have an evidence_eligible that their attempts do not give (EVIDENCE_RULE)"}


def pair(x: dict) -> tuple:
    """The pair of a manifest line, an accepted_missing entry or a score row: (record_id, model, condition, variant,
    seed); a condition has one format, so fmt adds nothing to it."""
    return x.get("record_id"), x.get("model"), x.get("condition"), x.get("variant") or "main", x.get("seed")


def has_evidence(line: dict) -> bool:
    """The evidence rule (fixed 26/09/2026; design specification §5 at 7af1673), EVIDENCE_RULE: a `missing` line whose
    job has at least min_attempts attempts (the generation run's chronological `attempts`), each with a sha256 and at
    least min_attempts of them with a status (a receipt that parsed, as the generation run's evidence_eligible), the
    last one with a status of `statuses` (CRASHED, INCOMPLETE). REFUSED_* (operator-made), RECHECK_MISMATCH and an
    absent job (no attempt) never count."""
    att = line.get("attempts")
    return (line.get("status") == "missing" and isinstance(att, list) and len(att) >= EVIDENCE_RULE["min_attempts"]
            # as the generation run's evidence_eligible, attempts that read as a
            # receipt count
            and sum(isinstance(a, dict) and isinstance(a.get("status"), str) for a in att) >= EVIDENCE_RULE["min_attempts"]
            and all(isinstance(a, dict) and isinstance(a.get("sha256"), str) and a["sha256"] for a in att)
            and att[-1].get("status") in EVIDENCE_RULE["statuses"])


def manifest_line_rules(x: dict) -> dict[str, bool]:
    """Which MANIFEST_RULES a manifest line breaks (the generation run's schema): the fields; the status;
    line_sha256 non-empty exactly when ok; attempts null when ok or excluded, a list of objects when missing;
    receipt_status and receipt_sha256 those of the last attempt ("absent" and null without one); evidence_eligible, a
    convenience that is never trusted, equal to has_evidence when present."""
    st = x.get("status")
    need = MANIFEST_FIELDS + {"missing": ("receipt_status", "receipt_sha256"), "excluded": ("reason",)}.get(st, ())
    if any(k not in x for k in need):
        return {"fields": True}
    att = x["attempts"]
    listed = isinstance(att, list) and all(isinstance(a, dict) for a in att)
    last = (att[-1].get("status"), att[-1].get("sha256")) if listed and att else ("absent", None)
    return {"status": st not in ("ok", "excluded", "missing"),
            "line": (st == "ok") != (isinstance(x["line_sha256"], str) and bool(x["line_sha256"])),
            "attempts": att is not None if st in ("ok", "excluded") else st == "missing" and not listed,
            "receipt": st == "missing" and listed and (x["receipt_status"], x["receipt_sha256"]) != last,
            "eligible": "evidence_eligible" in x and x["evidence_eligible"] != has_evidence(x)}


def read_manifests(files: list[tuple[Path, Path]]) -> tuple[dict[tuple, dict], dict[str, str], list[str]]:
    """The generation manifests of the scored runs, each with its coverage file (fixed 26/09/2026): (pair ->
    line over every manifest, manifest sha256 -> coverage sha256, problems). Problems: lines that break
    manifest_line_rules (one message per rule); a pair listed twice over the manifests; a coverage file that is not a
    JSON object, whose manifest_sha256 is not its manifest's sha256, whose status is PASS while the manifest has
    missing lines, or whose status and status_except_missing are both not PASS (RECHECK_MISMATCH fails both: the
    generation is not complete). A manifest that is not JSONL raises SapInputError (jsonl_rows)."""
    lines: dict = {}
    shas: dict = {}
    out = []
    for m, c in files:
        rows = jsonl_rows(m)
        broken: dict = {k: [] for k in MANIFEST_RULES}
        twice = []
        for x in rows:
            for name, bad in manifest_line_rules(x).items():
                if bad:
                    broken[name].append(pair(x))
            if pair(x) in lines:
                twice.append(pair(x))
            lines.setdefault(pair(x), x)
        out += [f"{m}: {len(xs)} {MANIFEST_RULES[name]}, e.g. {min(xs, key=str)}" for name, xs in broken.items() if xs]
        if twice:
            out.append(f"{m}: {len(twice)} lines list a pair already listed (model, condition, seed, variant, "
                       f"record_id), e.g. {min(twice, key=str)}")
        msha, n_missing = hashlib.sha256(m.read_bytes()).hexdigest(), sum(x.get("status") == "missing" for x in rows)
        try:
            cov = json.loads(c.read_text(encoding="utf-8"))
        except ValueError:
            cov = None
        if not isinstance(cov, dict):
            out.append(f"{c}: not a JSON object")
            continue
        if cov.get("manifest_sha256") != msha:
            out.append(f"{c}: manifest_sha256 is not the sha256 of {m}")
        elif cov.get("status") == "PASS":
            if n_missing:
                out.append(f"{c}: status PASS, but {m} has {n_missing} missing lines")
        elif cov.get("status_except_missing") != "PASS":
            out.append(f"{c}: status {cov.get('status')} and status_except_missing {cov.get('status_except_missing')}: "
                       "the generation is not complete (a RECHECK_MISMATCH fails both)")
        shas[msha] = hashlib.sha256(c.read_bytes()).hexdigest()
    return lines, shas, out


def manifest_design_errors(lines: dict[tuple, dict], cfg: dict, ids: dict[str, set[str]], dropped: dict[str, set[str]],
                           test: set[str], known: set[str], deferred: bool = False) -> list[str]:
    """The manifests against the design (fixed 26/09/2026). A line of a record of no locked partition (`known`:
    the test and aux_test records) is an error. Over the test records (`test`), the ok and missing lines are exactly
    the design: a line of a run outside design() (the deferred runs are allowed with `deferred`, --with-deferred), of a
    design run under a seed the design does not have, or of a record outside `ids[model]` (design_records) that compose
    did not drop for that model (`dropped[model]`, compose_dropped), is outside the design; a design cell (every record
    of ids[model] for every design run and seed) with no line, or with an excluded one (the locked compose file keeps
    the record), is absent. The lines of a compose-dropped (model, record) are free (compose may drop one format only);
    the lines of aux_test records belong to the aux run."""
    runs = design(cfg)
    names, cells = {rk for rk, _ in runs}, set(runs)
    allowed = names | (deferred_runs(cfg) if deferred else set())
    unknown, outside = [], []
    for k in lines:
        rid, model, cond, variant, seed = k
        if rid not in known:
            unknown.append(k)
        elif rid in test:
            rk = (model, cond, variant)
            if rk not in allowed or rk in names and ((rk, seed) not in cells or rid not in ids.get(model, set())
                                                     and rid not in dropped.get(model, set())):
                outside.append(k)
    absent = [(rid, *rk, s) for rk, s in runs for rid in sorted(ids.get(rk[0], ()))
              if lines.get((rid, *rk, s), {}).get("status") not in ("ok", "missing")]
    out = [f"{len(unknown)} manifest lines of records of no locked partition, e.g. {min(unknown, key=str)}"] \
        if unknown else []
    if outside:
        out.append(f"{len(outside)} manifest lines outside the design (model, condition, variant, seed, record), e.g. "
                   f"{min(outside, key=str)}")
    if absent:
        out.append(f"{len(absent)} design cells have no ok or missing manifest line (none, or excluded while the "
                   f"locked compose file keeps the record), e.g. {min(absent)}")
    return out


ACCEPTED_FIELDS = MANIFEST_KEY + ("reason", "receipt_sha256")  # an entry of accepted_missing.jsonl (as score_run's)


def accepted_missing_errors(entries: list[dict] | None, lines: dict[tuple, dict]) -> list[str]:
    """Design specification §5 at 7af1673 (fixed 26/09/2026): the entries of the accepted_missing.jsonl (None: no file
    given) against the manifests' lines. Each entry has ACCEPTED_FIELDS, lists its pair once, is a `missing` line (same
    key and fmt) with that line's receipt_sha256 (its last attempt's), and the line has evidence (has_evidence,
    EVIDENCE_RULE); every missing line is listed. One message per rule broken, with the count and one example."""
    missing = {k for k, x in lines.items() if x.get("status") == "missing"}
    if entries is None:
        return [f"{len(missing)} missing manifest lines and no --accepted-missing, e.g. {min(missing, key=str)}"] \
            if missing else []
    broken: dict = {"fields": [], "twice": [], "not_missing": [], "receipt": [], "evidence": []}
    seen: set = set()
    for e in entries:
        k = pair(e)
        if any(f not in e for f in ACCEPTED_FIELDS):
            broken["fields"].append(k)
            continue
        if k in seen:
            broken["twice"].append(k)
            continue
        seen.add(k)
        line = lines.get(k, {})
        if line.get("status") != "missing" or line.get("fmt") != e["fmt"]:
            broken["not_missing"].append(k)
        elif e["receipt_sha256"] != line.get("receipt_sha256"):
            broken["receipt"].append(k)
        elif not has_evidence(line):
            broken["evidence"].append(f"{k}: {line.get('receipt_status')}")
    what = {"fields": "lack a field (model, condition, seed, variant, record_id, fmt, reason, receipt_sha256)",
            "twice": "list a pair twice", "not_missing": "are not a missing line of the manifests (same key and fmt)",
            "receipt": "have a receipt_sha256 that is not their manifest line's",
            "evidence": "have no evidence (last receipt CRASHED or INCOMPLETE over >= 2 attempts, each with a sha256; "
                        "REFUSED_* and absent never count)"}
    out = [f"{len(xs)} accepted_missing entries {what[name]}, e.g. {min(xs, key=str)}"
           for name, xs in broken.items() if xs]
    unlisted = sorted(missing - seen, key=str)
    if unlisted:
        out.append(f"{len(unlisted)} missing manifest lines are not in accepted_missing, e.g. {unlisted[0]}")
    return out


def utc_time(x) -> dt.datetime | None:
    """An ISO 8601 time with its offset ("Z" or +hh:mm), else None."""
    try:
        t = dt.datetime.fromisoformat(x)
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo is not None else None


def scores_meta_errors(metas: list[tuple[Path, dict]], shas: dict[str, str], partition: str, accepted: str | None,
                       commit: str | None, committed: str | None, scores_shas: list[str]) -> list[str]:
    """Fixed 26/09/2026: each SCORES_META.json of the scored runs (3C's score_run --outputs) is of mode outputs and of
    the partition analysed; its manifest_sha256 is a given --manifest's and its coverage_sha256 that manifest's
    --coverage's (`shas`); its scored_utc is a time; when it names an accepted_missing file (accepted_missing_sha256),
    that is --accepted-missing (`accepted`, its sha256), its accepted_missing_commit is (a prefix of) the resolved
    --accepted-missing-commit (`commit`), and the scoring comes after that commit (`committed`, its %cI date: missing
    pairs are accepted before the scoring, as §5 of the design specification says). Every --manifest, and
    --accepted-missing, is named by some SCORES_META. With `scores_shas` (the actual sha256 of each --scores file,
    computed by the caller over the file's current bytes, index-aligned with `metas`), a SCORES_META's own scores_sha256
    must equal it, missing included (None is not any file's sha256 either); this is the only thing that ties a score
    row's content to the moment it was scored, since flipping a value inside an already-scored file changes no manifest
    or accepted_missing byte at all."""
    out, named, acc_named = [], set(), False
    for (p, m), sc_expected in zip(metas, scores_shas):
        if m.get("mode") != "outputs":
            out.append(f"{p}: mode {m.get('mode')!r} is not outputs")
        if m.get("partition") != partition:
            out.append(f"{p}: partition {m.get('partition')!r} is not {partition}")
        sc = m.get("scores_sha256")
        if sc != sc_expected:
            out.append(f"{p}: scores_sha256 {sc} is not the sha256 of its --scores file ({sc_expected})")
        ms = m.get("manifest_sha256")
        if ms not in shas:
            out.append(f"{p}: manifest_sha256 {ms} is not the sha256 of a given --manifest")
        else:
            named.add(ms)
            if m.get("coverage_sha256") != shas[ms]:
                out.append(f"{p}: coverage_sha256 is not the sha256 of the --coverage of its manifest")
        t = utc_time(m.get("scored_utc"))
        if t is None:
            out.append(f"{p}: scored_utc {m.get('scored_utc')!r} is not a UTC time")
        a = m.get("accepted_missing_sha256")
        if a is None:
            continue
        if accepted is None:
            out.append(f"{p}: accepted_missing_sha256 {a} names an accepted_missing file, but no --accepted-missing "
                       "is given")
        elif a != accepted:
            out.append(f"{p}: accepted_missing_sha256 {a} is not the sha256 of --accepted-missing ({accepted})")
        else:
            acc_named = True
            c = m.get("accepted_missing_commit")
            if not (isinstance(c, str) and len(c) >= 7 and (commit or "").startswith(c)):
                out.append(f"{p}: accepted_missing_commit {c} is not --accepted-missing-commit {commit}")
            if t is not None and not (utc_time(committed) is not None and t > utc_time(committed)):
                out.append(f"{p}: scored_utc {m['scored_utc']} is not after the commit date {committed} of the "
                           "accepted_missing file")
    out += [f"--manifest {s} is named by no SCORES_META" for s in shas if s not in named]
    if accepted is not None and not acc_named:
        out.append("--accepted-missing is named by no SCORES_META")
    return out


ROBUST_RULE = ("Quy tắc diễn giải (khai trước, P22): một kết luận xác nhận chỉ \"vững\" khi CA và CA_prec cho cùng "
               "nhãn P9, xét theo từng phép so và theo từng RQ; supported, contrary và conditional_only_± mang sẵn "
               "chiều, nên cùng nhãn là cùng chiều; với no_difference, dấu của một ước lượng không có ý nghĩa thống kê "
               "là nhiễu, không làm mất nhãn \"vững\", và khi hai ước lượng trái dấu thì ghi \"(ước lượng trái dấu)\" "
               "cạnh nhãn; nếu khác nhãn, kết luận là \"chưa vững\", không chọn một trong hai thước đo.")


def robust_label(r: dict) -> str:
    """P22 (fixed 27/09/2026): the label of a robust_verdicts entry, "vững" or "chưa vững"; a robust one whose CA and
    CA_prec estimates are of opposite signs (two no_difference verdicts: the same label hides the signs) adds
    "(ước lượng trái dấu)"."""
    return "chưa vững" if not r["robust"] else "vững" + (" (ước lượng trái dấu)" if r.get("opposite_estimates") else "")


def robust_verdicts(conf: dict, conf_prec: dict) -> dict:
    """Per comparison and per RQ, the CA and CA_prec verdicts and whether the conclusion is robust
    ("vững": the same verdict, which carries its direction), under ROBUST_RULE. P22 (fixed 27/09/2026): each comparison
    also keeps both estimates (ca_estimate, ca_prec_estimate; None when absent) and opposite_estimates, true when
    both are there and of opposite signs (a zero has none); the rule is unchanged: two no_difference verdicts are
    robust whatever their signs, and robust_label notes the opposite signs. Since 28/09/2026, every comparison
    and every RQ entry also carries "label", robust_label(...) of the entry itself, as text next to the boolean, so
    SAP_RESULTS.json states the same "vững" / "chưa vững" / "vững (ước lượng trái dấu)" the report prints, not just
    the boolean a reader would have to re-derive it from."""
    prec = lambda fam: conf_prec.get(fam, {})
    labeled = lambda e: {**e, "label": robust_label(e)}

    def entry(v: dict, p: dict) -> dict:
        a, b = v.get("ca", {}).get("estimate"), p.get("ca_prec", {}).get("estimate")
        return labeled({"ca": v["verdict"], "ca_prec": p.get("verdict"), "robust": v["verdict"] == p.get("verdict"),
                        "ca_estimate": a, "ca_prec_estimate": b,
                        "opposite_estimates": a is not None and b is not None and a * b < 0})
    return {"rule": ROBUST_RULE,
            "comparisons": {fam: {name: entry(v, prec(fam).get("comparisons", {}).get(name, {}))
                                  for name, v in f["comparisons"].items()} for fam, f in conf.items()},
            "rq": {fam: labeled({"ca": f["rq_verdict"], "ca_prec": prec(fam).get("rq_verdict"),
                                 "robust": f["rq_verdict"] == prec(fam).get("rq_verdict")})
                   for fam, f in conf.items()}}


def denominator_rows(records: list[dict], controls: list[dict], exclusions: list[dict], compose_excluded: list[dict],
                     model: str) -> list[dict]:
    """Per repo and partition of the locked records, the records and scenarios in all, those usable for CA
    (usable_records: the controls' gold records minus the mechanical exclusions minus those compose dropped for
    `model`, the anchor), and the records mechanically excluded and compose-dropped."""
    keep = usable_records(controls, exclusions, compose_excluded, model)
    excl, dropped = {x["record_id"] for x in exclusions}, compose_dropped(compose_excluded, model)
    scen = lambda rs: len({f"{r['repo']}|{r['feature_path']}|{r['scenario_line']}" for r in rs})
    groups: dict = defaultdict(list)
    for r in records:
        groups[(r["repo"], r["partition"])].append(r)
    out = []
    for (repo, part), rs in sorted(groups.items()):
        ca = [r for r in rs if r["id"] in keep]
        out.append({"repo": repo, "partition": part, "records": len(rs), "scenarios": scen(rs), "ca_records": len(ca),
                    "ca_scenarios": scen(ca), "excluded": sum(r["id"] in excl for r in rs),
                    "dropped": sum(r["id"] in dropped for r in rs)})
    return out


def type_i_rates(calibration: dict) -> list[dict]:
    """The measured rejection rates under H0 (delta = 0) of test (1) (sign
    flip), test (2) (two-level bootstrap) and the "supported" rule, with their Monte Carlo SE, per comparison type and
    tau, at each alpha of the locked CALIBRATION.json (sap_calibrate)."""
    out = []
    for name, v in calibration.get("results", {}).items():
        mode, tau, delta = (x.split("=")[-1] for x in name.split("|"))
        if float(delta) == 0:
            out.append({"type": mode, "tau": tau, "n": v["n"],
                        "alphas": {a: {t: x[t] for t in ("sign_flip", "two_level", "supported")}
                                   for a, x in v.items() if a.startswith("alpha=")}})
    return out


CODE_FILES = ("v2tools/stats.py", "v2tools/sap.py")  # and every scripts/sap_*.py present (code_files)


def code_files(root: Path) -> tuple[str, ...]:
    """The 3D analysis code the lock covers (fixed 05/10/2026): v2tools/stats.py, v2tools/sap.py and
    EVERY scripts/sap_*.py present under `root` (sap_verify.py and sap_swrap.py included), by path from the repo root;
    a sap_*.py script added or removed after the lock makes the code differ."""
    return CODE_FILES + tuple(sorted(f"scripts/{p.name}" for p in (root / "scripts").glob("sap_*.py") if p.is_file()))


def code_digest(root: Path) -> dict[str, str | None]:
    """sha256 of the 3D analysis code only (fixed 24/09/2026; code_files), by path from the repo root: a change to any
    other code file (3A, 3C) leaves the lock valid; a fixed file that is missing gives None (sap_lock refuses)."""
    return {p: sha256_of(root / p) for p in code_files(root)}


def code_tree_errors(root: Path) -> list[str]:
    """Why the 3D code under `root` is not a clean committed tree, so that the lock can be rebuilt from git (added
    05/10/2026):
    `git status --porcelain --untracked-files=all --ignored=matching` of v2tools/stats.py, v2tools/sap.py and
    scripts/sap_*.py must be empty (an untracked sap_*.py, one that .gitignore, .git/info/exclude or a global excludes
    file ignores, or a change not committed is named), git must answer (`root` in a git checkout), and no file
    code_files names may be a symlink. Other files, 3A's and 3C's included, do not count.
    sap_lock refuses (exit 2) on any of these, before it hashes the code."""
    st = git_run(root, "status", "--porcelain", "--untracked-files=all", "--ignored=matching", "--",
                 *CODE_FILES, "scripts/sap_*.py")  # "!!" lines too: code_files hashes an ignored file (05/10/2026)
    if st.returncode != 0:
        return [f"git status failed in {root} (the lock is taken in a git checkout): "
                f"{st.stderr.decode('utf-8', errors='replace').strip()[:200]}"]
    out = [f"not a clean committed file: {line}" for line in st.stdout.decode("utf-8", errors="replace").splitlines()
           if line.strip()]
    return out + [f"{p} is a symlink, not a regular file" for p in code_files(root) if (root / p).is_symlink()]


def differing(locked: dict, now: dict) -> list[str]:
    """Names whose sha256 is not the locked one (present on one side only included)."""
    return sorted(k for k in set(locked) | set(now) if locked.get(k) != now.get(k))


def git_head(root: Path) -> str | None:
    """HEAD of the repo holding the code, or None outside git: recorded for information, never compared."""
    head = git_run(root, "rev-parse", "HEAD")
    return head.stdout.decode().strip() if head.returncode == 0 else None
