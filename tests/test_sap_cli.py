import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from tests.test_sap import rows
from v2tools.sap import code_files

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {"config", "mde", "calibration", "dev_selection", "dev_scores[0]", "dev_scores[1]", "dev_exclusions", "controls",
          "exclusions", "compose_excluded", "records", "subsets", "self_contained", "phase2_sums",
          "er_controls[0]", "er_manifest", "future_methods", "aux_controls", "aux_exclusions", "aux_future_methods",
          "aux_records", "pre_t_flags"}
OTHER_SEEDS = ("base", "s42", "s11", "s23")  # the seeds of CodeLlama and Mistral
REHEARSAL = "**KHÔNG PHẢI KẾT QUẢ — chạy thử trên đối chứng**"  # sap_run --rehearsal (added 26/09/2026)
SHAS = ("prompts_sha256", "prompt_sha256", "adapter_tensor_sha256", "decision_sha256", "gen_code_sha256",
        "leakcheck_receipt_sha256")  # the generation run's manifest sha fields, which score rows carry
FMT = {"zero_shot": "F0", "ft_only": "F0", "rag_only": "F1", "full_neutral": "F1", "full_aware": "F2"}
KEY = ("model", "condition", "seed", "variant", "record_id", "fmt")
SCORED = "2099-01-01T00:00:00Z"  # the fixture's scored_utc: after every commit a test makes
DEV_ID = "Sylius/Sylius|features/api/dev.feature|12|default"  # a dev record of the same ER controls run


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l]


def write(path: Path, xs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(x) + "\n" for x in xs))
    return path


ER_DONE = {"missing": [], "extra": [], "duplicates": [], "suspect": [],
           "scope": {"conditions": ["gold", "stub"], "limit": None}, "scope_missing": [], "complete": True,
           "running": False}


def manifest_line(rid: str, cond: str) -> str:
    """The fixture's line of the ER controls run's manifest.jsonl for (record, condition)."""
    return json.dumps({"record_id": rid, "condition": cond, "dir": f"{rid.split('|')[1]}/{cond}", "files": {}})


def write_er(path: Path, xs: list[dict]) -> Path:
    """An ER control results file whose rows carry the overlay_sha256 of their line in the manifest.jsonl next to it
    (a row that has one keeps it), and, next to it, the er_complete.json of a finished, complete run of that manifest
    (the ER harness's er_results contract, 0c42865)."""
    done = {"expected": len(xs), "rows": len(xs), **ER_DONE, "manifest_sha256": sha(path.parent / "manifest.jsonl")}
    (path.parent / "er_complete.json").write_text(json.dumps(done))
    return write(path, [{"overlay_sha256": hashlib.sha256(manifest_line(x["record_id"], x["condition"]).encode())
                        .hexdigest(), **x} for x in xs])


def other_bytes(data: bytes) -> bytes:
    """The same JSONL rows with other bytes (compact separators), or, for an empty file, one exclusion of a record no
    input has: a new sha256 with no change of meaning, still JSONL (a stray blank line is refused)."""
    xs = [json.loads(l) for l in data.decode("utf-8").split("\n") if l]
    if not xs:
        return (json.dumps({"record_id": "R|none.feature|0|a", "reason": "gold_fails_bv"}) + "\n").encode("utf-8")
    return "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in xs).encode("utf-8")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup(tmp: Path) -> tuple:
    """A copy of the code (so the lock's code rule is exercised whatever the state of the real tree) and every input of
    the SAP; residual_clean is a real subset, the same in the rows and in sensitivity_subsets.json. PHASE2_SHA256SUMS
    lists the records file as out/phase2/records/test.jsonl and sensitivity_subsets.json; the ER control run covers the
    one exec_core record (ids[5]), whose score rows carry er, and its manifest lists (neg, gold, stub) of every test
    record and of DEV_ID. ids[:2] are mechanical exclusions with CA undefined (not excl_bv), as 3C writes
    them. The records of every third scenario are not self_contained (subset (iv)), ids[5]'s scenario is. CodeLlama
    and Mistral run too (the design covers every model; Mistral has no row of ids[3], which compose dropped for it);
    the records carry language, suite, partition and create; the aux_test partition has its own controls, records and
    DeepSeek scores (repo X), all locked."""
    repo = tmp / "repo"
    for d in ("v2tools", "scripts"):
        shutil.copytree(ROOT / d, repo / d, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = {**json.loads((ROOT / "config" / "sap.json").read_text()), "reps_bootstrap": 300, "reps_permutation": 299}
    (repo / "config").mkdir()
    (repo / "config" / "sap.json").write_text(json.dumps(cfg))
    test_rows = rows({"full_aware": 0.3}, seed=7)
    ids = sorted({r["record_id"] for r in test_rows})
    clean = ids[::3]
    apart = set(sorted({i.rsplit("|", 1)[0] for i in ids})[1::3])
    sc = {i: i.rsplit("|", 1)[0] not in apart for i in ids}
    assert sc[ids[5]] and not all(sc.values())
    for r in test_rows:
        r["sub_residual_clean"] = r["record_id"] in clean
        r["self_contained"] = sc[r["record_id"]]
        r["fence_truncated"] = False
        if r["record_id"] in ids[10:14] and (r["condition"], r["seed"]) == ("full_aware", "s42"):  # the fifth status
            r["extract_status"], r["fence_truncated"] = "multi_file_fence", True
        r["ca_defined"] = r["record_id"] not in ids[:2]
        r["er"] = r["trace_action"] = r["ca"] if r["record_id"] == ids[5] else None
        r["er_status"] = None if r["er"] is None else ("PASS" if r["er"] == 1 else "ERROR")  # the ER run's status
        r["exec_core"] = r["record_id"] == ids[5]
        r["ca_ref_v3all"], r["ca_ref_top4"] = 0, int(r["record_id"] == ids[6])  # D11 (d): top4 reaches ids[6] only
        r["php_tag_stripped"] = r["record_id"] in ids[14:17] and r["condition"] == "zero_shot"  # §1.4 (26/09/2026)
        r["php_docblock_steps"] = int(r["record_id"] in ids[5:7] and r["condition"] == "rag_only")  # §2.3
        r["php_attribute_steps"] = 2
        # §2.2 of the design specification: ids[7] shows two step lines the reader does not read on rag_only
        r["unread_step_annotations"] = 2 * int(r["record_id"] == ids[7] and r["condition"] == "rag_only")
        r["cuke_escaped_patterns"] = r["cuke_custom_types"] = 0
        # P26 (fixed 06/10/2026): ids[30:33] have a long Full-aware prompt, ids[33] no Full-aware length; ids[3:5]
        # rewrite a REUSE step, ids[3:6] match the gold verbatim; Full-aware follows the annotation style more
        r["len_bucket"] = None if (r["record_id"], r["condition"]) == (ids[33], "full_aware") else (
            "gt8192" if r["record_id"] in ids[30:33] else "le8192")
        r["duplicate_reuse"], r["exact"] = r["record_id"] in ids[3:5], r["record_id"] in ids[3:6]
        r["conv_style"] = 0.5 if r["condition"] == "full_neutral" else 1.0
    test_rows += [{**r, "model": m, "er": None, "er_status": None} for r in list(test_rows) if r["seed"] in OTHER_SEEDS
                  for m in ("codellama", "mistral") if not (m == "mistral" and r["record_id"] == ids[3])]
    aux_rows = [{**r, "record_id": "X" + r["record_id"][1:], "scenario_key": "X" + r["scenario_key"][1:], "repo": "X",
                 "partition": "aux_test", "exec_core": False, "self_contained": True, "er": None, "ca_defined": True,
                 "sub_post_cutoff_all": False, "sub_residual_clean": False, "ca_ref_top4": None,
                 "php_tag_stripped": None, "php_docblock_steps": None, "php_attribute_steps": None,
                 "unread_step_annotations": None, "cuke_escaped_patterns": None, "cuke_custom_types": None}
                for r in test_rows if r["model"] == "deepseek" and r["feature_path"] in ("f0.feature", "f1.feature")]
    aux_ids = sorted({r["record_id"] for r in aux_rows})
    for r in test_rows + aux_rows:  # the sha256 of the pair's manifest line (score_run copies them)
        r.update(shas_of(r))
        r["line_sha256"] = mline(r)["line_sha256"]  # the ok line's line_sha256, copied too
    test_rows = [r for r in test_rows if r["record_id"] != ids[2]]  # compose dropped it: no prompt, no output
    ac = tmp / "ac"
    ac.mkdir()
    write(ac / "controls.jsonl", [{"record_id": i, "control": c, "ca_defined": True, "ca": ca, "n_calls": nc,
                                   "call_precision": cp} for i in aux_ids
                                  for c, (ca, nc, cp) in (("gold", (1, 3, 1.0)), ("stub-v3", (0, 0, None)))])
    write(ac / "exclusions.jsonl", [])
    write(ac / "future_methods.jsonl", [{"record_id": i, "path": f"src/{i.split('|')[1]}.php", "future": []}
                                        for i in aux_ids])
    tsv = tmp / "FLAGGED_PRE_T.tsv"  # the frozen pre-T list: one test scenario, a dev record, one aux scenario
    tsv.write_text(f"test\t{ids[20]}\ntest\t{ids[21]}\ndev\tR|dev.feature|1|a\naux_test\t{aux_ids[0]}\n"
                   f"aux_test\t{aux_ids[1]}\n")
    cfg["pre_t_flags"] = {"path": str(tsv), "sha256": sha(tsv)}  # the fixture's list, named by the config
    (repo / "config" / "sap.json").write_text(json.dumps(cfg))
    rec = lambda i, part, ec: {"id": i, "repo": i.split("|")[0], "feature_path": i.split("|")[1],
                               "scenario_line": int(i.split("|")[2]), "suite": i.split("|")[3], "language": "php",
                               "partition": part, "exec_core": ec, "create": [{"preexisting": False}]}
    tc = tmp / "tc"
    tc.mkdir()
    ctl = {"gold": (1, 3, 1.0), "stub-v3": (0, 0, None), "stub-v3names": (1, 12, 0.1),
           "stub-v3top4": (1, 4, 0.25)}  # ca, n_calls, call_precision
    write(tc / "controls.jsonl", [{"record_id": i, "control": c, "ca_defined": i not in ids[:2], "ca": ca,
                                   "n_calls": nc, "call_precision": cp}
                                  for i in ids for c, (ca, nc, cp) in ctl.items()])
    write(tc / "exclusions.jsonl", [{"record_id": i, "reason": "ca_undefined", "excl_bv": False, "excl_ca": True}
                                    for i in ids[:2]])
    write(tc / "future_methods.jsonl", [{"record_id": i, "path": f"src/{i.split('|')[1]}.php",
                                         "future": ["Ctx:probeHelper"] if i == ids[5] else []} for i in ids])
    (tmp / "manifest.jsonl").write_text("".join(manifest_line(i, c) + "\n" for i in [*ids, DEV_ID]
                                                for c in ("neg", "gold", "stub")))
    dc = tmp / "dc"
    dc.mkdir()
    write(dc / "exclusions.jsonl", [])
    dev = [r for r in rows({"full_aware": 0.1}, seed=8, variant="dev_pilot_v0") if r["seed"] in ("s42", "base")]
    k4 = [r for r in rows({"full_aware": -0.3}, seed=9, variant="k4") if r["seed"] in ("s42", "base")]
    (tmp / "sensitivity_subsets.json").write_text(json.dumps({"post_cutoff_all": ids, "time_coherent_30d": [], "residual_clean": clean}))
    records = write(tmp / "records.jsonl", [rec(i, "test", i == ids[5]) for i in ids])
    (tmp / "PHASE2_SHA256SUMS").write_text("0" * 64 + "  out/phase2/records/dev.jsonl\n"
                                           + sha(records) + "  out/phase2/records/test.jsonl\n"
                                           + sha(tmp / "sensitivity_subsets.json")
                                           + "  out/phase2/sensitivity_subsets.json\n")
    inputs = {"scores": write(tmp / "scores.jsonl", test_rows), "dev": write(tmp / "dev_pilot_v0.jsonl", dev),
              "dev_k4": write(tmp / "dev_k4.jsonl", k4), "records": records,
              "compose_excluded": write(tmp / "excluded.jsonl", [{"record_id": ids[2], "model": None, "fmt": None, "reason": "too_long_union"},
                                                                  {"record_id": ids[3], "model": "mistral", "fmt": "F2", "reason": "control_token"}]),
              "subsets": tmp / "sensitivity_subsets.json", "phase2_sums": tmp / "PHASE2_SHA256SUMS",
              "er_manifest": tmp / "manifest.jsonl",
              "self_contained": write(tmp / "self_contained.jsonl",
                                      [{"record_id": i, "self_contained": sc[i]} for i in ids]
                                      + [{"record_id": i, "self_contained": True} for i in aux_ids]),
              "aux_controls": ac, "aux_records": write(tmp / "aux_records.jsonl", [rec(i, "aux_test", False)
                                                                                   for i in aux_ids]),
              "aux_scores": write(tmp / "aux_scores.jsonl", aux_rows), "pre_t_flags": tsv,
              "er": write_er(tmp / "er_controls.jsonl",
                             [{"record_id": ids[5], "condition": "gold", "er_status": "PASS", "trace_action": 1},
                              {"record_id": ids[5], "condition": "stub", "er_status": "FAIL", "trace_action": 0}]),
              "outputs": write(tmp / "outputs.jsonl", [{"record_id": r["record_id"], "model": "deepseek", "seed": r["seed"],
                                                        "condition": r["condition"], "variant": "main", "text": "// FILE: x"} for r in test_rows])}
    write_generation(tmp, inputs)
    git(repo, "init", "-q")  # sap_lock locks the 3D code of a clean git checkout only
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the code of the fixture")
    return repo, tc, dc, ids, inputs


def shas_of(r: dict) -> dict:
    h = lambda k: hashlib.sha256(f"{k}|{r['model']}|{r['condition']}|{r['seed']}".encode()).hexdigest()
    return {k: {"v2gpu/generate.py": h(k), "v2gpu/runs.py": h(k + "/runs")} if k == "gen_code_sha256" else h(k)
            for k in SHAS}  # gen_code_sha256 is the generation run's {path: sha256}


def mline(r: dict, status: str = "ok", last: str = "CRASHED") -> dict:
    """The fixture's manifest line of the pair of `r` (the run's manifest contract): ok with a line_sha256, excluded
    with a reason, or missing after two attempts, each with a sha256, the last one `last` (CRASHED: evidence)."""
    x = {k: r[k] for k in ("model", "condition", "seed", "record_id")}
    x.update(variant=r.get("variant") or "main", fmt=FMT[r["condition"]], **shas_of(r), status=status,
             line_sha256=hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest() if status == "ok" else None,
             attempts=None, receipt_status="COMPLETE", receipt_sha256="r" * 64)
    if status == "excluded":
        x["reason"] = "compose"
    if status == "missing":
        x.update(attempts=[{"file": "receipt-generate.attempt-1.json", "status": "CRASHED", "sha256": "1" * 64},
                           {"file": "receipt-generate.json", "status": last, "sha256": "2" * 64}],
                 receipt_status=last, receipt_sha256="2" * 64)
    return x


def write_generation(tmp: Path, x: dict, missing=(), accepted: Path | None = None, commit: str | None = None,
                     scored: str = SCORED, extra=(), drop=(), last: str = "CRASHED",
                     cfg_repo: Path | None = None) -> None:
    """manifest.main.jsonl, coverage.main.json, SCORES_META.json (test) and SCORES_META.aux.json (aux_test) of the
    fixture's score files: an ok line per score row; for every design cell of a test record without a row, an
    excluded line where compose's excluded.jsonl drops it (model null or the cell's, fmt null or the cell's), else an
    ok line (compose dropped the record for another format); the pairs of `missing` (record_id, model, condition, seed)
    become missing lines (their last attempt `last`), the pairs of `drop` are left out and `extra` lines are added.
    Written into `tmp` (an accepted_missing scenario's own repo, fresh_generation_repo, so its canonical
    SCORES_META write never collides with another scenario's); the code copy that names the design is read from
    `cfg_repo` instead when given (`tmp` itself holds no "repo" of its own)."""
    cfg = json.loads(((cfg_repo or tmp) / "repo" / "config" / "sap.json").read_text())
    read = lambda f: [json.loads(l) for l in f.read_text(encoding="utf-8").split("\n") if l]
    compose, ids = read(x["compose_excluded"]), [r["id"] for r in read(x["records"])]
    lines = {(r["record_id"], r["model"], r["condition"], r["seed"]): mline(r)
             for r in read(x["scores"]) + read(x["aux_scores"])}
    for m, seeds in cfg["seeds"].items():
        for c in cfg["base_conditions"] + cfg["adapter_conditions"]:
            for s in ["base"] if c in cfg["base_conditions"] else seeds:
                for rid in ids:
                    out = any(e["record_id"] == rid and e.get("model") in (None, m) and e.get("fmt") in (None, FMT[c])
                              for e in compose)
                    lines.setdefault((rid, m, c, s), mline({"record_id": rid, "model": m, "condition": c, "seed": s},
                                                           "excluded" if out else "ok"))
    for k in missing:
        lines[k] = mline(dict(zip(("record_id", "model", "condition", "seed"), k)), "missing", last)
    for k in drop:
        del lines[k]
    m = tmp / "manifest.main.jsonl"
    m.write_text("".join(json.dumps(v) + "\n" for _, v in sorted(lines.items())) + "".join(json.dumps(v) + "\n"
                                                                                         for v in extra))
    c = tmp / "coverage.main.json"
    c.write_text(json.dumps({"manifest_sha256": sha(m), "status": "FAIL" if missing else "PASS",
                             "status_except_missing": "PASS"}))
    scores_of = {"test": x["scores"], "aux_test": x["aux_scores"]}
    for name, part in (("SCORES_META.json", "test"), ("SCORES_META.aux.json", "aux_test")):
        (tmp / name).write_text(json.dumps({"partition": part, "mode": "outputs", "manifest_sha256": sha(m),
                                            "coverage_sha256": sha(c), "accepted_missing_sha256":
                                            accepted and sha(accepted), "accepted_missing_commit": commit,
                                            "scored_utc": scored,
                                            "scores_sha256": sha(scores_of[part])}))  # the scores file's own sha256


def accept(tmp: Path, x: dict, pairs, scored: str = SCORED, entries=None, push: bool = True,
           last: str = "CRASHED", commit_date: str | None = None, cfg_repo: Path | None = None) -> list:
    """The acceptance of the missing `pairs` (record_id, model, condition, seed), whose score rows the test
    leaves out: accepted_missing.jsonl (`entries`, by default one per pair with its manifest line's key and
    receipt_sha256), committed and pushed to origin (dated `commit_date` when given; see publish), and the generation
    files rewritten with those pairs missing and the SCORES_META naming the file and its commit; the sap_run
    arguments. Each acceptance is a new file, accepted/<uuid>/accepted_missing.jsonl, written once (since 27/09/2026 its
    path is written on origin by one commit only). `tmp` may be a fresh, otherwise-empty repo of its own
    (fresh_generation_repo, one repo per accepted_missing scenario); `cfg_repo` then names where the
    fixture's code copy (and so the design's config) actually lives, for write_generation."""
    f = tmp / "accepted" / uuid.uuid4().hex[:12] / "accepted_missing.jsonl"  # fresh, even after a `git reset`
    f.parent.mkdir(parents=True)
    lines = [mline(dict(zip(("record_id", "model", "condition", "seed"), k)), "missing") for k in pairs]
    write(f, entries if entries is not None else [{**{k: x[k] for k in KEY}, "reason": "rerun failed",
                                                   "receipt_sha256": x["receipt_sha256"]} for x in lines])
    commit = publish(tmp, f, push, date=commit_date)
    write_generation(tmp, x, pairs, f, commit, scored, last=last, cfg_repo=cfg_repo)
    return ["--accepted-missing", f, "--accepted-missing-commit", commit]


def table_shape_errors(report: str) -> list[str]:
    """The rows of a Markdown table in `report` whose number of "|" is not its header's (a cell that splits)."""
    out, header = [], None
    for line in report.split("\n"):
        if not line.startswith("|"):
            header = None
        elif header is None:
            header = line.count("|")
        elif line.count("|") != header:
            out.append(line)
    return out


def git(repo: Path, *args, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                           "commit.gpgsign=false", *map(str, args)], check=True, capture_output=True, text=True,
                          env={**os.environ, **env} if env else None).stdout.strip()


def publish(home: Path, path: Path | list[Path], push: bool = True, date: str | None = None,
            branch: str = "main") -> str:
    """`path` (one file or several, anywhere under `home`) committed to the git repository of `home` and, with `push`,
    pushed to its origin (a bare remote there; both made on first use), on `branch` (forced when it is not main); the
    commit. With `date` (an ISO time git accepts as GIT_AUTHOR_DATE/GIT_COMMITTER_DATE), the commit is dated there
    instead of now (used to keep a commit's ordering test independent of the earlier lock-precedes-scoring check,
    both of which land on "today" otherwise)."""
    if not (home / ".git").exists():
        subprocess.run(["git", "init", "-q", "--bare", str(home / "remote.git")], check=True)
        git(home, "init", "-q", "-b", "main")
        git(home, "remote", "add", "origin", home / "remote.git")
    paths = path if isinstance(path, list) else [path]
    git(home, "add", "-f", *(Path(q).resolve().relative_to(home.resolve()) for q in paths))
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date} if date else None
    git(home, "commit", "-q", "--allow-empty", "-m", f"add {', '.join(Path(q).name for q in paths)}", env=env)
    if push:
        git(home, "push", "-q", *(["-f"] if branch != "main" else []), "origin", f"HEAD:refs/heads/{branch}")
        git(home, "fetch", "-q", "origin")
    return git(home, "rev-parse", "HEAD")


def publish_metas(home: Path, metas: list[Path], date: str | None = None) -> str:
    """The SCORES_META files committed after the scoring (rule of 27/09/2026; an hour after their latest scored_utc
    unless `date`) and pushed to origin (branch scores-meta, forced: a test may move main back), as is done right after
    scoring; the commit. Not pushed when HEAD is not on origin: it holds a commit the test left unpublished on purpose
    (a lock or an accepted_missing file, or one pushed to a fork only), which must stay so for its own check to refuse
    the run first. Metas committed before and unchanged are not committed again: the commit that wrote them is returned
    (a META is written once, since 27/09/2026)."""
    rel = [Path(q).resolve().relative_to(home.resolve()) for q in metas]
    if not git(home, "status", "--porcelain", "--", *rel):
        return git(home, "log", "-1", "--format=%H", "--", *rel)
    if date is None:
        times = []
        for q in metas:
            try:
                times.append(dt.datetime.fromisoformat(json.loads(Path(q).read_text())["scored_utc"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        date = (max(times) + dt.timedelta(hours=1)).isoformat() if times else None
    pushed = any(b.startswith("origin/") for b in git(home, "branch", "-r", "--contains", "HEAD").split())
    return publish(home, list(metas), push=pushed, date=date, branch="scores-meta")


def anchor(lock: Path) -> list:
    """--repo and --lock-commit for `lock`: the lock committed to a git repository in its folder and pushed to a bare
    remote there (both made on first use). Since 28/09/2026 a lock unchanged since its last commit at that path is not
    committed again (the same idiom as publish_metas' skip-if-unchanged), so repeatedly anchoring an untouched lock
    (sap_run's implicit auto-fill on every call) never advances the lock commit PAST a SCORES_META already anchored on
    top of the earlier one -- which would flip the ancestor relationship the lock must keep with it."""
    home = lock.parent
    rel = lock.resolve().relative_to(home.resolve()).as_posix()
    if (home / ".git").exists() and not git(home, "status", "--porcelain", "--", rel):
        return ["--repo", home, "--lock-commit", git(home, "log", "-1", "--format=%H", "--", rel)]
    return ["--repo", home, "--lock-commit", publish(home, lock)]


def canonical_meta(repo: Path, partition: str = "test") -> Path:
    """The one path a SCORES_META.json may sit at under --repo (fixed 28/09/2026),
    out/phase3/scores/<partition>/SCORES_META.json ("test" for the confirmatory run, "aux_test" for --descriptive-only);
    its parent directories are created so a test can write straight into it."""
    p = repo / "out" / "phase3" / "scores" / partition / "SCORES_META.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fresh_generation_repo(base_dir: Path, lock_bytes: bytes, name: str) -> tuple[Path, Path, list]:
    """A repository of its own (its own bare origin), under base_dir/repos/<name>, holding a committed and pushed copy
    of the lock (`lock_bytes`): every distinct SCORES_META scenario gets its own canonical-path history, so that
    different content at that one path never collides across scenarios sharing the SAME repo (and the lock
    committed here always precedes anything committed on top of it in this repo, whatever other repos hold). Returns
    (the lock copy inside it, the repo directory, anchor()'s --repo/--lock-commit for it)."""
    home = base_dir / "repos" / name
    home.mkdir(parents=True)
    lock_copy = home / "SAP_LOCK.json"
    lock_copy.write_bytes(lock_bytes)
    return lock_copy, home, anchor(lock_copy)


def run(repo: Path, script: str, *args):
    """A script of the fixture's code copy; sap_run's --lock is first committed and published (anchor) unless the test
    passes --lock-commit itself, and sap_run gets the fixture's --manifest, --coverage and --scores-meta (the aux one
    with --descriptive-only) unless the test passes them, then --scores-meta-commit (publish_metas of those metas)
    unless the test passes it. The auto-filled --scores-meta is the fixture's base one with scores_sha256 patched to the
    actual current sha256 of the --scores file it is paired with (recomputed on every call), so a test that substitutes
    its own --scores file (a tampered or a shortened copy) still gets a SCORES_META that agrees with it and exercises
    whatever OTHER rule it means to, unless it deliberately wants the scores_sha256 rule itself (which passes
    --scores-meta itself to opt out of this auto-fill). Since 28/09/2026 the auto-filled META is always written at its
    canonical path (canonical_meta), one per partition, so exactly one --scores file may be auto-filled per call; when
    the call needs BOTH an implicit --lock-commit AND an implicit --scores-meta and carries no --accepted-missing (the
    common, isolated case), it gets its own fresh repository (fresh_generation_repo) so its canonical write never
    collides with another such call's in the same test, and the lock-ancestor rule holds trivially (the lock committed
    first, in the same repo). A call that also carries --accepted-missing keeps the --lock file's own directory as
    --repo instead, since accept() always anchors that file there: every such call in the suite reuses the same --scores
    bytes throughout, so the canonical write there is made at most once (a repeat is a same-bytes no-op, never a second
    genuine write)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    args = list(args)
    is_sap_run = script == "sap_run.py"
    gen_home = Path(args[args.index("--lock") + 1]).parent if is_sap_run and "--lock" in args else None
    if is_sap_run and "--lock" in args and "--manifest" not in args:
        args += ["--manifest", gen_home / "manifest.main.jsonl", "--coverage", gen_home / "coverage.main.json"]
    needs_lock_commit = is_sap_run and "--lock" in args and "--lock-commit" not in args
    needs_scores_meta = is_sap_run and "--scores" in args and "--scores-meta" not in args
    if needs_lock_commit and needs_scores_meta and "--accepted-missing" not in args:
        lock = Path(args[args.index("--lock") + 1])
        lock_copy, home, anch = fresh_generation_repo(gen_home, lock.read_bytes(), uuid.uuid4().hex[:12])
        args[args.index("--lock") + 1] = lock_copy
        args += anch
    elif needs_lock_commit:
        args += anchor(Path(args[args.index("--lock") + 1]))
    if needs_scores_meta:
        home = Path(args[args.index("--repo") + 1]) if "--repo" in args else gen_home
        i = args.index("--scores") + 1
        n = next((j for j in range(i, len(args)) if str(args[j]).startswith("--")), len(args)) - i
        if n != 1:
            raise ValueError("the canonical SCORES_META path holds one --scores file per run")
        partition = "aux_test" if "--descriptive-only" in args else "test"
        base_meta = json.loads((gen_home / ("SCORES_META.aux.json" if partition == "aux_test"
                                           else "SCORES_META.json")).read_text())
        meta = canonical_meta(home, partition)
        meta.write_text(json.dumps({**base_meta, "scores_sha256": sha(Path(args[i]))}))
        args += ["--scores-meta", meta]
    if is_sap_run and "--scores" in args and "--scores-meta-commit" not in args:  # publish the metas
        home = Path(args[args.index("--repo") + 1]) if "--repo" in args else gen_home
        i = args.index("--scores-meta") + 1
        n = next((j for j in range(i, len(args)) if str(args[j]).startswith("--")), len(args)) - i
        args += ["--scores-meta-commit", publish_metas(home, [Path(q) for q in args[i:i + n]])]
    return subprocess.run([sys.executable, str(repo / "scripts" / script), *map(str, args)], cwd=repo, env=env,
                          capture_output=True, text=True)


def lock_args(repo, tmp, tc, dc, x, er=None):
    return ["--config", repo / "config" / "sap.json", "--mde", tmp / "MDE.json", "--calibration", tmp / "CALIBRATION.json",
            "--dev-selection", tmp / "DEV_SELECTION.json", "--dev-scores", x["dev"], x["dev_k4"], "--dev-controls-dir", dc,
            "--controls-dir", tc, "--compose-excluded", x["compose_excluded"], "--records", x["records"],
            "--subsets", x["subsets"], "--self-contained", x["self_contained"], "--phase2-sums", x["phase2_sums"],
            "--er-controls", er or x["er"], "--er-manifest", x["er_manifest"], "--aux-controls-dir", x["aux_controls"],
            "--aux-records", x["aux_records"], "--pre-t-flags", x["pre_t_flags"]]


def test_sap_lock_refuses_a_code_tree_that_is_not_clean(tmp_path):
    """Since 05/10/2026 the lock hashes the 3D code of a clean git checkout only. An untracked scripts/sap_*.py, a
    sap_run.py changed and not committed, an ignored scripts/sap_*.py, or a symlink among the hashed files stops
    sap_lock (exit 2, naming the file) before it writes the lock; the clean tree locks."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    args = [*lock_args(repo, tmp_path, tc, dc, x), "--out", lock]
    stray = repo / "scripts" / "sap_scratch.py"
    stray.write_text("print(1)\n")
    r = run(repo, "sap_lock.py", *args)
    assert r.returncode == 2 and "scripts/sap_scratch.py" in r.stderr and not lock.exists(), r.stderr
    stray.unlink()
    run_py = repo / "scripts" / "sap_run.py"
    good = run_py.read_bytes()
    run_py.write_bytes(good + b"# changed\n")
    r = run(repo, "sap_lock.py", *args)
    assert r.returncode == 2 and "scripts/sap_run.py" in r.stderr and not lock.exists(), r.stderr
    run_py.write_bytes(good)
    ignored = repo / "scripts" / "sap_ignored.py"  # since 05/10/2026 an ignored file is hashed too, so it is refused
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("scripts/sap_ignored.py\n")
    ignored.write_text("print(1)\n")
    r = run(repo, "sap_lock.py", *args)
    assert r.returncode == 2 and "scripts/sap_ignored.py" in r.stderr and not lock.exists(), r.stderr
    ignored.unlink()
    (repo / "scripts" / "sap_link.py").symlink_to(repo / "scripts" / "sap_mde.py")
    git(repo, "add", "scripts/sap_link.py")
    git(repo, "commit", "-q", "-m", "a committed symlink")
    r = run(repo, "sap_lock.py", *args)
    assert r.returncode == 2 and "scripts/sap_link.py" in r.stderr and "symlink" in r.stderr and not lock.exists()
    git(repo, "rm", "-q", "scripts/sap_link.py")
    git(repo, "commit", "-q", "-m", "no symlink")
    r = run(repo, "sap_lock.py", *args)
    assert r.returncode == 0, r.stderr


def mde_args(repo, tmp, tc, dc, x, *dev, out="MDE.json"):
    return ["--config", repo / "config" / "sap.json", "--dev-scores", *(dev or (x["dev"], x["dev_k4"])), "--dev-controls-dir", dc,
            "--test-controls-dir", tc, "--compose-excluded", x["compose_excluded"], "--calibration", tmp / "CALIBRATION.json",
            "--out", tmp / out]


def prepare(tmp: Path, edit=None) -> tuple:
    """setup, `edit` of its inputs if given, then CALIBRATION.json, DEV_SELECTION.json and MDE.json from them."""
    repo, tc, dc, ids, x = setup(tmp)
    if edit:
        edit(repo, tc, dc, ids, x)
    cfg = repo / "config" / "sap.json"
    r = run(repo, "sap_calibrate.py", "--config", cfg, "--records", x["records"], "--controls-dir", tc, "--compose-excluded",
            x["compose_excluded"], "--sims", 4, "--reps", 49, "--jobs", 2, "--out", tmp / "CALIBRATION.json")
    assert r.returncode == 0, r.stderr
    r = run(repo, "sap_dev_select.py", "--config", cfg, "--dev-scores", x["dev"], x["dev_k4"], "--dev-controls-dir", dc,
            "--out", tmp / "DEV_SELECTION.json")
    assert r.returncode == 0, r.stderr
    r = run(repo, "sap_mde.py", *mde_args(repo, tmp, tc, dc, x))
    assert r.returncode == 0, r.stderr
    return repo, tc, dc, ids, x


SEPARATORS = "Mua\u2028hai\u2029món\x85xong"  # str.splitlines breaks at each; json.dumps(ensure_ascii=False) keeps them


def raw_rows(path: Path, **extra) -> None:
    """Rewrite a JSONL file with `extra` in every row, UTF-8 with ensure_ascii=False (U+2028, U+2029, U+0085 raw)."""
    xs = [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    path.write_bytes("".join(json.dumps({**x, **extra}, ensure_ascii=False) + "\n" for x in xs).encode("utf-8"))


def test_jsonl_readers_split_on_newlines_only(tmp_path):
    """Every 3D JSONL reader is sap.jsonl_rows (3A's symcache.jsonl_rows rule: "\\n" only), so
    a string field holding U+2028, U+2029 or U+0085, which json.dumps(ensure_ascii=False) leaves raw, no longer breaks
    its row. The test controls, both dev score files and the test scores carry one, so sap_calibrate, sap_dev_select,
    sap_mde, sap_lock and sap_run each read such a file; every row is read whole. A blank line inside an input is
    refused as bad input: exit 2 for sap_calibrate and sap_lock, 3 for sap_dev_select, sap_mde and sap_run."""
    def edit(repo, tc, dc, ids, x):
        for p in (tc / "controls.jsonl", x["dev"], x["dev_k4"], x["scores"]):
            raw_rows(p, scenario_name=SEPARATORS)
    repo, tc, dc, ids, x = prepare(tmp_path, edit)
    text = x["scores"].read_bytes().decode("utf-8")
    assert "\u2028" in text and "\x85" in text and len(text.splitlines()) > text.count("\n")
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 0, r.stderr
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text(encoding="utf-8"))
    assert res["n_rows"] == text.count("\n") and res["completeness"]["missing_filled"] == 0
    tc_blank = tmp_path / "tc_blank"  # each input below has a blank line 2
    shutil.copytree(tc, tc_blank)
    dev, scores, er = (tmp_path / f"{k}_blank.jsonl" for k in ("dev", "scores", "er"))
    for src, dst in ((tc / "controls.jsonl", tc_blank / "controls.jsonl"), (x["dev"], dev), (x["scores"], scores),
                     (x["er"], er)):
        dst.write_bytes(src.read_bytes().replace(b"\n", b"\n\n", 1))
    cfg = repo / "config" / "sap.json"
    calibrate = ["--config", cfg, "--records", x["records"], "--controls-dir", tc_blank, "--compose-excluded",
                 x["compose_excluded"], "--sims", 2, "--reps", 9, "--out", tmp_path / "c.json"]
    select = ["--config", cfg, "--dev-scores", dev, x["dev_k4"], "--dev-controls-dir", dc, "--out", tmp_path / "d.json"]
    for script, args, code, bad in (
            ("sap_calibrate.py", calibrate, 2, tc_blank / "controls.jsonl"),
            ("sap_dev_select.py", select, 3, dev),
            ("sap_mde.py", mde_args(repo, tmp_path, tc, dc, x, dev, out="m.json"), 3, dev),
            ("sap_lock.py", [*lock_args(repo, tmp_path, tc, dc, x, er=er), "--out", tmp_path / "LOCK2.json"], 2, er),
            ("sap_run.py", [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", scores,
                            "--out", tmp_path / "sap2"], 3, scores)):
        r = run(repo, script, *args)
        assert r.returncode == code and r.stderr.startswith(f"refusing: {bad} line 2 is not a JSON row"), \
            (script, r.stderr)


def test_calibrate_select_mde_lock_run(tmp_path):
    repo, tc, dc, ids, x = prepare(tmp_path)
    cal = json.loads((tmp_path / "CALIBRATION.json").read_text())
    assert cal["seeds"] == 5 and cal["scenarios"] == len({i.rsplit("|", 1)[0] for i in ids[3:]}) and set(cal["r"]) == {"base", "adapter"}
    assert all(v["n"] == 4 for v in cal["results"].values())
    assert cal["inputs"] == {"records": sha(x["records"]), "controls": sha(tc / "controls.jsonl"),
                             "exclusions": sha(tc / "exclusions.jsonl"), "compose_excluded": sha(x["compose_excluded"])}
    sel = json.loads((tmp_path / "DEV_SELECTION.json").read_text())
    assert sel["choice"] == "dev_pilot_v0" and set(sel["scores"]) == {"dev_pilot_v0", "k4"}
    assert sel["inputs"] == {str(p): sha(p) for p in (x["dev"], x["dev_k4"], dc / "exclusions.jsonl")}
    m = json.loads((tmp_path / "MDE.json").read_text())
    usable = ids[3:]  # 2 mechanical exclusions, 1 compose exclusion for every model; mistral's does not count
    assert m["n_target"] == len({i.rsplit("|", 1)[0] for i in usable}) and m["dev_variant"] == "dev_pilot_v0"
    assert m["mde"]["RQ2:full_aware_vs_full_neutral"]["mde_two_level@tau_0"] > 0
    assert m["dev_inputs"] == {str(p): sha(p) for p in (x["dev"], x["dev_k4"], dc / "exclusions.jsonl")}
    # k4's file is read but left out; the MDE is that of the locked variant alone
    r = run(repo, "sap_mde.py", *mde_args(repo, tmp_path, tc, dc, x, x["dev"], out="MDE_v0.json"))
    assert r.returncode == 0 and json.loads((tmp_path / "MDE_v0.json").read_text())["mde"] == m["mde"]
    r = run(repo, "sap_mde.py", *mde_args(repo, tmp_path, tc, dc, x, x["dev_k4"], out="MDE_k4.json"))
    assert r.returncode == 3 and "locked variant" in r.stderr
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 0, r.stderr
    locked = json.loads(lock.read_text())
    assert locked["code"] == {p: sha(repo / p) for p in code_files(repo)} and set(locked["inputs"]) == INPUTS
    assert locked["inputs"]["phase2_sums"] == sha(x["phase2_sums"]) and locked["inputs"]["dev_scores[1]"] == sha(x["dev_k4"])
    assert locked["inputs"]["records"] == sha(x["records"]) and locked["inputs"]["er_manifest"] == sha(x["er_manifest"])
    assert locked["inputs"]["self_contained"] == sha(x["self_contained"])  # it decides subset (iv)
    assert locked["inputs"]["future_methods"] == sha(tc / "future_methods.jsonl")  # it decides CA (helper expansion)
    assert locked["inputs"]["pre_t_flags"] == sha(x["pre_t_flags"])  # the frozen pre-T list
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 2
    out = tmp_path / "sap"
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--outputs", x["outputs"], "--out", out)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "SAP_RESULTS.json").read_text())
    rq2 = res["confirmatory"]["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]
    assert rq2["verdict"] == "supported" and rq2["ca"]["two_level"]["p_holm"] < 0.05 and len(rq2["ca"]["per_seed"]) == 5
    assert res["completeness"] == {"usable_records": len(usable), "missing_filled": 0,
                                   "records_covered": {"deepseek": len(ids) - 1, "codellama": len(ids) - 1,
                                                       "mistral": len(ids) - 2}}  # every model
    assert set(res["descriptive"]["rq3_full_aware_by_model"]) == {"deepseek", "codellama", "mistral"}
    # P14, the RQ3 profile (fixed 07/10/2026): each model's Full-aware levels; CA as
    # rq3_full_aware_by_model; ER of ids[5] for the anchor only here (the fixture's
    # ER rows), so the other models' ER metrics are null with their reason, never 0
    prof = res["descriptive"]["rq3_profile"]
    assert list(prof) == ["codellama", "deepseek", "mistral"]
    for model, lv in res["descriptive"]["rq3_full_aware_by_model"].items():
        assert prof[model]["ca"] == {**{k: lv[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")},
                                     "reason": None}  # the very level, one CI per number
    assert prof["deepseek"]["er"]["n_scenarios"] == 1 and prof["deepseek"]["er"]["reason"] is None
    assert all(prof[m]["er"]["estimate"] is None and "does not cover" in prof[m]["er"]["reason"]
               for m in ("codellama", "mistral"))
    rq3 = (out / "SAP_REPORT.md").read_text().split("## RQ3: hồ sơ theo mô hình ở Full-aware (mô tả)")[1]
    rq3 = rq3.split("\n## ")[0]
    d = prof["deepseek"]["er"]
    assert f"| deepseek | {prof['deepseek']['ca']['estimate']:.3f} [" in rq3 and f"{d['estimate']:.3f} [" in rq3
    assert "- codellama · ER: no ER value: the ER run does not cover this model" in rq3
    assert "CHƯA ĐO" not in rq3 and "aux_test chỉ có" not in rq3
    covered = res["completeness"]["records_covered"]  # a model's own denominator (compose per model)
    assert ("Bản ghi trong thiết kế của từng mô hình (P4; compose loại riêng theo mô hình): "
            + ", ".join(f"{m} {covered[m]}" for m in sorted(covered)) + ".") in rq3
    assert covered["mistral"] < covered["deepseek"]
    # the lock was authenticated by a published commit, whose dates are printed
    assert res["lock_commit"]["path"] == "SAP_LOCK.json" and "CommitDate:" in res["lock_commit"]["fuller"]
    assert len(res["answers"]) == 6
    assert res["accepted_missing"] is None  # the outputs are tied to the manifest, and nothing is missing
    assert res["generation"]["manifests"] == [{"manifest": str(tmp_path / "manifest.main.jsonl"),
                                               "sha256": sha(tmp_path / "manifest.main.jsonl"),
                                               "coverage": str(tmp_path / "coverage.main.json"),
                                               "coverage_sha256": sha(tmp_path / "coverage.main.json")}]
    assert res["generation"]["scores_meta"][0]["partition"] == "test"
    assert "Manifest không có dòng `missing`" in (out / "SAP_REPORT.md").read_text()
    assert res["rehearsal"] is False and REHEARSAL not in (out / "SAP_REPORT.md").read_text()  # 26/09/2026: no banner
    assert rq2["descriptive"]["trace_then@self_contained"]["n_scenarios"] == 0
    # ids[2] (compose-excluded for every model) is out of BV too; ids[:2] (mechanical, not excl_bv) only of CA
    pr = res["descriptive"]["per_run"]["deepseek|full_aware|main|s42"]
    assert pr["bv"]["n"] == len(ids) - 1 and pr["ca"]["n"] == len(usable)
    scored = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    rs_of = lambda cond, seed=None: [r for r in scored if (r["model"], r["variant"], r["condition"]) ==
                                     ("deepseek", "main", cond) and r["record_id"] in usable
                                     and seed in (None, r["seed"])]
    # the fifth extract status and the fence_truncated rate (over the rows with a text)
    n = len(ids) - 1
    assert pr["extract"] == {"ok": n - 4, "multi_file_fence": 4, "unclosed": 0, "malformed": 0, "none": 0, "n": n}
    assert pr["fence_truncated"] == {"n": n, "rate": 4 / n}
    # the robustness views: the four subsets (subset (iv) from the locked self_contained.jsonl), strata,
    # groups, each repo left out and without the pre-T records, exploratory: n, estimate and CI, no test
    rob = res["descriptive"]["robustness"]
    assert not {"sensitivity", "strata", "preexisting"} & set(res["descriptive"])
    assert list(rob) == ["subset:post_cutoff_all", "subset:time_coherent_30d", "subset:residual_clean",
                         "subset:self_contained", "repo=R", "language=php", "preexisting=True", "preexisting=False",
                         "without_repo=R", "without_pre_t", "len_bucket=le8192", "len_bucket=gt8192"]
    assert 0 < rob["subset:residual_clean"]["full_aware_vs_full_neutral"]["n_scenarios"] < rq2["ca"]["n_scenarios"]
    assert rob["subset:time_coherent_30d"]["full_aware_vs_full_neutral"]["n_scenarios"] == 0
    assert rob["without_pre_t"]["full_aware_vs_full_neutral"]["n_scenarios"] == \
        rob["subset:post_cutoff_all"]["full_aware_vs_full_neutral"]["n_scenarios"] - 1  # ids[20:22]: one scenario
    sc = {o["record_id"]: o["self_contained"] for o in map(json.loads, x["self_contained"].read_text().splitlines())}
    iv = rob["subset:self_contained"]["full_aware_vs_full_neutral"]
    assert 0 < iv["n_scenarios"] == len({i.rsplit("|", 1)[0] for i in usable if sc[i]}) < rq2["ca"]["n_scenarios"]
    assert set(iv) == {"n_scenarios", "n_clusters", "estimate", "ci95"}
    report = (out / "SAP_REPORT.md").read_text()
    section = report.split("## Độ vững (khám phá)")[1].split("\n## ")[0]
    assert "Holm" not in section.split("|", 1)[1] and "supported" not in section and "RQ1" not in section
    assert (f"| subset:self_contained | full_aware_vs_full_neutral | {iv['n_scenarios']} / {iv['n_clusters']} | "
            f"{iv['estimate']:+.3f} | [{iv['ci95'][0]:+.3f}, {iv['ci95'][1]:+.3f}] |\n") in section
    assert res["pre_t"] == {"file": str(x["pre_t_flags"]), "sha256": sha(x["pre_t_flags"]),
                            "counts": {"aux_test": 2, "dev": 1, "test": 2}, "partition": "test", "dropped": 2}
    assert (f"danh sách pre-T đã khoá `{x['pre_t_flags']}` (sha256 `{sha(x['pre_t_flags'])}`): góc nhìn "
            f"without_pre_t bỏ 2 bản ghi của phần test.") in section
    # the robust verdicts and the denominators table; the dose rule without a locked table
    assert res["conclusion_robust"]["rq"]["RQ2"] == {"ca": "full", "ca_prec": "full", "robust": True,
                                                     "label": "vững"}  # the text label next to "robust"
    ca2 = res["confirmatory"]["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]["ca"]["estimate"]
    assert "không chọn một trong hai thước đo" in report and f"| supported | {ca2:+.3f} | supported | vững |" in report
    main = report.split("## Kiểm định xác nhận")[1].split("\n## ")[0]  # P22 (27/09/2026): next to every conclusion
    assert [l for l in main.split("\n") if l.startswith("| RQ")] and all(
        l.endswith("| supported | vững |") for l in main.split("\n") if l.startswith("| RQ"))
    assert "- RQ1: **full** (vững)\n" in main and "- RQ2: **full** (vững)\n" in main
    assert all(a.endswith(" Độ vững (CA và CA_prec): vững.") for a in res["answers"])
    scen = lambda rs: len({i.rsplit("|", 1)[0] for i in rs})
    assert res["denominators"][0] == {"repo": "R", "partition": "test", "records": len(ids), "scenarios": scen(ids),
                                      "ca_records": len(usable), "ca_scenarios": scen(usable), "excluded": 2,
                                      "dropped": 1} and res["denominators"][1]["partition"] == "aux_test"
    assert f"| R | test | {len(ids)} | {scen(ids)} | {len(usable)} | {scen(usable)} | 2 | 1 |\n" in report
    # P26 (fixed 06/10/2026): the expected directions of §2.4, the label beside the P11 flag, duplicate_reuse and exact
    # per run, the prompt length groups by the anchor's Full-aware prompt (every record of the fixture is le8192)
    con = res["descriptive"]["contrasts"]
    assert list(con) == ["hallucinated_rate", "conv_style", "conv_pattern", "conv_naming"]
    assert all((c["n_scenarios"], c["estimate"], c["ci95"], c["direction"]) ==
               (scen([i for i in ids if i != ids[2]]), 0.0, [0.0, 0.0], "không rõ chiều")
               for k, c in con.items() if k != "conv_style")
    assert (con["conv_style"]["estimate"], con["conv_style"]["ci95"], con["conv_style"]["direction"]) == (
        0.5, [0.5, 0.5], "theo chiều kỳ vọng")
    h, cs = con["hallucinated_rate"], con["conv_style"]
    section = report.split("## Chiều kỳ vọng của thước đo mô tả")[1].split("\n## ")[0]
    assert (f"| ký hiệu bịa (hallucinated / members_used) | < 0 | {h['n_scenarios']} / {h['n_clusters']} | +0.000 | "
            "[+0.000, +0.000] | không rõ chiều |\n") in section
    assert (f"| quy ước: dạng annotation | > 0 | {cs['n_scenarios']} / {cs['n_clusters']} | +0.500 | "
            "[+0.500, +0.500] | theo chiều kỳ vọng |\n") in section
    bv2 = rq2["descriptive"]["bv"]
    assert bv2["direction"] == "theo chiều kỳ vọng" and bv2["opposite"] is False
    assert (f"| full_aware_vs_full_neutral | bv | {bv2['n_scenarios']} / {bv2['n_clusters']} | "
            f"{bv2['estimate']:+.3f} | [{bv2['ci95'][0]:+.3f}, {bv2['ci95'][1]:+.3f}] | theo chiều kỳ vọng "
            "|  |\n") in report
    assert "| Phép so | Thước đo | n kịch bản / cụm | Ước lượng | KTC 95% | Chiều (quy ước §2.4) | Cờ P11 |" in report
    er2 = rq2["descriptive"]["er"]  # ER of ids[5] alone: one cluster, so no direction
    assert er2["n_clusters"] == 1 and er2["direction"] == "không rõ chiều"
    assert f"| full_aware_vs_full_neutral | er | {er2['n_scenarios']} / 1 | {er2['estimate']:+.3f} | " in report
    dose = res["descriptive"]["dose"]  # P26 (e): each bucket labelled, the rise is not
    assert list(dose) == ["0-0", "1-2", "3-"] and all(d["direction"] for d in dose.values())
    assert all((f"| {b} | {d['n_scenarios']} / {d['n_clusters']} | {d['estimate']:+.3f} | [{d['ci95'][0]:+.3f}, "
                f"{d['ci95'][1]:+.3f}] | {d['direction']} |\n") in section for b, d in dose.items())
    assert "KHÔNG gắn nhãn xu hướng" in section
    assert (pr["duplicate_reuse"], pr["exact"]) == ({"n": n, "rate": 2 / n}, {"n": n, "rate": 3 / n})
    assert f"| deepseek · full_aware · main · s42 | {2 / n:.3f} ({n}) | {3 / n:.3f} ({n}) |\n" in report
    assert res["descriptive"]["len_strata"] == {"records": {"le8192": len(ids) - 5, "gt8192": 3}, "unplaced": 1}
    assert rob["len_bucket=gt8192"]["full_aware_vs_full_neutral"]["n_scenarios"] == scen(ids[30:33])
    assert rob["len_bucket=le8192"]["full_aware_vs_full_neutral"]["n_scenarios"] == scen(
        [i for i in usable if i not in ids[30:34]])
    assert (f"{len(ids) - 5} bản ghi ≤ 8.192 token (len_bucket=le8192), 3 bản ghi > 8.192 (len_bucket=gt8192); 1 bản "
            "ghi không có dòng Full-aware thật nên ra khỏi cả hai nhóm.") in report.split("## Độ vững (khám phá)")[1]
    assert [(t["type"], t["tau"]) for t in res["calibration_type_i"]] == [(m, t) for m in ("base", "adapter")
                                                                          for t in ("0", "0.02", "0.04")]
    assert "## Sai lầm loại I đo được của hai phép thử" in report
    assert res["dose_checked"] is False and "dose chưa được đối chiếu với đầu vào đã khoá" in report
    assert f"| deepseek · full_aware · main · s42 | {n} | {n - 4} | 4 | 0 | 0 | 0 | {4 / n:.3f} ({n}) |" in report
    # a run key is shown as its label (" · "), never with a raw "|" that would split a table cell
    pm = lambda v: "–" if v is None else f"{v:+.3f}"
    ca, bv = (res["descriptive"]["by_repo"]["R"][m]["deepseek|full_aware|main"] for m in ("ca", "bv"))
    lz = res["descriptive"]["lazy_by_repo"]["R"]["deepseek|full_aware|main"]  # the lazy references too
    assert (f"| R | deepseek · full_aware · main | {pm(ca['estimate'])} [{pm(ca['ci95'][0])}, {pm(ca['ci95'][1])}] | "
            f"{pm(bv['estimate'])} [{pm(bv['ci95'][0])}, {pm(bv['ci95'][1])}] | 0.000 | 0.000 | 0.000 | "
            f"{lz['ca_ref_top4']['rate']:.3f} |  |\n") in report
    assert "deepseek|" not in report and table_shape_errors(report) == []
    # D11 (design specification at 3739dd6): CA_prec's comparisons next to CA's
    # (call_precision 0.5 >= 0.25 everywhere: the same conclusions, said so), the
    # controls' CA_prec rates, ca_ref_v3names next to v3m and v3all, and the call spread
    cp = res["ca_prec"]
    assert cp["tau"] == 0.25 and cp["differences"] == ["Kết luận theo CA_prec trùng với CA ở mọi phép so và mọi RQ."]
    prec = cp["confirmatory"]["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]
    assert prec["ca_prec"] == rq2["ca"] and prec["verdict"] == rq2["verdict"] == "supported"
    assert cp["controls"]["gold"] == {"n": len(usable), "ca": 1.0, "ca_prec@0.1": 1.0, "ca_prec@0.25": 1.0,
                                      "ca_prec@0.5": 1.0}
    assert cp["controls"]["stub-v3names"] == {"n": len(usable), "ca": 1.0, "ca_prec@0.1": 1.0, "ca_prec@0.25": 0.0,
                                              "ca_prec@0.5": 0.0} and cp["controls"]["swap"]["n"] == 0
    assert pr["ca_ref_v3names"] == {"n": len(usable), "rate": 0.0} and pr["ca_above_v3names"] == pr["ca"]
    assert pr["call_precision"] == {"n": len(usable), "median": 0.5, "q1": 0.5, "q3": 0.5}
    assert pr["n_calls"] == {"n": len(usable), "median": 2, "p90": 2}
    by_repo = res["descriptive"]["calls_by_repo"]["R"]["deepseek|full_aware|main"]
    assert by_repo["n_calls"]["n"] == 5 * len(usable) and by_repo["call_precision"]["median"] == 0.5
    assert "- Kết luận theo CA_prec trùng với CA ở mọi phép so và mọi RQ.\n" in report
    assert f"| stub-v3names | {len(usable)} | 1.000 | 1.000 | 0.000 | 0.000 |\n" in report
    # D11 (d) (design specification at 4d8bf2a): stub-v3top4 in the controls table,
    # ca_ref_top4 next to the other lazy references, and "CA_prec vượt mức lười top4"
    # next to CA_prec for the runs of every RQ1/RQ2 comparison (CA_prec = CA here)
    assert cp["controls"]["stub-v3top4"] == {"n": len(usable), "ca": 1.0, "ca_prec@0.1": 1.0, "ca_prec@0.25": 1.0,
                                             "ca_prec@0.5": 0.0}
    assert f"| stub-v3top4 | {len(usable)} | 1.000 | 1.000 | 1.000 | 0.000 |\n" in report
    assert pr["ca_ref_top4"] == {"n": len(usable), "rate": 1 / len(usable)}
    above = sum(r["ca"] == 1 and r["record_id"] != ids[6] for r in rs_of("full_aware", "s42"))
    assert pr["ca_prec_above_top4"] == {"n": len(usable), "rate": above / len(usable)}
    top4 = cp["above_top4"]
    assert list(top4) == ["full_aware", "zero_shot", "rag_only", "ft_only", "full_neutral"]
    for cond, v in top4.items():
        xs = rs_of(cond)
        assert v == {"n": len(xs), "rate": sum(r["ca"] == 1 and r["record_id"] != ids[6] for r in xs) / len(xs)}
    cell4 = lambda c: f"{top4[c]['rate']:.3f} ({top4[c]['n']})"
    assert any(l.startswith("| RQ2 | full_aware_vs_full_neutral | ") and l.endswith(
        f"| supported | {ca2:+.3f} | supported | vững | {cell4('full_aware')} · {cell4('full_neutral')} |")
               for l in report.split("\n"))
    assert "| v3names | top4 | CA vượt v3m | CA vượt v3all | CA vượt v3names | CA_prec vượt mức lười top4 |" in report
    # §1.4 (fixed 26/09/2026): the rate of php_tag_stripped per run, over the PHP rows with a text
    tag = res["descriptive"]["per_run"]["deepseek|zero_shot|main|base"]["php_tag_stripped"]
    assert tag == {"n": len(ids) - 1, "rate": 3 / (len(ids) - 1)} and pr["php_tag_stripped"]["rate"] == 0
    assert "| fence_truncated (n) | php_tag_stripped (n) |" in report
    # §2.3 (fixed 26/09/2026): docblock steps next to ER per repo and run, and P16 also on attribute-only outputs
    db = res["descriptive"]["docblock_by_repo"]["R"]["deepseek|rag_only|main"]
    er5 = next(r["er"] for r in scored if (r["record_id"], r["model"], r["condition"]) == (ids[5], "deepseek",
                                                                                            "rag_only"))
    assert db == {"n": len(ids) - 1, "docblock_rate": 2 / (len(ids) - 1), "docblock_steps": 2,
                  "attribute_steps": 2 * (len(ids) - 1), "er": {"n": 1, "rate": er5}}
    assert (f"| R | deepseek · rag_only · main | {er5:.3f} (1) | {2 / (len(ids) - 1):.3f} ({len(ids) - 1}) | 2 | "
            f"{2 * (len(ids) - 1)} |\n") in report.split("## Bước docblock PHP cạnh ER (mô tả)")[1]
    kap = res["descriptive"]["kappa_ca_trace_action"]
    assert list(kap) == ["trace", "trace@attribute_only", "trace@self_contained", "trace@self_contained@attribute_only"]
    assert kap["trace@attribute_only"]["ALL"]["n"] == kap["trace"]["ALL"]["n"] - 1  # ids[5] rag_only: the anchor's
    assert kap["trace"]["ALL"]["n"] == sum(v["n"] for k, v in kap["trace"].items() if k.startswith("deepseek|"))
    assert any(k.startswith("codellama|") for k in kap["trace"])  # P16, 07/10/2026: ALL pools the anchor's runs only
    section = report.split("## Đồng thuận CA × ER-trace (P16, mô tả)")[1].split("\n## ")[0]
    assert f"| trace@attribute_only | {kap['trace@attribute_only']['ALL']['n']} | " in section
    assert f"| {3 / (len(ids) - 1):.3f} ({len(ids) - 1}) | 0.000 ({len(ids) - 1}; 0) | 0.000 ({len(ids) - 1}; 0) | " in report
    # §2.2 of the design specification (fixed 27/09/2026), descriptive, per run: each reader count over the rows with a
    # text (share with >= 1, sum), and the rate of er_status ERROR over the rows with one (ids[5] only has an ER value)
    k7 = [k for k in res["descriptive"]["per_run"] if k.startswith("deepseek|rag_only|main|")]
    assert len(k7) == 1  # the fixture's rag_only run (one seed)
    v7 = res["descriptive"]["per_run"][k7[0]]
    un = v7["unread_step_annotations"]
    assert un == {"n": un["n"], "rate": 1 / un["n"], "sum": 2} and un["n"] == len(ids) - 1
    assert v7["cuke_escaped_patterns"] == v7["cuke_custom_types"] == {"n": un["n"], "rate": 0.0, "sum": 0}
    s5 = next(r["er_status"] for r in scored if (r["record_id"], r["model"], r["condition"], r["seed"]) ==
              (ids[5], "deepseek", "rag_only", k7[0].rsplit("|", 1)[1]))
    assert v7["er_error"] == {"n": 1, "rate": float(s5 == "ERROR")}
    assert (f"| {1 / un['n']:.3f} ({un['n']}; 2) | 0.000 ({un['n']}; 0) | 0.000 ({un['n']}; 0) | "
            f"{float(s5 == 'ERROR'):.3f} (1) |\n") in report
    assert ("| step không đọc được (n; tổng) | expression thoát (n; tổng) | kiểu tự định nghĩa (n; tổng) "
            "| ER ERROR (n) |") in report
    assert f"| 0.500 [0.500, 0.500] ({len(usable)}) | 2, p90 2 ({len(usable)}) |" in report
    assert f"| R | deepseek · full_aware · main | 0.500 [0.500, 0.500] ({5 * len(usable)}) | " in report
    assert "p Holm (2) hai tầng" in report and "Ước lượng từng seed" in report and (out / "SHA256SUMS").exists()
    # §0b of the design specification: deferred analyses do not run by default
    assert "learning_curve" not in res["descriptive"] and not (out / "semv_blind.jsonl").exists() and "Tạm bỏ" in report
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--outputs", x["outputs"], "--with-deferred", "--out", tmp_path / "full")
    assert r.returncode == 0, r.stderr
    assert "learning_curve" in json.loads((tmp_path / "full" / "SAP_RESULTS.json").read_text())["descriptive"]
    blind = [json.loads(l) for l in (tmp_path / "full" / "semv_blind.jsonl").read_text().splitlines()]
    assert len(blind) == 100 and all(set(b) == {"item_id", "record_id", "text"} and b["text"] for b in blind)
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["aux_scores"],
            "--descriptive-only", "--out", tmp_path / "aux")
    assert r.returncode == 0, r.stderr
    aux = json.loads((tmp_path / "aux" / "SAP_RESULTS.json").read_text())
    assert aux["confirmatory"] == {} and aux["answers"] == [] and aux["descriptive"]["levels"]
    assert set(aux["descriptive"]["by_repo"]) == {"X"}
    assert "deepseek|full_aware|main" in aux["descriptive"]["by_repo"]["X"]["ca"]
    assert "CHƯA ĐO" in (tmp_path / "aux" / "SAP_REPORT.md").read_text()
    assert "CHƯA ĐO" in (tmp_path / "aux" / "SAP_REPORT.md").read_text().split("\n## ")[0]  # not measured
    arq3 = (tmp_path / "aux" / "SAP_REPORT.md").read_text().split("## RQ3: hồ sơ theo mô hình ở Full-aware (mô tả)")[1]
    arq3 = arq3.split("\n## ")[0]  # aux has the anchor model only and no ER, by design (P14)
    assert "CHƯA ĐO" in arq3 and "aux_test chỉ có các lượt của mô hình neo và không có lượt ER" in arq3
    assert aux["descriptive"]["rq3_profile"]["deepseek"]["ca"]["reason"] is None
    assert aux["pre_t"]["partition"] == "aux_test" and aux["pre_t"]["dropped"] == 2
    arob = aux["descriptive"]["robustness"]  # the aux_test records of the pre-T list leave its robustness view
    assert arob["without_pre_t"]["full_aware_vs_full_neutral"]["n_scenarios"] == \
        arob["repo=X"]["full_aware_vs_full_neutral"]["n_scenarios"] - 1


def test_report_puts_the_direction_label_beside_the_p11_flag(tmp_path):
    """P11 and design specification §2.4 (fixed 06/10/2026): with a supported CA, a negative BV estimate whose CI holds
    0 reads "không rõ chiều" in the §2.4 direction column and CÓ in the column of the P11 `opposite` flag (kept as
    declared); no cell says "ngược chiều"."""
    def edit(repo, tc, dc, ids, x):
        rs = load_rows(x["scores"])
        first = sorted({r["scenario_key"] for r in rs if r["record_id"] in ids[3:]})[0]
        write(x["scores"], [{**r, "bv": int((r["model"], r["condition"], r["scenario_key"]) !=
                                            ("deepseek", "full_aware", first))} for r in rs])
    repo, tc, dc, ids, x = prepare(tmp_path, edit)
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 0, r.stderr
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    v = json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text())["confirmatory"]["RQ2"]["comparisons"][
        "full_aware_vs_full_neutral"]
    d = v["descriptive"]["bv"]
    assert v["verdict"] == "supported" and d["estimate"] < 0 and d["ci95"][0] < 0 <= d["ci95"][1]
    assert d["opposite"] is True and d["direction"] == "không rõ chiều"
    report = (tmp_path / "sap" / "SAP_REPORT.md").read_text()
    assert (f"| full_aware_vs_full_neutral | bv | {d['n_scenarios']} / {d['n_clusters']} | {d['estimate']:+.3f} | "
            f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | không rõ chiều | CÓ |\n") in report
    assert "| ngược chiều |" not in report


def test_lock_refuses_missing_inputs_or_stale_provenance(tmp_path):
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x, er=tmp_path / "no_er_run.jsonl"), "--out", lock)
    assert r.returncode == 2 and "missing" in r.stderr and "er_controls[0]" in r.stderr and not lock.exists()
    er_empty = write_er(tmp_path / "er_empty.jsonl", [])
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x, er=er_empty), "--out", lock)
    assert r.returncode == 2 and "empty" in r.stderr and not lock.exists()
    gold_only = write_er(tmp_path / "er_gold_only.jsonl", [json.loads(x["er"].read_text().splitlines()[0])])
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x, er=gold_only), "--out", lock)
    assert r.returncode == 2 and "partial ER control file" in r.stderr and not lock.exists()
    r = run(repo, "sap_calibrate.py", "--records", x["records"], "--compose-excluded", x["compose_excluded"], "--out", tmp_path / "c.json")
    assert r.returncode == 2 and "--controls-dir" in r.stderr
    sums = x["phase2_sums"].read_bytes()
    x["phase2_sums"].unlink()
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "phase2_sums" in r.stderr
    # the records file and sensitivity_subsets.json given here are the ones PHASE2_SHA256SUMS lists
    listed = sums.decode().splitlines()
    for i, what in ((1, "the records file given here is not PHASE2_SHA256SUMS's records/test.jsonl"),
                    (2, "the subsets file given here is not PHASE2_SHA256SUMS's sensitivity_subsets.json")):
        x["phase2_sums"].write_text("".join(("f" * 64 + l[64:] if j == i else l) + "\n" for j, l in enumerate(listed)))
        r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
        assert r.returncode == 2 and what in r.stderr and r.stderr.count("is not PHASE2_SHA256SUMS's") == 1
        assert "CALIBRATION.json" not in r.stderr and not lock.exists()
    x["phase2_sums"].write_bytes(sums)
    mde = lambda: run(repo, "sap_mde.py", *mde_args(repo, tmp_path, tc, dc, x)).returncode
    # a calibration on records/dev.jsonl (listed in PHASE2_SHA256SUMS too) does not pass for the test records
    cal_path = tmp_path / "CALIBRATION.json"
    cal = cal_path.read_bytes()
    dev_records = write(tmp_path / "dev_records.jsonl", [{**json.loads(l), "partition": "dev"}
                                                         for l in x["records"].read_text().splitlines()])
    x["phase2_sums"].write_text(sha(dev_records) + "  out/phase2/records/dev.jsonl\n"
                                + "".join(l + "\n" for l in listed[1:]))
    r = run(repo, "sap_calibrate.py", "--config", repo / "config" / "sap.json", "--records", dev_records,
            "--controls-dir", tc, "--compose-excluded", x["compose_excluded"], "--sims", 4, "--reps", 49,
            "--out", cal_path)
    assert r.returncode == 0 and mde() == 0
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "CALIBRATION.json inputs records is not the records file given here" in r.stderr
    assert "PHASE2_SHA256SUMS" not in r.stderr and "MDE.json" not in r.stderr and not lock.exists()
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, {**x, "records": dev_records}), "--out", lock)
    assert r.returncode == 2 and "the records file given here is not PHASE2_SHA256SUMS's records/test.jsonl" in r.stderr
    assert "CALIBRATION.json" not in r.stderr and not lock.exists()
    cal_path.write_bytes(cal)
    x["phase2_sums"].write_bytes(sums)
    assert mde() == 0
    # a test, dev-chosen or dev exclusions file changed after sap_calibrate / sap_dev_select, with sap_mde rerun on it
    for path, what in ((tc / "exclusions.jsonl", "CALIBRATION.json inputs ['exclusions'] are not the test files"),
                       (x["dev"], "DEV_SELECTION.json inputs do not hold the chosen variant's dev score files"),
                       (dc / "exclusions.jsonl", "DEV_SELECTION.json inputs do not hold the dev exclusions")):
        before = path.read_bytes()
        path.write_bytes(other_bytes(before))
        assert mde() == 0
        r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
        assert r.returncode == 2 and what in r.stderr and "MDE.json" not in r.stderr and not lock.exists()
        path.write_bytes(before)
        assert mde() == 0
    before = x["dev_k4"].read_bytes()  # another variant's dev file: DEV_SELECTION.json need not hold it
    x["dev_k4"].write_bytes(other_bytes(before))
    assert mde() == 0
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", tmp_path / "LOCK_k4.json").returncode == 0
    x["dev_k4"].write_bytes(before)
    assert mde() == 0
    cfg_path = repo / "config" / "sap.json"
    cfg = cfg_path.read_bytes()
    cfg_path.write_text(json.dumps({**json.loads(cfg), "power": 0.9}))  # edited after sap_calibrate and sap_mde
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "MDE.json config_sha256" in r.stderr and "CALIBRATION.json config_sha256" in r.stderr
    cfg_path.write_bytes(cfg)
    cal_path = tmp_path / "CALIBRATION.json"
    cal = cal_path.read_bytes()
    cal_path.write_text(json.dumps(json.loads(cal)))
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "MDE.json calibration_sha256" in r.stderr and "config_sha256" not in r.stderr
    cal_path.write_bytes(cal)
    dev = x["dev_k4"].read_bytes()
    x["dev_k4"].write_bytes(other_bytes(dev))
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "dev_inputs" in r.stderr
    x["dev_k4"].write_bytes(dev)
    args = lock_args(repo, tmp_path, tc, dc, x)
    i = args.index(x["dev_k4"])
    r = run(repo, "sap_lock.py", *args[:i], *args[i + 1:], "--out", lock)
    assert r.returncode == 2 and "dev_inputs" in r.stderr and not lock.exists()
    assert run(repo, "sap_lock.py", *args, "--out", lock).returncode == 0


def test_lock_and_run_refuse_er_controls_that_miss_an_exec_core_record(tmp_path):
    """The ER control run covers exactly the exec_core records of records/test.jsonl, with one gold and one
    stub row each: an exec_core record with no row at all, a duplicated row or a row of another record stops the lock,
    and the run under a lock that covers them."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    more = write(tmp_path / "records_more.jsonl", [{**r, "exec_core": r["id"] in ids[5:7]}
                                                   for r in map(json.loads, x["records"].read_text().splitlines())])
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, {**x, "records": more}), "--out", lock)
    assert r.returncode == 2 and r.stderr.startswith("refusing: partial ER control file: 1 exec_core test records")
    assert r.stderr.rstrip().endswith(f"e.g. {ids[6]}") and not lock.exists()
    er = [json.loads(l) for l in x["er"].read_text().splitlines()]
    for bad, what in ((er + er[:1], "1 duplicated (record_id, condition) rows"),
                      (er + [{**e, "record_id": ids[7]} for e in er],
                       "1 records in the ER control results are not exec_core")):
        er_bad = write_er(tmp_path / "er_bad.jsonl", bad)
        r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x, er=er_bad), "--out", lock)
        assert r.returncode == 2 and r.stderr.startswith("refusing: ") and what in r.stderr and not lock.exists()
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    forged = json.loads(lock.read_text())
    forged["inputs"]["records"] = sha(more)
    (tmp_path / "forged.json").write_text(json.dumps(forged))
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, {**x, "records": more}),
            "--lock", tmp_path / "forged.json", "--scores", x["scores"], "--out", tmp_path / "sap")
    assert r.returncode == 3 and "partial ER control file" in r.stderr and ids[6] in r.stderr
    assert not (tmp_path / "sap" / "SAP_RESULTS.json").exists()


def test_lock_and_run_leave_out_the_dev_rows_of_the_er_controls_run(tmp_path):
    """The ER controls run also holds dev records (write_overlays' default ids: test exec_core + dev Sylius api*);
    their rows are not in records/test.jsonl and are left out, so the lock and the run go ahead (their overlay_sha256
    is not checked either)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    er = [json.loads(l) for l in x["er"].read_text().splitlines()]
    assert DEV_ID not in ids
    mixed = write_er(tmp_path / "er_mixed.jsonl", [er[0], {**er[0], "record_id": DEV_ID, "overlay_sha256": "0" * 64},
                                                   er[1], {**er[1], "record_id": DEV_ID}])
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x, er=mixed), "--out", lock)
    assert r.returncode == 0, r.stderr
    assert json.loads(lock.read_text())["inputs"]["er_controls[0]"] == sha(mixed)
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x, er=mixed), "--lock", lock,
            "--scores", x["scores"], "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "sap" / "SAP_RESULTS.json").exists()


def test_lock_and_run_refuse_er_results_not_marked_complete(tmp_path):
    """The ER harness's er_results contract (0c42865): the ER control results are read only when the er_complete.json
    next to them says complete and not running; the lock refuses (exit 2), and so does the run (exit 3), aux_test
    included."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    status = x["er"].parent / "er_complete.json"
    done = json.loads(status.read_text())
    status.unlink()
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and r.stderr.startswith("refusing: ") and f"{status} is missing" in r.stderr
    assert not lock.exists()
    for st, what in (({"complete": False}, f"{status} complete is not true"),
                     ({"complete": False, "running": True}, f"{status} says the batch is still running")):
        status.write_text(json.dumps({**done, **st}))
        r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
        assert r.returncode == 2 and r.stderr.startswith("refusing: ") and what in r.stderr and not lock.exists()
    status.write_text(json.dumps(done))
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    status.write_text(json.dumps({**done, "complete": False}))  # the ER files are the locked ones: only this changed
    for aux in ((), ("--descriptive-only",)):
        r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
                *aux, "--out", tmp_path / "sap")
        assert r.returncode == 3 and r.stderr.startswith("refusing: ") and f"{status} complete is not true" in r.stderr
        assert not (tmp_path / "sap" / "SAP_RESULTS.json").exists()
    status.write_text(json.dumps(done))
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr


def test_run_refuses_score_flags_that_differ_from_the_locked_controls(tmp_path):
    """Under a valid lock, ca_defined False on the full_aware rows of a usable record (CA n 227 -> 226, exit 0 before),
    excl_bv True on the rows of a usable record (BV n 229 -> 228, exit 0 before), or no er on the anchor main rows of
    the ER-eligible record now stop the run (exit 3); aux_test (--descriptive-only, on its own locked controls)
    checks the two flags, not er."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    test_rows = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    assert ids[10] in {r["record_id"] for r in test_rows}
    assert all(r["er"] is not None for r in test_rows if (r["record_id"], r["model"]) == (ids[5], "deepseek"))
    changed = {"ca_defined": [{**r, "ca_defined": False} if r["record_id"] == ids[10] and r["condition"] == "full_aware"
                              else r for r in test_rows],
               "excl_bv": [{**r, "excl_bv": True} if r["record_id"] == ids[5] else r for r in test_rows],
               "er": [{**r, "er": None} if (r["record_id"], r["condition"], r["seed"]) == (ids[5], "full_aware", "s7")
                      else r for r in test_rows]}
    ax = [json.loads(l) for l in x["aux_scores"].read_text().splitlines()]
    a0 = ax[0]["record_id"]
    aux_changed = {"ca_defined": [{**r, "ca_defined": False} if r["record_id"] == a0 else r for r in ax],
                   "excl_bv": [{**r, "excl_bv": True} if r["record_id"] == a0 else r for r in ax],
                   "er": [{**r, "er": None} for r in ax]}
    for name, what, aux_exit in (("ca_defined", "score rows have a ca_defined that is not the one of their record's "
                                  "gold row in the locked controls.jsonl", 3),
                                 ("excl_bv", "score rows have an excl_bv that is not the one of the locked "
                                  "exclusions.jsonl", 3),
                                 ("er", "1 score rows of ER-eligible records (anchor model, main) have no er "
                                  "value", 0)):
        scores = write(tmp_path / f"scores_{name}.jsonl", changed[name])
        r = run(repo, "sap_run.py", *base, "--scores", scores, "--out", tmp_path / name)
        assert r.returncode == 3 and r.stderr.startswith("refusing: ") and what in r.stderr, (name, r.stderr)
        assert not (tmp_path / name / "SAP_RESULTS.json").exists()
        aux_scores = write(tmp_path / f"aux_{name}.jsonl", aux_changed[name])
        r = run(repo, "sap_run.py", *base, "--scores", aux_scores, "--descriptive-only",
                "--out", tmp_path / f"aux_{name}")
        assert r.returncode == aux_exit, (name, r.stderr)
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr


def test_run_refuses_rows_that_move_the_denominators(tmp_path):
    """Under a valid lock, each case of `probes` that once moved a denominator with exit 0, or crashed, now stops
    the run (exit 3): P2 (an extra full_aware seed s5 on 20 usable records; two-level n 114 -> 10, verdict flipped),
    P1b and P1c (the whole zero_shot run again under s42, changed or copied), P3 (the full_aware rows of the two
    CA-undefined records dropped; BV n 115 -> 114), P4b (rows of a record with no gold row; BV n 115 -> 116), P5b
    (the scenario_key of a scenario's full_aware rows; CA n 114 -> 113), P9 (ca None on rows that count in CA; exit
    1) and P10 (input rows carrying filled_missing; CA n 114 -> 113). Since 26/09/2026 the rows of P2, P1b and P1c are
    not ok lines of the manifest, which now stops them first. P6 (a missing row of the ER-eligible record, accepted by
    the committed accepted_missing.jsonl) still runs: the filled row scores CA = 0 and BV = 0 and has no ER value."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    usable = sorted(set(ids) - set(ids[:4]))
    some, k = set(usable[:20]), usable[5]
    sk = next(r["scenario_key"] for r in rs if r["record_id"] == k)
    ghost = [{**r, "record_id": "R|f999.feature|0|a", "scenario_key": "R|f999.feature|0",
              "feature_path": "f999.feature", "sub_post_cutoff_all": False, "sub_residual_clean": False,
              "sub_time_coherent_30d": False} for r in rs if r["record_id"] == usable[0]]
    seed = "score rows are not an ok line of the manifests"  # met before check_complete's seed rule
    probes = {"P2": (rs + [{**r, "seed": "s5"} for r in rs if r["condition"] == "full_aware" and r["seed"] == "s42"
                           and r["record_id"] in some], seed),
              "P1b": (rs + [{**r, "seed": "s42", "ca": 1 - r["ca"]} for r in rs if r["condition"] == "zero_shot"],
                      seed),
              "P1c": (rs + [{**r, "seed": "s42"} for r in rs if r["condition"] == "zero_shot"], seed),
              "P3": ([r for r in rs if not (r["record_id"] in ids[:2] and r["condition"] == "full_aware")],
                     "missing rows"),
              "P4b": (rs + ghost, "score rows are of records with no gold row in the locked controls.jsonl"),
              "P5b": ([{**r, "scenario_key": sk + "x"} if r["scenario_key"] == sk and r["condition"] == "full_aware"
                       else r for r in rs], "score rows have a scenario_key, repo or feature_path that is not"),
              "P9": ([{**r, "ca": None} if r["record_id"] == k and r["condition"] == "full_neutral" else r for r in rs],
                     "score rows have no ca value although ca_defined"),
              "P10": ([{**r, "ca_defined": False, "excl_bv": True, "filled_missing": True}
                       if r["scenario_key"] == sk and r["condition"] == "full_aware" else r for r in rs],
                      "score rows already carry filled_missing")}
    for name, (changed, what) in probes.items():
        scores = write(tmp_path / f"scores_{name}.jsonl", changed)
        r = run(repo, "sap_run.py", *base, "--scores", scores, "--out", tmp_path / name)
        assert r.returncode == 3 and r.stderr.startswith("refusing: ") and what in r.stderr, (name, r.stderr[-400:])
        assert not (tmp_path / name / "SAP_RESULTS.json").exists()
    er_row = next(r for r in rs if (r["record_id"], r["model"], r["condition"], r["seed"])
                  == (ids[5], "deepseek", "full_aware", "s7"))
    scores = write(tmp_path / "scores_P6.jsonl", [r for r in rs if r is not er_row])
    # accept()'s commit and the canonical SCORES_META commit share one --repo, so this scenario (the only
    # accepted_missing one here) gets its own fresh repo, its own canonical write, cleanly.
    p6_lock, p6_home, p6_anch = fresh_generation_repo(tmp_path, lock.read_bytes(), "p6")
    acc = accept(p6_home, x, [(ids[5], "deepseek", "full_aware", "s7")], cfg_repo=tmp_path)
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", p6_lock, *p6_anch, "--scores",
            scores, *acc, "--out", tmp_path / "P6")
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "P6" / "SAP_RESULTS.json").read_text())["completeness"]["missing_filled"] == 1


def test_lock_and_run_refuse_er_results_of_another_manifest(tmp_path):
    """The ER control results are those of the --er-manifest given (locked): the er_complete.json next to
    them has its sha256, and every ER row of a test record has the overlay_sha256 of the manifest line of its
    (record_id, condition); a pair the manifest does not list is refused too. The lock refuses (exit 2), and so does
    the run (exit 3). Rows of non-test records are not checked
    (test_lock_and_run_leave_out_the_dev_rows_of_the_er_controls_run)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    manifest = x["er_manifest"].read_bytes()
    x["er_manifest"].unlink()
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "missing inputs ['er_manifest']" in r.stderr and not lock.exists()
    x["er_manifest"].write_bytes(manifest)
    status = x["er"].parent / "er_complete.json"
    done = json.loads(status.read_text())
    assert done["manifest_sha256"] == sha(x["er_manifest"])
    other = f"{status} manifest_sha256 is not the sha256 of the ER manifest given here"
    status.write_text(json.dumps({**done, "manifest_sha256": "0" * 64}))
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and r.stderr.startswith("refusing: ") and other in r.stderr and not lock.exists()
    er = [json.loads(l) for l in x["er"].read_text().splitlines()]
    swapped = [er[0], {**er[1], "overlay_sha256": er[0]["overlay_sha256"]}]  # the stub row scored on gold's overlay
    stale = ("1 ER control rows have an overlay_sha256 that is not the sha256 of their manifest line, "
             f"e.g. ('{ids[5]}', 'stub')")
    unlisted = f"1 ER control rows are not in the ER manifest given here, e.g. ('{ids[5]}', 'stub-v9')"
    for bad, what in ((swapped, stale), ([*er, {**er[1], "condition": "stub-v9"}], unlisted)):
        f = write_er(tmp_path / "er_other.jsonl", bad)
        r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x, er=f), "--out", lock)
        assert r.returncode == 2 and r.stderr.startswith("refusing: ") and what in r.stderr and not lock.exists()
    write_er(x["er"], er)  # er_complete.json of the fixture's run again
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    status.write_text(json.dumps({**done, "manifest_sha256": "0" * 64}))  # checked, not locked
    for aux in ((), ("--descriptive-only",)):
        r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
                *aux, "--out", tmp_path / "sap")
        assert r.returncode == 3 and r.stderr.startswith("refusing: ") and other in r.stderr
    f = write_er(tmp_path / "er_other.jsonl", swapped)
    forged = json.loads(lock.read_text())
    forged["inputs"]["er_controls[0]"] = sha(f)
    (tmp_path / "forged.json").write_text(json.dumps(forged))
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x, er=f), "--lock", tmp_path / "forged.json",
            "--scores", x["scores"], "--out", tmp_path / "sap")
    assert r.returncode == 3 and r.stderr.startswith("refusing: ") and stale in r.stderr
    assert not (tmp_path / "sap" / "SAP_RESULTS.json").exists()


def test_lock_and_run_cover_future_methods(tmp_path):
    """3C's controls_<part>/future_methods.jsonl decides CA (through the expansion of helper calls), so it is
    locked like controls.jsonl, exclusions.jsonl and self_contained.jsonl: sap_lock refuses without it (exit 2), the
    lock holds its sha256, and sap_run refuses (exit 2) when it changed after the lock."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock, fm = tmp_path / "SAP_LOCK.json", tc / "future_methods.jsonl"
    before = fm.read_bytes()
    fm.unlink()
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "refusing: missing inputs ['future_methods']" in r.stderr and not lock.exists()
    fm.write_bytes(before)
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    assert json.loads(lock.read_text())["inputs"]["future_methods"] == sha(fm)
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"], "--out", tmp_path / "sap"]
    rows = [json.loads(l) for l in before.decode().splitlines()]
    write(fm, [{**o, "future": ["Ctx:otherHelper"]} if o["record_id"] == ids[6] else o for o in rows])
    r = run(repo, "sap_run.py", *base)
    assert r.returncode == 2 and "refusing: inputs differ from the lock: ['future_methods']" in r.stderr
    assert not (tmp_path / "sap" / "SAP_RESULTS.json").exists()
    fm.write_bytes(before)
    assert run(repo, "sap_run.py", *base).returncode == 0


def test_run_says_where_ca_prec_and_ca_conclude_differently(tmp_path):
    """D11 (c): CA_prec's comparisons are reported next to CA's and every difference of conclusion is stated, whatever
    its direction: with the full_aware outputs below τ (call_precision 0.1), full_aware's CA_prec is 0, so every
    comparison is contrary on CA_prec while CA supports it, and the report says so for each comparison and each RQ.
    A score row without call_precision stops the run (exit 3)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    low = write(tmp_path / "scores_low.jsonl", [{**r, "call_precision": 0.1} if r["condition"] == "full_aware" else r
                                                for r in rs])
    r = run(repo, "sap_run.py", *base, "--scores", low, "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text(encoding="utf-8"))
    report = (tmp_path / "sap" / "SAP_REPORT.md").read_text(encoding="utf-8")
    comps = lambda conf: [v for f in conf.values() for v in f["comparisons"].values()]
    assert {v["verdict"] for v in comps(res["confirmatory"])} == {"supported"}
    assert {v["verdict"] for v in comps(res["ca_prec"]["confirmatory"])} == {"contrary"}
    diff = res["ca_prec"]["differences"]
    assert len(diff) == 6 and all(f"- {d}\n" in report for d in diff)
    assert "- RQ2: kết luận mức RQ khác nhau giữa CA (full) và CA_prec (contrary).\n" in report
    assert table_shape_errors(report) == []
    # P22 (27/09/2026): the main table, the RQ lines and every answer
    # carry the robustness label, not only the CA_prec section
    main = report.split("## Kiểm định xác nhận")[1].split("\n## ")[0]
    assert "| Seed dương | Kết luận | Vững |" in main
    rows_ = [l for l in main.split("\n") if l.startswith("| RQ")]
    assert len(rows_) == 4 and all(l.endswith("| supported | chưa vững |") for l in rows_)
    assert "- RQ1: **full** (chưa vững)\n" in main and "- RQ2: **full** (chưa vững)\n" in main
    assert len(res["answers"]) == 6 and all(a.endswith(" Độ vững (CA và CA_prec): chưa vững.") for a in res["answers"])
    assert all(f"- {a}\n" in report for a in res["answers"])
    missing = write(tmp_path / "scores_missing.jsonl", [{k: v for k, v in r.items() if k != "call_precision"}
                                                        if r is rs[0] else r for r in rs])
    r = run(repo, "sap_run.py", *base, "--scores", missing, "--out", tmp_path / "sap2")
    assert r.returncode == 3 and "1 score rows lack call_precision or n_calls" in r.stderr


def test_report_marks_the_prestashop_rows_of_the_call_table():
    """D11 (design specification at 3739dd6, §2.2b): call_precision and n_calls per repo and condition, and the
    PrestaShop rows of that table carry the sentence that CA is easy to reach there through infrastructure calls
    (sap.REPO_NOTES)."""
    spec = importlib.util.spec_from_file_location("sap_run_script", ROOT / "scripts" / "sap_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = {"call_precision": {"n": 2, "median": 0.5, "q1": 0.25, "q3": 0.75},
             "n_calls": {"n": 2, "median": 3, "p90": 5.4}}
    lines = mod.calls_by_repo_lines({"PrestaShop/PrestaShop": {"deepseek|zero_shot|main": stats},
                                     "R": {"deepseek|zero_shot|main": stats}})
    assert lines[2] == ("| PrestaShop/PrestaShop | deepseek · zero_shot · main | 0.500 [0.250, 0.750] (2) | "
                        "3, p90 5.4 (2) | Ở PrestaShop, CA dễ đạt chỉ bằng các lời gọi hạ tầng (D11): đọc CA ở repo "
                        "này cùng call_precision và CA_prec. |")
    assert lines[3] == "| R | deepseek · zero_shot · main | 0.500 [0.250, 0.750] (2) | 3, p90 5.4 (2) |  |"
    assert table_shape_errors("\n".join(lines)) == []


def test_report_gives_the_lazy_references_and_the_prestashop_note_per_repo():
    """The per-repo CA and BV table carries the lazy references of each run (sap.lazy_by_repo:
    v3m, v3all, v3names, top4; "–" where the rows have none) and the PrestaShop sentence (sap.REPO_NOTES)."""
    spec = importlib.util.spec_from_file_location("sap_run_script", ROOT / "scripts" / "sap_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    level = {"estimate": 0.5, "ci95": [0.25, 0.75]}
    ps, run = "PrestaShop/PrestaShop", "deepseek|zero_shot|main"
    res = {"aux": False, "descriptive": {
        "by_repo": {r: {"ca": {run: level}, "bv": {run: level}} for r in (ps, "R")},
        "lazy_by_repo": {ps: {run: {"ca_ref_v3m": {"n": 4, "rate": 0.25}, "ca_ref_v3all": {"n": 4, "rate": 0.5},
                                    "ca_ref_v3names": {"n": 4, "rate": 0.75}, "ca_ref_top4": {"n": 4, "rate": 1.0}}},
                         "R": {run: {"ca_ref_v3m": {"n": 4, "rate": 0.0}}}}}}
    lines = mod.by_repo_lines(res)
    head = lines.index("| Repo | Lượt chạy | CA | BV | v3m | v3all | v3names | top4 | Ghi chú |")
    assert lines[head + 2] == ("| PrestaShop/PrestaShop | deepseek · zero_shot · main | +0.500 [+0.250, +0.750] | "
                               "+0.500 [+0.250, +0.750] | 0.250 | 0.500 | 0.750 | 1.000 | Ở PrestaShop, CA dễ đạt chỉ "
                               "bằng các lời gọi hạ tầng (D11): đọc CA ở repo này cùng call_precision và CA_prec. |")
    assert lines[head + 3] == ("| R | deepseek · zero_shot · main | +0.500 [+0.250, +0.750] | +0.500 [+0.250, +0.750] "
                               "| 0.000 | – | – | – |  |")
    assert table_shape_errors("\n".join(lines)) == []


def test_run_needs_the_lock_committed_and_published(tmp_path):
    """sap_run authenticates SAP_LOCK.json by --lock-commit: the file equals, byte for byte, git show <sha>:<its path in
    --repo>, the commit is an ancestor of HEAD and on a branch of origin (a push to a fork only does not do); otherwise
    exit 2. A lock whose digest was edited after the commit (to fit an edited config) is refused, committed or not; the
    commit's fuller dates are printed into SAP_RESULTS and SAP_REPORT."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"]]
    good = anchor(lock)
    r = run(repo, "sap_run.py", *base, *good, "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text())
    assert res["lock_commit"]["sha"] == good[-1] and "CommitDate:" in res["lock_commit"]["fuller"]
    report = (tmp_path / "sap" / "SAP_REPORT.md").read_text()
    assert f"`{good[-1]}`" in report and "CommitDate:" in report
    cfg_path, lock_bytes = repo / "config" / "sap.json", lock.read_bytes()
    cfg = cfg_path.read_bytes()
    cfg_path.write_text(json.dumps({**json.loads(cfg), "alpha": 0.2}))
    forged = json.loads(lock_bytes)
    forged["inputs"]["config"] = sha(cfg_path)
    lock.write_text(json.dumps(forged, indent=1) + "\n")
    r = run(repo, "sap_run.py", *base, *good, "--out", tmp_path / "f1")
    assert r.returncode == 2 and "is not, byte for byte, the one committed at" in r.stderr
    git(tmp_path, "commit", "-q", "-am", "forged lock")
    forged_sha = git(tmp_path, "rev-parse", "HEAD")
    r = run(repo, "sap_run.py", *base, "--repo", tmp_path, "--lock-commit", forged_sha, "--out", tmp_path / "f2")
    assert r.returncode == 2 and f"commit {forged_sha} is on no branch of origin" in r.stderr
    subprocess.run(["git", "init", "-q", "--bare", str(tmp_path / "fork.git")], check=True)  # a lock commit
    git(tmp_path, "remote", "add", "fork", tmp_path / "fork.git")                          # pushed to a fork only
    git(tmp_path, "push", "-q", "fork", "HEAD:refs/heads/main")
    git(tmp_path, "fetch", "-q", "fork")
    r = run(repo, "sap_run.py", *base, "--repo", tmp_path, "--lock-commit", forged_sha, "--out", tmp_path / "f2b")
    assert r.returncode == 2 and f"commit {forged_sha} is on no branch of origin" in r.stderr, r.stderr
    git(tmp_path, "push", "-q", "origin", "HEAD:refs/heads/side")
    git(tmp_path, "fetch", "-q", "origin")
    git(tmp_path, "reset", "-q", "--hard", good[-1])  # HEAD back on the true lock; the forged one only on side
    lock.write_text(json.dumps(forged, indent=1) + "\n")
    r = run(repo, "sap_run.py", *base, "--repo", tmp_path, "--lock-commit", forged_sha, "--out", tmp_path / "f3")
    assert r.returncode == 2 and f"commit {forged_sha} is not an ancestor of HEAD" in r.stderr
    r = run(repo, "sap_run.py", *base, "--repo", tmp_path, "--lock-commit", "0" * 40, "--out", tmp_path / "f4")
    assert r.returncode == 2 and "is not a commit of" in r.stderr
    lock.write_bytes(lock_bytes)
    cfg_path.write_bytes(cfg)
    assert run(repo, "sap_run.py", *base, *good, "--out", tmp_path / "ok").returncode == 0


def test_run_covers_every_model_of_the_design(tmp_path):
    """Completeness covers every model × condition × seed of the design (RQ3 and the descriptive tables read them): a
    missing Mistral row stops the run unless that pair is accepted as missing (an entry of another pair, e.g. the
    anchor's, does not do); a row of a run outside the design, or of a seed a model does not have, is no ok line of the
    manifest and stops it too, and so does such a line in the manifest."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    gone = next(r for r in rs if (r["record_id"], r["model"], r["condition"], r["seed"])
                == (ids[7], "mistral", "full_aware", "s23"))
    short = write(tmp_path / "short.jsonl", [r for r in rs if r is not gone])
    r = run(repo, "sap_run.py", *base, "--scores", short, "--out", tmp_path / "s1")
    assert r.returncode == 3
    assert f"1 missing rows, e.g. ('{ids[7]}', ('mistral', 'full_aware', 'main'), 's23')" in r.stderr
    mine = (ids[7], "mistral", "full_aware", "s23")
    anchors = [{**{k: v for k, v in mline(dict(zip(("record_id", "model", "condition", "seed"), p)), "missing").items()
                   if k in KEY}, "reason": "rerun failed", "receipt_sha256": "2" * 64}
               for p in [(ids[7], "deepseek", "full_aware", "s23")]]
    # each accepted_missing scenario below changes the template SCORES_META the harness auto-fills from
    # (accept() rewrites accepted_missing_sha256/commit into it), so each gets its own fresh repo -- accept()'s commit
    # and the canonical META commit share one --repo, and two DIFFERENT canonical writes in the same repo would break
    # the "once" rule the good scenario (s3) must pass cleanly.
    lock2, home2, anch2 = fresh_generation_repo(tmp_path, lock.read_bytes(), "s2")
    acc = accept(home2, x, [mine], entries=anchors, cfg_repo=tmp_path)  # the anchor's pair listed: not a missing line
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock2, *anch2, "--scores", short,
            *acc, "--out", tmp_path / "s2")
    assert r.returncode == 3 and "1 accepted_missing entries are not a missing line of the manifests" in r.stderr
    assert "1 missing manifest lines are not in accepted_missing" in r.stderr
    lock3, home3, anch3 = fresh_generation_repo(tmp_path, lock.read_bytes(), "s3")
    acc = accept(home3, x, [mine], cfg_repo=tmp_path)
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock3, *anch3, "--scores", short,
            *acc, "--out", tmp_path / "s3")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "s3" / "SAP_RESULTS.json").read_text())
    assert res["completeness"]["missing_filled"] == 1
    assert res["accepted_missing"]["filled"] == [[ids[7], "mistral", "full_aware", "s23"]]
    write_generation(tmp_path, x)
    codellama = [r for r in rs if (r["model"], r["seed"]) == ("codellama", "s42")]
    for extra in ([{**r, "variant": "k4"} for r in codellama], [{**r, "model": "gpt"} for r in codellama],
                  [{**r, "seed": "s7"} for r in codellama]):
        f = write(tmp_path / "extra.jsonl", rs + extra)
        r = run(repo, "sap_run.py", *base, "--scores", f, "--out", tmp_path / "s4")
        assert r.returncode == 3 and f"{len(codellama)} score rows are not an ok line of the manifests" in r.stderr
        write_generation(tmp_path, x, extra=[mline(e) for e in extra])  # the same pairs as manifest lines
        r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "s5")
        assert r.returncode == 3 and f"{len(codellama)} manifest lines outside the design" in r.stderr, r.stderr
        write_generation(tmp_path, x)


def test_run_refuses_record_fields_or_doses_that_differ_from_the_locked_inputs(tmp_path):
    """Language, suite, exec_core, partition and preexisting of a score row are its record's in the locked records (exit
    3 otherwise); with a locked 3C dose table (sap_lock --dose; fixed 25/09/2026) every row's dose is the table's for
    its (model, record), and the report no longer says it is unchecked."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    table = sorted({(r["model"], r["record_id"]): r["dose"] for r in rs}.items())
    dose = write(tmp_path / "dose.jsonl", [{"record_id": rid, "model": m, "dose": d} for (m, rid), d in table])
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--dose", dose, "--out", lock)
    assert r.returncode == 0, r.stderr
    assert json.loads(lock.read_text())["inputs"]["dose"] == sha(dose)
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--dose", dose, "--lock", lock]
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text())["dose_checked"] is True
    assert "dose chưa được đối chiếu" not in (tmp_path / "sap" / "SAP_REPORT.md").read_text()
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--out", tmp_path / "nodose")
    assert r.returncode == 2 and "inputs differ from the lock: ['dose']" in r.stderr
    dosed = "score rows have a dose that is not the one of the locked dose table for their (model, record)"
    rec = "score rows have a language, suite, exec_core, partition or preexisting that is not their record's"
    for name, change, what in (("dose", {"dose": 9}, dosed), ("language", {"language": "java"}, rec),
                               ("suite", {"suite": "z"}, rec), ("exec_core", {"exec_core": True}, rec),
                               ("partition", {"partition": "dev"}, rec), ("preexisting", {"preexisting": True}, rec)):
        f = write(tmp_path / f"scores_{name}.jsonl", [{**r, **change} if r["record_id"] == ids[9] else r for r in rs])
        r = run(repo, "sap_run.py", *base, "--scores", f, "--out", tmp_path / name)
        assert r.returncode == 3 and what in r.stderr, (name, r.stderr)


def test_run_refuses_score_rows_without_the_d11d_columns(tmp_path):
    """D11 (d) (design specification at 4d8bf2a), §1.4 and §2.3: a score row without ca_ref_top4, php_tag_stripped or
    the PHP step counts, or (§2.2) a reader count, or with a value outside their contract, stops the run (exit 3)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    what = ("score rows lack ca_ref_v3m, ca_ref_v3all, ca_ref_v3names, ca_ref_top4, php_tag_stripped, "
            "unread_step_annotations, cuke_escaped_patterns, cuke_custom_types, php_docblock_steps or php_attribute_steps")
    for name, edit in (("no_top4", lambda r: {k: v for k, v in r.items() if k != "ca_ref_top4"}),
                       ("no_tag", lambda r: {k: v for k, v in r.items() if k != "php_tag_stripped"}),
                       ("no_docblock", lambda r: {k: v for k, v in r.items() if k != "php_docblock_steps"}),
                       ("top4", lambda r: {**r, "ca_ref_top4": 2}), ("tag", lambda r: {**r, "php_tag_stripped": 1}),
                       ("steps", lambda r: {**r, "php_attribute_steps": -1}),
                       ("no_unread", lambda r: {k: v for k, v in r.items() if k != "unread_step_annotations"}),
                       ("custom", lambda r: {**r, "cuke_custom_types": 1.5})):
        f = write(tmp_path / f"scores_{name}.jsonl", [edit(r) if r["record_id"] == ids[9] else r for r in rs])
        r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", f,
                "--out", tmp_path / name)
        assert r.returncode == 3 and what in r.stderr, (name, r.stderr)


def test_a_rehearsal_run_can_never_be_read_as_results(tmp_path):
    """With --rehearsal (added 26/09/2026; a run on control outputs), SAP_REPORT.md opens with the bold banner, the same
    sentence heads every table of the confirmatory and CA_prec sections and the pre-written answers, SAP_RESULTS.json
    says "rehearsal": true, and the answers printed on stdout follow the banner."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--rehearsal", "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[0] == REHEARSAL
    assert json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text())["rehearsal"] is True
    report = (tmp_path / "sap" / "SAP_REPORT.md").read_text()
    assert report.split("\n")[0] == REHEARSAL
    sections = ("## Kiểm định xác nhận", "## Ước lượng từng seed", "## Thước đo mô tả cạnh mỗi phép so",
                "## Câu trả lời", "## Độ nhạy CA_prec")
    for head in sections:
        body = report.split(head)[1].split("\n## ")[0]
        blocks = body.split("\n\n")
        for i, block in enumerate(blocks):  # every table (and the answers) right after the banner
            if block.startswith("|") or block.startswith("- ") and head == "## Câu trả lời":
                assert blocks[i - 1].strip().split("\n")[-1] == REHEARSAL, (head, block[:60])
    assert report.count(REHEARSAL) == 1 + 3 + 1 + 2  # the top, three confirmatory tables, the answers, two CA_prec


def test_run_ties_the_outputs_to_the_manifest_and_the_accepted_missing_pairs(tmp_path):
    """Design specification §5 at 7af1673 (fixed 26/09/2026), exit 3 each: a SCORES_META that does not name the given
    manifest; a manifest with a design cell absent, or a line of a record of no locked partition; a coverage file whose
    two statuses fail (RECHECK_MISMATCH); a score row whose sha256 is not its manifest line's; and, for missing pairs,
    an accepted_missing.jsonl that is not pushed to origin, differs from its commit, or was committed after the scoring,
    or an accepted pair without evidence (REFUSED_* last), or whose evidence_eligible the attempts do not give."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    go = lambda name, *more, scores=x["scores"]: run(repo, "sap_run.py", *base, "--scores", scores, *more,
                                                     "--out", tmp_path / name)
    m = tmp_path / "manifest.main.jsonl"
    m.write_bytes(b"".join(sorted(m.read_bytes().splitlines(keepends=True))))  # the same lines, other bytes
    r = go("meta")
    assert r.returncode == 3 and "is not the sha256 of a given --manifest" in r.stderr, r.stderr
    cell = (ids[7], "codellama", "rag_only", "base")
    write_generation(tmp_path, x, drop=[cell])
    r = go("absent")
    assert r.returncode == 3 and f"1 design cells have no ok or missing manifest line" in r.stderr
    stray = mline({"record_id": "Q|q.feature|1|a", "model": "deepseek", "condition": "zero_shot", "seed": "base"})
    write_generation(tmp_path, x, extra=[stray])
    r = go("stray")
    assert r.returncode == 3 and "1 manifest lines of records of no locked partition" in r.stderr
    write_generation(tmp_path, x)
    cov = tmp_path / "coverage.main.json"
    cov.write_text(json.dumps({**json.loads(cov.read_text()), "status": "FAIL", "status_except_missing": "FAIL"}))
    r = go("recheck")
    assert r.returncode == 3 and "the generation is not complete (a RECHECK_MISMATCH fails both)" in r.stderr
    write_generation(tmp_path, x)
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    f = write(tmp_path / "sha.jsonl", [{**r, "decision_sha256": "0" * 64} if r is rs[0] else r for r in rs])
    r = go("sha", scores=f)
    assert r.returncode == 3 and "1 score rows lack decision_sha256 (64 lowercase hex) or gen_code_sha256" in r.stderr
    gone = (ids[9], "deepseek", "full_neutral", "s11")
    short = write(tmp_path / "short.jsonl", [r for r in rs if (r["record_id"], r["model"], r["condition"], r["seed"])
                                             != gone])
    # accept() rewrites accepted_missing_sha256/commit into the template SCORES_META the harness auto-
    # fills from, so each of the accepted_missing scenarios below (an evolving one for edited/recommitted, otherwise
    # one each) gets its own fresh repo -- accept()'s commit and the canonical META commit share one --repo, and the
    # "ok" scenario at the end must still see a clean, single write there.
    def am_go(name, lock_copy, anch, *more):
        return run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock_copy, *anch, "--scores",
                  short, *more, "--out", tmp_path / name)
    lock_u, home_u, anch_u = fresh_generation_repo(tmp_path, lock.read_bytes(), "unpushed")
    r = am_go("unpushed", lock_u, anch_u, *accept(home_u, x, [gone], push=False, cfg_repo=tmp_path))
    assert r.returncode == 3 and "is on no branch of origin" in r.stderr, r.stderr
    # the accepted_missing commit is dated in the future (the earlier lock-precedes-scoring check also lands
    # on "today" otherwise) so scored_utc can sit strictly after today (the lock's real commit date) and still be
    # "late" relative to this one, in isolation
    lock_l, home_l, anch_l = fresh_generation_repo(tmp_path, lock.read_bytes(), "late")
    r = am_go("late", lock_l, anch_l, *accept(home_l, x, [gone], scored="2030-01-01T00:00:00Z",
                                              commit_date="2050-01-01T00:00:00+00:00", cfg_repo=tmp_path))
    assert r.returncode == 3 and "is not after the commit date" in r.stderr
    lock_e, home_e, anch_e = fresh_generation_repo(tmp_path, lock.read_bytes(), "edited")
    acc = accept(home_e, x, [gone], cfg_repo=tmp_path)
    write(acc[1], [{**e, "reason": "rerun failed twice"} for e in load_rows(acc[1])])
    r = am_go("edited", lock_e, anch_e, *acc)
    assert r.returncode == 3 and "is not, byte for byte, the one committed at" in r.stderr
    again = publish(home_e, acc[1])  # the edit committed and pushed again (refused since 27/09/2026)
    write_generation(home_e, x, [gone], acc[1], again, cfg_repo=tmp_path)
    r = am_go("recommitted", lock_e, anch_e, "--accepted-missing", acc[1], "--accepted-missing-commit", again)
    assert r.returncode == 3 and f"is written on origin by {again}, {acc[3]}, not by --accepted-missing-commit" in (
        r.stderr), r.stderr
    lock_r, home_r, anch_r = fresh_generation_repo(tmp_path, lock.read_bytes(), "refused")
    r = am_go("refused", lock_r, anch_r, *accept(home_r, x, [gone], last="REFUSED_POLICY", cfg_repo=tmp_path))
    assert r.returncode == 3 and "1 accepted_missing entries have no evidence" in r.stderr
    assert "REFUSED_POLICY" in r.stderr
    lock_el, home_el, anch_el = fresh_generation_repo(tmp_path, lock.read_bytes(), "eligible")
    acc = accept(home_el, x, [gone], cfg_repo=tmp_path)
    m_el = home_el / "manifest.main.jsonl"
    lines = [json.loads(l) for l in m_el.read_text().splitlines()]
    trusted = [{**l, "evidence_eligible": False} if l["status"] == "missing" else l for l in lines]
    m_el.write_text("".join(json.dumps(l) + "\n" for l in trusted))
    r = am_go("eligible", lock_el, anch_el, *acc)
    assert r.returncode == 3 and "1 lines have an evidence_eligible that their attempts do not give" in r.stderr
    lock_ok, home_ok, anch_ok = fresh_generation_repo(tmp_path, lock.read_bytes(), "ok")
    acc = accept(home_ok, x, [gone], cfg_repo=tmp_path)
    r = am_go("ok", lock_ok, anch_ok, *acc)
    assert r.returncode == 0, r.stderr


def test_run_refuses_a_scores_file_edited_after_it_was_scored(tmp_path):
    """SCORES_META.json's own scores_sha256 names the exact bytes of the --scores file it was written
    for; a scores.jsonl edited after scoring (flipping ca on rows while the manifest and SCORES_META both
    still describe the file as it was before) is refused (exit 3), and so is a SCORES_META that lacks the
    field entirely (an older writer that never recorded it; its own repo: its content genuinely differs
    from the good meta's below, and the canonical path holds only one genuine write per repo). Passing
    --scores-meta explicitly (unlike every other test here) opts out of run()'s auto-patch, so the raw
    contract is exercised; the meta still has to sit at its canonical path (since 28/09/2026)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    meta = canonical_meta(tmp_path)
    meta.write_bytes((tmp_path / "SCORES_META.json").read_bytes())
    assert run(repo, "sap_run.py", *base, "--scores", x["scores"], "--scores-meta", meta,
               "--out", tmp_path / "before").returncode == 0
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    flipped = [{**r, "ca": 1 - r["ca"]} if i < 20 and r.get("ca") is not None else r for i, r in enumerate(rs)]
    x["scores"].write_text("".join(json.dumps(r) + "\n" for r in flipped))
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--scores-meta", meta, "--out", tmp_path / "tampered")
    assert r.returncode == 3 and "is not the sha256 of its --scores file" in r.stderr, r.stderr
    assert not (tmp_path / "tampered" / "SAP_RESULTS.json").exists()
    x["scores"].write_text("".join(json.dumps(r) + "\n" for r in rs))  # restore the scored bytes
    # a SCORES_META lacking scores_sha256 (an older writer): its own repo (its bytes genuinely differ
    # from `meta`'s, and the canonical path is anchored once per repo), with --manifest/--coverage given explicitly
    # since its --lock now sits in a different directory than the fixture's generation files
    stale = json.loads(meta.read_bytes())
    del stale["scores_sha256"]
    no_field_lock, no_field_repo, no_field_anchor = fresh_generation_repo(tmp_path, lock.read_bytes(), "no_field")
    no_field = canonical_meta(no_field_repo)
    no_field.write_text(json.dumps(stale))
    r = run(repo, "sap_run.py", *base, "--lock", no_field_lock, *no_field_anchor, "--manifest",
            tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json", "--scores", x["scores"],
            "--scores-meta", no_field, "--out", tmp_path / "nofield")
    assert r.returncode == 3 and "scores_sha256 None is not the sha256 of its --scores file" in r.stderr, r.stderr
    assert run(repo, "sap_run.py", *base, "--scores", x["scores"], "--scores-meta", meta,
               "--out", tmp_path / "after").returncode == 0


def test_run_refuses_an_exploit_of_a_scores_meta_anchored_in_a_new_folder(tmp_path):
    """Until 28/09/2026 the exploit against scores_meta_commit_errors alone -- (1) copy an edited scores.jsonl into a
    NEW folder, (2) write a new SCORES_META.json there naming the edited file's scores_sha256, (3) commit and push it
    once -- passed every check that function alone enforces (byte match, ancestor of HEAD, on a branch of origin,
    written once) since nothing there names a required path; on the real dataset it moved RQ1 from 0.2711 to 0.5482 at
    exit 0. Fix (a), canonical_meta_errors, refuses any META not at out/phase3/scores/<partition>/SCORES_META.json under
    --repo before scores_meta_commit_errors (or any content check) ever runs, so the copy-and-anchor-elsewhere trick no
    longer has anywhere valid to land."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "before")
    assert r.returncode == 0, r.stderr
    # the exploit: an edited copy of scores.jsonl (every full_aware CA forced to 1, moving RQ1 upward),
    # a brand-new folder, a new META there naming the edited copy's own scores_sha256, committed and pushed once --
    # exactly what committed_file_errors and scores_meta_commit_errors alone accept.
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    edited = [{**r, "ca": 1} if r["condition"] == "full_aware" and r.get("ca") is not None else r for r in rs]
    exploit_lock, exploit_repo, exploit_anch = fresh_generation_repo(tmp_path, lock.read_bytes(), "exploit")
    folder = exploit_repo / "attacker_folder"
    folder.mkdir()
    scores_copy = write(folder / "scores.jsonl", edited)
    template = json.loads((tmp_path / "SCORES_META.json").read_text())
    meta = folder / "SCORES_META.json"
    meta.write_text(json.dumps({**template, "scores_sha256": sha(scores_copy)}))
    meta_commit = publish(exploit_repo, meta, branch="scores-meta")
    r = run(repo, "sap_run.py", *base, "--lock", exploit_lock, *exploit_anch, "--manifest",
            tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json", "--scores", scores_copy,
            "--scores-meta", meta, "--scores-meta-commit", meta_commit, "--out", tmp_path / "exploit")
    assert r.returncode != 0, r.stdout  # was exit 0 before the fix
    assert "canonical path" in r.stderr and "copying the scores" in r.stderr, r.stderr
    assert not (tmp_path / "exploit" / "SAP_RESULTS.json").exists()
    # the same content, legitimately at the canonical path in its own repo, is read (the fix narrows the path, not
    # what a correctly-anchored META may say): the point is that this is the ONLY way in now
    legit = canonical_meta(exploit_repo)
    legit.write_text(meta.read_text())
    legit_commit = publish(exploit_repo, legit, date="2099-06-01T00:00:00+00:00", branch="scores-meta-2")  # after
    # SCORED (2099-01-01), which the META names as its scored_utc
    r2 = run(repo, "sap_run.py", *base, "--lock", exploit_lock, *exploit_anch, "--manifest",
             tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json", "--scores", scores_copy,
             "--scores-meta", legit, "--scores-meta-commit", legit_commit, "--out", tmp_path / "legit")
    assert r2.returncode == 0, r2.stderr
    rq1_before = json.loads((tmp_path / "before" / "SAP_RESULTS.json").read_text())["confirmatory"]["RQ1"][
        "comparisons"]["full_aware_vs_zero_shot"]["ca"]["estimate"]
    rq1_legit = json.loads((tmp_path / "legit" / "SAP_RESULTS.json").read_text())["confirmatory"]["RQ1"][
        "comparisons"]["full_aware_vs_zero_shot"]["ca"]["estimate"]
    assert rq1_legit != rq1_before  # the edit really does move the estimate: the canonical path is a real check,
    # not a coincidental no-op that happens to refuse everything


def edited_scores(x: dict) -> list[dict]:
    """The edit of the exploit tests: every full_aware CA forced to 1, which moves RQ1 upward."""
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    return [{**r, "ca": 1} if r["condition"] == "full_aware" and r.get("ca") is not None else r for r in rs]


def test_run_refuses_a_symlink_at_the_canonical_meta_path_or_in_place_of_its_folder(tmp_path):
    """Bypass (a), closed 28/09/2026: an edited scores.jsonl and a META naming it, committed and pushed once in a folder
    of their own (copied_meta), then reached from the canonical path by a symlink there, or by a symlink in place of the
    folder scores/test. canonical_meta_errors compared the RESOLVED paths and committed_file_errors took the relative
    path given to git from the resolved one, so both followed the link to copied_meta and the run passed at exit 0 on
    the edited table. Now any symlink from --repo down to the META is refused (sap.link_errors) and nothing is
    written."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    template = json.loads((tmp_path / "SCORES_META.json").read_text())
    for name in ("file", "folder"):
        a_lock, a_repo, a_anch = fresh_generation_repo(tmp_path, lock.read_bytes(), f"copied_meta_{name}")
        folder = a_repo / "copied_meta"
        folder.mkdir()
        scores_copy = write(folder / "scores.jsonl", edited_scores(x))
        meta = folder / "SCORES_META.json"
        meta.write_text(json.dumps({**template, "scores_sha256": sha(scores_copy)}))
        meta_commit = publish(a_repo, meta, date="2099-06-01T00:00:00+00:00", branch="scores-meta")  # after SCORED
        canon = a_repo / "out" / "phase3" / "scores" / "test" / "SCORES_META.json"
        if name == "file":
            canon.parent.mkdir(parents=True)
            canon.symlink_to(Path("../../../../copied_meta/SCORES_META.json"))
        else:
            canon.parent.parent.mkdir(parents=True)
            canon.parent.symlink_to(Path("../../../copied_meta"), target_is_directory=True)
        assert canon.read_bytes() == meta.read_bytes()
        out = tmp_path / f"exploit_{name}"
        r = run(repo, "sap_run.py", *base, "--lock", a_lock, *a_anch, "--manifest", tmp_path / "manifest.main.jsonl",
                "--coverage", tmp_path / "coverage.main.json", "--scores", scores_copy, "--scores-meta", canon,
                "--scores-meta-commit", meta_commit, "--out", out)
        assert r.returncode == 3, (name, r.returncode, r.stderr)  # once exit 0
        at = "out/phase3/scores/test/SCORES_META.json" if name == "file" else "out/phase3/scores/test"
        assert f": {at} is a symlink under --repo" in r.stderr, r.stderr
        assert not (out / "SAP_RESULTS.json").exists()


def test_run_refuses_a_repo_below_the_top_level_of_its_checkout(tmp_path):
    """Bypass (b), closed 28/09/2026: with --repo a subfolder, `git show <sha>:<rel>` read <rel> from the checkout's
    root while `git log -- <rel>` read it from the subfolder. The legitimate META sits at the root's canonical path,
    written once; the attack rewrites it for an edited scores copy and puts the same bytes at sub/'s canonical path, in
    one commit. From the top level the rewrite is a second write and is refused; from sub/ the bytes came from the root
    and the written-once history from sub/, so the edited table passed at exit 0. Now --repo must be the top level of
    its checkout (sap.top_level_errors), met first by the lock's check."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--manifest", tmp_path / "manifest.main.jsonl",
            "--coverage", tmp_path / "coverage.main.json"]
    template = json.loads((tmp_path / "SCORES_META.json").read_text())
    a_lock, a_repo, a_anch = fresh_generation_repo(tmp_path, lock.read_bytes(), "sub_probe")
    lock_commit = a_anch[a_anch.index("--lock-commit") + 1]
    root_meta = canonical_meta(a_repo)
    root_meta.write_text(json.dumps({**template, "scores_sha256": sha(x["scores"])}))
    legit = publish(a_repo, root_meta, date="2099-06-01T00:00:00+00:00", branch="scores-meta")
    scores_copy = write(tmp_path / "edited_scores.jsonl", edited_scores(x))
    root_meta.write_text(json.dumps({**template, "scores_sha256": sha(scores_copy)}))
    sub = a_repo / "sub"
    sub_meta = canonical_meta(sub)
    sub_meta.write_bytes(root_meta.read_bytes())
    attack = publish(a_repo, [root_meta, sub_meta], date="2099-07-01T00:00:00+00:00", branch="scores-meta")
    sub_lock = sub / "SAP_LOCK.json"  # git show <lock commit>:SAP_LOCK.json reads the root's lock: same bytes
    sub_lock.write_bytes(lock.read_bytes())
    top = run(repo, "sap_run.py", *base, "--lock", a_lock, "--repo", a_repo, "--lock-commit", lock_commit,
              "--scores", scores_copy, "--scores-meta", root_meta, "--scores-meta-commit", attack,
              "--out", tmp_path / "top")
    assert top.returncode == 3 and f"is written on origin by {attack}, {legit}, not by" in top.stderr, top.stderr
    below = run(repo, "sap_run.py", *base, "--lock", sub_lock, "--repo", sub, "--lock-commit", lock_commit,
                "--scores", scores_copy, "--scores-meta", sub_meta, "--scores-meta-commit", attack,
                "--out", tmp_path / "below")
    assert below.returncode == 2, (below.returncode, below.stderr)  # once exit 0
    assert f"--repo {sub} is not the top level of its git checkout (" in below.stderr, below.stderr
    assert not (tmp_path / "below" / "SAP_RESULTS.json").exists()


def test_run_refuses_a_scores_meta_commit_that_does_not_descend_from_the_lock(tmp_path):
    """The rule that the SCORES_META commit descends from the lock commit (sap.scores_meta_commit_errors; fixed
    28/09/2026) was once pinned by a unit test only; a mutant of sap_run.generation passing --scores-meta-commit in its
    place (the check then always true) survived the CLI tests. Two histories a run must refuse through sap_run itself
    (exit 3): (1) the META committed on a branch cut BEFORE the lock, dated after it and after the scoring, merged later
    (an ancestor of HEAD, written once, on origin); (2) the META committed first and the lock on top of it (the META an
    ancestor of the lock) though dated after it. sap_run hands the check the full sha the lock check verified:
    --lock-commit is given abbreviated here and the message names the full sha."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--manifest", tmp_path / "manifest.main.jsonl",
            "--coverage", tmp_path / "coverage.main.json"]
    meta_bytes = json.dumps({**json.loads((tmp_path / "SCORES_META.json").read_text()),
                             "scores_sha256": sha(x["scores"])}).encode()
    for case in ("side_branch", "meta_before_lock"):
        home = tmp_path / "repos" / case
        home.mkdir(parents=True)
        meta, lock_copy = canonical_meta(home), home / "SAP_LOCK.json"
        if case == "side_branch":
            (home / "README").write_text("base\n")
            publish(home, home / "README")
            git(home, "checkout", "-q", "-b", "side")
            meta.write_bytes(meta_bytes)
            meta_commit = publish(home, meta, date="2099-06-01T00:00:00+00:00", branch="side")
            git(home, "checkout", "-q", "main")
            lock_copy.write_bytes(lock.read_bytes())
            lock_commit = publish(home, lock_copy)
            git(home, "merge", "-q", "--no-ff", "-m", "merge side", "side")
            git(home, "push", "-q", "origin", "HEAD:main")
            git(home, "fetch", "-q", "origin")
        else:
            meta.write_bytes(meta_bytes)
            meta_commit = publish(home, meta, date="2099-06-01T00:00:00+00:00")
            lock_copy.write_bytes(lock.read_bytes())
            lock_commit = publish(home, lock_copy)
        r = run(repo, "sap_run.py", *base, "--lock", lock_copy, "--repo", home, "--lock-commit", lock_commit[:10],
                "--scores", x["scores"], "--scores-meta", meta, "--scores-meta-commit", meta_commit,
                "--out", tmp_path / case)
        assert r.returncode == 3, (case, r.returncode, r.stderr)
        assert (f"--lock-commit {lock_commit} is not an ancestor of --scores-meta-commit {meta_commit}: the lock "
                "must precede the scores meta") in r.stderr, (case, r.stderr)
        assert not (tmp_path / case / "SAP_RESULTS.json").exists()


def bare_run(repo: Path, *args):
    """sap_run with exactly `args`: none of run()'s auto-fill (a default --scores-meta stays the default)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(repo / "scripts" / "sap_run.py"), *map(str, args)], cwd=repo, env=env,
                          capture_output=True, text=True)


def anchored_repo(tmp_path: Path, lock: Path, x: dict, name: str) -> tuple[Path, Path, str, Path, str]:
    """A repository of its own holding the committed and pushed lock (L) and the honest SCORES_META at its canonical path
    (M, after SCORED), as a finished run leaves it: (repo, lock copy, L, META path, M)."""
    a_lock, a_repo, anch = fresh_generation_repo(tmp_path, lock.read_bytes(), name)
    meta = canonical_meta(a_repo)
    meta.write_text(json.dumps({**json.loads((tmp_path / "SCORES_META.json").read_text()), "scores_sha256": sha(x["scores"])}))
    return a_repo, a_lock, anch[3], meta, publish(a_repo, meta, date="2099-06-01T00:00:00+00:00")


def test_run_reads_the_anchored_files_from_their_commits(tmp_path):
    """Until 28/09/2026 the anchors were checked at the canonical file but READ from the path given, and `x/..` through
    a symlink `x` lexically is the canonical path while it reads the file next to x's target. dd1: --scores-meta
    <canonical folder>/x/../SCORES_META.json, an edited META there and an edited scores table; dd2: the default META
    of --scores <canonical folder>/x/../scores.jsonl; dd3: --lock <repo>/y/../SAP_LOCK.json, a forged lock there that
    hashes an edited config (alpha 0.2). All three passed at exit 0 with the registered commits (RQ1 0.2711 ->
    0.5482); now the path given must lead to the file checked, and the content is read from the commit."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    gen = ["--manifest", tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json"]
    a_repo, a_lock, L, meta, M = anchored_repo(tmp_path, lock, x, "dd")
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", a_lock, "--repo", a_repo, "--lock-commit", L, *gen]
    assert bare_run(repo, *base, "--scores", x["scores"], "--scores-meta", meta, "--scores-meta-commit", M,
                    "--out", tmp_path / "honest").returncode == 0
    evil = a_repo / "copied_meta"
    (evil / "inner").mkdir(parents=True)
    scores_edit = write(evil / "scores.jsonl", edited_scores(x))
    (evil / "SCORES_META.json").write_text(json.dumps({**json.loads(meta.read_text()), "scores_sha256": sha(scores_edit)}))
    (meta.parent / "x").symlink_to(evil / "inner", target_is_directory=True)
    dd1 = bare_run(repo, *base, "--scores", scores_edit, "--scores-meta", meta.parent / "x" / ".." / "SCORES_META.json",
                   "--scores-meta-commit", M, "--out", tmp_path / "dd1")
    dd2 = bare_run(repo, *base, "--scores", meta.parent / "x" / ".." / "scores.jsonl", "--scores-meta-commit", M,
                   "--out", tmp_path / "dd2")
    for name, r in (("dd1", dd1), ("dd2", dd2)):
        assert r.returncode == 3 and "does not lead to out/phase3/scores/test/SCORES_META.json" in r.stderr, (name, r.stderr)
        assert not (tmp_path / name / "SAP_RESULTS.json").exists()
    cfg = Path(base[base.index("--config") + 1])
    alpha = tmp_path / "cfg_alpha.json"
    alpha.write_text(json.dumps({**json.loads(cfg.read_text()), "alpha": 0.2}))
    forged = json.loads(lock.read_text())
    forged["inputs"]["config"] = sha(alpha)
    (a_repo / "evil" / "inner").mkdir(parents=True)
    (a_repo / "evil" / "SAP_LOCK.json").write_text(json.dumps(forged))
    (a_repo / "y").symlink_to(a_repo / "evil" / "inner", target_is_directory=True)
    swap = lambda args, flag, v: [v if i and args[i - 1] == flag else a for i, a in enumerate(args)]
    dd3 = bare_run(repo, *swap(swap(base, "--lock", a_repo / "y" / ".." / "SAP_LOCK.json"), "--config", alpha),
                   "--scores", x["scores"], "--scores-meta", meta, "--scores-meta-commit", M, "--out", tmp_path / "dd3")
    assert dd3.returncode == 2 and "does not lead to SAP_LOCK.json" in dd3.stderr, dd3.stderr
    real = bare_run(repo, *swap(base, "--config", alpha), "--scores", x["scores"], "--scores-meta", meta,
                    "--scores-meta-commit", M, "--out", tmp_path / "dd3_real")
    assert real.returncode == 2 and "inputs differ from the lock: ['config']" in real.stderr, real.stderr


def test_run_refuses_a_replaced_or_grafted_history(tmp_path):
    """Until 28/09/2026, with the registered commits unchanged, a local `git replace` of the META blob (rp1), of the
    lock blob after an edited config (rp3), or `git replace --graft` making a META cut before the lock its descendant
    (rp2) once passed at exit 0. Replace objects are now ignored and a --repo holding replace refs
    is refused (exit 2, at the lock, the first anchor read)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    gen = ["--manifest", tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json"]
    rel = "out/phase3/scores/test/SCORES_META.json"
    # rp1: the META blob replaced
    a_repo, a_lock, L, meta, M = anchored_repo(tmp_path, lock, x, "rp1")
    scores_edit = write(tmp_path / "rp1_scores.jsonl", edited_scores(x))
    orig = git(a_repo, "rev-parse", f"{M}:{rel}")
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "scores_sha256": sha(scores_edit)}))
    git(a_repo, "replace", orig, git(a_repo, "hash-object", "-w", meta))
    r = bare_run(repo, *lock_args(repo, tmp_path, tc, dc, x), "--lock", a_lock, "--repo", a_repo, "--lock-commit", L, *gen,
                 "--scores", scores_edit, "--scores-meta", meta, "--scores-meta-commit", M, "--out", tmp_path / "rp1")
    assert r.returncode == 2 and "--repo has replace refs" in r.stderr, r.stderr
    # rp3: the lock blob replaced after an edited config
    a_repo, a_lock, L, meta, M = anchored_repo(tmp_path, lock, x, "rp3")
    base_args = lock_args(repo, tmp_path, tc, dc, x)
    cfg = Path(base_args[base_args.index("--config") + 1])
    alpha = tmp_path / "cfg_alpha3.json"
    alpha.write_text(json.dumps({**json.loads(cfg.read_text()), "alpha": 0.2}))
    forged = json.loads(lock.read_text())
    forged["inputs"]["config"] = sha(alpha)
    orig = git(a_repo, "rev-parse", f"{L}:SAP_LOCK.json")
    a_lock.write_text(json.dumps(forged))
    git(a_repo, "replace", orig, git(a_repo, "hash-object", "-w", a_lock))
    r = bare_run(repo, *[alpha if i and base_args[i - 1] == "--config" else a for i, a in enumerate(base_args)],
                 "--lock", a_lock, "--repo", a_repo, "--lock-commit", L, *gen, "--scores", x["scores"], "--scores-meta", meta,
                 "--scores-meta-commit", M, "--out", tmp_path / "rp3")
    assert r.returncode == 2 and "--repo has replace refs" in r.stderr, r.stderr
    # rp2: the META on a branch cut before the lock, merged later; refused (not a descendant) until grafted
    home = tmp_path / "repos" / "rp2"
    home.mkdir(parents=True)
    (home / "README").write_text("base\n")
    publish(home, home / "README")
    git(home, "checkout", "-q", "-b", "off")
    m2 = canonical_meta(home)
    m2.write_text(json.dumps({**json.loads((tmp_path / "SCORES_META.json").read_text()), "scores_sha256": sha(x["scores"])}))
    M2 = publish(home, m2, date="2099-06-01T00:00:00+00:00", branch="off")
    git(home, "checkout", "-q", "main")
    lock2 = home / "SAP_LOCK.json"
    lock2.write_bytes(lock.read_bytes())
    L2 = publish(home, lock2)
    git(home, "merge", "-q", "--no-ff", "-m", "merge off", "off")
    git(home, "push", "-q", "origin", "HEAD:main")
    git(home, "fetch", "-q", "origin")
    args2 = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock2, "--repo", home, "--lock-commit", L2, *gen,
             "--scores", x["scores"], "--scores-meta", m2, "--scores-meta-commit", M2]
    r = bare_run(repo, *args2, "--out", tmp_path / "rp2_0")
    assert r.returncode == 3 and "is not an ancestor of --scores-meta-commit" in r.stderr, r.stderr
    git(home, "replace", "--graft", M2, L2)
    r = bare_run(repo, *args2, "--out", tmp_path / "rp2")
    assert r.returncode == 2 and "--repo has replace refs" in r.stderr, r.stderr


def test_a_fresh_clone_catches_a_forged_remote_tracking_ref(tmp_path):
    """`git update-ref refs/remotes/origin/main <an unpushed commit>` makes a local tampered META
    look published: the checks read the local remote-tracking refs, so the run in that checkout
    passes (the documented limit: whoever controls the local checkout). The last, mandatory step
    of the analysis (added 28/09/2026) clones origin afresh and runs sap_run there: the forged
    commit is not in origin, and the run is refused; the honest commits pass there too."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    gen = ["--manifest", tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json"]
    a_repo, a_lock, L, meta, M = anchored_repo(tmp_path, lock, x, "lr")
    scores_edit = write(tmp_path / "lr_scores.jsonl", edited_scores(x))
    git(a_repo, "checkout", "-q", "-b", "evil", L)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({**json.loads((tmp_path / "SCORES_META.json").read_text()), "scores_sha256": sha(scores_edit)}))
    git(a_repo, "add", "-f", meta.relative_to(a_repo))
    git(a_repo, "commit", "-q", "-m", "tampered META, never pushed", env={"GIT_AUTHOR_DATE": "2099-06-02T00:00:00+00:00",
                                                                         "GIT_COMMITTER_DATE": "2099-06-02T00:00:00+00:00"})
    E = git(a_repo, "rev-parse", "HEAD")
    git(a_repo, "update-ref", "refs/remotes/origin/main", E)
    local = bare_run(repo, *lock_args(repo, tmp_path, tc, dc, x), "--lock", a_lock, "--repo", a_repo, "--lock-commit", L,
                     *gen, "--scores", scores_edit, "--scores-meta", meta, "--scores-meta-commit", E, "--out", tmp_path / "lr")
    assert local.returncode == 0, local.stderr  # the limit: local refs are the checkout holder's
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", str(a_repo / "remote.git"), str(fresh)], check=True)
    step = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", fresh / "SAP_LOCK.json", "--repo", fresh, "--lock-commit", L,
            *gen, "--scores-meta", canonical_meta(fresh)]
    forged = bare_run(repo, *step, "--scores", scores_edit, "--scores-meta-commit", E, "--out", tmp_path / "lr_fresh")
    assert forged.returncode == 3, forged.stderr
    assert f"--scores-meta-commit {E} holds no out/phase3/scores/test/SCORES_META.json" in forged.stderr, forged.stderr
    honest = bare_run(repo, *step, "--scores", x["scores"], "--scores-meta-commit", M, "--out", tmp_path / "lr_honest")
    assert honest.returncode == 0, honest.stderr


def test_step_8_verifies_the_fresh_clone_run_against_the_registry(tmp_path):
    """The last step (added 28/09/2026): sap_run runs again from a fresh clone of origin, and sap_verify.py, from that
    clone, checks the SAP results row ("§6. Neo kết quả SAP") of PREREG_LEDGER.md on origin's v2-phase3-spec (six
    columns, the commit date of META_COMMIT in +07 included) and compares the run's SAP_RESULTS.json with the working
    checkout's (step6 here) in every field but RESULTS_IGNORED. A ledger date one second off, or any field outside
    RESULTS_IGNORED changed, stops it (exit 3); a field inside it changed does not."""
    import datetime as dt
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    gen = ["--manifest", tmp_path / "manifest.main.jsonl", "--coverage", tmp_path / "coverage.main.json"]
    a_repo, a_lock, L, meta, M = anchored_repo(tmp_path, lock, x, "step8")
    common = [*lock_args(repo, tmp_path, tc, dc, x), *gen, "--lock-commit", L, "--scores", x["scores"],
              "--scores-meta-commit", M]
    step6 = tmp_path / "step6"
    assert bare_run(repo, *common, "--lock", a_lock, "--repo", a_repo, "--scores-meta", meta, "--out", step6).returncode == 0
    when = dt.datetime.fromisoformat(git(a_repo, "show", "-s", "--format=%cI", M))
    row = ["test", L, sha(a_lock), M, sha(meta), json.loads(meta.read_text())["scores_sha256"],
           when.astimezone(dt.timezone(dt.timedelta(hours=7))).isoformat()]

    def ledger(cells: list[str]) -> None:  # the ledger's SAP results row on origin's v2-phase3-spec
        spec = tmp_path / "spec"
        if not spec.exists():
            subprocess.run(["git", "clone", "-q", str(a_repo / "remote.git"), str(spec)], check=True)
            git(spec, "checkout", "-q", "--orphan", "v2-phase3-spec")
            git(spec, "rm", "-rqf", ".")
        (spec / "PREREG_LEDGER.md").write_text(
            "# PREREG_LEDGER\n\n## §6. Neo kết quả SAP\n\n| phần | LOCK_COMMIT (40 ký tự) | sha256(SAP_LOCK.json) | "
            "META_COMMIT | sha256(SCORES_META.json) | scores_sha256 | ngày commit của META_COMMIT (git %cI, quy về +07) |\n"
            "|---|---|---|---|---|---|---|\n| " + " | ".join(cells) + " |\n")
        git(spec, "add", "PREREG_LEDGER.md")
        git(spec, "commit", "-q", "-m", "§6")
        git(spec, "push", "-q", "origin", "HEAD:refs/heads/v2-phase3-spec")
    ledger(row)
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", str(a_repo / "remote.git"), str(fresh)], check=True)
    check = tmp_path / "check"
    assert bare_run(repo, *common, "--lock", fresh / "SAP_LOCK.json", "--repo", fresh,
                    "--scores-meta", canonical_meta(fresh), "--out", check).returncode == 0
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    verify = lambda step: subprocess.run([sys.executable, str(repo / "scripts" / "sap_verify.py"), "--clone", str(fresh),
                                          "--part", "test", "--results", str(check / "SAP_RESULTS.json"),
                                          "--step-results", str(step)], cwd=repo, env=env, capture_output=True, text=True)
    ok = verify(step6 / "SAP_RESULTS.json")
    assert ok.returncode == 0 and "test: khớp sổ §6 và khớp SAP_RESULTS của bước trước" in ok.stdout, ok.stderr
    ref = json.loads((step6 / "SAP_RESULTS.json").read_text())
    for name, edit, want in (("tool_commit", lambda r: r.update(tool_commit="0" * 40), 0),
                             ("n_rows", lambda r: r.update(n_rows=r["n_rows"] + 1), 3),
                             ("confirmatory", lambda r: r.update(confirmatory={}), 3)):
        edited = json.loads(json.dumps(ref))
        edit(edited)
        f = tmp_path / f"step6_{name}.json"
        f.write_text(json.dumps(edited))
        r = verify(f)
        assert r.returncode == want, (name, r.stderr)
        assert want == 0 or f"differs from the earlier step at {name}" in r.stderr, (name, r.stderr)
    ledger(row[:6] + [(when + dt.timedelta(seconds=1)).astimezone(dt.timezone(dt.timedelta(hours=7))).isoformat()])
    git(fresh, "fetch", "-q", "origin")
    late = verify(step6 / "SAP_RESULTS.json")
    assert late.returncode == 3 and "ngày commit của META_COMMIT (+07)" in late.stderr, late.stderr


def test_run_refuses_a_scores_meta_not_committed_after_the_scoring(tmp_path):
    """Every SCORES_META.json is committed once (fixed 27/09/2026), right after the scoring, and pushed to origin;
    sap_run authenticates them by --scores-meta-commit (sap.scores_meta_commit_errors, exit 3). The tampered case, ca =
    1 on every DeepSeek full_aware row with scores_sha256 recomputed in the META (it moved RQ1 from +0.27 to +0.55 and
    passed), is refused: the META is no longer the committed one; so is the same edit committed and pushed again (the
    META's path is then written on origin by two commits), a META commit not pushed to origin and one not after the
    scoring. The commit and its fuller dates go into SAP_RESULTS and SAP_REPORT. Since 28/09/2026 the META also has to
    sit at its canonical path; m_ok/probe/again are one evolving scenario (the same path, re-edited and recommitted, so
    they share one repo on purpose) but m_unpushed, m_early and m_last are unrelated cases that would otherwise collide
    on that one canonical path, so each gets its own fresh repo (fresh_generation_repo)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    lock_bytes = lock.read_bytes()
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--manifest", tmp_path / "manifest.main.jsonl", "--coverage",
            tmp_path / "coverage.main.json", "--scores", x["scores"]]
    meta_bytes = (tmp_path / "SCORES_META.json").read_bytes()
    go = lambda name, lock_copy, anch, meta, commit: run(repo, "sap_run.py", *base, "--lock", lock_copy, *anch,
                                                         "--scores-meta", meta, "--scores-meta-commit", commit,
                                                         "--out", tmp_path / name)
    ok_lock, ok_repo, ok_anchor = fresh_generation_repo(tmp_path, lock_bytes, "ok")
    meta = canonical_meta(ok_repo)
    meta.write_bytes(meta_bytes)
    good = publish_metas(ok_repo, [meta])
    r = go("ok", ok_lock, ok_anchor, meta, good)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "ok" / "SAP_RESULTS.json").read_text())
    smc = res["generation"]["scores_meta_commit"]
    assert smc["sha"] == good and smc["paths"] == ["out/phase3/scores/test/SCORES_META.json"] and (
        "CommitDate:" in smc["fuller"])
    report = (tmp_path / "ok" / "SAP_REPORT.md").read_text()
    assert f"`{good}`" in report and smc["fuller"].rstrip() in report
    scores = x["scores"].read_bytes()
    rs = [json.loads(l) for l in scores.decode().splitlines()]
    probe = [{**r, "ca": 1} if (r["model"], r["condition"]) == ("deepseek", "full_aware") and r["ca"] is not None
             else r for r in rs]
    x["scores"].write_text("".join(json.dumps(r) + "\n" for r in probe))
    meta.write_text(json.dumps({**json.loads(meta_bytes), "scores_sha256": sha(x["scores"])}))
    r = go("probe", ok_lock, ok_anchor, meta, good)
    assert r.returncode == 3 and f"is not, byte for byte, the one committed at {good}" in r.stderr, r.stderr
    assert not (tmp_path / "probe" / "SAP_RESULTS.json").exists()
    again = publish(ok_repo, meta, date="2099-01-01T02:00:00+00:00", branch="scores-meta")
    r = go("again", ok_lock, ok_anchor, meta, again)
    assert r.returncode == 3 and (f"out/phase3/scores/test/SCORES_META.json is written on origin by {again}, {good}, "
                                  f"not by --scores-meta-commit {again} alone") in r.stderr, r.stderr
    x["scores"].write_bytes(scores)
    unpushed_lock, unpushed_repo, unpushed_anchor = fresh_generation_repo(tmp_path, lock_bytes, "unpushed")
    m = canonical_meta(unpushed_repo)
    m.write_bytes(meta_bytes)
    unpushed = publish(unpushed_repo, m, push=False, date="2099-01-01T03:00:00+00:00", branch="scores-meta")
    r = go("unpushed", unpushed_lock, unpushed_anchor, m, unpushed)
    assert r.returncode == 3 and f"commit {unpushed} is on no branch of origin" in r.stderr, r.stderr
    early_lock, early_repo, early_anchor = fresh_generation_repo(tmp_path, lock_bytes, "early")
    m = canonical_meta(early_repo)
    m.write_bytes(meta_bytes)
    r = go("early", early_lock, early_anchor, m, publish(early_repo, m, date=SCORED.replace("Z", "+00:00"),
                                                         branch="scores-meta"))
    assert r.returncode == 3 and f"is not after its scored_utc {SCORED}" in r.stderr, r.stderr
    last_lock, last_repo, last_anchor = fresh_generation_repo(tmp_path, lock_bytes, "last")
    m = canonical_meta(last_repo)
    m.write_bytes(meta_bytes)
    assert go("last", last_lock, last_anchor, m, publish_metas(last_repo, [m])).returncode == 0

def test_descriptive_only_checks_the_completeness_of_the_aux_design(tmp_path):
    """--descriptive-only goes through check_complete over the aux design (since 27/09/2026;
    sap.aux_design: the anchor model's 17 runs, each over the aux_test records): a duplicated
    aux row with its SCORES_META agreeing (once n 10 -> 11 and exit 0) or a missing aux row
    stops the run (exit 3); the completeness of the aux run is recorded and reported."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--descriptive-only"]
    r = run(repo, "sap_run.py", *base, "--scores", x["aux_scores"], "--out", tmp_path / "aux")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "aux" / "SAP_RESULTS.json").read_text())
    n_aux = len({json.loads(l)["record_id"] for l in x["aux_scores"].read_text().splitlines()})
    assert res["completeness"]["records_covered"] == {"deepseek": n_aux} and res["completeness"]["missing_filled"] == 0
    assert "của thiết kế aux_test (chỉ mô hình neo)" in (tmp_path / "aux" / "SAP_REPORT.md").read_text()
    ax = x["aux_scores"].read_text().splitlines()
    dup = tmp_path / "aux_dup.jsonl"
    dup.write_text("\n".join(ax + [ax[0]]) + "\n")
    r = run(repo, "sap_run.py", *base, "--scores", dup, "--out", tmp_path / "dup")
    assert r.returncode == 3 and "1 duplicated (record, run, seed) rows" in r.stderr, r.stderr
    short = tmp_path / "aux_short.jsonl"
    short.write_text("\n".join(ax[1:]) + "\n")
    r = run(repo, "sap_run.py", *base, "--scores", short, "--out", tmp_path / "short")
    assert r.returncode == 3 and "1 missing rows" in r.stderr, r.stderr

def test_run_refuses_a_lock_committed_after_the_scoring(tmp_path):
    """The lock's own commit date (today, since the fixture commits it right now) must be
    strictly before every SCORES_META's scored_utc; a scored_utc that is not after it is refused (exit 2, before any
    of the generation or score-row checks run), distinct from exit 3's checks that a scored_utc is after some OTHER
    commit (accepted_missing's)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    write_generation(tmp_path, x, scored="2000-01-01T00:00:00Z")  # before today, the lock's real commit date
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "before_lock")
    assert r.returncode == 2 and r.stderr.startswith("refusing: ") and (
        "scored_utc 2000-01-01T00:00:00Z is not after the lock's commit date" in r.stderr), r.stderr
    assert not (tmp_path / "before_lock" / "SAP_RESULTS.json").exists()
    write_generation(tmp_path, x)  # SCORED (2099-...): back to after today
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "after_lock")
    assert r.returncode == 0, r.stderr


def test_ca_prec_is_read_over_the_ca_denominator(tmp_path):
    """With a mechanical exclusion whose CA is defined (gold fails BV) and a scenario that compose drops for the anchor
    in one format (F1) only, CA_prec's comparisons keep CA's n (sap_run refuses any other:
    sap.ca_prec_denominator_errors), so a CA_prec analysis without the exclusions or the dropped records cannot pass."""
    def edit(repo, tc, dc, ids, x):
        excl = [json.loads(l) for l in (tc / "exclusions.jsonl").read_text().splitlines()]
        write(tc / "exclusions.jsonl", excl + [{"record_id": i, "reason": "gold_fails_bv", "excl_bv": True,
                                                "excl_ca": True} for i in ids[8:10]])
        compose = [json.loads(l) for l in x["compose_excluded"].read_text().splitlines()]
        write(x["compose_excluded"], compose + [{"record_id": i, "model": "deepseek", "fmt": "F1",
                                                 "reason": "control_token"} for i in ids[24:26]])
        rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
        write(x["scores"], [{**r, "excl_bv": r["record_id"] in ids[8:10] or r["excl_bv"],
                             "excl_ca": r["record_id"] in ids[8:10] or r["excl_ca"]} for r in rs
                            if not (r["record_id"] in ids[24:26] and r["model"] == "deepseek"
                                    and FMT[r["condition"]] == "F1")])
        write_generation(x["records"].parent, x)
    repo, tc, dc, ids, x = prepare(tmp_path, edit)
    lock = tmp_path / "SAP_LOCK.json"
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 0, r.stderr
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--scores", x["scores"],
            "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text())
    for fam, f in res["confirmatory"].items():
        for name, v in f["comparisons"].items():
            p = res["ca_prec"]["confirmatory"][fam]["comparisons"][name]["ca_prec"]
            assert (p["n_scenarios"], p["n_clusters"]) == (v["ca"]["n_scenarios"], v["ca"]["n_clusters"])
    scen = lambda rs: len({i.rsplit("|", 1)[0] for i in rs})
    n = res["confirmatory"]["RQ1"]["comparisons"]["full_aware_vs_zero_shot"]["ca"]["n_scenarios"]
    assert n == scen(set(ids) - set(ids[:3]) - set(ids[8:10]) - set(ids[24:26]))


def test_descriptive_only_checks_every_locked_input_and_reads_the_aux_inputs(tmp_path):
    """sap_lock needs the aux_test controls and records (locked); --descriptive-only (aux_test) compares every locked
    input with the lock, and reads its controls, exclusions and records from the locked aux inputs; only completeness is
    relaxed."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    aux_records = x["aux_records"].read_bytes()
    x["aux_records"].unlink()
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "missing inputs ['aux_records']" in r.stderr and not lock.exists()
    x["aux_records"].write_bytes(aux_records)
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--descriptive-only"]
    r = run(repo, "sap_run.py", *base, "--scores", x["aux_scores"], "--out", tmp_path / "aux")
    assert r.returncode == 0, r.stderr
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "t")  # test rows: not aux records
    assert r.returncode == 3 and "score rows are of records with no gold row in the locked controls.jsonl" in r.stderr
    for path, key in ((tc / "exclusions.jsonl", "exclusions"), (x["aux_controls"] / "controls.jsonl", "aux_controls"),
                      (x["aux_records"], "aux_records"), (x["records"], "records"), (x["subsets"], "subsets")):
        before = path.read_bytes()
        path.write_bytes(other_bytes(before))
        r = run(repo, "sap_run.py", *base, "--scores", x["aux_scores"], "--out", tmp_path / key)
        assert r.returncode == 2 and f"inputs differ from the lock: ['{key}']" in r.stderr, (key, r.stderr)
        path.write_bytes(before)
    ax = [json.loads(l) for l in x["aux_scores"].read_text().splitlines()]
    f = write(tmp_path / "aux_lang.jsonl", [{**r, "language": "java"} if r is ax[0] else r for r in ax])
    r = run(repo, "sap_run.py", *base, "--scores", f, "--out", tmp_path / "lang")
    assert r.returncode == 3 and "that is not their record's in the locked records" in r.stderr


def test_lock_and_run_check_the_pre_t_flags_against_the_config(tmp_path):
    """The frozen pre-T list (30706f2) is a locked input whose sha256 config/sap.json names
    (pre_t_flags): sap_lock refuses another file (exit 2); sap_run refuses a file that differs from the lock, and one
    that differs from the config even under a lock forged to cover it (exit 2)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock, tsv = tmp_path / "SAP_LOCK.json", x["pre_t_flags"]
    before = tsv.read_bytes()
    tsv.write_bytes(before + f"test\t{ids[30]}\n".encode())
    r = run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock)
    assert r.returncode == 2 and "is not the sha256 config/sap.json names for the pre-T flags" in r.stderr
    assert not lock.exists()
    tsv.write_bytes(before)
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--scores", x["scores"]]
    tsv.write_bytes(before + f"test\t{ids[30]}\n".encode())
    r = run(repo, "sap_run.py", *base, "--lock", lock, "--out", tmp_path / "s1")
    assert r.returncode == 2 and "inputs differ from the lock: ['pre_t_flags']" in r.stderr
    forged = json.loads(lock.read_text())
    forged["inputs"]["pre_t_flags"] = sha(tsv)
    (tmp_path / "forged.json").write_text(json.dumps(forged))
    r = run(repo, "sap_run.py", *base, "--lock", tmp_path / "forged.json", "--out", tmp_path / "s2")
    assert r.returncode == 2 and "is not the sha256 config/sap.json names for the pre-T flags" in r.stderr
    tsv.write_bytes(before)
    assert run(repo, "sap_run.py", *base, "--lock", lock, "--out", tmp_path / "s3").returncode == 0


def test_run_refuses_self_contained_that_differs_from_the_locked_file(tmp_path):
    """self_contained.jsonl is in the lock, so a file changed after the lock stops the run (exit 2); subset (iv) is
    read from that file, so a row whose self_contained is not its record's flag in it stops the run (exit 3),
    aux_test included (on its own rows; it compares every locked input too)."""
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    assert json.loads(lock.read_text())["inputs"]["self_contained"] == sha(x["self_contained"])
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock]
    sc = {o["record_id"]: o["self_contained"] for o in map(json.loads, x["self_contained"].read_text().splitlines())}
    k = next(i for i in ids[3:] if not sc[i])  # a usable record outside subset (iv)
    rs = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    flipped = write(tmp_path / "flipped_sc.jsonl",
                    [{**r, "self_contained": True} if r["record_id"] == k else r for r in rs])
    ax = [json.loads(l) for l in x["aux_scores"].read_text().splitlines()]
    aux_flipped = write(tmp_path / "aux_flipped_sc.jsonl",
                        [{**r, "self_contained": False} if r["record_id"] == ax[0]["record_id"] else r for r in ax])
    differs = "rows have a self_contained that differs from self_contained.jsonl"
    for scores, aux in ((flipped, ()), (aux_flipped, ("--descriptive-only",))):
        r = run(repo, "sap_run.py", *base, "--scores", scores, *aux, "--out", tmp_path / "sap")
        assert r.returncode == 3 and r.stderr.startswith("refusing: ") and differs in r.stderr, r.stderr
        assert not (tmp_path / "sap" / "SAP_RESULTS.json").exists()
    before = x["self_contained"].read_bytes()
    write(x["self_contained"], [{"record_id": i, "self_contained": sc[i] or i == k} for i in sc])
    for scores, aux in ((x["scores"], ()), (x["aux_scores"], ("--descriptive-only",))):
        r = run(repo, "sap_run.py", *base, "--scores", scores, *aux, "--out", tmp_path / "sap")
        assert r.returncode == 2 and "inputs differ from the lock: ['self_contained']" in r.stderr
    x["self_contained"].write_bytes(before)
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"], "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr


def test_run_refuses_changed_inputs_code_subsets_or_missing_rows(tmp_path):
    repo, tc, dc, ids, x = prepare(tmp_path)
    lock = tmp_path / "SAP_LOCK.json"
    assert run(repo, "sap_lock.py", *lock_args(repo, tmp_path, tc, dc, x), "--out", lock).returncode == 0
    base = [*lock_args(repo, tmp_path, tc, dc, x), "--lock", lock, "--out", tmp_path / "sap"]
    # a partial ER control file (which sap_lock refuses) under a lock that covers it: the run stops too
    stub_only = write_er(tmp_path / "er_stub_only.jsonl", [json.loads(x["er"].read_text().splitlines()[1])])
    forged = json.loads(lock.read_text())
    forged["inputs"]["er_controls[0]"] = sha(stub_only)
    (tmp_path / "forged.json").write_text(json.dumps(forged))
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x, er=stub_only), "--lock", tmp_path / "forged.json",
            "--scores", x["scores"], "--out", tmp_path / "partial")
    assert r.returncode == 3 and "partial ER control file" in r.stderr
    test_rows = [json.loads(l) for l in x["scores"].read_text().splitlines()]
    gone = next(r for r in test_rows if r["record_id"] == ids[4] and r["condition"] == "ft_only" and r["seed"] == "s99")
    short = write(tmp_path / "short.jsonl", [r for r in test_rows if r is not gone])
    r = run(repo, "sap_run.py", *base, "--scores", short)
    assert r.returncode == 3 and "missing" in r.stderr
    # accept()'s commit and the canonical SCORES_META commit share one --repo, so this scenario (the only
    # accepted_missing one here) gets its own fresh repo, its own canonical write, cleanly.
    acc_lock, acc_home, acc_anch = fresh_generation_repo(tmp_path, lock.read_bytes(), "acc")
    acc = accept(acc_home, x, [(ids[4], "deepseek", "ft_only", "s99")], cfg_repo=tmp_path)
    r = run(repo, "sap_run.py", *lock_args(repo, tmp_path, tc, dc, x), "--lock", acc_lock, *acc_anch, "--scores",
            short, *acc, "--out", tmp_path / "sap")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "sap" / "SAP_RESULTS.json").read_text())
    assert res["completeness"]["missing_filled"] == 1
    # the accepted file is not in the lock (it follows it): its sha256, its commit with the fuller
    # dates and every pair are in the results and the report
    am, f = res["accepted_missing"], acc[1]
    assert {k: am[k] for k in ("file", "sha256", "sha", "path", "accepted", "filled")} == {
        "file": str(f), "sha256": sha(f), "sha": acc[3], "path": f.relative_to(acc_home).as_posix(),
        "accepted": [[ids[4], "deepseek", "ft_only", "main", "s99"]],
        "filled": [[ids[4], "deepseek", "ft_only", "s99"]]}
    assert "CommitDate:" in am["fuller"] and am["committed"]
    report = (tmp_path / "sap" / "SAP_REPORT.md").read_text()
    assert sha(f) in report
    assert f"- `{ids[4]}` · deepseek · ft_only · main · s99 (thiếu, đã tính CA = BV = 0)\n" in report
    assert "Dòng thiếu: được chấp nhận trước khi chấm" in report and am["fuller"].rstrip() in report
    write_generation(tmp_path, x)
    # the subsets come from the locked file: a row whose flag differs, or lacks one, stops the run
    flipped = write(tmp_path / "flipped.jsonl", [{**r, "sub_residual_clean": not r["sub_residual_clean"]} if r is test_rows[0] else r
                                                 for r in test_rows])
    r = run(repo, "sap_run.py", *base, "--scores", flipped)
    assert r.returncode == 3 and r.stderr.startswith("refusing:") and "sensitivity_subsets.json" in r.stderr
    lacking = write(tmp_path / "lacking.jsonl", [{k: v for k, v in r.items() if k != "sub_time_coherent_30d"} for r in test_rows])
    r = run(repo, "sap_run.py", *base, "--scores", lacking)
    assert r.returncode == 3 and "sensitivity_subsets.json" in r.stderr
    excl = (tc / "exclusions.jsonl").read_bytes()
    write(tc / "exclusions.jsonl", [{"record_id": i, "reason": "ca_undefined", "excl_bv": False, "excl_ca": True}
                                    for i in ids[:3]])
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"])
    assert r.returncode == 2 and r.stderr.startswith("refusing:") and "exclusions" in r.stderr
    # aux_test compares every locked input too
    r = run(repo, "sap_run.py", *base, "--scores", x["aux_scores"], "--descriptive-only")
    assert r.returncode == 2 and "inputs differ from the lock: ['exclusions']" in r.stderr
    (tc / "exclusions.jsonl").write_bytes(excl)
    dev = x["dev"].read_bytes()
    x["dev"].write_bytes(other_bytes(dev))
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"])
    assert r.returncode == 2 and "dev_scores[0]" in r.stderr
    x["dev"].write_bytes(dev)
    cfg = json.loads((repo / "config" / "sap.json").read_text())
    (repo / "config" / "sap.json").write_text(json.dumps({**cfg, "alpha": 0.1}))
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"])
    assert r.returncode == 2 and "config" in r.stderr
    assert run(repo, "sap_run.py", *base, "--scores", x["scores"], "--descriptive-only").returncode == 2
    (repo / "config" / "sap.json").write_text(json.dumps(cfg))
    assert run(repo, "sap_run.py", *base, "--scores", x["scores"]).returncode == 0
    # the 3D analysis code is locked by sha256; any other code file (3A, 3C) may change
    for p in ("scripts/sap_run.py", "v2tools/sap.py", "v2tools/stats.py"):
        code = (repo / p).read_bytes()
        (repo / p).write_bytes(code + b"\n# edit\n")
        r = run(repo, "sap_run.py", *base, "--scores", x["scores"])
        assert r.returncode == 2 and r.stderr.startswith("refusing:") and "analysis code" in r.stderr and p in r.stderr
        assert run(repo, "sap_run.py", *base, "--scores", x["scores"], "--descriptive-only").returncode == 2
        (repo / p).write_bytes(code)
    for p in ("v2tools/evaluate.py", "scripts/score_run.py"):
        (repo / p).write_text((repo / p).read_text() + "\n# edit\n")
    r = run(repo, "sap_run.py", *base, "--scores", x["scores"])
    assert r.returncode == 0, r.stderr
