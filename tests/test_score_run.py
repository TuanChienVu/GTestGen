import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.phase3c_fixture import PLAN_FMT, a3_repo, attempts_of, gen_plan, make_phase2, plan_line

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)


def plan(*gens, **kw) -> list[str]:
    """--manifest and --coverage of the generation run's contract, built from the rows of `gens`
    (phase3c_fixture.gen_plan)."""
    manifest, coverage = gen_plan(list(gens), **kw)
    return ["--manifest", str(manifest), "--coverage", str(coverage)]


def with_out(args: list[str], out) -> list[str]:
    """`args` with another --out (score_run refuses an --out that already holds a run, so a test that scores again
    names a new one)."""
    i = args.index("--out")
    return [*args[:i + 1], str(out), *args[i + 2:]]


def common(w):
    return ["--phase2", str(w / "phase2"), "--pairs", str(w / "pairs.jsonl"), "--config", str(w / "sources.json"),
            "--full-root", str(w / "full")]


def test_score_run_controls_then_outputs(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls", "--out", str(ctl))
    assert res.returncode == 0, res.stderr
    summary = json.loads((ctl / "CONTROLS.json").read_text())["ALL"]
    assert (summary["gold.bv"], summary["gold.ca"], summary["stub-v2.bv"], summary["stub-v3.ca"]) == (1, 1, 0, 0)
    assert (ctl / "exclusions.jsonl").read_text() == ""
    assert json.loads((ctl / "OPEN_RESOLUTION.json").read_text())["Acme/Shop|gold"] == {"open": 0, "n": 1, "rate": 0.0}
    gold_text = next(json.loads(l)["text"] for l in (ctl / "control_outputs.jsonl").read_text().splitlines()
                     if json.loads(l)["condition"] == "gold")
    assert json.loads((ctl / "ca_ref.jsonl").read_text()) == {"record_id": r["id"], "ca_ref_v3m": 0, "ca_ref_v3all": 0,
                                                              "ca_ref_v3names": 0, "ca_ref_top4": 0}
    # call precision (D11): controls rows carry n_calls and call_precision; CONTROLS.json carries ca_prec@τ; the gold's
    # are 1.0 and 1
    rows = {x["control"]: x for x in map(json.loads, (ctl / "controls.jsonl").read_text().splitlines())}
    assert (rows["gold"]["n_calls"], rows["gold"]["call_precision"], rows["stub-v3names"]["ca"]) == (1, 1.0, 0)
    assert {summary[f"gold.ca_prec@{t}"] for t in (0.1, 0.25, 0.5)} == {1} and summary["stub-v3names.n"] == 1
    assert json.loads((ctl / "CA_PREC_EXCLUDED.json").read_text()) == []
    gen = w / "gen.jsonl"
    o = {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main",
         "text": gold_text, "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True,
         "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64, "fmt": "F2"}
    gen.write_text(json.dumps(o) + "\n")
    out = w / "scores"
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"), "--out", str(out),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3")]
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    row = json.loads((out / "scores.jsonl").read_text())
    assert (row["ca"], row["ca_defined"], row["ca_ref_v3m"], row["bv"], row["cap_hit"], row["open_resolution"]) == (1, True, 0, 1, False, False)
    assert (row["excl_bv"], row["excl_ca"], row["prompt_sha256"], row["sub_residual_clean"]) == (False, False, "b" * 64, True)
    assert row["ca_ref_v3names"] == 0
    # the row's line_sha256 is the manifest ok line's own (the raw bytes of gen's line)
    assert row["line_sha256"] == hashlib.sha256(json.dumps(o).encode()).hexdigest()
    meta = json.loads((out / "SCORES_META.json").read_text())
    assert meta["runs"] == {"deepseek.s42.full_aware": 1} and str(gen) in meta["inputs"] and "dirty" in meta
    assert len(meta["code_sha256"]) == 64 and set(meta["a3_sha256"]) == {f"v2tools/{m}.py" for m in (
        "state", "compose", "records", "outline", "inherit", "symcache", "symtime")}
    # SCORES_META carries scores_sha256, the sha256 of the scores.jsonl it wrote
    assert meta["scores_sha256"] == hashlib.sha256((out / "scores.jsonl").read_bytes()).hexdigest()
    assert (out / "OPEN_RESOLUTION.json").exists()
    # The generation run writes JSONL with json.dumps(ensure_ascii=False): U+2028 and U+0085 stay raw inside a text, so
    # the reader splits lines on "\n" only, as its jsonl_lines and 3A's symcache.jsonl_rows do
    out = w / "scores2"  # each run its own --out
    args = with_out(args, out)
    odd = dict(o, text=gold_text + "\n// a\u2028b\x85c")
    gen.write_text(json.dumps(odd, ensure_ascii=False) + "\n", encoding="utf-8")
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    row = json.loads((out / "scores.jsonl").read_text(encoding="utf-8"))
    assert (row["ca"], row["text_sha256"]) == (1, hashlib.sha256(odd["text"].encode("utf-8")).hexdigest())
    fut = [json.loads(l) for l in (ctl / "future_methods.jsonl").read_text().splitlines()]
    assert fut == [{"record_id": r["id"], "path": "src/Behat/Context/ShopContext.php",
                    "future": sorted(fut[0]["future"])}] and "src/Behat/Context/ShopContext.php:computeTotal" in fut[0]["future"]
    # with helper-call expansion (§2.2b of the design specification), the visible / future sets belong to the RECORD, so
    # the same output scores the same under every format and model
    runs = [dict(o, model=m, seed=sd, condition=c, fmt=PLAN_FMT[c]) for m in ("deepseek", "codellama", "mistral")
            for sd, c in (("base", "zero_shot"), ("base", "rag_only"), ("s42", "full_aware"))]
    gen.write_text("".join(json.dumps(x) + "\n" for x in runs))
    out = w / "scores3"
    args = with_out(args, out)
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    got = {(x["bv"], x["ca"], x["ca_defined"]) for x in map(json.loads, (out / "scores.jsonl").read_text().splitlines())}
    assert got == {(1, 1, True)}
    (ctl / "future_methods.jsonl").write_text(json.dumps({**fut[0], "future": []}) + "\n")
    res = run("scripts/score_run.py", *with_out(args, w / "scores4"), *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "future methods" in res.stderr


def test_score_run_fails_closed(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    o = {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main", "text": "x",
         "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True,
         "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64, "fmt": "F2"}
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"), "--out", str(w / "o"),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3")]
    cases = {"missing": [{k: v for k, v in o.items() if k != "prompt_sha256"}], "twice": [o, o],
             "outside": [{**o, "record_id": "Acme/Shop|features/none.feature|1|s"}]}
    for name, rows in cases.items():
        gen = w / f"{name}.jsonl"
        gen.write_text("".join(json.dumps(x) + "\n" for x in rows))
        res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
        assert res.returncode == 3 and "refusing" in res.stderr, name
    gen = w / "blank.jsonl"  # 3A's reader (symcache.jsonl_rows): a blank line inside a JSONL file is an input fault
    gen.write_text(json.dumps(o) + "\n\n" + json.dumps({**o, "seed": "s11"}) + "\n")
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "not JSONL" in res.stderr
    gen = w / "one.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    # the 3A files must be those of the commit that built the prompts; the code must be the controls run's
    res = run("scripts/score_run.py", *args[:-4], *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "--prompts-meta" in res.stderr
    drifted = w / "PROMPTS_META.drift.json"
    drifted.write_text(json.dumps({"v2tools_commit": a3_repo(w / "a3", edit="\n# edited\n")}))
    res = run("scripts/score_run.py", *[str(drifted) if a == str(w / "PROMPTS_META.json") else a for a in args],
        *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "v2tools/state.py differs" in res.stderr
    meta = json.loads((ctl / "SCORES_META.json").read_text())
    (ctl / "SCORES_META.json").write_text(json.dumps({**meta, "code_sha256": "0" * 64}))
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "other code" in res.stderr
    (ctl / "SCORES_META.json").write_text(json.dumps(meta))
    manifest = w / "manifest.jsonl"
    line = json.dumps({"record_id": r["id"], "condition": "deepseek.s42.full_aware"})
    manifest.write_text(line + "\n")
    overlay = hashlib.sha256(line.encode()).hexdigest()
    done = {"complete": True, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (w / "er.jsonl").write_text("")
    for state, why in ((None, "missing input"), ({**done, "complete": False}, "not complete"),
                       ({**done, "running": True}, "not complete"), ([], "not a JSON object"),
                       ("{complete: true", "not a JSON object"), ({**done, "manifest_sha256": "0" * 64}, "manifest_sha256")):
        if state is not None:  # the ER harness's contract (0c42865): er_complete.json next to er_results.jsonl
            (w / "er_complete.json").write_text(state if isinstance(state, str) else json.dumps(state))
        res = run("scripts/score_run.py", *args,
            *plan(gen), "--outputs", str(gen), "--er-results", str(w / "er.jsonl"), "--er-manifest", str(manifest))
        assert res.returncode == 3 and why in res.stderr and "er_complete.json" in res.stderr, state
    (w / "er_complete.json").write_text(json.dumps(done))
    res = run("scripts/score_run.py", *args,
        *plan(gen), "--outputs", str(gen), "--er-results", str(w / "er.jsonl"), "--er-manifest", str(manifest))
    assert res.returncode == 3 and "no er_results row" in res.stderr
    er = {"record_id": r["id"], "condition": "deepseek.s42.full_aware", "er": 0, "er_status": "SKIP", "trace_action": None,
          "trace_then": None, "er_trace": None, "n_then_evaluable": 0, "overlay_sha256": overlay}
    flagged = {**er, "er": 1, "er_status": "PASS", "tamper_flags": ["env"]}  # er.report: a flagged PASS needs a verdict
    two = w / "manifest2.jsonl"  # one run per manifest, in order; a pair in two manifests is refused
    two.write_text(line + "\n")
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen), "--er-results", str(w / "er.jsonl"),
              "--er-manifest", str(manifest), str(two))
    assert res.returncode == 3 and "--er-results and --er-manifest" in res.stderr
    for rows_, why in (([er, er], "twice"), ([er, {**er, "condition": "deepseek.s11.full_aware"}], "no ER manifest"),
                       ([{**er, "overlay_sha256": "0" * 64}], "overlay_sha256"), ([flagged], "tamper")):
        (w / "er.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows_))
        res = run("scripts/score_run.py", *args,
            *plan(gen), "--outputs", str(gen), "--er-results", str(w / "er.jsonl"), "--er-manifest", str(manifest))
        assert res.returncode == 3 and why in res.stderr, why
    # The ER harness's er.report.verdicts_mismatch: a row with a tamper verdict
    # must carry the sha256 of the CURRENT er_tamper_verdicts.json of its run and the verdict the file gives its key
    # (record_id\tcondition\toverlay_sha256)
    vfile = w / "er_tamper_verdicts.json"
    vkey = f"{er['record_id']}\t{er['condition']}\t{er['overlay_sha256']}"
    vfile.write_text(json.dumps({vkey: "ok"}))
    vsha = hashlib.sha256(vfile.read_bytes()).hexdigest()
    er_run = lambda: run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen), "--er-results",
                         str(w / "er.jsonl"), "--er-manifest", str(manifest))
    for row, why in (({**flagged, "tamper_verdict": "ok"}, "tamper verdicts"),  # no sha of the file
                     ({**flagged, "tamper_verdict": "cheat", "tamper_verdicts_sha256": vsha}, "tamper verdicts"),
                     ({**flagged, "tamper_verdict": "ok", "tamper_verdicts_sha256": "0" * 64}, "tamper verdicts")):
        (w / "er.jsonl").write_text(json.dumps(row) + "\n")
        res = er_run()
        assert res.returncode == 3 and why in res.stderr, row
    (w / "er.jsonl").write_text(json.dumps({**flagged, "tamper_verdict": "ok", "tamper_verdicts_sha256": vsha}) + "\n")
    res = er_run()
    assert res.returncode == 0, res.stderr
    args = with_out(args, w / "o2")  # the runs below must not meet the run above
    vfile.unlink()  # the file gone after apply: refused
    res = er_run()
    assert res.returncode == 3 and "tamper verdicts" in res.stderr
    (w / "sc_empty.jsonl").write_text("")
    empty_sc = [a if a != str(w / "self_contained.jsonl") else str(w / "sc_empty.jsonl") for a in args]
    res = run("scripts/score_run.py", *empty_sc, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "is not in" in res.stderr
    res = run("scripts/score_run.py", *[a if a != str(w / "symtimes.jsonl") else str(w / "none.jsonl") for a in args],
        *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "missing input" in res.stderr
    labels = w / "phase2" / "labels.jsonl"
    labels.write_text("".join(l + " \n" for l in labels.read_text().split("\n") if l))  # same rows, other bytes: the
    # controls run no longer matches
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "was not built from this labels.jsonl" in res.stderr
    res = run("scripts/score_run.py", *args[:-6], "--out", str(w / "o3"),
        *plan(gen), "--outputs", str(gen), "--subsets", str(w / "none.json"))
    assert res.returncode == 3


def test_score_run_refuses_a_record_whose_er_tree_cannot_be_built(tmp_path):
    """A record the ER tree cannot be built for (here a t_vis None, then no t_vis key; ertree.removal) stops score_run
    with exit 3 and a message, not a traceback. The record's id is named once, though ertree's message already carries
    it and score_run prefixes every such refusal with it."""
    w, r = make_phase2(tmp_path)
    recs = w / "phase2" / "records" / "test.jsonl"
    rows = [json.loads(l) for l in recs.read_text().splitlines()]
    for name, edit in (("none", lambda x: {**x, "t_vis": None}),
                       ("no_key", lambda x: {k: v for k, v in x.items() if k != "t_vis"})):
        recs.write_text("".join(json.dumps(edit(x) if x["id"] == r["id"] else x) + "\n" for x in rows))
        res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
                  "--controls", "--out", str(w / f"ctl_{name}"))
        assert res.returncode == 3 and "Traceback" not in res.stderr, (name, res.stderr)
        assert f"refusing: {r['id']}: no t_vis" in res.stderr and res.stderr.count(r["id"]) == 1, (name, res.stderr)
    # the same for a DEV record, which score_run --controls of test prepares for stub-v3top4's prior (D11 (d))
    from tests.phase3c_fixture import make_phase2_masked
    (tmp_path / "masked").mkdir()
    w, dev, test = make_phase2_masked(tmp_path / "masked")
    recs = w / "phase2" / "records" / "dev.jsonl"
    recs.write_text("".join(json.dumps({**x, "t_vis": None} if x["id"] == dev["id"] else x) + "\n"
                            for x in map(json.loads, recs.read_text().splitlines())))
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls", "--out", str(w / "ctl"))
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr
    assert f"refusing: {dev['id']}: no t_vis" in res.stderr and res.stderr.count(dev["id"]) == 1, res.stderr


def test_score_run_prepares_a_dev_record_with_the_masked_answer_infrastructure(tmp_path):
    """Helper-call expansion under the answer mask (§2.2b of the design specification): computeTotal, written with a
    TEST answer, is answer infrastructure that 3A masks in every dev prompt, so it is FUTURE for the dev record "Grand
    total" although it existed at its t_vis. score_run prepares each record with shown4().masked; a run that passed
    frozenset() would write a future_methods.jsonl that the outputs-mode recheck finds consistent, and score the
    record's outputs against the wrong future set."""
    from tests.phase3_fixture import SHOP
    from tests.phase3c_fixture import GRAND, block, make_phase2_masked
    from v2tools.compose import Composer, gold_answer
    from v2tools.evaluate import Repo, future_methods, prepare
    from v2tools.state import RepoContext
    w, dev, test = make_phase2_masked(tmp_path)
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    # the premise, read with 3A's Composer and evaluate directly: the mask alone makes computeTotal future
    ctx = RepoContext("Acme/Shop", "php", w / "full" / "shop.git", dev["commit"], cat)
    masked = Composer({"Acme/Shop": ctx}, {"Acme/Shop": cat}, {}, {x["id"]: x for x in (dev, test)}).masked(dev)
    assert (SHOP, "computeTotal") in masked
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ctx, frozenset(), {})
    lab = next(x for x in map(json.loads, (w / "phase2" / "labels.jsonl").read_text().splitlines())
               if x["feature_path"] == dev["feature_path"])
    fut = lambda m: {f.name for f in future_methods(prepare(dev, repo, lab, m))[SHOP].values()}
    assert fut(masked) == {"computeTotal"} and fut(frozenset()) == set()
    ctl = w / "ctl"
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "dev",
              "--controls", "--out", str(ctl))
    assert res.returncode == 0, res.stderr
    assert (ctl / "exclusions.jsonl").read_text() == ""
    assert [json.loads(l) for l in (ctl / "future_methods.jsonl").read_text().splitlines()] == [
        {"record_id": dev["id"], "path": SHOP, "future": [f"{SHOP}:computeTotal"]}]
    # helper expansion: an output that redefines the future helper replaces it (unmasked, it would collide), and a call
    # to it counts as the calls of its ER-tree body (none), so cart->add alone does not meet the gold's
    # checkout->assertTotal
    gold = gold_answer(dev, cat)
    helper = "private function computeTotal(string $total): string\n{\n    return $total . '';\n}"
    texts = {"deepseek": gold[:gold.rindex("`" * 3)] + helper + "\n" + "`" * 3,
             "mistral": block("#[Then('" + GRAND + "')]\npublic function grand(string $total): void\n{\n"
                              "    $this->cart->add($this->computeTotal($total));\n}")}
    o = {"record_id": dev["id"], "seed": "s42", "condition": "full_aware", "variant": "main", "prompt_tokens": 100,
         "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True, "prompts_sha256": "a" * 64,
         "prompt_sha256": "b" * 64, "fmt": "F2"}
    gen = w / "gen.jsonl"
    gen.write_text("".join(json.dumps({**o, "model": m, "text": t}) + "\n" for m, t in texts.items()))
    out = w / "scores"
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "dev",
              "--controls-dir", str(ctl), "--self-contained", str(w / "self_contained.jsonl"), "--subsets",
              str(w / "subsets.json"), "--out", str(out), "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo",
              str(w / "a3"), *plan(gen), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    rows = {x["model"]: x for x in map(json.loads, (out / "scores.jsonl").read_text().splitlines())}
    assert [rows["deepseek"][k] for k in ("insert_status", "bv", "ca_defined", "ca")] == ["ok", 1, True, 1]
    assert [rows["mistral"][k] for k in ("insert_status", "bv", "ca_defined", "ca")] == ["ok", 1, True, 0]


def test_score_run_writes_the_dose_table_before_the_lock(tmp_path):
    """--dose-table writes dose.jsonl, one row {record_id, model, dose} per (F2 prompt model, record) of the partition,
    with describe.dose on the record prepared as every score_run mode prepares it and the model's F2 prompt text
    (messages[0].content of its --prompts-f2 file), the value describe.row puts in a scores row; the SAP checks each
    row's dose against this locked table. Fail closed: a record of the partition with no F2 line for a model (unless
    compose's excluded.jsonl drops it for that model), a line twice, a line of another partition, model or format, or a
    line for a record compose excluded, stops the run (exit 3)."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    line = {"record_id": r["id"], "partition": "test", "fmt": "F2", "model": "deepseek", "answer": None,
            "messages": [{"role": "user", "content": "Scenario\nProject knowledge:\n private Checkout $checkout,\n"}]}
    f2 = {m: w / f"{m}.all.test.F2.jsonl" for m in ("deepseek", "mistral")}
    f2["deepseek"].write_text(json.dumps(line) + "\n")
    f2["mistral"].write_text(json.dumps({**line, "model": "mistral", "messages": [{"role": "user", "content": "none"}]})
                             + "\n")
    excluded = w / "excluded.jsonl"
    excluded.write_text("")
    base = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--compose-excluded",
            str(excluded), "--dose-table", "--prompts-f2", *(f"{m}={f}" for m, f in f2.items())]
    res = run("scripts/score_run.py", *base, "--out", str(w / "dose"))
    assert res.returncode == 0, res.stderr
    table = [json.loads(x) for x in (w / "dose" / "dose.jsonl").read_text().splitlines()]
    assert table == [{"dose": 1, "model": "deepseek", "record_id": r["id"]}, {"dose": 0, "model": "mistral",
                                                                              "record_id": r["id"]}]
    meta = json.loads((w / "dose" / "SCORES_META.json").read_text())
    assert meta["mode"] == "dose" and str(f2["deepseek"]) in meta["inputs"] and "dose.jsonl" in (
        w / "dose" / "SHA256SUMS").read_text()
    # the value describe.row puts in the scores row of an output of that model
    gold_text = next(json.loads(x)["text"] for x in (ctl / "control_outputs.jsonl").read_text().splitlines()
                     if json.loads(x)["condition"] == "gold")
    o = {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main",
         "text": gold_text, "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True,
         "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64, "fmt": "F2"}
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls-dir", str(ctl), "--self-contained", str(w / "self_contained.jsonl"), "--subsets",
              str(w / "subsets.json"), "--out", str(w / "scores"), "--prompts-meta", str(w / "PROMPTS_META.json"),
              "--a3-repo", str(w / "a3"), "--prompts-f2", f"deepseek={f2['deepseek']}",
                  *plan(gen), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    assert json.loads((w / "scores" / "scores.jsonl").read_text())["dose"] == table[0]["dose"]
    bad = w / "bad.jsonl"
    cases = {"missing": "", "twice": json.dumps(line) + "\n" + json.dumps(line) + "\n",
             "partition": json.dumps({**line, "partition": "dev"}) + "\n",
             "model": json.dumps({**line, "model": "mistral"}) + "\n",
             "format": json.dumps({**line, "fmt": "F1"}) + "\n",
             "outside": json.dumps(line) + "\n"
             + json.dumps({**line, "record_id": "Acme/Shop|features/none.feature|1|s"}) + "\n"}
    for name, text in cases.items():
        bad.write_text(text)
        res = run("scripts/score_run.py", *base[:-2], f"deepseek={bad}", "--out", str(w / f"d_{name}"))
        assert res.returncode == 3 and "refusing" in res.stderr, (name, res.stderr)
    for model, want in ((None, 0), ("deepseek", 0), ("mistral", 3)):  # compose dropped the record for every model
        excluded.write_text(json.dumps({"record_id": r["id"], "model": model, "fmt": None, "reason": "x"}) + "\n")
        bad.write_text("")
        res = run("scripts/score_run.py", *base[:-2], f"deepseek={bad}", "--out", str(w / f"d_excl_{model}"))
        assert res.returncode == want, (model, res.stderr)
    excluded.write_text(json.dumps({"record_id": r["id"], "model": None, "fmt": None, "reason": "x"}) + "\n")
    res = run("scripts/score_run.py", *base, "--out", str(w / "d_excl2"))  # a line for a record compose dropped
    assert res.returncode == 3 and "excluded" in res.stderr
    res = run("scripts/score_run.py", *base[:-3], "--out", str(w / "d_none"))  # --dose-table needs --prompts-f2
    assert res.returncode == 3 and "--prompts-f2" in res.stderr


def test_score_run_checks_outputs_against_the_generation_manifest(tmp_path):
    """The generation run's manifest contract (fixed 26/09/2026): an outputs run needs --manifest and --coverage unless
    every row is a probe (model "probe"). Refused (exit 3): the manifest's sha256 is not coverage.manifest_sha256;
    coverage.status is not PASS and status_except_missing is not PASS; a row whose key (model, condition, seed, variant,
    record_id, fmt) is not an `ok` line; a row whose raw bytes do not hash to the line's line_sha256; a row sha field
    that differs from the line's; an `ok` line of a (model, variant) in the outputs, record of the partition, with no
    row; a manifest that is not sorted, repeats a key, or pairs a condition with the wrong fmt. Score rows copy
    decision_sha256 and gen_code_sha256 from the manifest; SCORES_META records the manifest, coverage and
    accepted-missing sha256, the --accepted-missing-commit, scored_utc and skipped_other_partition."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    gold_text = next(json.loads(x)["text"] for x in (ctl / "control_outputs.jsonl").read_text().splitlines()
                     if json.loads(x)["condition"] == "gold")
    o = {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main",
         "fmt": "F2", "text": gold_text, "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144,
         "stopped_at_eos": True, "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64,
         "adapter_tensor_sha256": "c" * 64, "leakcheck_receipt_sha256": "f" * 64}  # decision, gen code: manifest only
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3"), "--outputs", str(gen)]
    score = lambda *extra, out="o": run("scripts/score_run.py", *args, "--out", str(w / out), *extra)
    res = score(*plan(gen))
    assert res.returncode == 0, res.stderr
    row = json.loads((w / "o" / "scores.jsonl").read_text())
    assert (row["decision_sha256"], row["gen_code_sha256"]) == \
        ("d" * 64, {"v2gpu/generate.py": "e" * 64, "v2gpu/runs.py": "f" * 64})  # the writer's object, copied
    # line_sha256 is copied from the manifest ok line too (the raw bytes of gen's line)
    assert row["line_sha256"] == hashlib.sha256(gen.read_bytes().rstrip(b"\n")).hexdigest()
    meta = json.loads((w / "o" / "SCORES_META.json").read_text())
    manifest = w / "manifest.main.jsonl"
    assert meta["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert meta["coverage_sha256"] == hashlib.sha256((w / "coverage.main.json").read_bytes()).hexdigest()
    assert (meta["accepted_missing_sha256"], meta["accepted_missing_commit"], meta["skipped_other_partition"]) == (
        None, None, 0) and meta["scored_utc"].endswith("Z")
    assert meta["scores_sha256"] == hashlib.sha256((w / "o" / "scores.jsonl").read_bytes()).hexdigest()
    res = score(out="no_manifest")
    assert res.returncode == 3 and "--manifest" in res.stderr
    probe = w / "probe.jsonl"
    probe.write_text(json.dumps({**o, "model": "probe"}) + "\n")
    res = run("scripts/score_run.py", *args[:-1], str(probe), "--out", str(w / "p"))
    assert res.returncode == 0, res.stderr
    # a probe row has no manifest line: line_sha256 (like decision_sha256, gen_code_sha256) stays None
    assert json.loads((w / "p" / "scores.jsonl").read_text())["line_sha256"] is None
    other = {**o, "seed": "s11"}
    cases = {
        "manifest_sha256": dict(status=None), "status": dict(status="FAIL", status_except_missing="FAIL"),
        "line_sha256": dict(edit=lambda xs: [{**x, "line_sha256": "0" * 64} for x in xs]),
        "prompt_sha256": dict(edit=lambda xs: [{**x, "prompt_sha256": "9" * 64} for x in xs]),
        "not an ok line": dict(edit=lambda xs: [{**x, "status": "excluded", "reason": "r", "line_sha256": None}
                                                 for x in xs]),
        "no output row": dict(extra=[plan_line(other, line_sha256="2" * 64)]),
        "sorted": dict(extra=[plan_line({**o, "model": "codellama"}, line_sha256="2" * 64)],
                       edit=lambda xs: list(reversed(xs))),
        "twice": dict(edit=lambda xs: xs + xs),
        "fmt": dict(edit=lambda xs: [{**x, "fmt": "F1"} for x in xs]),
    }
    for name, kw in cases.items():
        extra = plan(gen, **kw)
        if name == "manifest_sha256":
            cov = w / "coverage.main.json"
            cov.write_text(json.dumps({**json.loads(cov.read_text()), "manifest_sha256": "0" * 64}))
        res = score(*extra, out=name.replace(" ", "_"))
        assert res.returncode == 3 and name in res.stderr, (name, res.stderr)
    res = score(*plan(gen, extra=[plan_line({**o, "model": "codellama"}, line_sha256="2" * 64)]), out="last")
    assert res.returncode == 0, res.stderr  # completeness is per (model, variant) in the outputs


def test_score_run_accepts_only_evidenced_missing_rows(tmp_path):
    """The generation run's manifest contract, the evidence rule (fixed 26/09/2026; score_run.EVIDENCE_STATUSES): a
    `missing` line is accepted only when coverage.status_except_missing is PASS and --accepted-missing (with
    --accepted-missing-commit) lists EXACTLY the manifest's missing lines, each with the line's key and receipt_sha256,
    and each missing line is evidence: receipt_status CRASHED or INCOMPLETE, >= 2 attempts, each with a sha256,
    receipt_status/receipt_sha256 = the last attempt's, and evidence_eligible (when present) equal to that. REFUSED_*,
    RECHECK_MISMATCH or "absent" never count."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    o = {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main",
         "fmt": "F2", "text": "x", "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144,
         "stopped_at_eos": True, "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64}
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3"), "--outputs", str(gen)]
    key = {**o, "seed": "s11"}
    accepted = w / "accepted.jsonl"

    def attempt(line: dict, entries=None, commit=True, out="o", **kw):
        entries = [{**{k: line[k] for k in ("model", "condition", "seed", "variant", "record_id", "fmt")},
                    "reason": "crashed twice", "receipt_sha256": line["receipt_sha256"]}] \
            if entries is None else entries
        accepted.write_text("".join(json.dumps(x) + "\n" for x in entries))
        extra = ["--accepted-missing", str(accepted)] + (["--accepted-missing-commit", "abc123"] if commit else [])
        return run("scripts/score_run.py", *args, "--out", str(w / out), *plan(gen, extra=[line], **kw), *extra)
    good = plan_line(key, "missing")
    bare = lambda statuses: {k: v for k, v in plan_line(key, "missing", attempts=attempts_of(statuses)).items()
                             if k != "evidence_eligible"}
    res = attempt(good)
    assert res.returncode == 0, res.stderr
    meta = json.loads((w / "o" / "SCORES_META.json").read_text())
    assert meta["accepted_missing_sha256"] == hashlib.sha256(accepted.read_bytes()).hexdigest()
    assert meta["accepted_missing_commit"] == "abc123"
    assert attempt(plan_line(key, "missing", attempts=attempts_of(["INCOMPLETE", "INCOMPLETE"])), out="o2").returncode == 0
    bad = {
        "no --accepted-missing": run("scripts/score_run.py", *args, "--out", str(w / "b0"), *plan(gen, extra=[good])),
        "commit": attempt(good, commit=False, out="b1"),
        "receipt_sha256": attempt(good, entries=[{**{k: good[k] for k in ("model", "condition", "seed", "variant",
                                                                           "record_id", "fmt")},
                                                  "reason": "x", "receipt_sha256": "0" * 64}], out="b2"),
        "not missing": attempt(good, entries=[{**{k: good[k] for k in ("model", "condition", "seed", "variant",
                                                                        "record_id", "fmt")},
                                               "reason": "x", "receipt_sha256": good["receipt_sha256"]},
                                              {**{k: o[k] for k in ("model", "condition", "seed", "variant",
                                                                     "record_id", "fmt")},
                                               "reason": "x", "receipt_sha256": "1" * 64}], out="b3"),
        "evidence refused": attempt(bare(["CRASHED", "REFUSED_PREFLIGHT"]), out="b4"),
        "evidence one": attempt(bare(["CRASHED"]), out="b5"),
        "evidence absent": attempt(bare([]), out="b6"),
        "evidence mismatch": attempt(bare(["CRASHED", "RECHECK_MISMATCH"]), out="b7"),
        "evidence_eligible": attempt({**good, "evidence_eligible": False}, out="b8"),
        "last attempt": attempt({**good, "receipt_sha256": "7" * 64}, out="b9"),
        "status PASS": attempt(good, status="PASS", out="b10"),
        "status_except_missing": attempt(good, status_except_missing="FAIL", out="b11"),
        "no sha256": attempt({k: v for k, v in plan_line(key, "missing", attempts=[
            {**a, "sha256": None} if i == 0 else a for i, a in enumerate(attempts_of(["CRASHED", "CRASHED"]))]).items()
            if k != "evidence_eligible"}, out="b12"),
    }
    for name, res in bad.items():
        assert res.returncode == 3 and "refusing" in res.stderr, (name, res.stderr)
        # without evidence_eligible (a convenience field), the attempts alone decide
        assert name not in ("evidence refused", "evidence one", "evidence absent", "evidence mismatch", "no sha256") \
            or "have no evidence" in res.stderr, (name, res.stderr)


def test_score_run_skips_rows_of_another_partition(tmp_path):
    """The generation run's genqueue: one workdir's outputs.jsonl holds test and aux_test rows. score_run --partition P
    skips, and counts in SCORES_META (skipped_other_partition), a row whose record is a KNOWN record of P's
    test/aux_test partner ONLY (the only two partitions that ever share a generation workdir); a row of an unknown
    record is still refused; the manifest's completeness is checked on P's records only."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    aux_id = "Acme/Shop|features/aux.feature|3|s"
    (w / "phase2" / "records" / "aux_test.jsonl").write_text(json.dumps({"id": aux_id, "repo": "Acme/Shop"}) + "\n")
    o = {"model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main", "fmt": "F2", "text": "x",
         "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True,
         "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64}
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({**o, "record_id": aux_id}) + "\n" + json.dumps({**o, "record_id": r["id"]}) + "\n")
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3"), "--outputs", str(gen)]
    extra = [plan_line({**o, "record_id": aux_id, "seed": "s11"}, line_sha256="2" * 64)]  # the partner's own line
    res = run("scripts/score_run.py", *args, "--out", str(w / "o"), *plan(gen, extra=extra))
    assert res.returncode == 0, res.stderr
    assert [json.loads(x)["record_id"] for x in (w / "o" / "scores.jsonl").read_text().splitlines()] == [r["id"]]
    assert json.loads((w / "o" / "SCORES_META.json").read_text())["skipped_other_partition"] == 1


def test_score_run_refuses_a_train_or_dev_row_scored_under_test(tmp_path):
    """A row of a KNOWN train or dev record is REFUSED (exit 3), never skipped, when scoring test or aux_test (only a
    test/aux_test pair may share a generation workdir; a train or dev row never belongs there)."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    dev_id, train_id = "Acme/Shop|features/dev.feature|3|s", "Acme/Shop|features/train.feature|3|s"
    (w / "phase2" / "records" / "dev.jsonl").write_text(json.dumps({"id": dev_id, "repo": "Acme/Shop"}) + "\n")
    (w / "phase2" / "records" / "train.jsonl").write_text(json.dumps({"id": train_id, "repo": "Acme/Shop"}) + "\n")
    o = {"model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main", "fmt": "F2", "text": "x",
         "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True,
         "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64}
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3")]
    for name, rid in (("dev", dev_id), ("train", train_id)):
        gen = w / f"gen_{name}.jsonl"
        gen.write_text(json.dumps({**o, "record_id": rid}) + "\n" + json.dumps({**o, "record_id": r["id"]}) + "\n")
        res = run("scripts/score_run.py", *args, "--outputs", str(gen), "--out", str(w / f"o_{name}"), *plan(gen))
        assert res.returncode == 3 and "refusing" in res.stderr, (name, res.stderr)


def test_score_run_refuses_a_test_row_scored_under_dev(tmp_path):
    """The reverse direction: a test or aux_test record's row is refused when scoring dev, since dev shares no
    generation workdir with test/aux_test either."""
    from tests.phase3c_fixture import make_phase2_masked
    w, dev, test = make_phase2_masked(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "dev",
               "--controls", "--out", str(ctl)).returncode == 0
    o = {"model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main", "fmt": "F2", "text": "x",
         "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True,
         "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64}
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({**o, "record_id": dev["id"]}) + "\n" + json.dumps({**o, "record_id": test["id"]}) + "\n")
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "dev", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3"), "--outputs", str(gen)]
    res = run("scripts/score_run.py", *args, "--out", str(w / "o"), *plan(gen))
    assert res.returncode == 3 and "refusing" in res.stderr, res.stderr


def test_score_run_gives_stub_v3top4_the_dev_prior(tmp_path):
    """Stub-v3top4's prior (D11 (d), fixed 26/09/2026): score_run --controls prepares the DEV partition's records once
    (as for scoring dev: evaluate.prepare with Shown4.masked), builds controls.dev_prior from them and passes it to
    every record's controls(); ca_ref.jsonl gains ca_ref_top4 (the CA of stub-v3top4; None for Ruby). The dev record's
    gold calls $this->checkout->assertTotal($this->computeTotal(...)): both checkout and computeTotal are
    promoted/declared alongside the TEST record's own answer (at its own t_vis), i.e. FUTURE for the test record too
    (top4_calls.writable excludes a future receiver, as visible_calls does), so the test record's stub-v3top4 does not
    write them from the dev prior and its ca_ref_top4 stays 0."""
    from tests.phase3c_fixture import make_phase2_masked
    w, dev, test = make_phase2_masked(tmp_path)
    ctl = w / "ctl"
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls", "--out", str(ctl))
    assert res.returncode == 0, res.stderr
    text = next(json.loads(x)["text"] for x in (ctl / "control_outputs.jsonl").read_text().splitlines()
                if json.loads(x)["condition"] == "stub-v3top4")
    assert "    $this->cart->add();\n    $this->indexPage->open();\n" in text
    assert "checkout" not in text and "computeTotal" not in text
    assert json.loads((ctl / "ca_ref.jsonl").read_text())["ca_ref_top4"] == 0


def score_run_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_run_for_tests", ROOT / "scripts" / "score_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_controls_summary_counts_ca_prec_over_the_ca_at_each_tau():
    """D11 (c): CONTROLS.json's <control>.ca_prec@τ = the sum of CA × [call_precision ≥ τ] over the rows, like .ca (a
    record without CA adds 0, whatever its precision), for each of 0.1, 0.25, 0.5."""
    sr = score_run_module()
    rows = [{"repo": "R", "control": "c", "bv": 1, "ca": 1, "ca_defined": True, "fallback": 0, "call_precision": 0.3},
            {"repo": "R", "control": "c", "bv": 1, "ca": 0, "ca_defined": True, "fallback": 2, "call_precision": 0.9},
            {"repo": "R", "control": "c", "bv": 0, "ca": 0, "ca_defined": False, "fallback": 0, "call_precision": None}]
    s = sr.controls_summary(rows)
    assert s["R"] == s["ALL"] == {"c.n": 3, "c.bv": 2, "c.ca": 1, "c.ca_defined": 2, "c.fallback": 1,
                                  "c.ca_prec@0.1": 1, "c.ca_prec@0.25": 1, "c.ca_prec@0.5": 0}


def test_ca_refs_read_each_reference_from_its_own_control():
    """ca_ref.jsonl: each ca_ref column is the CA of its own control (v3m, v3all, v3names, v3top4), None when the record
    has no such control row (Ruby)."""
    sr = score_run_module()
    res = {f"stub-{v}": (v,) for v in ("v3m", "v3all", "v3names", "v3top4")}
    assert sr.ca_refs(res, lambda s: s) == {"ca_ref_v3m": "v3m", "ca_ref_v3all": "v3all", "ca_ref_v3names": "v3names",
                                            "ca_ref_top4": "v3top4"}
    ruby = sr.ca_refs({"stub-v3m": ("x",), "stub-v3all": ("y",)}, lambda s: s)
    assert (ruby["ca_ref_v3names"], ruby["ca_ref_top4"]) == (None, None)


# ----- The official v0 prompts were built at 37ed1a2, before a later 3A commit changed symcache.py;
# symcache.py is checked against an explicit allowlist (added 04/10/2026), the data it reads by symtimes_sha256

def commit_edit(repo, f: str, text: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    (repo / f).write_text(text)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", f"edit {f}"], check=True, capture_output=True, env=env)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_symcache_is_checked_against_an_explicit_allowlist(tmp_path, monkeypatch):
    """v2tools/symcache.py, at the prompts' commit and here, must each be one of the versions A3_ALLOWED names by
    sha256 (37ed1a2, which built the official v0 prompts; c6b16c6, the commit that closed 3A, which added
    jsonl_rows; and the public tree's file, c6b16c6's with cleaned docstrings), never "any commit"; the six other
    3A files stay byte for byte the prompts' commit's."""
    m = score_run_module()
    assert m.A3_ALLOWED == {"v2tools/symcache.py": frozenset({
        "c21ee44837a0479aef92914663536020f7013d71dc07a8dc5b9935808b3e35ed",  # 37ed1a2
        "aa30b8cdba11ad0cfca2e340dab0cc5ebdbb6469782e51299b2ede68f6950dc7",  # c6b16c6
        "f1c41842ac05c37fb17e55d18ecfac14823c045fc77eab743ad3e62470e54402"})}  # public tree (docstrings only)
    here = hashlib.sha256((ROOT / "v2tools" / "symcache.py").read_bytes()).hexdigest()
    assert here in m.A3_ALLOWED["v2tools/symcache.py"]  # a new symcache.py must be added to the allowlist
    repo = tmp_path / "a3"
    assert m.a3_errors(repo, a3_repo(repo)) == []
    edited = commit_edit(repo, "v2tools/symcache.py", (repo / "v2tools" / "symcache.py").read_text() + "\n# edited\n")
    assert m.a3_errors(repo, edited) == [f"v2tools/symcache.py at {edited} is not an allowed version (A3_ALLOWED)"]
    other = hashlib.sha256((repo / "v2tools" / "symcache.py").read_bytes()).hexdigest()
    monkeypatch.setattr(m, "A3_ALLOWED", {"v2tools/symcache.py": frozenset({other})})  # the prompts' one only
    assert m.a3_errors(repo, edited) == ["v2tools/symcache.py here is not an allowed version (A3_ALLOWED)"]
    monkeypatch.setattr(m, "A3_ALLOWED", {"v2tools/symcache.py": frozenset({other, here})})  # as v0: two allowed versions
    assert m.a3_errors(repo, edited) == []
    state = commit_edit(repo, "v2tools/state.py", (repo / "v2tools" / "state.py").read_text() + "\n# edited\n")
    assert m.a3_errors(repo, state) == [f"v2tools/state.py differs from the one at {state} that built the prompts"]


def test_the_prompts_symtimes_are_the_symtimes_scored_with(tmp_path):
    """PROMPTS_META.symtimes_sha256 must be the sha256 of --symtimes, the data symcache.load reads: a META whose
    value differs, or that has none, is refused (exit 3) before any scoring; the fixture's own META scores."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    o = {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main",
         "fmt": "F2", "text": "x", "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144,
         "stopped_at_eos": True, "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64,
         "adapter_tensor_sha256": "c" * 64, "leakcheck_receipt_sha256": "f" * 64}
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    args = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"),
            "--a3-repo", str(w / "a3"), "--outputs", str(gen), *plan(gen)]
    pm = json.loads((w / "PROMPTS_META.json").read_text())
    assert pm["symtimes_sha256"] == hashlib.sha256((w / "symtimes.jsonl").read_bytes()).hexdigest()
    for name, meta in (("other", {**pm, "symtimes_sha256": "0" * 64}),
                       ("none", {k: v for k, v in pm.items() if k != "symtimes_sha256"})):
        (w / f"PROMPTS_META.{name}.json").write_text(json.dumps(meta))
        res = run("scripts/score_run.py", *args, "--prompts-meta", str(w / f"PROMPTS_META.{name}.json"),
                  "--out", str(w / f"o_{name}"))
        assert res.returncode == 3 and "symtimes_sha256" in res.stderr and "Traceback" not in res.stderr, res.stderr
        assert not (w / f"o_{name}" / "scores.jsonl").exists()
    res = run("scripts/score_run.py", *args, "--prompts-meta", str(w / "PROMPTS_META.json"), "--out", str(w / "o"))
    assert res.returncode == 0, res.stderr


# ----- Tests added after mutation testing of score_run.py: each kills mutants of the code that survived the earlier
# tests, as its docstring or section line says -------------------------------------------------------------------------

NOW = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")


def git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=NOW).stdout.strip()


def controls(w, out="ctl", partition="test"):
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", partition,
              "--controls", "--out", str(w / out))
    assert res.returncode == 0, res.stderr
    return w / out


def gold_text(ctl):
    return next(json.loads(l)["text"] for l in (ctl / "control_outputs.jsonl").read_text().splitlines()
                if json.loads(l)["condition"] == "gold")


def output(r, text, **kw):
    return {"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main",
            "fmt": "F2", "text": text, "prompt_tokens": 100, "new_tokens": 50, "max_new_tokens": 6144,
            "stopped_at_eos": True, "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64, **kw}


def outputs_args(w, ctl, out):
    return [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls-dir", str(ctl),
            "--self-contained", str(w / "self_contained.jsonl"), "--subsets", str(w / "subsets.json"), "--out", str(out),
            "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3")]


def score(w, ctl, rows, *extra, out="o", with_plan=True):
    gen = w / "gen.jsonl"
    gen.write_text("".join(json.dumps(x) + "\n" for x in rows))
    return run("scripts/score_run.py", *outputs_args(w, ctl, w / out), *(plan(gen) if with_plan else []), *extra,
               "--outputs", str(gen))


def scores(w, out="o"):
    return [json.loads(l) for l in (w / out / "scores.jsonl").read_text().splitlines()]


# ---- the machine exclusions (real data: aux 14 = 4 BV + 10 CA-only, dev 17 = 15 + 2) ---------------------------------

def test_each_exclusion_reason_is_written_with_its_flags(tmp_path):
    """exclusions.jsonl: gold fails BV (excl_bv, the failed conditions), HEAD file does not parse (excl_bv; BV_EXCLUDED),
    CA undefined (excl_ca only). The earlier tests never wrote a non-empty exclusions.jsonl: 7 mutants of the rule
    (BV_EXCLUDED, the reason order, ca_undefined dropped, failed, excl_ca, repo) survived them."""
    head = tmp_path / "head"
    head.mkdir()
    w, r = make_phase2(head)
    bare = w / "full" / "shop.git"
    wc = head / "wc"
    git("clone", "-q", str(bare), str(wc))
    f = wc / "src/Behat/Context/ShopContext.php"
    f.write_text(f.read_text() + "\n}}} garbage\n")
    git("commit", "-q", "-am", "break the head file", cwd=wc)
    git("push", "-q", str(bare), "HEAD:refs/heads/main", cwd=wc)
    (w / "phase2" / "records" / "test.jsonl").write_text(json.dumps({**r, "commit": git("-C", str(bare), "rev-parse",
                                                                                         "refs/heads/main")}) + "\n")
    row = lambda reason, bv, failed: {"excl_bv": bv, "excl_ca": True, "failed": failed, "reason": reason,
                                      "record_id": r["id"], "repo": "Acme/Shop"}
    got = [json.loads(l) for l in (controls(w) / "exclusions.jsonl").read_text().splitlines()]
    assert got == [row("head_unparsable", True, ["syntax", "binding", "no_ambiguity", "arity", "grounded", "resolved"])]
    (tmp_path / "bv").mkdir()
    w, r = make_phase2(tmp_path / "bv")          # labels name another def for the CREATE step: gold fails binding
    labels = [json.loads(l) for l in (w / "phase2" / "labels.jsonl").read_text().splitlines()]
    labels[0]["step_def_ids"] = [labels[0]["step_def_ids"][0]] * 2
    (w / "phase2" / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    got = [json.loads(l) for l in (controls(w) / "exclusions.jsonl").read_text().splitlines()]
    assert got == [row("gold_fails_bv", True, ["binding", "no_ambiguity", "arity", "grounded"])]
    (tmp_path / "ca").mkdir()
    w, r = make_phase2(tmp_path / "ca")          # gold passes BV (a member is referenced) but calls nothing: CA undefined
    code = "#[Then('I should see the total :total')]\npublic function iShouldSeeTotal(string $total): void\n{\n    $x = $this->checkout;\n}"
    (w / "phase2" / "records" / "test.jsonl").write_text(json.dumps({**r, "create": [{**r["create"][0], "code": code}]}) + "\n")
    cat = [json.loads(l) for l in (w / "phase2" / "catalog.jsonl").read_text().splitlines()]
    (w / "phase2" / "catalog.jsonl").write_text("".join(json.dumps({**f, "code": code} if f["fn_id"] == r["create"][0]["fn_id"]
                                                                  else f) + "\n" for f in cat))
    ctl = controls(w)
    assert [json.loads(l) for l in (ctl / "exclusions.jsonl").read_text().splitlines()] == [row("ca_undefined", False, [])]
    gold = next(json.loads(l) for l in (ctl / "controls.jsonl").read_text().splitlines() if json.loads(l)["control"] == "gold")
    assert (gold["bv"], gold["ca_defined"]) == (1, False)


def test_score_rows_carry_the_denominator_flags_of_the_exclusions(tmp_path):
    """excl_bv only for gold_fails_bv and head_unparsable, excl_ca for every excluded record (3D's
    locked_flags refuses a row that disagrees, but no 3C test pinned the copy)."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    text = gold_text(ctl)
    for reason, want in (("gold_fails_bv", (True, True)), ("head_unparsable", (True, True)), ("ca_undefined", (False, True)),
                         (None, (False, False))):
        (ctl / "exclusions.jsonl").write_text("" if reason is None else json.dumps(
            {"record_id": r["id"], "repo": "Acme/Shop", "reason": reason, "excl_bv": want[0], "excl_ca": True,
             "failed": []}) + "\n")
        res = score(w, ctl, [output(r, text)], out=f"o_{reason}")
        assert res.returncode == 0, res.stderr
        (row,) = scores(w, f"o_{reason}")
        assert (row["excl_bv"], row["excl_ca"]) == want, reason


# ---- cap_hit (the rate of outputs that hit the token cap) ------------------------------------------------------------

def test_cap_hit_needs_no_eos_and_the_whole_budget(tmp_path):
    """cap_hit = not stopped_at_eos and new_tokens >= max_new_tokens. The earlier tests only ever scored a row with
    cap_hit False: a `>` for the `>=` (the rate of capped outputs becomes 0 everywhere) and a dropped EOS test survived."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    text = gold_text(ctl)
    rows = [output(r, text, seed=s, stopped_at_eos=eos, new_tokens=n) for s, eos, n in
            (("s42", False, 6144), ("s11", False, 6143), ("s23", True, 6144), ("s7", True, 100))]
    assert score(w, ctl, rows).returncode == 0
    assert {x["seed"]: x["cap_hit"] for x in scores(w)} == {"s42": True, "s11": False, "s23": False, "s7": False}


# ---- ER columns reach the scores row ---------------------------------------------------------------------------------

def er_run(w, name, rows, manifest_line):
    d = w / name
    d.mkdir()
    manifest = d / "manifest.jsonl"
    manifest.write_text(manifest_line + "\n")
    (d / "er_complete.json").write_text(json.dumps({"complete": True, "manifest_sha256": hashlib.sha256(
        manifest.read_bytes()).hexdigest()}))
    (d / "er_results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return d / "er_results.jsonl", manifest


def test_er_results_are_joined_into_the_row_of_their_run(tmp_path):
    """The ER columns of a scores row are the er_results row of (record, run), when the run is in an ER manifest. No
    earlier test read an ER value back: `er.get(k) if k in in_manifest else None` replaced by None survived all of
    test_score_run and test_score_independent. Also: a flagged output whose TRACE matches (er 0) needs a verdict, and a
    (record, run) in two ER manifests is refused."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    text = gold_text(ctl)
    cond = "deepseek.s42.full_aware"
    line = json.dumps({"record_id": r["id"], "condition": cond})
    er = {"record_id": r["id"], "condition": cond, "er": 1, "er_status": "PASS", "trace_action": 1, "trace_then": 0,
          "er_trace": "x", "n_then_evaluable": 2, "overlay_sha256": hashlib.sha256(line.encode()).hexdigest()}
    res_f, man = er_run(w, "er1", [er], line)
    res = score(w, ctl, [output(r, text)], "--er-results", str(res_f), "--er-manifest", str(man))
    assert res.returncode == 0, res.stderr
    (row,) = scores(w)
    assert [row[k] for k in ("er", "er_status", "trace_action", "trace_then", "er_trace", "n_then_evaluable")] == [
        1, "PASS", 1, 0, "x", 2]
    flagged = {**er, "er": 0, "trace_action": 1, "tamper_flags": ["env"]}
    res_f, man = er_run(w, "er2", [flagged], line)
    res = score(w, ctl, [output(r, text)], "--er-results", str(res_f), "--er-manifest", str(man), out="o2")
    assert res.returncode == 3 and "tamper" in res.stderr, res.stderr
    res_a, man_a = er_run(w, "er3", [er], line)
    res_b, man_b = er_run(w, "er4", [er], line)
    res = score(w, ctl, [output(r, text)], "--er-results", str(res_a), str(res_b), "--er-manifest", str(man_a), str(man_b),
                out="o3")
    assert res.returncode == 3 and "twice in the ER manifests" in res.stderr, res.stderr


# ---- dose, per model -------------------------------------------------------------------------------------------------

def test_dose_comes_from_the_f2_prompt_of_the_rows_own_model(tmp_path):
    """Every model has its own F2 prompt file; the dose of a row is read from its model's. Reading the first model's file
    for every row survived: the earlier tests gave --prompts-f2 to one model only."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    text = gold_text(ctl)
    line = {"record_id": r["id"], "partition": "test", "fmt": "F2", "model": "deepseek", "answer": None,
            "messages": [{"role": "user", "content": "Scenario\nProject knowledge:\n private Checkout $checkout,\n"}]}
    f2 = {"deepseek": w / "deepseek.F2.jsonl", "mistral": w / "mistral.F2.jsonl"}
    f2["deepseek"].write_text(json.dumps(line) + "\n")
    f2["mistral"].write_text(json.dumps({**line, "model": "mistral", "messages": [{"role": "user", "content": "none"}]}) + "\n")
    rows = [output(r, text, model=m) for m in ("deepseek", "mistral")]
    res = score(w, ctl, rows, "--prompts-f2", *(f"{m}={p}" for m, p in f2.items()))
    assert res.returncode == 0, res.stderr
    assert {x["model"]: x["dose"] for x in scores(w)} == {"deepseek": 1, "mistral": 0}


# ---- accepted missing lines: exactly the missing ones, each once -----------------------------------------------------

def test_accepted_missing_lists_every_missing_line_once(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    o = output(r, gold_text(ctl))
    keys = [{**o, "seed": s} for s in ("s11", "s23")]
    lines = [plan_line(k, "missing", attempts=attempts_of(["CRASHED", "CRASHED"])) for k in keys]
    entry = lambda line: {**{k: line[k] for k in ("model", "condition", "seed", "variant", "record_id", "fmt")},
                          "reason": "crashed twice", "receipt_sha256": line["receipt_sha256"]}
    accepted = w / "accepted.jsonl"
    for k, (entries, ok, why) in enumerate((([entry(lines[0]), entry(lines[1])], True, ""),
                                            ([entry(lines[0])], False, "not listed"),
                                            ([entry(lines[0]), entry(lines[1]), entry(lines[1])], False, "twice"))):
        accepted.write_text("".join(json.dumps(x) + "\n" for x in entries))
        gen = w / "gen.jsonl"
        gen.write_text(json.dumps(o) + "\n")
        res = run("scripts/score_run.py", *outputs_args(w, ctl, w / f"o{k}"), *plan(gen, extra=lines, status="FAIL"),
                  "--accepted-missing", str(accepted), "--accepted-missing-commit", "abc123", "--outputs", str(gen))
        assert (res.returncode == 0) == ok and (ok or (res.returncode == 3 and why in res.stderr)), (why, res.stderr)


# ---- what SCORES_META pins -------------------------------------------------------------------------------------------

def test_scores_meta_binds_the_tool_the_prompts_and_the_generation_files(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    o = output(r, gold_text(ctl))
    assert score(w, ctl, [o]).returncode == 0
    meta = json.loads((w / "o" / "SCORES_META.json").read_text())
    pm = json.loads((w / "PROMPTS_META.json").read_text())
    assert meta["prompts_v2tools_commit"] == pm["v2tools_commit"]
    assert str(w / "manifest.main.jsonl") in meta["inputs"] and str(w / "coverage.main.json") in meta["inputs"]
    cm = json.loads((ctl / "SCORES_META.json").read_text())      # a controls run of another tool commit is refused
    (ctl / "SCORES_META.json").write_text(json.dumps({**cm, "tool_commit": "0" * 40}))
    res = score(w, ctl, [o], out="o2")
    assert res.returncode == 3 and "tool commit" in res.stderr, res.stderr


def test_a3_errors_name_each_file_that_differs_and_a_missing_commit(tmp_path):
    sr = score_run_module()
    assert sr.a3_errors(tmp_path / "none", None) == ["PROMPTS_META.json has no v2tools_commit"]
    assert sr.a3_errors(tmp_path / "none", "") == ["PROMPTS_META.json has no v2tools_commit"]
    for i, f in enumerate(sr.A3_FILES):   # the whole list decides V(r): an edit of any one of the seven is refused, and named
        repo = tmp_path / f"a3_{i}"
        base = a3_repo(repo)
        assert sr.a3_errors(repo, base) == []
        (repo / f).write_text((repo / f).read_text() + "\n# edited\n")
        git("commit", "-q", "-am", f"edit {f}", cwd=repo)
        edited = git("rev-parse", "HEAD", cwd=repo)
        if f in sr.A3_ALLOWED:  # symcache.py is checked against its allowlist, not byte for byte
            assert sr.a3_errors(repo, edited) == [f"{f} at {edited} is not an allowed version (A3_ALLOWED)"], f
        else:
            assert [e.split(" differs")[0] for e in sr.a3_errors(repo, edited)] == [f], f


def test_code_sha256_and_tool_state_follow_the_scoring_code(tmp_path):
    """code_sha256 digests v2tools/*.py AND scripts/*.py; `dirty` is True once either is edited."""
    sr = score_run_module()
    (tmp_path / "v2tools").mkdir()
    (tmp_path / "scripts").mkdir()
    for f in sr.A3_FILES:
        (tmp_path / f).write_text("x = 1\n")
    (tmp_path / "scripts" / "s.py").write_text("y = 1\n")
    git("init", "-q", cwd=tmp_path)
    git("add", "-A", cwd=tmp_path)
    git("commit", "-q", "-m", "c", cwd=tmp_path)
    sr.ROOT = tmp_path
    clean = sr.tool_state()
    assert clean["dirty"] is False and clean["tool_commit"] == git("rev-parse", "HEAD", cwd=tmp_path)
    (tmp_path / "scripts" / "s.py").write_text("y = 2\n")
    edited = sr.tool_state()
    assert edited["dirty"] is True and edited["code_sha256"] != clean["code_sha256"]


# ---- determinism of future_methods.jsonl -----------------------------------------------------------------------------

def test_future_rows_are_sorted_whatever_the_hash_seed():
    """The set of future methods of a file is written sorted: SAP_LOCK hashes the file, and a set iterates in a
    per-process order. 12 names: the mutant that wrote `list(set)` agrees with `sorted` once in 479 million."""
    sr = score_run_module()
    names = [f"m{i:02d}" for i in range(12)][::-1]
    sr.future_methods = lambda p: {"a.php": {i: SimpleNamespace(origin="o", name=n) for i, n in enumerate(names)}}
    assert sr.future_rows({"rid": None}) == [{"record_id": "rid", "path": "a.php", "future": [f"o:{n}" for n in sorted(names)]}]


# ---- the refusals that the earlier tests reach for another reason, or not at all -------------------------------------

def test_missing_or_malformed_inputs_are_refused_with_exit_3_and_their_own_message(tmp_path):
    """test_score_run_fails_closed's "missing subsets" case drops --prompts-meta too (args[:-6]), so it is refused for the
    missing --prompts-meta and never reaches the subsets check. The same file never lets the manifest's own field,
    status, attempts and line_sha256 rules, the --manifest/--coverage pairing, a --controls run with a manifest, a ca_ref
    row or an ER verdicts file that is not an object speak for themselves."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    o = output(r, gold_text(ctl))
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    base = outputs_args(w, ctl, w / "o")
    sub = [a if a != str(w / "subsets.json") else str(w / "none.json") for a in base]
    res = run("scripts/score_run.py", *sub, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 3 and "missing input" in res.stderr and "none.json" in res.stderr, res.stderr
    key = {k: o[k] for k in ("model", "condition", "seed", "variant", "record_id")}
    manifest, coverage = plan(gen)[1], plan(gen)[3]
    res = run("scripts/score_run.py", *base, "--manifest", manifest, "--outputs", str(gen))
    assert res.returncode == 3 and "go together" in res.stderr, res.stderr
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls", "--out", str(w / "c2"), "--manifest", manifest, "--coverage", coverage)
    assert res.returncode == 3 and "belong to --outputs" in res.stderr, res.stderr
    bad = {"status": (lambda xs: [{**xs[0], "status": "bogus"}], "has status 'bogus'"),
           "attempts": (lambda xs: [{**xs[0], "attempts": []}], "has attempts"),
           "line_sha256": (lambda xs: [{**xs[0], "status": "missing", "line_sha256": "1" * 64, "attempts": [],
                                        "receipt_status": "absent", "receipt_sha256": None}], "has line_sha256"),
           "field": (lambda xs: [{k: v for k, v in xs[0].items() if k != "attempts"}], "missing ['attempts']")}
    for name, (edit, why) in bad.items():
        res = run("scripts/score_run.py", *base, *plan(gen, edit=edit), "--outputs", str(gen))
        assert res.returncode == 3 and why in res.stderr and "Traceback" not in res.stderr, (name, res.stderr)
    ref = (ctl / "ca_ref.jsonl").read_text()
    (ctl / "ca_ref.jsonl").write_text("")
    res = score(w, ctl, [o])
    assert res.returncode == 3 and "no ca_ref row" in res.stderr and "Traceback" not in res.stderr, res.stderr
    (ctl / "ca_ref.jsonl").write_text(ref)
    cond = "deepseek.s42.full_aware"
    line = json.dumps({"record_id": r["id"], "condition": cond})
    er = {"record_id": r["id"], "condition": cond, "er": 0, "er_status": "SKIP", "trace_action": None, "trace_then": None,
          "er_trace": None, "n_then_evaluable": 0, "overlay_sha256": hashlib.sha256(line.encode()).hexdigest()}
    res_f, man = er_run(w, "er", [er], line)
    (res_f.parent / "er_tamper_verdicts.json").write_text("[]")
    res = score(w, ctl, [o], "--er-results", str(res_f), "--er-manifest", str(man))
    assert res.returncode == 3 and "not a JSON object" in res.stderr and "Traceback" not in res.stderr, res.stderr


# ---- OPEN_RESOLUTION.json of an outputs run --------------------------------------------------------------------------

def test_the_outputs_run_writes_open_resolution_by_repo_model_seed_condition_variant(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    text = gold_text(ctl)
    rows = [output(r, text), output(r, text, model="mistral", seed="s11", condition="ft_only", fmt="F0")]
    assert score(w, ctl, rows).returncode == 0
    got = json.loads((w / "o" / "OPEN_RESOLUTION.json").read_text())
    assert got == {"Acme/Shop|deepseek|s42|full_aware|main": {"open": 0, "n": 1, "rate": 0.0},
                   "Acme/Shop|mistral|s11|ft_only|main": {"open": 0, "n": 1, "rate": 0.0}}, got


# ---- hallucinated of a DEV row uses the mask of the answer infrastructure --------------------------------------------

def test_a_dev_rows_hallucinated_is_read_with_the_mask(tmp_path):
    """describe.row(..., masked): computeTotal, written with the TEST answer, is masked in every dev prompt (the
    scorer reads V(r) exactly as prompt building did), so an output that calls it without defining it invents it.
    Passing frozenset() instead of Shown4.masked survived the earlier dev test, which asserts insert_status, bv,
    ca_defined and ca only (0 / 0 / 0 instead of 1 / 1 / 2)."""
    from tests.phase3c_fixture import GRAND, block, make_phase2_masked
    from v2tools.compose import gold_answer
    (tmp_path / "m").mkdir()
    w, dev, test = make_phase2_masked(tmp_path / "m")
    ctl = controls(w, partition="dev")
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    gold = gold_answer(dev, cat)
    helper = "private function computeTotal(string $total): string\n{\n    return $total . '';\n}"
    texts = {"deepseek": gold[:gold.rindex("`" * 3)] + helper + "\n" + "`" * 3,
             "mistral": block("#[Then('" + GRAND + "')]\npublic function grand(string $total): void\n{\n"
                              "    $this->cart->add($this->computeTotal($total));\n}"),
             "codellama": gold}
    o = {"record_id": dev["id"], "seed": "s42", "condition": "full_aware", "variant": "main", "prompt_tokens": 100,
         "new_tokens": 50, "max_new_tokens": 6144, "stopped_at_eos": True, "prompts_sha256": "a" * 64,
         "prompt_sha256": "b" * 64, "fmt": "F2"}
    gen = w / "gen.jsonl"
    gen.write_text("".join(json.dumps({**o, "model": m, "text": t}) + "\n" for m, t in texts.items()))
    args = [a if a != "test" else "dev" for a in outputs_args(w, ctl, w / "o")]   # --partition dev
    res = run("scripts/score_run.py", *args, *plan(gen), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    assert {x["model"]: x["hallucinated"] for x in scores(w)} == {"deepseek": 1, "mistral": 1, "codellama": 2}


# ---- the dose table is a mode of its own and names each model once; canonical keys -----------------------------------

def test_dose_table_refusals_and_canonical_row_keys(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    for name in ("controls.jsonl", "ca_ref.jsonl"):      # one JSON object per line, keys sorted: the bytes do not depend on
        first = json.loads((ctl / name).read_text().splitlines()[0])      # how a row happened to be built
        assert list(first) == sorted(first), name
    assert score(w, ctl, [output(r, gold_text(ctl))]).returncode == 0
    first = json.loads((w / "o" / "scores.jsonl").read_text().splitlines()[0])
    assert list(first) == sorted(first)
    f2 = w / "deepseek.F2.jsonl"
    f2.write_text(json.dumps({"record_id": r["id"], "partition": "test", "fmt": "F2", "model": "deepseek", "answer": None,
                              "messages": [{"role": "user", "content": "x"}]}) + "\n")
    excluded = w / "excluded.jsonl"
    excluded.write_text("")
    base = [*common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--compose-excluded", str(excluded),
            "--dose-table", "--out", str(w / "dose")]
    res = run("scripts/score_run.py", *base, "--prompts-f2", f"deepseek={f2}", f"deepseek={f2}")
    assert res.returncode == 3 and "each model once" in res.stderr, res.stderr
    res = run("scripts/score_run.py", *base, "--prompts-f2", f"deepseek={f2}", "--controls")
    assert res.returncode == 3 and "mode of its own" in res.stderr, res.stderr


# ---- the exit-3 wrapping (the repo named once, no traceback) and SHA256SUMS ------------------------------------------

def test_a_catalog_that_disagrees_with_head_is_refused_with_the_repo_named_once(tmp_path):
    """A catalog row with no annotation (phantoms raises ValueError) stops score_run with exit 3, no traceback, and
    `Acme/Shop` appears once in the line (phantoms' own ' (repo …)' is stripped). Without this test, the wrapper and
    the strip could be removed with no earlier test turning red."""
    w, r = make_phase2(tmp_path)
    cat = [json.loads(l) for l in (w / "phase2" / "catalog.jsonl").read_text().splitlines()]
    (w / "phase2" / "catalog.jsonl").write_text("".join(json.dumps({**f, "annotations": []} if f["fn_id"] == r["create"][0]["fn_id"]
                                                                  else f) + "\n" for f in cat))
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--controls",
              "--out", str(w / "c"))
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr
    assert res.stderr.startswith("refusing: Acme/Shop: ") and res.stderr.count("Acme/Shop") == 1, res.stderr


def test_sha256sums_lists_every_file_written_with_its_digest(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    got = dict(reversed(l.split("  ")) for l in (ctl / "SHA256SUMS").read_text().splitlines())
    want = {"CA_PREC_EXCLUDED.json", "CONTROLS.json", "OPEN_RESOLUTION.json", "SCORES_META.json", "ca_ref.jsonl",
            "control_outputs.jsonl", "controls.jsonl", "exclusions.jsonl", "future_methods.jsonl"}
    assert set(got) == want and all(hashlib.sha256((ctl / n).read_bytes()).hexdigest() == d for n, d in got.items())


# ----- fail-closed fixes of score_run.py (added 04/10/2026), each test red on ab949af ---------------------------------

def att(*statuses):
    return [{"file": f"f{i}", "status": s, "sha256": f"{i}" * 64} for i, s in enumerate(statuses)]


def test_evidence_counts_the_attempts_that_read_as_a_receipt_as_generation_evidence_eligible_does():
    """The generation run's gencoverage.evidence_eligible: the last receipt CRASHED or INCOMPLETE and >= 2
    attempts that each read as a receipt (a status). An attempt whose JSON is torn has a sha256 and no status: with
    [torn, CRASHED] the writer says False and score_run said True, so "evidence_eligible False disagrees with the
    attempts" refused the whole manifest (reproduced on the real writer: 59 missing lines). Over all 781 status
    sequences of length <= 4 the two now agree."""
    ev = score_run_module().evidence
    assert [ev({"attempts": a}) for a in (att(None, "CRASHED"), att(None, "CRASHED", "CRASHED"), att("INCOMPLETE", "CRASHED"),
                                           att("CRASHED"), att("CRASHED", "REFUSED_PREFLIGHT"), att("CRASHED", None))] == [
        False, True, True, False, False, False]


def test_a_run_with_nothing_to_do_is_refused(tmp_path):
    """No --controls, --outputs <files> or --dose-table (a glob that expanded to nothing) wrote a META with
    mode "outputs" and exit 0."""
    w, r = make_phase2(tmp_path)
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--out",
              str(w / "x"))
    assert res.returncode == 3 and "nothing to do" in res.stderr and not (w / "x" / "SCORES_META.json").exists(), res.stderr
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test", "--out",
              str(w / "x"), "--outputs")
    assert res.returncode == 3 and "nothing to do" in res.stderr and not (w / "x" / "SCORES_META.json").exists(), res.stderr


def test_the_controls_run_must_come_from_the_inputs_of_this_run(tmp_path):
    """The controls run was matched to this run on records, labels and catalog only; the scoring also reads
    forbidden.json, the pairs, the source config (styles) and symtimes. A controls run built with other ones was
    accepted (exit 0, reproduced with another sources.json); now all seven shared inputs are compared."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    o = output(r, gold_text(ctl))
    assert score(w, ctl, [o]).returncode == 0
    src = json.loads((w / "sources.json").read_text())
    src[0]["styles"] = ["php_docblock"]
    (w / "sources.json").write_text(json.dumps(src))
    res = score(w, ctl, [o], out="o2")
    assert res.returncode == 3 and "was not built from this sources.json" in res.stderr, res.stderr


def test_an_existing_scoring_run_is_never_written_over(tmp_path):
    """Re-running a scoring run's controls command overwrote a sealed controls run, and an outputs run that stopped
    left the previous scores.jsonl in place. An --out that already holds a SCORES_META.json is refused (exit 3), before
    anything is read or written; its files keep their bytes."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    before = {f.name: f.read_bytes() for f in ctl.iterdir()}
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls", "--out", str(ctl))
    assert res.returncode == 3 and "already holds a scoring run" in res.stderr, res.stderr
    assert {f.name: f.read_bytes() for f in ctl.iterdir()} == before
    o = output(r, gold_text(ctl))
    assert score(w, ctl, [o]).returncode == 0
    first = (w / "o" / "scores.jsonl").read_bytes()
    res = score(w, ctl, [{**o, "text": "no code"}])
    assert res.returncode == 3 and "already holds a scoring run" in res.stderr, res.stderr
    assert (w / "o" / "scores.jsonl").read_bytes() == first


def test_malformed_inputs_are_refusals_never_tracebacks(tmp_path):
    """Every malformed (not missing) input of 14 cases, plus the ER manifest ones, ended in a traceback (exit 1); each
    is now a refusal (exit 3, "refusing:") that names what is wrong, and nothing is written."""
    import shutil
    (tmp_path / "w").mkdir()
    w, r = make_phase2(tmp_path / "w")
    ctl = controls(w)
    o = output(r, gold_text(ctl))
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps(o) + "\n")
    n = iter(range(99))

    def refused(res, why):
        assert res.returncode == 3 and res.stderr.startswith("refusing: ") and why in res.stderr, res.stderr
        assert "Traceback" not in res.stderr, res.stderr

    def edited(rel, text, mode="controls"):
        w2 = tmp_path / f"w{next(n)}"
        shutil.copytree(w, w2, symlinks=True)
        (w2 / rel).write_text(text)
        if mode == "controls":
            return run("scripts/score_run.py", *common(w2), "--symtimes", str(w2 / "symtimes.jsonl"), "--partition", "test",
                       "--controls", "--out", str(w2 / "c2"))
        args = [a.replace(str(w), str(w2)) for a in outputs_args(w, ctl, w / "o_new")]
        return run("scripts/score_run.py", *args, *plan(w2 / "gen.jsonl"), "--outputs", str(w2 / "gen.jsonl"))
    refused(edited("phase2/forbidden.json", "{"), "forbidden.json")
    refused(edited("sources.json", "["), "sources.json")
    refused(edited("phase2/labels.jsonl", ""), "labels.jsonl")
    refused(edited("sources.json", "[]"), "sources.json")
    refused(edited("phase2/records/test.jsonl", "[1]\n"), "not a JSON object")
    refused(edited("ctl/SCORES_META.json", "{", "outputs"), "SCORES_META.json")
    refused(edited("ctl/SCORES_META.json", json.dumps({"code_sha256": "x"}), "outputs"), "SCORES_META.json")
    refused(edited("PROMPTS_META.json", "{", "outputs"), "PROMPTS_META.json")
    refused(edited("subsets.json", "{", "outputs"), "subsets.json")
    refused(edited("self_contained.jsonl", json.dumps({"record_id": r["id"]}) + "\n", "outputs"), "self_contained")

    def scored(rows, *extra):
        g = w / f"gen{next(n)}.jsonl"
        g.write_text("".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in rows))
        return run("scripts/score_run.py", *outputs_args(w, ctl, w / f"o{next(n)}"), *plan(gen), *extra,
                   "--outputs", str(g))
    refused(scored(["[1]"]), "not a JSON object")
    for field, value in (("new_tokens", "x"), ("text", 5), ("stopped_at_eos", "yes"), ("prompt_tokens", True)):
        g = w / f"gen{next(n)}.jsonl"
        g.write_text(json.dumps({**o, field: value}) + "\n")
        res = run("scripts/score_run.py", *outputs_args(w, ctl, w / f"o{next(n)}"), *plan(g), "--outputs", str(g))
        refused(res, field)
    refused(scored([o], "--prompts-f2", "deepseek"), "--prompts-f2")
    refused(scored([o], "--prompts-f2", "deepseek=/nonexistent.jsonl"), "missing input")
    (w / "er.jsonl").write_text("")
    refused(scored([o], "--er-results", str(w / "er.jsonl"), "--er-manifest", str(w / "nope.jsonl")), "nope.jsonl")
    (w / "mbad.jsonl").write_text("not json\n")
    refused(scored([o], "--er-results", str(w / "er.jsonl"), "--er-manifest", str(w / "mbad.jsonl")), "mbad.jsonl")
    (w / "mbad2.jsonl").write_text("{}\n")
    refused(scored([o], "--er-results", str(w / "er.jsonl"), "--er-manifest", str(w / "mbad2.jsonl")), "record_id")
    assert not any((d / "scores.jsonl").exists() for d in w.glob("o*") if d.is_dir())


def test_unparsable_or_missing_json_inputs_are_refusals(tmp_path):
    """A JSON input that does not parse is exit 3, not a traceback."""
    w, r = make_phase2(tmp_path)
    (w / "phase2" / "forbidden.json").write_text("{")
    res = run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
              "--controls", "--out", str(w / "c"))
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr


def test_require_clean_refuses_a_scoring_tree_that_is_not_a_clean_commit(tmp_path):
    """`dirty` and `tool_commit` were only recorded; the scoring runs of record now run with --require-clean, which
    refuses (exit 3, nothing written) a tree outside git or with uncommitted changes under v2tools/ or scripts/."""
    import shutil
    (tmp_path / "w").mkdir()
    w, r = make_phase2(tmp_path / "w")
    t = tmp_path / "tree"
    for d in ("v2tools", "scripts"):
        shutil.copytree(ROOT / d, t / d, ignore=shutil.ignore_patterns("__pycache__"))
    (t / ".gitignore").write_text("__pycache__/\n")
    args = [str(t / "scripts" / "score_run.py"), *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition",
            "test", "--controls", "--require-clean"]
    res = run(*args, "--out", str(w / "c0"))  # not a git checkout
    assert res.returncode == 3 and "not a git checkout" in res.stderr and not (w / "c0").exists(), res.stderr
    git("init", "-q", cwd=t)
    git("add", "-A", cwd=t)
    git("commit", "-q", "-m", "tree", cwd=t)
    res = run(*args, "--out", str(w / "c1"))
    assert res.returncode == 0, res.stderr
    assert json.loads((w / "c1" / "SCORES_META.json").read_text())["dirty"] is False
    (t / "scripts" / "score_run.py").write_text((t / "scripts" / "score_run.py").read_text() + "\n# edited\n")
    res = run(*args, "--out", str(w / "c2"))
    assert res.returncode == 3 and "uncommitted changes" in res.stderr and not (w / "c2").exists(), res.stderr


def test_the_f2_prompts_of_an_outputs_run_are_checked_as_the_dose_table_checks_them(tmp_path):
    """The outputs mode read --prompts-f2 with none of dose_rows' checks: a dev / F0 / mistral line given as
    DeepSeek's F2 file scored a dose of 1 on a test row (exit 0). Now a line of another partition, format or model, of a
    record outside the partition, a record twice or a model twice is refused; the right file still scores its dose."""
    w, r = make_phase2(tmp_path)
    ctl = controls(w)
    o = output(r, gold_text(ctl))
    line = {"record_id": r["id"], "partition": "test", "fmt": "F2", "model": "deepseek", "answer": None,
            "messages": [{"role": "user", "content": "Scenario\nProject knowledge:\n private Checkout $checkout,\n"}]}
    f2 = w / "f2.jsonl"
    n = iter(range(99))
    for bad, why in (({**line, "partition": "dev"}, "not test/F2/deepseek"), ({**line, "fmt": "F0"}, "not test/F2/deepseek"),
                     ({**line, "model": "mistral"}, "not test/F2/deepseek"),
                     ({**line, "record_id": "Acme/Shop|features/none.feature|1|s"}, "not in partition test")):
        f2.write_text(json.dumps(bad) + "\n")
        res = score(w, ctl, [o], "--prompts-f2", f"deepseek={f2}", out=f"o{next(n)}")
        assert res.returncode == 3 and why in res.stderr and "Traceback" not in res.stderr, (bad, res.stderr)
    f2.write_text(json.dumps(line) + "\n" + json.dumps(line) + "\n")
    res = score(w, ctl, [o], "--prompts-f2", f"deepseek={f2}", out=f"o{next(n)}")
    assert res.returncode == 3 and "twice" in res.stderr, res.stderr
    f2.write_text(json.dumps(line) + "\n")
    res = score(w, ctl, [o], "--prompts-f2", f"deepseek={f2}", f"deepseek={f2}", out=f"o{next(n)}")
    assert res.returncode == 3 and "each model once" in res.stderr, res.stderr
    out = f"o{next(n)}"
    assert score(w, ctl, [o], "--prompts-f2", f"deepseek={f2}", out=out).returncode == 0
    assert [x["dose"] for x in scores(w, out)] == [1]
