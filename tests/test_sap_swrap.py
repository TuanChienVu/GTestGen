"""S-wrap (fixed 05/10/2026, before the SAP lock): the descriptive sensitivity of RQ1 to outputs wrapped in a
namespace/class shell. sap_swrap unwrap (one fixed rule, inputs checked against the generation run's manifest) and
sap_swrap report (the SAP's own denominators, pairing and cluster bootstrap, no p-value; accepted-missing pairs filled
as sap_run fills them (P4), sap_run's bootstrap seeds, and the S-wrap scores tied to the official scoring's code and
controls)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tests.test_sap import CFG, rows
from v2tools.sap import (check_complete, compare, compose_dropped, confirmatory, design_records, locked_flags,
                         per_seed_pairs, pooled, seed_values, usable, usable_records)

ROOT = Path(__file__).resolve().parents[1]
TICKS = "`" * 3
_spec = importlib.util.spec_from_file_location("sap_swrap", ROOT / "scripts" / "sap_swrap.py")
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

PHP_CLASS = ("<?php\n\nnamespace App\\Ctx;\n\nuse Behat\\Behat\\Context\\Context;\nuse App\\Svc;\n\n"
             "class A implements Context\n{\n    /** @When x */\n    public function x(): void\n    {\n"
             "        $this->s->go();\n    }\n}")
PHP_BODY = ("use Behat\\Behat\\Context\\Context;\nuse App\\Svc;\n\n    /** @When x */\n    public function x(): void\n"
            "    {\n        $this->s->go();\n    }")


def fence(path: str, code: str, info: str = "php") -> str:
    return f"{TICKS}{info}\n// FILE: {path}\n{code}\n{TICKS}\n"


def test_a_php_namespace_use_and_one_class_keep_the_imports_and_the_members():
    text = "Here:\n\n" + fence("src/A.php", PHP_CLASS) + "Done."
    new, wrapped, unwrapped = sw.unwrap_text(text, "php", ["src/A.php"])
    assert new == "Here:\n\n" + fence("src/A.php", PHP_BODY) + "Done."
    assert (wrapped, unwrapped) == (1, 1)


def test_a_java_package_imports_and_one_class():
    code = ("package org.x;\n\nimport io.cucumber.java.en.When;\n\npublic class S {\n    @When(\"^x$\")\n"
            "    public void x() {\n        api.go();\n    }\n}")
    new, wrapped, unwrapped = sw.unwrap_text(fence("src/S.java", code, "java"), "java", ["src/S.java"])
    assert new == fence("src/S.java", "import io.cucumber.java.en.When;\n\n    @When(\"^x$\")\n    public void x() {\n"
                                      "        api.go();\n    }", "java")
    assert (wrapped, unwrapped) == (1, 1)


def test_what_the_rule_leaves_alone():
    methods = "    /** @When x */\n    public function x(): void\n    {\n    }"
    two = PHP_CLASS + "\n\nclass B\n{\n}"
    broken = PHP_CLASS.replace("$this->s->go();", "$this-s = 1;")  # a syntax error inside the class body
    loose = PHP_CLASS + "\n\nfunction helper() {}"  # something beside the one type
    for code, counts in ((methods, (0, 0)), (two, (1, 0)), (broken, (1, 0)), (loose, (1, 0))):
        text = fence("src/A.php", code)
        assert sw.unwrap_text(text, "php", ["src/A.php"]) == (text, *counts), code
    text = fence("src/Other.php", PHP_CLASS)  # not a target: never read
    assert sw.unwrap_text(text, "php", ["src/A.php"]) == (text, 0, 0)
    text = fence("features/step_definitions/a.rb", "class A\nend", "ruby")  # Ruby (and C#): not in the rule
    assert sw.unwrap_text(text, "ruby", ["features/step_definitions/a.rb"]) == (text, 0, 0)
    assert sw.unwrap_text("no fence at all\nclass A {}", "php", ["src/A.php"]) == ("no fence at all\nclass A {}", 0, 0)
    text = f"{TICKS}php\nclass A {{}}\n// FILE: src/A.php\n{PHP_CLASS}\n{TICKS}\n"  # code before the FILE line
    assert sw.unwrap_text(text, "php", ["src/A.php"]) == (text, 0, 0)


def test_each_file_of_a_multi_file_fence_on_its_own_and_the_reopen_line_stays():
    b_class = PHP_CLASS.replace("class A", "class B").replace("<?php\n\n", "")
    b_body = PHP_BODY
    text = (f"{TICKS}php\n// FILE: src/A.php\n{PHP_CLASS}\n{TICKS}php\n// FILE: src/B.php\n{b_class}\n"
            f"// FILE: ./src/C.php\n{PHP_CLASS.replace('$this->s->go();', '$this-s = 1;')}\n{TICKS}\n")
    new, wrapped, unwrapped = sw.unwrap_text(text, "php", ["src/A.php", "src/B.php", "src/C.php"])
    assert new == (f"{TICKS}php\n// FILE: src/A.php\n{PHP_BODY}\n{TICKS}php\n// FILE: src/B.php\n{b_body}\n"
                   f"// FILE: ./src/C.php\n{PHP_CLASS.replace('$this->s->go();', '$this-s = 1;')}\n{TICKS}\n")
    assert (wrapped, unwrapped) == (3, 2)


def test_a_php_tag_line_before_the_file_line_and_an_unclosed_fence():
    text = f"{TICKS}php\n<?php\n// FILE: src/A.php\n{PHP_CLASS}\n"  # design specification §1.4; no closing fence
    new, wrapped, unwrapped = sw.unwrap_text(text, "php", ["src/A.php"])
    assert new == f"{TICKS}php\n<?php\n// FILE: src/A.php\n{PHP_BODY}\n" and (wrapped, unwrapped) == (1, 1)


def test_a_trait_and_the_comments_the_rule_allows():
    """A trait is one of the types; a PHP comment or docblock at the head of the file, and Java line and block
    comments, are allowed nodes (dropped with the shell)."""
    trait = "<?php\nnamespace App;\n\ntrait T\n{\n    public function x(): void {}\n}"
    new, wrapped, unwrapped = sw.unwrap_text(fence("src/T.php", trait), "php", ["src/T.php"])
    assert new == fence("src/T.php", "    public function x(): void {}") and (wrapped, unwrapped) == (1, 1)
    head = "<?php\n/**\n * The context.\n */\n// generated\nnamespace App;\nuse App\\Svc;\n\nclass A\n{\n    public function x(): void {}\n}"
    new, wrapped, unwrapped = sw.unwrap_text(fence("src/A.php", head), "php", ["src/A.php"])
    assert new == fence("src/A.php", "use App\\Svc;\n\n    public function x(): void {}") and (wrapped, unwrapped) == (1, 1)
    java = ("// header\n/* block */\npackage org.x;\n\nimport a.B;\n\n/** doc */\npublic class S {\n    public void x() {\n"
            "    }\n}")
    new, wrapped, unwrapped = sw.unwrap_text(fence("src/S.java", java, "java"), "java", ["src/S.java"])
    assert new == fence("src/S.java", "import a.B;\n\n    public void x() {\n    }", "java") and (wrapped, unwrapped) == (1, 1)


def test_a_braced_php_namespace():
    code = "namespace App {\n    use App\\Svc;\n    class A\n    {\n        public function x(): void {}\n    }\n}"
    new, wrapped, unwrapped = sw.unwrap_text(fence("src/A.php", code), "php", ["src/A.php"])
    assert new == fence("src/A.php", "use App\\Svc;\n\n        public function x(): void {}")
    assert (wrapped, unwrapped) == (1, 1)


# ----- the CLI ---------------------------------------------------------------------------------------------------------

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gen(tmp: Path, bad_line: bool = False, first_status: str = "ok") -> dict:
    """Three generation workdirs as the generation run writes them (DeepSeek base: zero_shot and rag_only; DeepSeek s42:
    full_aware; CodeLlama base: zero_shot, not the anchor model), the manifest
    with each line's line_sha256 and its file's outputs_sha256, the coverage that hashes it, and the records."""
    recs = [{"id": f"R|f{i}.feature|1|a", "repo": "R", "language": "php", "partition": "test",
             "create": [{"path": f"src/C{i}.php"}]} for i in range(3)]
    (tmp / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    files, lines = {}, []
    for wd, conds, seed, model in (("gen-base", ("zero_shot", "rag_only"), "base", "deepseek"),
                                   ("gen-s42", ("full_aware",), "s42", "deepseek"), ("gen-cl", ("zero_shot",), "base", "codellama")):
        out = []
        for c in conds:
            for i, r in enumerate(recs):
                code = PHP_CLASS.replace("class A", f"class C{i}") if (c == "zero_shot" or i == 0) else PHP_BODY
                out.append({"record_id": r["id"], "model": model, "seed": seed, "condition": c, "variant": "main",
                            "fmt": {"zero_shot": "F0", "rag_only": "F1", "full_aware": "F2"}[c],
                            "text": fence(f"src/C{i}.php", code), "prompt_tokens": 10, "new_tokens": 5,
                            "max_new_tokens": 9, "stopped_at_eos": True, "prompts_sha256": "p", "prompt_sha256": "q"})
        (tmp / wd).mkdir()
        f = tmp / wd / "outputs.jsonl"
        f.write_text("".join(json.dumps(o) + "\n" for o in out))
        files[wd] = f
        for o, raw in zip(out, f.read_bytes().split(b"\n")):
            lines.append({**{k: o[k] for k in ("model", "condition", "seed", "variant", "record_id", "fmt")},
                          "status": first_status if not lines else "ok", "outputs_sha256": sha(f),
                          "line_sha256": hashlib.sha256(raw + (b"x" if bad_line else b"")).hexdigest()})
    (tmp / "manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (tmp / "coverage.json").write_text(json.dumps({"status": "PASS", "manifest_sha256": sha(tmp / "manifest.jsonl")}))
    return files


def cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "sap_swrap.py"), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True)


def unwrap_args(tmp: Path, files: dict, out: Path) -> list:
    return ["unwrap", "--config", ROOT / "config" / "sap.json", "--outputs", *files.values(), "--manifest",
            tmp / "manifest.jsonl", "--coverage", tmp / "coverage.json", "--records", tmp / "records.jsonl", "--out", out]


def test_unwrap_writes_the_anchor_base_rows_as_probes_and_the_wrap_rate_of_every_run(tmp_path):
    files = gen(tmp_path)
    r = cli(*unwrap_args(tmp_path, files, tmp_path / "sw"))
    assert r.returncode == 0, r.stderr
    out = [json.loads(l) for l in (tmp_path / "sw" / "swrap_outputs.jsonl").read_text().split("\n") if l]
    assert {(o["condition"], o["model"], o["swrap_model"]) for o in out} == {("zero_shot", "probe", "deepseek"),
                                                                            ("rag_only", "probe", "deepseek")}
    assert len(out) == 6 and all(o["text"].count("class ") == 0 for o in out if o["condition"] == "zero_shot")
    meta = json.loads((tmp_path / "sw" / "SWRAP_META.json").read_text())
    assert meta["name"] == "S-wrap" and meta["descriptive_only"] is True
    assert meta["script_sha256"] == sha(ROOT / "scripts" / "sap_swrap.py")
    assert meta["sources"] == {str(f): sha(f) for f in files.values()}
    assert meta["out_sha256"] == sha(tmp_path / "sw" / "swrap_outputs.jsonl")
    assert meta["wrap"] == {"deepseek|zero_shot|R": {"rows": 3, "wrapped": 3, "unwrapped": 3},
                            "deepseek|rag_only|R": {"rows": 3, "wrapped": 1, "unwrapped": 1},
                            "deepseek|full_aware|R": {"rows": 3, "wrapped": 1, "unwrapped": 1},
                            "codellama|zero_shot|R": {"rows": 3, "wrapped": 3, "unwrapped": 3}}
    assert sorted(meta["unwrapped"]) == sorted([[f"R|f{i}.feature|1|a", "zero_shot", "base", "main"] for i in range(3)]
                                               + [["R|f0.feature|1|a", "rag_only", "base", "main"]])


def test_unwrap_reads_only_outputs_that_pass_the_generation_contract(tmp_path):
    files = gen(tmp_path)
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(files["gen-base"].read_bytes().replace(b"Svc", b"Xvc"))
    r = cli(*unwrap_args(tmp_path, {**files, "gen-base": tampered}, tmp_path / "a"))
    assert r.returncode == 3 and "not the outputs_sha256 of its manifest lines" in r.stderr, r.stderr
    (tmp_path / "coverage.json").write_text(json.dumps({"status": "PASS", "manifest_sha256": "0" * 64}))
    r = cli(*unwrap_args(tmp_path, files, tmp_path / "b"))
    assert r.returncode == 3 and "manifest_sha256" in r.stderr
    (tmp_path / "coverage.json").write_text(json.dumps({"status": "FAIL", "status_except_missing": "FAIL",
                                                        "manifest_sha256": sha(tmp_path / "manifest.jsonl")}))
    r = cli(*unwrap_args(tmp_path, files, tmp_path / "c"))
    assert r.returncode == 3 and "not complete" in r.stderr
    sub = tmp_path / "bad"
    sub.mkdir()
    r = cli(*unwrap_args(sub, gen(sub, bad_line=True), tmp_path / "d"))
    assert r.returncode == 3 and "line_sha256" in r.stderr
    sub = tmp_path / "excl"
    sub.mkdir()
    r = cli(*unwrap_args(sub, gen(sub, first_status="excluded"), tmp_path / "f"))  # a row of an excluded pair
    assert r.returncode == 3 and "is not an ok line" in r.stderr, r.stderr
    (tmp_path / "e").mkdir()
    r = cli(*unwrap_args(tmp_path, files, tmp_path / "e"))  # an --out that exists: refused before anything is read
    assert r.returncode == 3 and "never writes over" in r.stderr
    assert not any((tmp_path / x).exists() for x in "abcdf")


# ----- report ----------------------------------------------------------------------------------------------------------

CFG_ONE = {**CFG, "reps_bootstrap": 200, "reps_permutation": 199, "seeds": {"deepseek": CFG["seeds"]["deepseek"]}}
CTL_FILES = ("ca_ref.jsonl", "exclusions.jsonl", "future_methods.jsonl")


def write_rows(path: Path, xs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(x) + "\n" for x in xs))
    return path


def write_meta(scores: Path, ctl: Path, extra: dict | None = None, **kw) -> None:
    """The SCORES_META.json score_run writes next to a scores.jsonl:
    its sha256, code, 3A files and the controls read."""
    meta = {"partition": "test", "code_sha256": "c" * 64, "a3_sha256": {"v2tools/state.py": "a" * 64},
            "scores_sha256": sha(scores), "accepted_missing_sha256": None,
            "inputs": {**{str(ctl / n): sha(ctl / n) for n in CTL_FILES}, **(extra or {})}, **kw}
    (scores.parent / "SCORES_META.json").write_text(json.dumps(meta))


def report_inputs(tmp: Path, flip: bool = True, gone: list[dict] = (), accept: bool = True):
    """The test score rows of the SAP fixture (DeepSeek only), minus `gone` (listed in an accepted_missing file when
    `accept`); the S-wrap scores of the anchor's base runs (3C's probe mode: model "probe"), where the unwrapped
    records score CA = BV = 1; and the other inputs of report, with the SCORES_META.json of both scorings."""
    test_rows = rows({"full_aware": 0.3}, seed=7)
    ids = sorted({r["record_id"] for r in test_rows})
    official = [r for r in test_rows if r not in gone]
    unwrapped = [[r["record_id"], r["condition"], r["seed"], r["variant"]] for r in official
                 if r["condition"] in ("zero_shot", "rag_only") and r["record_id"] in ids[::2]]
    keys = {tuple(k) for k in unwrapped}
    swrap = [{**r, "model": "probe", **({"ca": 1, "bv": 1} if flip and (r["record_id"], r["condition"], r["seed"],
                                                                         r["variant"]) in keys else {})}
             for r in official if r["condition"] in ("zero_shot", "rag_only")]
    ctl = tmp / "controls_test"
    ctl.mkdir()
    controls = [{"record_id": i, "control": "gold", "ca_defined": True} for i in ids]
    exclusions = [{"record_id": ids[1], "reason": "ca_undefined"}]
    write_rows(ctl / "controls.jsonl", controls)
    write_rows(ctl / "exclusions.jsonl", exclusions)
    write_rows(ctl / "ca_ref.jsonl", [{"record_id": i} for i in ids])
    write_rows(ctl / "future_methods.jsonl", [])
    compose = [{"record_id": ids[3], "model": "deepseek"}]
    write_rows(tmp / "compose.jsonl", compose)
    (tmp / "official").mkdir()
    scores = write_rows(tmp / "official" / "scores.jsonl", official)
    am = None
    if gone and accept:
        am = write_rows(tmp / "accepted_missing.jsonl", [{"record_id": r["record_id"], "model": r["model"],
                                                         "condition": r["condition"], "variant": "main",
                                                         "seed": r["seed"]} for r in gone])
    write_meta(scores, ctl, accepted_missing_sha256=sha(am) if am else None)
    (tmp / "sw").mkdir()
    (tmp / "sw" / "swrap_outputs.jsonl").write_text("{}\n")
    out_sha = sha(tmp / "sw" / "swrap_outputs.jsonl")
    (tmp / "sw" / "SWRAP_META.json").write_text(json.dumps({"name": "S-wrap", "out_sha256": out_sha,
                                                            "unwrapped": unwrapped, "wrap": {"x": 1}}))
    (tmp / "swscore").mkdir()
    put_swrap(tmp, swrap)
    (tmp / "cfg.json").write_text(json.dumps(CFG_ONE))
    args = ["report", "--config", tmp / "cfg.json", "--scores", scores, "--swrap-scores", tmp / "swscore" / "scores.jsonl",
            "--swrap-meta", tmp / "sw" / "SWRAP_META.json", "--controls-dir", ctl, "--compose-excluded",
            tmp / "compose.jsonl", "--out", tmp / "SWRAP.json"] + (["--accepted-missing", am] if am else [])
    return args, official, swrap, {"controls": controls, "exclusions": exclusions, "compose": compose,
                                   "accepted": {(r["record_id"], r["model"], r["condition"], r["seed"]) for r in gone}}


def put_swrap(tmp: Path, xs: list[dict], **kw) -> None:
    """The S-wrap scores and their SCORES_META.json (same code, 3A files and controls as the official scoring)."""
    f = write_rows(tmp / "swscore" / "scores.jsonl", xs)
    write_meta(f, tmp / "controls_test", {str(tmp / "sw" / "swrap_outputs.jsonl"): sha(tmp / "sw" / "swrap_outputs.jsonl")},
               **kw)


def sap_run_rq1(official: list[dict], ctx: dict) -> dict:
    """What sap_run reports for RQ1 (CA) and for BV beside it: check_complete (P4, which fills accepted-missing pairs),
    then confirmatory."""
    filled, _ = check_complete(official, {"deepseek": design_records(ctx["controls"], ctx["compose"], "deepseek")},
                               CFG_ONE, ctx["accepted"], locked_flags(ctx["controls"], ctx["exclusions"]))
    keep = usable_records(ctx["controls"], ctx["exclusions"], ctx["compose"], "deepseek")
    conf = confirmatory(filled, CFG_ONE, {r["record_id"] for r in filled} - keep, {},
                        compose_dropped(ctx["compose"], "deepseek"))
    return {b: {"ca": conf["RQ1"]["comparisons"][f"full_aware_vs_{b}"]["ca"],
                "bv": conf["RQ1"]["comparisons"][f"full_aware_vs_{b}"]["descriptive"]["bv"]}
            for b in ("zero_shot", "rag_only")}


def test_report_puts_the_unwrapped_base_runs_beside_rq1_with_the_sap_procedure(tmp_path):
    args, official, swrap, ctx = report_inputs(tmp_path)
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "SWRAP.json").read_text())
    assert res["name"] == "S-wrap" and res["descriptive_only"] is True and res["wrap"] == {"x": 1}
    sap = sap_run_rq1(official, ctx)
    ids = sorted({x["record_id"] for x in official})
    excluded, dropped = {ids[1], ids[3]}, {ids[3]}
    after = [x for x in official if x["condition"] not in ("zero_shot", "rag_only")] + [{**x, "model": "deepseek"} for x in swrap]
    for i, base in enumerate(("zero_shot", "rag_only")):  # their index in families.RQ1
        for m, off in (("ca", 0), ("bv", 50)):
            got = res["comparisons"][f"RQ1:full_aware_vs_{base}"][m]
            # the official column IS sap_run's RQ1 (CA) and BV beside it, interval included
            assert (got["official"]["estimate"], got["official"]["ci95"], got["official"]["n_scenarios"]) == \
                (sap[base][m]["estimate"], sap[base][m]["ci95"], sap[base][m]["n_scenarios"])
            sv = seed_values(after, m, usable(m, excluded, dropped=dropped))
            want = compare(pooled(per_seed_pairs(sv, ("deepseek", "full_aware", "main"), ("deepseek", base, "main"))),
                           200, 0, CFG_ONE["seed"] + 100 * i + off, with_p=False)  # the same seed as the official
            assert got["swrap"]["estimate"] == want.estimate and got["swrap"]["ci95"] == list(want.ci)
            assert got["swrap"]["n_scenarios"] == want.n_scenarios and "p" not in got["swrap"]
            assert got["swrap"]["estimate"] < got["official"]["estimate"]  # the baseline gained: the gap shrinks
        keep = usable("ca", excluded, dropped=dropped)
        assert res["rates"][base]["ca"]["official"] == [sum(x["ca"] for x in official if x["condition"] == base and keep(x)),
                                                        sum(1 for x in official if x["condition"] == base and keep(x))]
    assert "full_aware_vs_ft_only" not in json.dumps(res["comparisons"])


def test_report_fills_the_accepted_missing_pairs_as_sap_run_does(tmp_path):
    """19 accepted-missing pairs of the anchor's RQ1 runs (zero_shot base and full_aware s42) score CA = BV = 0 in both
    columns (check_complete, P4); the official column is then sap_run's RQ1 (CA 0.3026, BV 0.2983 on this fixture),
    where it was 0.2809 and 0.2765 without the filling."""
    rs = rows({"full_aware": 0.3}, seed=7)
    ids = sorted({r["record_id"] for r in rs})
    gone = [r for r in rs if r["condition"] == "zero_shot" and r["record_id"] in ids[4:40:3]] + \
           [r for r in rs if r["condition"] == "full_aware" and r["seed"] == "s42" and r["record_id"] in ids[5:30:4]]
    assert len(gone) == 19
    args, official, swrap, ctx = report_inputs(tmp_path, flip=False, gone=gone)
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "SWRAP.json").read_text())
    sap = sap_run_rq1(official, ctx)
    got = res["comparisons"]["RQ1:full_aware_vs_zero_shot"]
    assert (round(got["ca"]["official"]["estimate"], 4), round(got["bv"]["official"]["estimate"], 4)) == (0.3026, 0.2983)
    for base in ("zero_shot", "rag_only"):
        for m in ("ca", "bv"):
            g = res["comparisons"][f"RQ1:full_aware_vs_{base}"][m]
            assert (g["official"]["estimate"], g["official"]["ci95"]) == (sap[base][m]["estimate"], sap[base][m]["ci95"])
            assert g["swrap"] == g["official"]  # nothing unwrapped (flip=False): both columns are sap_run's
    assert res["filled_missing"] == 19


def test_report_refuses_a_missing_pair_that_was_not_accepted(tmp_path):
    rs = rows({"full_aware": 0.3}, seed=7)
    gone = [r for r in rs if r["condition"] == "zero_shot"][:1]
    args, *_ = report_inputs(tmp_path, gone=gone, accept=False)
    r = cli(*args)
    assert r.returncode == 3 and "missing rows" in r.stderr, r.stderr
    sub = tmp_path / "b"
    sub.mkdir()
    args, *_ = report_inputs(sub, gone=gone)  # the official scores accepted it, but report is not told
    r = cli(*[a for i, a in enumerate(args) if a != "--accepted-missing" and (i == 0 or args[i - 1] != "--accepted-missing")])
    assert r.returncode == 3 and "accepted_missing_sha256" in r.stderr, r.stderr
    assert not (tmp_path / "SWRAP.json").exists() and not (sub / "SWRAP.json").exists()


def test_report_refuses_rows_that_do_not_match(tmp_path):
    args, official, swrap = report_inputs(tmp_path)[:3]
    put_swrap(tmp_path, swrap[1:])  # one base row missing
    r = cli(*args)
    assert r.returncode == 3 and "the official base rows" in r.stderr, r.stderr
    put_swrap(tmp_path, [{**x, "model": "deepseek"} for x in swrap])
    r = cli(*args)
    assert r.returncode == 3 and "probe" in r.stderr
    flipped = [dict(x) for x in swrap]
    unwrapped = {tuple(k) for k in json.loads((tmp_path / "sw" / "SWRAP_META.json").read_text())["unwrapped"]}
    other = next(x for x in flipped if (x["record_id"], x["condition"], x["seed"], x["variant"]) not in unwrapped)
    other["ca"] = 1 - other["ca"]  # a row S-wrap did not change scores otherwise: other code or other controls
    put_swrap(tmp_path, flipped)
    r = cli(*args)
    assert r.returncode == 3 and "not unwrapped" in r.stderr
    put_swrap(tmp_path, swrap)
    meta = tmp_path / "swscore" / "SCORES_META.json"
    good = meta.read_text()
    meta.write_text(json.dumps({**json.loads(good), "inputs": {"x": "0" * 64}}))
    r = cli(*args)
    assert r.returncode == 3 and "SWRAP_META" in r.stderr
    meta.write_text(good)
    assert not (tmp_path / "SWRAP.json").exists()
    (tmp_path / "SWRAP.json").write_text("{}")
    r = cli(*args)
    assert r.returncode == 3 and "never writes over" in r.stderr


def test_report_ties_the_swrap_scores_to_the_official_code_and_controls(tmp_path):
    """The S-wrap SCORES_META must have the official one's code_sha256, a3_sha256 and controls, and each SCORES_META
    its scores' sha256; --controls-dir must be those controls."""
    args, official, swrap = report_inputs(tmp_path)[:3]
    for kw, word in (({"code_sha256": "d" * 64}, "code_sha256"), ({"a3_sha256": {"v2tools/state.py": "b" * 64}}, "a3_sha256"),
                     ({"scores_sha256": "0" * 64}, "scores_sha256"), ({"partition": "dev"}, "partition")):
        put_swrap(tmp_path, swrap, **kw)
        r = cli(*args)
        assert r.returncode == 3 and word in r.stderr, (word, r.stderr)
    put_swrap(tmp_path, swrap)
    ctl = tmp_path / "controls_test"
    other = tmp_path / "other_ctl"
    other.mkdir()
    for n in CTL_FILES:
        (other / n).write_bytes((ctl / n).read_bytes())
    (other / "ca_ref.jsonl").write_text("{}\n")
    write_meta(tmp_path / "swscore" / "scores.jsonl", other,
               {str(tmp_path / "sw" / "swrap_outputs.jsonl"): sha(tmp_path / "sw" / "swrap_outputs.jsonl")})
    r = cli(*args)
    assert r.returncode == 3 and "controls" in r.stderr, r.stderr
    put_swrap(tmp_path, swrap)
    official_meta = tmp_path / "official" / "SCORES_META.json"
    official_meta.write_text(json.dumps({**json.loads(official_meta.read_text()), "scores_sha256": "0" * 64}))
    r = cli(*args)
    assert r.returncode == 3 and "scores_sha256" in r.stderr
    write_meta(tmp_path / "official" / "scores.jsonl", ctl)
    (ctl / "exclusions.jsonl").write_text("")  # --controls-dir no longer holds the exclusions both were scored with
    r = cli(*args)
    assert r.returncode == 3 and "--controls-dir" in r.stderr, r.stderr
    assert not (tmp_path / "SWRAP.json").exists()
