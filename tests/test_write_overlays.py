import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.phase3c_fixture import make_phase2

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)


def test_write_overlays_base_and_conditions(tmp_path):
    w, r = make_phase2(tmp_path)
    ids = w / "ids.txt"
    ids.write_text(r["id"] + "\n")
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware",
                               "text": "no\u2028code\x85"}, ensure_ascii=False) + "\n", encoding="utf-8")  # JSONL: "\n" only
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov"), "--run", "t",
              "--ids", str(ids), "--controls", "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    rows = {x["condition"]: x for x in map(json.loads, (w / "ov" / "t" / "manifest.jsonl").read_text().splitlines())}
    assert set(rows) == {"neg", "gold", "stub", "deepseek.s42.full_aware"}
    base = w / "ov" / rows["neg"]["base"]
    shop = "src/Behat/Context/ShopContext.php"
    assert "iShouldSeeTotal" not in (base / shop).read_text() and rows["neg"]["files"] == {} and rows["neg"]["dir"] is None
    gold = (w / "ov" / rows["gold"]["dir"] / shop).read_text()
    assert "public function iShouldSeeTotal" in gold and list(rows["gold"]["files"]) == [shop]
    stub = (w / "ov" / rows["stub"]["dir"] / shop).read_text()
    assert "#[Then('I should see the total :total')]\n    public function stub1($total)\n    {\n    }" in stub
    assert rows["deepseek.s42.full_aware"]["skip"] == "no_usable_block" and rows["deepseek.s42.full_aware"]["dir"] is None
    digest = (w / "ov" / "t" / "MANIFEST_SHA256").read_text().split()[0]
    assert digest == hashlib.sha256((w / "ov" / "t" / "manifest.jsonl").read_bytes()).hexdigest()
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov2"), "--run", "t",
              "--ids", str(ids), "--controls", "--skip-gold", str(ids))
    assert res.returncode == 0, res.stderr
    rows = {x["condition"]: x for x in map(json.loads, (w / "ov2" / "t" / "manifest.jsonl").read_text().splitlines())}
    assert rows["gold"]["skip"] == "flaky_gold" and rows["gold"]["dir"] is None and rows["stub"]["skip"] is None
    other = w / "other.txt"
    other.write_text("Acme/Shop|features/none.feature|1|s\n")
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov3"), "--run", "t",
              "--ids", str(ids), "--controls", "--skip-gold", str(other))
    assert res.returncode == 2 and "not in this run" in res.stderr
    java = w / "sources_java.json"  # the ER runner is PHP (Sylius) only: another language is refused
    java.write_text(json.dumps([{**json.loads((w / "sources.json").read_text())[0], "language": "java"}]))
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(java),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov4"), "--run", "t",
              "--ids", str(ids), "--controls")
    assert res.returncode == 2 and "PHP" in res.stderr
    gen.write_text(gen.read_text(encoding="utf-8") + "\n" + gen.read_text(encoding="utf-8"), encoding="utf-8")  # blank line
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov5"), "--run", "t",
              "--ids", str(ids), "--outputs", str(gen))
    assert res.returncode == 2 and "not JSONL" in res.stderr


def test_a_future_helper_in_the_output_replaces_the_er_tree_one(tmp_path):
    """The ER overlay uses the same insertion as the scorers, so an output that defines
    the record's FUTURE helper (computeTotal, written with the answer and hidden from V(r)) replaces it instead of
    colliding; every condition file stays among the record's CREATE paths (the ER harness requires it)."""
    from v2tools.compose import gold_answer
    w, r = make_phase2(tmp_path)
    ids = w / "ids.txt"
    ids.write_text(r["id"] + "\n")
    cat = {json.loads(l)["fn_id"]: json.loads(l) for l in (w / "phase2" / "catalog.jsonl").read_text().splitlines()}
    gold = gold_answer(r, cat)  # one fenced block with its FILE line
    helper = "private function computeTotal(string $total): string\n{\n    return $total . '';\n}"
    text = gold[:gold.rindex("`" * 3)] + helper + "\n" + "`" * 3
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware",
                               "text": text}) + "\n")
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov"), "--run", "t",
              "--ids", str(ids), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    row = json.loads((w / "ov" / "t" / "manifest.jsonl").read_text().splitlines()[0])
    assert row["skip"] is None and row["insert_status"] == "ok", row
    shop = "src/Behat/Context/ShopContext.php"
    text = (w / "ov" / row["dir"] / shop).read_text()
    assert text.count("function computeTotal(") == 1 and "return $total . '';" in text
    assert set(row["files"]) <= {c["path"] for c in r["create"]}


def test_a_catalog_the_er_tree_refuses_stops_with_a_clean_error(tmp_path):
    w, r = make_phase2(tmp_path)
    ids = w / "ids.txt"
    ids.write_text(r["id"] + "\n")
    cat = w / "phase2" / "catalog.jsonl"
    rows = [json.loads(l) for l in cat.read_text().splitlines()]
    cat.write_text("".join(json.dumps({**x, "annotations": []} if i == 0 else x) + "\n" for i, x in enumerate(rows)))
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--out", str(w / "ov"), "--run", "t",
              "--ids", str(ids), "--controls")
    assert res.returncode == 2 and "no annotation" in res.stderr and "Traceback" not in res.stderr
    assert not (w / "ov" / "t" / "manifest.jsonl").exists()
    # the repo named once, not also as ertree's " (repo Acme/Shop)"
    assert "Acme/Shop: 1 catalog rows with no annotation: [" in res.stderr and res.stderr.count("Acme/Shop") == 1, \
        res.stderr


def test_a_dev_record_replaces_a_helper_that_3a_masks_as_test_answer_infrastructure(tmp_path):
    """write_overlays prepares a dev record with the composer's mask (shown4 of a Composer over
    every partition), so computeTotal, written with a TEST answer and masked in dev prompts, is FUTURE for the dev
    record "Grand total" although it existed at its t_vis: an output that defines it with another body replaces it in
    the overlay (unmasked, the helper is not future and the output would be skipped as a collision)."""
    from tests.phase3_fixture import SHOP
    from tests.phase3c_fixture import make_phase2_masked
    from v2tools.compose import gold_answer
    w, dev, _ = make_phase2_masked(tmp_path)
    ids = w / "ids.txt"
    ids.write_text(dev["id"] + "\n")
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    gold = gold_answer(dev, cat)
    helper = "private function computeTotal(string $total): string\n{\n    return $total . '';\n}"
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({"record_id": dev["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware",
                               "text": gold[:gold.rindex("`" * 3)] + helper + "\n" + "`" * 3}) + "\n")
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop",
              "--out", str(w / "ov"), "--run", "t", "--ids", str(ids), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    row = json.loads((w / "ov" / "t" / "manifest.jsonl").read_text().splitlines()[0])
    assert row["skip"] is None and row["insert_status"] == "ok", row
    base = (w / "ov" / row["base"] / SHOP).read_text()  # the ER tree keeps the helper and drops the dev answer
    assert "return $total;" in base and "iShouldSeeGrandTotal" not in base
    text = (w / "ov" / row["dir"] / SHOP).read_text()
    assert text.count("function computeTotal(") == 1 and "return $total . '';" in text and "return $total;" not in text
    assert "function iShouldSeeGrandTotal(" in text and list(row["files"]) == [SHOP]


def test_bad_inputs_are_refused_cleanly(tmp_path):
    """A missing input file, an unknown or empty --ids list end in a clean refusal (exit 2),
    never a traceback, and no manifest is written."""
    w, r = make_phase2(tmp_path)
    base = ["scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
            "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop", "--run", "t",
            "--controls"]
    cases = {"unknown": "Acme/Shop|features/none.feature|1|s\n", "empty": "\n"}
    for name, text in cases.items():
        ids = w / f"ids_{name}.txt"
        ids.write_text(text)
        res = run(*base, "--out", str(w / f"ov_{name}"), "--ids", str(ids))
        assert res.returncode == 2 and "Traceback" not in res.stderr, (name, res.stderr)
        assert not (w / f"ov_{name}" / "t" / "manifest.jsonl").exists()
    res = run(*base, "--out", str(w / "ov_noids"), "--ids", str(w / "absent.txt"))
    assert res.returncode == 2 and "Traceback" not in res.stderr
    (w / "phase2" / "labels.jsonl").unlink()
    ids = w / "ids.txt"
    ids.write_text(r["id"] + "\n")
    res = run(*base, "--out", str(w / "ov_nolabels"), "--ids", str(ids))
    assert res.returncode == 2 and "Traceback" not in res.stderr and "labels.jsonl" in res.stderr


def test_an_overlay_never_registers_a_pattern_twice_on_a_rewritten_shell(tmp_path):
    """Behat 4.0.0-alpha1 reads attributes only, and a pattern registered twice throws RedundantStepException: the
    record's preexisting shell iAddToCart, rewritten by the output with its HEAD pattern in the other form (docblock)
    or with another keyword, registers that pattern once in the overlay, by the HEAD attribute; the output's duplicate
    annotation is dropped (insert.php_kept)."""
    from collections import Counter

    from tests.phase3_fixture import SHOP
    from v2tools.compose import gold_answer
    from v2tools.stepdefs import extract as extract_defs
    w, r = make_phase2(tmp_path)
    ids = w / "ids.txt"
    ids.write_text(r["id"] + "\n")
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    cart = next(fid for fid, f in cat.items() if "function iAddToCart(" in f["code"])
    r["create"].append({"fn_id": cart, "path": SHOP, "code": cat[cart]["code"], "preexisting": True,
                        "post_cutoff_code": False})
    (w / "phase2" / "records" / "test.jsonl").write_text(json.dumps(r) + "\n")
    gold, head = gold_answer(r, cat), "#[When('I add :product to the cart')]"
    assert gold.count(head) == 1
    gen = w / "gen.jsonl"
    news = ("/**\n * @When I add :product to the cart\n */", "#[Given('I add :product to the cart')]")
    gen.write_text("".join(json.dumps({"record_id": r["id"], "model": "m", "seed": f"s{k}", "condition": "full_aware",
                                       "text": gold.replace(head, new)}) + "\n" for k, new in enumerate(news)))
    res = run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
              "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop",
              "--out", str(w / "ov"), "--run", "t", "--ids", str(ids), "--outputs", str(gen))
    assert res.returncode == 0, res.stderr
    rows = [json.loads(x) for x in (w / "ov" / "t" / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["skip"] is None and row["insert_status"] == "ok", row
        text = (w / "ov" / row["dir"] / SHOP).read_text()
        pats = Counter(d.pattern for st in ("php_attribute", "php_docblock") for d in extract_defs(SHOP, text, st))
        assert pats["I add :product to the cart"] == 1 and set(pats.values()) == {1}, text
        assert text.count(head) == 1 and "@When I add" not in text and "#[Given('I add" not in text, text


# ----- pins of the writer's contract (each kills its mutants), and every bad input ends in exit 2 before any file is
# written --------------------------------------------------------------------------------------------------------------

SHOP = "src/Behat/Context/ShopContext.php"


def go(w, ids, out, *extra):
    return run("scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"),
               "--full-root", str(w / "full"), "--symtimes", str(w / "symtimes.jsonl"), "--repo", "Acme/Shop",
               "--out", str(out), "--run", "t", "--ids", str(ids), *extra)


def rows_of(out):
    return {x["condition"]: x for x in map(json.loads, (out / "t" / "manifest.jsonl").read_text().split("\n")[:-1])}


def phase2_ids(tmp_path):
    w, r = make_phase2(tmp_path)
    ids = w / "ids.txt"
    ids.write_text(r["id"] + "\n")
    return w, r, ids


def test_pin_manifest_fields_and_statuses(tmp_path):
    w, r, ids = phase2_ids(tmp_path)
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware",
                               "text": "no code"}) + "\n")
    assert go(w, ids, w / "ov", "--controls", "--outputs", str(gen)).returncode == 0
    rows = rows_of(w / "ov")
    for c in ("neg", "gold", "stub"):
        assert rows[c]["skip"] is None and rows[c]["insert_status"] == "ok" and rows[c]["extract_status"] == "ok"
    x = rows["deepseek.s42.full_aware"]
    assert (x["extract_status"], x["insert_status"], x["skip"]) == ("none", "empty", "no_usable_block")
    for c, x in rows.items():
        assert (x["suite"], x["scenario_name"], x["t_vis"]) == (r["suite"], r["scenario_name"], r["t_vis"])
        assert x["target"] == f"{r['feature_path']}:{r['scenario_line']}", c
        assert x["stats"]["answer_fns_removed"] >= 1 and set(x["stats"]) == {
            "answer_fns_removed", "answer_annotations_removed", "future_fns_removed", "phantom_fns"}
    assert rows["neg"]["base_files"] and rows["neg"]["base_files"] == rows["gold"]["base_files"] == rows["stub"]["base_files"]


def test_pin_collision_rows(tmp_path):
    from v2tools.compose import gold_answer
    w, r, ids = phase2_ids(tmp_path)
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    gold = gold_answer(r, cat)
    body = gold[:gold.rindex("`" * 3)]
    dup = body + body[body.index("#["):] + "`" * 3        # the same method twice in the output
    other = "public function iOpenTheShop(): void\n{\n}\n"       # a method of the file that the answer does not replace
    clash = body + other + "`" * 3
    gen = w / "gen.jsonl"
    gen.write_text("".join(json.dumps({"record_id": r["id"], "model": "m", "seed": f"s{k}", "condition": "c", "text": t}) + "\n"
                           for k, t in enumerate((dup, clash))))
    assert go(w, ids, w / "ov", "--outputs", str(gen)).returncode == 0
    rows = rows_of(w / "ov")
    for c in ("m.s0.c", "m.s1.c"):
        assert (rows[c]["skip"], rows[c]["insert_status"], rows[c]["dir"]) == ("collision", "collision", None), (c, rows[c])
        assert rows[c]["files"] == {} and rows[c]["base_files"]


def test_pin_ids_must_be_test_or_dev_records(tmp_path):    # the id has a labels row, so only the partition refuses it
    w, r, ids = phase2_ids(tmp_path)
    tid = r["id"].replace("|s", "|t")
    (w / "phase2" / "records" / "train.jsonl").write_text(json.dumps({**r, "id": tid, "partition": "train"}) + "\n")
    lab = json.loads((w / "phase2" / "labels.jsonl").read_text().splitlines()[0])
    (w / "phase2" / "labels.jsonl").write_text((w / "phase2" / "labels.jsonl").read_text() + json.dumps({**lab, "suite": "t"}) + "\n")
    ids.write_text(tid + "\n")
    res = go(w, ids, w / "ov", "--controls")
    assert res.returncode == 2 and "not test/dev records" in res.stderr and "Traceback" not in res.stderr


def test_pin_duplicate_ids_are_refused(tmp_path):
    w, r, ids = phase2_ids(tmp_path)
    ids.write_text(r["id"] + "\n" + r["id"] + "\n")
    res = go(w, ids, w / "ov", "--controls")
    assert res.returncode == 2 and not (w / "ov" / "t" / "manifest.jsonl").exists()


def test_pin_a_run_directory_is_never_overwritten(tmp_path):
    w, r, ids = phase2_ids(tmp_path)
    assert go(w, ids, w / "ov", "--controls").returncode == 0
    before = (w / "ov" / "t" / "manifest.jsonl").read_bytes()
    res = go(w, ids, w / "ov", "--controls")
    assert res.returncode == 2 and (w / "ov" / "t" / "manifest.jsonl").read_bytes() == before


def _wo():
    import importlib.util
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("wo_pin", ROOT / "scripts" / "write_overlays.py")
    wo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wo)
    return wo


def test_pin_stub_parameters_and_forms():                  # also the er prefix and $multiline
    wo = _wo()
    assert wo.stub_params("I pay :a for :b", False) == ["a", "b"]
    assert wo.stub_params(r'^I pay "([^"]*)" for (\d+) (?:units|pcs) (?P<who>\w+) (?P<n2>x)$', True) == ["arg1", "arg2", "who", "n2"]
    assert wo.stub_params(r"^a \(b\) [(x] (y)$", True) == ["arg1"]
    assert wo.stub_params(r"^(?P<a>x) (?<b>y)$", True) == ["a", "b"]
    assert wo.stub_params(r"^(?=a)(?!b)(?<=c)(x)$", True) == ["arg1"]


def test_pin_er_stub_docblock_regex_table_and_prefix(tmp_path):     # also $multiline and the er prefix
    import copy
    wo = _wo()
    w, r = make_phase2(tmp_path)
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    fid = r["create"][0]["fn_id"]
    fn = copy.deepcopy(cat[fid])
    ann = fn["annotations"][0]
    ann["pattern"] = '^I pay "([^"]*)" (?P<n>\\d+)$'
    lines = ["x"] * 60
    lines[ann["line"] - 1] = '     * @Then /^I pay "([^"]*)" (?P<n>\\d+)$/'
    r2 = dict(r, scenario=r["scenario"] + "\n    | a |")                      # a table follows the CREATE step
    labels = {"step_def_ids": [[], [ann["def_id"]]]}
    got = wo.er_stub(r2, {**cat, fid: fn}, labels, {SHOP: "class X { public function stub1() {} }"}, lambda p: "\n".join(lines))
    assert got[SHOP] == ('/**\n * @Then /^I pay "([^"]*)" (?P<n>\\d+)$/\n */\n'
                         'public function erstub1($arg1, $n, $multiline = null)\n{\n}')


def test_pin_slug_and_manifest_digest_file(tmp_path):
    wo = _wo()
    rid = "Acme/Shop|features/total.feature|3|s"
    assert wo.slug(rid) == f"total_3_{hashlib.sha1(rid.encode()).hexdigest()[:6]}"
    assert wo.slug(rid) != wo.slug("Acme/Shop|features/total.feature|3|t")
    w, r, ids = phase2_ids(tmp_path)
    assert go(w, ids, w / "ov", "--controls").returncode == 0
    m = w / "ov" / "t"
    assert (m / "MANIFEST_SHA256").read_text() == f"{hashlib.sha256((m / 'manifest.jsonl').read_bytes()).hexdigest()}  manifest.jsonl\n"


def test_pin_overlay_is_the_scorers_tree(tmp_path):         # ties write_overlays to evaluate (any drift)
    from tests.phase3c_fixture import make_prep
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w, r, ids = phase2_ids(tmp_path / "a")
    p = make_prep(tmp_path / "b")                             # the same fixture, scored in-process
    assert go(w, ids, w / "ov", "--controls").returncode == 0
    rows = rows_of(w / "ov")
    assert (w / "ov" / rows["neg"]["base"] / SHOP).read_text() == p.er_files[SHOP]
    assert (w / "ov" / rows["gold"]["dir"] / SHOP).read_text() == p.gold_output().after[SHOP]


def test_pin_er_stub_only_for_create_annotations(tmp_path):
    import copy
    wo = _wo()
    w, r = make_phase2(tmp_path)
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    fid = r["create"][0]["fn_id"]
    fn = copy.deepcopy(cat[fid])
    fn["annotations"] = fn["annotations"] + [{**fn["annotations"][0], "def_id": "other-scenario-step", "pattern": "another :x"}]
    got = wo.er_stub(r, {**cat, fid: fn}, {"step_def_ids": [[], []]}, {SHOP: "class X {}"}, lambda p: "\n".join(["#[Then('a :b')]"] * 60))
    assert got[SHOP].count("public function stub") == 1 and "stub2" not in got[SHOP]


def test_pin_nothing_requested_is_refused(tmp_path):        # else an empty manifest, exit 0
    w, r, ids = phase2_ids(tmp_path)
    res = go(w, ids, w / "ov")
    assert res.returncode == 2 and not (w / "ov" / "t" / "manifest.jsonl").exists()


def test_pin_a_malformed_output_row_is_refused_cleanly(tmp_path):
    w, r, ids = phase2_ids(tmp_path)
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({"model": "m", "seed": "s1", "condition": "c"}) + "\n")
    res = go(w, ids, w / "ov", "--controls", "--outputs", str(gen))
    assert res.returncode == 2 and "Traceback" not in res.stderr


def test_pin_an_output_declaring_its_preexisting_shell_twice_collides(tmp_path):   # duplicate_keys
    """targets_after marks `collision` when the output declares a method key twice, as the scorers do: the fixture record
    with its preexisting shell iAddToCart (as in test_an_overlay_never_registers_a_pattern_twice_on_a_rewritten_shell)
    and an output that is the gold plus the shell's code once more (2/149 such outputs on real records)."""
    from v2tools.compose import gold_answer
    w, r, ids = phase2_ids(tmp_path)
    cat = {f["fn_id"]: f for f in map(json.loads, (w / "phase2" / "catalog.jsonl").read_text().splitlines())}
    cart = next(fid for fid, f in cat.items() if "function iAddToCart(" in f["code"])
    r["create"].append({"fn_id": cart, "path": SHOP, "code": cat[cart]["code"], "preexisting": True,
                        "post_cutoff_code": False})
    (w / "phase2" / "records" / "test.jsonl").write_text(json.dumps(r) + "\n")
    gold = gold_answer(r, cat)
    gen = w / "gen.jsonl"
    gen.write_text("".join(json.dumps({"record_id": r["id"], "model": "m", "seed": f"s{k}", "condition": "c", "text": t})
                           + "\n" for k, t in enumerate((gold, gold[:gold.rindex("`" * 3)] + cat[cart]["code"] + "\n" + "`" * 3))))
    assert go(w, ids, w / "ov", "--outputs", str(gen)).returncode == 0
    rows = rows_of(w / "ov")
    assert rows["m.s0.c"]["skip"] is None and rows["m.s0.c"]["dir"]
    assert (rows["m.s1.c"]["skip"], rows["m.s1.c"]["insert_status"], rows["m.s1.c"]["dir"]) == ("collision", "collision", None)


def test_bad_inputs_end_in_exit_2_before_any_file_is_written(tmp_path):
    """A record of --ids without t_vis (as score_run's `timed`), a record of the repo
    without t_vis when a dev id needs the composer, a missing forbidden.json, --config or --skip-gold, a --repo absent
    from the config and a --full-root without the repo's clone end in exit 2 naming the input, not in a traceback;
    --ids at two commits is refused (RepoContext reads one). In every case nothing is written, not even
    <out>."""
    from tests.phase3c_fixture import make_phase2_masked

    def refused(res, out, why):
        assert res.returncode == 2 and why in res.stderr and "Traceback" not in res.stderr, res.stderr
        assert not out.exists()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w, r, ids = phase2_ids(tmp_path / "a")
    test = w / "phase2" / "records" / "test.jsonl"
    test.write_text(json.dumps({k: v for k, v in r.items() if k != "t_vis"}) + "\n")
    refused(go(w, ids, w / "ov1", "--controls"), w / "ov1", "no t_vis")
    other = dict(r, id=r["id"].replace("|3|", "|4|"), scenario_line=4, commit="0" * 40)
    test.write_text(json.dumps(r) + "\n" + json.dumps(other) + "\n")
    ids.write_text(r["id"] + "\n" + other["id"] + "\n")
    refused(go(w, ids, w / "ov2", "--controls"), w / "ov2", "2 commits")
    ids.write_text(r["id"] + "\n")
    base = ["scripts/write_overlays.py", "--phase2", str(w / "phase2"), "--symtimes", str(w / "symtimes.jsonl"), "--run", "t",
            "--ids", str(ids), "--controls"]
    refused(run(*base, "--config", str(w / "absent.json"), "--full-root", str(w / "full"), "--repo", "Acme/Shop",
                "--out", str(w / "ov3")), w / "ov3", "missing input")
    refused(run(*base, "--config", str(w / "sources.json"), "--full-root", str(w / "full"), "--repo", "Acme/Other",
                "--out", str(w / "ov4")), w / "ov4", "Acme/Other is not in")
    refused(run(*base, "--config", str(w / "sources.json"), "--full-root", str(w / "nowhere"), "--repo", "Acme/Shop",
                "--out", str(w / "ov5")), w / "ov5", "--full-root")
    refused(go(w, ids, w / "ov6", "--controls", "--skip-gold", str(w / "absent.txt")), w / "ov6", "absent.txt")
    (w / "phase2" / "forbidden.json").unlink()
    refused(go(w, ids, w / "ov7", "--controls"), w / "ov7", "forbidden.json")
    mw, dev, tst = make_phase2_masked(tmp_path / "b")
    (mw / "phase2" / "records" / "test.jsonl").write_text(json.dumps({k: v for k, v in tst.items() if k != "t_vis"}) + "\n")
    (mw / "ids.txt").write_text(dev["id"] + "\n")
    refused(go(mw, mw / "ids.txt", mw / "ov8", "--controls"), mw / "ov8", "no t_vis")


def test_run_names_are_checked_against_the_harness_when_writing(tmp_path):
    """A run name that the harness's CONDITION regex refuses (a
    capital, a `.` in the model) or a run given twice for a record would make the harness refuse the whole manifest at
    load; write_overlays refuses it first (exit 2, nothing written). The names of the plan's runs pass."""
    w, r, ids = phase2_ids(tmp_path)
    gen, n = w / "gen.jsonl", iter(range(9))

    def outputs(*runs):
        gen.write_text("".join(json.dumps({"record_id": r["id"], "text": "no code", **x}) + "\n" for x in runs))
        return go(w, ids, w / f"ov{next(n)}", "--outputs", str(gen))
    ok = {"model": "deepseek", "seed": "s42", "condition": "full_aware"}
    for bad in ({**ok, "model": "DeepSeek"}, {**ok, "model": "deepseek.v3"}, {**ok, "variant": "dev-pilot"}):
        res = outputs(bad)
        assert res.returncode == 2 and "CONDITION" in res.stderr and "Traceback" not in res.stderr, (bad, res.stderr)
    res = outputs(ok, ok)
    assert res.returncode == 2 and "twice" in res.stderr and "Traceback" not in res.stderr
    assert not any((d / "t").exists() for d in w.glob("ov*"))
    res = outputs({**ok, "seed": "base", "condition": "zero_shot", "variant": "dev_pilot_v0"}, {**ok, "variant": "main"})
    assert res.returncode == 0, res.stderr
    assert set(rows_of(w / "ov4")) == {"deepseek.base.zero_shot.dev_pilot_v0", "deepseek.s42.full_aware"}
