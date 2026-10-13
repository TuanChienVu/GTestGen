"""The Phase 3 fixture repo (phase3_fixture.py) prepared for scoring: record "I open the shop" (REUSE) + "I should see
the total \"5\"" (CREATE, gold iShouldSeeTotal), suite "s" also holding "I add \"apple\" to the cart"."""
from __future__ import annotations

from tests.phase3_fixture import SHOP, catalog, make_repo, record
from v2tools.evaluate import Repo, prepare
from v2tools.state import RepoContext


def make_prep(tmp_path, partition: str = "test"):
    bare, head = make_repo(tmp_path)
    cat = catalog()
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    def_of = lambda text: next(a["def_id"] for f in cat.values() for a in f["annotations"] if text in a["pattern"])
    suite = {"s": [("I open the shop", (def_of("open the shop"),)), ('I should see the total "5"', (def_of("total"),)),
                   ('I add "apple" to the cart', (def_of("cart"),))]}
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ctx, frozenset(), suite)
    r = record(cat, partition)
    r["commit"] = head
    return prepare(r, repo, {"step_def_ids": [[def_of("open the shop")], [def_of("total")]]})


def block(code: str) -> str:
    ticks = "`" * 3
    return f"{ticks}php\n// FILE: {SHOP}\n{code}\n{ticks}"


def make_phase2(tmp_path):
    """A Phase 2 + Phase 3 input tree for the CLIs: out/phase2 (one test record), pairs, config, repos-full, symtimes."""
    import json
    (tmp_path / "git").mkdir()
    p = make_prep(tmp_path / "git")
    r, cat = p.record, p.repo.fns
    root = tmp_path / "work"
    (root / "phase2" / "records").mkdir(parents=True)
    for part in ("train", "dev", "aux_test"):
        (root / "phase2" / "records" / f"{part}.jsonl").write_text("")
    (root / "phase2" / "records" / "test.jsonl").write_text(json.dumps(r) + "\n")
    (root / "phase2" / "catalog.jsonl").write_text("".join(json.dumps(f) + "\n" for f in cat.values()))
    (root / "phase2" / "forbidden.json").write_text("{}")
    rows = [("I open the shop", 'I should see the total "5"'), ('I add "apple" to the cart',)]
    labels, pairs = [], []
    for i, steps in enumerate(rows):
        k = {"repo": "Acme/Shop", "feature_path": r["feature_path"] if i == 0 else "features/cart.feature",
             "scenario_line": 3, "suite": "s"}
        ids = [[a for s2, a in p.repo.suite_steps["s"] if s2 == s][0] for s in steps]
        labels.append({**k, "step_def_ids": [list(x) for x in ids]})
        pairs.append({**k, "steps": list(steps)})
    (root / "phase2" / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (root / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pairs))
    (root / "sources.json").write_text(json.dumps([{"dir": "shop", "repo": "Acme/Shop", "language": "php",
                                                     "styles": ["php_attribute", "php_docblock"]}]))
    (root / "full").mkdir()
    (p.repo.ctx.git_dir).rename(root / "full" / "shop.git")
    (root / "symtimes.jsonl").write_text("")
    (root / "self_contained.jsonl").write_text(json.dumps({"record_id": r["id"], "self_contained": True}) + "\n")
    (root / "subsets.json").write_text(json.dumps({"post_cutoff_all": [r["id"]], "time_coherent_30d": [], "residual_clean": [r["id"]]}))
    (root / "PROMPTS_META.json").write_text(json.dumps({"v2tools_commit": a3_repo(root / "a3"), "symtimes_sha256":
                                                       sha256_of(root / "symtimes.jsonl")}))
    return root, r


A3_FILES = ("v2tools/state.py", "v2tools/compose.py", "v2tools/records.py", "v2tools/outline.py", "v2tools/inherit.py",
            "v2tools/symcache.py", "v2tools/symtime.py")


def sha256_of(path) -> str:
    """sha256 of a file's bytes, as 3A writes PROMPTS_META.symtimes_sha256."""
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def a3_repo(path, edit: str = "") -> str:
    """A git repo holding this tree's 3A files (the commit that "built the prompts", for score_run
    --a3-repo); `edit` is appended to state.py in a second commit. Returns the HEAD commit."""
    import os
    import subprocess
    from pathlib import Path
    src = Path(__file__).resolve().parents[1]
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, text=True, env=env).stdout
    if not (path / ".git").exists():
        (path / "v2tools").mkdir(parents=True)
        git("init", "-q")
        for f in A3_FILES:
            (path / f).write_bytes((src / f).read_bytes())
        git("add", "-A")
        git("commit", "-q", "-m", "3A")
    if edit:
        (path / A3_FILES[0]).write_text((path / A3_FILES[0]).read_text() + edit)
        git("commit", "-q", "-am", "edit")
    return git("rev-parse", "HEAD").strip()


T2025 = 1735689600  # 2025-01-01: the dev record's answer (make_phase2_masked)
GRAND = "I should see the grand total :total"


def make_phase2_masked(tmp_path):
    """Helper-call expansion under the answer mask (§2.2b of the design specification: 3C reads V(r) exactly as 3A did):
    an input tree like make_phase2 whose shop history gets a fourth commit (2025-01-01) that adds iShouldSeeGrandTotal
    (GRAND, calls $this->computeTotal), the answer of a DEV record "Grand total" (t_vis 2025-01-01). The TEST record
    "Total" of make_phase2 writes the helper computeTotal with its answer (2024-06-01), so 3A's Composer masks
    (ShopContext.php, computeTotal) as answer infrastructure in every train and dev prompt of the repo: for the dev
    record, which comes after the helper, computeTotal is FUTURE (④(a) does not show it) only when the record is
    prepared with that mask; without it, the helper is shown and not future. Returns (root, dev record, test record)."""
    import json
    from tests.phase3_fixture import SHOP_2024, _fn, _git
    (tmp_path / "git").mkdir()
    bare, _ = make_repo(tmp_path / "git")
    shop = SHOP_2024.replace("        return $total;\n    }\n", "        return $total;\n    }\n\n"
                             f"    #[Then('{GRAND}')]\n    public function iShouldSeeGrandTotal(string $total): void\n"
                             "    {\n        $this->checkout->assertTotal($this->computeTotal($total));\n    }\n")
    work = tmp_path / "git" / "work"  # make_repo's working clone: one more commit, pushed to its bare clone
    (work / SHOP).write_text(shop)
    _git(work, "commit", "-q", "-am", f"c{T2025}", date=T2025)
    _git(work, "push", "-q", str(bare), "main")
    head = _git(bare, "rev-parse", "HEAD").strip()
    cat = catalog()  # rows read from SHOP_2024: the new method comes after them, so their lines still hold at HEAD
    body = next(i for i, l in enumerate(shop.splitlines(), 1) if "function iShouldSeeGrandTotal(" in l) + 1
    grand = _fn(body, "iShouldSeeGrandTotal", "Then", GRAND, T2025, text=shop)
    cat[grand["fn_id"]] = grand
    test = dict(record(cat, "test"), commit=head)
    dev = dict(record(cat, "dev", "Acme/Shop|features/grand.feature|3|s"), commit=head, scenario_name="Grand total",
               scenario_time=T2025, t_vis=T2025, steps=["I open the shop", 'I should see the grand total "5"'],
               scenario='Scenario: Grand total\n  Given I open the shop\n  Then I should see the grand total "5"',
               create=[{"fn_id": grand["fn_id"], "path": SHOP, "code": grand["code"], "preexisting": False,
                        "post_cutoff_code": True}], create_def_ids=[grand["annotations"][0]["def_id"]])
    dev["step_fns"] = [dev["step_fns"][0], grand["fn_id"]]
    root = tmp_path / "work"
    (root / "phase2" / "records").mkdir(parents=True)
    for part, rows in (("train", []), ("dev", [dev]), ("test", [test]), ("aux_test", [])):
        (root / "phase2" / "records" / f"{part}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (root / "phase2" / "catalog.jsonl").write_text("".join(json.dumps(f) + "\n" for f in cat.values()))
    (root / "phase2" / "forbidden.json").write_text("{}")
    def_of = {a["pattern"]: a["def_id"] for f in cat.values() for a in f["annotations"]}
    pattern = {"I open the shop": "I open the shop", 'I should see the total "5"': "I should see the total :total",
               'I should see the grand total "5"': GRAND, 'I add "apple" to the cart': "I add :product to the cart"}
    labels, pairs = [], []
    for x in (test, dev, {"feature_path": "features/cart.feature", "steps": ['I add "apple" to the cart']}):
        k = {"repo": "Acme/Shop", "feature_path": x["feature_path"], "scenario_line": 3, "suite": "s"}
        labels.append({**k, "step_def_ids": [[def_of[pattern[s]]] for s in x["steps"]]})
        pairs.append({**k, "steps": list(x["steps"])})
    (root / "phase2" / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (root / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pairs))
    (root / "sources.json").write_text(json.dumps([{"dir": "shop", "repo": "Acme/Shop", "language": "php",
                                                     "styles": ["php_attribute", "php_docblock"]}]))
    (root / "full").mkdir()
    bare.rename(root / "full" / "shop.git")
    (root / "symtimes.jsonl").write_text("")
    (root / "self_contained.jsonl").write_text("".join(json.dumps({"record_id": x["id"], "self_contained": True}) + "\n"
                                                       for x in (test, dev)))
    (root / "subsets.json").write_text(json.dumps({"post_cutoff_all": [dev["id"]], "time_coherent_30d": [],
                                                   "residual_clean": [dev["id"]]}))
    (root / "PROMPTS_META.json").write_text(json.dumps({"v2tools_commit": a3_repo(root / "a3"), "symtimes_sha256":
                                                       sha256_of(root / "symtimes.jsonl")}))
    return root, dev, test


# ----- the same on the Java twin of the fixture (tests/test_score_independent.py, jpair, here with a class chain):
# ShopSteps extends BaseSteps (field cart); the gold iShouldSeeTotal calls the future private helper computeTotal, which
# calls BaseSteps.baseNew, future too; the constructor added with them is future but never a helper (a constructor
# never expands). Shared here (not in tests/test_controls.py) so tests/test_describe.py can use it too without
# importing another test file.

JPATH = "src/test/java/acme/steps/ShopSteps.java"
JBASE = "src/test/java/acme/steps/BaseSteps.java"
CART = ("package acme;\n\npublic class Cart {\n    public void open() {\n    }\n\n    public void assertTotal(String total) {\n    }\n\n"
        "    public String total() {\n        return \"\";\n    }\n}\n")
BASE_2020 = ("package acme.steps;\n\nimport acme.Cart;\n\npublic abstract class BaseSteps {\n\n    protected final Cart cart = new Cart();\n\n"
             "    protected void baseOld() {\n        cart.open();\n    }\n}\n")
BASE_2024 = BASE_2020.replace("        cart.open();\n    }\n", "        cart.open();\n    }\n\n    protected void baseNew() {\n"
                              "        cart.total();\n    }\n")
JAVA_2020 = ('package acme.steps;\n\nimport io.cucumber.java.en.Given;\nimport io.cucumber.java.en.Then;\n\n'
             'public class ShopSteps extends BaseSteps {\n\n    @Given("I open the shop")\n    public void iOpenTheShop() {\n'
             '        cart.open();\n    }\n}\n')
JAVA_2024 = JAVA_2020.replace("        cart.open();\n    }\n", '        cart.open();\n    }\n\n    public ShopSteps() {\n        super();\n'
                              '    }\n\n    @Then("I should see the total {string}")\n    public void iShouldSeeTotal(String total) {\n'
                              '        cart.assertTotal(computeTotal(total));\n    }\n\n    private String computeTotal(String total) {\n'
                              '        baseNew();\n        return total;\n    }\n')


def make_jprep(tmp_path):
    import subprocess
    from tests.phase3_fixture import T2020, T2024, _git, record
    from v2tools.evaluate import Repo, prepare
    from v2tools.state import RepoContext
    work = tmp_path / "jwork"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in [(T2020, {"src/main/java/acme/Cart.java": CART, JBASE: BASE_2020, JPATH: JAVA_2020}),
                        (T2024, {JBASE: BASE_2024, JPATH: JAVA_2024})]:
        for path, text in files.items():
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp_path / "java.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    head = _git(bare, "rev-parse", "HEAD").strip()

    def jfn(keyword, pattern, time):
        lines = JAVA_2024.splitlines()
        ann = next(i for i, l in enumerate(lines, 1) if f'("{pattern}")' in l)
        end = next(i for i in range(ann, len(lines) + 1) if lines[i - 1] == "    }")
        return {"repo": "Acme/Java", "fn_id": f"{JPATH}:{ann + 1}", "path": JPATH, "first_line": ann, "body_end": end,
                "code": "\n".join(l[4:] for l in lines[ann - 1:end]),
                "annotations": [{"def_id": f"{JPATH}:{ann}", "line": ann, "keyword": keyword, "pattern": pattern, "time": time}],
                "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}
    opn, tot = jfn("Given", "I open the shop", T2020), jfn("Then", "I should see the total {string}", T2024)
    cat = {f["fn_id"]: f for f in (opn, tot)}
    r = record(cat)
    r.update(id="Acme/Java|features/total.feature|3|s", repo="Acme/Java", commit=head, language="java", framework="Cucumber",
             step_fns=[opn["fn_id"], tot["fn_id"]], create_def_ids=[tot["annotations"][0]["def_id"]],
             create=[{"fn_id": tot["fn_id"], "path": JPATH, "code": tot["code"], "preexisting": False, "post_cutoff_code": True}])
    d_open, d_tot = opn["annotations"][0]["def_id"], tot["annotations"][0]["def_id"]
    suite = {"s": [("I open the shop", (d_open,)), ('I should see the total "5"', (d_tot,))]}
    repo = Repo("Acme/Java", "java", ["java_annotation"], cat, RepoContext("Acme/Java", "java", bare, head, cat), frozenset(), suite)
    return prepare(r, repo, {"step_def_ids": [[d_open], [d_tot]]})


# ----- The generation run's manifest and coverage (its manifest contract, fixed 26/09/2026; score_run --manifest) -----

PLAN_FMT = {"zero_shot": "F0", "ft_only": "F0", "rag_only": "F1", "full_neutral": "F1", "full_aware": "F2"}


def attempts_of(statuses: list[str]) -> list[dict]:
    """A missing job's receipts, oldest first: receipt-generate.attempt-K.json, the last one receipt-generate.json."""
    import hashlib
    names = [f"receipt-generate.attempt-{k}.json" for k in range(1, len(statuses))] + ["receipt-generate.json"]
    return [{"file": n, "status": s, "sha256": hashlib.sha256(f"{n}|{s}".encode()).hexdigest()}
            for n, s in zip(names[-len(statuses):] if statuses else [], statuses)]


def plan_line(key: dict, status: str = "ok", **kw) -> dict:
    """One manifest line with every key of the contract: an `ok` line takes line_sha256 etc. from `kw`; a `missing` line
    its `attempts` (receipt_status and receipt_sha256 = the last attempt's; [] = an absent job: "absent", None);
    an `excluded` line a `reason`. evidence_eligible: last status CRASHED or INCOMPLETE, >= 2 attempts with sha256."""
    row = {"model": key["model"], "condition": key["condition"], "seed": key["seed"], "variant": key["variant"],
           "record_id": key["record_id"], "fmt": key.get("fmt") or PLAN_FMT[key["condition"]],
           "prompts_sha256": key.get("prompts_sha256", "a" * 64), "prompt_sha256": key.get("prompt_sha256", "b" * 64),
           "adapter_tensor_sha256": key.get("adapter_tensor_sha256", None if key["seed"] == "base" else "c" * 64),
           "decision_sha256": key.get("decision_sha256", "d" * 64),
           "gen_code_sha256": key.get("gen_code_sha256", {"v2gpu/generate.py": "e" * 64, "v2gpu/runs.py": "f" * 64}),
           "leakcheck_receipt_sha256": key.get("leakcheck_receipt_sha256", "f" * 64),
           "workdir": f"gen-{key['model']}-{key['variant']}", "receipt_sha256": "1" * 64, "outputs_sha256": None,
           "line_sha256": None, "status": status, "attempts": None}
    if status == "missing":
        att = kw.pop("attempts", attempts_of(["CRASHED", "CRASHED"]))
        row.update(attempts=att, receipt_status=att[-1]["status"] if att else "absent",
                   receipt_sha256=att[-1]["sha256"] if att else None, outputs_sha256=None,
                   evidence_eligible=len(att) >= 2 and all(a.get("sha256") for a in att)
                   and att[-1]["status"] in ("CRASHED", "INCOMPLETE"))
    if status == "excluded":
        row["reason"] = kw.pop("reason", "control_token")
    row.update(kw)
    return row


def gen_plan(outputs: list, extra: list[dict] = (), variant: str = "main", status: str | None = None,
             status_except_missing: str = "PASS", edit=lambda rows: rows) -> tuple:
    """(manifest path, coverage path) next to the first outputs file, as the generation run writes them: one `ok` line
    per output row (its key; its sha fields; line_sha256 = sha256 of the row's raw bytes between b"\\n"; outputs_sha256
    of its file), plus `extra` lines, sorted by (model, condition, seed, record_id, fmt), JSON sort_keys;
    coverage.<variant>.json with manifest_sha256, manifest_path, status (PASS unless a line is missing) and
    status_except_missing. `edit` may change the lines before they are written (fail-closed cases). A row that repeats
    a key, or does not parse, is left out."""
    import hashlib
    import json
    from pathlib import Path
    lines, seen = [], set()
    for f in map(Path, outputs):
        data = f.read_bytes()
        for raw in data.split(b"\n"):
            try:
                o = json.loads(raw)
            except ValueError:
                continue
            k = (o["model"], o["condition"], o["seed"], o.get("variant"), o["record_id"], o.get("fmt"))
            if k in seen:
                continue
            seen.add(k)
            lines.append(plan_line({**{s: o.get(s) for s in ("prompts_sha256", "prompt_sha256")}, **o},
                                   line_sha256=hashlib.sha256(raw).hexdigest(),
                                   outputs_sha256=hashlib.sha256(data).hexdigest()))
    lines = edit(sorted(lines + list(extra), key=lambda x: (x["model"], x["condition"], x["seed"], x["record_id"],
                                                               x["fmt"])))
    first = Path(outputs[0])
    manifest, coverage = first.parent / f"manifest.{variant}.jsonl", first.parent / f"coverage.{variant}.json"
    manifest.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    missing = any(x["status"] == "missing" for x in lines)
    coverage.write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                    "manifest_path": manifest.name, "status": status or ("FAIL" if missing else "PASS"),
                                    "status_except_missing": status_except_missing}, sort_keys=True))
    return manifest, coverage
