"""Manual error reading (P25, fixed 06/10/2026, before any test output; descriptive): sap_errsample draws the 30 test
records with CA = 0 at Full-aware in EVERY seed of the anchor model, stratified by repo in proportion to the eligible
population (largest remainder), ordered by sha256(seed|record_id) within a repo, each read at the seed sha256(record_id)
picks (the rule of the blind semantic-validity sheet), and writes the diagnostic columns of that row for the reader;
no number of it is a conclusion."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tests.test_sap import CFG, SEEDS
from v2tools.sap import code_files

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sap_errsample", ROOT / "scripts" / "sap_errsample.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)
H = lambda s: hashlib.sha256(s.encode()).hexdigest()


def row(rid, repo, seed, ca, cond="full_aware", **kw):
    return {"record_id": rid, "repo": repo, "language": "php", "scenario_key": rid.rsplit("|", 1)[0], "model": "deepseek",
            "seed": seed, "condition": cond, "variant": "main", "ca": ca, "ca_defined": True, "extract_status": "ok",
            "fence_truncated": False, "bv": 0, **{f"bv_{c}": 1 for c in es.BV_CONDS}, "cap_hit": False,
            "open_resolution": False, "hallucinated": 1, "members_used": 2, "conv_style": 1.0, "conv_pattern": 0.5,
            "conv_naming": 1.0, "n_calls": 3, "call_precision": 0.0, "ca_ref_v3m": 0, "ca_ref_v3all": 0,
            "php_tag_stripped": False, "unread_step_annotations": 0, "duplicate_reuse": False, **kw}


def population(n_a=10, n_b=5, n_c=3):
    """Three repos; every record has CA = 0 at all five seeds unless its name says otherwise."""
    rows = []
    for repo, n in (("A", n_a), ("B", n_b), ("C", n_c)):
        for i in range(n):
            rid = f"{repo}|f{i}.feature|1|a"
            rows += [row(rid, repo, sd, 0) for sd in SEEDS]
    return rows


def test_the_population_is_ca_zero_at_every_seed_of_usable_records_with_a_real_row_each():
    rows = population(2, 0, 0)
    rid_one = "A|one.feature|1|a"  # CA = 1 at one seed: not a failure of Full-aware
    rows += [row(rid_one, "A", sd, int(sd == SEEDS[2])) for sd in SEEDS]
    rid_gap = "A|gap.feature|1|a"  # a seed without a real row (filled or absent): not readable at every seed
    rows += [row(rid_gap, "A", sd, 0) for sd in SEEDS[:-1]] + [row(rid_gap, "A", SEEDS[-1], 0, filled_missing=True)]
    rid_absent = "A|absent.feature|1|a"  # a seed with no row at all
    rows += [row(rid_absent, "A", sd, 0) for sd in SEEDS[:-1]]
    rid_undef = "A|undef.feature|1|a"  # CA undefined
    rows += [row(rid_undef, "A", sd, 0, ca_defined=False) for sd in SEEDS]
    rid_out = "A|out.feature|1|a"  # not a usable record (mechanical exclusion or compose)
    rows += [row(rid_out, "A", sd, 0) for sd in SEEDS]
    rows += [row("A|f0.feature|1|a", "A", sd, 1, cond="full_neutral") for sd in SEEDS]  # other condition: ignored
    rows += [row("A|f1.feature|1|a", "A", sd, 1, variant="lc600") for sd in SEEDS]  # other variant: ignored
    keep = {r["record_id"] for r in rows} - {rid_out}
    pool, info = es.error_pool(rows, CFG, keep)
    assert sorted(pool) == ["A|f0.feature|1|a", "A|f1.feature|1|a"]
    assert info == {"incomplete": 2, "ca_one": 1, "ca_undefined": 1, "not_usable": 1}


def test_largest_remainder_allocation_and_the_order_within_a_repo():
    assert es.allocate({"A": 10, "B": 5, "C": 3}, 6) == {"A": 3, "B": 2, "C": 1}  # 3.33, 1.67, 1.0
    assert es.allocate({"A": 1, "B": 1, "C": 1}, 2) == {"A": 1, "B": 1, "C": 0}  # ties: by repo name
    assert es.allocate({"A": 4, "B": 0}, 30) == {"A": 4, "B": 0}  # a population smaller than n: all of it
    # remainders 4/7, 4/7, 6/7 are a tie for A and B in exact arithmetic (by name: A), which floats broke (B's
    # 26.571... came out above A's 2.571...); exact arithmetic since 06/10/2026
    assert es.allocate({"A": 3, "B": 31, "C": 1}, 30) == {"A": 3, "B": 26, "C": 1}
    rows = population()
    keep = {r["record_id"] for r in rows}
    cfg = {**CFG, "error_reading": {**CFG["error_reading"], "n_records": 6}}
    sample, info = es.error_sample(rows, cfg, keep)
    assert info["population"] == {"A": 10, "B": 5, "C": 3} and info["allocation"] == {"A": 3, "B": 2, "C": 1}
    seed = cfg["error_reading"]["seed"]
    for repo, k in info["allocation"].items():
        ids = sorted({r["record_id"] for r in rows if r["repo"] == repo}, key=lambda x: H(f"{seed}|{x}"))
        assert [s["record_id"] for s in sample if s["repo"] == repo] == ids[:k]
    assert es.error_sample(rows, cfg, keep) == (sample, info)  # deterministic


def test_each_case_is_read_at_the_seed_of_its_hash_with_the_diagnostic_columns_of_that_row():
    rows = population(3, 0, 0)
    for r in rows:
        r["n_calls"] = SEEDS.index(r["seed"]) + 10  # tells the rows of the seeds apart
    sample, _ = es.error_sample(rows, CFG, {r["record_id"] for r in rows})
    assert len(sample) == 3
    for s in sample:
        sd = SEEDS[int(H(s["record_id"]), 16) % len(SEEDS)]
        assert s["seed"] == sd and s["n_calls"] == SEEDS.index(sd) + 10
        assert set(es.COLUMNS) <= set(s) and s["reason"] == "" and s["example"] is False and "ca" not in s
        assert {"extract_status", "bv_syntax", "bv_grounded", "cap_hit", "open_resolution", "hallucinated",
                "members_used", "conv_style", "conv_pattern", "conv_naming", "n_calls", "call_precision", "ca_ref_v3m",
                "ca_ref_v3all", "ca_ref_v3names", "ca_ref_top4", "unread_step_annotations",
                "duplicate_reuse"} <= set(s)  # P25's diagnostic columns, the four lazy-stub references among them


# ----- the CLI ---------------------------------------------------------------------------------------------------------

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p: Path, xs: list[dict]) -> Path:
    p.write_text("".join(json.dumps(x) + "\n" for x in xs))
    return p


def cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "sap_errsample.py"), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True)


def inputs(tmp: Path, rows: list[dict], with_text: bool = True) -> list:
    ids = sorted({r["record_id"] for r in rows})
    ctl = tmp / "ctl"
    ctl.mkdir()
    write(ctl / "controls.jsonl", [{"record_id": i, "control": "gold", "ca_defined": True} for i in ids + ["Z|x.feature|1|a"]])
    write(ctl / "exclusions.jsonl", [{"record_id": "Z|x.feature|1|a", "reason": "ca_undefined"}])
    write(tmp / "compose.jsonl", [])
    scores = write(tmp / "scores.jsonl", rows)
    args = ["--config", ROOT / "config" / "sap.json", "--scores", scores, "--controls-dir", ctl,
            "--compose-excluded", tmp / "compose.jsonl", "--out", tmp / "es"]
    if with_text:
        outs = write(tmp / "outputs.jsonl", [{k: r[k] for k in ("record_id", "model", "seed", "condition", "variant")}
                                             | {"text": f"text of {r['record_id']} at {r['seed']}"} for r in rows])
        args += ["--outputs", outs]
    return args


def scored(rows: list[dict]) -> list[dict]:
    """The rows with the text_sha256 3C writes (describe.row) for the text inputs() gives each of them."""
    return [{**r, "text_sha256": H(f"text of {r['record_id']} at {r['seed']}")} for r in rows]


def test_cli_writes_the_sample_and_a_meta_that_hashes_it_before_any_reading(tmp_path):
    rows = scored(population())
    r = cli(*inputs(tmp_path, rows))
    assert r.returncode == 0, r.stderr
    sample = [json.loads(l) for l in (tmp_path / "es" / "sample.jsonl").read_text().split("\n") if l]
    assert len(sample) == 18  # 18 eligible < 30: all of them
    assert all(s["text"] == f"text of {s['record_id']} at {s['seed']}" for s in sample)
    meta = json.loads((tmp_path / "es" / "ERRSAMPLE_META.json").read_text())
    assert meta["descriptive_only"] is True and meta["seed"] == 20261006 and meta["n_records"] == 30
    assert meta["sample_sha256"] == sha(tmp_path / "es" / "sample.jsonl")
    assert meta["script_sha256"] == sha(ROOT / "scripts" / "sap_errsample.py")
    assert meta["population"] == {"A": 10, "B": 5, "C": 3} and meta["allocation"] == {"A": 10, "B": 5, "C": 3}
    named = (tmp_path / "scores.jsonl", tmp_path / "ctl" / "controls.jsonl", tmp_path / "ctl" / "exclusions.jsonl",
             tmp_path / "compose.jsonl", tmp_path / "outputs.jsonl")
    assert meta["inputs"] == {str(p): sha(p) for p in named}  # every named input, with its sha256
    assert meta["config_sha256"] == sha(ROOT / "config" / "sap.json")
    assert meta["code"] == {p: sha(ROOT / p) for p in code_files(ROOT)}  # hashed as SAP_LOCK.json "code" is
    assert all(s["text_sha256"] == H(s["text"]) for s in sample)  # each text is the text that was scored
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("es")] == ["es"]  # written whole, then renamed


def test_cli_refusals(tmp_path):
    rows = population()
    (tmp_path / "a").mkdir()
    args = inputs(tmp_path / "a", rows)
    (tmp_path / "a" / "es").mkdir()
    r = cli(*args)
    assert r.returncode == 3 and "never writes over" in r.stderr
    (tmp_path / "b").mkdir()
    r = cli(*inputs(tmp_path / "b", rows + rows[:1]))  # the same (record, seed) twice
    assert r.returncode == 3 and "twice" in r.stderr
    (tmp_path / "c").mkdir()
    args = inputs(tmp_path / "c", rows)
    write(Path(args[args.index("--outputs") + 1]), [])  # the outputs lack the text of a sampled case
    r = cli(*args)
    assert r.returncode == 3 and "no output text" in r.stderr
    (tmp_path / "d").mkdir()
    args = inputs(tmp_path / "d", scored(rows))
    o = Path(args[args.index("--outputs") + 1])
    write(o, [{**x, "text": x["text"] + " (another run)"} for x in map(json.loads, o.read_text().split("\n")[:-1])])
    r = cli(*args)  # the text read must be the text 3C scored
    assert r.returncode == 3 and "text_sha256" in r.stderr
    (tmp_path / "e").mkdir()
    r = cli(*inputs(tmp_path / "e", [{**x, "variant": "dev_pilot_v0"} for x in rows]))  # dev scores are refused
    assert r.returncode == 3 and "no score row" in r.stderr
    assert not any((tmp_path / d / "es").exists() for d in "bcde")


def test_a_failure_part_way_leaves_no_out_directory(tmp_path, monkeypatch):
    """Since 06/10/2026 the two files are written into <out>.partial-<pid> and renamed to <out> once whole, so a failure
    while writing ERRSAMPLE_META.json leaves no <out> to block the rerun (sap_errsample never writes over)."""
    args = inputs(tmp_path, scored(population()))
    real = Path.write_text

    def failing(self, *a, **kw):
        if self.name == "ERRSAMPLE_META.json":
            raise OSError("disk full")
        return real(self, *a, **kw)
    monkeypatch.setattr(Path, "write_text", failing)
    monkeypatch.setattr(sys, "argv", ["sap_errsample.py", *map(str, args)])
    try:
        es.main()
    except OSError:
        pass
    else:
        raise AssertionError("the META write should have failed")
    assert not (tmp_path / "es").exists()
    assert [p.name.startswith("es.partial-") for p in tmp_path.iterdir() if p.name.startswith("es")] == [True]
