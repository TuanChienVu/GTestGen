import hashlib
import json

import pytest

from tests.phase3_fixture import catalog, make_repo, record
from v2tools.compose import TICKS, Composer, Params
from v2tools.prompts import build, pool_sha256, held_out_exclusions
from v2tools.state import RepoContext


def words(text: str) -> int:
    return len(text.split())


def short(msgs: list[dict], gold: str) -> int:
    return 10


@pytest.fixture(scope="module")
def comp(tmp_path_factory):
    bare, head = make_repo(tmp_path_factory.mktemp("repo"))
    cat = catalog()
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    q = record(cat)
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])
    tr = dict(record(cat, "train", "Acme/Shop|features/cart.feature|3|s"), feature_path="features/cart.feature",
              steps=["I add \"x\" to the cart"], step_kinds=["create"], neighbours=[],
              create=[{"fn_id": add["fn_id"], "path": add["path"], "code": add["code"], "preexisting": False}],
              create_def_ids=[add["annotations"][0]["def_id"]])
    return Composer({"Acme/Shop": ctx}, {"Acme/Shop": cat}, {}, {q["id"]: q, tr["id"]: tr}, Params())


def test_build_writes_rows_meta_excluded_and_sums(comp, tmp_path):
    pool = [i for i, r in comp.records.items() if r["partition"] == "train"]
    res = build(comp, tmp_path, "v0", ["m1"], ["train", "test"], {"all": pool}, {"m1": words}, {"m1": short},
                {"m1": ["<s>"]}, {"v2tools_commit": "abc"})
    assert res == {"files": 6, "excluded": 0}
    test_rows = [json.loads(l) for l in (tmp_path / "m1.all.test.F2.jsonl").read_text().splitlines()]
    assert len(test_rows) == 1 and test_rows[0]["answer"] is None and test_rows[0]["pool_sha256"] == pool_sha256(pool)
    assert set(test_rows[0]) == {"record_id", "partition", "fmt", "model", "pool_sha256", "messages", "answer", "provenance"}
    train_rows = [json.loads(l) for l in (tmp_path / "m1.all.train.F0.jsonl").read_text().splitlines()]
    assert train_rows[0]["answer"].startswith(TICKS + "php\n// FILE: ")
    sums = dict(reversed(l.split("  ")) for l in (tmp_path / "PROMPTS_SHA256SUMS").read_text().splitlines())
    assert set(sums) == {p.name for p in tmp_path.iterdir()} - {"PROMPTS_SHA256SUMS"} and "pools.json" in sums
    assert json.loads((tmp_path / "pools.json").read_text()) == {"all": {"sha256": pool_sha256(pool), "ids": pool}}
    assert all(hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() == h for n, h in sums.items())
    meta = json.loads((tmp_path / "PROMPTS_META.json").read_text())
    assert meta["compose"]["variant"] == "v0" and meta["v2tools_commit"] == "abc" and meta["compose"]["k"] == 8
    # the test answer uses $this->checkout and computeTotal(), both written with it (at its t_vis): infrastructure
    assert json.loads((tmp_path / "infra.json").read_text()) == {
        "Acme/Shop": [["src/Behat/Context/ShopContext.php", "checkout"], ["src/Behat/Context/ShopContext.php", "computeTotal"]]}


def test_control_tokens_exclude_a_record_for_that_model_only(comp, tmp_path):
    pool = [i for i, r in comp.records.items() if r["partition"] == "train"]
    build(comp, tmp_path, "v0", ["m1", "m2"], ["train"], {"all": pool}, {"m1": words, "m2": words},
          {"m1": short, "m2": short}, {"m1": ["iAddToCart"], "m2": []}, {})
    ex = [json.loads(l) for l in (tmp_path / "excluded.jsonl").read_text().splitlines()]
    assert {(e["model"], e["reason"]) for e in ex} == {("m1", "control_token")} and len(ex) == 3
    assert (tmp_path / "m1.all.train.F0.jsonl").read_text() == ""
    assert (tmp_path / "m2.all.train.F0.jsonl").read_text() != ""


def test_held_out_exclusions_use_the_union_of_tokenizers(comp):
    q = next(r for r in comp.records.values() if r["partition"] == "test")
    pool = [i for i, r in comp.records.items() if r["partition"] == "train"]
    both_short = {"a": short, "b": short}
    assert held_out_exclusions(comp, [q], {"all": pool}, both_short, {"a": words, "b": words}, {"a": [], "b": []}) == []
    long_b = {"a": short, "b": lambda msgs, gold: 20000}
    assert held_out_exclusions(comp, [q], {"all": pool}, long_b, {"a": words, "b": words}, {"a": [], "b": []}) == [
        {"record_id": q["id"], "model": None, "fmt": None, "reason": "too_long_union"}]


def test_a_directory_of_test_prompts_alone_still_lists_every_infrastructure_user(comp, tmp_path):
    q = next(r for r in comp.records.values() if r["partition"] == "test")
    user = dict(next(r for r in comp.records.values() if r["partition"] == "train"), id="Acme/Shop|features/infra.feature|3|s",
                feature_path="features/infra.feature")
    user["create"] = [dict(user["create"][0], code="#[When('x')]\npublic function x(): void\n{\n    $this->checkout->go();\n}")]
    recs = {**comp.records, user["id"]: user}
    c2 = Composer(comp.contexts, comp.catalog, comp.forbidden, recs, comp.params)
    pool = [i for i, r in recs.items() if r["partition"] == "train"]
    build(c2, tmp_path, "v0", ["m1"], ["test"], {"all": pool}, {"m1": words}, {"m1": short}, {"m1": []}, {})
    ex = [json.loads(l) for l in (tmp_path / "excluded.jsonl").read_text().splitlines()]
    assert ex == [{"record_id": user["id"], "model": None, "fmt": None, "reason": "uses_test_infra"}]
    assert json.loads((tmp_path / "pools.json").read_text())["all"]["ids"] == sorted(set(pool) - {user["id"]})
    assert q["id"] in (tmp_path / "m1.all.test.F2.jsonl").read_text()
