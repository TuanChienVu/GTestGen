import importlib.util
import json
from pathlib import Path

import pytest

from tests.phase3_fixture import catalog, make_repo, record
from v2tools.compose import Composer, Params
from v2tools.prompts import build
from v2tools.state import RepoContext

ROOT = Path(__file__).resolve().parents[1]


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return make_repo(tmp_path_factory.mktemp("repo"))


def test_scan_prompts_accepts_honest_files_and_catches_corruptions(repo, tmp_path):
    bare, head = repo
    cat = catalog()
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    q = record(cat)
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])
    tr = record(cat, "train", "Acme/Shop|features/cart.feature|3|s")
    tr.update(feature_path="features/cart.feature", steps=["I add \"x\" to the cart"], step_kinds=["create"], neighbours=[],
              t_vis=1640995200, create=[{"fn_id": add["fn_id"], "path": add["path"], "code": add["code"], "preexisting": False}],
              create_def_ids=[add["annotations"][0]["def_id"]])
    recs = {q["id"]: q, tr["id"]: tr}
    comp = Composer({"Acme/Shop": ctx}, {"Acme/Shop": cat}, {}, recs, Params())
    out = tmp_path / "p"
    build(comp, out, "v0", ["m"], ["train", "test"], {"all": [tr["id"]]}, {"m": words}, {"m": lambda msgs, gold: 10},
          {"m": []}, {})
    scan = script("scan_prompts").scan
    assert scan(out, recs, {"Acme/Shop": cat}) == {}
    f = out / "m.all.test.F2.jsonl"
    row = json.loads(f.read_text().splitlines()[0])
    row["messages"][0]["content"] += "\n\nProject knowledge:\n#[Then('I should see the total :total')]\n$this->iShouldSeeTotal();"
    row["answer"] = "leak"
    f.write_text(json.dumps(row) + "\n")
    found = scan(out, recs, {"Acme/Shop": cat})
    assert found["held_out_answer"] == 1 and found["create_annotation_line"] == 1 and found["answer_method_name"] == 1
    assert found["sums_mismatch"] == 1
