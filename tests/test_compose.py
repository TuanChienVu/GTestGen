import copy

import pytest

from tests.phase3_fixture import SHOP, T2022, T2024, catalog, make_repo, record
from v2tools.compose import TICKS, Answer, Composer, Params, aligned_slices, gold_answer
from v2tools.state import RepoContext

OTHER = "src/Behat/Context/ApiShopContext.php"


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    bare, head = make_repo(tmp_path_factory.mktemp("repo"))
    cat = catalog()
    # a function in another (unversioned) file with the SAME pattern as the answer: a pattern twin
    twin = copy.deepcopy(next(f for f in cat.values() if "iShouldSeeTotal" in f["code"]))
    twin.update(fn_id=f"{OTHER}:20", path=OTHER, code=twin["code"].replace("iShouldSeeTotal", "apiTotal"))
    twin["annotations"][0]["def_id"] = f"{OTHER}:17"
    cat[twin["fn_id"]] = twin
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    q = record(cat)
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])

    def train(rid, fn, feature, steps, t):
        return dict(record(cat, "train", rid), feature_path=feature, steps=steps, step_kinds=["create"] * len(steps),
                    scenario="Scenario: x\n  " + "\n  ".join(steps), t_vis=t, scenario_time=t, neighbours=[],
                    create=[{"fn_id": fn["fn_id"], "path": fn["path"], "code": fn["code"], "preexisting": False}],
                    create_def_ids=[fn["annotations"][0]["def_id"]])
    e_ok = train("Acme/Shop|features/cart.feature|3|s", add, "features/cart.feature", ["I add \"x\" to the cart"], T2022)
    e_same_file = train("Acme/Shop|features/total.feature|9|s", add, "features/total.feature", ["I add \"y\" to the cart"], T2022)
    e_pattern = train("Acme/Shop|features/api.feature|3|s", twin, "features/api.feature", ["I should see the total \"1\""], T2024)
    recs = {r["id"]: r for r in (q, e_ok, e_same_file, e_pattern)}
    comp = Composer({"Acme/Shop": ctx}, {"Acme/Shop": cat}, {}, recs, Params(budget_examples=400, budget_knowledge=400))
    return comp, q, [e_ok["id"], e_same_file["id"], e_pattern["id"]]


def test_gold_answer_is_one_fenced_block_per_file_with_a_file_line():
    r = {"language": "php", "t_vis": 1, "create": [{"path": "a.php", "code": "f1", "preexisting": False},
                                                   {"path": "b.php", "code": "g", "preexisting": False},
                                                   {"path": "a.php", "code": "f2", "preexisting": False}]}
    assert gold_answer(r) == f"{TICKS}php\n// FILE: a.php\nf1\n\nf2\n{TICKS}\n\n{TICKS}php\n// FILE: b.php\ng\n{TICKS}"


def test_f0_is_one_user_turn_with_instructions_and_undefined_steps_only(env):
    comp, q, pool = env
    msgs, prov = comp.compose_prompt(q, "F0", words, pool)
    assert [m["role"] for m in msgs] == ["user"]
    text = msgs[0]["content"]
    assert f"- {SHOP} (class ShopContext)" in text
    assert "Undefined steps:\n- I should see the total \"5\"" in text
    assert "Examples from this repository" not in text and "Project knowledge" not in text
    assert prov == {"examples": [], "knowledge": [], "skeleton": [], "dropped": []}


def test_f1_examples_exclude_same_file_and_shared_create_pattern(env):
    comp, q, pool = env
    _, prov = comp.compose_prompt(q, "F1", words, pool)
    assert prov["examples"] == ["Acme/Shop|features/cart.feature|3|s"]
    assert prov["knowledge"] == [] and prov["skeleton"] == []


def test_f2_knowledge_hides_the_answer_and_shows_visible_code(env):
    comp, q, pool = env
    msgs, prov = comp.compose_prompt(q, "F2", words, pool)
    text = msgs[0]["content"]
    ids = {f.rsplit(":", 1)[0] for f in prov["knowledge"]}
    assert ids == {SHOP} and not any("iShouldSeeTotal" in comp.catalog["Acme/Shop"][f]["code"] for f in prov["knowledge"])
    assert "iShouldSeeTotal" not in text and "I should see the total :total" not in text
    assert "Class outline of " + SHOP in text and "$checkout" not in text
    assert len(prov["skeleton"]) == 2  # iOpenTheShop and iAddToCart signatures


def test_preexisting_item_is_shown_without_its_create_annotation(env):
    comp, q, pool = env
    add = next(f for f in comp.catalog["Acme/Shop"].values() if "iAddToCart" in f["code"])
    r = dict(q, create=[{"fn_id": add["fn_id"], "path": SHOP, "code": add["code"], "preexisting": True}],
             create_def_ids=[add["annotations"][0]["def_id"]], steps=["I add \"x\" to the cart"], step_kinds=["create"])
    msgs, prov = comp.compose_prompt(r, "F2", words, pool)
    text = msgs[0]["content"]
    assert add["fn_id"] in prov["knowledge"]  # its old body may be shown ...
    assert "I add :product to the cart" not in text  # ... but never the CREATE pattern, although it is older than T
    assert "Acme/Shop|features/cart.feature|3|s" not in prov["examples"]  # the example shares that CREATE pattern


def test_answer_method_name_in_a_visible_body_drops_that_function(env):
    comp, q, pool = env
    code = "#[Then('I should see the total :total')]\npublic function open(string $total): void\n{\n}"
    r = dict(q, create=[dict(q["create"][0], code=code)])
    _, prov = comp.compose_prompt(r, "F2", words, pool)
    assert {"kind": "knowledge", "id": next(f for f, x in comp.catalog["Acme/Shop"].items() if "iOpenTheShop" in x["code"]),
            "rule": "name"} in prov["dropped"]


def test_text_rule_catches_slices_names_and_signature_lines():
    code = "x" * 450
    ans = Answer(set(), set(), {"doIt"}, {"public function doIt(): void"}, aligned_slices(code), set(), set())
    assert Composer.text_rule(ans, "prefix " + "x" * 400) == "slice"
    assert Composer.text_rule(ans, "$this->doIt();") == "name"
    assert Composer.text_rule(ans, "$this->doItAgain();") is None
    assert Composer.text_rule(ans, "  public function   doIt(): void\n") in ("name", "signature")


def test_budgets_are_respected_and_ablations_select_one_part(env):
    comp, q, pool = env
    tiny = Composer(comp.contexts, comp.catalog, comp.forbidden, comp.records, Params(budget_examples=5, budget_knowledge=5))
    _, prov = tiny.compose_prompt(q, "F2", words, pool)
    assert prov["examples"] == [] and prov["knowledge"] == [] and prov["skeleton"] == []
    only_a = Composer(comp.contexts, comp.catalog, comp.forbidden, comp.records, Params(budget_knowledge=400, mode="4a"))
    msgs, prov = only_a.compose_prompt(q, "F2", words, pool)
    assert prov["knowledge"] == [] and prov["skeleton"] and "Existing step definitions" not in msgs[0]["content"]
    only_b = Composer(comp.contexts, comp.catalog, comp.forbidden, comp.records, Params(budget_knowledge=400, mode="4b"))
    msgs, prov = only_b.compose_prompt(q, "F2", words, pool)
    assert prov["skeleton"] == [] and prov["knowledge"] and "Class outline" not in msgs[0]["content"]
