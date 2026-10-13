import pytest

from tests.phase3_fixture import SHOP, T2020, T2022, T2024, catalog, make_repo, record
from v2tools.state import RepoContext, future_ref, render_visible, self_contained, skeleton
from v2tools.symtime import first_times


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    bare, head = make_repo(tmp_path_factory.mktemp("repo"))
    return RepoContext("Acme/Shop", "php", bare, head, catalog())


def answer_id(ctx):
    return next(f for f, row in ctx.fns.items() if "iShouldSeeTotal" in row["code"])


def test_first_times_use_declaration_contexts_across_versions(ctx):
    t = first_times(ctx.git_dir, ctx.commit, SHOP, "php")
    assert t["file"] == T2020
    assert t["names"]["indexPage"] == T2020 and t["names"]["iOpenTheShop"] == T2020
    assert t["names"]["cart"] == T2022 and t["names"]["checkout"] == T2024
    assert t["names"]["computeTotal"] == T2024


def test_skeleton_masks_future_members_and_answers_and_keeps_inherited_non_private(ctx):
    gone = {answer_id(ctx)}
    s = skeleton(ctx, SHOP, T2024, gone)
    assert "private Cart $cart," in s and "private IndexPage $indexPage," in s
    assert "$checkout" not in s  # promoted parameter first declared at T: dropped from the constructor
    assert "iOpenTheShop" in s and "iAddToCart" in s
    assert "iShouldSeeTotal" not in s and "computeTotal" not in s
    assert "#[" not in s and "Given(" not in s  # signatures only, no annotations
    assert "// from BaseContext (src/Behat/Context/BaseContext.php)" in s
    assert "helperOld" in s and "$sharedStorage" in s and "secret" not in s
    assert s.startswith("namespace App\\Behat\\Context;\n\nfinal class ShopContext extends BaseContext\n{")


def test_skeleton_at_an_earlier_time_masks_more(ctx):
    s = skeleton(ctx, SHOP, T2022, set())
    assert "$cart" not in s and "iAddToCart" not in s and "iOpenTheShop" in s


def test_new_file_has_no_skeleton(ctx):
    assert skeleton(ctx, SHOP, T2020, set()) is None
    assert ctx.is_new_file(SHOP, T2020) and not ctx.is_new_file(SHOP, T2020 + 1)


def test_render_visible_drops_create_annotations_at_any_time_and_future_annotations(ctx):
    fid = next(f for f, row in ctx.fns.items() if "iAddToCart" in row["code"])
    fn = ctx.fns[fid]
    cdef = frozenset({fn["annotations"][0]["def_id"]})
    assert "#[When" in render_visible(fn, T2024, frozenset())
    assert "#[When" not in render_visible(fn, T2024, cdef)  # the record's own CREATE pattern, although older than T
    assert "#[When" not in render_visible(fn, T2022, frozenset())  # not strictly before T


def test_future_ref_flags_a_body_that_uses_a_future_member(ctx):
    fn = ctx.fns[next(f for f, row in ctx.fns.items() if "iAddToCart" in row["code"])]
    assert future_ref(ctx, fn, T2022) and not future_ref(ctx, fn, T2024)


def test_self_contained_depends_on_what_the_answer_uses(ctx):
    r = record(ctx.fns)
    assert not self_contained(ctx, r, {answer_id(ctx)})  # uses $this->checkout, first declared at T
    r2 = dict(r, create=[dict(r["create"][0], code="#[Then('x')]\npublic function x(): void\n{\n    $this->cart->go();\n}")])
    assert self_contained(ctx, r2, {answer_id(ctx)})
