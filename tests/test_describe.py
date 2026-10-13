import dataclasses
import hashlib
from collections import Counter

import pytest

import v2tools.describe as d
from tests.phase3c_fixture import block, make_prep
from v2tools.describe import (InputError, call_overlap, conventions, dose, hallucinated, naming,
                              open_resolution_rates, rouge_l, row, similarity, subsets, tokens, unread_steps)
from v2tools.evaluate import score_output
from v2tools.extract import extract


@pytest.fixture()
def prep(tmp_path):
    return make_prep(tmp_path)


def test_naming():
    assert (naming("iAddToCart"), naming("i_add_to_cart"), naming("AddToCart")) == ("camel", "snake", "pascal")


def test_gold_is_similar_and_follows_conventions(prep):
    g = prep.gold()
    assert similarity(prep, {prep.targets[0]: g.items[next(iter(g.items))][2]}) == {"rouge_l": 1.0, "exact": True}
    assert conventions(prep, g) == {"conv_style": 1.0, "conv_pattern": 1.0, "conv_naming": 1.0}


def test_docblock_regex_snake_case_breaks_conventions(prep):
    s = score_output(prep, block("/**\n * @Then /^I should see the total \"([^\"]*)\"$/\n */\npublic function i_see_total(string $t): void\n{\n    $this->checkout->assertTotal($t);\n}"))
    assert conventions(prep, s) == {"conv_style": 0.0, "conv_pattern": 0.0, "conv_naming": 0.0}


def test_hallucinated_counts_members_not_in_v_r(prep):
    s = score_output(prep, block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->ghostService->x($total);\n    $this->cart->add($total);\n}"))
    assert hallucinated(prep, s) == (1, 2)


def test_hallucinated_reads_v_r_through_t_view(prep, monkeypatch):
    """Rule 5: V(r) is what 3A's state.t_view shows; with nothing shown (a file new at t_vis) $this->cart is
    invented."""
    import v2tools.describe as d
    s = score_output(prep, block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->cart->add($total);\n}"))
    assert hallucinated(prep, s) == (0, 1)
    assert hallucinated(prep, s, frozenset({("src/Behat/Context/ShopContext.php", "cart")})) == (1, 1)  # masked
    monkeypatch.setattr(d, "t_view", lambda *a, **k: None)
    assert hallucinated(prep, s) == (1, 1)


def test_dose_counts_gold_members_shown_in_block_4(prep):
    assert dose(prep, "Scenario...\nProject knowledge:\n private Checkout $checkout,\n") == 1
    assert dose(prep, "no knowledge block, checkout") == 0


def test_row_has_the_sap_columns(prep):
    text = block(prep.gold().items[next(iter(prep.gold().items))][2])
    s = score_output(prep, text)
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 1, "ca_ref_v3names": 1, "ca_ref_top4": 0}
    excl = {"excl_bv": False, "excl_ca": False}
    sealed = {"post_cutoff_all": [prep.record["id"]], "time_coherent_30d": [], "residual_clean": []}
    run = {"model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "main", "prompt_tokens": 9000,
           "text": text, "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64}
    out = row(prep, run, s, extract(text, prep.targets).by_path(), refs_, excl, sealed)
    assert (out["ca"], out["ca_defined"], out["ca_ref_v3m"], out["ca_ref_v3all"], out["bv"], out["len_bucket"]) == (1, True, 0, 1, 1, "gt8192")
    assert (out["ca_ref_v3names"], out["ca_ref_top4"]) == (1, 0)  # D11 (a), (d): the lazy references v3names and top4
    with pytest.raises(InputError, match="ca_ref_v3names"):
        row(prep, run, s, {}, {"ca_ref_v3m": 0, "ca_ref_v3all": 1, "ca_ref_top4": 0}, excl, sealed)
    with pytest.raises(InputError, match="ca_ref_top4"):
        row(prep, run, s, {}, {"ca_ref_v3m": 0, "ca_ref_v3all": 1, "ca_ref_v3names": 1}, excl, sealed)
    assert out["scenario_key"] == "Acme/Shop|features/total.feature|3" and out["sub_post_cutoff_all"] is True
    # the gold's ("self", computeTotal) is a future helper whose body makes no project call: n_calls 1
    assert (out["excl_bv"], out["excl_ca"], out["prompts_sha256"]) == (False, False, "a" * 64)
    assert (out["n_calls"], out["call_precision"]) == (1, 1.0)
    assert out["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    er = {"er": 1, "er_status": "PASS", "trace_action": 1, "trace_then": None, "n_then_evaluable": 0, "er_trace": None}
    out = row(prep, {"model": "m", "seed": "s11", "condition": "ft_only"}, s, {}, refs_, excl, sealed, er=er)
    assert (out["er"], out["er_status"], out["trace_action"], out["trace_then"], out["er_trace"]) == (1, "PASS", 1, None, None)
    skip = {**er, "er": 0, "er_status": "SKIP", "trace_action": 0, "er_trace": 0}
    assert row(prep, run, s, {}, refs_, excl, sealed, er=skip)["er"] == 0
    with pytest.raises(InputError, match="SKIP"):
        row(prep, run, s, {}, refs_, excl, sealed, er={**skip, "er": 1})
    with pytest.raises(InputError, match="trace_action"):
        row(prep, run, s, {}, refs_, excl, sealed, er={"er": 1, "er_status": "PASS"})
    with pytest.raises(InputError, match="excl_ca"):
        row(prep, run, s, {}, refs_, {"excl_bv": False}, sealed)


def test_row_copies_line_sha256_from_the_manifest_for_provenance(prep):
    """A scores row carries `line_sha256` = the manifest ok line's line_sha256, copied like decision_sha256 and
    gen_code_sha256 (describe.PROVENANCE) from score_run's `run` dict; None for a probe row with no manifest (the field
    simply absent from `run`)."""
    from v2tools.describe import PROVENANCE
    assert "line_sha256" in PROVENANCE
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": None, "ca_ref_top4": None}
    excl = {"excl_bv": False, "excl_ca": False}
    sealed = {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []}
    s = score_output(prep, "")
    out = row(prep, {"line_sha256": "a" * 64}, s, {}, refs_, excl, sealed)
    assert out["line_sha256"] == "a" * 64
    assert row(prep, {}, s, {}, refs_, excl, sealed)["line_sha256"] is None


def test_run_name_adds_variant_unless_main():
    from v2tools.evaluate import run_name
    assert run_name({"model": "deepseek", "seed": "s42", "condition": "full_aware"}) == "deepseek.s42.full_aware"
    assert run_name({"model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": "lc600"}) == "deepseek.s42.full_aware.lc600"


def test_rouge_l_and_tokens():
    assert rouge_l(tokens("a b c"), tokens("a b c")) == 1.0
    assert rouge_l(tokens("a b c d"), tokens("a c")) == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_subsets_flags():
    r = {"id": "x", "create": [{"post_cutoff_code": True, "preexisting": False}], "t_vis": 100 + 31 * 86400, "scenario_time": 100}
    sealed = {"post_cutoff_all": ["x"], "time_coherent_30d": [], "residual_clean": {"test": ["x"], "aux_test": []}}
    assert subsets(r, sealed) == {"sub_post_cutoff_all": True, "sub_time_coherent_30d": False, "sub_residual_clean": True}
    with pytest.raises(InputError, match="residual_clean"):
        subsets(r, {"post_cutoff_all": [], "time_coherent_30d": []})


def test_open_resolution_rates_count_parsed_runs_only():
    rows = [{"repo": "A", "condition": "c", "bv_syntax": 1, "open_resolution": True},
            {"repo": "A", "condition": "c", "bv_syntax": 1, "open_resolution": False},
            {"repo": "A", "condition": "c", "bv_syntax": 0, "open_resolution": False}]
    assert open_resolution_rates(rows) == {"A|c": {"open": 1, "n": 2, "rate": 0.5}}


def test_conventions_survive_code_cut_off_at_the_token_cap(prep):
    for code in ("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal(",
                 "#[Then('I should see the total :total')]\n"):
        s = score_output(prep, block(code))
        assert not s.conds["syntax"] and conventions(prep, s) == {"conv_style": None, "conv_pattern": None, "conv_naming": None}



def test_call_overlap_reads_the_expanded_call_sets(prep):
    """n_calls and call_precision read the call sets after helper expansion (describe.call_overlap)."""
    from v2tools.describe import call_overlap
    out = ("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n"
           "    $this->probeHelper();\n}\n\nprivate function probeHelper(): void\n{\n"
           "    $this->checkout->assertTotal('x');\n    $this->cart->add('y');\n}")
    assert call_overlap(prep, score_output(prep, block(out))) == {"n_calls": 2, "call_precision": 0.5}


STEP = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{{\n    {}\n}}"


def test_call_overlap_folds_php_method_names_only(prep):
    """PHP call matching ignores case, as calls.agreement: n_calls and call_precision compare (receiver,
    casefold(method)) for PHP, so two calls that differ only in the method's case are one call; the receiver, a
    property name, stays as written."""
    from v2tools.describe import call_overlap
    up = score_output(prep, block(STEP.format("$this->checkout->AssertTotal($total);")))
    assert call_overlap(prep, up) == {"n_calls": 1, "call_precision": 1.0}
    both = score_output(prep, block(STEP.format("$this->checkout->AssertTotal($total);\n"
                                                "    $this->checkout->assertTotal($total);")))
    assert call_overlap(prep, both) == {"n_calls": 1, "call_precision": 1.0}
    prop = score_output(prep, block(STEP.format("$this->Checkout->assertTotal($total);")))
    assert call_overlap(prep, prop) == {"n_calls": 1, "call_precision": 0.0}


def test_hallucinated_matches_php_method_names_in_any_case(prep):
    """As in condition 6, a PHP METHOD call is not hallucinated when its casefold is a method V(r) shows or a method the
    output adds (`$this->PROBEFILL()`); a property name stays case-sensitive, and a call to a name V(r) shows only as a
    property is still invented; a parameter an output `__CONSTRUCT` promotes is a member the output defines, as with
    `__construct`."""
    visible = score_output(prep, block(STEP.format("$this->IADDTOCART($total);\n    $this->cart->add($total);")))
    assert hallucinated(prep, visible) == (0, 2)
    helper = "\n\nprivate function probeFill(string $t): void\n{\n    $this->cart->add($t);\n}"
    own = score_output(prep, block(STEP.format("$this->PROBEFILL($total);") + helper))
    assert hallucinated(prep, own) == (0, 2)
    assert hallucinated(prep, score_output(prep, block(STEP.format("$this->Cart->add($total);")))) == (1, 1)
    assert hallucinated(prep, score_output(prep, block(STEP.format("$this->CART($total);")))) == (1, 1)
    prep._shown, prep._future = {prep.targets[0]: set()}, None  # the constructor is future: the output's replaces it
    for name in ("__construct", "__CONSTRUCT"):
        ctor = f"\n\npublic function {name}(private Foo $probeSvc)\n{{\n}}"
        s = score_output(prep, block(STEP.format("$this->probeSvc->go($total);") + ctor))
        assert s.detail["replaced"] == 1 and hallucinated(prep, s) == (0, 1), name



def test_row_flags_a_fence_closed_early(prep):
    """`fence_truncated` = extract.Extraction.truncated of the run's text (None when it has none). A heredoc's rest that
    closes its bare fence is not flagged (§1.4, a documented limit fixed 28/09/2026); the same output cut before that
    last fence is."""
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": None, "ca_ref_top4": None}
    excl = {"excl_bv": False, "excl_ca": False}
    sealed = {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []}
    ticks = "`" * 3
    good = block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n"
                 "    $this->checkout->assertTotal($total);\n}")
    cut = good.replace("    $this->checkout", f"    $s = <<<EOT\n{ticks}\nx\n{ticks}\nEOT;\n    $this->checkout")
    for text, want in ((good, False), (cut, False), (cut.rsplit("\n", 1)[0], True)):
        blocks = extract(text, prep.targets).by_path()
        out = row(prep, {"text": text}, score_output(prep, text), blocks, refs_, excl, sealed)
        assert out["fence_truncated"] is want
    assert row(prep, {}, score_output(prep, good), {}, refs_, excl, sealed)["fence_truncated"] is None


def test_ca_prec_is_ca_with_a_call_precision_floor():
    """CA_prec(τ) (D11 (c), fixed 25/09/2026) = CA × [call_precision ≥ τ], call_precision as describe.call_overlap
    (after helper expansion); no call (call_precision None) gives 0. τ = 0.25 is the pre-registered one; 0.1 and 0.5 are
    reported beside it for the controls."""
    from v2tools.describe import CA_PREC_TAUS, ca_prec
    assert CA_PREC_TAUS == (0.1, 0.25, 0.5)
    assert ca_prec(1, 0.25, 0.25) == 1 and ca_prec(1, 0.2499, 0.25) == 0 and ca_prec(0, 1.0, 0.1) == 0
    assert ca_prec(1, None, 0.1) == 0 and ca_prec(True, 1.0, 0.5) == 1


def test_ca_prec_exclusions_list_gold_and_gold_helpers_only():
    """D11 (c), the CA_prec check made before the analysis plan was locked: every controls row of gold or gold-helpers
    with CA 1 whose CA_prec at τ = 0.25 is 0 (expected: none)."""
    from v2tools.describe import ca_prec_excluded
    rows = [{"record_id": "a", "repo": "R", "control": "gold", "ca": 1, "call_precision": 0.2, "n_calls": 5},
            {"record_id": "b", "repo": "R", "control": "gold-helpers", "ca": 1, "call_precision": None, "n_calls": 0},
            {"record_id": "c", "repo": "R", "control": "gold", "ca": 1, "call_precision": 0.25, "n_calls": 4},
            {"record_id": "d", "repo": "R", "control": "gold", "ca": 0, "call_precision": 0.0, "n_calls": 1},
            {"record_id": "e", "repo": "R", "control": "stub-v3names", "ca": 1, "call_precision": 0.01, "n_calls": 90}]
    assert [(x["record_id"], x["control"]) for x in ca_prec_excluded(rows)] == [("a", "gold"), ("b", "gold-helpers")]


def test_row_counts_php_step_styles_and_a_stripped_open_tag(prep):
    """Descriptive PHP columns of a scores row with a text (None otherwise): php_tag_stripped (a block had a `<?php` /
    `?>` tag that insert.php_open_tag removed), and php_docblock_steps / php_attribute_steps, the output's step
    definitions by style in the method items BV binds from (evaluate's Score.detail["step_styles"]): the ER image (Behat
    4.0.0-alpha1) registers attribute steps only (26/09/2026, §2.3), while BV and CA take both forms."""
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": 0, "ca_ref_top4": 0}
    excl = {"excl_bv": False, "excl_ca": False}
    sealed = {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []}
    both = ("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n"
            "    $this->checkout->assertTotal($total);\n}\n\n/**\n * @Then I should see nothing at all\n */\n"
            "public function u(): void\n{\n    $this->cart->add('x');\n}\n\n/**\n * @When I do nothing\n */\n"
            "public function v(): void\n{\n    $this->cart->add('y');\n}")
    for code, tag in ((both, False), ("<?php\n" + both, True)):
        text = block(code)
        s = score_output(prep, text)
        out = row(prep, {"text": text}, s, extract(text, prep.targets).by_path(), refs_, excl, sealed)
        assert (out["php_tag_stripped"], out["php_docblock_steps"], out["php_attribute_steps"]) == (tag, 2, 1), code
    text = block(both) + "\n" + block("<?php\n\n/**\n * @When I do more\n */\npublic function w(): void\n{\n}")
    s = score_output(prep, text)  # §1.4, tags are stripped per block: the tag of the second fence counts too
    out = row(prep, {"text": text}, s, extract(text, prep.targets).by_path(), refs_, excl, sealed)
    assert (out["php_tag_stripped"], out["php_docblock_steps"], out["php_attribute_steps"]) == (True, 3, 1)
    out = row(prep, {}, s, {}, refs_, excl, sealed)
    assert (out["php_tag_stripped"], out["php_docblock_steps"], out["php_attribute_steps"]) == (None, None, None)


def test_a_method_that_replaces_a_future_one_is_output_defined(prep):
    """A member is hallucinated (§2.4) only when it is neither in V(r) nor defined by the output: every member the
    output's own items declare is output-defined, a method that replaces a future or shell method included, although the
    ER tree already had it; writing the developer's helper name then counts as writing it under a new name (gold-helpers
    vs gold-helpers-renamed: 116/429 test records differed)."""
    step = STEP.format("$this->checkout->assertTotal($this->computeTotal($total));")
    helper = "\n\nprivate function computeTotal(string $t): string\n{\n    return $t;\n}"
    same = score_output(prep, block(step + helper))
    renamed = score_output(prep, block(step.replace("computeTotal", "probeTotal")
                                       + "\n\nprivate function probeTotal(string $t): string\n{\n    return $t;\n}"))
    assert same.detail["replaced"] == 1 and renamed.detail["replaced"] == 0
    assert hallucinated(prep, same) == hallucinated(prep, renamed) == (1, 2)  # checkout: added with the answer
    upper = score_output(prep, block(step.replace("$this->computeTotal", "$this->COMPUTETOTAL") + helper))
    assert hallucinated(prep, upper) == (1, 2)  # PHP: the output's method, called in another case


from tests.phase3c_fixture import JPATH, make_jprep  # noqa: E402  (the Java twin of the fixture)


@pytest.fixture()
def jprep(tmp_path):
    return make_jprep(tmp_path)


def test_a_java_method_that_replaces_a_future_one_is_output_defined(jprep):
    """The same on Java (no case folding): the gold plus its future helper computeTotal (replaced) and the same with the
    helper renamed count the same invented members."""
    gold = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n' \
           '    cart.assertTotal(computeTotal(total));\n}'
    helper = "\n\nprivate String computeTotal(String total) {\n    return total;\n}"
    fence = lambda code: f"{'`' * 3}java\n// FILE: {JPATH}\n{code}\n{'`' * 3}"
    same = score_output(jprep, fence(gold + helper))
    renamed = score_output(jprep, fence((gold + helper).replace("computeTotal", "probeTotal")))
    assert same.detail["replaced"] == 1 and hallucinated(jprep, same) == hallucinated(jprep, renamed) == (0, 2)



def test_call_overlap_is_the_copy_the_evaluate_and_controls_tests_use_on_java(jprep):
    """The copy is pinned on Java too (27/09/2026), where calls are not folded (the `set` key): the gold (its future
    helper computeTotal is read, so cart.total counts), the same with the helper written out, one with an extra call,
    and one calling assertTotal in another case, a different call on Java."""
    from tests.test_evaluate import call_precision
    gold = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n' \
           '    cart.assertTotal(computeTotal(total));\n}'
    helper = "\n\nprivate String computeTotal(String total) {\n    baseNew();\n    return total;\n}"
    fence = lambda code: f"{'`' * 3}java\n// FILE: {JPATH}\n{code}\n{'`' * 3}"
    extra = gold.replace("    cart.assertTotal", "    cart.open();\n    cart.assertTotal") + helper
    for code, want in ((gold, (2, 1.0)), (gold + helper, (2, 1.0)), (extra, (3, 0.6667)),
                       (gold.replace("assertTotal", "AssertTotal") + helper, (2, 0.5))):
        s = score_output(jprep, fence(code))
        assert call_overlap(jprep, s) == call_precision(jprep, s) == {"n_calls": want[0], "call_precision": want[1]}

def test_annotation_form_columns_read_the_outputs_own_blocks(prep):
    """php_docblock_steps, php_attribute_steps and conv_style count the step definitions the OUTPUT wrote (its own
    blocks), not the file after insertion: rewriting a preexisting shell keeps HEAD's annotation lines it lacks
    (insert.with_annotations, the kept-annotation rule), which must not change these descriptive counts (26/09/2026).
    Here the output rewrites the shell iAddToCart with a docblock step; insertion keeps its HEAD attribute line."""
    from v2tools.evaluate import prepare
    cart = next(fid for fid, f in prep.repo.fns.items() if "iAddToCart" in f["code"])
    shell_row = {"fn_id": cart, "path": prep.targets[0], "code": prep.repo.fns[cart]["code"], "preexisting": True,
                 "post_cutoff_code": False}
    r = {**prep.record, "create": [prep.record["create"][0], shell_row]}
    p = prepare(r, prep.repo, {"step_def_ids": [[i[0]] for _, i in prep.repo.suite_steps["s"][:2]]})
    shell = ("/**\n * @When I add :product to the cart\n */\npublic function iAddToCart(string $product): void\n{\n"
             "    $this->cart->add($product);\n}")
    text = block(STEP.format("$this->checkout->assertTotal($total);") + "\n\n" + shell)
    s = score_output(p, text)
    assert s.detail["replaced"] == 1 and s.detail["kept_annotations"] == 1
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": 0, "ca_ref_top4": 0}
    out = row(p, {"text": text}, s, extract(text, p.targets).by_path(), refs_, {"excl_bv": False, "excl_ca": False},
              {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []})
    assert (out["php_docblock_steps"], out["php_attribute_steps"], out["conv_style"]) == (1, 1, 0.5)


# ----- 26/09/2026: the describe-level half of the two equivalent step-annotation forms ------------------------------
# (the describe half of tests.test_evaluate.test_a_function_scores_the_same_in_either_annotation_form, split so that
# tests/test_evaluate.py need not import v2tools.describe; the Score/evaluate-level half -- conditions, binders, calls,
# helper expansion, CA -- stays there and is not repeated here.)


def test_a_function_in_either_annotation_form_agrees_on_conventions_and_hallucination(prep):
    """Since 26/09/2026 the SAME function scores identically in either annotation form on every describe-level field
    too: call_overlap (n_calls, call_precision), conv_pattern, conv_naming and hallucinated; only conv_style differs, by
    design (the share in the file's majority form): 1.0 for the file's majority style (here the attribute), 0.0 for the
    docblock. Turnip and regex patterns (N = every capture group, Behat), a `\\'` escape of the attribute literal
    included; Ruby `Then(/re/) do` and the parenthesised form too."""
    def dview(p, s):
        conv = {k: v for k, v in conventions(p, s).items() if k != "conv_style"}
        return (call_overlap(p, s), conv, hallucinated(p, s))
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    attr = "#[Then('I should see the total :total')]"
    body = gold.split(attr, 1)[1]
    forms = [(attr, "/**\n * @Then I should see the total :total\n */")]
    for pattern in ("I should see the (customer's) total :total", "/^I should see the (?:grand )?total \"(\\d+)\"$/"):
        forms.append(("#[Then('" + pattern.replace("'", "\\'") + "')]", "/**\n * @Then " + pattern + "\n */"))
    for a, d in forms:  # a `'` is written `\'` in the attribute literal
        sa, sd = score_output(prep, block(a + body)), score_output(prep, block(d + body))
        assert dview(prep, sa) == dview(prep, sd), a
        assert (conventions(prep, sa)["conv_style"], conventions(prep, sd)["conv_style"]) == (1.0, 0.0)
    from tests.test_evaluate import RPATH, fence, mini_prep
    rb = mini_prep("ruby", ["ruby_block"], RPATH, "Given(/^I open the shop$/) do\n  open_shop\nend\n")
    for re_ in ("/^the total is (\\d+)$/", "%r{^the total is (\\d+)$}"):  # the gold in renalware HEAD's form
        rb._gold = bare = score_output(rb, fence("ruby", RPATH, f"Then {re_} do |n|\n  check_total(n)\nend"))
        paren = score_output(rb, fence("ruby", RPATH, f"Then({re_}) do |n|\n  check_total(n)\nend"))
        assert dview(rb, paren) == dview(rb, bare), re_


def test_call_overlap_is_the_copy_the_evaluate_and_controls_tests_use(tmp_path):
    """tests.test_evaluate.call_precision copies describe.call_overlap so tests/test_evaluate.py and
    tests/test_controls.py need not import v2tools.describe; here the copy is pinned to it on a visible preexisting
    shell (D11; the gold's call set gains ("self", name); PHP folds the name's case), the gold control (gold_output)
    included."""
    from tests.test_evaluate import PUT_STEP, call_precision, shell_prep
    p = shell_prep(tmp_path)
    for call, want in (("$this->iAddToCart($product);", 1.0), ("$this->IADDTOCART($product);", 1.0),
                       ("self::iAddToCart($product);", 1.0), ("$this->iOpenTheShop();", 0.0)):
        s = score_output(p, block(PUT_STEP.format(call)))
        assert call_overlap(p, s) == call_precision(p, s) == {"n_calls": 1, "call_precision": want}, call
    raw = p.gold_output()
    assert call_overlap(p, raw) == call_precision(p, raw) == {"n_calls": 1, "call_precision": 1.0}



def test_row_counts_step_lines_the_reader_does_not_read(prep, jprep):
    r"""Descriptive columns (§2.2, pre-registered 27/09/2026, per condition): unread_step_annotations counts, in the
    heads of the output's own items, what looks like a step declaration but the reader itself does not take there
    through the file's imports (stepdefs.extract's `rejected`: an alias of a foreign class and a foreign qualified name
    count, a "then" in a docblock's prose does not, where the old pattern count missed the first two and counted the
    third); cuke_escaped_patterns and cuke_custom_types count the Cucumber-expression patterns read with an escaped
    `\(`, `\{` or `\/` and with a parameter type the Phase 1 binder does not know (its documented limits (a)-(d)); None
    without a text."""
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": 0, "ca_ref_top4": 0}
    excl, sealed = {"excl_bv": False, "excl_ca": False}, {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []}
    keys = ("unread_step_annotations", "cuke_escaped_patterns", "cuke_custom_types")

    def counts(p, text):
        out = row(p, {"text": text}, score_output(p, text), extract(text, p.targets).by_path(), refs_, excl, sealed)
        return tuple(out[k] for k in keys)
    body = "\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}"
    read = "#[Then('I should see the total :total')]"
    assert counts(prep, block(read + body)) == (0, 0, 0)
    assert counts(prep, block(read + "\n#[Other\\Then('I see :total')]\n#[\\behat\\step\\then('I get :total')]" + body)) \
        == (2, 0, 0)  # PHP has no Cucumber expressions
    assert counts(prep, block("/**\n * @Then I should see the total :total\n */" + body)) == (0, 0, 0)
    # two examples: an alias of a foreign class next to a read alias of Behat's; a docblock's prose
    aliases = "use Behat\\Step\\Then as T;\nuse Foo\\Bar\\Then as X;\n\n#[T('I should see the total :total')]\n#[X('I see :total')]"
    assert counts(prep, block(aliases + body)) == (1, 0, 0)
    assert counts(prep, block(read + "\n#[\\Foo\\Then('I get :total')]" + body)) == (1, 0, 0)
    prose = "/**\n * If the cart is empty, then (the total) is zero; when (not) empty, given (a) total.\n */\n"
    assert counts(prep, block(prose + read + body)) == (0, 0, 0)
    # §2.2 (28/09/2026): no annotation of a method cut mid-body counts, whatever its form and the order of the
    # annotations; a whole method before it is still read in that item, which never binds
    cut = "\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal("
    foreign = "#[\\Foo\\Then('I get :total')]"
    assert counts(prep, block(foreign + cut)) == (0, 0, 0)  # was 1
    assert counts(prep, block(foreign + "\n" + read + cut)) == (0, 0, 0)  # was 1
    assert counts(prep, block(read + "\n" + foreign + cut)) == (0, 0, 0)
    assert counts(prep, block(read + body + "\n\n" + read + cut)) == (1, 0, 0)  # was 0: the reader stopped there
    fence = lambda code: f"{'`' * 3}java\n// FILE: {JPATH}\n{code}\n{'`' * 3}"
    jbody = "\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(total);\n}"
    assert counts(jprep, fence('@Then("I should see the total {string}")' + jbody)) == (0, 0, 0)
    assert counts(jprep, fence('@Then("I should see the total {string}")\n@And("I see {string}")' + jbody)) == (1, 0, 0)
    jcut = "\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal("
    assert counts(jprep, fence('@foo.Then("I see {string}")' + jcut)) == (0, 0, 0)  # was 1
    assert counts(jprep, fence('@foo.Then("I see {string}")\n@Then("I should see the total {string}")' + jcut)) == \
        (0, 0, 0)
    # the `{` of a later annotation's {string} is not the cut method's body (unread 1 before)
    assert counts(jprep, fence('@Then("I should see the total {string}")\n@Then("I see the sum {string}")' + jcut)) == \
        (0, 0, 0)
    assert counts(jprep, fence('@Then("I should see the \\\\(net\\\\) total {string}")' + jbody)) == (0, 1, 0)
    assert counts(jprep, fence('@Then("I should see the total {money}")\n@Given("I pay {string} {amount}")' + jbody)) \
        == (0, 0, 2)
    out = row(prep, {}, score_output(prep, block(read + body)), {}, refs_, excl, sealed)
    assert tuple(out[k] for k in keys) == (None, None, None)


# ----- open-path names and the describe rules (04/10/2026), each pinned by a counter-example ------------------------

REFS = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": 0, "ca_ref_top4": 0}
EXCL = {"excl_bv": False, "excl_ca": False}
SEALED = {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []}
UNREAD = ("unread_step_annotations", "cuke_escaped_patterns", "cuke_custom_types")


def jfence(code):
    return f"{'`' * 3}java\n// FILE: {JPATH}\n{code}\n{'`' * 3}"


def test_a_static_import_or_lombok_log_is_not_an_invented_member(jprep):
    """describe.hallucinated subtracts what symbols.unresolved resolves on an open path, an explicit static import the
    output adds and Lombok's `log` (and the root's members), so "self_contained iff hallucinated == 0" holds on Java;
    an invented member is still counted."""
    step = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n'
    imp = "import static org.assertj.core.api.Assertions.assertThat;\n\n"
    s = score_output(jprep, jfence(imp + step + "    assertThat(total).isNotNull();\n    cart.assertTotal(total);\n}"))
    assert s.bv == 1 and s.detail.get("open_resolution") is True  # BV resolves it, through an open path
    assert hallucinated(jprep, s) == (0, 2)  # members used: assertThat, cart
    s = score_output(jprep, jfence(step + "    cart.assertTotal(total + hashCode());\n}"))
    assert s.bv == 1 and hallucinated(jprep, s) == (0, 2)  # a member of the root object (sc.root)
    er = jprep.er_files[JPATH]
    jprep.er_files[JPATH] = er.replace("public class ShopSteps", "@Slf4j\npublic class ShopSteps")
    s = score_output(jprep, jfence(step + "    log.info(total);\n    cart.assertTotal(total);\n}"))
    assert s.conds["resolved"] and hallucinated(jprep, s) == (0, 2)
    ghost = score_output(jprep, jfence(step + "    ghostService.x(total);\n    cart.assertTotal(total);\n}"))
    assert hallucinated(jprep, ghost) == (1, 2)  # an invented member is still counted


def test_naming_ignores_edge_underscores_and_majority_needs_a_strict_lead():
    """naming ignores leading and trailing underscores (only an inner `_` makes a name snake case), and _majority needs
    a strict lead (a tie is no majority)."""
    assert (naming("helper_"), naming("Helper_"), naming("a_b"), naming("_a_b_")) == ("camel", "pascal", "snake", "snake")
    assert d._majority(Counter()) is None and d._majority(Counter({"a": 1, "b": 1})) is None
    assert d._majority(Counter({"a": 2, "b": 1})) == "a" and d._majority(Counter({"a": 1})) == "a"


def test_similarity_reads_the_target_blocks_only_and_keeps_four_decimals(prep):
    """similarity reads only the output's target blocks, not every block, and rouge_l keeps four decimals, not two."""
    g = prep.gold()
    code = g.items[next(iter(g.items))][2]
    t = prep.targets[0]
    assert similarity(prep, {t: code, "src/Other.php": "echo 1;"}) == {"rouge_l": 1.0, "exact": True}
    out = similarity(prep, {t: code.replace("assertTotal", "assertSum")})
    assert out["exact"] is False and out["rouge_l"] == 0.9787


ER_FILE = """<?php

namespace App\\Behat\\Context;

use Behat\\Step\\{Given, Then, When};

final class ShopContext
{
    public function __construct(private Cart $cart)
    {
    }

    public function __invoke(): void
    {
    }

    #[Given('I open the shop')]
    public function i_open_the_shop(): void
    {
        $this->cart->open();
    }
}
"""


def test_conventions_judge_naming_without_dunder_methods_and_pattern_share_in_four_decimals(prep):
    """The file's naming majority leaves out `__construct`/`__invoke`, conv_* keep four decimals, not one, and a tie is
    no majority."""
    prep.er_files = {prep.targets[0]: ER_FILE}
    three = ("#[Then('I should see the total :total')]\npublic function i_see_total(string $total): void\n{\n"
             "    $this->cart->add($total);\n}\n\n#[Then('I see the sum :sum')]\npublic function i_see_sum(string $sum): void\n{\n"
             "    $this->cart->add($sum);\n}\n\n#[Then('/^I see the item \"(.*)\"$/')]\npublic function i_see_item(string $x): void\n"
             "{\n    $this->cart->add($x);\n}")
    s = score_output(prep, block(three))
    blocks = extract(block(three), prep.targets).by_path()
    assert conventions(prep, s, blocks) == {"conv_style": 1.0, "conv_pattern": 0.6667, "conv_naming": 1.0}


def test_hallucinated_scope_rules_open_all_statics_and_open_calls(prep, monkeypatch):
    """A class with an ancestor outside the repo is not judged (open_all), statically imported names are not invented,
    and a bare call left to an open interface or `.*` static import is not invented (open_calls)."""
    s = score_output(prep, block(STEP.format("$this->ghostService->x($total);\n    $this->ghostCall();\n    $this->cart->add($total);")))
    real = d.scope
    assert hallucinated(prep, s) == (2, 3)
    patched = lambda **kw: lambda *a, **k: dataclasses.replace(real(*a, **k), **kw)
    monkeypatch.setattr(d, "scope", patched(open_all=True))
    assert hallucinated(prep, s) == (0, 0)
    monkeypatch.setattr(d, "scope", patched(statics=frozenset({"ghostService"})))
    assert hallucinated(prep, s) == (1, 3)
    monkeypatch.setattr(d, "scope", patched(open_calls=True))
    assert hallucinated(prep, s) == (1, 3)  # ghostCall is a bare call, left open; ghostService is not


def test_dose_counts_every_gold_item(prep):
    """dose reads the members of every gold item, not only the first one's."""
    import types
    t = prep.targets[0]
    prep.gold = lambda: types.SimpleNamespace(items={(t, "a"): (t, "method", STEP.format("$this->cart->add($total);")),
                                                     (t, "b"): (t, "method", STEP.format("$this->checkout->assertTotal($total);"))})
    assert dose(prep, "Project knowledge:\n private Cart $cart,\n private Checkout $checkout,\n") == 2
    assert dose(prep, "Project knowledge:\n private Checkout $checkout,\n") == 1


def test_row_copies_the_record_and_run_fields_it_documents(prep):
    """The row copies the fields it documents: preexisting is any, not all; the len_bucket boundary; dose None without
    a prompt; excl_bv and excl_ca, and hallucinated and members_used, each in its own column; text_sha256 of non-ASCII
    text."""
    cart = next(fid for fid, f in prep.repo.fns.items() if "iAddToCart" in f["code"])
    shell = {"fn_id": cart, "path": prep.targets[0], "code": prep.repo.fns[cart]["code"], "preexisting": True,
             "post_cutoff_code": False}
    text = block("// é café\n" + STEP.format("$this->ghostService->x($total);\n    $this->cart->add($total);"))
    s = score_output(prep, text)
    blocks = extract(text, prep.targets).by_path()
    run = {"text": text, "prompt_tokens": 8192, "cap_hit": True}
    out = row(prep, run, s, blocks, REFS, {"excl_bv": True, "excl_ca": False}, SEALED, True)
    assert (out["excl_bv"], out["excl_ca"], out["self_contained"], out["cap_hit"], out["len_bucket"]) == (True, False, True, True, "le8192")
    assert (out["hallucinated"], out["members_used"]) == hallucinated(prep, s) == (1, 2)
    assert out["dose"] is None and out["preexisting"] is False
    assert out["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert row(prep, {**run, "prompt_tokens": 8193}, s, blocks, REFS, EXCL, SEALED)["len_bucket"] == "gt8192"
    assert row(prep, {"text": text}, s, blocks, REFS, EXCL, SEALED)["len_bucket"] is None
    prep.record["create"] = [prep.record["create"][0], shell]
    assert row(prep, run, s, blocks, REFS, EXCL, SEALED)["preexisting"] is True


def unread_counts(p, text):
    out = row(p, {"text": text}, score_output(p, text), extract(text, p.targets).by_path(), REFS, EXCL, SEALED)
    return tuple(out[k] for k in UNREAD)


def test_a_candidate_in_a_body_or_in_a_docblock_is_judged_by_its_place(prep):
    """The head of a method ends where its body starts (an annotation inside the body does not count), and every style
    is read: a docblock `@And` is a candidate the reader declines, as `[Given]` without a pattern is."""
    read = "#[Then('I should see the total :total')]"
    nested = ("\npublic function t(string $total): void\n{\n    $o = new class { #[\\Foo\\Then('x')] public function y(): void {} };\n"
              "    $this->checkout->assertTotal($total);\n}")
    assert unread_counts(prep, block(read + nested)) == (0, 0, 0)  # registered by nobody, declined by nobody: in the body
    doc = "/**\n * @And I see the sum :sum\n */\npublic function u(): void\n{\n    $this->cart->add('x');\n}"
    assert unread_counts(prep, block(read + "\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}\n\n" + doc)) \
        == (1, 0, 0)


def test_a_non_php_run_has_no_php_columns(jprep):
    """The language gate of php_steps: a Java row's php_* columns are None, not 0 (SAP's attribute-only stratum and
    docblock rate read 0 as a PHP output with no docblock step)."""
    code = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(total);\n}'
    text = jfence(code)
    out = row(jprep, {"text": text}, score_output(jprep, text), extract(text, jprep.targets).by_path(), REFS, EXCL, SEALED)
    assert (out["php_tag_stripped"], out["php_docblock_steps"], out["php_attribute_steps"]) == (None, None, None)


def test_cucumber_expression_columns_separate_each_escape_and_each_known_type(jprep):
    """Each of `\\{` and `\\/` is an escape the binder mishandles; `{word}` and `{}` are known types; a regex pattern is
    no Cucumber expression; a member item never binds; one count per pattern."""
    body = "\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(total);\n}"
    keys = ("cuke_escaped_patterns", "cuke_custom_types")
    def c(ann):
        text = jfence(ann + body)
        out = row(jprep, {"text": text}, score_output(jprep, text), extract(text, jprep.targets).by_path(), REFS, EXCL, SEALED)
        return tuple(out[k] for k in keys)
    assert c('@Then("a \\\\{b\\\\} {string}")') == (1, 0)  # `\{`
    assert c('@Then("a\\\\/b {string}")') == (1, 0)  # `\/`
    assert c('@Then("a \\\\(b\\\\) {string}")') == (1, 0)  # `\(`
    assert c('@Then("the {word} {} {int} {float} total")') == (0, 0)  # known types, the anonymous one included
    assert c('@Then("^the \\\\(x\\\\) total {money}$")') == (0, 0)  # a regular expression, not an expression
    assert c('@Then("the {money} and {amount} total")') == (0, 1)  # two custom types, one pattern
    member = ('private Object o = new Object() {\n    @Then("the {money} total")\n    public void m() {\n    }\n};')
    text = jfence(member + "\n\n" + '@Then("I should see the total {string}")' + body)
    out = row(jprep, {"text": text}, score_output(jprep, text), extract(text, jprep.targets).by_path(), REFS, EXCL, SEALED)
    assert (out["cuke_escaped_patterns"], out["cuke_custom_types"]) == (0, 0)  # the anonymous class's method never binds


def test_a_step_call_nested_in_a_ruby_block_body_is_not_unread():
    """Ruby: the head of a block item is its first line only."""
    from tests.test_evaluate import RPATH, fence, mini_prep
    rb = mini_prep("ruby", ["ruby_block"], RPATH, "Given(/^I open the shop$/) do\n  open_shop\nend\n")
    def n(code):
        text = fence("ruby", RPATH, code)
        return unread_steps(rb, {"text": text}, extract(text, rb.targets).by_path())["unread_step_annotations"]
    assert n("Then(/^the total is (\\d+)$/) do |n|\n  check_total(n)\nend") == 0
    assert n("Then(:total) do |n|\n  check_total(n)\nend") == 1  # a symbol pattern: the reader declines it, on the first line
    assert n("Then(/^the total is (\\d+)$/) do |n|\n  Given(:sym) do\n  end\n  check_total(n)\nend") == 0  # nested: no step


def test_php_tag_stripped_sees_the_tag_of_the_second_target(tmp_path):
    """php_tag_stripped reads every target's block, not the first target only."""
    prep = make_prep(tmp_path)
    first = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}"
    text1 = block(first)
    s, blocks = score_output(prep, text1), extract(text1, prep.targets).by_path()
    other = "src/Behat/Context/OtherContext.php"
    prep.targets = [*prep.targets, other]  # a record with two target files
    text = text1 + "\n\n" + "`" * 3 + "php\n// FILE: " + other + "\n<?php\npublic function u(): void\n{\n}\n" + "`" * 3
    assert row(prep, {"text": text1}, s, blocks, REFS, EXCL, SEALED)["php_tag_stripped"] is False
    assert row(prep, {"text": text}, s, blocks, REFS, EXCL, SEALED)["php_tag_stripped"] is True


def test_hallucinated_hides_the_answers_a_dev_prompt_hides(tmp_path):
    """hallucinated: V(r) of a dev record hides the test/aux answers (hidden_for)."""
    prep = make_prep(tmp_path, "dev")
    cart = next(fid for fid, f in prep.repo.fns.items() if "iAddToCart" in f["code"])
    s = score_output(prep, block(STEP.format("$this->iAddToCart($total);")))
    assert hallucinated(prep, s) == (0, 1)
    prep.repo.forbidden = {"Acme/Shop": {"test": [cart]}}  # forbidden.json: a test answer, hidden from every dev prompt
    assert hallucinated(prep, s) == (1, 1)


def test_a_member_items_initializer_is_not_counted_by_hallucinated(jprep, prep):
    """A field, constant or PHP property of the output is a `member` item, and `hallucinated` skips it, so the names
    read in its initializer count in neither `hallucinated` nor `members_used` (the initializer is BV condition 6's
    business). Without the skip the Java ghost cases would read (1, 2) and the real-name case (0, 2); on the real data
    that would move 679 probe rows of test and 205 of dev, no gold or control row."""
    step = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n' \
           '    cart.assertTotal(total);\n}'
    fence = lambda code: f"{'`' * 3}java\n// FILE: {JPATH}\n{code}\n{'`' * 3}"
    plain = score_output(jprep, fence(step))
    ghost = score_output(jprep, fence(step + "\n\nprivate String f = ghostZ();"))
    assert hallucinated(jprep, plain) == (0, 1) and plain.bv == 1
    assert sorted(k for _, k, _ in ghost.items.values()) == ["member", "method"]  # the field IS a member item
    assert hallucinated(jprep, ghost) == (0, 1) and ghost.bv == 0  # ghostZ() fails BV (condition 6), not this column
    for field in ("private Object f = ghostFieldZ.make();", "private String f = cart.total();",
                  'private static final String K = prefixZ() + "a";', 'private String f = "a";'):
        assert hallucinated(jprep, score_output(jprep, fence(step + "\n\n" + field))) == (0, 1), field
    php = STEP.format("$this->iAddToCart($total);")
    assert hallucinated(prep, score_output(prep, block(php))) == (0, 1)
    assert hallucinated(prep, score_output(prep, block(php + "\n\nprivate $x = self::ghostZ();"))) == (0, 1)
