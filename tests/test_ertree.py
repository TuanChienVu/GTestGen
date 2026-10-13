import pytest

from v2tools.ertree import Removal, cut, er_tree, phantoms, removal

DOC = "<?php\nclass S\n{\n    /**\n     * @Given I open the shop\n     */\n    public function a(): void\n    {\n    }\n}\n"


def fn(fid, first, end, anns, code=""):
    return {"fn_id": fid, "path": "S.php", "first_line": first, "body_end": end, "code": code,
            "annotations": [{"def_id": f"S.php:{l}", "line": l, "time": t} for l, t in anns]}


def rec(create, t=100, cdefs=()):
    return {"t_vis": t, "create_def_ids": list(cdefs), "language": "php", "create": create}


def test_removal_new_answer_whole_and_future_functions():
    fns = {"S.php:5": fn("S.php:5", 3, 6, [(3, 200)]), "S.php:9": fn("S.php:9", 8, 10, [(8, 50)]),
           "S.php:13": fn("S.php:13", 12, 14, [(12, 150)])}
    rem = removal(rec([{"fn_id": "S.php:5", "path": "S.php", "preexisting": False}], cdefs=["S.php:3"]), fns)
    assert rem.lines == {"S.php": frozenset({3, 4, 5, 6, 12, 13, 14})}
    assert rem.defs == {"S.php:3", "S.php:12"} and (rem.answer_fns, rem.future_fns) == (1, 1)


def test_removal_preexisting_keeps_body_drops_create_and_late_annotations():
    fns = {"S.php:5": fn("S.php:5", 2, 7, [(2, 10), (3, 20), (4, 300)])}
    rem = removal(rec([{"fn_id": "S.php:5", "path": "S.php", "preexisting": True}], cdefs=["S.php:3"]), fns)
    assert rem.lines == {"S.php": frozenset({3, 4})} and rem.answer_annotations == 2 and rem.defs == {"S.php:3", "S.php:4"}


def test_phantom_functions_are_never_cut():
    java = 'class A {\n    // @Then("x")\n    // void x() {\n    void y() {\n    }\n}\n'
    fns = {"A.java:3": {"fn_id": "A.java:3", "path": "A.java", "first_line": 2, "body_end": 5, "code": "",
                        "annotations": [{"def_id": "A.java:2", "line": 2, "time": 999}]}}
    ghost = phantoms("java", fns, lambda p: java)
    assert ghost == {"A.java:3"}
    rem = removal({"t_vis": 100, "create_def_ids": [], "language": "java", "create": []}, fns, ghost)
    assert rem.lines == {} and rem.defs == {"A.java:2"} and rem.phantom_fns == 1


def test_php_docblock_is_not_a_phantom():
    fns = {"S.php:8": fn("S.php:8", 4, 9, [(5, 1)])}
    assert phantoms("php", fns, lambda p: DOC) == frozenset()


def test_cut_and_er_tree_number_lines_by_newline():
    assert cut("a\nb\nc\n", frozenset({2})) == "a\nc\n"
    rem = Removal({"S.php": frozenset({1})}, frozenset(), 0, 0, 0, 0)
    assert er_tree(rem, lambda p: "x\ny") == {"S.php": "y"}


def test_cut_keeps_crlf_line_endings():
    assert cut("a\r\nb\r\nc\r\n", frozenset({2})) == "a\r\nc\r\n"


def test_catalog_dead_flag_is_a_phantom_too():
    fns = {"S.php:8": fn("S.php:8", 4, 9, [(5, 1)]), "S.php:3": {**fn("S.php:3", 2, 3, [(2, 999), (3, 999)]), "dead": True}}
    assert phantoms("php", fns, lambda p: DOC) == frozenset({"S.php:3"})


def test_phantoms_must_agree_with_the_catalog_flags():
    java = 'class A {\n    // @Then("x")\n    // void x() {\n    void y() {\n    }\n}\n'
    ghost = {"fn_id": "A.java:3", "path": "A.java", "first_line": 2, "body_end": 5, "code": "",
             "annotations": [{"def_id": "A.java:2", "line": 2, "time": 999}]}
    assert phantoms("java", {"A.java:3": {**ghost, "dead": True}}, lambda p: java) == {"A.java:3"}
    with pytest.raises(ValueError, match="dead flag"):
        phantoms("java", {"A.java:3": {**ghost, "dead": False}}, lambda p: java)
    live = {**ghost, "fn_id": "A.java:4", "annotations": [{"def_id": "A.java:4", "line": 4, "time": 999}]}
    with pytest.raises(ValueError, match="dead flag"):
        phantoms("java", {"A.java:4": {**live, "dead": False, "contains_fn": True}}, lambda p: java)


def jfn(fid, first, end, anns, **flags):
    return {"fn_id": fid, "path": "A.java", "first_line": first, "body_end": end, "code": "",
            "annotations": [{"def_id": f"A.java:{l}", "line": l, "time": t} for l, t in anns], **flags}


def test_removal_keeps_a_function_with_any_annotation_before_t_vis():
    rem = removal(rec([], t=100), {"S.php:5": fn("S.php:5", 3, 7, [(3, 50), (4, 150)])})
    assert rem.lines == {} and rem.defs == frozenset() and rem.future_fns == 0


def test_removal_annotation_at_exactly_t_vis_is_future():
    rem = removal(rec([], t=100), {"S.php:5": fn("S.php:5", 3, 6, [(3, 100)])})
    assert rem.lines == {"S.php": frozenset({3, 4, 5, 6})} and rem.defs == {"S.php:3"} and rem.future_fns == 1
    fns = {"S.php:5": fn("S.php:5", 2, 7, [(2, 10), (3, 100), (4, 100)])}
    rem = removal(rec([{"fn_id": "S.php:5", "path": "S.php", "preexisting": True}], t=100, cdefs=["S.php:3"]), fns)
    assert rem.lines == {"S.php": frozenset({3, 4})} and rem.defs == {"S.php:3", "S.php:4"}
    assert (rem.answer_annotations, rem.future_fns) == (2, 0)


def test_removal_none_time_is_not_before_t_vis():
    rem = removal(rec([], t=100), {"S.php:5": fn("S.php:5", 3, 6, [(3, None)])})
    assert rem.lines == {"S.php": frozenset({3, 4, 5, 6})} and rem.defs == {"S.php:3"} and rem.future_fns == 1


def test_removal_new_answer_drops_every_def_id_it_carries():
    fns = {"S.php:5": fn("S.php:5", 2, 6, [(2, 100), (3, 100)])}
    rem = removal(rec([{"fn_id": "S.php:5", "path": "S.php", "preexisting": False}], cdefs=["S.php:2"]), fns)
    assert rem.lines == {"S.php": frozenset({2, 3, 4, 5, 6})} and rem.defs == {"S.php:2", "S.php:3"}


def test_function_with_one_live_annotation_is_not_a_phantom():
    java = 'class A {\n    // @Given("old")\n    @Given("new")\n    void y() {\n    }\n}\n'
    row = jfn("A.java:4", 2, 5, [(2, 10), (3, 50)])
    assert phantoms("java", {"A.java:4": row}, lambda p: java) == frozenset()
    assert phantoms("java", {"A.java:4": {**row, "dead": False}}, lambda p: java) == frozenset()


def test_live_contains_fn_row_must_be_flagged_dead():
    java = 'class A {\n    @Given("y")\n    void y() {\n    }\n}\n'
    row = jfn("A.java:3", 2, 4, [(2, 5)], dead=False)
    assert phantoms("java", {"A.java:3": row}, lambda p: java) == frozenset()
    assert phantoms("java", {"A.java:3": {**row, "dead": True, "contains_fn": True}}, lambda p: java) == {"A.java:3"}
    with pytest.raises(ValueError, match="dead flag"):
        phantoms("java", {"A.java:3": {**row, "contains_fn": True}}, lambda p: java)


def test_phantoms_count_bytes_after_multibyte_text():
    java = ('class A {\n    String s = "é日本";\n    // @Then("x")\n    // void x() {}\n'
            '    @Given("y")\n    void y() {\n    }\n}\n')
    fns = {"A.java:4": jfn("A.java:4", 3, 4, [(3, 999)]), "A.java:6": jfn("A.java:6", 5, 7, [(5, 50)])}
    assert phantoms("java", fns, lambda p: java) == {"A.java:4"}


def test_java_block_comment_step_is_a_phantom():
    java = 'class A {\n    /*\n    @Then("x")\n    void x() {\n    }\n    */\n    @Given("y") void y() {\n    }\n}\n'
    fns = {"A.java:4": jfn("A.java:4", 3, 8, [(3, 999)]), "A.java:7": jfn("A.java:7", 7, 8, [(7, 50)])}
    assert phantoms("java", fns, lambda p: java) == {"A.java:4"}


def test_er_tree_raises_on_a_path_missing_from_head():
    with pytest.raises(ValueError, match="not in HEAD: S.php$"):
        er_tree(Removal({"S.php": frozenset({1})}, frozenset(), 0, 0, 0, 0), lambda p: None)


def test_phantoms_refuse_a_catalog_row_with_no_annotation():
    fns = {"S.php:8": fn("S.php:8", 4, 9, [(5, 1)]), "S.php:3": fn("S.php:3", 2, 3, [])}
    with pytest.raises(ValueError, match=r"no annotation: \['S\.php:3'\]"):
        phantoms("php", fns, lambda p: DOC)
    with pytest.raises(ValueError, match=r"no annotation: \['S\.php:3'\]"):
        phantoms("php", {**fns, "S.php:3": {**fns["S.php:3"], "dead": True}}, lambda p: DOC)
    # a row without the key is one without annotations (a ValueError, with the count and the repo), and a record
    # without t_vis stops removal with a ValueError
    keyless = {k: v for k, v in fns["S.php:3"].items() if k != "annotations"}
    with pytest.raises(ValueError, match=r"^1 catalog rows with no annotation \(repo R\): \['S\.php:3'\]$"):
        phantoms("php", {**fns, "S.php:3": {**keyless, "repo": "R"}}, lambda p: DOC)
    with pytest.raises(ValueError, match="R\\|x: no t_vis"):
        removal({"id": "R|x", "t_vis": None, "create_def_ids": [], "create": []}, fns)
    # a record with no t_vis key at all is refused the same way; the count is all the bare
    # rows, not the three the message lists
    with pytest.raises(ValueError, match="^R\\|x: no t_vis$"):
        removal({"id": "R|x", "create_def_ids": [], "create": []}, fns)
    four = {f"S.php:{k}": {**keyless, "fn_id": f"S.php:{k}", "repo": "R"} for k in (3, 30, 31, 32)}
    with pytest.raises(ValueError, match=r"^4 catalog rows with no annotation \(repo R\): \['S\.php:3', 'S\.php:30', "
                                         r"'S\.php:31'\]$"):
        phantoms("php", {**fns, **four}, lambda p: DOC)


def test_phantoms_refuse_a_catalog_file_absent_from_head():
    fns = {"S.php:8": fn("S.php:8", 4, 9, [(5, 1)]), "T.php:4": {**fn("T.php:4", 2, 5, [(3, 1)]), "path": "T.php"}}
    with pytest.raises(ValueError, match="not in HEAD: T.php$"):
        phantoms("php", fns, {"S.php": DOC}.get)
    fns["T.php:4"]["repo"] = "Sylius/Sylius"
    with pytest.raises(ValueError, match=r"not in HEAD: T\.php \(repo Sylius/Sylius\)$"):
        phantoms("php", fns, {"S.php": DOC}.get)
    # a file whose every row is dead is still looked up in HEAD
    with pytest.raises(ValueError, match=r"not in HEAD: T\.php"):
        phantoms("php", {**fns, "T.php:4": {**fns["T.php:4"], "dead": True}}, {"S.php": DOC}.get)
