from v2tools.compose import gold_answer
from v2tools.extract import extract


def fenced(s: str) -> str:
    return s.replace("~~~", "`" * 3)


T = ["src/A.php", "src/B.php"]


def test_ok_two_files_and_prose_around():
    out = fenced("Here you go:\n~~~php\n// FILE: src/A.php\npublic function a() {}\n~~~\ntext\n~~~php\n// FILE: src/B.php\nfunction b() {}\n~~~\n")
    e = extract(out, T)
    assert e.status == "ok"
    assert e.by_path() == {"src/A.php": "public function a() {}", "src/B.php": "function b() {}"}


def test_none_when_no_fence():
    e = extract("// FILE: src/A.php\npublic function a() {}", T)
    assert e.status == "none" and e.by_path() == {}


def test_unclosed_last_block_is_kept():
    e = extract(fenced("~~~php\n// FILE: src/A.php\npublic function a() {\n"), T)
    assert e.status == "unclosed" and e.by_path() == {"src/A.php": "public function a() {"}


def test_malformed_blocks_are_dropped_others_kept():
    out = fenced("~~~php\npublic function x() {}\n~~~\n~~~php\n// FILE: src/Other.php\nf();\n~~~\n~~~php\n// FILE: ./src/A.php\na();\n~~~")
    e = extract(out, T)
    assert e.status == "malformed"
    assert [b.usable for b in e.blocks] == [False, False, True]
    assert e.by_path() == {"src/A.php": "a();"}


def test_ruby_hash_file_line_and_blank_lines_before_it():
    e = extract(fenced("~~~ruby\n\n# FILE: features/step_definitions/a_steps.rb\nGiven(/^x$/) do\nend\n~~~"), ["features/step_definitions/a_steps.rb"])
    assert e.status == "ok" and e.by_path()["features/step_definitions/a_steps.rb"] == "Given(/^x$/) do\nend"


def test_two_blocks_for_one_file_are_joined_in_order():
    out = fenced("~~~php\n// FILE: src/A.php\na();\n~~~\n~~~php\n// FILE: src/A.php\nb();\n~~~")
    assert extract(out, T).by_path() == {"src/A.php": "a();\n\nb();"}


def test_php_tags_are_dropped_per_block_before_the_blocks_are_joined():
    """§1.4 (27/09/2026): a PHP block's own leading `<?php` and trailing `?>` (php_open_tag) are dropped per block
    before by_path joins the blocks of one file, so an output that opens, or closes, every fence with a tag reads as the
    same code; php_tag_stripped says whether a usable PHP block had one. A block that is only a tag is no code; a block
    of another language keeps its text."""
    code = lambda text: [l for l in text.split("\n") if l.strip()]
    for a, b in (("<?php\n\na();", "<?php\n\nb();"), ("a();\n?>", "b();\n?>"), ("<?php\na();", "b();"),
                 ("<?php\na();\n?>", "<?php\nb();\n?>"), ("\n<?PHP\na();", "b();\n\n?>")):
        e = extract(fenced(f"~~~php\n// FILE: src/A.php\n{a}\n~~~\n~~~php\n// FILE: src/A.php\n{b}\n~~~"), T)
        assert code(e.by_path()["src/A.php"]) == ["a();", "b();"] and e.php_tag_stripped, (a, b)
    assert not extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~"), T).php_tag_stripped
    only = extract(fenced("~~~php\n// FILE: src/A.php\n<?php\n~~~\n~~~php\n// FILE: src/A.php\na();\n~~~"), T)
    assert only.by_path() == {"src/A.php": "a();"} and only.php_tag_stripped
    rb = ["features/step_definitions/a_steps.rb"]
    e = extract(fenced(f"~~~ruby\n# FILE: {rb[0]}\n<?php\nx\n~~~"), rb)
    assert e.by_path() == {rb[0]: "<?php\nx"} and not e.php_tag_stripped


def test_gold_answer_round_trips():
    rec = {"language": "php", "t_vis": 10, "create_def_ids": [],
           "create": [{"fn_id": "src/A.php:3", "path": "src/A.php", "code": "#[When('a')]\npublic function a(): void\n{\n}", "preexisting": False},
                      {"fn_id": "src/B.php:3", "path": "src/B.php", "code": "/**\n * @Then b\n */\npublic function b(): void\n{\n}", "preexisting": False},
                      {"fn_id": "src/A.php:9", "path": "src/A.php", "code": "#[Then('c')]\npublic function c(): void\n{\n}", "preexisting": False}]}
    e = extract(gold_answer(rec), T)
    assert e.status == "ok"
    assert e.by_path() == {"src/A.php": rec["create"][0]["code"] + "\n\n" + rec["create"][2]["code"], "src/B.php": rec["create"][1]["code"]}


def test_a_file_line_inside_a_fence_starts_a_new_file():
    """A FILE line inside a fence starts a new file, the text before it belongs to the previous one, and the output's
    status says so (multi_file_fence) when nothing worse applies (none > malformed > unclosed)."""
    one = fenced("~~~php\n// FILE: src/A.php\npublic function a() {}\n\n// FILE: src/B.php\n"
                 "public function b() {}\n~~~")
    e = extract(one, T)
    assert e.status == "multi_file_fence" and not e.truncated
    assert e.by_path() == {"src/A.php": "public function a() {}", "src/B.php": "public function b() {}"}
    assert extract(one.replace("src/B.php", "src/C.php"), T).status == "malformed"
    assert extract(one.rstrip("`"), T).status == "unclosed"
    lead = extract(fenced("~~~php\nnote\n// FILE: src/A.php\na();\n~~~"), T)
    assert lead.status == "malformed" and lead.by_path() == {"src/A.php": "a();"}


def test_a_fence_closes_on_at_least_as_many_backticks():
    """As in CommonMark, a fence opened with four backticks keeps a three-backtick line as code; a closing line may be
    longer than the opening one."""
    inner = "public function a() {\n    $s = <<<EOT\n~~~\nx\n~~~\nEOT;\n}"
    four = "`" * 4
    e = extract(fenced(four + "php\n// FILE: src/A.php\n" + inner + "\n" + four), T)
    assert e.status == "ok" and e.by_path() == {"src/A.php": fenced(inner)} and not e.truncated
    assert extract(fenced("~~~php\n// FILE: src/A.php\na();\n" + four), T).status == "ok"


def test_a_fence_closed_early_is_detected():
    """The documented limit: a three-backtick line inside a three-backtick fence closes it early. The rest reopens as a
    bare fence (backticks only) whose first line is not a FILE line, right after a closed FILE block:
    Extraction.truncated when that bare fence never closes or holds a FILE line further down (§1.4, 28/09/2026). A
    deliberate extra block with an info string (bash) is not flagged. Here the heredoc's rest closes its bare fence,
    holding no FILE line, so it is not flagged, a documented limit of §1.4 (BV, CA and the status, malformed, do not
    change: the code is still cut); the same output cut at the end is flagged."""
    out = fenced("~~~php\n// FILE: src/A.php\npublic function a() {\n    $s = <<<EOT\n~~~\nx\n~~~\nEOT;\n}\n~~~")
    e = extract(out, T)
    assert not e.truncated and e.status == "malformed"
    assert e.by_path() == {"src/A.php": "public function a() {\n    $s = <<<EOT"}
    assert extract(out.rsplit("\n", 1)[0], T).truncated
    bash = fenced("~~~php\n// FILE: src/A.php\na();\n~~~\nRun it:\n~~~bash\nvendor/bin/behat\n~~~")
    assert not extract(bash, T).truncated


def test_any_info_string_opens_a_fence():
    """As in CommonMark, an opening fence is three or more backticks and any info string without a backtick:
    " php", "php:src/A.php", "php {1,3}" after the backticks, and a bare fence, all open one."""
    for opener in ("~~~ php", "~~~php:src/A.php", "~~~php {1,3}", "~~~"):
        e = extract(fenced(f"{opener}\n// FILE: src/A.php\na();\n~~~\n{opener}\n// FILE: src/B.php\nb();\n~~~"), T)
        assert e.status == "ok" and e.by_path() == {"src/A.php": "a();", "src/B.php": "b();"}, opener


def test_lines_split_on_newline_only():
    """Lines are split on "\\n" (a trailing "\\r" dropped, for CRLF input), never str.splitlines(): U+2028, U+2029,
    U+0085, \\x0b, \\x0c and \\x1c-\\x1e stay inside the code."""
    code = 'String s = "a b c\u0085d\x0be\x0cf\x1cg\x1dh\x1ei";'
    assert extract(fenced(f"~~~java\n// FILE: src/A.php\n{code}\n~~~"), T).by_path() == {"src/A.php": code}
    crlf = extract(fenced("~~~php\r\n// FILE: src/A.php\r\na();\r\n  b();\r\n\r\n~~~\r\n"), T)
    assert crlf.status == "ok" and crlf.by_path() == {"src/A.php": "a();\n  b();"}


def test_status_pins():
    """An off-target path alone is malformed; all blocks malformed is malformed, not none; malformed wins over
    unclosed; text after the path on a FILE line is malformed; an empty usable block leaves the file out of by_path
    (status ok); a line that starts with the ticks but holds more is code, not a closing fence."""
    assert extract(fenced("~~~php\n// FILE: src/Other.php\nf();\n~~~"), T).status == "malformed"
    assert extract(fenced("~~~php\n// FILE: src/A.php (new)\nf();\n~~~"), T).status == "malformed"
    two = fenced("~~~php\n// FILE: src/Other.php\nf();\n~~~\n~~~php\n// FILE: src/A.php\na();\n")
    assert extract(two, T).status == "malformed"
    empty = extract(fenced("~~~php\n// FILE: src/A.php\n\n~~~"), T)
    assert empty.status == "ok" and empty.by_path() == {}
    kept = extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~php\nb();\n~~~"), T)
    assert kept.status == "ok" and kept.by_path() == {"src/A.php": fenced("a();\n~~~php\nb();")}


def test_extraction_edge_case_pins():
    """One trailing "\\r" is dropped per line, not all of them; a bare fence that starts with a FILE line is not a
    truncation; a truncation needs the fence RIGHT before it to be a closed FILE fence (a `bash` block in between
    resets it); a segment that a FILE line ends is closed even when its fence never closes."""
    assert extract(fenced("~~~php\r\n// FILE: src/A.php\r\na();\r\r\nb();\r\n~~~\r\n"), T).by_path() == \
        {"src/A.php": "a();\r\nb();"}
    bare_file = extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~\n~~~\n// FILE: src/B.php\nb();\n~~~"), T)
    assert bare_file.status == "ok" and not bare_file.truncated
    reset = extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~\n~~~bash\nls\n~~~\n~~~\nnote\n~~~"), T)
    assert reset.status == "malformed" and not reset.truncated
    open_ = extract(fenced("~~~php\n// FILE: src/A.php\na();\n// FILE: src/B.php\nb();\n"), T)
    assert open_.status == "unclosed" and [b.closed for b in open_.blocks] == [True, False]


def test_fence_edge_rules():
    """The rules at the edges of a fence in v2tools/extract.py (§1.4, fixed 28/09/2026, before any output): head tag
    line, a line of only `<?php` before a fence's first FILE line is blank and a stripped tag, the block the FILE line's
    (with other text before the FILE line the fence is still malformed, and without a FILE line nothing changes);
    reopen, inside an open fence a fence-open line RIGHT before a FILE line is a close and reopen, dropped, the status
    still multi_file_fence (with a blank line between, the line is code); narrowed truncation, a bare fence right after
    a closed FILE fence, whose first non-blank line is not a FILE line, is a truncation only when it never closes or
    holds a FILE line further down: a closed shell command is none, a real closing fence that reopens and never closes
    is one, prose and then `~~~php // FILE: B` in it is one, and a bare fence that starts with a FILE line is none."""
    m1 = extract(fenced("~~~php\n<?php\n// FILE: src/A.php\na();\n~~~"), T)
    assert (m1.status, m1.php_tag_stripped, m1.by_path()) == ("ok", True, {"src/A.php": "a();"})
    assert extract(fenced("~~~php\n<?PHP  \n\n// FILE: src/A.php\na();\n~~~"), T).status == "ok"
    assert extract(fenced("~~~php\n<?php\nnote\n// FILE: src/A.php\na();\n~~~"), T).status == "malformed"
    # the head tag line rule takes a line of the open tag ALONE, not `<?php echo 1;`
    code_after_tag = extract(fenced("~~~php\n<?php echo 1;\n// FILE: src/A.php\na();\n~~~"), T)
    assert (code_after_tag.status, code_after_tag.php_tag_stripped) == ("malformed", False)
    assert extract(fenced("~~~php\n<?php\na();\n~~~"), T).status == "malformed"
    assert not extract(fenced("~~~php\n// FILE: src/Other.php\n<?php\na();\n~~~"), T).php_tag_stripped  # off target
    m2 = extract(fenced("~~~php\n// FILE: src/A.php\na();\n\n~~~php\n// FILE: src/B.php\nb();\n~~~"), T)
    assert (m2.status, m2.by_path()) == ("multi_file_fence", {"src/A.php": "a();", "src/B.php": "b();"})
    gap = extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~php\n\n// FILE: src/B.php\nb();\n~~~"), T)
    assert gap.by_path() == {"src/A.php": fenced("a();\n~~~php"), "src/B.php": "b();"}
    shell = fenced("~~~php\n// FILE: src/A.php\na();\n~~~\n\nRun it with:\n~~~\nvendor/bin/behat\n~~~\n")
    assert not extract(shell, T).truncated
    assert extract(fenced("~~~php\n// FILE: src/A.php\na();\n/*\n~~~\n*/\n~~~"), T).truncated  # reopens, never closes
    prose = fenced("~~~php\n// FILE: src/A.php\na();\n/*\n~~~\n*/\n~~~\nNow B:\n~~~php\n// FILE: src/B.php\nb();\n~~~")
    assert extract(prose, T).truncated
    assert not extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~\n~~~\n// FILE: src/B.php\nb();\n~~~"), T).truncated
    # narrowed truncation flags a BARE fence only (an unclosed `bash` fence after a FILE fence is none); only the first
    # block of a fence has the `<?php` line before it (a target second block is no stripped tag); the reopen rule drops
    # a bare fence-open line too (a bare line of three backticks right before a FILE line inside a four-backtick fence)
    assert not extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~\n\nRun:\n~~~bash\nvendor/bin/behat"), T).truncated
    assert not extract(fenced("~~~php\n<?php\n// FILE: src/Other.php\nx();\n// FILE: src/A.php\na();\n~~~"), T).php_tag_stripped
    four = extract(fenced("~~~`php\n// FILE: src/A.php\na();\n~~~\n// FILE: src/B.php\nb();\n~~~`"), T)
    assert (four.status, four.by_path()) == ("multi_file_fence", {"src/A.php": "a();", "src/B.php": "b();"})


def test_fence_line_pins():
    """An info string holding a backtick opens no fence; an indented opening line opens one; a closing line with text
    after its backticks ("~~~ php") does not close; an indented closing line closes."""
    assert extract(fenced("~~~ph`p\n// FILE: src/A.php\na();\n~~~"), T).status != "ok"
    assert extract(fenced("   ~~~php\n// FILE: src/A.php\na();\n~~~"), T).by_path() == {"src/A.php": "a();"}
    assert extract(fenced("~~~php\n// FILE: src/A.php\na();\n~~~ php\nb();\n~~~"), T).by_path() == \
        {"src/A.php": fenced("a();\n~~~ php\nb();")}
    assert extract(fenced("~~~php\n// FILE: src/A.php\na();\n    ~~~\nprose"), T).by_path() == {"src/A.php": "a();"}
