from v2tools.body import MIN_CHARS, body_key, body_key_nosig, containments, near_duplicates, nosig_text

BODY = "    $this->indexPage->filterByCode($phrase);\n    $this->indexPage->filter();\n}"


def fn(code, first_line=10, ann_lines=(10,)):
    return {"code": code, "first_line": first_line, "annotations": [{"line": l} for l in ann_lines]}


def test_body_key_drops_annotations_docblocks_and_comments_but_keeps_the_signature():
    attribute = fn("#[When('I filter by code containing :phrase')]\npublic function iFilter(string $phrase): void\n{\n" + BODY)
    docblock = fn("/**\n * @When I filter by code :phrase\n */\npublic function iFilter(string $phrase): void\n{\n"
                  "    // narrow the grid first\n" + BODY, ann_lines=(11,))
    renamed = fn("#[When('x')]\npublic function iFilterOther(string $phrase): void\n{\n" + BODY)
    assert body_key(attribute) == body_key(docblock) == (
        "public function iFilter(string $phrase): void { $this->indexPage->filterByCode($phrase); $this->indexPage->filter(); }")
    assert body_key(renamed) != body_key(attribute)


def test_body_key_is_none_when_short_and_keeps_php_attributes_that_are_not_annotations():
    assert body_key(fn("#[Given('a')]\npublic function a(): void {}")) is None
    ruby = fn("Given(/^the patient (.*) exists$/) do |name|\n  # look the patient up\n  @patient = Patient.find_by!(name: name)\n"
              "  expect(@patient).to be_present\nend", first_line=1, ann_lines=(1,))
    assert body_key(ruby) == "@patient = Patient.find_by!(name: name) expect(@patient).to be_present end"
    attributed = fn("#[When('x')]\n#[Deprecated]\npublic function iFilter(string $phrase): void\n{\n" + BODY)
    assert body_key(attributed).startswith("#[Deprecated] public function iFilter(")


def test_body_key_nosig_drops_the_declaration_so_renamed_copies_match():
    one_line = fn("#[When('x')]\npublic function iFilter(string $phrase): void\n{\n" + BODY)
    multi_line = fn("#[When('y')]\npublic function iFilterByAnotherName(\n    string $phrase,\n): void {\n" + BODY)
    assert body_key(one_line) != body_key(multi_line)
    assert body_key_nosig(one_line, "php") == body_key_nosig(multi_line, "php") == (
        "$this->indexPage->filterByCode($phrase); $this->indexPage->filter(); }")
    lam_a = fn('Given("/^a (.*)$/",\n        (String name) -> {\n            context.put("name", name);\n'
               '            context.put("created", java.time.Instant.now());\n        });', first_line=1, ann_lines=(1,))
    lam_b = fn('Given("/^b (.*)$/",\n        (String other) -> {\n            context.put("name", name);\n'
               '            context.put("created", java.time.Instant.now());\n        });', first_line=1, ann_lines=(1,))
    assert body_key(lam_a) != body_key(lam_b) and body_key_nosig(lam_a, "java") == body_key_nosig(lam_b, "java")
    ruby = fn("Given(/^x$/) do\n  create(:patient, attributes: { name: 'Jo' }, family_name: 'Blogs')\n  visit patient_path(1)\nend",
              first_line=1, ann_lines=(1,))
    assert body_key_nosig(ruby, "ruby") == body_key(ruby)  # Ruby: the declaration is the annotation line; a `{` is a hash
    assert body_key_nosig(ruby, "java") != body_key(ruby)   # the brace rule would have cut it


def test_min_chars_boundary_is_inclusive():
    assert MIN_CHARS == 60
    exact = fn("#[When('x')]\n" + "a" * 60)
    assert body_key(exact) == "a" * 60 and body_key(fn("#[When('x')]\n" + "a" * 59)) is None


def test_nosig_text_cuts_at_the_first_brace_only_after_a_call_like_prefix():
    nested = fn("#[When('x')]\npublic function f($a): void\n{\n    if ($a) {\n        return;\n    }\n}")
    assert nosig_text(nested, "php") == "if ($a) { return; } }"                      # the FIRST `{`, not the last
    statement_first = fn("#[When('x')]\n$x = 1; if ($y) {\n    z();\n}")
    assert nosig_text(statement_first, "php") == "$x = 1; if ($y) { z(); }"          # a `;` before the brace: no cut
    no_call = fn("#[When('x')]\nstatic {\n    init();\n}")
    assert nosig_text(no_call, "java") == "static { init(); }"                       # no `(` before the brace: no cut
    at_60 = fn("#[When('x')]\npublic function f(): void {" + "a" * 59 + "}")
    assert body_key_nosig(at_60, "php") == "a" * 59 + "}"                                    # 60 characters: kept
    assert body_key_nosig(fn("#[When('x')]\npublic function f(): void {" + "a" * 58 + "}"), "php") is None   # 59


def test_near_duplicates_keep_the_best_answer_and_ignore_short_or_distant_texts():
    a = "$this->client->addFilter('code', $phrase); $this->client->filter(); $this->client->sort('code'); }"
    close = a.replace("'code', $phrase", "'name', $phrase")
    got = near_duplicates({"x:1": a, "y:1": close, "z:1": close},
                          {"c:1": close, "d:1": "$this->x(); }", "e:1": "completely different text " * 4})
    assert got == {"c:1": ("y:1", 1.0)}                     # best ratio wins; the tie with z:1 keeps y:1 (met first)
    assert 0.9 <= near_duplicates({"x:1": a}, {"c:1": close})["c:1"][1] < 1.0
    assert near_duplicates({"x:1": a}, {"c:1": close[:40]}) == {}   # under MIN_CHARS: ignored


def test_near_ratio_threshold_is_0_90_inclusive():
    base = "".join(chr(0x4E00 + i) for i in range(100))            # 100 distinct characters: ratio = (100 - k) / 100
    swap = lambda k: "".join(chr(0x5E00 + i) if i % 9 == 4 and i // 9 < k else c for i, c in enumerate(base))
    got = near_duplicates({"x:1": base}, {"k7": swap(7), "k10": swap(10), "k11": swap(11)})
    assert got == {"k7": ("x:1", 0.93), "k10": ("x:1", 0.9)}


def test_near_duplicates_pin_autojunk_off_orientation_and_min_chars():
    popular = "".join(chr(0x4E00 + i % 40) for i in range(240))   # every character "popular": autojunk would drop all
    got = near_duplicates({"x:1": popular}, {"c:1": "Z" + popular[1:]})
    assert list(got) == ["c:1"] and got["c:1"][1] > 0.99            # with autojunk on, the ratio would be 0.0
    fn_text = "abbcaacabbbabaacacaabaccbaabbacabcbcbcbaaaacccaacbccccbbcbcbacbbc"    # a = the function
    ans_text = "abbbcaacabbbaabaacacaabccbaabbacabcbcbcbaaaacccaacbcccacbbcbbcbbc"   # b = the answer
    assert near_duplicates({"x:1": ans_text}, {"c:1": fn_text})["c:1"][1] >= 0.9    # ratio(a, b) = 0.9538
    assert near_duplicates({"x:1": fn_text}, {"c:1": ans_text}) == {}                # ratio(b, a) = 0.8769
    short = "".join(chr(0x4E00 + i) for i in range(59))
    assert near_duplicates({"x:1": short}, {"c:1": short}) == {}                    # identical but under MIN_CHARS
    assert near_duplicates({"x:1": short + "!"}, {"c:1": short + "!"}) == {"c:1": ("x:1", 1.0)}


def test_containments_hold_most_of_a_long_answer_inside_a_longer_body():
    answer = "".join(chr(0x4E00 + i) for i in range(250))                           # 250 distinct characters
    holder = "HEAD " + answer[:240] + "".join(chr(0x6000 + i) for i in range(100))  # holds 240 of 250 = 0.96
    partial = answer[:200] + "".join(chr(0x7000 + i) for i in range(200))          # holds 200 of 250 = 0.80
    got = containments({"a:1": answer, "s:1": answer[:199]}, {"h:1": holder, "p:1": partial})
    assert got == {"h:1": ("a:1", 0.96)}   # s:1 is under 200 characters; share is of the ANSWER (reversed: 240/345)
    popular = "".join(chr(0x4E00 + i % 40) for i in range(240))
    assert containments({"a:1": popular}, {"h:1": "Q" + popular + "tail"}) == {"h:1": ("a:1", 1.0)}   # autojunk off
