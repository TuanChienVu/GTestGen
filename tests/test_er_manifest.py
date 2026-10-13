import hashlib
import json
from pathlib import Path

import pytest

from er.manifest import ManifestError, added_lines, load_manifest, tamper_flags, verify_entry, write_run
from er.spike_manifest import convert

A = "src/Sylius/Behat/Context/Api/A.php"
REC = {"id": "Sylius/Sylius|f.feature|3|api_x", "suite": "api_x", "feature_path": "f.feature", "scenario_line": 3,
       "scenario_name": "s", "t_vis": 1, "create": [{"path": A}]}
BASE = {A: "<?php\nclass A\n{\n}\n"}
GOLD = {A: "<?php\nclass A\n{\n    public function x(): void\n    {\n    }\n}\n"}


def make(tmp_path, files=GOLD, condition="gold", base=BASE, run="r"):
    run_dir = write_run(tmp_path, run, [{"record": REC, "condition": "neg", "slug": "f_3", "base": base, "files": {}},
                                        {"record": REC, "condition": condition, "slug": "f_3", "base": base,
                                         "files": files}])
    return {e.condition: e for e in load_manifest(run_dir)}


def test_round_trip_and_verify(tmp_path):
    entries = make(tmp_path)
    verify_entry(tmp_path, entries["neg"], REC)
    verify_entry(tmp_path, entries["gold"], REC)
    assert entries["neg"].dir is None and entries["gold"].files
    assert entries["gold"].overlay_sha256 != entries["neg"].overlay_sha256


def test_changed_file_is_rejected(tmp_path):
    entries = make(tmp_path)
    (tmp_path / "r/f_3/gold" / A).write_text("<?php\n")
    with pytest.raises(ManifestError, match="sha256"):
        verify_entry(tmp_path, entries["gold"], REC)


def test_extra_file_is_rejected(tmp_path):
    entries = make(tmp_path)
    (tmp_path / "r/f_3/base/src/Sylius/Behat/Context/Api/B.php").write_text("<?php\n")
    with pytest.raises(ManifestError, match="differ"):
        verify_entry(tmp_path, entries["gold"], REC)


def test_edited_manifest_is_rejected(tmp_path):
    make(tmp_path)
    m = tmp_path / "r/manifest.jsonl"
    m.write_text(m.read_text().replace('"gold"', '"stub"'))
    with pytest.raises(ManifestError, match="MANIFEST_SHA256"):
        load_manifest(tmp_path / "r")


def test_path_traversal_is_rejected(tmp_path):
    entries = make(tmp_path)
    e = entries["gold"]
    bad = type(e)(e.record_id, e.condition, e.dir, e.base, {"../etc/passwd": "x"}, e.files, e.skip, e.line)
    with pytest.raises(ManifestError):
        verify_entry(tmp_path, bad, REC)


def test_condition_files_must_be_create_targets_php_under_behat(tmp_path):
    other = "src/Sylius/Behat/Context/Api/Other.php"
    e = make(tmp_path, files={**GOLD, other: "<?php\n"}, condition="m.s42.full_aware")["m.s42.full_aware"]
    with pytest.raises(ManifestError, match="CREATE targets"):
        verify_entry(tmp_path, e, REC)
    yml = make(tmp_path, files={"src/Sylius/Behat/Resources/config/x.yml": "a: 1\n"}, run="y")["gold"]
    with pytest.raises(ManifestError, match="not a .php file"):
        verify_entry(tmp_path, yml, REC)
    for name in ("_gold_ref", "a/b", "Deepseek.s42.full_aware"):
        e = make(tmp_path, condition=name, run=f"n{len(name)}")[name]
        with pytest.raises(ManifestError, match="condition name"):
            verify_entry(tmp_path, e, REC)


def test_unicode_line_separators_inside_a_json_string_survive_load_manifest(tmp_path):
    rec = {**REC, "scenario_name": "a\u2028b\u2029c\x85d"}  # json.dumps(ensure_ascii=False) keeps them raw
    run_dir = write_run(tmp_path, "u", [{"record": rec, "condition": "neg", "slug": "f_3", "base": BASE, "files": {}},
                                        {"record": rec, "condition": "gold", "slug": "f_3", "base": BASE,
                                         "files": GOLD}])
    raw = (run_dir / "manifest.jsonl").read_bytes()
    assert "\u2028".encode() in raw
    entries = load_manifest(run_dir)
    assert [e.condition for e in entries] == ["neg", "gold"]
    raw_lines = raw.split(b"\n")[:2]
    assert [e.line.encode("utf-8") for e in entries] == raw_lines
    assert [e.overlay_sha256 for e in entries] == [hashlib.sha256(line).hexdigest() for line in raw_lines]
    assert json.loads(entries[0].line)["scenario_name"] == rec["scenario_name"]


def test_spike_manifest_reads_utf8_whatever_the_locale(tmp_path, monkeypatch):
    spike = tmp_path / "spike"
    neg = {A: "<?php\n// café\nclass A\n{\n}\n"}
    trees = {"neg": neg, "pos": {A: neg[A].replace("{\n}", "{\n    // gold é\n}")}, "stub_f": neg}
    for cond, tree in trees.items():
        for rel, text in tree.items():
            p = spike / "f_3" / cond / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))
    rec = {**REC, "id": REC["id"].replace("f.feature", "fé.feature")}  # non-ASCII in a field convert() reads
    (spike / "manifest.json").write_bytes(json.dumps([{"record_id": rec["id"], "dir": "x/f_3"}],
                                                     ensure_ascii=False).encode("utf-8"))
    read_text = Path.read_text

    def latin1_default(self, encoding=None, errors=None, **kw):  # a non-UTF-8 locale (e.g. Windows cp1252)
        return read_text(self, encoding or "latin-1", errors, **kw)
    monkeypatch.setattr(Path, "read_text", latin1_default)
    entries = {e.condition: e for e in load_manifest(convert(spike, tmp_path / "root", "s", {rec["id"]: rec}))}
    assert {e.record_id for e in entries.values()} == {rec["id"]}
    assert set(entries) == {"neg", "gold", "stub"} and entries["stub"].files == {}
    gold = entries["gold"]
    assert (tmp_path / "root" / gold.dir / A).read_bytes() == trees["pos"][A].encode("utf-8")
    assert (tmp_path / "root" / gold.base / A).read_bytes() == neg[A].encode("utf-8")


def test_added_lines_is_a_multiset_difference():
    assert added_lines("a\nb\n}\n", "a\nb\nx\n}\n}\n") == ["x", "}"]


def test_tamper_flags_only_on_added_lines_and_catch_known_bypasses(tmp_path):
    base = {A: "<?php\nclass A\n{\n    // file_put_contents('/tmp/x', 'y');\n}\n"}
    forged = ("    public function x(): void\n    {\n"
              "        $GLOBALS['erx'] = new class {\n"
              "            public function __destruct() {\n"
              "                $p = getenv('ER_TRACE_OUT');\n"
              "                $w = 'file_put' . '_contents';\n"
              "                $w($p, '{}');\n"
              "            }\n"
              "        };\n"
              "        $pdo = new \\PDO('mysql:host=db');\n"
              "        $pdo->exec('CREATE TRIGGER t BEFORE INSERT ON x');\n"
              "    }\n}\n")
    run_dir = write_run(tmp_path, "r", [{"record": REC, "condition": "m.s42.full_aware", "slug": "f_3", "base": base,
                                         "files": {A: base[A].replace("}\n", forged, 1)}}])
    (e,) = load_manifest(run_dir)
    assert {"dynamic_call", "env", "global_hook", "harness", "raw_db"} <= set(tamper_flags(tmp_path, e))
    clean = write_run(tmp_path, "c", [{"record": REC, "condition": "gold", "slug": "f_3", "base": base, "files": base}])
    assert tamper_flags(tmp_path, load_manifest(clean)[0]) == []


@pytest.mark.parametrize("line, flag", [
    ("public function __destruct() {", "global_hook"),  # rewrites the trace after the Recorder's shutdown function
    ("$GLOBALS['erx'] = new class {", "global_hook"),
    ("$w = 'file_put' . '_contents';", "dynamic_call"),  # a function name built by concatenation
    ("$w($p, '{}');", "dynamic_call"),
    ("$p = getenv('ER_TRACE_OUT');", "env"),
    ("$p = getenv('ER_TRACE_OUT');", "harness"),
    ("$pdo = new \\PDO('mysql:host=db');", "raw_db"),
    ("$pdo->exec('CREATE TRIGGER t BEFORE INSERT ON x');", "raw_db"),
    ("($w)($p, '{}');", "dynamic_call"),  # a call through a parenthesised variable
    ("(implode('', ['file_put', '_contents']))($p, '{}');", "dynamic_call"),  # a built name, called at once
    ("sprintf('%s_%s', 'file_put', 'contents')($p, '{}');", "dynamic_call"),
    ("(str_replace('X', '_', 'fileXputXcontents'))($p, '{}');", "dynamic_call"),
    ("strrev('stnetnoc_tup_elif')($p, '{}');", "dynamic_call"),
    ("(base64_decode('ZmlsZV9wdXRfY29udGVudHM='))($p, '{}');", "dynamic_call"),
    ("(chr(102) . 'ile_put_contents')($p, '{}');", "dynamic_call"),
])
def test_each_known_bypass_is_flagged_on_its_own_line(tmp_path, line, flag):
    base = {A: "<?php\nclass A\n{\n}\n"}
    run_dir = write_run(tmp_path, "r", [{"record": REC, "condition": "m.s42.full_aware", "slug": "f_3", "base": base,
                                         "files": {A: base[A].replace("}\n", f"    {line}\n}}\n", 1)}}])
    assert flag in tamper_flags(tmp_path, load_manifest(run_dir)[0])


@pytest.mark.parametrize("line", [  # gold lines of the control overlays: a built string that is not called
    "sprintf('Type \"%s\" is invalid. Allowed types are:', $type),",
    "sprintf('order[%s]', lcfirst(str_replace(' ', '', ucwords($field)))),",
])
def test_a_built_string_that_is_not_called_is_not_a_dynamic_call(tmp_path, line):
    base = {A: "<?php\nclass A\n{\n}\n"}
    run_dir = write_run(tmp_path, "r", [{"record": REC, "condition": "m.s42.full_aware", "slug": "f_3", "base": base,
                                         "files": {A: base[A].replace("}\n", f"    {line}\n}}\n", 1)}}])
    assert "dynamic_call" not in tamper_flags(tmp_path, load_manifest(run_dir)[0])
