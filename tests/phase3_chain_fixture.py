"""Small dated repos for tests/test_state_pins.py (rules 1–5 of v2tools/state.py), built like tests/phase3_fixture.py.

PHP chain (namespace App\\Ctx, src/Ctx/): ChildContext extends BaseContext and uses OwnTrait; BaseContext uses
ParentTrait and promotes `protected Foo $bar`; Late gets `extends LateParent` only in 2022.
  2020  every file; movedUp in the child, movedDown in the parent, stepLater without annotation, private tock in
        ParentTrait
  2022  movedUp → parent, movedDown → child, stepLater gets #[When('I do later')], Late extends LateParent
  2024  the answer iAnswer; helper gains `int $b`, the constructor gains `NewDep $newDep`, editedStep's body changes,
        the parent adds addedAtT and takes tock over from ParentTrait
Rebase: A (2020) → B (authored 2024) → C (authored 2022, committed after B).
C#: Steps (block namespace A.B) gains `NewDep newDep` and `_newDep` in 2024. Java: two constructors, two go() overloads.
Deep: L0 extends L1 … extends L6 (depth 6). Rename: A.php → B.php, weak (R054) or exact (R100).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from tests.phase3_fixture import T2020, T2022, T2024, _git

T2021, T2023 = 1609459200, 1672531200
REPO = "Acme/Chain"
COMPOSER = '{"autoload": {"psr-4": {"App\\\\": "src/"}}}\n'
CHILD, BASE, OWN, PTRAIT, LATE, LATE_PARENT = (f"src/Ctx/{n}.php" for n in (
    "ChildContext", "BaseContext", "OwnTrait", "ParentTrait", "Late", "LateParent"))
REB = "src/Ctx/R.php"
CS = "Tests/Steps.cs"
JAVA = "src/test/java/org/x/Steps.java"
DEEP = [f"src/Deep/L{i}.php" for i in range(7)]


def build(tmp: Path, history: list[tuple[int, int, dict[str, str | None]]]) -> tuple[Path, str]:
    """Commit each (author time, committer time, {path: text, or None to delete}) in order, then clone bare; returns
    (bare clone, HEAD)."""
    work = tmp / "work"
    work.mkdir(parents=True)
    _git(work, "init", "-q", "-b", "main")
    for at, ct, files in history:
        for p, text in files.items():
            if text is None:
                (work / p).unlink()
            else:
                (work / p).parent.mkdir(parents=True, exist_ok=True)
                (work / p).write_text(text)
        _git(work, "add", "-A")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t", "GIT_AUTHOR_DATE": f"@{at} +0000", "GIT_COMMITTER_DATE": f"@{ct} +0000"}
        subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", f"c{at}"], check=True, capture_output=True, env=env)
    bare = tmp / "repo.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    return bare, _git(bare, "rev-parse", "HEAD").strip()


def _php(kind_name: str, body: str) -> str:
    return f"<?php\n\nnamespace App\\Ctx;\n\n{kind_name}\n{{\n{body}}}\n"


def _method(sig: str, body: str = "") -> str:
    return f"    {sig}\n    {{\n" + (f"        {body}\n" if body else "") + "    }\n"


PTRAIT_2020 = _php("trait ParentTrait", _method("public function parentTraitPublic(): void") + "\n"
                   + _method("private function parentTraitSecret(): void") + "\n" + _method("private function tock(): void"))
PTRAIT_2024 = _php("trait ParentTrait", _method("public function parentTraitPublic(): void") + "\n"
                   + _method("private function parentTraitSecret(): void"))
OWN_2020 = _php("trait OwnTrait", _method("private function ownTraitSecret(): void"))

_BASE_HEAD = "    use ParentTrait;\n\n" + _method("public function __construct(protected Foo $bar)") + "\n" \
    + _method("public function sharedName(): string", "return 'base';") + "\n" \
    + _method("protected function helperInfra(): void") + "\n"
_BASE_TAIL = _method("private function baseSecret(): void")
BASE_2020 = _php("abstract class BaseContext", _BASE_HEAD + _method("protected function movedDown(): void") + "\n" + _BASE_TAIL)
BASE_2022 = _php("abstract class BaseContext", _BASE_HEAD + _method("protected function movedUp(): void") + "\n" + _BASE_TAIL)
BASE_2024 = _php("abstract class BaseContext", _BASE_HEAD + _method("protected function movedUp(): void") + "\n"
                 + _method("protected function addedAtT(): void") + "\n" + _method("protected function tock(): void") + "\n"
                 + _BASE_TAIL)


def _child(ctor: str, moved: str, helper: str, later_ann: str, edited: str, answer: bool) -> str:
    body = "    use OwnTrait;\n\n" + _method(f"public function __construct({ctor})") + "\n" \
        + _method("public function sharedName(): string", "return 'child';") + "\n" \
        + _method(f"protected function {moved}(): void") + "\n" \
        + _method(f"public function helper({helper}): void") + "\n" \
        + _method("public function stable(string $a): void") + "\n" \
        + later_ann + _method("public function stepLater(): void") + "\n" \
        + "    #[Given('I edit')]\n" + _method("public function editedStep(): void", f"$this->stable('{edited}');") + "\n" \
        + "    #[Given('I stay')]\n" + _method("public function untouchedStep(): void", "$this->helperInfra();")
    if answer:
        body += "\n    #[Then('I answer')]\n" + _method("public function iAnswer(): void", "$this->helper('x', 2);")
    return _php("final class ChildContext extends BaseContext", body)


CHILD_2020 = _child("OldDep $oldDep", "movedUp", "string $a", "", "one", False)
CHILD_2022 = _child("OldDep $oldDep", "movedDown", "string $a", "    #[When('I do later')]\n", "one", False)
CHILD_2024 = _child("OldDep $oldDep, NewDep $newDep", "movedDown", "string $a, int $b", "    #[When('I do later')]\n",
                    "two", True)
LATE_PARENT_2020 = _php("class LateParent", _method("public function old(): void"))
LATE_2020 = _php("class Late", _method("public function own(): void"))
LATE_2022 = _php("class Late extends LateParent", _method("public function own(): void"))


def chain_repo(tmp: Path) -> tuple[Path, str]:
    return build(tmp, [
        (T2020, T2020, {"composer.json": COMPOSER, PTRAIT: PTRAIT_2020, OWN: OWN_2020, BASE: BASE_2020, CHILD: CHILD_2020,
                        LATE_PARENT: LATE_PARENT_2020, LATE: LATE_2020}),
        (T2022, T2022, {BASE: BASE_2022, CHILD: CHILD_2022, LATE: LATE_2022}),
        (T2024, T2024, {BASE: BASE_2024, CHILD: CHILD_2024, PTRAIT: PTRAIT_2024})])


def _step(name: str, time: int, text: str = CHILD_2024, path: str = CHILD) -> dict:
    lines = text.splitlines()
    decl = next(i for i, l in enumerate(lines, 1) if f"function {name}(" in l)
    keyword, pattern = re.fullmatch(r"\s*#\[(\w+)\('(.*)'\)\]", lines[decl - 2]).groups()
    end = next(i for i in range(decl, len(lines) + 1) if lines[i - 1] == "    }")
    return {"repo": REPO, "fn_id": f"{path}:{decl + 1}", "path": path, "first_line": decl - 1, "body_end": end,
            "code": "\n".join(l[4:] for l in lines[decl - 2:end]),
            "annotations": [{"def_id": f"{path}:{decl - 1}", "line": decl - 1, "keyword": keyword, "pattern": pattern,
                             "time": time}],
            "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}


def chain_catalog() -> dict[str, dict]:
    rows = [_step("editedStep", T2020), _step("untouchedStep", T2020), _step("stepLater", T2022), _step("iAnswer", T2024)]
    return {r["fn_id"]: r for r in rows}


def rebase_repo(tmp: Path) -> tuple[Path, str]:
    a = _php("class R", _method("public function a(): void"))
    b = _php("class R", _method("public function a(): void") + _method("public function b(): void"))
    c = _php("class R", _method("public function a(): void") + _method("public function b(): void")
             + _method("public function c(): void"))
    return build(tmp, [(T2020, T2020, {"composer.json": COMPOSER, REB: a}), (T2024, T2024, {REB: b}),
                       (T2022, T2024 + 3600, {REB: c})])


CS_2020 = '''using System;

namespace A.B
{
    public class Steps
    {
        private readonly OldDep _oldDep;

        public Steps(OldDep oldDep)
        {
            _oldDep = oldDep;
        }

        public void Go(string a)
        {
            _oldDep.Run(a);
        }
    }
}
'''
CS_2024 = CS_2020.replace("        private readonly OldDep _oldDep;\n", "        private readonly OldDep _oldDep;\n"
                          "        private readonly NewDep _newDep;\n").replace(
    "public Steps(OldDep oldDep)", "public Steps(OldDep oldDep, NewDep newDep)").replace(
    "            _oldDep = oldDep;\n", "            _oldDep = oldDep;\n            _newDep = newDep;\n")


def csharp_repo(tmp: Path) -> tuple[Path, str]:
    return build(tmp, [(T2020, T2020, {CS: CS_2020}), (T2024, T2024, {CS: CS_2024})])


JAVA_2020 = '''package org.x;

public class Steps {
    private final Other other;
    private final int size;

    public Steps(Other other) {
        this.other = other;
        this.size = 0;
    }

    public Steps(Dep dep, int size) {
        this.other = null;
        this.size = size;
    }

    public void go(String a) {
    }

    public void go(String a, int b) {
    }
}
'''


def java_repo(tmp: Path) -> tuple[Path, str]:
    return build(tmp, [(T2020, T2020, {JAVA: JAVA_2020})])


def deep_repo(tmp: Path) -> tuple[Path, str]:
    files = {p: f"<?php\n\nnamespace App\\Deep;\n\nclass L{i}" + (f" extends L{i + 1}" if i < 6 else "") + "\n{\n}\n"
             for i, p in enumerate(DEEP)}
    return build(tmp, [(T2020, T2020, {"composer.json": COMPOSER, **files})])


def rename_repo(tmp: Path, keep: int) -> tuple[Path, str]:
    """A.php (2020, 20 methods) renamed to B.php in 2022 keeping its first `keep` methods: 11 gives R054, 20 R100."""
    old = [f"    public function old{i}(): void {{ $this->x{i}(); }}" for i in range(20)]
    new = old[:keep] + [f"    public function new{i}(): void {{ $this->y{i}(); }}" for i in range(20 - keep)]
    return build(tmp, [(T2020, T2020, {"A.php": "<?php\nclass A\n{\n" + "\n".join(old) + "\n}\n"}),
                       (T2022, T2022, {"A.php": None, "B.php": "<?php\nclass A\n{\n" + "\n".join(new) + "\n}\n"})])


def copy_repo(tmp: Path) -> tuple[Path, str]:
    """A.php (2020) is copied to B.php in 2022 with one method renamed and the source kept: git --follow reports C094."""
    body = "\n".join(f"    public function old{i}(): void {{ $this->x{i}(); }}" for i in range(20))
    return build(tmp, [(T2020, T2020, {"A.php": "<?php\nclass A\n{\n" + body + "\n}\n"}),
                       (T2022, T2022, {"B.php": "<?php\nclass B\n{\n" + body.replace("old19", "fresh19") + "\n}\n"})])


TWO = "src/Ctx/Two.php"


def two_class_repo(tmp: Path) -> tuple[Path, str]:
    """Two.php holds a Helper class before TwoContext until 2022; from 2022 only TwoContext remains."""
    helper = "final class Helper\n{\n    public function helperOnly(): void {}\n}\n\n"
    ctx = "final class TwoContext\n{\n    public function stepOne(): void {}\n}\n"
    head = "<?php\n\nnamespace App\\Ctx;\n\n"
    return build(tmp, [(T2020, T2020, {"composer.json": COMPOSER, TWO: head + helper + ctx}),
                       (T2022, T2022, {TWO: head + ctx})])
