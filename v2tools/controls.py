"""Pre-declared controls of Phase 3 scoring (§2.2 and §2.2b of the design specification; fixed 24/09/2026), run on the
sealed data before the SAP is locked and reported next to every condition, per repo:
- gold: the gold answer (positive control of BV and CA);
- stub-v1: one function per CREATE step, the literal step as an anchored regex, no parameter, empty body;
- stub-v2: values and numbers as groups, the right parameter count, one meaningless statement (fails BV condition 5);
- stub-v3: as v2, the body touches a real project member (the ceiling of BV);
- stub-v3c: as v2, the body calls the first public method of the first injected service of the target class, in the ④(a)
order of V(r);
- stub-v3m: as v2, the body makes the (member, method) call most frequent in the VISIBLE step functions of the target
file at t_vis (ties alphabetical): a lazy but informed writer, using only what a model is shown;
- stub-v3all: as v2, the body makes EVERY call seen in those visible step functions, `$this->f()` included, in sorted
order: a writer that sprays what it sees without knowing what the step means;
- swap: every gold function keeps its annotations and signature but takes the body of the gold function of ANOTHER
record with the same target file (chosen by sha256), parameters renamed by position.
Helper controls (`d10_controls`), for PHP, Java and C# records only (Ruby has no future methods: no class; fixed
25/09/2026):
- gold-helpers: the gold answer plus, appended to each target file's block, every FUTURE method the gold's expansion
reads (evaluate.expanded_futures: to depth 3, a constructor never), verbatim from the ER tree (positive control);
- gold-helpers-renamed: the same, each helper renamed probeHelper<k> at its definition and every call site;
- stub-v3m-h, stub-v3all-h: stub-v3m, stub-v3all whose steps only call one self-written helper (`HELPER`) holding the
stub's body (the v3m call; every visible call): a helper contributes exactly its own calls, so CA is the stub's.
D11 name and top-4 controls (the design specification at 3739dd6, fixed 25/09/2026; `d11_controls`), for PHP, Java and
C# records (Ruby: 3A's ④(a) lists no method):
- stub-v3names: stub-v3all with, at the top of every stub step body, a call without arguments on the class itself to
every METHOD name V(r) shows for the file (`shown_methods`), sorted: the third lazy reference (a name spray);
- stub-v3top4 (D11 (d), added 26/09/2026): stub-v3all's functions, each body the at most 4 calls of `top4_calls` (the
file's most frequent visible calls, filled up from the dev golds' most frequent calls; a new file: the methods its
ancestors show, ranked by the dev golds), when controls() gets the dev `priors`.
v3c, v3m, v3all, v3names, v3top4 and swap are the reference levels of CA ("CA a lazy writer reaches"); the CA of v3m,
v3all, v3names and v3top4 are the ca_ref_v3m, ca_ref_v3all, ca_ref_v3names and ca_ref_top4 columns (score_run). What the
lazy writer reads is what 3A shows (rule 5 of V(r), state.py): ④(a) through state.t_view, the step functions ④(b) could
show through records.snapshot_fns, state.pre_t_code and state.future_ref, less what 3A's composer keeps out of ④ for the
record (`Shown4`: its leak rules, its masked infrastructure)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from .compose import gold_answer
from .evaluate import Prepared, Score, expanded_futures, score_blocks, shown_members
from .extract import extract
from .ertree import parser
from .insert import IMPORT, IMPORT_NODES, STEP_ANNOTATION, split_code
from .records import hidden_for, snapshot_fns
from .state import future_ref, pre_t_code, render_visible, t_view
from .symbols import class_status, lambda_text, refs, scope

MODIFIERS = {"public", "private", "protected", "readonly", "final", "static", "transient", "volatile", "var"}


def _escape(s: str) -> str:
    return re.sub(r"([\\.^$|?*+()\[\]{}/#-])", r"\\\1", s)


def generalize(step: str) -> tuple[str, int]:
    """Anchored regex of a step with each double-quoted value and each number as a group, and the group count."""
    out, pos, n = [], 0, 0
    for m in re.finditer(r'"[^"]*"|(?<![\w.])-?\d+(?:[.,]\d+)*(?![\w])', step):
        out.append(_escape(step[pos:m.start()]))
        out.append('"([^"]*)"' if m.group(0).startswith('"') else r"(-?\d+(?:[.,]\d+)*)")
        n += 1
        pos = m.end()
    out.append(_escape(step[pos:]))
    return "^" + "".join(out) + "$", n


def stub_item(style: str, i: int, pattern: str, n_params: int, body: str) -> str:
    """One step function in the repo's main style: regex pattern, `n_params` parameters, `body` statements."""
    if style.startswith("php"):
        params = ", ".join(f"$p{k}" for k in range(1, n_params + 1))
        quoted = ("/" + pattern + "/").replace("'", "\\'")  # single-quoted PHP: only the quote is escaped
        head = f"#[When('{quoted}')]" if style == "php_attribute" else f"/**\n * @When /{pattern}/\n */"
        return f"{head}\npublic function stepStub{i}({params}): void\n{{\n{body}}}"
    if style.startswith("java"):
        params = ", ".join(f"String p{k}" for k in range(1, n_params + 1))
        lit = pattern.replace("\\", "\\\\").replace('"', '\\"')
        return f'@When("{lit}")\npublic void stepStub{i}({params}) {{\n{body}}}'
    if style == "csharp_attribute":
        params = ", ".join(f"string p{k}" for k in range(1, n_params + 1))
        return f'[When(@"{pattern.replace(chr(34), chr(34) * 2)}")]\npublic void StepStub{i}({params})\n{{\n{body}}}'
    params = ", ".join(f"p{k}" for k in range(1, n_params + 1))
    return f"When(/{pattern}/) do{f' |{params}|' if params else ''}\n{body}end"


def meaningless(language: str) -> str:
    return {"php": "    $x = 1;\n", "java": "    int x = 1;\n", "csharp": "    var x = 1;\n", "ruby": "  x = 1\n"}[language]


def touch(language: str, member: str) -> str:
    return {"php": f"    $this->{member};\n", "java": f"    Object o = this.{member};\n",
            "csharp": f"    var o = this.{member};\n", "ruby": f"  {member}\n"}[language]


def call_body(language: str, call: tuple[str, str]) -> str:
    m, f = call
    if language == "php":
        if m == "self":
            return f"    $this->{f}();\n"
        return f"    {m}::{f}();\n" if m[:1].isupper() else f"    $this->{m}->{f}();\n"
    if language == "ruby":
        return f"  {f}\n"
    return f"    {m}.{f}();\n" if m != "self" else f"    {f}();\n"


# ----- the member a lazy writer would use -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Shown4:
    """What 3A's composer keeps out of ④ for one record: `dropped` = the ④(b) functions its
    leak rules drop (Composer.knowledge_candidates), `masked` = the test-answer infrastructure it masks (Composer.masked;
    empty for test and aux_test). The default keeps nothing out."""
    dropped: frozenset = frozenset()
    masked: frozenset = frozenset()


def shown4(comp, r: dict) -> Shown4:
    """Shown4 of record `r` from a compose.Composer holding every record of the repo (all partitions)."""
    dropped: list = []
    comp.knowledge_candidates(r, comp.answer(r), dropped)
    return Shown4(frozenset(d["id"] for d in dropped if d["kind"] == "knowledge"), comp.masked(r))


def _gone(p: Prepared) -> set[str]:
    r = p.record
    return {c["fn_id"] for c in r["create"] if not c["preexisting"]} | hidden_for(r["partition"], p.repo.forbidden.get(r["repo"], {}))


def member_type(member) -> str | None:
    """Declared class type of a property, promoted parameter or field (outline.Member), or None."""
    t = re.sub(r"@\w+(\([^)]*\))?", " ", member.text).split("=")[0].replace(";", " ").replace(",", " ")
    toks = [x for x in t.split() if x not in MODIFIERS]
    i = next((k for k, x in enumerate(toks) if x.lstrip("$") in member.names), None)
    if not i:
        return None
    typ = toks[i - 1].lstrip("?").split("<", 1)[0]
    return typ if typ[:1].isupper() else None


def first_service_call(p: Prepared, path: str, k: Shown4 = Shown4()) -> tuple[str, str] | None:
    """stub-v3c: (first injected member of ④(a) whose class resolves in the repo, its first public method), read as the
    model is shown them (rule 5 of V(r), 3A's state.t_view): the members in their pre-T form, the service class as it
    was before t_vis."""
    ctx, t = p.repo.ctx, p.record["t_vis"]
    for s in t_view(ctx, path, t, _gone(p), k.masked) or ():
        m = s.member
        if m.kind not in ("property", "promoted"):
            continue
        typ, o0 = member_type(m), ctx.outline_before(s.origin, t)
        par = ctx.resolver.resolve(o0, "ref", typ) if typ and o0 is not None else None
        o = ctx.outline_before(par.path, t) if par is not None and par.status == "resolved" and par.path else None
        if o is None:
            continue
        f = next((x.names[0] for x in o.members if x.kind == "method" and x.visibility in ("public", "")
                  and not x.names[0].startswith("__")), None)
        if f:
            return m.names[0], f
    return None


def visible_calls(p: Prepared, path: str, k: Shown4 = Shown4()) -> Counter:
    """(member or "self", method) -> number of step functions of `path` that ④(b) could show that make that call: 3A's
    records.snapshot_fns (suite or scope, catalog flags, a pattern before the scenario, the partition's hidden
    functions, the record's new answers), each read in the form the model is shown (rule 5 of V(r) through
    state.pre_t_code: the HEAD code when unchanged since t_vis, else its pre-T form); a function whose pre-T form cannot
    be located, whose shown code uses a future or masked member (rule 3, state.future_ref), or that the composer's leak
    rules drop (`k.dropped`), is not counted. Unlike ④(b), no BM25 cut: every function it could show counts."""
    from .calls import calls
    from .outline import parse
    repo, r = p.repo, p.record
    t, cdefs, gone = r["t_vis"], frozenset(r["create_def_ids"]), _gone(p)
    text = p.er_files.get(path, repo.ctx.text(path))
    sc = scope(repo.ctx, path, text, text, p.er_files) if repo.language != "ruby" else None
    o = parse(repo.language, text) if repo.language != "ruby" else None
    fut = repo.ctx.future(path, t, k.masked) if repo.language != "ruby" else set()
    counts: Counter = Counter()
    for fid in snapshot_fns(r, repo.fns, gone):
        f = repo.fns[fid]
        if f["path"] != path or fid in k.dropped:
            continue
        pre = pre_t_code(repo.ctx, fid, t)
        if pre == "" or future_ref(repo.ctx, f, t, k.masked, pre):
            continue
        code = render_visible(f, t, cdefs) if pre is None else pre
        found = calls(repo.language, code, ctx=repo.ctx, o_after=o, members=sc.available if sc else frozenset(),
                      world=repo.world().methods if repo.language == "ruby" else frozenset(),
                      loggers=sc.loggers if sc else frozenset())
        if repo.language == "php" and sc is not None and not sc.open_all:  # the ER-tree filter
            # holds for PHP too: a pre-T call to a member or a method the ER tree no longer has is not counted (as
            # jvm_calls never counts one); a class call (X::f) names a class php_calls found in the tree, kept
            classes = {c.lstrip("\\") for c in refs("php", code).classes}
            found = {(m, fn) for m, fn in found if m in classes or (fn.casefold() in sc.methods if m == "self"
                                                                         else m in sc.available)}
        for m, fn in found:
            if m not in fut and (m != "self" or fn not in fut):
                counts[(m, fn)] += 1
    return counts


def frequent_call(p: Prepared, path: str, k: Shown4 = Shown4()) -> tuple[str, str] | None:
    """stub-v3m: the (member, method) call most frequent in the visible step functions of `path` at t_vis (calls on
    the class itself excluded, except in Ruby where every call is one)."""
    counts = {c: v for c, v in visible_calls(p, path, k).items() if c[0] != "self" or p.repo.language == "ruby"}
    if not counts:
        return None
    best = max(counts.values())
    return min(c for c, v in counts.items() if v == best)


def lazy_body(p: Prepared, version: str, path: str, k: Shown4 = Shown4(), memo: dict | None = None) -> tuple[str, bool]:
    """(body, fell back) of the stub functions of `path` for stub-v3m or stub-v3all (one body per file); `memo` shares
    them between a stub and its -h variant within one controls() call."""
    if memo is not None and (version, path) in memo:
        return memo[(version, path)]
    lang = p.repo.language
    if version == "v3m":
        call = frequent_call(p, path, k)
        got = (call_body(lang, call), False) if call else (meaningless(lang), True)
    else:
        seen = sorted(visible_calls(p, path, k))
        got = ("".join(call_body(lang, c) for c in seen), False) if seen else (meaningless(lang), True)
    if memo is not None:
        memo[(version, path)] = got
    return got


def stubs(p: Prepared, version: str, k: Shown4 = Shown4(), memo: dict | None = None,
          prior: "DevPrior | None" = None) -> tuple[dict[str, str], int]:
    """(blocks, functions whose body fell back to a meaningless statement) of stub-v1/v2/v3/v3c/v3m/v3all and of the
    helper variants v3m-h/v3all-h: one function per CREATE step (stub-v1) or per generalised pattern (the others), in
    the file of the function that binds the step at HEAD. A -h variant (not for Ruby) writes the same functions, each
    calling only the helper HELPER instead, and appends to the file's block that one helper, whose body is the stub's;
    when the stub falls back, so does the -h variant, with no helper (`memo`: see lazy_body). D11's v3names (not for
    Ruby) writes stub-v3all's functions, each body headed by a call to every method `shown_methods` gives for the file.
    D11 (d)'s v3top4 (added 26/09/2026; not for Ruby) writes them with the body of top4_calls (`prior`: the repo's
    DevPrior)."""
    r, lang, style = p.record, p.repo.language, p.repo.styles[0]
    path_of = {s: fid.rsplit(":", 1)[0] for s, fid in zip(r["steps"], r["step_fns"])}
    out: dict[str, list[str]] = {}
    done, fallback, helpers = set(), 0, {}
    for i, (s, has_arg) in enumerate(p.create_steps, start=1):
        path = path_of[s]
        if version == "v1":
            out.setdefault(path, []).append(stub_item(style, i, "^" + _escape(s) + "$", 0, ""))
            continue
        pattern, n = generalize(s)
        if pattern in done:
            continue
        done.add(pattern)
        body = meaningless(lang)
        if version == "v3":
            if lang == "ruby":
                member = min(p.repo.world().methods, default=None)
            else:
                text = p.er_files.get(path, p.repo.ctx.text(path))
                sc = scope(p.repo.ctx, path, text, text, p.er_files)
                member = min(sc.project) if sc and sc.project else None
            body = touch(lang, member) if member else body
            fallback += member is None
        elif version == "v3c":
            call = first_service_call(p, path, k) if lang != "ruby" else None
            body = call_body(lang, call) if call else body
            fallback += call is None
        elif version == "v3names":  # D11: every method V(r) shows, then stub-v3all's own body
            body, fell = lazy_body(p, "v3all", path, k, memo)
            names = shown_methods(p, path)
            body = "".join(call_body(lang, ("self", n)) for n in names) + body
            fallback += fell and not names
        elif version == "v3top4":  # D11 (d): at most 4 calls, the same for every stub of the file
            body, fell = top4_body(p, path, k, prior, memo)
            fallback += fell
        elif version in ("v3m", "v3all", "v3m-h", "v3all-h"):
            body, fell = lazy_body(p, version.removesuffix("-h"), path, k, memo)
            fallback += fell
            if version.endswith("-h") and not fell:
                helpers.setdefault(path, body)
                body = call_body(lang, ("self", helper_name(lang)))
        out.setdefault(path, []).append(stub_item(style, i, pattern, n + has_arg, body))
    for path, body in helpers.items():
        out[path].append(helper_def(lang, helper_name(lang), body))
    # design specification §2.2 at 2ae2b6f: the stubs' `When` resolves through the block's own import
    head = step_import(style)
    return {k: (head + "\n\n" if head else "") + "\n\n".join(v) for k, v in out.items()}, fallback


def step_import(style: str) -> str | None:
    """The import a stub block opens with, so that its step annotation `When` names the framework's class wherever the
    target file's own imports stand (design specification §2.2 at 2ae2b6f: the reader resolves names as the language
    does), as a valid output would: PHP attributes `use Behat\\Step\\When;`, Java `import io.cucumber.java.en.When;`, C#
    `using TechTalk.SpecFlow;`; None for a PHP docblock and Ruby, which import nothing."""
    if style == "php_attribute":
        return "use Behat\\Step\\When;"
    if style.startswith("java"):
        return "import io.cucumber.java.en.When;"
    return "using TechTalk.SpecFlow;" if style == "csharp_attribute" else None


# ----- swap ----------------------------------------------------------------------------------------------------------

def body_span(language: str, text: str) -> tuple[int, int] | None:
    """Character span of the body braces of one function (a Java lambda step: of the lambda)."""
    lam = lambda_text(language, text)
    if lam is not None:
        j = text.find(lam)
        return j + lam.find("{"), j + len(lam)
    pre = "<?php class __X {\n" if language == "php" else "class __X {\n"
    src = (pre + text + "\n}").encode()
    stack = [parser(language).parse(src).root_node]
    while stack:
        n = stack.pop()
        if n.type == "method_declaration" and n.child_by_field_name("body") is not None:
            b, off = n.child_by_field_name("body"), len(pre.encode())
            return len(src[off:b.start_byte].decode("utf-8", "replace")), len(src[off:b.end_byte].decode("utf-8", "replace"))
        stack.extend(reversed(n.children))
    return None


def param_names(language: str, text: str) -> list[str]:
    """Parameter names of one function (a Java lambda step: of the lambda), in order."""
    lam = lambda_text(language, text)
    if lam is not None:
        inner = lam[:lam.find("->")].strip().strip("()")
        return [x.split()[-1] for x in inner.split(",") if x.strip()]
    pre = "<?php class __X {\n" if language == "php" else "class __X {\n"
    src = (pre + text + "\n}").encode()
    stack = [parser(language).parse(src).root_node]
    while stack:
        n = stack.pop()
        if n.type == "method_declaration" and n.child_by_field_name("parameters") is not None:
            out = []
            for prm in n.child_by_field_name("parameters").named_children:
                name = prm.child_by_field_name("name") or next((c for c in prm.children if c.type == "variable_name"), None)
                if name is None:
                    name = next((c for c in reversed(prm.named_children) if c.type in ("identifier", "variable_declarator")), None)
                if name is not None:
                    out.append(src[name.start_byte:name.end_byte].decode("utf-8", "replace").lstrip("$"))
            return out
        stack.extend(reversed(n.children))
    return []


def donors(preps: list[Prepared]) -> dict[str, list[tuple[str, str]]]:
    """Target file -> (record id, gold function text) over the given records (one partition)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for p in preps:
        for path, code in extract(gold_answer(p.record, p.repo.fns), p.targets).by_path().items():
            for it in split_code(p.repo.language, code)[1]:
                out.setdefault(path, []).append((p.record["id"], it.text))
    return out


def swap(p: Prepared, pool: dict[str, list[tuple[str, str]]]) -> dict[str, str] | None:
    """Gold blocks with every function's body taken from a donor; None when some function has no donor. A donor is
    taken in a fixed order (sha256), skipping one whose body, after the positional renaming of its parameters, is the
    function's own body (the same or a twin function reached through another
    record's gold made 4/342 test swaps byte-identical to the gold); the renaming is one pass, so a new name that is a
    later old name is not renamed again."""
    lang, rid = p.repo.language, p.record["id"]
    if lang == "ruby":
        return None
    out = {}
    for path, code in extract(gold_answer(p.record, p.repo.fns), p.targets).by_path().items():
        new = []
        for it in split_code(lang, code)[1]:
            cands = [(d, t) for d, t in pool.get(path, []) if d != rid and t != it.text]
            span = body_span(lang, it.text)
            if not cands or span is None or it.kind not in ("method", "lambda"):
                return None
            body = None
            for d, dt in sorted(cands, key=lambda c: hashlib.sha256(f"{rid}|{it.text}|{c[0]}|{c[1]}".encode()).hexdigest()):
                dspan = body_span(lang, dt)
                if dspan is None:
                    return None
                got = dt[dspan[0]:dspan[1]]
                ren = {("$" + o if lang == "php" else o): ("$" + n if lang == "php" else n)
                       for o, n in zip(param_names(lang, dt), param_names(lang, it.text))}
                if ren:  # one pass: a new name that is a later old name is not renamed again
                    got = re.sub(rf"(?<![\w$])(?:{'|'.join(re.escape(k) for k in sorted(ren, key=len, reverse=True))})(?!\w)",
                                 lambda m: ren[m.group(0)], got)
                if got != it.text[span[0]:span[1]]:  # never the function's own body
                    body = got
                    break
            if body is None:
                return None
            new.append(it.text[:span[0]] + body + it.text[span[1]:])
        out[path] = "\n\n".join(new)
    return out


# ----- the helper controls --------------------------------------------------------------------------------------------

# Ruby has no future methods (no class; fixed 25/09/2026): no helper control for its records
D10_LANGUAGES = ("php", "java", "csharp")
HELPER = {"php": "probeHelper", "java": "probeHelper", "csharp": "ProbeHelper"}  # C#: PascalCase, like StepStub<i>


def helper_name(language: str, k: int | None = None) -> str:
    """The self-written helper of stub-v3m-h and stub-v3all-h (`k` None), or the k-th renamed helper of
    gold-helpers-renamed."""
    return HELPER[language] + ("" if k is None else str(k))


def helper_def(language: str, name: str, body: str) -> str:
    """A private helper `name`, no parameter, no result, whose body is the statements `body`, in the language's form."""
    if language == "php":
        return f"private function {name}(): void\n{{\n{body}}}"
    if language == "csharp":
        return f"private void {name}()\n{{\n{body}}}"
    return f"private void {name}() {{\n{body}}}"


def rename_calls(language: str, text: str, names: dict[str, str]) -> str:
    """`text` with each method of `names` (old name -> new name) renamed at its definition and at its calls on the class
    itself, on exact identifier boundaries. PHP (names case-insensitive): `function f(`, `$this->f(`, `$this?->f(`,
    `self::f(`, `static::f(`; `parent::f(` stays (the ancestor's own method, which the ER tree keeps). Java, C#: a bare
    `f(` (the definition included) or `this.f(`; never `x.f(`, `super.f(`, `base.f(`, `new f(`. Anything else (a string,
    a method reference, a callable array) is left as it is; the ER tree still declares the old name, so it resolves."""
    if not names:
        return text
    fold = str.casefold if language == "php" else (lambda n: n)
    new = {fold(old): nw for old, nw in names.items()}
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    if language == "php":
        pat = (r"(?P<pre>(?:\$this\s*\??->|(?<![\w$\\])(?i:self|static)\s*::|(?<![\w$])(?i:function)\s+&?)\s*)"
               rf"(?P<name>(?i:{alt}))(?![\w])(?=\s*\()")
        return re.sub(pat, lambda m: m.group("pre") + new[fold(m.group("name"))], text)
    pat = rf"(?P<pre>(?<![\w$.])this\s*\.\s*)?(?<![\w$])(?P<name>{alt})(?![\w$])(?=\s*(?:<[^<>()]*>\s*)?\()"

    def repl(m):
        before = text[:m.start()].rstrip()
        if m.group("pre") is None and (before.endswith((".", "::")) or re.search(r"(?<![\w$])new$", before)):
            return m.group(0)
        return (m.group("pre") or "") + new[m.group("name")]
    return re.sub(pat, repl, text)


def _norm(line: str) -> str:
    return " ".join(line.split())


def import_lines(language: str, text: str) -> list[str]:
    """The file's top-level import lines (PHP namespace `use`, Java `import`, C# `using`; inside a namespace block too,
    where insert._merge_imports reads them), in order. Only a one-line import of the form insert.split_code takes as a
    block's import line (insert.IMPORT) is kept: a multi-line group use or a comma list is left out; a trait `use` in a
    class body is never one."""
    if language not in D10_LANGUAGES or not text:
        return []
    src, out = text.encode(), []

    def walk(n):
        for c in n.children:
            if c.type in IMPORT_NODES:
                line = src[c.start_byte:c.end_byte].decode("utf-8", errors="replace").strip()
                if "\n" not in line and IMPORT[language].match(line):
                    out.append(line)
            elif c.type == "namespace_definition" and c.child_by_field_name("body") is not None:
                walk(c.child_by_field_name("body"))
            elif c.type in ("namespace_declaration", "file_scoped_namespace_declaration", "declaration_list"):
                walk(c)
    walk(parser(language).parse(src).root_node)
    return out


def import_name(language: str, line: str) -> str | None:
    """The short name an import line binds: PHP its alias or last segment (a class, function or constant), Java the
    last segment (a class, or a static member), C# an alias; None for a PHP group use, a Java wildcard or a C#
    namespace (or static) using, which bind no single name."""
    if language == "php":
        m = re.match(r"\s*use\s+(?:function\s+|const\s+)?\\?([\w\\]+)(?:\s+as\s+(\w+))?\s*;", line)
        return None if m is None else m.group(2) or m.group(1).rsplit("\\", 1)[-1]
    if language == "java":
        m = re.match(r"\s*import\s+(?:static\s+)?([\w.]+)\s*;", line)
        return None if m is None else m.group(1).rsplit(".", 1)[-1]
    m = re.match(r"\s*using\s+(?:static\s+)?(\w+)\s*=", line)
    return None if m is None else m.group(1)


def needed_imports(language: str, origin_text: str, helper_text: str, have: list[str]) -> list[str]:
    """The import lines of `origin_text` (the file a copied helper comes from) that the helper needs and the target
    lacks (gold-helpers: the helper as at HEAD, with its imports; fixed 25/09/2026), in order: an import whose
    short name the helper's text uses as a name (PHP: case-insensitive, not as `$x`, `->x` or `\\x`; Java, C#: not as
    `.x`), or one that binds no single name (a wildcard, a C# namespace); never a line already in `have` (the target
    file's imports and those of the block), nor one whose short name `have` already binds (the target's wins)."""
    fold = str.casefold if language == "php" else (lambda n: n)
    seen = {_norm(x) for x in have}
    bound = {fold(n) for n in (import_name(language, x) for x in have) if n}
    out = []
    for line in import_lines(language, origin_text):
        name = import_name(language, line)
        if _norm(line) in seen:
            continue
        if name is not None:
            use = (rf"(?<![\w$\\])(?<!->)(?i:{re.escape(name)})(?!\w)" if language == "php"
                   else rf"(?<![\w$.]){re.escape(name)}(?![\w$])")
            if fold(name) in bound or not re.search(use, helper_text):
                continue
            bound.add(fold(name))
        out.append(line)
        seen.add(_norm(line))
    return out


def namespace_imports(p: Prepared, path: str, helper, have: list[str]) -> list[str]:
    """PHP: for `helper` (an evaluate.Future) copied from a file of another
    namespace than the target file `path`, `use <origin namespace>\\<Class>;` for each unqualified class it names that
    its file does not import and that resolves there (in the origin's namespace, inherit.Resolver) but not in the target
    file, in sorted order; never a name `have` (the lines already imported: the target file's and the block's) binds."""
    from .outline import parse
    ctx = p.repo.ctx
    oo = parse("php", p.er_files.get(helper.origin, ctx.text(helper.origin)) or "")
    to = parse("php", p.er_files.get(path, ctx.text(path)) or "")
    if oo is None or to is None or oo.namespace == to.namespace:
        return []
    bound = {n.casefold() for n in (import_name("php", x) for x in have) if n}
    out = []
    for c in sorted(refs("php", helper.text).classes):
        if "\\" in c or c in oo.imports or c.casefold() in bound:
            continue
        st = ctx.resolver.resolve(oo, "ref", c)
        if st.status == "resolved" and ctx.resolver.resolve(to, "ref", c).status != "resolved":
            out.append(f"use {st.fqn};")
            bound.add(c.casefold())
    return out


def strip_steps(language: str, text: str) -> str:
    """`text` without its step-annotation lines (insert.STEP_ANNOTATION: a PHP step attribute or docblock step line, a
    Java or C# step annotation): the renamed copy of a helper that is also a step function of the ER tree never binds
    that step a second time (condition 3)."""
    return "\n".join(line for line in text.split("\n") if not STEP_ANNOTATION[language].match(line))


def gold_helpers(p: Prepared, renamed: bool = False) -> tuple[dict[str, str], int]:
    """(blocks, fallback) of gold-helpers: the gold blocks with, appended to each target file's block, every FUTURE
    method the gold's expansion reads (evaluate.expanded_futures: to depth 3, in order; a constructor is never expanded,
    so never a helper), verbatim from the ER tree (Future.text; one of the file's own class replaces the ER tree's copy;
    an ancestor's is an override). A helper copied from another file (an ancestor or a trait) brings the import lines of
    that file it needs (needed_imports) and, in PHP, a `use` for each class of that file's own namespace it names
    (namespace_imports), put before the block's code, so that the insertion rule merges them into the target file's
    imports: the helper as at HEAD, with its imports. `renamed` (gold-helpers-renamed): each helper loses its step
    annotations (strip_steps) and its name becomes helper_name(language, k), k = 1, 2, … over the record in that order,
    one k per name (overloads stay overloads), at its definition and every call site in the gold functions and the
    helpers of the file (rename_calls). fallback = 1 when the gold reads no future helper: the control is then the gold
    answer itself."""
    lang, ctx = p.repo.language, p.repo.ctx
    gold = extract(gold_answer(p.record, p.repo.fns), p.targets).by_path()
    fut = expanded_futures(p, p.gold())
    fold = str.casefold if lang == "php" else (lambda n: n)
    out, k = {}, 0
    for path, code in gold.items():
        helpers = fut.get(path, [])
        have, imports = import_lines(lang, p.er_files.get(path, ctx.text(path)) or "") + split_code(lang, code)[0], []
        for f in helpers:
            if f.origin != path:
                imports += needed_imports(lang, p.er_files.get(f.origin, ctx.text(f.origin)) or "", f.text, have + imports)
                if lang == "php":
                    imports += namespace_imports(p, path, f, have + imports)
        text = "\n\n".join([code] + [strip_steps(lang, f.text) if renamed else f.text for f in helpers])
        if renamed and helpers:
            names: dict[str, str] = {}
            for f in helpers:
                if fold(f.name) not in {fold(n) for n in names}:
                    k += 1
                    names[f.name] = helper_name(lang, k)
            text = rename_calls(lang, text, names)
        out[path] = "\n".join(imports) + "\n\n" + text if imports else text
    return out, int(not any(fut.get(path) for path in gold))


# ----- D11 (fixed 25/09/2026; the design specification at 3739dd6): stub-v3names ---------------------------

V3NAMES_LANGUAGES = ("php", "java", "csharp")  # Ruby: 3A's ④(a) is None for a Ruby file (state.skeleton_parts)


def shown_methods(p: Prepared, path: str) -> list[str]:
    """D11 (a): the METHOD names V(r) shows for `path`, sorted, no repeat: the `method` members of
    evaluate.shown_members (state.t_view; for a file new at t_vis, what t_view shows for its in-repo ancestors, less
    their constructors and the privates outside RepoContext.access, as evaluate.shown_names); constructors, properties,
    constants and promoted properties are left out."""
    return sorted({n for s in shown_members(p, path) if s.member.kind == "method" for n in s.member.names})


# ----- D11 (d) (added 26/09/2026; design specification §0 D11, §2.2b): stub-v3top4 -------------------------------

TOP4 = 4  # D11 (d): the number of calls of a stub-v3top4 body


def _fold_call(language: str, call: tuple[str, str]) -> tuple[str, str]:
    """A call as CA compares it: a PHP method name casefolded (calls.agreement), never the receiver."""
    return (call[0], call[1].casefold()) if language == "php" else call


def _scenario(rid: str) -> str:
    """The "repo|feature_path|scenario_line" scenario key of a record id (rid rsplit once
    on "|"): every suite of the same scenario shares it (a scenario can bind the same steps under more than one
    suite), and a dev record can read its own gold through such a sibling suite's identical gold answer."""
    return rid.rsplit("|", 1)[0]


@dataclass(frozen=True)
class DevPrior:
    """D11 (d) (added 26/09/2026): the prior of stub-v3top4 for one repo (`dev_prior`): `golds` = (dev record id, the
    project calls of its gold, i.e. the union of Prepared.gold's call sets over its functions that bind CREATE steps),
    sorted by id."""
    repo: str
    language: str
    golds: tuple = ()

    def counts(self, exclude: str | None = None) -> Counter:
        """Call (as _fold_call) -> number of dev records whose gold holds it, every dev record of `exclude`'s own
        SCENARIO left out (a dev record never reads its own gold, nor that of a sibling
        suite of the SAME scenario — same repo, feature_path, scenario_line; record ids are unique across
        partitions)."""
        excl = _scenario(exclude) if exclude is not None else None
        out: Counter = Counter()
        for rid, atoms in self.golds:
            if excl is None or _scenario(rid) != excl:
                out.update({_fold_call(self.language, a) for a in atoms})
        return out

    def ranked(self, exclude: str | None = None) -> list[tuple[str, str]]:
        """The calls of `counts(exclude)`, most frequent first, ties by sorted (casefolded) call, each written in its
        smallest spelling among those dev golds."""
        excl = _scenario(exclude) if exclude is not None else None
        n, spell = self.counts(exclude), {}
        for rid, atoms in self.golds:
            if excl is not None and _scenario(rid) == excl:
                continue
            for a in atoms:
                f = _fold_call(self.language, a)
                spell[f] = min(spell.get(f, a), a)
        return [spell[f] for f in sorted(n, key=lambda f: (-n[f], f))]


def dev_prior(preps) -> dict[str, DevPrior]:
    """D11 (d): {repo: DevPrior} over the DEV partition's prepared records `preps` (prepared as for scoring the dev
    partition: evaluate.prepare with the record's Shown4.masked), PHP, Java and C# repos only; a record of another
    partition raises ValueError (no test gold may reach a prior)."""
    golds: dict[str, list] = {}
    language: dict[str, str] = {}
    for p in preps:
        r = p.record
        if r["partition"] != "dev":
            raise ValueError(f"{r['id']}: the stub-v3top4 prior reads dev records only, not {r['partition']}")
        if p.repo.language not in V3NAMES_LANGUAGES:
            continue
        g = p.gold()
        atoms = frozenset().union(*(g.calls.get(f, ()) for f in set(g.binders.values())))
        golds.setdefault(r["repo"], []).append((r["id"], atoms))
        language[r["repo"]] = p.repo.language
    return {repo: DevPrior(repo, language[repo], tuple(sorted(v))) for repo, v in sorted(golds.items())}


def top4_calls(p: Prepared, path: str, k: Shown4 = Shown4(), prior: DevPrior | None = None) -> list[tuple[str, str]]:
    """D11 (d) (added 26/09/2026): the calls, at most TOP4, of every stub-v3top4 body of `path`, read without the step
    text. `prior` is the repo's DevPrior (None: no dev record), read without the record's own gold. For a file NEW at
    t_vis (RepoContext.is_new_file: no visible step function): the METHODS the shown set holds (shown_methods, as
    stub-v3names), as ("self", name), by the prior's count of ("self", name) (PHP: any case), ties and names the prior
    never holds by name, the first TOP4. Else: first the file's visible calls (visible_calls, self calls included) most
    frequent first, ties by sorted call; then the prior's calls (DevPrior.ranked) the record can write in the file:
    ("self", f) with f a method shown_methods gives (any case in PHP, written as shown), a PHP static X::f whose class X
    resolves in the file (a project call, symbols.class_status), else (m, f) with m an available member of the class
    (Scope.available) that is not FUTURE or masked test-answer infrastructure (ctx.future(path, t_vis, k.masked), the
    same filter visible_calls applies: the lazy writer reads only what 3A shows); a call already taken (as CA compares
    it) is skipped; the first TOP4."""
    from .outline import parse
    lang, ctx = p.repo.language, p.repo.ctx
    fold = str.casefold if lang == "php" else (lambda n: n)
    names: dict[str, str] = {}
    for n in shown_methods(p, path):
        names.setdefault(fold(n), n)
    count = prior.counts(p.record["id"]) if prior is not None else Counter()
    if ctx.is_new_file(path, p.record["t_vis"]):
        ranked = sorted(names.values(), key=lambda n: (-count[("self", fold(n))], n))
        return [("self", n) for n in ranked[:TOP4]]
    text = p.er_files.get(path, ctx.text(path))
    sc = scope(ctx, path, text, text, p.er_files)
    o = parse(lang, text) if lang == "php" else None
    fut = ctx.future(path, p.record["t_vis"], k.masked)

    def writable(call: tuple[str, str]) -> tuple[str, str] | None:
        m, f = call
        if m == "self":
            return ("self", names[fold(f)]) if fold(f) in names else None
        if lang == "php" and m[:1].isupper():
            return call if o is not None and class_status(ctx, o, m) in ("resolved", "global") else None
        return call if sc is not None and m in sc.available and m not in fut else None
    vis = [c for c, _ in sorted(visible_calls(p, path, k).items(), key=lambda x: (-x[1], x[0]))]
    out, seen = [], set()
    for c in vis + [w for w in map(writable, prior.ranked(p.record["id"]) if prior is not None else []) if w]:
        if _fold_call(lang, c) not in seen and len(out) < TOP4:
            seen.add(_fold_call(lang, c))
            out.append(c)
    return out


def top4_body(p: Prepared, path: str, k: Shown4, prior: DevPrior | None, memo: dict | None) -> tuple[str, bool]:
    """(body, fell back) of the stub-v3top4 functions of `path`: its top4_calls, or a meaningless statement when there
    is none (`memo`: as lazy_body)."""
    if memo is not None and ("v3top4", path) in memo:
        return memo[("v3top4", path)]
    calls = top4_calls(p, path, k, prior)
    lang = p.repo.language
    got = ("".join(call_body(lang, c) for c in calls), False) if calls else (meaningless(lang), True)
    if memo is not None:
        memo[("v3top4", path)] = got
    return got


def d11_controls(p: Prepared, k: Shown4 = Shown4(), memo: dict | None = None,
                 priors: dict[str, DevPrior] | None = None) -> dict[str, tuple]:
    """D11's name and top-4 controls of a PHP, Java or C# record, scored like outputs; none for Ruby: stub-v3names,
    whose fallback counts the stub functions with no call at all (stub-v3all fell back and V(r) shows no method), then,
    when `priors` (dev_prior of the dev partition; a repo it lacks has no dev record) is given, D11 (d)'s stub-v3top4
    (added 26/09/2026), whose fallback counts the stub functions with no call (top4_calls empty)."""
    if p.repo.language not in V3NAMES_LANGUAGES:
        return {}
    blocks, fallback = stubs(p, "v3names", k, memo)
    out = {"stub-v3names": (score_blocks(p, blocks), fallback, blocks)}
    if priors is not None:
        blocks, fallback = stubs(p, "v3top4", k, memo, priors.get(p.record["repo"]))
        out["stub-v3top4"] = (score_blocks(p, blocks), fallback, blocks)
    return out


def d10_controls(p: Prepared, k: Shown4 = Shown4(), memo: dict | None = None) -> dict[str, tuple[Score, int, dict[str, str]]]:
    """The four helper controls of a PHP, Java or C# record, scored like outputs; none for Ruby (no future methods)."""
    if p.repo.language not in D10_LANGUAGES:
        return {}
    out = {}
    for name, renamed in (("gold-helpers", False), ("gold-helpers-renamed", True)):
        blocks, fallback = gold_helpers(p, renamed)
        out[name] = (score_blocks(p, blocks), fallback, blocks)
    for v in ("v3m-h", "v3all-h"):
        blocks, fallback = stubs(p, v, k, memo)
        out[f"stub-{v}"] = (score_blocks(p, blocks), fallback, blocks)
    return out


def controls(p: Prepared, pool: dict | None = None, k: Shown4 = Shown4(),
             priors: dict[str, DevPrior] | None = None) -> dict[str, tuple[Score, int, dict[str, str]]]:
    """Every control scored like an output: name -> (score, fallback functions, the blocks that were scored); `k` is
    what 3A keeps out of ④ for the record (Shown4); `priors` = dev_prior of the dev partition (D11 (d): stub-v3top4,
    only when given). The helper controls (d10_controls), then D11's name and top-4 controls (d11_controls), come last.
    The gold control is the gold answer scored exactly like an output (Prepared.gold_output: without D11 (e), the
    visible-shell delegation, which only the CA reference Prepared.gold holds)."""
    gold = extract(gold_answer(p.record, p.repo.fns), p.targets).by_path()
    out = {"gold": (p.gold_output(), 0, gold)}
    memo: dict = {}
    for v in ("v1", "v2", "v3", "v3c", "v3m", "v3all"):
        blocks, fallback = stubs(p, v, k, memo)
        out[f"stub-{v}"] = (score_blocks(p, blocks), fallback, blocks)
    if pool is not None:
        blocks = swap(p, pool)
        if blocks is not None:
            out["swap"] = (score_blocks(p, blocks), 0, blocks)
    out.update(d10_controls(p, k, memo))
    out.update(d11_controls(p, k, memo, priors))
    return out
