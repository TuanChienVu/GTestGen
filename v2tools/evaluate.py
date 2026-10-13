"""Scoring of one output (§2.2 BV and §2.2b CA of the design specification): the ER tree of insert.py (HEAD minus the
answer and minus future step functions, nothing masked) with the output inserted, the six BV conditions, and CA, the
decisive metric (fixed 24/09/2026): BV times agreement of project calls with the gold answer, step by step (calls.py).

BV = 1 when all six hold (when 1 fails, 2–6 are 0). Before them, an output that declares the same method key twice (in
one block or across blocks of one file) is a `collision`, like a clash with a method of the file.
1 syntax      every target file that received code parses (tree-sitter, no ERROR or MISSING node), and no inserted item
is raw (code that is not class members, or a type declared in the output);
2 binding     every CREATE step matches exactly one function after insertion, and that function is generated (a method,
a Java lambda step or a Ruby block of the output, never a member); an item registers only the step annotations before
the `{` that opens its body (`item_defs`);
3 no_ambiguity every step that is bound in the ER tree matches the same functions after insertion (identity = path +
method key; a replaced preexisting shell is the same function), in the record's suite and in every suite that a function
of a target file serves (catalog `suites`); a generated function counts in a suite only when its file serves that suite;
4 arity       the generated function that binds a CREATE step takes the N arguments its framework hands over (the
pattern's placeholders; a regex: for Behat the groups up to the highest-numbered one that matches the step, since
preg_match drops trailing groups that do not (design specification §2.2 at 25ba873), every capture group for SpecFlow,
the outermost ones for cucumber-jvm and cucumber-ruby) plus one for a table or doc string, compared as its framework
does (argcheck.GROUP_RULE, ARITY_RULE; fixed 25/09/2026), and no double-quoted value or number of the step is written
literally in the pattern;
5 grounded    every generated function that binds a CREATE step uses a project member of the ER tree (not one the output
defines, not a logger, not an output-added static import) or an in-repo class, in its body or in the body of a helper
its CA call set expands, read in the file that declares it (1b287d0; `grounds`);
6 resolved    no member reference of the generated code is unresolvable on the ER tree plus the output; the initializer
of a member item (field, property, constant) is read like a body, and counts toward open_resolution too.
Future methods: an output method with the key of a FUTURE method of the target class (one ④(a) does not show,
`future_methods`) replaces it (insert.insert(future=...)); and every call set, gold or output, is expanded through
calls.expand: a ("self", f) call to a method the output defines, or to a future method of the class or an in-repo
ancestor, becomes the project calls of f's body, to depth 3 (Score.calls; the cuts in Score.cuts)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from .argcheck import arity_ok as arity_fits, hardcoded, params_of, placeholders, step_args
from .bind import Library
from .calls import Expansion, agreement, calls, expand
from .compose import gold_answer
from .extract import extract
from .ertree import Removal, er_tree, parser, removal
from .insert import (LAMBDA, METHOD_NODES, class_body, class_methods, duplicate_keys, insert, is_constructor,
                     method_key, shell_keys, split_code)
from .outline import parse
from .records import hidden_for
from .state import t_view
from .stepdefs import extract as extract_defs
from .stepdefs import file_imports
from .symbols import (Refs, grounded, initializer_refs, open_paths, refs, ruby_grounded, ruby_unresolved, ruby_world,
                      scope, unresolved)

CONDITIONS = ("syntax", "binding", "no_ambiguity", "arity", "grounded", "resolved")


# ----- per-repo and per-record context -------------------------------------------------------------------------------

@dataclass
class Repo:
    name: str
    language: str
    styles: list[str]
    fns: dict  # fn_id -> catalog row
    ctx: object  # state.RepoContext
    ghost: frozenset
    suite_steps: dict  # suite -> list of (step text, HEAD def ids), distinct
    forbidden: dict = field(default_factory=dict)  # forbidden.json: repo -> {test, dev, aux} (controls use V(r))
    def_fn: dict = field(default_factory=dict)
    _ident: dict = field(default_factory=dict)

    def __post_init__(self):
        self.def_fn = {a["def_id"]: fid for fid, f in self.fns.items() for a in f["annotations"]}
        self._world = None

    def world(self):
        if self._world is None:
            self._world = ruby_world(self.ctx)
        return self._world

    def ident(self, fid: str) -> tuple:
        """Identity of a catalog function: (path, method key) for a method, (path, fn_id) for a lambda or a block."""
        if fid not in self._ident:
            f = self.fns[fid]
            items = split_code(self.language, f["code"])[1] if self.language != "ruby" else []
            m = next((it for it in items if it.kind == "method"), None)
            self._ident[fid] = (f["path"], method_key(self.language, m.name, m.arity)) if m else (f["path"], fid)
        return self._ident[fid]


def suite_index(labels: list[dict], pairs: dict) -> dict:
    """(repo, suite) -> distinct (step text, HEAD def ids) over every Phase 1 pair (labels.jsonl + pairs.jsonl)."""
    out = defaultdict(set)
    for lab in labels:
        k = (lab["repo"], lab["feature_path"], lab["scenario_line"], lab["suite"])
        for s, ids in zip(pairs[k]["steps"], lab["step_def_ids"]):
            if ids:
                out[(lab["repo"], lab["suite"])].add((s, tuple(ids)))
    return {k: sorted(v) for k, v in out.items()}


@dataclass
class Prepared:
    record: dict
    repo: Repo
    rem: Removal
    er_files: dict
    targets: list[str]
    shells: dict  # path -> replaceable method keys
    create_steps: list[tuple[str, bool]]  # (text, has table/doc string), distinct, in order
    reuse_steps: list[str]
    bound: list[tuple[str, str, frozenset]]  # steps bound in the ER tree: (suite, text, identities)
    served: dict = field(default_factory=dict)  # target file -> the suites its functions serve (the record's included)
    _gold: object = None
    masked: frozenset = frozenset()  # what 3A masks in the record's prompt (controls.Shown4.masked; train/dev only)
    _future: dict | None = None  # future_methods(), cached
    _shown: dict = field(default_factory=dict)  # shown_names(), cached per path
    _overloads: dict = field(default_factory=dict)  # future_overloads(), cached per path
    _view: dict = field(default_factory=dict)  # shown_members(), cached per path (D11: stub-v3names)
    _gold_out: object = None  # gold_output(), cached (D11 (e))

    def gold(self) -> "Score":
        """The CA reference (cached): the gold answer scored like any output (gold_output), whose binders and call sets
        anchor CA and call_precision, with D11 (e), the visible-shell delegation (added 26/09/2026), applied to its call
        sets (`delegated`)."""
        if self._gold is None:
            self._gold = delegated(self, self.gold_output())
        return self._gold

    def gold_output(self) -> "Score":
        """The gold answer scored exactly like an output (cached), without D11 (e), the visible-shell delegation: the
        gold control."""
        if self._gold_out is None:
            e = extract(gold_answer(self.record, self.repo.fns), self.targets)
            self._gold_out = score_blocks(self, e.by_path(), e.status)
        return self._gold_out


def prepare(record: dict, repo: Repo, labels_row: dict, masked: frozenset = frozenset()) -> Prepared:
    """`masked` = controls.Shown4.masked of the record (empty for test and aux_test): future_methods reads it."""
    rem = removal(record, repo.fns, repo.ghost)
    files = er_tree(rem, repo.ctx.text)
    targets = list(dict.fromkeys(c["path"] for c in record["create"]))
    shells = {p: shell_keys(repo.language, [repo.fns[c["fn_id"]]["code"] for c in record["create"] if c["preexisting"] and c["path"] == p])
              for p in targets}
    args = step_args(record)
    create, seen = [], set()
    for s, k, a in zip(record["steps"], record["step_kinds"], args):
        if k == "create" and s not in seen:
            seen.add(s)
            create.append((s, a))
    reuse = [s for s, k in zip(record["steps"], record["step_kinds"]) if k != "create"]
    own = record["suite"]
    served = {t: {own} | {su for f in repo.fns.values() if f["path"] == t for su in f.get("suites", [])} for t in targets}
    bound: dict = defaultdict(set)
    for su in sorted(set().union(*served.values())):
        for s, ids in repo.suite_steps.get(su, []):
            bound[(su, s)] |= {repo.ident(repo.def_fn[i]) for i in ids if i not in rem.defs and i in repo.def_fn}
    for s, ids in zip(record["steps"], labels_row["step_def_ids"]):  # the record's own steps, whatever their suite
        bound[(own, s)] |= {repo.ident(repo.def_fn[i]) for i in ids if i not in rem.defs and i in repo.def_fn}
    return Prepared(record, repo, rem, files, targets, shells, create, reuse,
                    sorted((su, s, frozenset(v)) for (su, s), v in bound.items() if v), served, masked=masked)


# ----- future methods -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Future:
    name: str
    origin: str  # the file that declares it: the target file or an in-repo ancestor
    text: str  # the method as the ER tree has it, dedented, docblock and annotations included


def shown_names(p: Prepared, path: str) -> set[str]:
    """The member names 3A shows in block ④(a) for `path`: state.t_view with the `gone` and `masked` of
    describe.hallucinated and controls.Shown4. Future is defined by time (fixed 25/09/2026): for a file NEW at t_vis
    (t_view None) the names are those t_view shows for each in-repo ancestor (RepoContext.ancestors, same `gone` and
    `masked`), less what class_view would not inherit: an ancestor's constructor, and a member private at HEAD, or in
    the declaration taken from before t_vis, in a file outside RepoContext.access(path). The new file's own methods, and
    what the ancestors got at or after t_vis, stay unshown, so future. Cached on `p`."""
    if path not in p._shown:
        p._shown[path] = {n for s in shown_members(p, path) for n in s.member.names}
    return p._shown[path]


def shown_members(p: Prepared, path: str) -> list:
    """The state.Shown members behind shown_names (the same rule, by time for a new file), in t_view's order;
    D11's stub-v3names calls their METHODS (controls.shown_methods). Cached on `p`."""
    if path not in p._view:
        r, ctx = p.record, p.repo.ctx
        gone = {c["fn_id"] for c in r["create"] if not c["preexisting"]}
        gone |= hidden_for(r["partition"], p.repo.forbidden.get(r["repo"], {}))
        view = t_view(ctx, path, r["t_vis"], gone, p.masked)
        if view is None:
            ok = ctx.access(path)
            view = [s for par, _ in ctx.ancestors(path) for s in t_view(ctx, par.path, r["t_vis"], gone, p.masked) or ()
                    if s.head.kind != "constructor" and (s.head.visibility != "private" or s.head_origin in ok)
                    and (s.member.visibility != "private" or s.origin in ok)]
        p._view[path] = view
    return p._view[path]


def future_methods(p: Prepared) -> dict[str, dict[object, Future]]:
    """Per target path, method key -> Future for every method (constructors included, as in insert's keys) of its class
    and of its in-repo ancestors in the ER tree (p.er_files, else HEAD) whose name `shown_names` does not hold; a
    private method is future only in a file of RepoContext.access(path), the target and the traits it uses, through
    traits: never an ancestor's, nor one of a trait that an ancestor uses, which is that ancestor's private (fixed
    25/09/2026); in C#, a method of a class, struct or record with no access modifier is private (`csharp_access`); the
    most derived FILE declaring a key wins, with the first of its declarations of that key here and every one of them in
    `future_overloads` (Java/C# overloads with one key). Ruby: none (no class). Cached on `p`: `p.masked` must be set
    before the first call."""
    if p._future is None:
        lang, ctx, p._future = p.repo.language, p.repo.ctx, {}
        for path in p.targets:
            out: dict = {}
            every: list = []
            if lang != "ruby":
                shown, ok = shown_names(p, path), ctx.access(path)
                for origin in [path] + [par.path for par, _ in ctx.ancestors(path)]:
                    er_text = p.er_files.get(origin, ctx.text(origin)) or ""
                    implicit = lang == "csharp" and origin not in ok and not csharp_interface(er_text)
                    here: dict = {}
                    for name, key, text in class_methods(lang, er_text, private=origin in ok):
                        if implicit and not csharp_access(text):  # private by default
                            continue
                        if name not in shown and key not in out:
                            here.setdefault(key, Future(name, origin, text))
                            every.append((key, Future(name, origin, text)))
                    out.update(here)
            p._future[path] = out
            p._overloads[path] = every
    return p._future


def future_overloads(p: Prepared, path: str) -> list[tuple[object, Future]]:
    """(key, Future) of EVERY future method of `path`'s class chain, in future_methods' order: a name resolves to all of
    them, Java/C# overloads that share a key (name, arity) included (fixed 25/09/2026: calls carry no
    arity, so the union of the overloads), where future_methods keeps one per key (the replacement set,
    future_methods.jsonl and gold-helpers)."""
    fut = future_methods(p).get(path, {})
    return p._overloads[path] if path in p._overloads else list(fut.items())


CSHARP_ACCESS = frozenset({"public", "protected", "internal", "private"})


def csharp_interface(text: str) -> bool:
    """The class-like declaration of the C# file `text` (insert.class_body's) is an interface: members are public."""
    body = class_body(parser("csharp").parse(text.encode()).root_node)
    return body is not None and body.parent is not None and body.parent.type == "interface_declaration"


def csharp_access(text: str) -> set[str]:
    """The access modifiers written on the C# method item `text`. None written, in a class, struct or record: the
    member is private (fixed 25/09/2026), so an ancestor's is never future."""
    src = ("class __X {\n" + text + "\n}\n").encode()
    body = class_body(parser("csharp").parse(src).root_node)
    m = next((c for c in body.named_children if c.type in METHOD_NODES), None) if body is not None else None
    if m is None:
        return set(CSHARP_ACCESS)
    return {src[c.start_byte:c.end_byte].decode() for c in m.children if c.type == "modifier"} & CSHARP_ACCESS


def future_keys(p: Prepared, path: str) -> frozenset:
    """Keys of the future methods of `path`'s own class, the ones an output method replaces (insert.insert(future=...)).
    An output method with the key of an ANCESTOR's future method is a normal override."""
    return frozenset(k for k, f in future_methods(p).get(path, {}).items() if f.origin == path)


def bodyless(lang: str, text: str) -> bool:
    """Design specification §2.2b (fixed 25/09/2026): the method item `text` parses as a method without a body (no
    block, no C# `=> …`): an abstract or interface method, never expanded, like a constructor."""
    prefix = "<?php\nclass __X {\n" if lang == "php" else "class __X {\n"
    body = class_body(parser(lang).parse((prefix + text + "\n}\n").encode()).root_node)
    m = next((c for c in body.named_children if c.type in METHOD_NODES), None) if body is not None else None
    return m is not None and m.child_by_field_name("body") is None


@dataclass(frozen=True)
class Body:
    """A helper body that the expansion reads, and the file whose outline and Scope read it: the target file for a
    method the output defines, the declaring file (target or ancestor) for a future method (1b287d0 (2))."""
    origin: str
    text: str


def expanded_futures(p: Prepared, s: "Score") -> dict[str, list[Future]]:
    """Per target path, the FUTURE methods whose ER-tree bodies the expansion of `s`'s step-binding functions read (to
    depth 3, in order, no repeats), one per method key: the first read (an output that declares a
    key twice is a collision; insert replaces the first method of the file with that key, and the other
    overloads stay in the file). `expanded_futures(p, p.gold())` = the helpers of the positive controls gold-helpers
    and gold-helpers-renamed."""
    out: dict[str, list[Future]] = {}
    keys: dict[str, set] = {}
    for ident in dict.fromkeys(s.binders.values()):
        by_body = {Body(f.origin, f.text): (k, f) for k, f in future_overloads(p, ident[0])}
        exp = s.expansions.get(ident)
        for _, body in exp.helpers if exp is not None else ():
            k, f = by_body.get(body, (None, None))
            if f is not None and k not in keys.setdefault(ident[0], set()):
                keys[ident[0]].add(k)
                out.setdefault(ident[0], []).append(f)
    return out


def grounds(ctx, lang: str, body: Body, exp: Expansion, own: set[str], fold, read) -> bool:
    """Condition 5 of one function: its own references, or those of any helper body its call-set expansion read, each
    read with the outline and Scope of the file that declares it (`read(origin)` -> (Scope, outline); 1b287d0), use a
    project member or an in-repo class; the name of a NEW method the output defines (`own`: not one V(r) shows, nor one
    with a future method's key) never counts on its own."""
    def part(text: str) -> Refs:
        r = refs(lang, text)
        keep = lambda names: frozenset(n for n in names if fold(n) not in own)
        return Refs(keep(r.members), keep(r.calls), r.classes)
    for b in [body] + [h for _, h in exp.helpers]:
        sc, o = read(b.origin)
        if sc is not None and grounded(ctx, o, part(b.text), sc):
            return True
    return False


# ----- the step definitions of one inserted item ----------------------------------------------------------------------

def body_open(lang: str, kind: str, text: str) -> int | None:
    """Index in `text` of the `{` that opens the body of a method item (tree-sitter: the method or constructor node's
    `body` field; a C# expression body `=> …` counts from its `=>`) or of a java8 lambda item (the lambda's `{` in its
    registration call, insert.LAMBDA); None when there is none (a method without a body)."""
    if kind == "lambda":
        m = LAMBDA.search(text)
        return m.end() - 1 if m is not None else None
    prefix = "<?php\nclass __X {\n" if lang == "php" else "class __X {\n"
    src = (prefix + text + "\n}\n").encode()
    body = class_body(parser(lang).parse(src).root_node)
    m = next((c for c in body.named_children if c.type in METHOD_NODES), None) if body is not None else None
    b = m.child_by_field_name("body") if m is not None else None
    return None if b is None else len(src[:b.start_byte].decode("utf-8", errors="replace")) - len(prefix)


def item_defs(lang: str, kind: str, path: str, text: str, styles: list[str], imports=None) -> list:
    """The step definitions one inserted method, java8 lambda or Ruby block item registers (fixed 25/09/2026):
    stepdefs.extract (Phase 1, sealed) reads the whole item text, so only the definitions whose annotation starts BEFORE
    the `{` that opens the item's body are kept (`body_open`): a PHP attribute or docblock line, a Java annotation or a
    C# attribute of the method itself; for a lambda item, its own registration call; for a Ruby block item, the step
    call on its first line (the one that opens the block). An annotation on a method of an anonymous or local class, on
    a local function, or on a registration nested in a lambda or block body is never registered by Behat, Cucumber or
    SpecFlow, so it never binds. A style whose extraction fails (ValueError: code cut off) gives nothing, as before.
    `imports` (stepdefs.file_imports of the file after insertion; design specification §2.2) resolves annotation names
    as the language does."""
    out = []
    opener = body_open(lang, kind, text) if kind != "block" else None
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    for style in styles:
        try:
            defs = extract_defs(path, text, style, imports)
        except ValueError:
            continue
        if kind == "block":
            out += [d for d in defs if d.line == 1]
        else:
            out += [d for d in defs if opener is None or text.find(d.code, starts[d.line - 1]) < opener]
    return out


# ----- scoring one output --------------------------------------------------------------------------------------------

@dataclass
class Score:
    extract_status: str
    insert_status: str  # "ok" | "collision" | "empty" (no usable block)
    conds: dict
    bv: int
    detail: dict
    binders: dict = field(default_factory=dict)  # CREATE step -> identity of the generated function that binds it
    items: dict = field(default_factory=dict)  # identity -> (path, kind, final text) of every inserted item
    after: dict = field(default_factory=dict)  # path -> file text after insertion (targets and ER tree changes)
    calls: dict = field(default_factory=dict)  # identity -> CA call set of every inserted function, helpers expanded
    expansions: dict = field(default_factory=dict)  # identity -> calls.Expansion
    cuts: dict = field(default_factory=lambda: {"depth": 0, "cycle": 0})  # depth and cycle cuts in the step binders


def score_blocks(p: Prepared, blocks: dict[str, str], extract_status: str = "ok") -> Score:
    r, repo, lang = p.record, p.repo, p.repo.language
    conds = dict.fromkeys(CONDITIONS, False)
    detail: dict = {}
    if not blocks:
        return Score(extract_status, "empty", conds, 0, detail)
    dup = [k for code in blocks.values() for k in duplicate_keys(lang, code)]
    if dup:
        detail["duplicate_keys"] = dup
        return Score(extract_status, "collision", conds, 0, detail)
    after, reports = dict(p.er_files), {}
    for path, code in blocks.items():
        base = after.get(path, repo.ctx.text(path))
        reports[path] = insert(lang, base, code, p.shells.get(path, set()), future_keys(p, path))
        after[path] = reports[path].text
    collisions = [n for x in reports.values() for n in x.collisions]
    detail["replaced"] = sum(len(x.replaced) for x in reports.values())
    detail["kept_annotations"] = sum(x.kept_annotations for x in reports.values())
    if collisions:
        detail["collisions"] = collisions
        return Score(extract_status, "collision", conds, 0, detail)
    raw = lang != "ruby" and any(kind == "raw" for x in reports.values() for kind, _, _ in x.items)
    conds["syntax"] = not raw and not any(parser(lang).parse(after[path].encode()).root_node.has_error for path in reports)
    if raw:
        detail["raw"] = True
    gen, items, replaced = [], {}, set()
    for path, rep in reports.items():
        for i, (kind, key, text) in enumerate(rep.items):
            ident = (path, key) if kind == "method" else (path, f"gen:{i}")
            items[ident] = (path, kind, text)
            # the output's step library: methods, Java lambda steps and Ruby blocks only, never a member or raw item:
            # a step hidden in a field's anonymous class or a PHP `new class` never binds; nor
            # does one nested in the body of a method, lambda or block item (item_defs)
            if kind in ("method", "lambda", "block"):  # names resolved via imports (design specification §2.2)
                gen += [(d, ident) for d in item_defs(lang, kind, path, text, repo.styles,
                                                      file_imports(lang, after[path]))]
        replaced |= {(path, key) for kind, key, _ in rep.items
                     if kind == "method" and (key in p.shells.get(path, ()) or key in future_keys(p, path))}
    # style -> step definitions of the INSERTED items (a rewritten shell's kept HEAD lines included); the scores row's
    # php_docblock_steps / php_attribute_steps read the output's own blocks instead (describe.own_step_styles)
    detail["step_styles"] = dict(sorted(Counter(d.style for d, _ in gen).items()))
    if not conds["syntax"]:  # a file that does not parse is not judged further (conditions 2-6 = 0)
        return Score(extract_status, "ok", conds, 0, detail, {}, items, after, {})
    lib = Library([d for d, _ in gen])
    ident_of = {id(d): ident for d, ident in gen}
    gen_ids = set(items)

    def matches(step):
        return {ident_of[id(d)] for d in lib.matches(step)}
    binders, arity_ok = {}, True
    own = p.record["suite"]
    b0_of = {s: ids for su, s, ids in p.bound if su == own}
    active = lambda g, su: su == own or su in p.served.get(g[0], ())  # a new function counts where its file is loaded
    for s, has_arg in p.create_steps:
        b1 = {x for x in b0_of.get(s, frozenset()) if x not in replaced} | matches(s)
        if len(b1) == 1 and b1 <= gen_ids:
            binders[s] = next(iter(b1))
        else:
            detail.setdefault("unbound_or_ambiguous", []).append(s)
    conds["binding"] = len(binders) == len(p.create_steps)
    changed = [s for su, s, b0 in p.bound
               if ({x for x in b0 if x not in replaced} | {g for g in matches(s) if active(g, su)}) != b0]
    conds["no_ambiguity"] = not changed
    if changed:
        detail["suite_steps_changed"] = changed[:5]
        detail["duplicate_reuse"] = any(s in p.reuse_steps for s in changed)
    for s, has_arg in p.create_steps:
        f = binders.get(s)
        if f is None:
            arity_ok = False
            continue
        path, kind, text = items[f]
        it_arity = next((it.arity for it in split_code(lang, text)[1] if it.kind == kind), None) if kind == "lambda" else None
        par = params_of(lang, kind, text, it_arity)
        ok = False
        for d in lib.matches(s):
            if ident_of[id(d)] != f:
                continue
            want = placeholders(d.kind, d.pattern, lang, s) + has_arg  # N read on the step itself (PHP)
            ok |= arity_fits(lang, want, *par, table=has_arg, takes_table=par.takes_table) \
                and not hardcoded(d.kind, d.pattern, s)
        arity_ok &= ok
    conds["arity"] = arity_ok and conds["binding"]
    ok5, ok6, bad, opened = bool(binders), True, set(), set()
    scopes, call_sets, expansions, memo, helpers = {}, {}, {}, {}, {}
    fold = str.casefold if lang == "php" else (lambda n: n)

    def resolver(path: str):
        """(resolve, folded names of the output's NEW methods) for the functions of `path`: ("self", f) with f visible
        in V(r) stays (a project member, even when the output rewrites it); else it expands to the union of the output's
        methods named f and the FUTURE methods named f that the ER tree still has after insertion (masking is by KEY,
        not by name; fixed 25/09/2026): every overload of a key (future_overloads), except, for a key the output
        declares, an ancestor's declarations (the target now declares the key: the most derived wins) and the first
        declaration in the target file (the one insert replaced; the file's other declarations of that key stay in it);
        else it stays. A constructor never expands, nor does a method without a body (`bodyless`), nor, from the ER
        tree, a declaration in the target file with the key of one of the record's preexisting shells (p.shells, fixed
        25/09/2026: the ER tree keeps its HEAD body, the answer; an output that redefines it expands its own version).
        The second value is what is kept out of condition 5: neither a visible name nor that of an output method with a
        future method's key, which grounds like the member it replaces or overrides; a bodyless output method's name is
        kept out too."""
        visible = {fold(n) for n in shown_names(p, path)} if lang != "ruby" else set()
        mine, er, empty, keys, replaced = defaultdict(list), defaultdict(list), {}, set(), set()
        for kind, key, text in reports[path].items:
            if kind == "method":
                keys.add(key)
                if not is_constructor(lang, text):
                    mine[fold(key if lang == "php" else key[0])].append(Body(path, text))
        for key, f in future_overloads(p, path):
            if key in keys and (f.origin != path or key not in replaced):
                replaced.add(key)
                continue
            if f.origin == path and key in p.shells.get(path, ()):  # a preexisting shell: its ER body is the answer
                continue
            if not is_constructor(lang, f.text):
                er[fold(f.name)].append(Body(f.origin, f.text))
        future_named = {fold(key if lang == "php" else key[0]) for kind, key, _ in reports[path].items
                        if kind == "method" and key in future_methods(p).get(path, {})}

        def resolve(f: str):
            if fold(f) in visible:
                return None
            got = mine.get(fold(f), []) + er.get(fold(f), [])
            for b in got:
                if b not in empty:
                    empty[b] = bodyless(lang, b.text)
            return [b for b in got if not empty[b]] or None
        return resolve, set(mine) - visible - future_named

    def read(origin: str):
        """(Scope, outline) of a file after insertion: a body is read with the file that declares it (1b287d0 (2))."""
        if origin not in scopes:
            er = p.er_files.get(origin, repo.ctx.text(origin))
            now = after.get(origin, er)
            scopes[origin] = (scope(repo.ctx, origin, er, now, p.er_files), parse(lang, now))
        return scopes[origin]

    def body_calls(b: Body) -> set:  # a body read like a step body: live(), the loggers and the Scope of its own file
        if b not in memo:
            sc, o = read(b.origin)
            memo[b] = set() if sc is None else calls(lang, b.text, ctx=repo.ctx, o_after=o, members=sc.available,
                                                         loggers=sc.loggers)
        return memo[b]
    for ident, (path, kind, text) in items.items():
        rf = initializer_refs(lang, text) if kind == "member" else None  # a member's initializer
        if kind == "member" and rf is None:
            continue
        if lang == "ruby":
            rf, world = refs(lang, text), repo.world()
            u = ruby_unresolved(rf, world)
            bad |= u
            ok6 &= not u
            if ident in binders.values():
                ok5 &= ruby_grounded(rf, world)
            opened |= set(rf.calls) - world.methods  # receiverless calls are never judged
            call_sets[ident] = calls(lang, text, world=world.methods)
            continue
        sc, o_after = read(path)
        rf = refs(lang, text) if rf is None else rf
        if sc is None:
            ok6 = False
            if kind != "member":
                call_sets[ident] = set()
            continue
        u = unresolved(rf, sc)
        bad |= u
        ok6 &= not u
        opened |= open_paths(repo.ctx, o_after, rf, sc)
        if kind == "member":  # judged by condition 6 only: condition 5 and the CA call sets read the functions
            continue
        if path not in helpers:
            helpers[path] = resolver(path)
        resolve, own = helpers[path]
        root = Body(path, text) if kind == "method" else None  # its own body is on the expansion path
        exp = expand(body_calls(Body(path, text)), resolve, body_calls, root)
        expansions[ident], call_sets[ident] = exp, set(exp.calls)
        if ident in binders.values():
            ok5 &= grounds(repo.ctx, lang, Body(path, text), exp, own, fold, read)
    conds["grounded"], conds["resolved"] = ok5, ok6
    detail["open_resolution"] = ok6 and bool(opened)  # condition 6 passed through an open path
    if opened:
        detail["open_paths"] = sorted(opened)[:10]
    if bad:
        detail["unresolved"] = sorted(bad)[:10]
    bound_exp = [expansions[f] for f in set(binders.values()) if f in expansions]  # the call sets CA reads (their cuts)
    cuts = {"depth": sum(e.depth_cuts for e in bound_exp), "cycle": sum(e.cycle_cuts for e in bound_exp)}
    return Score(extract_status, "ok", conds, int(all(conds.values())), detail, binders, items, after, call_sets,
                 expansions, cuts)


def run_name(run: dict) -> str:
    """The name of a model run in overlays and ER results: <model>.<seed>.<condition>, plus .<variant> unless "main"."""
    v = run.get("variant") or "main"
    return f"{run['model']}.{run['seed']}.{run['condition']}" + ("" if v == "main" else f".{v}")


def delegated(p: Prepared, gold: Score) -> Score:
    """D11 (e), the visible-shell delegation (added 26/09/2026; design specification §2.2b, "Uỷ quyền cho vỏ preexisting
    nhìn thấy"): the gold Score `gold` whose call set of each binder (a function that binds a CREATE step) that is a
    preexisting shell of the record (its method key in p.shells of its file) and whose name ④(a) shows for that file
    (shown_names; PHP compares in any case) also holds ("self", the shell's name as the gold writes it): the gold only
    adds an annotation to a function the model is shown, so a new step that calls it does what the gold does, and a
    visible name keeps that call as ("self", f). An EMPTY call set gains nothing (the design specification is silent;
    D11 (e) then never adds a CA condition nor changes ca_defined). Only the gold's call sets change (Prepared.gold, the
    CA and call_precision reference); an output, and the gold answer scored as one (Prepared.gold_output, the gold
    control), are scored as before. Ruby: none (no ④(a) method list, no shell method)."""
    lang = p.repo.language
    if lang == "ruby":
        return gold
    fold = str.casefold if lang == "php" else (lambda n: n)
    calls = dict(gold.calls)
    for ident in dict.fromkeys(gold.binders.values()):
        path, key = ident
        if key not in p.shells.get(path, ()) or not calls.get(ident) or ident not in gold.items:
            continue
        _, kind, text = gold.items[ident]
        name = next((it.name for it in split_code(lang, text)[1] if it.kind == kind == "method"), None)
        if name is not None and fold(name) in {fold(n) for n in shown_names(p, path)}:
            calls[ident] = set(calls[ident]) | {("self", name)}
    return replace(gold, calls=calls) if calls != gold.calls else gold


def ca(p: Prepared, s: Score) -> tuple[bool, int]:
    """(ca_defined, ca) of a scored output against the record's gold answer (§2.2b of the design specification); PHP
    method names compare in any case."""
    g = p.gold()
    fold = str.casefold if p.repo.language == "php" else None
    defined, agree = agreement(g.binders, g.calls, s.binders, s.calls, fold)
    return defined, int(bool(defined and s.bv and agree))


def score_output(p: Prepared, text: str) -> Score:
    e = extract(text, p.targets)
    return score_blocks(p, e.by_path(), e.status) if e.status != "none" else Score("none", "empty", dict.fromkeys(CONDITIONS, False), 0, {})
