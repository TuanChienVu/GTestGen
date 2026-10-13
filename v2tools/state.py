"""The visible state V(r): what a model may see of the repo for record r, at T = r["t_vis"].
Rule 1 drops answer functions (preexisting items keep their body minus the record's CREATE annotations and every
annotation at or after T); rule 2 drops step functions whose every annotation is at or after T; rule 3 masks class
members whose NAME no file of the class chain declared before T (RepoContext.declared_before: the pre-T version of
each chain file, the same rebase-proof choice as rule 5, so a member that moved between the class and its ancestors
stays visible) and constructor parameters first declared at or after T, and drops a ④(b) function whose body uses
such a member; rule 4 adds inherited members from in-repo parents (inherit.Resolver): the non-private ones, plus the
private ones of traits the class uses directly or through other traits. Rule 5 (24/09/2026): what ④ shows is the text
each file had BEFORE T, never HEAD text written with or after the answer: ④(a) renders every kept member from the
chain's pre-T versions (t_view), and a ④(b) or ③ function is shown from HEAD when its HEAD code is unchanged since T
(RepoContext.unchanged), else in its pre-T form (pre_t_code), and left out only when that form cannot be located."""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from . import gitio
from .inherit import Parent, Resolver
from .outline import Member, Outline, parse, parse_snippet
from .stepdefs import _java_kind, _php_kind
from .symtime import first_times

MAX_DEPTH = 5  # deeper chains raise instead of being cut silently
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
_KEYWORD = r"(?:Given|When|Then|And|But)"
_PHP_ATTRIBUTE = re.compile(rf"#\[{_KEYWORD}\(\s*(['\"])(.*?)(?<!\\)\1\s*\)\]", re.I)
_PHP_DOCBLOCK = re.compile(rf"^\s*(?:/\*\*|\*).*?@{_KEYWORD}\s+(.+?)\s*(?:\*/|$)", re.I)
_JAVA_ANNOTATION = re.compile(rf'@{_KEYWORD}\(\s*"((?:[^"\\]|\\.)*)"', re.I)
_CSHARP_ATTRIBUTE = re.compile(rf'\[{_KEYWORD}\(\s*(?:@"((?:[^"]|"")*)"|"((?:[^"\\]|\\.)*)")', re.I)


def render_visible(fn: dict, t: int, create_def_ids: frozenset[str]) -> str:
    """Catalog function code without the annotation lines of the record's CREATE patterns (any time) and without every
    annotation that did not exist strictly before t (rule 1)."""
    drop = {a["line"] for a in fn["annotations"]
            if a["def_id"] in create_def_ids or not (a["time"] is not None and a["time"] < t)}
    return "\n".join(l for i, l in enumerate(fn["code"].splitlines(), start=fn["first_line"]) if i not in drop)


def annotation_pattern(language: str, line: str) -> str | None:
    """The step pattern that one annotation line declares, as the catalog stores it (stepdefs.extract): PHP
    `#[Given('…')]` or a docblock line `* @Given …`, a /…/ regex without its delimiters; Java `@Given("…")`, escapes
    undone, a /…/ regex without its delimiters; C# `[Given(@"…")]` or `[Given("…")]`. Keywords Given, When, Then, And,
    But in any case. None for any other line, for Ruby, and for a cucumber-java8 lambda step (a call, not an
    annotation). Rule 1 reads the annotations of a pre-T text with it, where catalog line numbers do not apply."""
    if language == "php":
        m = _PHP_ATTRIBUTE.search(line)
        if m:
            return _php_kind(m.group(2).replace("\\'", "'"))[1]
        m = _PHP_DOCBLOCK.match(line)
        return _php_kind(m.group(1).strip())[1] if m else None
    if language == "java":
        m = _JAVA_ANNOTATION.search(line)
        return _java_kind(m.group(1))[1] if m else None
    m = _CSHARP_ATTRIBUTE.search(line) if language == "csharp" else None
    if m is None:
        return None
    if m.group(1) is not None:
        return m.group(1).replace('""', '"')
    return re.sub(r"\\(.)", lambda e: e.group(1), m.group(2))


def body_of(code: str) -> str:
    """Text after the first "{" (the function body), or the whole code when there is none (Ruby)."""
    j = code.find("{")
    return code[j + 1:] if j >= 0 else code


_DECL = re.compile(r"\b(?:[A-Z][\w.]*(?:<[^;(){}]*?>)?(?:\[\])*|int|long|boolean|double|float|char|byte|short|var)"
                   r"\s+([a-z_]\w*)\s*(?==|:|;|,|\))")
_LAMBDA = re.compile(r"\(([^()]*)\)\s*->")


def member_refs(language: str, code: str) -> set[str]:
    """Names a function's body uses as members of its class. PHP: `$this->x`, `self::x`, `static::x`, `parent::x`.
    Java and C#: `this.x`, bare calls `x(`, bare capitalised names (constants, C# properties such as `Patients`), and
    bare identifiers that are not declared in the
    function itself (parameters, lambda parameters, locals). String literals are blanked first, so words inside them
    (and a "{" inside an annotation such as `@Then("… {int}")`) are not read as code. Callers intersect the result with
    real member names."""
    if language == "ruby":
        return set()
    if language == "php":
        body = body_of(code)
        return set(re.findall(r"\$this->(\w+)", body)) | set(re.findall(r"\b(?:self|static|parent)::\$?(\w+)", body))
    code = _STRING.sub('""', code)
    body = body_of(code)
    local = set(_DECL.findall(code))
    for params in _LAMBDA.findall(code):
        local |= {p.split()[-1] for p in params.split(",") if p.strip()}
    refs = set(re.findall(r"\bthis\.(\w+)", body))
    refs |= set(re.findall(r"(?<![.\w])([A-Za-z_]\w*)\s*\(", body))
    refs |= set(re.findall(r"(?<![.\w])([A-Z]\w*)\b", body))  # constants and PascalCase members (C# properties)
    refs |= set(re.findall(r"(?<![.\w\"'])([a-z_]\w*)\b(?!\s*\()", body)) - local
    return refs


@dataclass
class ClassView:
    path: str
    outline: Outline
    members: list[tuple[Member, str, str]] = field(default_factory=list)  # (member, origin path, origin class name)


class RepoContext:
    """Per-repo caches over a full bare clone at the pinned commit. `symtimes` may be preloaded from
    out/phase3/symtimes.jsonl so that compose never re-reads git history."""

    def __init__(self, repo: str, language: str, git_dir: Path, commit: str, fns: dict[str, dict],
                 symtimes: dict[str, dict] | None = None, strict: bool = False):
        self.repo, self.language, self.git_dir, self.commit, self.fns = repo, language, git_dir, commit, fns
        self.tree = gitio.ls_tree(git_dir, commit)
        self._paths = set(self.tree)
        self.resolver = Resolver(language, self.tree, gitio.show(git_dir, commit, "composer.json")
                                 if language == "php" and "composer.json" in self._paths else None)
        self._outlines: dict[str, Outline | None] = {}
        self._text: dict[str, str | None] = {}
        self._ancestors: dict[str, list] = {}
        self._chain: dict[str, dict[str, int]] = {}
        self._access: dict[str, set[str]] = {}
        self._old: dict[tuple[str, str], str] = {}
        self._old_outline: dict[tuple, Outline | None] = {}
        self.pre_t: dict[tuple[str, int | None], str] = {}  # pre_t_code by (fn_id, pre-t version index)
        self.symtimes = dict(symtimes or {})
        self.strict = strict  # True: symtimes is a sealed input and a missing path is an error, not a git read
        self.step_at = {(f["path"], int(fid.rsplit(":", 1)[1])): fid for fid, f in fns.items()}

    def text(self, path: str) -> str | None:
        """File content at HEAD; None only for a path absent from the HEAD tree (a read failure raises)."""
        if path not in self._text:
            self._text[path] = gitio.show(self.git_dir, self.commit, path) if path in self._paths else None
        return self._text[path]

    def outline(self, path: str) -> Outline | None:
        if path not in self._outlines:
            t = self.text(path)
            self._outlines[path] = None if t is None or self.language == "ruby" else parse(self.language, t)
        return self._outlines[path]

    def times(self, path: str) -> dict:
        if path not in self.symtimes:
            if self.strict:
                raise KeyError(f"{self.repo}: {path} is not in the sealed symtimes")
            o = self.outline(path)
            self.symtimes[path] = first_times(self.git_dir, self.commit, path, self.language,
                                              [f for f in self.fns.values() if f["path"] == path], o.name if o else None)
        return self.symtimes[path]

    def version_index(self, path: str, t: int) -> int | None:
        """Index in times(path)["versions"] (newest first) of the file as it was before t (rule 5): the version just
        older, in history order, than the oldest commit authored at or after t that touched the file, so that a commit
        with an older author time placed after such a commit (a rebase) is not taken as "before t"; the newest version
        when no commit is authored at or after t; None when the file did not exist before t."""
        vs = self.times(path)["versions"]
        later = [i for i, (_, at, _) in enumerate(vs) if at >= t]
        i = later[-1] + 1 if later else 0
        return i if i < len(vs) else None

    def text_before(self, path: str, t: int) -> str | None:
        i = self.version_index(path, t)
        if i is None:
            return None
        sha, _, p = self.times(path)["versions"][i]
        if (sha, p) not in self._old:
            self._old[(sha, p)] = gitio.show(self.git_dir, sha, p)
        return self._old[(sha, p)]

    def outline_before(self, path: str, t: int) -> Outline | None:
        """Outline of the file as it was before t (rule 5); None for Ruby, a file absent before t, or no class."""
        i = self.version_index(path, t) if self.language != "ruby" else None
        if i is None:
            return None
        key = tuple(self.times(path)["versions"][i][::2])
        if key not in self._old_outline:
            o = self.outline(path)
            self._old_outline[key] = parse(self.language, self.text_before(path, t), o.name if o else None)
        return self._old_outline[key]

    def unchanged(self, fid: str, t: int) -> bool:
        """Rule 5 for a catalog function: its HEAD code (annotation and comment lines aside) is present in its file as
        it was before t, i.e. nothing that ④ or ③ would show from HEAD was written at or after t."""
        path = self.fns[fid]["path"]
        i = self.version_index(path, t)
        bits = self.times(path)["present"].get(fid)
        return i is not None and bits is not None and bits[i] == "1"

    def parents(self, path: str) -> list[Parent]:
        o = self.outline(path)
        return [] if o is None else [self.resolver.resolve(o, rel, w) for rel, w in o.parents]

    def ancestors(self, path: str) -> list[tuple[Parent, int]]:
        """Resolved in-repo ancestors, breadth first, with depth; each file once. A chain deeper than MAX_DEPTH raises."""
        if path in self._ancestors:
            return self._ancestors[path]
        out, seen, frontier = [], {path}, [(path, 0)]
        while frontier:
            p, d = frontier.pop(0)
            for par in self.parents(p):
                if par.status == "resolved" and par.path not in seen:
                    if d >= MAX_DEPTH:
                        raise RuntimeError(f"{self.repo}: class chain of {path} is deeper than {MAX_DEPTH}")
                    seen.add(par.path)
                    out.append((par, d + 1))
                    frontier.append((par.path, d + 1))
        self._ancestors[path] = out
        return out

    def access(self, path: str) -> set[str]:
        """Files whose private members the class in `path` can use: itself and the traits it uses, directly or through
        other traits (a parent's traits are the parent's own members)."""
        if path not in self._access:
            out, frontier = {path}, [path]
            while frontier:
                for par in self.parents(frontier.pop()):
                    if par.relation == "trait" and par.status == "resolved" and par.path not in out:
                        out.add(par.path)
                        frontier.append(par.path)
            self._access[path] = out
        return self._access[path]

    def chain(self, path: str) -> list[str]:
        return [path] + [par.path for par, _ in self.ancestors(path)]

    def chain_before(self, path: str, t: int) -> list[str]:
        """The files of chain(path) that the class reached before t: breadth first over the parents declared in the
        pre-t versions (resolved with the HEAD resolver), kept inside chain(path). A parent added at or after t is not
        part of V(r), even if one of its members is older (as PrestaShop's checkFixtureExists)."""
        head, out, frontier = set(self.chain(path)), [path], [path]
        while frontier:
            o = self.outline_before(frontier.pop(0), t)
            for rel, w in (o.parents if o is not None else ()):
                par = self.resolver.resolve(o, rel, w)
                if par.status == "resolved" and par.path in head and par.path not in out:
                    out.append(par.path)
                    frontier.append(par.path)
        return out

    def chain_times(self, path: str) -> dict[str, int]:
        """First declaration time of each member name over the class and its in-repo ancestors (minimum over the
        files; reports only, rule 3 reads declared_before). A name that an ancestor outside access(path) declares
        private at HEAD does not count for that ancestor."""
        if path not in self._chain:
            names: dict[str, int] = {}
            ok = self.access(path)
            for p in self.chain(path):
                o = self.outline(p)
                private = set() if p in ok or o is None else \
                    {n for m in o.members if m.visibility == "private" for n in m.names}
                for n, at in self.times(p)["names"].items():
                    if n not in private and (n not in names or at < names[n]):
                        names[n] = at
            self._chain[path] = names
        return self._chain[path]

    def declared_before(self, path: str, t: int) -> set[str]:
        """Member names that the class chain of `path` declared before t (rule 3): the names of each chain file's pre-t
        version (version_index). A member that moved between the class and its ancestors is still declared somewhere; a
        name that only a commit with an old author date, placed after a commit authored at or after t, introduced is not
        (a rebase: PrestaShop createApiClient). Private names of a chain file outside access(path) do not count: the
        class could never use them, so they must not make a new member of the class look old."""
        out, ok = set(), self.access(path)
        for p in self.chain(path):
            i = self.version_index(p, t)
            if i is not None:
                tm = self.times(p)
                out.update(tm["vnames"][i] if p in ok else set(tm["vnames"][i]) - set(tm["vprivate"][i]))
        return out

    def future(self, path: str, t: int, masked: frozenset = frozenset()) -> set[str]:
        """Member names of the class in `path` (own or inherited, every name its chain ever declared) that the chain did
        not declare before t, plus the names in `masked` declared by a file of that chain (test-answer infrastructure
        hidden from train and dev: `masked` holds (declaring path, name) pairs)."""
        chain = set(self.chain(path))
        return (set(self.chain_times(path)) - self.declared_before(path, t)) | {n for q, n in masked if q in chain}

    def declarer(self, path: str, name: str) -> str | None:
        """The file of the class chain of `path` that declares member `name` at HEAD (own class first)."""
        for p in self.chain(path):
            o = self.outline(p)
            if o is not None and any(name in m.names for m in o.members):
                return p
        return None

    def is_new_file(self, path: str, t: int) -> bool:
        """The file does not exist at HEAD, or did not exist before t (no pre-t version: the choice of rule 5)."""
        return self.text(path) is None or self.version_index(path, t) is None


def _hidden_step(ctx: RepoContext, path: str, m: Member, t: int, gone: set[str]) -> bool:
    """A method that is a catalog step function removed from V(r): an answer (rule 1), hidden for the partition, or a
    future step function (rule 2: every annotation at or after t)."""
    fid = ctx.step_at.get((path, m.body_line)) if m.body_line else None
    if fid is None:
        return False
    if fid in gone:
        return True
    return not any(a["time"] is not None and a["time"] < t for a in ctx.fns[fid]["annotations"])


def class_view(ctx: RepoContext, path: str, t: int, gone: set[str], masked: frozenset = frozenset(),
               removed: list | None = None) -> ClassView | None:
    """Members of the class in `path` that V(r) keeps, own first, then inherited (non-private, plus the private ones of
    RepoContext.access), filtered by rules 1–2 and by rule 3 over the whole chain (RepoContext.future). A name already
    kept from a NEARER class (an override) is not repeated; overloads within one class all stay. `removed`, when given,
    receives (member, origin, "rule3" | "rule12") for each member the rules take out, for the STATE_REPORT rates."""
    o = ctx.outline(path)
    if o is None:
        return None
    view = ClassView(path, o)
    fut, ok, seen, out = ctx.future(path, t, masked), ctx.access(path), set(), set()
    for origin in ctx.chain(path):
        oo = ctx.outline(origin)
        if oo is None:
            continue
        here = set()
        for m in oo.members:
            if origin != path and m.kind == "constructor":
                continue
            if origin not in ok and m.visibility == "private":
                continue
            if any(n in seen or n in out for n in m.names):
                continue
            why = "rule3" if any(n in fut for n in m.names) else \
                "rule12" if m.kind == "method" and _hidden_step(ctx, origin, m, t, gone) else None
            if why:
                if removed is not None:
                    removed.append((m, origin, why))
                if why == "rule3":  # the same name in a parent is just as new (chain minimum): count it once
                    out.update(m.names)
                continue
            view.members.append((m, origin, oo.name))
            here.update(m.names)
        seen |= here
    return view


_FIELDS = ("property", "promoted", "const")


def _group(kind: str) -> str:
    return "field" if kind in _FIELDS else kind  # a property may become a promoted parameter, and back


def _plain(name: str) -> str:
    return name.lstrip("_").casefold()


def param_names(language: str, signature: str) -> tuple[str, ...]:
    """Parameter names of a method signature, in order (types and defaults ignored): what a call has to match."""
    inner = signature[signature.find("(") + 1:signature.rfind(")")] if "(" in signature else ""
    inner = re.sub(r"'(?:\\.|[^'\\])*'", "''", _STRING.sub('""', inner))  # a comma inside a default string
    if language == "php":
        return tuple(re.findall(r"\$(\w+)", inner))
    prev = None
    while prev != inner:
        prev, inner = inner, re.sub(r"<[^<>]*>", "", inner)
    return tuple(w[-1] for part in inner.split(",") if (w := re.findall(r"\w+", part.split("=")[0])))


@dataclass(frozen=True)
class Shown:
    member: Member  # the declaration as it was before T (rule 5)
    origin: str  # the file of the chain that declared it before T
    origin_class: str
    head: Member  # the same member at HEAD
    head_origin: str


def _annotated(language: str, text: str, x: Member) -> bool:
    """The method declaration `x` of `text` carries a step annotation (in its attribute, annotation or docblock lines)."""
    if x.kind != "method" or x.span == (0, 0):
        return False
    return any(annotation_pattern(language, line) is not None for line in text.splitlines()[x.span[0] - 1:x.line - 1])


def t_view(ctx: RepoContext, path: str, t: int, gone: set[str], masked: frozenset = frozenset(),
           removed: list | None = None) -> list[Shown] | None:
    """④(a) content under rule 5: each member that the HEAD view keeps (class_view, rules 1–4), in the form it had
    before t, looked up in the class chain as it was before t (chain_before): its HEAD file first, then the other files
    of that chain (a member that moved). A method takes the pre-T declaration with its HEAD parameter names, else the
    one at the same position among its overloads; a member that is not a step at HEAD never takes a pre-T declaration
    that carries a step annotation (with a hidden test-answer step removed, a helper overload
    was matched to that step's pre-T form, and train prompts showed the test answer's signature). Own members first,
    then inherited in chain order, each in declaration order. A kept member that no file of the chain declared before t is left out (`removed` gets
    (member, origin, "rule5")), and so is a field declaration that also declares a name the view does not keep. None
    for a new file, or when there is no class at HEAD or before t."""
    if ctx.is_new_file(path, t):
        return None
    view = class_view(ctx, path, t, gone, masked, removed)
    if view is None or ctx.outline_before(path, t) is None:
        return None
    keep = {n for m, _, _ in view.members for n in m.names}
    chain = ctx.chain_before(path, t)
    count: dict[tuple, int] = {}
    taken: set[tuple] = set()  # pre-T declarations already used: (file, line, kind, names)
    out = []
    for m, origin, _ in view.members:
        key = (origin, m.names[0], _group(m.kind))
        k = count[key] = count.get(key, -1) + 1
        m_step = m.kind == "method" and (bool(m.body_line and ctx.step_at.get((origin, m.body_line)))
                                         or _annotated(ctx.language, ctx.text(origin), m))
        want = param_names(ctx.language, m.text) if m.kind in ("method", "constructor") else None
        hit = None
        for p in ([origin] if origin in chain else []) + [q for q in chain if q != origin]:
            o = ctx.outline_before(p, t)
            same = [] if o is None else [x for x in o.members if m.names[0] in x.names and _group(x.kind) == _group(m.kind)
                                         and (x.visibility != "private" or p in ctx.access(path))]
            allowed = [x for x in same if m_step or not _annotated(ctx.language, ctx.text_before(p, t), x)]
            pick = next((x for x in allowed if want is not None and (p, x.line, x.kind, x.names) not in taken
                         and param_names(ctx.language, x.text) == want), None)
            i = k if p == origin else 0  # position among ALL overloads of the name, then the step guard
            if pick is None and i < len(same) and same[i] in allowed:
                pick = same[i]
            if pick is not None:
                hit = (pick, p, o.name)
                break
        if hit is None:
            if removed is not None:
                removed.append((m, origin, "rule5"))
            continue
        x, p, cls_name = hit
        if (p, x.line, x.kind, x.names) in taken:
            continue
        if any(n not in keep for n in x.names):  # a field declaring a name the view does not keep
            if removed is not None:
                removed.append((m, origin, "rule3"))
            continue
        taken.add((p, x.line, x.kind, x.names))  # not the line alone: a one-line constructor shares it with its params
        out.append(Shown(x, p, cls_name, m, origin))
    order = {p: i for i, p in enumerate(chain)}
    out.sort(key=lambda s: (order.get(s.origin, len(order)), s.member.line))
    return out


def _render_ctor(o: Outline, m: Member, masked_names: set[str]) -> str:
    """The constructor signature as it was before T, with every parameter it had then (filtering
    them by the HEAD members dropped parameters that existed at T and left a post-answer shape), except a parameter
    named after a masked infrastructure member (C# `_field` ← `field`)."""
    params = o.params.get(m.line, ())
    masked_plain = {_plain(n) for n in masked_names}
    kept = [p for p, promoted, name in params if not ((promoted or name) and _plain(promoted or name) in masked_plain)]
    if len(kept) == len(params):
        return m.text
    head, _, _ = m.text.partition("(")
    tail = m.text[m.text.rfind(")"):]
    if "\n" in m.text.split("(", 1)[1].rsplit(")", 1)[0]:
        inner = "".join(f"\n{m.indent}    {p}," for p in kept)
        return f"{head}({inner}\n{m.indent}{tail}"
    return f"{head}({', '.join(kept)}{tail}"


@dataclass(frozen=True)
class SkeletonItem:
    text: str  # rendered member line(s), already indented
    member: Member  # as declared before T
    origin: str  # path of the file that declares the member (before T)
    origin_class: str
    fn_id: str | None  # catalog step function id when the member is a step method


def skeleton_parts(ctx: RepoContext, path: str, t: int, gone: set[str],
                   masked: frozenset = frozenset()) -> tuple[list[str], list[SkeletonItem]] | None:
    """④(a) outline of one target file as (head lines, member items), all as they were before t (rule 5): namespace,
    class header, then the members of t_view() as signatures with elided bodies, own first, inherited after. None for a
    new file, a Ruby file or no class."""
    shown = t_view(ctx, path, t, gone, masked)
    if shown is None:
        return None
    o = ctx.outline_before(path, t)
    masked_names = {n for q, n in masked if q in set(ctx.chain(path))}
    head = []
    if o.namespace:  # C# as a file-scoped namespace, so that the outline's braces balance
        head += [("package " if o.language == "java" else "namespace ") + o.namespace + ";", ""]
    head.append(o.header)
    items = []
    for s in shown:
        m = s.member
        if m.kind == "promoted" and s.origin == path:  # shown inside the constructor signature
            continue
        indent = m.indent
        if m.kind == "promoted":  # a parent's promoted property: shown as a property declaration
            text = m.text.rstrip().rstrip(",") + ";"
            indent = next((x.indent for x in ctx.outline_before(s.origin, t).members if x.kind == "constructor"), m.indent)
        elif m.kind == "constructor" and s.origin == path:
            text = _render_ctor(o, m, masked_names)
        else:
            text = m.text
        if m.kind in ("method", "constructor") and text.rstrip().endswith("{"):
            text = text + " /* … */ }"
        fid = ctx.step_at.get((s.head_origin, s.head.body_line)) if s.head.body_line else None
        items.append(SkeletonItem(indent + text, m, s.origin, s.origin_class, fid))
    return head, items


def render_skeleton(path: str, head: list[str], items: list[SkeletonItem]) -> str:
    lines, current = list(head), path
    for it in items:
        if it.origin != current:
            lines.append(f"    // from {it.origin_class} ({it.origin})")
            current = it.origin
        lines.append(it.text)
    lines.append("}")
    return "\n".join(lines)


def skeleton(ctx: RepoContext, path: str, t: int, gone: set[str], masked: frozenset = frozenset()) -> str | None:
    parts = skeleton_parts(ctx, path, t, gone, masked)
    return None if parts is None else render_skeleton(path, *parts)


def future_ref(ctx: RepoContext, fn: dict, t: int, masked: frozenset = frozenset(), code: str | None = None) -> bool:
    """Rule 3 for ④(b) and ③: the function body uses a member of its class (own or inherited) first declared at or after
    t, or a masked infrastructure member. `code`, when given, is the text shown instead of the HEAD code (its pre-T
    form)."""
    if ctx.language == "ruby" or ctx.text(fn["path"]) is None:
        return False
    return bool(member_refs(ctx.language, fn["code"] if code is None else code) & ctx.future(fn["path"], t, masked))


def pre_t_code(ctx: RepoContext, fid: str, t: int) -> str | None:
    """Rule 5 for a catalog function: None when its HEAD code is unchanged since t (callers show HEAD code); else its
    text as it was before t: the method of the same name in the file's pre-t version (among overloads, the one with the
    HEAD parameter names, else the first), from the docblock or comments directly above it and its attributes to the end
    of its body (Member.span), dedented like catalog code. "" when that text cannot be located: a cucumber-java8 lambda
    step (no method), Ruby, no method of that name before t, or no pre-t version of the file."""
    if ctx.unchanged(fid, t):
        return None
    if ctx.language == "ruby":  # no class outline to locate the function in
        return ""
    fn = ctx.fns[fid]
    key = (fid, ctx.version_index(fn["path"], t))
    if key not in ctx.pre_t:
        ctx.pre_t[key] = _pre_t_code(ctx, fn, t)
    return ctx.pre_t[key]


def _pre_t_code(ctx: RepoContext, fn: dict, t: int) -> str:
    head = parse_snippet(ctx.language, fn["code"])
    head = next((m for m in head.members if m.kind == "method"), None) if head is not None else None
    old = ctx.outline_before(fn["path"], t) if head is not None else None
    same = [m for m in old.members if m.kind == "method" and m.names == head.names] if old is not None else []
    if not same:
        return ""
    want = param_names(ctx.language, head.text)
    m = next((x for x in same if param_names(ctx.language, x.text) == want), same[0])
    return textwrap.dedent("\n".join(ctx.text_before(fn["path"], t).splitlines()[m.span[0] - 1:m.span[1]]))


def render_answer(fn: dict, t: int, create_def_ids: frozenset[str]) -> str:
    """Gold code of a preexisting CREATE item: the function as V(r) shows it at t, plus the record's own
    CREATE annotations, so that "replace the method of the same name" keeps every annotation the model was shown."""
    drop = {a["line"] for a in fn["annotations"]
            if not (a["def_id"] in create_def_ids or (a["time"] is not None and a["time"] < t))}
    return "\n".join(l for i, l in enumerate(fn["code"].splitlines(), start=fn["first_line"]) if i not in drop)


def head_members(ctx: RepoContext, path: str) -> set[str]:
    o = ctx.outline(path)
    if o is None:
        return set()
    names = set()
    for p in ctx.chain(path):
        po = ctx.outline(p)
        if po is not None:
            names |= {n for m in po.members for n in m.names}
    return names


def member_use(ctx: RepoContext, record: dict, gone: set[str],
               masked: frozenset = frozenset()) -> list[tuple[dict, set[str], set[str]]]:
    """For each CREATE item: (item, members of its target class at HEAD, own or inherited, that its code uses, the
    subset the answer does not define and ④(a) does not show in a usable form: not shown at all, or a method shown with
    other parameter names than at HEAD, since the gold calls the HEAD form). self_contained() and the manual-check
    sample of scripts/state_report.py both read this, so they cannot disagree."""
    t, defined = record["t_vis"], set()
    for c in record["create"]:
        o = parse_snippet(ctx.language, c["code"])
        if o is not None:
            defined |= {n for m in o.members for n in m.names}
    out = []
    for c in record["create"]:
        used = member_refs(ctx.language, c["code"]) & head_members(ctx, c["path"])
        usable = set()
        for s in t_view(ctx, c["path"], t, gone, masked) or []:
            if s.head.kind not in ("method", "constructor") or \
                    param_names(ctx.language, s.head.text) == param_names(ctx.language, s.member.text):
                usable.update(s.member.names)
        out.append((c, used, used - usable - defined))
    return out


def self_contained(ctx: RepoContext, record: dict, gone: set[str], masked: frozenset = frozenset()) -> bool:
    """True when every member of a target class (own or inherited, at HEAD) that the gold answer uses is kept in V(r),
    or is defined by the answer itself. Ruby records are True by definition (no class outline)."""
    if ctx.language == "ruby":
        return True
    return not any(missing for _, _, missing in member_use(ctx, record, gone, masked))
