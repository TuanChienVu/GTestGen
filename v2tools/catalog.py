"""Every step-definition function of a repository: its code, its annotations (each with its own earliest-evidence
time), the Behat suites / SpecFlow scopes it belongs to, and a per-step binder that reproduces Phase 1 binding."""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

from .bind import Library
from .history import earliest, first_seen, line_times
from .scopes import active_in_scope, specflow_scope_tags
from .stepdefs import extract
from .yield_report import _behat_suites, _classify_catch_all, _git


@dataclass(frozen=True)
class Annotation:
    def_id: str          # "path:line", the Phase 1 target id
    line: int
    keyword: str
    pattern: str
    time: int | None     # when this PATTERN first existed: see pattern_time()


@dataclass(frozen=True)
class Fn:
    fn_id: str           # "path:body_start", the Phase 1 target_fns key
    path: str
    first_line: int      # first line of the declaration (a PHP docblock's `/**`, else the first annotation)
    body_end: int
    code: str            # source lines first_line..body_end, common indentation removed
    annotations: tuple[Annotation, ...]
    suites: tuple[str, ...]            # Behat suites whose context files include `path`; ("default",) if unscoped
    scope_tags: tuple[tuple[str, ...], ...]  # SpecFlow [Scope] tag sets covering the function; () = unscoped
    catch_all: bool
    # dead = every annotation line is commented out (a `//`/`#` line or inside a non-docblock /* */ block), or
    # the entry has no body line at all (a stacked annotation split off its function); contains_fn = the span contains
    # another catalog function of the file (a mis-parsed dead one)
    dead: bool = False
    contains_fn: bool = False

    @property
    def time(self) -> int | None:
        return earliest(*(a.time for a in self.annotations))


def render(fn: Fn, before: int | None) -> str:
    """The function's code with every annotation line whose time is not strictly before `before` removed, so a
    context snapshot never shows a pattern that did not exist yet. `before=None` keeps everything."""
    drop = {a.line for a in fn.annotations if before is not None and not (a.time is not None and a.time < before)}
    lines = fn.code.splitlines()
    return "\n".join(l for i, l in enumerate(lines, start=fn.first_line) if i not in drop)


_QUOTED = re.compile(r"@\"((?:[^\"]|\"\")*)\"|'((?:[^'\\\\]|\\\\.)*)'|\"((?:[^\"\\\\]|\\\\.)*)\"")
_DOCBLOCK = re.compile(r"@(?:Given|When|Then|And|But)\s+(.+?)\s*(?:\*/)?\s*$")
_RUBY = re.compile(r"\b(?:Given|When|Then|And|But|Step)\s*\(?\s*(?:/(.+)/|%r\{(.+)\})")


def pattern_key(text: str) -> str:
    """One normal form for a step pattern whichever syntax wrapped it: no regex slashes or ^/$ anchors, C# doubled
    quotes and backslash-escaped quotes undone, whitespace collapsed."""
    t = text.strip()
    if len(t) > 1 and t[0] == "/" and t[-1] == "/":
        t = t[1:-1]
    t = t[1:] if t.startswith("^") else t
    t = t[:-1] if t.endswith("$") and not t.endswith("\\$") else t
    return " ".join(t.replace('""', '"').replace("\\'", "'").replace('\\"', '"').split())


def line_keys(line: str) -> set[str]:
    """Every pattern_key a source line could be declaring: quoted literals, a docblock `@When ...` tail, and Ruby
    `/.../` or `%r{...}` regexes. Over-generating keys is harmless: only keys equal to a real pattern are looked up."""
    keys = {pattern_key(next(g for g in m.groups() if g is not None)) for m in _QUOTED.finditer(line)
            if any(g is not None for g in m.groups())}
    for rx in (_DOCBLOCK, _RUBY):
        m = rx.search(line)
        if m:
            keys.add(pattern_key(next(g for g in m.groups() if g is not None)))
    keys.discard("")
    return keys


class Catalog:
    def __init__(self, repo: str, fns: list[Fn], defs: list, scoping_kind: str | None, spans: dict[str, list]):
        self.repo = repo
        self.fns = {f.fn_id: f for f in fns}
        self._defs = defs
        self._kind = scoping_kind
        self._spans = spans
        self._fn_of = {f"{d.path}:{d.line}": f"{d.path}:{d.body_start}" for d in defs}
        self._libraries: dict[str, Library] = {}

    def fn_of(self, def_id: str) -> str:
        return self._fn_of[def_id]

    def _library(self, suite: str, tags: list[str]) -> Library:
        if self._kind == "specflow_scope":
            key = "\x00".join(sorted(tags))
            if key not in self._libraries:
                tagset = set(tags)
                self._libraries[key] = Library([d for d in self._defs if not self.fns[self._fn_of[f"{d.path}:{d.line}"]].catch_all
                                                and active_in_scope(d, self._spans.get(d.path, []), tagset)])
            return self._libraries[key]
        if suite not in self._libraries:
            self._libraries[suite] = Library([d for d in self._defs
                                              if not self.fns[self._fn_of[f"{d.path}:{d.line}"]].catch_all
                                              and suite in self.fns[self._fn_of[f"{d.path}:{d.line}"]].suites])
        return self._libraries[suite]

    def bind_steps(self, steps: list[str], suite: str, tags: list[str]) -> list[list[str]]:
        """For each step, the sorted def ids (annotation lines) it matches in the pair's active library."""
        lib = self._library(suite, tags)
        return [sorted({f"{d.path}:{d.line}" for d in lib.matches(s)}) for s in steps]

    def active(self, fn: Fn, suite: str, tags: list[str]) -> bool:
        if fn.catch_all:
            return False
        if self._kind == "specflow_scope":
            return not fn.scope_tags or any(set(t) & {x.lower() for x in tags} for t in fn.scope_tags)
        return suite in fn.suites


_LINE_COMMENT = re.compile(r"^\s*(//|#(?!\[))")  # a line comment; `*` and `/**` are docblock syntax, not "commented out"
_ANY_COMMENT = re.compile(r"^\s*(/\*\*|\*|\*/|//|#(?!\[))")  # docblock and comment lines (body_key's definition)


def block_comment_lines(text: str) -> set[int]:
    """1-based lines covered by /* ... */ blocks that are NOT /** docblocks (a Behat docblock holds live annotations),
    skipping "..." and '...' literals and // line comments; for PHP, Java and C# (Ruby has no such blocks)."""
    out: set[int] = set()
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] not in (c, "\n"):
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        elif text.startswith("/*", i):
            docblock = text.startswith("/**", i) and not text.startswith("/**/", i)
            j = text.find("*/", i + 2)
            j = n if j < 0 else j
            end = line + text.count("\n", i, j)
            if not docblock:
                out.update(range(line, end + 1))
            line, i = end, j + 2
        else:
            i += 1
    return out


def build_catalog(source: dict, repo: Path, history: bool = True) -> Catalog:
    step_files = [p for p in _git(repo, "ls-files", *source["step_globs"]).splitlines() if p]
    texts = {p: (repo / p).read_text(encoding="utf-8", errors="replace") for p in step_files}
    defs = []
    for path in step_files:
        for style in source["styles"]:
            try:
                defs.extend(extract(path, texts[path], style))
            except ValueError:
                continue
    catch_all, _, _ = _classify_catch_all(defs)
    kind = (source.get("scoping") or {}).get("kind")
    suites = _behat_suites(source, repo) or []
    spans = {p: specflow_scope_tags(p, texts[p]) for p in step_files} if kind == "specflow_scope" else {}
    seen = first_seen(repo, source["step_globs"]) if history else {}
    key_time: dict[str, int] = {}
    for text, t in seen.items():
        for k in line_keys(text):
            key_time[k] = min(key_time.get(k, t), t)
    blame: dict[str, dict[int, int]] = {}

    def times(path: str) -> dict[int, int]:
        if path not in blame:
            blame[path] = line_times(repo, path) if history else {}
        return blame[path]

    groups: dict[str, list] = {}
    for d in defs:
        groups.setdefault(f"{d.path}:{d.body_start}", []).append(d)
    blocks = {p: set() if source.get("language") == "ruby" else block_comment_lines(texts[p]) for p in step_files}
    def_lines: dict[str, set[int]] = {}
    for d in defs:
        def_lines.setdefault(d.path, set()).add(d.line)
    fns = []
    for fn_id, ds in groups.items():
        ds = sorted(ds, key=lambda d: d.line)
        lines = texts[ds[0].path].splitlines()
        anns = tuple(Annotation(f"{d.path}:{d.line}", d.line, d.keyword, d.pattern,
                                earliest(times(d.path).get(d.line), seen.get(lines[d.line - 1].strip()),
                                         key_time.get(pattern_key(d.pattern))) if history else None) for d in ds)
        span_tags = tuple(tuple(sorted(tags)) for start, end, tags in spans.get(ds[0].path, [])
                          if start <= ds[0].body_start <= end)
        first = ds[0].line
        while first > 1 and (lines[first - 2].strip().startswith("/**")
                             or (lines[first - 2].strip().startswith("*") and not lines[first - 2].strip().startswith("*/"))):
            first -= 1  # climb to the docblock opener so the code is valid PHP, not "@Given ... */ public function"
        code = textwrap.dedent("\n".join(lines[first - 1:ds[0].body_end]))
        commented = all(_LINE_COMMENT.match(lines[d.line - 1]) or d.line in blocks[d.path] for d in ds)
        bodiless = all(i in def_lines[ds[0].path] or not lines[i - 1].strip() or _ANY_COMMENT.match(lines[i - 1])
                       for i in range(first, ds[0].body_end + 1))
        dead = commented or bodiless
        fns.append(Fn(fn_id, ds[0].path, first, ds[0].body_end, code, anns,
                      tuple(sorted(s.name for s in suites if ds[0].path in s.files)) if suites else ("default",),
                      span_tags, any(d in catch_all for d in ds), dead))
    spans_of: dict[str, list[tuple[int, int, str]]] = {}
    for f in fns:
        spans_of.setdefault(f.path, []).append((f.first_line, f.body_end, f.fn_id))
    fns = [replace(f, contains_fn=any(g != f.fn_id and f.first_line <= a and b <= f.body_end for a, b, g in spans_of[f.path]))
           for f in fns]
    return Catalog(source["repo"], fns, defs, kind, spans)
