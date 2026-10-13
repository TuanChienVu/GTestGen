"""Execution scope: which Behat suite (TagFilter) or SpecFlow [Scope(Tag)] span a step definition is active in,
plus the catch-all/negative-control machinery shared by both."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# 20 distinct, deliberately meaningless sentences: none resembles a common step phrasing, so a legitimate
# step pattern should never match more than a handful by accident. A pattern matching all 20 is a catch-all.
NEGATIVE_STEPS: tuple[str, ...] = (
    "zqx florp the widget 9",
    "the moon is made of 42 cheeses",
    "blorp glimmers under stairs quux",
    "xyzzy plugh frobozz",
    "seventeen cucumbers debate zorblax",
    "a quokka audits the snarfle ledger",
    "glarb the fnord 3 times sideways",
    "qwerty uiop asdf ghjkl",
    "an octopus merges zzyzx",
    "my shadow owes 12 wibbles",
    "the kettle forgets vlorn",
    "purple elephants juggle quietly nnn",
    "frobnicate the gronk 77",
    "snozzberry protocol 5 engaged",
    "wug wug blicket dax",
    "the teapot sings in B flat zorp",
    "eleventy flumphs abscond",
    "klaatu barada nikto 1951",
    "plumbus grumbo schleem",
    "the zanzibar gazebo hums 0",
)


def negative_hits(regex: re.Pattern, mode: str) -> int:
    """How many NEGATIVE_STEPS this compiled pattern matches, using its own fullmatch/search mode."""
    matcher = regex.fullmatch if mode == "fullmatch" else regex.search
    return sum(1 for s in NEGATIVE_STEPS if matcher(s))


def is_catch_all(regex: re.Pattern, mode: str) -> bool:
    """True only when every negative-control sentence matches; matching 1-19 is BROAD, not catch-all."""
    return negative_hits(regex, mode) == len(NEGATIVE_STEPS)


def tag_filter_matches(expr: str, tags: Iterable[str]) -> bool:
    """Behat's TagFilter::isTagsMatchCondition: '&&' separates parts that must ALL hold (AND), ',' inside a
    part is OR between terms, a '~' prefix negates a term, and a leading '@' is optional/ignored on both
    sides; case-insensitive. An empty expression matches everything."""
    expr = expr.strip()
    if not expr:
        return True
    have = {t.lower().removeprefix("@") for t in tags}

    def term_matches(term: str) -> bool:
        negate = term.startswith("~")
        name = (term[1:] if negate else term).strip().lower().removeprefix("@")
        present = name in have
        return (not present) if negate else present

    for part in expr.split("&&"):
        terms = [t.strip() for t in part.split(",") if t.strip()]
        if terms and not any(term_matches(t) for t in terms):
            return False
    return True


@dataclass(frozen=True)
class Suite:
    name: str
    tag_filter: str
    files: frozenset[str]
    missing_contexts: tuple[str, ...]
    context_args_total: int = 0
    context_args_resolved: int = 0


def _match_delim(text: str, open_index: int, open_ch: str, close_ch: str) -> int:
    """Index of the close_ch matching the open_ch at open_index; skips // and /* */ comments and quoted
    strings, including C#'s @"verbatim" strings where "" is an escaped quote."""
    depth, i = 0, open_index
    while i < len(text):
        c = text[i]
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = nl if nl != -1 else len(text)
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i)
            i = end + 2 if end != -1 else len(text)
            continue
        if c == "@" and i + 1 < len(text) and text[i + 1] == '"':
            i += 2
            while i < len(text):
                if text[i] == '"':
                    if text[i:i + 2] == '""':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c in "\"'":
            q, i = c, i + 1
            while i < len(text) and text[i] != q:
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced delimiters")


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


# --- Behat suites -----------------------------------------------------------------------------------------

_SUITE_RE = re.compile(r"new\s+Suite\(\s*(['\"])(.*?)\1", re.S)
_WITHCONTEXTS_OPEN_RE = re.compile(r"->withContexts\(")
_FILTER_RE = re.compile(r"new\s+TagFilter\(\s*(['\"])(.*?)\1", re.S)
_QUOTED_FULL_RE = re.compile(r"^(['\"])(.*)\1$", re.S)
_CLASS_REF_FULL_RE = re.compile(r"^([A-Za-z0-9_\\]+)::class$")
_USE_RE = re.compile(r"^\s*use\s+([A-Za-z0-9_\\]+)(?:\s+as\s+([A-Za-z0-9_]+))?\s*;", re.M)
_SET_RE = re.compile(r"->set\(\s*(['\"])(.*?)\1\s*,\s*([A-Za-z0-9_\\]+)::class")
_TRAIT_USE_RE = re.compile(r"^[ \t]*use\s+([A-Za-z0-9_\\,\s]+?)\s*;", re.M)
_CLASS_OPEN_RE = re.compile(r"\bclass\s+\w+[^{;]*\{")
_NAMESPACE_RE = re.compile(r"namespace\s+([A-Za-z0-9_\\]+)\s*;")


def _use_map(text: str) -> dict[str, str]:
    uses: dict[str, str] = {}
    for m in _USE_RE.finditer(text):
        fqcn = m.group(1).lstrip("\\")
        alias = m.group(2) or fqcn.rsplit("\\", 1)[-1]
        uses[alias] = fqcn
    return uses


_UNPARSED_CALL = "<unparsed withContexts call>"


def _withcontexts_calls(text: str) -> list[str | None]:
    """Raw argument-list text of every ->withContexts(...) call, via balanced-paren matching -- not a
    `[^)]*` regex, which would stop at the first ')' and mis-split an argument like `new Foo('x')`. A call
    whose closing paren can't be found (e.g. a PHP '#'-comment containing an apostrophe, which _match_delim's
    naive quote handling mistakes for an opening quote) is reported as None rather than silently vanishing,
    so the caller can still count and flag it instead of dropping it."""
    calls: list[str | None] = []
    for m in _WITHCONTEXTS_OPEN_RE.finditer(text):
        open_index = m.end() - 1
        try:
            close_index = _match_delim(text, open_index, "(", ")")
        except ValueError:
            calls.append(None)
            continue
        calls.append(text[open_index + 1:close_index])
    return calls


def _split_call_args(s: str) -> list[str]:
    """Top-level comma-separated arguments of a call's raw text, respecting quotes and nested (), [], {};
    empty/whitespace-only pieces (a trailing comma) are dropped."""
    args, depth, quote, cur = [], 0, None, []
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            cur.append(c)
            if c == "\\" and i + 1 < len(s):
                i += 1
                cur.append(s[i])
            elif c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    args.append("".join(cur))
    return [a.strip() for a in args if a.strip()]


def _classify_context_arg(arg: str, uses: dict[str, str]) -> str:
    """A raw ->withContexts(...) argument -> a service-id string, "class:<FQCN>" (resolved via this suite
    file's own `use`s), or -- for anything else -- the argument's own verbatim text, so the caller can report
    it in missing_contexts instead of silently dropping it."""
    quoted = _QUOTED_FULL_RE.match(arg)
    if quoted:
        return quoted.group(2)
    class_ref = _CLASS_REF_FULL_RE.match(arg)
    if class_ref:
        ref = class_ref.group(1)
        return "class:" + (ref.lstrip("\\") if "\\" in ref else uses.get(ref, ref))
    return arg


def _parse_suite_file(text: str) -> list[tuple[str, list[str], str]]:
    """Every `new Suite('name'` in the file, with every argument from its ->withContexts(...) calls (each
    classified by _classify_context_arg, or the _UNPARSED_CALL marker for a call whose own closing paren
    couldn't be found) and the filter expression from its ->withFilter(new TagFilter('...')), scoped to the
    text up to the next Suite."""
    uses = _use_map(text)
    matches = list(_SUITE_RE.finditer(text))
    result = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.end():end]
        contexts = []
        for call in _withcontexts_calls(chunk):
            if call is None:
                contexts.append(_UNPARSED_CALL)
            else:
                contexts.extend(_classify_context_arg(arg, uses) for arg in _split_call_args(call))
        filt = _FILTER_RE.search(chunk)
        result.append((m.group(2), contexts, filt.group(2) if filt else ""))
    return result


def _service_class_map(text: str) -> dict[str, str]:
    """Service id -> FQCN, from ->set('id', Name::class) with Name resolved through this file's `use`s."""
    uses = _use_map(text)
    out: dict[str, str] = {}
    for m in _SET_RE.finditer(text):
        service_id, class_ref = m.group(2), m.group(3)
        fqcn = class_ref.lstrip("\\") if "\\" in class_ref else uses.get(class_ref)
        if fqcn:
            out[service_id] = fqcn
    return out


def _psr4_path(fqcn: str, psr4: dict[str, str]) -> str | None:
    candidates = [(prefix, base) for prefix, base in psr4.items() if fqcn.startswith(prefix)]
    if not candidates:
        return None
    prefix, base = max(candidates, key=lambda pb: len(pb[0]))
    rest = fqcn[len(prefix):].replace("\\", "/")
    return f"{base.rstrip('/')}/{rest}.php"


def _split_header_body(text: str) -> tuple[str, str]:
    m = _CLASS_OPEN_RE.search(text)
    return (text, "") if m is None else (text[:m.start()], text[m.end():])


def _class_and_trait_files(repo: Path, fqcn: str, psr4: dict[str, str]) -> frozenset[str] | None:
    """The context class's own file plus any trait file it `use`s in its body; None if the class file itself
    doesn't resolve to an existing file. Trait names are resolved via the file's own `use` imports, falling
    back to the class's own namespace; only files that actually exist on disk are kept."""
    rel = _psr4_path(fqcn, psr4)
    if rel is None or not (repo / rel).is_file():
        return None
    text = (repo / rel).read_text(encoding="utf-8", errors="replace")
    header, body = _split_header_body(text)
    uses = _use_map(header)
    ns = _NAMESPACE_RE.search(header)
    namespace = ns.group(1) if ns else ""
    files = {rel}
    for m in _TRAIT_USE_RE.finditer(body):
        for token in m.group(1).split(","):
            token = token.strip()
            if not token:
                continue
            if "\\" in token:
                trait_fqcn = token.lstrip("\\")
            else:
                trait_fqcn = uses.get(token) or (f"{namespace}\\{token}" if namespace else token)
            trait_rel = _psr4_path(trait_fqcn, psr4)
            if trait_rel and (repo / trait_rel).is_file():
                files.add(trait_rel)
    return frozenset(files)


def load_behat_suites(repo: Path, suite_files: list[str], service_files: list[str], psr4: dict[str, str]) -> list[Suite]:
    service_map: dict[str, str] = {}
    for sf in service_files:
        service_map.update(_service_class_map((repo / sf).read_text(encoding="utf-8", errors="replace")))
    cache: dict[str, frozenset[str] | None] = {}

    def files_for(fqcn: str) -> frozenset[str] | None:
        if fqcn not in cache:
            cache[fqcn] = _class_and_trait_files(repo, fqcn, psr4)
        return cache[fqcn]

    suites: list[Suite] = []
    for suite_file in suite_files:
        text = (repo / suite_file).read_text(encoding="utf-8", errors="replace")
        for name, context_ids, tag_filter in _parse_suite_file(text):
            files: set[str] = set()
            missing: list[str] = []
            resolved_count = 0
            for cid in context_ids:
                fqcn = cid.removeprefix("class:") if cid.startswith("class:") else service_map.get(cid)
                resolved = files_for(fqcn) if fqcn else None
                if resolved is not None:
                    files.update(resolved)
                    resolved_count += 1
                else:
                    missing.append(cid)
            suites.append(Suite(name, tag_filter, frozenset(files), tuple(missing), len(context_ids), resolved_count))
    return suites


# --- SpecFlow [Scope(Tag = "...")] spans -------------------------------------------------------------------

_SCOPE_TAG_RE = re.compile(r'Scope\s*\(\s*Tag\s*=\s*"([^"]+)"\s*\)', re.I)


def _skip_attrs_and_ws(text: str, pos: int) -> int:
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] == "[":
            pos = _match_delim(text, pos, "[", "]") + 1
        else:
            return pos


def specflow_scope_tags(path: str, text: str) -> list[tuple[int, int, frozenset[str]]]:
    """One (start_line, end_line, tags) span per `[...]` attribute block that contains one or more
    Scope(Tag = "x") entries, where start_line is that block's own opening line and end_line is the line of
    the '}' closing the body of the class or method the (possibly further attribute-decorated) declaration
    right after it introduces. Tags are normalised to lowercase "@x"."""
    spans: list[tuple[int, int, frozenset[str]]] = []
    for m in re.finditer(r"\[", text):
        start = m.start()
        try:
            close = _match_delim(text, start, "[", "]")
        except ValueError:
            continue
        block = text[start:close + 1]
        tags = frozenset(f"@{t.lower()}" for t in _SCOPE_TAG_RE.findall(block))
        if not tags:
            continue
        after = _skip_attrs_and_ws(text, close + 1)
        brace = text.find("{", after)
        if brace == -1:
            continue
        try:
            body_close = _match_delim(text, brace, "{", "}")
        except ValueError:
            continue
        spans.append((_line_of(text, start), _line_of(text, body_close), tags))
    return spans


def active_in_scope(d, spans: list[tuple[int, int, frozenset[str]]], tags: Iterable[str]) -> bool:
    """A step definition outside every span is always active. One inside one or more spans is active if the
    scenario's tags intersect ANY single covering span's own tags -- not all of them: SpecFlow creates one
    independent binding per [Scope] attribute, so several attributes on the same method/class (e.g. a
    class-level one and a method-level one) OR together, they don't AND. Checked against the def's
    body_start (always inside the body a span covers), not its own attribute line, since a step attribute
    can be written above or below the [Scope(...)] attribute on the same method. Tags are compared
    case-insensitively (span tags from specflow_scope_tags are already lowercased; scenario tags are kept
    verbatim by the caller, so this helper lowercases its own side rather than relying on every caller to)."""
    tag_set = {t.lower() for t in tags}
    covering = [s for s in spans if s[0] <= d.body_start <= s[1]]
    return not covering or any(tag_set & s[2] for s in covering)


# --- Wrong-suite negative control --------------------------------------------------------------------------

def pick_wrong_suite(active: list[str], all_names: list[str]) -> str | None:
    """A suite NOT active for a scenario, to bind against as a negative control (a scenario should mostly
    NOT bind against a suite it wasn't routed to). Deterministic: among the inactive suites sorted by name,
    the first one greater than the active suites' minimum name, wrapping around to the first inactive suite
    if none is; None if there is no inactive suite left, or no suite is active in the first place."""
    if not active:
        return None
    inactive = sorted(n for n in all_names if n not in active)
    if not inactive:
        return None
    threshold = min(active)
    return next((n for n in inactive if n > threshold), inactive[0])
