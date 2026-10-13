"""CA, the decisive metric of Phase 3 (§2.2b of the design specification; fixed 24/09/2026, before any output): the
PROJECT calls of a step function, and the agreement of an output with the gold answer, matched step by step.

Calls of a function, read from its tree-sitter tree, only where the code can run (symbols.live: no comment, no string,
nothing after an unconditional return or throw, or a process exit, in the same block; Ruby: no line after
return/raise/fail):
- PHP: `$this->m->f(` -> (m, f); `$this->f(`, `self::f(`, `static::f(` -> ("self", f); `X::f(` with X a class in the
repo (PSR-4, or a global class with a <X>.php file outside vendor/) -> (X, f);
- Java, C#: `field.f(` and `this.field.f(` -> (field, f), field a member of the class or its in-repo ancestors; a bare
`f(` or `this.f(` with f such a member -> ("self", f);
- Ruby: a receiverless call to a method of the World -> ("self", f).
A call on a logger is not project knowledge: a receiver named log, logger, LOG, LOGGER, _log or _logger, or a field
whose declared type is a logger (symbols.Scope.loggers), never enters a call set.
Helper expansion (§2.2b of the design specification): in the call set of a function, gold or output, a ("self", f) call
to a method the output defines or to a FUTURE method of the ER tree is replaced by the project calls of f's body
(`expand`, the one expansion for the gold answer and every condition; evaluate.score_blocks decides what expands).
CA(record) = BV x [every gold function with a non-empty call set shares at least one call with the output functions
bound to the same steps]; a record whose gold functions all have empty call sets is undefined (ca_defined = False)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ertree import parser
from .symbols import DECLARES, LOGGER_NAME, class_status, in_anonymous_class, lambda_text, live, php_body, php_scoped, refs

Call = tuple[str, str]


def php_calls(ctx, o_after, code: str, loggers: frozenset[str] = frozenset()) -> set[Call]:
    src, body = php_body(code)
    text = lambda n: src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
    is_this = lambda n: n is not None and n.type == "variable_name" and text(n) == "$this"
    out = set()
    for n in live(body):
        if n.type in ("member_call_expression", "nullsafe_member_call_expression"):
            obj, name = n.child_by_field_name("object"), n.child_by_field_name("name")
            if name is None or name.type != "name":
                continue
            if is_this(obj):
                out.add(("self", text(name)))
            elif obj is not None and obj.type in ("member_access_expression", "nullsafe_member_access_expression") \
                    and is_this(obj.child_by_field_name("object")):
                m = obj.child_by_field_name("name")
                if m is not None and m.type == "name" and not (LOGGER_NAME.match(text(m)) or text(m) in loggers):
                    out.add((text(m), text(name)))
        elif n.type == "scoped_call_expression":
            scope, f = php_scoped(src, n)
            if scope.type == "relative_scope":
                out.add(("self", f))
            elif scope.type in ("name", "qualified_name") and re.match(r"\\?[A-Z]", text(scope)) \
                    and class_status(ctx, o_after, text(scope)) in ("resolved", "global"):
                out.add((text(scope).lstrip("\\"), f))
    return out


def jvm_calls(language: str, code: str, members: frozenset[str] | set[str], loggers: frozenset[str] = frozenset()) -> set[Call]:
    lam = lambda_text(language, code)
    src = ("class __X { void __m() { __f(" + lam + "); } }" if lam else "class __X {\n" + code + "\n}").encode()
    root = parser(language).parse(src).root_node
    text = lambda n: src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
    local, out = set(), set()

    def declare(n):
        p = n.parent
        if n.type == "implicit_parameter":
            local.add(text(n))
        elif n.type == "identifier" and p is not None:
            field = p.field_name_for_child(next(i for i, c in enumerate(p.children) if c.id == n.id))
            # the locals symbols.jvm_refs declares (added 04/10/2026; instanceof, type and record
            # patterns, C# `is T x`, tuples, LINQ from/let/join/into) hide a field receiver on the call side too
            if (p.type, field) in DECLARES or p.type in ("inferred_parameters", "type_pattern", "query_expression") \
                    or p.type in ("let_clause", "join_clause") \
                    and next(c for c in p.named_children if c.type == "identifier").id == n.id:
                local.add(text(n))
    for n in live(root):  # a local declared in dead code does not hide a member
        declare(n)

    def is_this(n):
        return n is not None and n.type in ("this", "this_expression")

    def use(n):
        if n.type == "method_invocation":
            obj, name = n.child_by_field_name("object"), text(n.child_by_field_name("name"))
            own = in_anonymous_class(n)  # a bare or this. call there is the anonymous class's
            if obj is None:
                if name in members and not own:
                    out.add(("self", name))
            elif is_this(obj):
                if not own:
                    out.add(("self", name))
            elif obj.type == "identifier" and text(obj) not in local and text(obj) in members:
                out.add((text(obj), name))
            elif obj.type == "field_access" and is_this(obj.child_by_field_name("object")) and not own:
                out.add((text(obj.child_by_field_name("field")), name))
        elif n.type == "invocation_expression":
            f = n.child_by_field_name("function")
            if f is None:
                return
            if f.type == "identifier":
                if text(f) in members:
                    out.add(("self", text(f)))
            elif f.type == "member_access_expression":
                e, nm = f.child_by_field_name("expression"), f.child_by_field_name("name")
                if e is None or nm is None:
                    return
                if is_this(e):
                    out.add(("self", text(nm)))
                elif e.type == "identifier" and text(e) not in local and text(e) in members:
                    out.add((text(e), text(nm)))
                elif e.type == "member_access_expression" and is_this(e.child_by_field_name("expression")):
                    out.add((text(e.child_by_field_name("name")), text(nm)))
    for n in live(root):
        use(n)
    return {(r, f) for r, f in out if r == "self" or not (LOGGER_NAME.match(r) or r in loggers)}


def calls(language: str, code: str, *, ctx=None, o_after=None, members=frozenset(), world=frozenset(),
          loggers=frozenset()) -> set[Call]:
    """Project calls of one function: `members` are the available members of its class (symbols.Scope.available),
    `world` the Ruby World methods, `loggers` the class's logger fields (symbols.Scope.loggers)."""
    if language == "php":
        return php_calls(ctx, o_after, code, loggers)
    if language == "ruby":
        return {("self", c) for c in refs("ruby", code).calls if c in world}
    return jvm_calls(language, code, members, loggers)


CUT_DEPTH = 3  # helpers expand to depth 3; a deeper call, or a cycle, stays ("self", f)


@dataclass(frozen=True)
class Expansion:
    calls: frozenset  # the CA call set: every expandable ("self", f) replaced by the project calls of f's body
    helpers: tuple  # (name, body) of every helper body expanded, in order, no repeats (condition 5 reads them)
    depth_cuts: int  # expandable calls kept as ("self", f) because a body of f was reached at depth > CUT_DEPTH
    cycle_cuts: int  # expandable calls kept as ("self", f) because a body of f was already on the expansion path


def expand(own, resolve, calls_of, root=None) -> Expansion:
    """Helper expansion (§2.2b of the design specification): the call set `own` of one function with every ("self", f)
    that `resolve(f)` expands (f's bodies, any hashable: evaluate.Body, the text and its declaring file; else None: a
    visible method or no method) replaced by the project calls of those bodies (`calls_of(body)`), recursively to depth
    CUT_DEPTH. The expansion path holds BODIES, the function's own body `root` first (fixed 25/09/2026: the overloads of
    one name are distinct bodies, so a step calling its own future overload expands that overload). Each body of a call
    is judged on its own: one already on the path is a cycle; else, at depth CUT_DEPTH, a depth cut (a cycle first);
    else it expands. When any body of the call is cut, the call is kept as ("self", f), and each counter counts that
    call once. Deterministic: calls are visited in sorted order, the bodies of a call in resolve's order."""
    out, helpers, cuts = set(), {}, {"depth": 0, "cycle": 0}

    def go(cs, path: tuple) -> None:
        for c in sorted(cs):
            bodies = list(dict.fromkeys(resolve(c[1]) or ())) if c[0] == "self" else []
            if not bodies:
                out.add(c)
                continue
            cycle = [b for b in bodies if b in path]
            deep = [b for b in bodies if b not in path] if len(path) - (root is not None) >= CUT_DEPTH else []
            cuts["cycle"] += bool(cycle)
            cuts["depth"] += bool(deep)
            if cycle or deep:
                out.add(c)
            for b in bodies:
                if b not in cycle and b not in deep:
                    helpers.setdefault((c[1], b), None)
                    go(calls_of(b), path + (b,))
    go(own, (root,) if root is not None else ())
    return Expansion(frozenset(out), tuple(helpers), cuts["depth"], cuts["cycle"])


def agreement(gold_binders: dict, gold_calls: dict, out_binders: dict, out_calls: dict, fold=None) -> tuple[bool, bool]:
    """(defined, agree): for each gold function with a non-empty call set, the union of the calls of the output
    functions bound to the steps that gold function binds must meet it. With `fold` (evaluate.ca: str.casefold for
    PHP), the METHOD name of every call, gold and output, is compared through it, never the receiver (a PHP property
    name is case-sensitive; design specification §2.2b: PHP call matching ignores case)."""
    key = (lambda cs: {(r, fold(f)) for r, f in cs}) if fold is not None else (lambda cs: set(cs))
    steps_of: dict = {}
    for step, g in gold_binders.items():
        steps_of.setdefault(g, set()).add(step)
    defined, agree = False, True
    for g, steps in steps_of.items():
        gc = key(gold_calls.get(g, set()))
        if not gc:
            continue
        defined = True
        oc = set().union(*(key(out_calls.get(out_binders.get(s), set())) for s in steps))
        agree &= bool(oc & gc)
    return defined, agree
