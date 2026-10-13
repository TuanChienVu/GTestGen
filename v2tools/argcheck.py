"""Argument checks of BV condition 4 (§2.2 of the design specification; corrected 24/09/2026; per framework since
25/09/2026): how many arguments N a step pattern hands to its function (placeholders; for a regex the capture groups its
framework passes, GROUP_RULE), how the function's parameters compare with N (ARITY_RULE), whether a quoted value or a
number of the step is written literally in the pattern, whether a step carries a table or a doc string, and how many
parameters a generated function takes."""
from __future__ import annotations

import math
import re

from .ertree import parser
from .patterns import _BEHAT_TOKEN, _CUKE_ARGS, _TOKEN, _ruby_to_python, compile_pattern, turnip_to_regex

try:  # Python 3.11+: the parser of the `re` module (sre_parse before)
    from re import _parser as _sre
except ImportError:  # pragma: no cover
    import sre_parse as _sre

STEP_LINE = re.compile(r"^\s*(?:Given|When|Then|And|But|\*)\s+\S")
RUBY_STEP = {"Given", "When", "Then", "And", "But", "Step"}  # the calls whose block is a Ruby step
QUOTED = re.compile(r'"[^"]*"')
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?![\w])")


def _captures(items, nested: bool) -> int:
    """Capture groups of a parsed regex: every one (`nested`), or those no other capture group holds. A non-capturing
    group, a branch, a repeat, an atomic group, a lookaround or a conditional is looked through; a named group is one
    group (the parser numbers it once)."""
    n = 0
    for op, av in items:
        name = str(op)
        if name == "SUBPATTERN" and av[0] is None:  # a non-capturing group (with flags)
            n += _captures(av[-1], nested)
        elif name == "SUBPATTERN":  # a capture group: counted; the groups inside it only when every group counts
            n += 1 + (_captures(av[-1], True) if nested else 0)
        elif name == "BRANCH":
            n += sum(_captures(b, nested) for b in av[1])
        elif name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
            n += _captures(av[2], nested)
        elif name == "ATOMIC_GROUP":
            n += _captures(av, nested)
        elif name in ("ASSERT", "ASSERT_NOT"):
            n += _captures(av[1], nested)
        elif name == "GROUPREF_EXISTS":
            n += _captures(av[1], nested) + (_captures(av[2], nested) if av[2] is not None else 0)
    return n


def _parsed(regex: str):
    """The pattern parsed by the `re` module's parser after patterns._ruby_to_python, exactly what compile_pattern
    compiles; None when it rejects the pattern (PCRE-only syntax), which then never binds a step (bind.Library)."""
    try:
        return _sre.parse(_ruby_to_python(regex))
    except re.error:
        return None


def all_groups(regex: str) -> int:
    """Every capture group, nested ones included, a named group once (Behat's PregMatchArgumentOrganiser, SpecFlow's
    Match.Groups minus group 0); 0 for a pattern the parser rejects. Counted by the parser, never by hand (§2.2 of the
    design specification)."""
    items = _parsed(regex)
    return 0 if items is None else _captures(items, True)


def top_groups(regex: str) -> int:
    """The outermost capture groups (cucumber-jvm's and cucumber-ruby's TreeRegexp: a group nested in another CAPTURE
    group goes to the same argument, while a non-capturing group, an alternation, a repeat or a lookaround hides
    nothing: `(?:customer named "(.*)"|the employee) … (\\d+)` has 2); 0 for a pattern the parser rejects."""
    items = _parsed(regex)
    return 0 if items is None else _captures(items, False)


# Condition 4 per framework (§2.2 of the design specification, condition 4, fixed 25/09/2026; each repo's framework read
# at its locked version and run for real on one feature file, regex_args/): how many arguments N a regex step hands to
# the method (a table or doc string adds 1), and how the method's parameters compare with N (arity_ok).
GROUP_RULE = {"php": all_groups, "csharp": all_groups, "java": top_groups, "ruby": top_groups}
ARITY_RULE = {  # (N, required, total, the step has a table or doc string, a parameter type-hinted to take it)
    # Behat: the arguments a method does not take are dropped, so no upper bound; but a table or doc string it
    # cannot take stops the run: it needs a TableNode/PyStringNode parameter, or at least N parameters
    "php": lambda n, req, total, table, takes: req <= n and (not table or takes or total >= n),
    "csharp": lambda n, req, total, table, takes: total == n,  # SpecFlow: every parameter, optional ones too; `params`
    "java": lambda n, req, total, table, takes: total == n,  # cucumber-jvm: exactly N; varargs is one parameter
    "ruby": lambda n, req, total, table, takes: n >= req if total == math.inf else n == req,  # Proc#arity
}


def regex_args(language: str, regex: str) -> int:
    return GROUP_RULE[language](regex)


class Params(tuple):
    """(required, total) of params_of, a pair; `takes_table`: a parameter type-hinted TableNode or PyStringNode (PHP:
    Behat binds a table or doc string to it by type)."""
    takes_table: bool

    def __new__(cls, required: int, total: float, takes_table: bool = False):
        self = super().__new__(cls, (required, total))
        self.takes_table = takes_table
        return self


def arity_ok(language: str, n: int, required: int, total: float, table: bool = False,
             takes_table: bool = False) -> bool:
    """Condition 4's comparison of N (the table or doc string included) with (required, total) = params_of(...) for the
    framework of `language`; `table`: the step has a table or doc string; `takes_table`: params_of(...).takes_table."""
    return ARITY_RULE[language](n, required, total, table, takes_table)


def placeholders(kind: str, pattern: str, language: str, step: str | None = None) -> int:
    """N of a step pattern: turnip :tokens and Cucumber {…} parameters count 1 each; a regex counts the capture groups
    its framework hands over (GROUP_RULE). Per §2.2 condition 4 of the design specification at 25ba873 (fixed
    28/09/2026), PHP's preg_match drops the TRAILING groups that take no part in the match, so Behat hands a method the
    groups up to the highest-numbered one that matched (shown by real Behat runs: ExactContext::p2 on "a and d" and ::p4
    without its colour fail for want of the last argument). For PHP and `step`, a regex's N is that number, read on the
    step's own match (the pattern as the step library compiles it, in its match mode; a named group once, as it is
    numbered once); without a step, or when the step does not match, every group."""
    if kind == "turnip":
        return len(re.findall(r":\w+", pattern))
    if kind == "cucumber":
        return sum(1 for m in _TOKEN.finditer(pattern) if m.group(0).startswith("{"))
    if language == "php" and step is not None:
        c = compile_pattern(kind, pattern)
        m = None if c is None else c[0].fullmatch(step) if c[1] == "fullmatch" else c[0].search(step)
        if m is not None:
            return max((i for i in range(1, c[0].groups + 1) if m.span(i) != (-1, -1)), default=0)
    return regex_args(language, pattern)


def capture_regex(kind: str, pattern: str) -> tuple[re.Pattern, str] | None:
    """The step pattern as a regex whose groups are its placeholders, and the match mode."""
    try:
        if kind == "turnip":
            return re.compile(turnip_to_regex(pattern).replace(_BEHAT_TOKEN, "(" + _BEHAT_TOKEN + ")"), re.I), "fullmatch"
        if kind == "cucumber":
            out, pos = [], 0
            for m in _TOKEN.finditer(pattern):
                out.append(re.escape(pattern[pos:m.start()]))
                tok = m.group(0)
                if tok.startswith("{"):
                    out.append("(" + _CUKE_ARGS.get(m.group(1), r".*?") + ")")
                elif m.group(2) is not None:
                    out.append(f"(?:{re.escape(m.group(2))})?")
                elif "/" in tok:
                    out.append("(?:" + "|".join(re.escape(p) for p in tok.split("/")) + ")")
                else:
                    out.append(re.escape(tok))
                pos = m.end()
            out.append(re.escape(pattern[pos:]))
            return re.compile("^" + "".join(out) + "$"), "fullmatch"
    except re.error:
        return None
    return compile_pattern(kind, pattern)


def hardcoded(kind: str, pattern: str, step: str) -> bool:
    """True when a double-quoted value or a number of the step is written LITERALLY in the pattern instead of being
    matched by a placeholder or a regex class (fixed 24/09/2026: `"[^"]+"` that matches without capturing is fine)."""
    c = capture_regex(kind, pattern)
    if c is None:
        return True
    m = c[0].fullmatch(step) if c[1] == "fullmatch" else c[0].search(step)
    if m is None:
        return True
    spans = [m.span(i) for i in range(1, c[0].groups + 1) if m.span(i) != (-1, -1)]
    plain = re.sub(r"\\(.)", r"\1", pattern)
    quoted = [(q.start(), q.end()) for q in QUOTED.finditer(step)]
    values = [(q.start() + 1, q.end() - 1) for q in QUOTED.finditer(step) if q.end() - q.start() > 2]
    values += [(x.start(), x.end()) for x in NUMBER.finditer(step) if not any(a <= x.start() < b for a, b in quoted)]
    for a, b in values:
        if any(a < e and s < b for s, e in spans):
            continue
        v = step[a:b]
        if re.search(rf"(?<![\w.]){re.escape(v)}(?![\w])" if v[:1].isalnum() or v[-1:].isalnum() else re.escape(v), plain):
            return True
    return False


def step_args(record: dict) -> list[bool]:
    """Per step of record["steps"]: True when a table or doc string follows it in the verbatim Gherkin."""
    lines = "\n".join(x for x in (record["background"], record["scenario"]) if x).split("\n")
    idx = [i for i, l in enumerate(lines) if STEP_LINE.match(l)]
    out = []
    for i in idx:
        nxt = next((l.strip() for l in lines[i + 1:] if l.strip() and not l.strip().startswith("#")), "")
        out.append(nxt.startswith(("|", '"""', "`" * 3)))
    return out if len(out) == len(record["steps"]) else [False] * len(record["steps"])


def ruby_step_params(text: str) -> list:
    """The block parameter nodes of a Ruby step: those of the do_block or block attached to the first call to Given,
    When, Then, And, But or Step, read from the tree: never a `do|` inside the step's regex, on the
    next line too, never an inner block's |i|, never a block-local variable after `;` (`|n; tmp|`: the
    `locals` field; Proc#arity ignores them)."""
    src = text.encode()
    stack = [parser("ruby").parse(src).root_node]
    while stack:
        n = stack.pop()
        name = n.child_by_field_name("method") if n.type == "call" else None
        if name is not None and src[name.start_byte:name.end_byte].decode() in RUBY_STEP:
            blk = n.child_by_field_name("block")
            ps = None if blk is None else next((c for c in blk.named_children if c.type == "block_parameters"), None)
            return [c for i, c in enumerate(ps.children) if c.is_named and ps.field_name_for_child(i) != "locals"] \
                if ps else []
        stack.extend(reversed(n.children))
    return []


TABLE_TYPES = {"tablenode", "pystringnode"}


def takes_table(typ: str) -> bool:
    """A PHP parameter type that takes a table or doc string: some alternative of the type (union `|`, nullable
    `?`, intersection `&`, DNF parentheses) whose last `\\` segment is TableNode or PyStringNode in any case, PHP
    class names being case-insensitive. Any class of that name counts (a documented limit)."""
    return any(alt.rsplit("\\", 1)[-1].casefold() in TABLE_TYPES for alt in re.split(r"[|&?()\s]+", typ) if alt)


def lambda_arity(text: str) -> int | None:
    """The parameters of the first lambda in a cucumber-java8 step (`Given("…", (…) -> {…});`), read from the tree:
    formal or inferred parameters, or one bare identifier; generics and nested brackets hold no extra parameter
    (`(java.util.Map<String, String> m) ->` has one). None when there is no lambda."""
    src = ("class __X { void __m() {\n" + text + "\n} }").encode()
    stack = [parser("java").parse(src).root_node]
    while stack:
        n = stack.pop()
        if n.type == "lambda_expression":
            ps = n.child_by_field_name("parameters")
            if ps is None:
                return 0
            return 1 if ps.type == "identifier" else sum(
                c.type in ("formal_parameter", "spread_parameter", "identifier") for c in ps.named_children)
        stack.extend(reversed(n.children))
    return None


def params_of(language: str, kind: str, text: str, arity: int | None = None) -> tuple[int, float]:
    """(required, total) parameters of one generated item, as arity_ok compares them with N (fixed 25/09/2026):
    - PHP: a parameter with a default is not required; a variadic makes total inf; `takes_table` when a parameter is
      type-hinted TableNode or PyStringNode;
    - C#: every parameter counts in total, optional ones too; a `params` array is ONE parameter (tree-sitter-c-sharp
      gives a `params` token, an array_type and an identifier in the parameter list);
    - Java: every parameter; varargs (`String... rest`) is ONE parameter; a cucumber-java8 lambda counts the
      parameters of its tree (lambda_arity; `arity` only when none is found);
    - a parameter is optional only when its tree gives it a default (PHP `default_value`, a C# `=`), never because
      of a `=` elsewhere in its text, such as inside an attribute;
    - Ruby, the step block's own parameters (ruby_step_params) read as Proc#arity does: a plain or destructured
      parameter is required, the required keywords together count one, a default, **kw or &blk does not count; a splat
      makes total inf, else total is the positional count."""
    if kind == "lambda":
        n = lambda_arity(text) if language == "java" else None
        n = (arity or 0) if n is None else n
        return Params(n, n)
    if language == "ruby":
        nodes = ruby_step_params(text)
        ps = [c.type for c in nodes]
        keyreq = any(c.type == "keyword_parameter" and c.child_by_field_name("value") is None for c in nodes)
        req = sum(t in ("identifier", "destructured_parameter") for t in ps) + keyreq
        return Params(req, math.inf if "splat_parameter" in ps else req + ps.count("optional_parameter"))
    wrapped = ("<?php class __X {\n" if language == "php" else "class __X {\n") + text + "\n}"
    src = wrapped.encode()
    root = parser(language).parse(src).root_node
    stack, node = [root], None
    while stack:
        n = stack.pop()
        if n.type in ("method_declaration", "constructor_declaration"):
            node = n
            break
        stack.extend(reversed(n.children))
    if node is None or node.child_by_field_name("parameters") is None:
        return Params(0, 0)
    req, total, takes = 0, 0.0, False
    total += sum(c.type == "params" for c in node.child_by_field_name("parameters").children)  # C# params T[] x
    for p in node.child_by_field_name("parameters").named_children:
        if p.type == "variadic_parameter":  # PHP ...$xs
            total = math.inf
        elif p.type == "spread_parameter":  # Java String... xs: one parameter
            total += 1
        elif p.type in ("simple_parameter", "property_promotion_parameter", "formal_parameter", "parameter"):
            total += 1
            req += p.child_by_field_name("default_value") is None and not any(c.type == "=" for c in p.children)
            typ = p.child_by_field_name("type")
            takes |= typ is not None and takes_table(src[typ.start_byte:typ.end_byte].decode("utf-8", errors="replace"))
    return Params(req, total, takes)
