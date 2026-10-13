"""Extract step definitions (pattern + function body) from PHP, Java, C# and Ruby step files."""
from __future__ import annotations

import re
from dataclasses import dataclass

KEYWORDS = r"Given|When|Then|And|But|Step|StepDefinition"  # what a step declaration LOOKS like (_looks_step, rejected)
# The names a cucumber-java8 lambda (io.cucumber.java8.En declares Given, When, Then, And, But only: javap, and `Step(`
# does not compile) and a cucumber-ruby step call (Cucumber::Glue::Dsl aliases the Gherkin keywords only: `Step` raises
# NoMethodError) register with, as real runs show.
_EN_KEYWORDS = r"Given|When|Then|And|But"
_RUBY_REGEX = r"(?:/(?P<slash>.*)/|%r\{(?P<brace>.*)\})[imxo]*"
# The reader used for scoring (§2.2, changed 27/09/2026, AFTER the Phase 1/2 seal): it takes every step declaration form
# the target framework registers, each shown by a real run, and no other; it reads code as the compiler does (nothing
# inside a string, a comment or a heredoc is code). The Phase 1/2 seal reproduces byte for byte only with its recorded
# tool commits of the research repository (5891f00, 8473872). On every source file of the six repositories at their
# pinned commits, this reader gives, without imports (Phase 1, _extract_sealed), the sealed reader's definitions except
# gpconnect ManualSteps.cs (3 -> 9: a group of three attributes); with them (3C) also except four commented-out
# definitions (fineract LoanStepDef.java:5446, :5458; gpconnect
# AccessStructuredMedicationSteps.cs:670, StructuredMultipleSteps.cs:185), the body of fineract
# WorkingCapitalNearBreachActionStepDef.java:78 ((79,79) -> (80,85)) and the doc line with a second `*` of Sylius
# ManagingProductOptionsContext.php:214 (no framework registers it).
# PHP (Behat 4.0.0-alpha1, shown by real runs): an attribute whose name resolves, as PHP resolves it, to exactly
# Behat\Step\{Given,When,Then} (Behat looks the resolved name up case for case: a short name through its `use`, or an
# alias, in any case; a fully qualified name `\Behat\Step\Given` in its own case), any whitespace, the pattern
# positional or named `pattern:` as a string literal or a concatenation of literals, several attributes in one `#[...]`,
# on a public method only. A class constant as the pattern is not read (a documented limit). Without `imports` (Phase 1:
# the catalog), a short name is read without its `use`, as before.
_PHP_LIT = r"""(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")"""
_PHP_ATTR = re.compile(rf"\s*(\\?[A-Za-z_]\w*(?:\\[A-Za-z_]\w*)*)\s*\(\s*(?:pattern\s*:\s*)?({_PHP_LIT}(?:\s*\.\s*{_PHP_LIT})*)"
                       r"\s*,?\s*\)\s*", re.S)
# What _code_only blanks in PHP besides comments: string literals and heredoc/nowdoc bodies.
_PHP_CODE_LIT = (rf"{_PHP_LIT}|<<<[ \t]*(?P<hq>[\"']?)(?P<hd>[A-Za-z_]\w*)(?P=hq)\r?\n.*?\n[ \t]*(?P=hd)\b")
# PHP docblock (Behat 3.26.0 and 3.33.0): _php_doc_steps; mergeMultilines, DOCLINE_TRIMMER_REGEX and the tag regex.
_DOC_MERGE = re.compile(r"\\$\s*\*\s*([^\\$]+)", re.M)
_DOC_TRIM = re.compile(r"^/\*\*\s*|^\s*\*\s*|\s*\*/$|\s*$")
_DOC_STEP = re.compile(r"@(given|when|then)\s+(.+)", re.I)
_PHP_STEPS = {"Behat\\Step\\Given": "Given", "Behat\\Step\\When": "When", "Behat\\Step\\Then": "Then"}
_PHP_USE = re.compile(r"^[ \t]*use\s+(?!function\b|const\b)\\?([\w\\]+?)\\?(?:\\\{([^}]*)\}|\s+as\s+(\w+))?\s*;", re.M)
_PHP_TYPE = re.compile(r"^[ \t]*(?:(?:abstract|final|readonly)\s+)*(?:class|interface|trait|enum)\s+\w", re.M)

# Java (Cucumber-JVM 7.34.3 and 7.8.1, shown by real runs): an annotation whose name resolves, as Java resolves it, to
# io.cucumber.java.en.{Given,When,Then,And,But} (the fully qualified name, or a simple name through a single-type or an
# on-demand import), any whitespace (`@ Given ( "x" )` included), the pattern positional or as `value =`, a string
# literal or a concatenation of literals, several annotations on one method, on a public method only (MethodScanner
# reads Class.getMethods()). A constant as the pattern is not read (a documented limit). Without `imports` (Phase 1: the
# catalog), a simple name is read without its import, as before.
_JAVA_LIT = r'"(?:[^"\\]|\\.)*"'
# What _code_only blanks in Java besides comments: text blocks, string literals and char literals ('"', '{').
_JAVA_CODE_LIT = rf'"""(?:[^\\]|\\.)*?"""|{_JAVA_LIT}|\'(?:[^\'\\\n]|\\.)+\''
_JAVA_ANN = re.compile(rf'@\s*((?:[A-Za-z_$][\w$]*\s*\.\s*)*[A-Za-z_$][\w$]*)\s*\(\s*(?:value\s*=\s*)?'
                       rf'({_JAVA_LIT}(?:\s*\+\s*{_JAVA_LIT})*)')
_JAVA_STEPS = {f"io.cucumber.java.en.{k}": k for k in ("Given", "When", "Then", "And", "But")}
_JAVA_IMPORT = re.compile(r"^\s*import\s+(?!static\b)([\w.]+?)(\.\*)?\s*;", re.M)
# C# (SpecFlow 3.0.213, shown by a real run): an attribute whose name resolves, as C# resolves it (with or without its
# "Attribute" suffix), to TechTalk.SpecFlow.{Given,When,Then,StepDefinition}Attribute: the fully qualified name, a using
# alias (`using G = TechTalk.SpecFlow.GivenAttribute;`), or a simple name through `using TechTalk.SpecFlow;`; any
# whitespace; the regex positional or as `Regex =` (other named arguments after it); a string literal or a concatenation
# of literals; several attributes in one `[...]`; on a method of any access level (SpecFlow scans NonPublic methods
# too). A constant as the regex, and `[Given]` without one (SpecFlow derives it from the method name), are not read
# (documented limits). Without `imports` (Phase 1), a simple name is read as before.
_CS_LIT = r'(?:@"(?:[^"]|"")*"|"(?:[^"\\]|\\.)*")'
# What _code_only blanks in C# besides comments: raw strings, verbatim and regular strings, char literals.
_CS_CODE_LIT = rf'"""(?:.|\n)*?"""|{_CS_LIT}|\'(?:[^\'\\\n]|\\.)+\''
_CS_ATTR = re.compile(rf'\s*((?:[A-Za-z_]\w*\s*\.\s*)*[A-Za-z_]\w*)\s*\(\s*(?:Regex\s*=\s*)?({_CS_LIT}(?:\s*\+\s*{_CS_LIT})*)'
                      r'\s*(?:,[^)]*)?\)\s*', re.S)
_CS_STEPS = {f"TechTalk.SpecFlow.{k}Attribute": k for k in ("Given", "When", "Then", "StepDefinition")}
_CS_USING = re.compile(r"^\s*using\s+(?!static\b)(?:(\w+)\s*=\s*)?([\w.]+)\s*;", re.M)

# Ruby (cucumber-ruby): besides a Regexp, a String pattern is a Cucumber expression.
_RUBY_STRING = r"""(?:'(?P<sq>(?:[^'\\]|\\.)*)'|"(?P<dq>(?:[^"\\]|\\.)*)")"""


@dataclass(frozen=True)
class StepDef:
    path: str
    style: str
    kind: str
    keyword: str
    pattern: str
    line: int
    body_start: int
    body_end: int
    code: str


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _close_brace(text: str, open_index: int, pair: str = "{}") -> int:
    """Index of the brace closing the one at open_index; skips strings and // or /* */ comments. `pair`: another pair of
    brackets, "()" for the parenthesis of a Java registration call."""
    (opening, closing), depth, quote, escaped, j = pair, 0, None, False, open_index
    while j < len(text):
        c = text[j]
        if quote:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quote:
                quote = None
        elif text.startswith("//", j):
            j = text.find("\n", j)
            if j < 0:
                break
        elif text.startswith("/*", j):
            j = text.find("*/", j) + 1
            if j <= 0:
                break
        elif c in "\"'":
            quote = c
        elif c == opening:
            depth += 1
        elif c == closing:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError("unbalanced braces")


def _code_only(text: str, lit: str, hash_comments: bool = False) -> str:
    """`text` with every string literal of `lit` and every `//` or `/* */` comment (and, for PHP, a `#` comment that is
    no `#[` attribute) blanked, same length and line breaks kept, in ONE left-to-right pass (a `//` inside a string, or
    a quote inside a comment, starts nothing): a keyword, a modifier, a brace or an annotation there is no code."""
    return re.sub(_code_regex(lit, hash_comments), lambda s: re.sub(r"[^\n]", " ", s.group(0)), text)


def _code_regex(lit: str, hash_comments: bool = False) -> re.Pattern:
    comment = r"//[^\n]*|/\*.*?\*/" + (r"|#(?!\[)[^\n]*" if hash_comments else "")
    return re.compile(rf"(?:{lit})|{comment}", re.S)


def _php_docblocks(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of every PHP doc comment of `text`: a `/**` followed by whitespace, as PHP's lexer takes it (a
    `/**` inside a string, a heredoc or another comment is none; `/**/` and `/**x` are plain comments), up to its first
    `*/` (read as the framework reads, getDocComment())."""
    return [(m.start(), m.end(), m.group(0)) for m in _code_regex(_PHP_CODE_LIT, hash_comments=True).finditer(text)
            if re.match(r"/\*\*\s", m.group(0))]


def _php_function(code: str, after: int) -> int:
    """Index of the first `function` keyword of `code` (_code_only) after `after`, -1 if none."""
    m = re.compile(r"\bfunction\b").search(code, after)
    return m.start() if m else -1


def _php_doc_steps(doc: str) -> list[tuple[int, str, str]]:
    r"""(index in `doc` of the `@`, keyword, pattern) of every step tag of the PHP doc comment `doc`, as Behat 3.26.0
    and 3.33.0 read one (AnnotatedContextReader::readDocBlockCallees and DefinitionAnnotationReader, shown by real
    runs): a line that ends in a backslash is joined with the next (mergeMultilines); each line, `/**`, a leading `*`, a
    trailing `*/` and blanks trimmed, is a tag only when it then starts with `@` and matches `@(given|when|then)\s+(.+)`
    in any case (the keyword capitalised). So `@And`, `@But`, `@Step`, `@Given:`, a tag after prose and a line with a
    second `*` are none."""
    removed, parts, last = [], [], 0
    for m in _DOC_MERGE.finditer(doc):
        parts.append(doc[last:m.start()])
        removed.append((sum(len(x) for x in parts), m.start(1) - m.start()))
        last = m.start(1)
    merged = "".join(parts) + doc[last:]
    out, at = [], 0
    for line in merged.split("\n"):
        lead = re.match(r"/\*\*\s*|\s*\*\s*", line)
        m = _DOC_STEP.fullmatch(_DOC_TRIM.sub("", line))
        if m:
            i = at + (lead.end() if lead else 0)
            out.append((i + sum(n for where, n in removed if where <= i), m.group(1).capitalize(), m.group(2)))
        at += len(line) + 1
    return out


def _body(text: str, after: int, code: str | None = None) -> tuple[int, int]:
    """(the `{` that opens the body after `after`, the `}` that closes it). With `code` (_code_only of `text`), that `{`
    is the first one OUTSIDE string literals and comments, not one of a later annotation's pattern (`{string}`, `{2}`);
    the sealed Phase 1 reader took the first `{` of the raw text."""
    open_index = (text if code is None else code).index("{", after)
    return open_index, _close_brace(text, open_index)


def _cut(read) -> bool:
    """Whether `read` (a call that finds a method's body) raises ValueError: the text ends before that body closes (no
    method, no `{`, or unbalanced braces), i.e. the method is cut off (§2.2)."""
    try:
        read()
        return False
    except ValueError:
        return True


def _php_cut(text: str, after: int, code: str) -> bool:
    """Whether the PHP method after `after` (the `function` that follows, as the reader finds it: outside strings and
    comments) is cut off."""
    def body():
        fn = _php_function(code, after)
        if fn < 0:
            raise ValueError("substring not found")
        _body(text, fn, code)
    return _cut(body)


def _body_cut(text: str, after: int, code: str) -> bool:
    """Whether the method after `after` (the end of an annotation or attribute group, in Java or C#) is cut off: no `{`
    outside strings and comments follows (`code`, _code_only), or that `{` never closes. A pattern's own braces
    (`{string}`, `{2}`) are no body."""
    return _cut(lambda: _body(text, after, code))


def _ruby_cut(lines: list[str], text: str, i: int) -> bool:
    """Whether the block that the Ruby step call on line `i` opens is cut off: a `do` block (the line ends with `do`,
    block parameters or a comment after it allowed) with no `end` at the call's indent (the reader's own end of a
    block), or a `{` block (the first `{` after the line's last `)`, or its first `{` without one) whose brace never
    closes; a call that opens no block is not."""
    line = lines[i]
    if re.search(r"\bdo\b\s*(?:\|[^|]*\|)?\s*(?:#.*)?$", line):
        indent = re.match(r"^(\s*)", line).group(1)
        return not any(re.match(rf"^{re.escape(indent)}end\b", lines[k]) for k in range(i + 1, len(lines)))
    brace = line.find("{", line.rfind(")") + 1)
    if brace < 0:
        return False
    return _cut(lambda: _close_brace(text, sum(len(x) + 1 for x in lines[:i]) + brace))


def _sealed_php_cut(text: str, after: int) -> bool:
    """The sealed path (_extract_sealed): whether the PHP method after `after` (the `function` that follows in the raw
    text) is cut off."""
    return _cut(lambda: _body(text, text.index("function", after)))


def _sealed_body_cut(text: str, after: int, lit: str) -> bool:
    """The sealed path (_extract_sealed): whether the method after `after` (the end of an annotation or attribute group,
    in Java or C#) is cut off: no `{` outside the string literals `lit` follows, or that `{` never closes."""
    def body():
        blank = re.sub(lit, lambda s: '"' + " " * (len(s.group(0)) - 2) + '"', text[after:])
        _close_brace(text, after + blank.index("{"))
    return _cut(body)


def _extract_sealed(path: str, text: str, style: str, rejected: list | None, read: list | None) -> list[StepDef]:
    """extract without `imports` for PHP, Java and C# (Phase 1: the sealed catalog, yield report and wrong-suite control
    call it so): the reader as it was before it read code as the compiler does (on every source file of the six
    repositories at their pinned commits it gives the sealed reader's definitions, except gpconnect ManualSteps.cs, 3 ->
    9), kept as it was so that Phase 1/2's sealed outputs and tests stand (28/09/2026): it reads a commented-out
    annotation (the catalog flags it `dead`) and takes a method's body from the first `{` after the annotation. 3C
    always passes imports and reads as the compiler and the framework do (extract)."""
    imports = None
    defs: list[StepDef] = [] if read is None else read
    if style == "php_attribute":
        for start, end, parts in _groups(text, "#["):
            for part in parts:
                m = _PHP_ATTR.fullmatch(part)
                keyword = _php_keyword(m.group(1), imports) if m else None
                if keyword is None:
                    if rejected is not None and not _sealed_php_cut(text, end):
                        _reject(rejected, start, part, "php", imports)
                    continue
                fn = text.find("function", end)
                if fn < 0:  # as text.index did: no method follows (cut off: nothing recorded)
                    raise ValueError("substring not found")
                if not _php_public(text, end, fn):
                    if rejected is not None and not _sealed_php_cut(text, end):
                        _reject(rejected, start, part, "php", imports)
                    continue
                kind, pattern = _php_kind(_php_literal(m.group(2)))
                defs.append(_make(path, style, kind, keyword, pattern, text, start, _body(text, fn)))
    elif style == "php_docblock":
        for block in re.finditer(r"/\*\*(.*?)\*/", text, re.S):
            anns = list(re.finditer(rf"@({KEYWORDS})\s+(.+?)\s*(?:\*/|$)", block.group(1), re.M))
            looks = [c for c in re.finditer(r"^[\s*]*@(\w+)", block.group(1), re.M) if c.group(1).lower() in _STEP_NAMES]
            taken = {a.start() for a in anns}
            whole = rejected is not None and not _sealed_php_cut(text, block.end())  # a cut method records nothing
            if whole:  # a tag at a line's start that looks like a step and is none (`@then x`, `@Given`)
                rejected += [(block.start() + 3 + c.start(1) - 1, "@" + c.group(1)) for c in looks
                             if c.start(1) - 1 not in taken]
            if not anns:
                continue
            fn = text.find("function", block.end())
            if fn < 0:  # as text.index did: no method follows (cut off: nothing recorded)
                raise ValueError("substring not found")
            if not _php_public(text, block.end(), fn):  # Behat reads public methods only
                if whole:
                    rejected += [(block.start() + 3 + a.start(), "@" + a.group(1)) for a in anns]
                continue
            body = _body(text, fn)
            for a in anns:
                kind, pattern = _php_kind(a.group(2).strip())
                defs.append(_make(path, style, kind, a.group(1), pattern, text, block.start() + 3 + a.start(), body))
    elif style == "java_annotation":
        taken, stop = set(), len(text)
        try:
            for m in _JAVA_ANN.finditer(text):
                keyword = _java_keyword(m.group(1), imports)
                if keyword is None:
                    continue
                # Cucumber-JVM reads public methods only: the modifiers before the method's `{`,
                # found with every string blanked (a later annotation's "{int}" is no brace); the body is read as it
                # always was
                blank = re.sub(_JAVA_LIT, lambda s: '"' + " " * (len(s.group(0)) - 2) + '"', text[m.end():])
                if not re.search(r"\bpublic\b", blank[:blank.find("{")]):
                    continue
                stop = m.start()  # where the reader stops if this method is cut off
                body = _body(text, m.end())
                if _sealed_body_cut(text, m.end(), _JAVA_LIT):  # the method's own `{` (not one in a later
                    raise ValueError("unbalanced braces")  # annotation's pattern, {string}) must close, or it is cut
                stop = len(text)
                kind, pattern = _java_kind(_literals(m.group(2), _JAVA_LIT, lambda s: s[1:-1]))
                defs.append(_make(path, style, kind, keyword, pattern, text, m.start(), body))
                taken.add(m.start())
        finally:
            if rejected is not None:  # `@Name(` that looks like a step and is none: no import, a constant, not
                # public; before the reader stopped, and not on a cut method
                rejected += [(c.start(), re.sub(r"\s+", "", c.group(1))) for c in _JAVA_NAME.finditer(text)
                             if c.start() < stop and c.start() not in taken and _looks_step(c.group(1), "java", imports)
                             and not _cut(lambda c=c: _close_brace(text, c.end() - 1, "()"))
                             and not _sealed_body_cut(text, _close_brace(text, c.end() - 1, "()") + 1, _JAVA_LIT)]
    elif style == "java_lambda":
        # cucumber-java8: Given("pattern", (String a) -> { ... }); the lambda may start on the next line
        stop = len(text)
        try:
            for m in re.finditer(rf'^[ \t]*({KEYWORDS})\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*(?:\([^)]*\)|\w+)\s*->\s*\{{', text, re.M):
                kind, pattern = _java_kind(m.group(2))
                stop = m.start(1)  # where the reader stops if this lambda is cut off
                defs.append(_make(path, style, kind, m.group(1), pattern, text, m.start(1), (m.end() - 1, _close_brace(text, m.end() - 1))))
                stop = len(text)
        finally:
            if rejected is not None:  # a registration call at a line's start in no form the reader takes; before the
                # reader stopped, and not a call cut off before its parenthesis closes
                taken = {d.line for d in defs}
                rejected += [(c.start(1), c.group(1)) for c in re.finditer(rf"^[ \t]*({KEYWORDS})\s*\(", text, re.M)
                             if c.start(1) < stop and _line_of(text, c.start(1)) not in taken
                             and not _cut(lambda c=c: _close_brace(text, c.end() - 1, "()"))]
    elif style == "csharp_attribute":
        for start, end, parts in _groups(text, "[", verbatim=True):
            for part in parts:
                m = _CS_ATTR.fullmatch(part)
                keyword = _cs_keyword(m.group(1), imports) if m else None
                if keyword is None:
                    if rejected is not None and not _sealed_body_cut(text, end, _CS_LIT):  # not a cut method
                        _reject(rejected, start, part, "csharp", imports)
                    continue
                pattern = _literals(m.group(2), _CS_LIT, lambda s: s[2:-1].replace('""', '"') if s.startswith("@")
                                    else re.sub(r"\\(.)", lambda e: e.group(1), s[1:-1]))
                body = _body(text, end)
                if _sealed_body_cut(text, end, _CS_LIT):  # the method's own `{` (not one in a later attribute's
                    raise ValueError("unbalanced braces")  # pattern, {2}) must close, or the method is cut off
                defs.append(_make(path, style, "regex_full", keyword, pattern, text, start, body))
    else:
        raise ValueError(f"unknown style {style!r}")
    return defs


def _php_kind(pattern: str) -> tuple[str, str]:
    if len(pattern) > 1 and pattern.startswith("/") and pattern.rstrip("imsxu").endswith("/"):
        return "regex_full", pattern[1:pattern.rstrip("imsxu").rfind("/")]
    return "turnip", pattern


def _java_kind(literal: str) -> tuple[str, str]:
    """Unescape a Java string literal; Cucumber JVM treats "^...", "...$" and "/.../" as regular expressions."""
    pattern = re.sub(r"\\(.)", lambda e: e.group(1), literal)
    if len(pattern) > 1 and pattern.startswith("/") and pattern.endswith("/"):
        return "regex_full", pattern[1:-1]
    return ("regex_full" if pattern.startswith("^") or pattern.endswith("$") else "cucumber"), pattern


def php_imports(text: str) -> dict[str, str]:
    r"""The `use` imports of a PHP file, before its first class, interface, trait or enum (a trait `use` inside a class
    is no import): lower-cased alias -> the imported name as written (`use A\B;`, `use A\B as C;`, `use A\{B, C as D};`;
    function and const imports left out). Behat resolves an attribute's name through these."""
    stop = _PHP_TYPE.search(text)
    out = {}
    for m in _PHP_USE.finditer(text, 0, stop.start() if stop else len(text)):
        if m.group(2) is not None:
            for item in m.group(2).split(","):
                words = item.split()
                if words:
                    alias = words[2] if len(words) == 3 and words[1].lower() == "as" else words[0].split("\\")[-1]
                    out[alias.lower()] = f"{m.group(1)}\\{words[0]}"
        else:
            out[(m.group(3) or m.group(1).split("\\")[-1]).lower()] = m.group(1)
    return out


def _php_keyword(name: str, imports: dict[str, str] | None) -> str | None:
    """Given, When or Then when the attribute `name` resolves to Behat's class (see _PHP_ATTR), else None."""
    if name.startswith("\\"):
        return _PHP_STEPS.get(name[1:])
    if imports is None:  # Phase 1 (the catalog): a short name, read without its `use`, in any case
        return name.capitalize() if name.lower() in ("given", "when", "then") else None
    head, _, rest = name.partition("\\")
    target = imports.get(head.lower())
    return None if target is None else _PHP_STEPS.get(target + (f"\\{rest}" if rest else ""))


def _php_literal(text: str) -> str:
    """The value of one or more PHP string literals joined by `.`, each unescaped as the catalog always did (`\'` -> `'`
    only)."""
    return "".join(m.group(0)[1:-1].replace("\\'", "'") for m in re.finditer(_PHP_LIT, text))


def _php_public(code: str, after: int, fn: int) -> bool:
    """No `private` or `protected` between the annotation and its `function` keyword (Behat reads public methods only),
    read in `code` (_code_only): the word in a later attribute's pattern or in a comment is no modifier."""
    return re.search(r"\b(?:private|protected)\b", code[after:fn]) is None


def java_imports(text: str) -> dict:
    """The imports of a Java file: {"single": {simple name: qualified name}, "star": [package, ...]} (static imports
    left out)."""
    out: dict = {"single": {}, "star": []}
    for m in _JAVA_IMPORT.finditer(text):
        if m.group(2):
            out["star"].append(m.group(1))
        else:
            out["single"][m.group(1).rsplit(".", 1)[-1]] = m.group(1)
    return out


def _java_keyword(name: str, imports: dict | None) -> str | None:
    """Given, When, Then, And or But when the annotation `name` resolves to Cucumber-JVM's (see _JAVA_ANN), else None."""
    name = re.sub(r"\s+", "", name)
    if "." in name:
        return _JAVA_STEPS.get(name)
    if imports is None:  # Phase 1 (the catalog): a simple name, read without its import
        return name if name in ("Given", "When", "Then", "And", "But") else None
    qualified = imports["single"].get(name)
    if qualified is not None:
        return _JAVA_STEPS.get(qualified)
    return next((_JAVA_STEPS[f"{pkg}.{name}"] for pkg in imports["star"] if f"{pkg}.{name}" in _JAVA_STEPS), None)


def csharp_usings(text: str) -> dict:
    """The using directives of a C# file: {"alias": {alias: name}, "namespaces": [namespace, ...]} (using static left
    out)."""
    out: dict = {"alias": {}, "namespaces": []}
    for m in _CS_USING.finditer(text):
        if m.group(1):
            out["alias"][m.group(1)] = m.group(2)
        else:
            out["namespaces"].append(m.group(2))
    return out


def _cs_keyword(name: str, usings: dict | None) -> str | None:
    """Given, When, Then or StepDefinition when the attribute `name` resolves to SpecFlow's (see _CS_ATTR), else None."""
    name = re.sub(r"\s+", "", name)
    named = lambda q: _CS_STEPS.get(q) or _CS_STEPS.get(q + "Attribute")
    if usings is None:  # Phase 1 (the catalog): with or without the namespace and the suffix, without its using
        return named(name if name.startswith("TechTalk.SpecFlow.") else f"TechTalk.SpecFlow.{name}")
    if "." in name:
        head, _, rest = name.partition(".")
        return named(usings["alias"][head] + "." + rest if head in usings["alias"] else name)
    if name in usings["alias"]:
        return named(usings["alias"][name])
    return next((named(f"{ns}.{name}") for ns in usings["namespaces"] if named(f"{ns}.{name}")), None)


def _literals(text: str, lit: str, unescape) -> str:
    """The value of one or more string literals (`lit`) joined by `+` or `.`."""
    return "".join(unescape(m.group(0)) for m in re.finditer(lit, text))


def file_imports(language: str, text: str):
    """The imports a file resolves its step annotations through: php_imports, java_imports or csharp_usings of `text`;
    None for Ruby. A byte-order mark (U+FEFF, which starts 105 of gpconnect's 128 .cs files) is no text to the
    compilers, so it is dropped first, wherever the callers' concatenation put it."""
    text = text.replace("\ufeff", "")
    return {"php": php_imports, "java": java_imports, "csharp": csharp_usings}.get(language, lambda t: None)(text)


def _groups(text: str, opener: str, verbatim: bool = False):
    """(start, end, [attribute text, ...]) of every attribute group of `text` opened by `opener` ("#[" for PHP, "[" for
    C#): the group closes on its `]` at depth 0 (strings, `( )` and `[ ]` inside skipped; with `verbatim`, a C# @"..."
    string, where "" is a quote) and its attributes are split on the commas at depth 0; a group that never closes, or
    meets a `)` at depth 0, is none."""
    i = text.find(opener)
    while i >= 0:
        parts, depth, j, last = [], 0, i + len(opener), i + len(opener)
        while j < len(text):
            c = text[j]
            if c in "'\"":
                raw = verbatim and c == '"' and text[j - 1:j] == "@"
                j += 1
                while j < len(text) and text[j] != c:
                    j += 2 if text[j] == "\\" and not raw else 1
                    if raw and text.startswith('""', j):
                        j += 2
                if raw:
                    while text.startswith('""', j):
                        j = text.find('"', j + 2) if text.find('"', j + 2) >= 0 else len(text)
            elif c in "([":
                depth += 1
            elif c in ")]":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                parts.append(text[last:j])
                last = j + 1
            j += 1
        if j < len(text) and text[j] == "]":
            parts.append(text[last:j])
            yield i, j + 1, parts
            i = text.find(opener, j + 1)
        else:
            i = text.find(opener, i + len(opener))


_STEP_NAMES = {k.lower() for k in KEYWORDS.split("|")}  # what a step annotation's last name looks like, in any case
_PHP_NAME = re.compile(r"\s*(\\?[A-Za-z_]\w*(?:\\[A-Za-z_]\w*)*)")
_CS_NAME = re.compile(r"\s*((?:[A-Za-z_]\w*\s*\.\s*)*[A-Za-z_]\w*)")
_JAVA_NAME = re.compile(r"@\s*((?:[A-Za-z_$][\w$]*\s*\.\s*)*[A-Za-z_$][\w$]*)\s*\(")


def _looks_step(name: str, language: str, imports) -> bool:
    """Whether an annotation or attribute name looks like a step declaration: its last name, after an alias of `imports`
    is resolved (PHP `use … as`, C# `using X = …`) and a C# "Attribute" suffix dropped, is one of KEYWORDS in any case.
    `\\Foo\\Then`, `then` and an alias of `Foo\\Then` look like one; `Route` does not."""
    raw = name.strip()
    parts = [x for x in re.split(r"\s*[\\.]\s*", raw.lstrip("\\")) if x]
    if not parts:
        return False
    if imports is not None and language == "php" and not raw.startswith("\\") and parts[0].lower() in imports:
        parts = imports[parts[0].lower()].split("\\") + parts[1:]
    elif imports is not None and language == "csharp" and parts[0] in imports["alias"]:
        parts = imports["alias"][parts[0]].split(".") + parts[1:]
    last = parts[-1].lower()
    if language == "csharp" and last.endswith("attribute") and last != "attribute":
        last = last[:-len("attribute")]
    return last in _STEP_NAMES


def _reject(rejected, index: int, candidate: str, language: str, imports) -> None:
    """Record in `rejected` (when given) a candidate of `candidate`'s leading name that looks like a step (_looks_step)
    and that the reader does not take: (index in the text, the name as written)."""
    if rejected is None:
        return
    m = (_PHP_NAME if language == "php" else _CS_NAME).match(candidate)
    if m and _looks_step(m.group(1), language, imports):
        rejected.append((index, re.sub(r"\s+", "", m.group(1))))


def _make(path, style, kind, keyword, pattern, text, ann_index, body):
    open_index, close_index = body
    return StepDef(path, style, kind, keyword, pattern, _line_of(text, ann_index), _line_of(text, open_index),
                   _line_of(text, close_index), text[ann_index:close_index + 1])


def extract(path: str, text: str, style: str, imports: dict[str, str] | None = None,
            rejected: list | None = None, read: list | None = None) -> list[StepDef]:
    """The step definitions of `text` in `style`. `imports` (file_imports of the file the text belongs to) resolves
    annotation names as PHP, Java or C# does; None (Phase 1) reads a short name without its import. `rejected`
    (describe's unread_step_annotations counts with this very reader), when given, receives (index, name) of every
    candidate of the style that looks like a step (_looks_step) but that the reader does not take: a name that resolves
    to no framework class, a pattern that is no literal, a method that is not public, `[Given]` without a pattern, a
    Ruby step call of no readable form. It changes nothing else. A candidate whose method is cut off (§2.2, 28/09/2026,
    before any output; the text ends before the method's body closes: _php_cut, _cut of its body, _ruby_cut) is not
    recorded, whatever its form and whatever the order of the annotations, since the reader never judges a cut method
    (it stops there); and the Java readers record what they declined BEFORE the point where they stopped. `read`, when
    given, is the list the definitions are appended to as they are read, so a caller keeps those read before the reader
    stops on a cut method (ValueError). The definitions returned and the ValueError raised do not change. Since
    28/09/2026, before any output, the text is read as the compiler reads it (_code_only): a method's body starts at its
    OWN `{` (the first one outside strings and comments; Ruby: the `{` the call opens) and its `function` is the first
    one in code, never one inside a later annotation's pattern (`{string}`, `{2}`, `{int}`, "call function x"), both to
    judge whether it is cut and to read its body; a modifier, an annotation, an attribute or a doc comment inside a
    comment, a string or a heredoc is none; a PHP doc comment is read as Behat 3 reads it (_php_doc_steps, the last one
    before the method); cucumber-java8 and cucumber-ruby read Given, When, Then, And, But only (_EN_KEYWORDS)."""
    if imports is None and style != "ruby_block":  # Phase 1: as sealed (28/09/2026)
        return _extract_sealed(path, text, style, rejected, read)
    defs: list[StepDef] = [] if read is None else read
    if style == "php_attribute":
        code = _code_only(text, _PHP_CODE_LIT, hash_comments=True)
        for start, end, parts in _groups(text, "#["):
            if code[start] != "#":  # an attribute inside a comment or a string is none
                continue
            for part in parts:
                m = _PHP_ATTR.fullmatch(part)
                keyword = _php_keyword(m.group(1), imports) if m else None
                if keyword is None:
                    if rejected is not None and not _php_cut(text, end, code):
                        _reject(rejected, start, part, "php", imports)
                    continue
                fn = _php_function(code, end)
                if fn < 0:  # as text.index did: no method follows (cut off: nothing recorded)
                    raise ValueError("substring not found")
                if not _php_public(code, end, fn):
                    if rejected is not None and not _php_cut(text, end, code):
                        _reject(rejected, start, part, "php", imports)
                    continue
                kind, pattern = _php_kind(_php_literal(m.group(2)))
                defs.append(_make(path, style, kind, keyword, pattern, text, start, _body(text, fn, code)))
    elif style == "php_docblock":
        code = _code_only(text, _PHP_CODE_LIT, hash_comments=True)
        blocks = _php_docblocks(text)
        for k, (b_start, b_end, doc) in enumerate(blocks):
            anns = _php_doc_steps(doc)
            looks = [c for c in re.finditer(r"^[\s*]*@(\w+)", doc[3:-2], re.M) if c.group(1).lower() in _STEP_NAMES]
            taken = {i for i, _, _ in anns}
            written = lambda i: "@" + re.match(r"@(\w+)", doc[i:]).group(1)
            whole = rejected is not None and not _php_cut(text, b_end, code)  # a cut method records nothing
            if whole:  # a tag at a line's start that looks like a step and is none (`@And x`, `@Given`)
                rejected += [(b_start + 3 + c.start(1) - 1, "@" + c.group(1)) for c in looks
                             if 3 + c.start(1) - 1 not in taken]
            if not anns:
                continue
            # The doc comment is the method's only (shown by a real run of Behat 3.26.0) when the method's `function`
            # comes before any `;`, `{` or `}` (a property, a constant, a class took it) and before the next doc comment
            # (PHP keeps the last one): ReflectionMethod::getDocComment
            nxt = blocks[k + 1][0] if k + 1 < len(blocks) else len(text)
            member = re.compile(r"\bfunction\b|[;{}]").search(code, b_end, nxt)
            if member is None and nxt == len(text):  # as text.index did: no method follows (cut off)
                raise ValueError("substring not found")
            fn = member.start() if member is not None and member.group(0) == "function" else -1
            if fn < 0 or not _php_public(code, b_end, fn):  # Behat reads public methods only
                if whole:
                    rejected += [(b_start + i, written(i)) for i, _, _ in anns]
                continue
            body = _body(text, fn, code)
            for i, keyword, pattern in anns:
                kind, pattern = _php_kind(pattern)
                defs.append(_make(path, style, kind, keyword, pattern, text, b_start + i, body))
    elif style == "java_annotation":
        code = _code_only(text, _JAVA_CODE_LIT)
        taken, stop = set(), len(text)
        try:
            for m in _JAVA_ANN.finditer(text):
                keyword = _java_keyword(m.group(1), imports)
                if keyword is None or code[m.start()] != "@":  # an annotation in a comment is none
                    continue
                # Cucumber-JVM reads public methods only: the modifiers before the method's `{`,
                # found with every string blanked (a later annotation's "{int}" is no brace); the body is read as it
                # always was
                blank = code[m.end():]
                if not re.search(r"\bpublic\b", blank[:blank.find("{")]):
                    continue
                stop = m.start()  # where the reader stops if this method is cut off
                body = _body(text, m.end(), code)  # the method's own `{`, not one of a later pattern
                stop = len(text)
                kind, pattern = _java_kind(_literals(m.group(2), _JAVA_LIT, lambda s: s[1:-1]))
                defs.append(_make(path, style, kind, keyword, pattern, text, m.start(), body))
                taken.add(m.start())
        finally:
            if rejected is not None:  # `@Name(` that looks like a step and is none: no import, a constant, not
                # public; before the reader stopped, and not on a cut method
                rejected += [(c.start(), re.sub(r"\s+", "", c.group(1))) for c in _JAVA_NAME.finditer(text)
                             if c.start() < stop and c.start() not in taken and code[c.start()] == "@"
                             and _looks_step(c.group(1), "java", imports)
                             and not _cut(lambda c=c: _close_brace(text, c.end() - 1, "()"))
                             and not _body_cut(text, _close_brace(text, c.end() - 1, "()") + 1, code)]
    elif style == "java_lambda":
        # cucumber-java8: Given("pattern", (String a) -> { ... }); the lambda may start on the next line
        code = _code_only(text, _JAVA_CODE_LIT)
        stop = len(text)
        try:
            for m in re.finditer(rf'^[ \t]*({_EN_KEYWORDS})\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*(?:\([^)]*\)|\w+)\s*->\s*\{{', text, re.M):
                if code[m.start(1)] == " ":  # a call inside a comment or a text block is none
                    continue
                kind, pattern = _java_kind(m.group(2))
                stop = m.start(1)  # where the reader stops if this lambda is cut off
                defs.append(_make(path, style, kind, m.group(1), pattern, text, m.start(1), (m.end() - 1, _close_brace(text, m.end() - 1))))
                stop = len(text)
        finally:
            if rejected is not None:  # a registration call at a line's start in no form the reader takes; before the
                # reader stopped, and not a call cut off before its parenthesis closes
                taken = {d.line for d in defs}
                rejected += [(c.start(1), c.group(1)) for c in re.finditer(rf"^[ \t]*({KEYWORDS})\s*\(", text, re.M)
                             if c.start(1) < stop and _line_of(text, c.start(1)) not in taken and code[c.start(1)] != " "
                             and not _cut(lambda c=c: _close_brace(text, c.end() - 1, "()"))]
    elif style == "csharp_attribute":
        code = _code_only(text, _CS_CODE_LIT)
        for start, end, parts in _groups(text, "[", verbatim=True):
            if code[start] != "[":  # an attribute inside a comment or a string is none
                continue
            for part in parts:
                m = _CS_ATTR.fullmatch(part)
                keyword = _cs_keyword(m.group(1), imports) if m else None
                if keyword is None:
                    if rejected is not None and not _body_cut(text, end, code):  # not a cut method
                        _reject(rejected, start, part, "csharp", imports)
                    continue
                pattern = _literals(m.group(2), _CS_LIT, lambda s: s[2:-1].replace('""', '"') if s.startswith("@")
                                    else re.sub(r"\\(.)", lambda e: e.group(1), s[1:-1]))
                body = _body(text, end, code)  # the method's own `{`, not one of a later pattern ({2})
                defs.append(_make(path, style, "regex_full", keyword, pattern, text, start, body))
    elif style == "ruby_block":
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = re.match(rf"^(\s*)({_EN_KEYWORDS})\s*\(?\s*(?:{_RUBY_REGEX}|{_RUBY_STRING}|(?P<var>[a-z_]\w*))\s*\)?\s*(do|\{{)",
                         line)
            step = re.match(rf"^\s*({KEYWORDS})\b", line)
            here = (sum(len(x) + 1 for x in lines[:i]) + step.start(1), step.group(1)) if step else None
            if not m:  # a step call of no form the reader takes (`Given(:sym) do`), its block not cut off
                if rejected is not None and here and not _ruby_cut(lines, text, i):
                    rejected.append(here)
                continue
            indent, keyword, opener = m.group(1), m.group(2), m.group(m.lastindex)
            opens = sum(len(x) + 1 for x in lines[:i]) + m.start(m.lastindex)  # the `{` the call opens
            if m.group("var"):  # pattern held in a variable assigned earlier in the file
                m = re.search(rf"^\s*{re.escape(m.group('var'))}\s*=\s*{_RUBY_REGEX}\s*$", text, re.M)
                if m is None:
                    if rejected is not None and here and not _ruby_cut(lines, text, i):
                        rejected.append(here)
                    continue
            sq, dq = m.groupdict().get("sq"), m.groupdict().get("dq")  # a pattern variable is a Regexp
            if sq is not None or dq is not None:  # a Cucumber expression
                kind = "cucumber"
                pattern = re.sub(r"\\(['\\])", r"\1", sq) if sq is not None else re.sub(r'\\(["\\#])', r"\1", dq)
            else:
                kind = "regex"
                pattern = m.group("slash").replace("\\/", "/") if m.group("slash") is not None else m.group("brace")
            if opener == "do":
                end = next((k for k in range(i + 1, len(lines)) if re.match(rf"^{re.escape(indent)}end\b", lines[k])), None)
                if end is None:  # the block never ends: cut off, neither read nor recorded
                    continue
                code = "\n".join(lines[i:end + 1])
                defs.append(StepDef(path, style, kind, keyword, pattern, i + 1, i + 1, end + 1, code))
            else:
                start = opens  # the block's own `{`, not one of the pattern ({int}, {2})
                close = _close_brace(text, start)
                defs.append(StepDef(path, style, kind, keyword, pattern, i + 1, i + 1, _line_of(text, close), text[start:close + 1]))
    else:
        raise ValueError(f"unknown style {style!r}")
    return defs
