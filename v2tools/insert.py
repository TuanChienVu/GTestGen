"""The one insertion rule of Phase 3, used for the gold answer, the stubs and every condition. Code to score is inserted
into a file of the ER tree (ertree.py):
- import lines join the file's import block, duplicates dropped (PHP: also a name already imported as the same
  class, whatever the form: _php_new_imports);
- a method whose key (PHP: name, case-insensitive; Java, C#: name and parameter count) equals that of a method of the
  class REPLACES it when that method is the shell of one of the record's preexisting CREATE items, or a FUTURE method of
  the class (not in V(r)), keeping the old step annotations that the new one lacks; any other clash is a `collision`
  (the item is not inserted and the run fails BV and ER); a method with a new key is added before the class's closing
  brace (also over a future method of an ancestor: a normal override); other members are added;
- a Java cucumber-java8 lambda step is added before the closing brace of the constructor that holds the file's lambda
  steps (else the first no-argument constructor, else the first one; a no-argument constructor is created if none); a
  lambda is such a step only as a top-level statement of the block or of a constructor body, where Cucumber-java8
  registers it; anywhere else it stays in the code around it (`_top_level`);
- Ruby step blocks are appended to the file;
- code that does not parse as class members is inserted verbatim before the class's closing brace, so that a syntax
  error is scored, never repaired; a class or other type declared in a block (a C# struct or record struct included) is
  such "raw" code too, and so is a Java field whose initializer holds an anonymous class body that declares a method;
  evaluate.py fails condition 1 on any raw item."""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

from .ertree import parser
from .extract import PHP_CLOSE, PHP_OPEN, php_open_tag  # noqa: F401 (§1.4: the extractor's, per block)
from .outline import _visibility
from .stepdefs import _EN_KEYWORDS, _PHP_ATTR, _PHP_LIT, _PHP_USE, _close_brace, _groups, _php_keyword, php_imports
from .stepdefs import extract as extract_defs

CLASS_NODES = {"class_declaration", "trait_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
TYPE_NODES = CLASS_NODES | {"struct_declaration", "record_struct_declaration"}  # declared in a block: raw
BODY_NODES = {"declaration_list", "class_body", "enum_body", "interface_body"}
METHOD_NODES = {"method_declaration", "constructor_declaration"}
PARAM_NODES = {"simple_parameter", "variadic_parameter", "property_promotion_parameter", "formal_parameter",
               "spread_parameter", "parameter"}
IMPORT_NODES = {"namespace_use_declaration", "import_declaration", "using_directive"}
HEAD_NODES = {"namespace_definition", "package_declaration", "php_tag"}
IMPORT = {"php": re.compile(r"^\s*use\s+(?:function\s+|const\s+)?\\?[\w\\]+(?:\\\{[^}]*\})?(?:\s+as\s+\w+)?\s*;\s*$"),
          "java": re.compile(r"^\s*import\s+(?:static\s+)?[\w.]+(?:\.\*)?\s*;\s*$"),
          "csharp": re.compile(r"^\s*using\s+(?:static\s+)?[\w.]+(?:\s*=\s*[\w.<>, ]+)?\s*;\s*$"),
          "ruby": re.compile(r"^\s*require(?:_relative)?\s*\(?\s*['\"][^'\"]+['\"]\s*\)?\s*$")}
LAMBDA = re.compile(rf'^[ \t]*(?:{_EN_KEYWORDS})\(\s*"(?:[^"\\]|\\.)*"\s*,\s*(\([^)]*\)|\w+)\s*->\s*\{{', re.M)
class _PhpStepLine:
    """STEP_ANNOTATION["php"]: a line that is a PHP step annotation (for the kept-annotation rule): a docblock line `*
    @Kw p`, or one attribute group alone on its line holding a step attribute of the reader (stepdefs; §2.2: a short
    name in any case or `\\Behat\\Step\\…`, whitespace, `pattern:`, several attributes). One line is read without the
    file's imports (an alias is not recognised here: a documented limit)."""
    DOC = re.compile(r"^\s*\*\s*@(?i:given|when|then)\s+\S.*$")  # Behat 3's tags (stepdefs._php_doc_steps)

    def match(self, line: str):
        if self.DOC.match(line):
            return True
        s = line.strip()
        g = next(_groups(s, "#["), None) if s.startswith("#[") else None
        if g is None or g[0] != 0 or g[1] != len(s):
            return None
        return any((m := _PHP_ATTR.fullmatch(part)) and _php_keyword(m.group(1), None) for part in g[2]) or None


STEP_ANNOTATION = {"php": _PhpStepLine(),  # the reader's forms (stepdefs, §2.2)
                   "java": re.compile(r"^\s*@\s*(?:io\s*\.\s*cucumber\s*\.\s*java\s*\.\s*en\s*\.\s*)?"
                                      r"(?:Given|When|Then|And|But)\s*\(.*\)\s*$"),
                   "csharp": re.compile(r"^\s*\[\s*(?:TechTalk\s*\.\s*SpecFlow\s*\.\s*)?(?:Given|When|Then|StepDefinition)"
                                        r"(?:Attribute)?\s*\(.*\)\s*\]\s*$")}
DECOR = re.compile(r"^\s*(?:#\[|@|\[|\*|/\*\*|//)")  # annotation, attribute, docblock or comment line


# ----- parsing -------------------------------------------------------------------------------------------------------

def first_class(node):
    """The first class-like declaration in pre-order (the file's top-level class)."""
    if node.type in CLASS_NODES:
        return node
    for c in node.children:
        found = first_class(c)
        if found is not None:
            return found
    return None


def class_body(root):
    cls = first_class(root)
    return None if cls is None else next((c for c in cls.children if c.type in BODY_NODES), None)


def arity(node) -> int:
    params = node.child_by_field_name("parameters")
    return 0 if params is None else sum(c.type in PARAM_NODES for c in params.named_children)


def node_name(src: bytes, node) -> str | None:
    n = node.child_by_field_name("name")
    return None if n is None else src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def span_start(src: bytes, node) -> int:
    """Start of a member including a docblock right before it (only whitespace in between)."""
    prev = node.prev_named_sibling
    if prev is not None and prev.type in ("comment", "block_comment") and src[prev.start_byte:prev.start_byte + 3] == b"/**" \
            and not src[prev.end_byte:node.start_byte].strip():
        return prev.start_byte
    return node.start_byte


def line_start(src: bytes, i: int) -> int:
    return src.rfind(b"\n", 0, i) + 1


def indent_at(src: bytes, i: int) -> str:
    return src[line_start(src, i):i].decode("utf-8", errors="replace")


def indented(text: str, indent: str) -> str:
    return "\n".join(indent + l if l.strip() else "" for l in text.split("\n"))


# ----- the code to insert --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    kind: str  # "method" | "member" | "lambda" | "block" (Ruby) | "raw" (does not parse as members)
    name: str | None
    arity: int | None
    text: str  # dedented source, docblock and annotations included


def _code_lines(text: str) -> tuple[dict[int, tuple[int, int | None]], dict[int, int]]:
    """({line start in code: (brace depth, index of the innermost open `{` or None)}, {index of a depth-0 `{`: start of
    its header}), scanning `text` as stepdefs._close_brace does (strings, character literals, // and /* */ comments
    skipped; a line start inside one is not in code). A header starts after the last `;` or `}` at brace and
    parenthesis depth 0 (or at 0): an annotation's `{…}` array or a lambda's body does not end it."""
    lines, head, stack, seg, parens, quote, escaped, j = {}, {}, [], 0, 0, None, False, 0
    while j < len(text):
        c = text[j]
        if quote is None and (j == 0 or text[j - 1] == "\n"):
            lines[j] = (len(stack), stack[-1] if stack else None)
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
            continue
        elif text.startswith("/*", j):
            j = text.find("*/", j + 2)
            if j < 0:
                break
            j += 2
            continue
        elif c in "\"'":
            quote = c
        elif c == "{":
            if not stack:
                head[j] = seg
            stack.append(j)
        elif c == "}" and stack:
            stack.pop()
            seg = j + 1 if not stack and not parens else seg
        elif c == "(" and not stack:
            parens += 1
        elif c == ")" and not stack:
            parens = max(parens - 1, 0)
        elif c == ";" and not stack and not parens:
            seg = j + 1
        j += 1
    return lines, head


def _is_constructor(header: str) -> bool:
    """`header {}` is one Java constructor declaration (tree-sitter), not a method, initializer or anonymous class."""
    src = ("class __X {\n" + header + "{}\n}\n").encode()
    root = parser("java").parse(src).root_node
    body = class_body(root)
    kids = [c for c in body.named_children if c.type not in ("comment", "line_comment", "block_comment")] \
        if body is not None and not root.has_error else []
    return len(kids) == 1 and kids[0].type == "constructor_declaration"


def _top_level(text: str, start: int, lines: dict, head: dict) -> bool:
    """A LAMBDA match at `start` (a line start; 25/09/2026) is a step only as a top-level statement of the block (brace
    depth 0) or of a constructor body (depth 1, inside a `{` whose header, from the end of the previous depth-0
    statement or member, is a constructor: `_is_constructor`); never in a string or a comment, a `static {}` or instance
    initializer, a method body, an anonymous class or a nested block."""
    depth, brace = lines.get(start, (None, None))
    return depth == 0 or depth == 1 and _is_constructor(text[head[brace]:brace])


def _take_lambdas(text: str) -> tuple[str, list[Item]]:
    items, keep, pos = [], [], 0
    lines = head = None
    for m in LAMBDA.finditer(text):
        if lines is None:
            lines, head = _code_lines(text)
        if m.start() < pos or not _top_level(text, m.start(), lines, head):
            continue
        try:
            close = _close_brace(text, m.end() - 1)
        except ValueError:
            break
        tail = re.match(r"\s*\)\s*;", text[close + 1:])
        if tail is None:
            continue
        end = close + 1 + tail.end()
        params = m.group(1).strip("()").strip()
        items.append(Item("lambda", None, len([p for p in params.split(",") if p.strip()]), textwrap.dedent(text[m.start():end])))
        keep.append(text[pos:m.start()])
        pos = end
    keep.append(text[pos:])
    return "".join(keep), items


def _anonymous_class(node) -> bool:
    """A Java anonymous class body (`new T(…) { … }`) that declares a method, in a field's initializer: that method
    never binds as a step of the class, so the field is raw. A body without a method (the Jackson idiom `new
    TypeReference<…>() {}`, fields only) leaves the field an ordinary member."""
    if node.type == "object_creation_expression" and any(c.type == "class_body" and any(
            m.type == "method_declaration" for m in c.named_children) for c in node.children):
        return True
    return any(_anonymous_class(c) for c in node.children)


def _members(language: str, code: str) -> list[Item]:
    prefix = "<?php\nclass __X {\n" if language == "php" else "class __X {\n"
    src = (prefix + code + "\n}\n").encode()
    root = parser(language).parse(src).root_node
    body = class_body(root)
    if root.has_error or body is None:
        return [Item("raw", None, None, code)]
    items = []
    for ch in body.named_children:
        if ch.type in ("comment", "block_comment"):
            continue
        text = textwrap.dedent(src[line_start(src, span_start(src, ch)):ch.end_byte].decode("utf-8", errors="replace"))
        if ch.type in METHOD_NODES:
            items.append(Item("method", node_name(src, ch), arity(ch), text))
        elif ch.type in TYPE_NODES or ch.type == "field_declaration" and _anonymous_class(ch):  # raw code
            items.append(Item("raw", None, None, text))  # a type declared in the output (e.g. the whole class again)
        else:
            items.append(Item("member", None, None, text))
    return items


def split_code(language: str, code: str) -> tuple[list[str], list[Item]]:
    """(leading import lines, items) of one extracted block (PHP: without its `<?php` / `?>` tags, php_open_tag, which
    extract.by_path already applied per block; again here for code that does not come through it)."""
    if language == "php":
        code = php_open_tag(code)[0]
    lines = code.split("\n")
    k, imports = 0, []
    while k < len(lines) and (not lines[k].strip() or IMPORT[language].match(lines[k])):
        if lines[k].strip():
            imports.append(lines[k].strip())
        k += 1
    rest = textwrap.dedent("\n".join(lines[k:])).strip("\n")
    if not rest.strip():
        return imports, []
    if language == "ruby":
        lines = rest.split("\n")
        try:
            defs = sorted(extract_defs("", rest, "ruby_block"), key=lambda d: d.line)
        except ValueError:  # a block that never closes (an output cut off): all of it is raw
            return imports, [Item("raw", None, None, rest)]
        items, covered = [], set()
        for d in defs:
            if d.line in covered:
                continue
            covered |= set(range(d.line, d.body_end + 1))
            items.append(Item("block", None, None, "\n".join(lines[d.line - 1:d.body_end])))
        left = "\n".join(l for i, l in enumerate(lines, start=1) if i not in covered)
        return imports, items + ([Item("raw", None, None, left.strip("\n"))] if left.strip() else [])
    items: list[Item] = []
    if language == "java":
        rest, items = _take_lambdas(rest)
    if rest.strip():
        items += _members(language, textwrap.dedent(rest).strip("\n"))
    return imports, items


# ----- insertion -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Inserted:
    text: str
    replaced: tuple[str, ...]  # names of replaced methods
    added: int
    kept_annotations: int  # old step annotations carried over by a replacement
    raw: int  # items inserted verbatim because they did not parse
    collisions: tuple[str, ...]  # names of methods that clash with a method that may not be replaced
    items: tuple[tuple[str, object, str], ...] = ()  # (kind, method key or None, final text) of every inserted item


def method_key(language: str, name: str, n_params: int | None):
    return name.casefold() if language == "php" else (name, n_params)


def class_methods(language: str, text: str, private: bool = True) -> list[tuple[str, object, str]]:
    """(name, method key, text) of every method and constructor of the file's class, in order, the private ones only
    when `private`; the text is dedented, its docblock and annotations included (the FUTURE methods of
    evaluate.future_methods, where an ancestor's private methods are left out)."""
    src = text.encode()
    body = class_body(parser(language).parse(src).root_node)
    out = []
    for ch in body.named_children if body is not None else []:
        if ch.type in METHOD_NODES and node_name(src, ch) is not None \
                and (private or _visibility(src, ch, language) != "private"):
            code = src[line_start(src, span_start(src, ch)):ch.end_byte].decode("utf-8", errors="replace")
            out.append((node_name(src, ch), method_key(language, node_name(src, ch), arity(ch)), textwrap.dedent(code)))
    return out


def duplicate_keys(language: str, code: str) -> list[str]:
    """Method keys that one output declares twice for the same file: a collision, whether or not the key is a
    replaceable shell."""
    keys = [method_key(language, it.name, it.arity) for it in split_code(language, code)[1] if it.kind == "method"]
    return sorted({str(k) for k in keys if keys.count(k) > 1})


def is_constructor(language: str, text: str) -> bool:
    """A constructor (Java/C# constructor_declaration, PHP __construct): replaced like a method when it is a shell or a
    FUTURE method, never expanded."""
    src = (("<?php\nclass __X {\n" if language == "php" else "class __X {\n") + text + "\n}\n").encode()
    body = class_body(parser(language).parse(src).root_node)
    m = next((c for c in body.named_children if c.type in METHOD_NODES), None) if body is not None else None
    return m is not None and (m.type == "constructor_declaration"
                              or (node_name(src, m) or "").casefold() == "__construct")


def shell_keys(language: str, codes: list[str]) -> set:
    """Keys of the methods declared in `codes` (the preexisting CREATE items of one target file)."""
    keys = set()
    for code in codes:
        for it in split_code(language, code)[1]:
            if it.kind == "method":
                keys.add(method_key(language, it.name, it.arity))
    return keys


def _norm(s: str) -> str:
    return " ".join(s.split())


PHP_STEP_DOC = re.compile(r"(?:^[ \t]*\*|/\*\*)[ \t]*@((?i:given|when|then))[ \t]+(.+?)[ \t]*(?=\*/|$)", re.M)


def _body_offset(language: str, text: str) -> int | None:
    """Index in `text` (one method) where its body starts (tree-sitter `body` field), None when it has none."""
    prefix = "<?php\nclass __X {\n" if language == "php" else "class __X {\n"
    src = (prefix + text + "\n}\n").encode()
    body = class_body(parser(language).parse(src).root_node)
    m = next((c for c in body.named_children if c.type in METHOD_NODES), None) if body is not None else None
    b = m.child_by_field_name("body") if m is not None else None
    return None if b is None else len(src[:b.start_byte].decode("utf-8", errors="replace")) - len(prefix)


def _php_step_groups(head: str) -> list[tuple[int, int, list[tuple[int, int, str]]]]:
    """(start, end, [(start, end, pattern) of each step attribute]) of every attribute group of `head` that holds a
    step attribute of the reader (stepdefs._groups, _PHP_ATTR; names without imports, as on one line)."""
    out = []
    for gs, ge, parts in _groups(head, "#["):
        at, steps = gs + 2, []
        for part in parts:
            m = _PHP_ATTR.fullmatch(part)
            if m and _php_keyword(m.group(1), None):
                lead = len(part) - len(part.lstrip())
                steps.append((at + lead, at + len(part.rstrip()), _unescaped(m.group(2))))
            at += len(part) + 1  # the comma after it
        if steps:
            out.append((gs, ge, steps))
    return out


def _unescaped(literals: str) -> str:
    """The value of one or more PHP string literals joined by `.`, each with backslash + its quote and backslash +
    backslash undone (as php_steps always read them)."""
    return "".join(re.sub(r"\\([\\" + m.group(0)[0] + "])", r"\1", m.group(0)[1:-1])
                   for m in re.finditer(_PHP_LIT, literals))


def php_steps(text: str) -> list[tuple[int, int, str, str]]:
    """(start, end, source, pattern) of every PHP step annotation of `text` (one method) before its body: each step
    attribute of the reader (§2.2: several may share one `#[...]`; a group of one is its whole `#[...]`, the span and
    source; one of several is that attribute alone, which php_kept drops by rewriting the group), a docblock line `* @Kw
    p` (the whole line) or the `@Kw p` of a one-line `/** @Kw p */`. The pattern is what Behat registers: form and
    keyword do not matter (26/09/2026)."""
    end = _body_offset("php", text)
    head = text if end is None else text[:end]
    out = []
    for gs, ge, steps in _php_step_groups(head):
        if len(steps) == 1 and _groups_parts(head, gs) == 1:
            out.append((gs, ge, head[gs:ge], steps[0][2]))
        else:
            out += [(a, b, head[a:b], pat) for a, b, pat in steps]
    for m in PHP_STEP_DOC.finditer(head):
        if head.startswith("/**", m.start()):
            at = head.index("@", m.start())
            out.append((at, m.end(2), head[at:m.end(2)], m.group(2)))
        else:
            eol = head.find("\n", m.start())
            out.append((m.start(), len(head) if eol < 0 else eol + 1, head[m.start():eol if eol >= 0 else None],
                        m.group(2)))
    return sorted(out)


def _groups_parts(head: str, start: int) -> int:
    """The number of attributes of the group of `head` that opens at `start`."""
    return next(len(parts) for gs, _, parts in _groups(head[start:], "#[") if gs == 0)


def _drop(text: str, spans: list[tuple[int, int]]) -> str:
    """`text` without the spans (and the blanks right after one), and without a line that one of them leaves blank."""
    for a, b in sorted(set(spans), reverse=True):
        inline = text[b - 1:b] != "\n"  # an inline attribute: the blanks after it go too; a whole line keeps the next
        text = text[:a] + (text[b:].lstrip(" \t") if inline else text[b:])
        ls, le = text.rfind("\n", 0, a) + 1, text.find("\n", a)
        le = len(text) if le < 0 else le
        if not text[ls:le].strip():
            text = text[:ls] + text[le + 1:]
    return text


def php_kept(new: str, old: str) -> tuple[str, list[str]]:
    """The kept-annotation rule for PHP, by pattern (26/09/2026), since Behat 4.0.0-alpha1 registers attributes only and
    refuses a pattern registered twice: (new method text, HEAD annotation lines to keep) when the PHP method `new`
    replaces `old`. Each OLD step annotation line (the ER tree has dropped the CREATE ones, so these are other
    scenarios' steps) is matched with the output's annotations by PATTERN (php_steps): (a) the output does not write it:
    the HEAD line is kept, as before; the output writes it with the same line: nothing changes; else the HEAD line is
    kept as HEAD wrote it and the output's annotations of that pattern are dropped. (b) An output annotation whose
    pattern is no OLD one (a CREATE step) is never changed. So no pattern is registered twice on the method."""
    heads, keep, drop = php_steps(new), [], []
    for line in old.split("\n"):
        if not STEP_ANNOTATION["php"].match(line):
            continue
        mine = php_steps(line.strip() + "\npublic function __x() {}")
        if not mine:  # not a pattern Behat reads from one line: compare the line itself, as before
            if all(_norm(line) != _norm(l) for l in new.split("\n")):
                keep.append(line.strip())
            continue
        patterns = {m[3] for m in mine}  # every pattern of a HEAD group, not the first
        same = [h for h in heads if h[3] in patterns]
        if not same or not any(_norm(h[2]) == _norm(line) for h in same):
            keep.append(line.strip())
        drop += [(a, b) for a, b, src, _ in same if _norm(src) != _norm(line)]
    return _drop_parts(new, drop), keep


def _drop_parts(text: str, spans: list[tuple[int, int]]) -> str:
    """_drop, where a span that is one attribute of a group of several rewrites that group without it: `#[` + the kept
    attributes joined by ", " + `]`, or the whole group dropped when none is kept."""
    end = _body_offset("php", text)
    head = text if end is None else text[:end]
    spans, edits = set(spans), []
    for gs, ge, parts in _groups(head, "#["):
        if len(parts) < 2:
            continue
        at, kept, gone = gs + 2, [], []
        for part in parts:
            a, b = at + len(part) - len(part.lstrip()), at + len(part.rstrip())
            (gone if (a, b) in spans else kept).append((a, b))
            at += len(part) + 1
        if gone:
            spans -= set(gone)
            edits.append((gs, ge, "#[" + ", ".join(text[a:b] for a, b in kept) + "]" if kept else None))
    for gs, ge, new in sorted(edits, reverse=True):  # a group rewritten in place; one left empty goes as a span
        if new is None:
            spans.add((gs, ge))
        else:
            shift = len(new) - (ge - gs)
            text = text[:gs] + new + text[ge:]
            spans = {(a + shift, b + shift) if a >= ge else (a, b) for a, b in spans}
    return _drop(text, sorted(spans))


DECOR_NODES = {"attribute_list", "annotation", "marker_annotation", "comment", "line_comment", "block_comment"}


def _head_line(language: str, text: str) -> int | None:
    """The 0-based line of `text` (one method, docblock and annotations included) before which kept annotation lines go:
    the line of the method's first token after its attributes/annotations (tree-sitter; PHP `attribute_list`, Java
    `annotation`/`marker_annotation`, C# `attribute_list`, comments skipped) when only whitespace precedes that token on
    its line, else the method's first line (before its whole attribute list); None when `text` holds no method."""
    prefix = "<?php\nclass __X {\n" if language == "php" else "class __X {\n"
    src = (prefix + text + "\n}\n").encode()
    body = class_body(parser(language).parse(src).root_node)
    m = next((c for c in body.named_children if c.type in METHOD_NODES), None) if body is not None else None
    if m is None:
        return None
    stack, tok = [m], None
    while stack and tok is None:
        n = stack.pop()
        if n.type in DECOR_NODES:
            continue
        if n.child_count == 0:
            tok = n
        stack.extend(reversed(n.children))
    at = lambda i: len(src[:i].decode("utf-8", errors="replace")) - len(prefix)  # char index in `text`
    i = at(tok.start_byte) if tok is not None else at(m.start_byte)
    if tok is None or text[text.rfind("\n", 0, i) + 1:i].strip():
        i = at(m.start_byte)
    return text.count("\n", 0, i)


def with_annotations(language: str, text: str, missing: list[str]) -> str:
    """`text` plus the old step annotation lines it lacks: docblock lines go inside its docblock (one is created when it
    has none), attribute/annotation lines go on their own lines before the head of the method (`_head_line`: never
    inside a multi-line attribute, never between the signature and the body)."""
    lines = text.split("\n")
    doc = [l for l in missing if l.lstrip().startswith("*")]
    other = [l for l in missing if not l.lstrip().startswith("*")]
    head = _head_line(language, text) if other else None
    if head is not None:
        lines[head:head] = other
    elif other:
        in_doc, s = False, 0
        for s, l in enumerate(lines):
            if l.lstrip().startswith("/**"):
                in_doc = "*/" not in l
                continue
            if in_doc:
                in_doc = "*/" not in l
                continue
            if l.strip() and not DECOR.match(l):
                break
        lines[s:s] = other
    if doc:
        doc = [" " + l.strip() for l in doc]
        if lines and lines[0].lstrip().startswith("/**"):
            j = next(i for i, l in enumerate(lines) if "*/" in l)
            if j == 0:
                inner = lines[0].strip()[3:].rsplit("*/", 1)[0].strip()
                lines[0:1] = ["/**"] + ([f" * {inner}"] if inner else []) + doc + [" */"]
            else:
                lines[j:j] = doc
        else:
            lines[0:0] = ["/**"] + doc + [" */"]
    return "\n".join(lines)


def _put(src: bytes, close: int, text: str, indent: str) -> bytes:
    """Insert `text` (indented) as new lines right before the brace at `close`."""
    ls = line_start(src, close)
    block = indented(text, indent) + "\n"
    if src[ls:close].strip():  # the brace shares its line with code: open a new line
        return src[:close] + b"\n" + block.encode() + src[close:]
    prev = src[line_start(src, ls - 1):ls - 1] if ls > 0 else b""  # the line right before the brace's line
    sep = b"" if not prev.strip() or prev.rstrip().endswith(b"{") else b"\n"
    return src[:ls] + sep + block.encode() + src[ls:]


def _member_indent(src: bytes, body) -> str:
    first = next((c for c in body.named_children if c.type not in ("comment", "block_comment")), None)
    return indent_at(src, span_start(src, first)) if first is not None else indent_at(src, body.end_byte - 1) + "    "


def _add_member(language: str, src: bytes, body, text: str) -> bytes:
    return _put(src, body.end_byte - 1, text, _member_indent(src, body))


def _add_lambda(language: str, src: bytes, body, item: Item) -> bytes:
    ctors = [c for c in body.named_children if c.type == "constructor_declaration"]
    has = [c for c in ctors if LAMBDA.search(src[c.start_byte:c.end_byte].decode("utf-8", errors="replace"))]
    ctor = (has or [c for c in ctors if arity(c) == 0] or ctors or [None])[0]
    if ctor is None:
        name = node_name(src, body.parent) or "Steps"
        return _add_member(language, src, body, f"public {name}() {{\n{indented(item.text, '    ')}\n}}")
    cbody = ctor.child_by_field_name("body")
    first = next(iter(cbody.named_children), None)
    indent = indent_at(src, first.start_byte) if first is not None else indent_at(src, span_start(src, ctor)) + "    "
    return _put(src, cbody.end_byte - 1, item.text, indent)


def _merge_imports(language: str, src: bytes, imports: list[str]) -> bytes:
    if not imports:
        return src
    if language == "ruby":
        text = src.decode("utf-8", errors="replace")
        have = {_norm(l) for l in text.split("\n") if IMPORT["ruby"].match(l)}
        new = [i for i in dict.fromkeys(imports) if _norm(i) not in have]
        return ("\n".join(new) + "\n" + text).encode() if new else src
    root = parser(language).parse(src).root_node
    nodes, heads = [], []

    def walk(n):
        for c in n.children:
            if c.type in IMPORT_NODES:
                nodes.append(c)
            elif c.type in HEAD_NODES:
                heads.append(c)
                if c.type == "namespace_definition" and c.child_by_field_name("body") is not None:
                    walk(c.child_by_field_name("body"))
            elif c.type in ("namespace_declaration", "file_scoped_namespace_declaration", "declaration_list"):
                heads.append(c) if c.type == "file_scoped_namespace_declaration" else None
                walk(c)
    walk(root)
    have = {_norm(src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")) for n in nodes}
    new = [i for i in dict.fromkeys(imports) if _norm(i) not in have]
    if language == "php":
        new = _php_new_imports(src.decode("utf-8", errors="replace"), new)
    if not new:
        return src
    anchor = nodes[-1] if nodes else (heads[-1] if heads else None)
    at = 0 if anchor is None else anchor.end_byte
    if anchor is not None and anchor.type == "file_scoped_namespace_declaration":
        at = anchor.start_byte + src[anchor.start_byte:anchor.end_byte].find(b";") + 1
    lead = b"\n" if nodes else b"\n\n"
    return src[:at] + lead + "\n".join(new).encode() + (b"" if nodes else b"\n") + src[at:]


def _php_new_imports(text: str, imports: list[str]) -> list[str]:
    r"""The PHP `use` lines of `imports` without the names that `text`, or an earlier line of `imports`, already
    imports. PHP refuses a second `use` of an alias in use ("Cannot use … because the name is already in use": a Fatal,
    for Behat 4 an ERROR of the output), whatever the form: a group `use A\{B, C};`, a leading `\`, blanks before `;`.
    So a name whose lower-cased alias (php_imports) is already imported as the same class (names compared
    case-insensitively, as PHP does) is dropped: a line all of whose names are dropped goes, a group keeps its other
    names as written. A name whose alias is imported as ANOTHER class is kept (the output's own clash: scored, never
    repaired); a function or const import is compared as text only."""
    have = {a: t.lower() for a, t in php_imports(text).items()}
    out = []
    for line in imports:
        pairs = {a: t.lower() for a, t in php_imports(line).items()}
        dup = {a for a, t in pairs.items() if have.get(a) == t}
        m = _PHP_USE.match(line)
        if dup and m is not None and m.group(2) is not None and len(dup) < len(pairs):
            items = [x.strip() for x in m.group(2).split(",") if x.strip()]
            kept = [x for x in items if (x.split()[2] if len(x.split()) == 3 and x.split()[1].lower() == "as"
                                         else x.split()[0].split("\\")[-1]).lower() not in dup]
            line = f"{line[:m.start(2)]}{', '.join(kept)}{line[m.end(2):]}"
        if not dup or len(dup) < len(pairs):
            out.append(line)
        have.update(pairs)
    return out


def _constructors_first(language: str, src: bytes, items: list[Item], replaceable: set) -> list[Item]:
    """For fineract's standard cucumber-java8 form: in a Java block holding lambda steps, the output's constructors go
    BEFORE the lambdas when the class declares no constructor, or when a constructor of the class with the same key may
    be replaced (a shell or a FUTURE method): _add_lambda then puts the lambdas into the output's constructor instead of
    creating `X() {}` that the output's constructor would clash with. Otherwise (the class has a constructor the output
    may not replace) the order stays: the lambdas go into the class's constructor and an output constructor with its key
    is a collision."""
    if language != "java" or not any(it.kind == "lambda" for it in items):
        return items
    src_body = class_body(parser(language).parse(src).root_node)
    have = {method_key(language, node_name(src, c), arity(c)) for c in (src_body.named_children if src_body else ())
            if c.type == "constructor_declaration" and node_name(src, c) is not None}
    first = [i for i, it in enumerate(items) if it.kind == "method" and is_constructor(language, it.text)
             and (not have or method_key(language, it.name, it.arity) in have & replaceable)]
    return [items[i] for i in first] + [it for i, it in enumerate(items) if i not in first]


def insert(language: str, text: str, code: str, shells: set = frozenset(), future: set = frozenset()) -> Inserted:
    """`text` (a file of the ER tree) with one extracted block inserted by the insertion rule; `shells` = shell_keys()
    of the record's preexisting items in this file and `future` = the keys of the FUTURE methods of this file's class
    (evaluate.future_keys): the only methods that may be replaced."""
    imports, items = split_code(language, code)
    src = _merge_imports(language, text.encode(), imports)
    replaced, added, kept, raw, clash, done = [], 0, 0, 0, [], []
    items = _constructors_first(language, src, items, set(shells) | set(future))
    for it in items:
        key = method_key(language, it.name, it.arity) if it.kind == "method" else None
        if language == "ruby" or class_body(parser(language).parse(src).root_node) is None:
            src = src.rstrip(b"\n") + b"\n\n" + it.text.encode() + b"\n"
            added += 1
            done.append((it.kind, key, it.text))
            continue
        body = class_body(parser(language).parse(src).root_node)
        if it.kind == "lambda":
            src = _add_lambda(language, src, body, it)
            added += 1
            done.append((it.kind, key, it.text))
            continue
        old = None
        if it.kind == "method":
            want = method_key(language, it.name, it.arity)
            old = next((ch for ch in body.named_children if ch.type in METHOD_NODES and node_name(src, ch) is not None
                        and method_key(language, node_name(src, ch), arity(ch)) == want), None)
            if old is not None and want not in shells and want not in future:
                clash.append(it.name)
                continue
        if old is None:
            src = _add_member(language, src, body, it.text)
            added += 1
            raw += it.kind == "raw"
            done.append((it.kind, key, it.text))
            continue
        start = span_start(src, old)
        old_text = src[start:old.end_byte].decode("utf-8", errors="replace")
        if language == "php":  # the kept-annotation rule by pattern (php_kept; 26/09/2026)
            new_text, missing = php_kept(it.text, old_text)
        else:
            new_anns = {_norm(l) for l in it.text.split("\n") if STEP_ANNOTATION[language].match(l)}
            new_text = it.text
            missing = [l.strip() for l in old_text.split("\n")
                       if STEP_ANNOTATION[language].match(l) and _norm(l) not in new_anns]
        indent = indent_at(src, start)
        final = with_annotations(language, new_text, missing)
        src = src[:start] + indented(final, indent)[len(indent):].encode() + src[old.end_byte:]
        replaced.append(it.name)
        kept += len(missing)
        done.append((it.kind, key, final))
    return Inserted(src.decode("utf-8", errors="replace"), tuple(replaced), added, kept, raw, tuple(clash), tuple(done))
