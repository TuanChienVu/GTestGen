"""Symbols that generated step code uses, for BV conditions 5 and 6 (§2.2) and the descriptive hallucinated-symbol rate
(§2.4). A reference is either a MEMBER of the current class (PHP `$this->x`, `self::x`, `static::x`, `parent::x`;
Java/C# `this.x`, Java `super.x`, C# `base.X`, a bare call `x(` (also `this.x(`, `super.x(`, `base.X(`; a bare call
inside a Java anonymous class body is that class's own), a bare identifier that is not a parameter or local (pattern,
tuple and LINQ variables are locals), an UPPER_CASE constant) or a CLASS (PHP `new X`, `X::`; Java/C# a capitalised
receiver or `new X`). Ruby: a receiverless call is a member of the World (methods defined under features/support or
features/step_definitions), a constant is a class resolved under app/ or lib/.

Only code that can run counts: every language is read from its tree-sitter tree, skipping comments, string literals
(heredocs and interpolations included) and every statement after an unconditional return or throw, or a statement that
ends the process, in the same block (`live`); Ruby by lines (`ruby_dead_rows`). `if (false)` and closures never called
still count: a documented limit."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ertree import parser
from .insert import LAMBDA, class_body
from .stepdefs import _close_brace

PACKAGE_ROOTS = {"java", "javax", "jakarta", "org", "com", "io", "net", "lombok", "System", "Microsoft"}
DEAD_NODES = {"comment", "line_comment", "block_comment", "string", "encapsed_string", "heredoc", "nowdoc",
              "shell_command_expression", "string_literal", "text_block", "character_literal", "verbatim_string_literal",
              "raw_string_literal", "interpolated_string_expression"}
BLOCK_NODES = {"compound_statement", "block", "constructor_body", "case_statement", "default_statement", "colon_block",
               "switch_block_statement_group", "switch_section"}
# The callee (whitespace removed) of a statement that ends the process, `java.lang.Runtime.…` and
# `global::System.Environment.…` included
PROCESS_EXIT = {
    "method_invocation": re.compile(r"(?:java\.lang\.)?(?:System\.exit|Runtime\.getRuntime\(\)\.(?:exit|halt))"),
    "invocation_expression": re.compile(r"(?:(?:global::)?System\.)?Environment\.(?:Exit|FailFast)")}
COMMENTS = {"comment", "line_comment", "block_comment"}


def _code_kids(n) -> list:
    """The named children of `n` that are not comments."""
    return [c for c in n.named_children if c.type not in COMMENTS]


def _callee(e, end: int) -> str:
    """The source of `e` before byte `end` (its arguments), without comments and whitespace (`System./*c*/exit(1);` is
    `System.exit`)."""
    parts, stack = [], [e]
    while stack:
        x = stack.pop()
        if x.type in COMMENTS or x.start_byte >= end:
            continue
        if x.child_count == 0:
            parts.append(x.text.decode("utf-8", errors="replace"))
        stack.extend(reversed(x.children))
    return re.sub(r"\s+", "", "".join(parts))


def ends_flow(n) -> bool:
    """An unconditional return or throw statement (also under a Java/C# statement label), or a statement that ends the
    process: PHP `exit`/`die` in any case, with or without arguments, as the whole statement (not `$a ?: exit(1)` or
    `f() or die()`), also under the error-suppression operator (`@exit;`, `@die(…);`); Java `System.exit(…)`,
    `Runtime.getRuntime().exit(…)` or `.halt(…)`, each also after `java.lang.`; C# `Environment.Exit(…)` or
    `Environment.FailFast(…)`, also `System.Environment.…` and `global::System.Environment.…`. Comments inside the
    statement never hide it: the callee is compared without comments and whitespace, and a comment is never the
    expression."""
    if n.type in ("return_statement", "throw_statement", "exit_statement"):
        return True
    if n.type == "labeled_statement" and _code_kids(n):
        return ends_flow(_code_kids(n)[-1])
    if n.type != "expression_statement" or not _code_kids(n):
        return False
    e = _code_kids(n)[0]
    while e is not None and e.type == "error_suppression_expression":  # PHP `@exit;`, `@die(…);`
        e = _code_kids(e)[0] if _code_kids(e) else None
    if e is not None and e.type == "function_call_expression":  # PHP `die(…)`, `@exit(…)`; `exit(…)`: exit_statement
        e = e.child_by_field_name("function")
    if e is not None and e.type == "name":  # PHP `die;`
        return e.text.decode("utf-8", errors="replace").lower() in ("exit", "die")
    args = e.child_by_field_name("arguments") if e is not None and e.type in PROCESS_EXIT else None
    if args is not None:
        return PROCESS_EXIT[e.type].fullmatch(_callee(e, args.start_byte)) is not None
    return e is not None and e.type == "throw_expression"


def live(node):
    """Nodes that can run, in pre-order: comments and string literals are skipped, and so is every statement after an
    unconditional return or throw, or a process exit (`ends_flow`), in the same block (a brace block, a switch case, or
    a PHP `:` block)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in DEAD_NODES:
            continue
        yield n
        kids = list(n.children)
        if n.type in BLOCK_NODES:
            cut = next((i for i, c in enumerate(kids) if ends_flow(c)), None)
            if cut is not None:
                kids = kids[:cut + 1] + [c for c in kids[cut + 1:] if not c.is_named]
        stack.extend(reversed(kids))


def php_body(code: str):
    """(source bytes, the node to read) of a PHP function: the body of its method, else the whole wrapped code."""
    src = ("<?php\nclass __X {\n" + code + "\n}\n").encode()
    root = parser("php").parse(src).root_node
    m = next((n for n in live(root) if n.type == "method_declaration"), None)
    body = m.child_by_field_name("body") if m is not None else None
    return src, body if body is not None else root


@dataclass(frozen=True)
class Refs:
    members: frozenset[str]
    # the members used as bare calls (x( / $this->x( ): an interface default method may serve them; PHP also self::x(,
    # static::x(, parent::x( (method calls, whose name resolves in any case)
    calls: frozenset[str]
    classes: frozenset[str]


def lambda_text(language: str, code: str) -> str | None:
    """For a Java lambda step, the lambda itself, `(params) -> { ... }` (not the Given(...) registration call)."""
    if language != "java":
        return None
    m = LAMBDA.search(code)
    if m is None:
        return None
    try:
        return code[m.start(1):_close_brace(code, m.end() - 1) + 1]
    except ValueError:
        return None


PHP_MEMBER = {"member_access_expression", "nullsafe_member_access_expression", "member_call_expression",
              "nullsafe_member_call_expression"}
PHP_SCOPED = {"scoped_call_expression", "scoped_property_access_expression", "class_constant_access_expression"}


def php_scoped(src: bytes, n) -> tuple[object, str]:
    """(scope node, accessed name without `$`) of `X::f()`, `X::$p` or `X::C`."""
    kids = n.named_children
    scope, name = n.child_by_field_name("scope") or kids[0], n.child_by_field_name("name") or kids[-1]
    return scope, src[name.start_byte:name.end_byte].decode("utf-8", errors="replace").lstrip("$")


def php_refs(code: str) -> Refs:
    src, body = php_body(code)
    text = lambda n: src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
    is_this = lambda n: n is not None and n.type == "variable_name" and text(n) == "$this"
    members, calls, classes = set(), set(), set()
    for n in live(body):
        if n.type in PHP_MEMBER and is_this(n.child_by_field_name("object")):
            name = n.child_by_field_name("name")
            if name is not None and name.type == "name":
                members.add(text(name))
                if n.type.endswith("call_expression"):
                    calls.add(text(name))
        elif n.type in PHP_SCOPED:  # `::class` in any case names no member (PHP ignores case)
            scope, name = php_scoped(src, n)
            if scope.type == "relative_scope":
                if name.lower() != "class":
                    members.add(name)
                if n.type == "scoped_call_expression":  # a METHOD call: any case resolves and grounds
                    calls.add(name)
            elif scope.type in ("name", "qualified_name") and re.match(r"\\?[A-Z]", text(scope)) \
                    and name.lower() != "class":
                classes.add(text(scope))
        elif n.type == "object_creation_expression":
            cls = next((c for c in n.named_children if c.type in ("name", "qualified_name")), None)
            if cls is not None:
                classes.add(text(cls))
    return Refs(frozenset(members), frozenset(calls), frozenset(classes - {"self", "static", "parent"}))


# Java and C#: identifier roles by (parent node type, field name).
DECLARES = {("variable_declarator", "name"), ("formal_parameter", "name"), ("catch_formal_parameter", "name"),
            ("enhanced_for_statement", "name"), ("lambda_expression", "parameters"), ("inferred_parameters", None),
            ("resource", "name"), ("type_pattern", None), ("record_pattern_component", None), ("parameter", "name"),
            ("foreach_statement", "left"), ("catch_declaration", "name"),
            ("declaration_expression", "name"), ("implicit_parameter", None), ("for_each_statement", "left"),
            # Java `instanceof T x`; C# `is T x` / `is var x`, `var (a, b)`, LINQ `from x`
            ("instanceof_expression", "name"), ("declaration_pattern", "name"), ("tuple_pattern", "name"),
            ("from_clause", "name")}
SKIP = {("method_invocation", "name"), ("field_access", "field"), ("method_reference", None),
        ("labeled_statement", None), ("break_statement", None), ("continue_statement", None),
        ("scoped_identifier", "name"), ("scoped_identifier", "scope"), ("marker_annotation", "name"),
        ("annotation", "name"), ("switch_label", None), ("element_value_pair", "key"),
        ("member_access_expression", "name"), ("argument", "name"), ("name_colon", None), ("attribute", "name"),
        ("qualified_name", None), ("generic_name", None), ("goto_statement", None), ("labeled_statement", "label"),
        ("method_declaration", "name"), ("constructor_declaration", "name"), ("class_declaration", "name"),
        ("local_function_statement", "name")}


def in_anonymous_class(n) -> bool:
    """Whether n lies inside a Java anonymous class body (`new HashMap<>() {{ put(…); }}`; 28/09/2026): a bare or
    `this.` call there is the anonymous class's own, resolved against its supertype first, never a member (jvm_refs) nor
    a call (calls.jvm_calls) of the step class."""
    p = n.parent
    while p is not None:
        if p.type == "class_body" and p.parent is not None and p.parent.type == "object_creation_expression":
            return True
        p = p.parent
    return False


def jvm_refs(language: str, code: str) -> Refs:
    lam = lambda_text(language, code)
    src = ("class __X { void __m() { __f(" + lam + "); } }" if lam else "class __X {\n" + code + "\n}").encode()
    root = parser(language).parse(src).root_node
    local, members, calls, classes = set(), set(), set(), set()

    def role(n):
        p = n.parent
        if p is None:
            return None
        idx = next(i for i, c in enumerate(p.children) if c.id == n.id)
        return p.type, p.field_name_for_child(idx)

    def text(n):
        return src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def declare(n):
        if n.type == "implicit_parameter":  # C#: `x => ...`
            local.add(text(n))
        p = n.parent
        if n.type == "identifier" and p is not None and (  # C# LINQ `let x =`, `join x in`, `into x`
                p.type in ("let_clause", "join_clause") and next(c for c in p.named_children if c.type == "identifier").id == n.id
                or p.type == "query_expression"):
            local.add(text(n))
        if n.type == "identifier" and (role(n) in DECLARES or (role(n) or ("", ""))[0] in
                                        ("inferred_parameters", "implicit_parameter", "type_pattern")):
            local.add(text(n))
    for n in live(root):  # a local declared in dead code does not hide a member
        declare(n)

    anonymous = in_anonymous_class

    def use(n):
        t = n.type
        if t in ("field_access", "member_access_expression"):
            obj = n.child_by_field_name("object") or n.child_by_field_name("expression")
            name = n.child_by_field_name("field") or n.child_by_field_name("name")
            # `super.x` and C# `base.X` name a member as `this.x` does
            if obj is not None and obj.type in ("this", "this_expression", "super", "base") and name is not None \
                    and not (obj.type == "this" and anonymous(n)):
                members.add(text(name))
                if t == "member_access_expression" and n.parent is not None and n.parent.type == "invocation_expression" \
                        and n.parent.child_by_field_name("function") == n:
                    calls.add(text(name))  # C# `this.F()`, `base.F()`: a call, as Java's
            return
        if t in ("method_invocation", "invocation_expression"):
            fname = n.child_by_field_name("function")
            obj = n.child_by_field_name("object")
            if t == "method_invocation" and (obj is None or obj.type in ("this", "super")):  # this./super.
                if obj is not None and obj.type == "super" or not anonymous(n):
                    calls.add(text(n.child_by_field_name("name")))
            elif fname is not None and fname.type == "identifier":
                calls.add(text(fname))
            return
        if t == "object_creation_expression":
            typ = n.child_by_field_name("type")
            if typ is not None:
                classes.add(text(typ).split("<", 1)[0])
            return
        if t != "identifier":
            return
        r = role(n)
        if r is None or r in DECLARES or r in SKIP or (r[0], None) in SKIP or r[0] in ("inferred_parameters", "type_pattern"):
            return
        if r in (("method_invocation", "name"), ("invocation_expression", "function")):
            return
        name = text(n)
        if name in local or name.startswith("__") or name in PACKAGE_ROOTS:
            return
        receiver = r[1] in ("object", "expression")
        if name[0].isupper() and (receiver or not (name.isupper() and len(name) > 1)):
            classes.add(name)
        else:
            members.add(name)
    for n in live(root):
        use(n)
    calls = {c for c in calls if not c.startswith("__")}
    return Refs(frozenset(members | calls), frozenset(calls), frozenset(classes))


RUBY_EXIT = re.compile(r"\s*(?:return|raise|fail)\b")
RUBY_MODIFIER = re.compile(r" (?:if|unless|while|until) ")
RUBY_OPENS = re.compile(r"\s*(?:if|unless|while|until|case|begin|def|class|module)\b")
RUBY_BRANCH = re.compile(r"\s*(?:else|elsif|when|in|rescue|ensure)\b")
RUBY_NEST = re.compile(r"\b(?:do|end)\b|[{}]")


def ruby_dead_rows(src: bytes, root) -> set[int]:
    """0-based rows of Ruby code that cannot run, read by lines once strings, heredocs and comments are blanked: after a
    line whose first token is return, raise or fail with no trailing modifier (` if `, ` unless `, ` while `, ` until `
    after the keyword), every line up to the one holding the end or } that closes its innermost block, or up to an
    else/elsif/when/in/rescue/ensure line at the same depth. Openers: do, {, and a line starting with if, unless, while,
    until, case, begin, def, class or module; closers: end, }. The tokens of the return line after the keyword count too
    (`return x.map do |i|` opens a block)."""
    buf, stack = bytearray(src), [root]
    while stack:
        n = stack.pop()
        if n.type in DEAD_NODES or n.type == "heredoc_body":
            buf[n.start_byte:n.end_byte] = re.sub(rb"[^\n]", b" ", src[n.start_byte:n.end_byte])
        else:
            stack.extend(n.children)
    lines = buf.decode("utf-8", errors="replace").split("\n")

    def nest(line: str, depth: int) -> int:  # the depth after `line`, -1 once a closer leaves the block
        depth += RUBY_OPENS.match(line) is not None
        for t in RUBY_NEST.finditer(line):
            depth += 1 if t.group() in ("do", "{") else -1
            if depth < 0:
                return -1
        return depth
    dead, i = set(), 0
    while i < len(lines):
        m = RUBY_EXIT.match(lines[i])
        i += 1
        if m is None or RUBY_MODIFIER.search(lines[i - 1], m.end()):
            continue
        depth = nest(lines[i - 1][m.end():], 0)
        while depth >= 0 and i < len(lines) and not (depth == 0 and RUBY_BRANCH.match(lines[i])):
            depth = nest(lines[i], depth)
            if depth >= 0:
                dead.add(i)
                i += 1
    return dead


def ruby_refs(code: str) -> Refs:
    src = code.encode()
    root = parser("ruby").parse(src).root_node
    calls, classes, local = set(), set(), set()
    dead = ruby_dead_rows(src, root)

    def text(n):
        return src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def visit(n):
        if n.type in DEAD_NODES or n.start_point[0] in dead:  # dead lines: no call, no constant
            return
        if n.type in ("block_parameters", "method_parameters", "lambda_parameters"):
            local.update(text(c) for c in n.named_children if c.type == "identifier")
        elif n.type == "assignment" and n.child_by_field_name("left") is not None and n.child_by_field_name("left").type == "identifier":
            local.add(text(n.child_by_field_name("left")))
        elif n.type == "call" and n.child_by_field_name("receiver") is None and n.child_by_field_name("method") is not None:
            calls.add(text(n.child_by_field_name("method")))
        elif n.type == "identifier" and n.parent is not None and n.parent.type in ("argument_list", "body_statement", "block_body", "do_block"):
            calls.add(text(n))
        elif n.type in ("constant", "scope_resolution") and (n.parent is None or n.parent.type != "scope_resolution"):
            classes.add(text(n))
            return
        for c in n.children:
            visit(c)
    visit(root)
    calls -= local
    calls -= {"Given", "When", "Then", "And", "But", "Step"}
    return Refs(frozenset(calls), frozenset(calls), frozenset(c for c in classes if c not in ("Given", "When", "Then")))


def refs(language: str, code: str) -> Refs:
    if language == "php":
        return php_refs(code)
    if language == "ruby":
        return ruby_refs(code)
    return jvm_refs(language, code)


DECLARATORS = {"variable_declarator", "property_element", "const_element", "property_declaration"}


def initializer_refs(language: str, code: str) -> Refs | None:
    """The references of a member item's initializers (what follows `=` in each of its declarators: a Java/C# field or
    constant, a C# property, a PHP property or constant), read as statements of a method body so that they resolve like
    body references; None when the member has no initializer."""
    src = (("<?php\nclass __X {\n" if language == "php" else "class __X {\n") + code + "\n}\n").encode()
    body = class_body(parser(language).parse(src).root_node)
    exprs = []
    for m in body.named_children if body is not None else []:
        for d in [m, *m.children, *(g for c in m.children if c.type == "variable_declaration" for g in c.children)]:
            eq = next((i for i, c in enumerate(d.children) if c.type == "="), None) if d.type in DECLARATORS else None
            if eq is not None:
                exprs += [src[c.start_byte:c.end_byte].decode("utf-8", errors="replace") for c in d.children[eq + 1:]
                          if c.is_named and c.type not in DEAD_NODES]
    if not exprs:
        return None
    if language == "php":
        return refs(language, "public function __init()\n{\n" + "".join(f"$__v = {e};\n" for e in exprs) + "}")
    typ = "Object" if language == "java" else "object"
    return refs(language, "void __init() {\n" + "".join(f"{typ} __v = {e};\n" for e in exprs) + "}")


# ----- what a target class offers ------------------------------------------------------------------------------------

STATIC_IMPORT = re.compile(r"^\s*(?:import|using)\s+static\s+([\w.]+?)(?:\.(\w+|\*))?\s*;", re.M)
MAGIC = {"__call", "__get", "__callStatic"}
LOMBOK_LOG = re.compile(r"^\s*@(?:lombok\.extern\.\w+\.)?(?:Slf4j|XSlf4j|Log4j2?|Log|CommonsLog|JBossLog|Flogger|CustomLog)\b", re.M)
# Loggers are not project knowledge: by name, or by the declared type of a field
LOGGER_NAME = re.compile(r"^(log|logger|LOG|LOGGER|_log|_logger)$")
JAVA_LOGGERS = ("org.slf4j.Logger", "org.apache.logging.log4j.Logger", "org.apache.log4j.Logger", "java.util.logging.Logger")
JAVA_LOGGER_FIELD = re.compile(r"\b((?:[\w.]+\.)?Logger)\s+(\w+)\s*[=;]")
# What every class inherits from its language's root object (PHP and Ruby have none)
ROOT_MEMBERS = {
    "java": frozenset({"getClass", "hashCode", "equals", "toString", "clone", "finalize", "notify", "notifyAll",
                       "wait"}),
    "csharp": frozenset({"GetType", "GetHashCode", "Equals", "ToString", "MemberwiseClone", "ReferenceEquals",
                         "Finalize"})}
PHP_LOGGER_FIELD = re.compile(r"\b(?:private|protected|public|readonly|var)\s+(?:readonly\s+)?\??\\?(?:Psr\\Log\\)?LoggerInterface\s+\$(\w+)")


def logger_fields(language: str, text: str) -> set[str]:
    """Fields whose declared type is an SLF4J, Log4j or JUL logger (Java) or a PSR-3 LoggerInterface (PHP)."""
    if language == "php":
        return set(PHP_LOGGER_FIELD.findall(text))
    if language != "java":
        return set()
    imported = any(re.search(rf"^\s*import\s+{re.escape(t)}\s*;", text, re.M) for t in JAVA_LOGGERS)
    return {name for typ, name in JAVA_LOGGER_FIELD.findall(text) if typ in JAVA_LOGGERS or typ == "Logger" and imported}


@dataclass(frozen=True)
class Scope:
    available: frozenset[str]  # members of the scored class (inserted code included) and of its in-repo ancestors
    project: frozenset[str]  # members that exist in the ER tree (inserted code excluded): own plus inherited
    statics: frozenset[str]  # Java/C#: static members of in-repo classes imported statically by the ER-tree file
    open_all: bool  # an ancestor outside the repo (class or trait), or a magic __call/__get: members cannot be judged
    open_calls: bool  # an interface outside the repo or a .* static import that is external or added by the output
    lombok_log: bool = False  # `log` is available only through a Lombok logger annotation
    open_statics: frozenset[str] = frozenset()  # names of explicit static imports that are external or added by the output
    loggers: frozenset[str] = frozenset()  # logger fields of the class and its in-repo ancestors, by name or declared type
    root: frozenset[str] = frozenset()  # members of the root object (ROOT_MEMBERS): resolved for condition 6, never an
    # open path; not in `available` or `project`, so they never ground (condition 5) nor enter a call set
    methods: frozenset[str] = frozenset()  # PHP: casefolded METHOD names of `available` (method names ignore case)
    project_methods: frozenset[str] = frozenset()  # PHP: casefolded METHOD names of `project`


def php_promoted(text: str) -> dict[str, str]:
    """PHP names are case-insensitive: name -> visibility of the promoted parameters of the constructor of the file's
    class, written `__construct` in ANY case (`__CONSTRUCT`): 3A's outline, which is not edited here, only knows the
    lower-case name, so symbols.scope adds these as properties."""
    from .outline import _visibility
    if not any(m.group(1) != "__construct" for m in re.finditer(r"(?i)\bfunction\s+(__construct)\b", text)):
        return {}  # the outline already has what a lower-case constructor promotes
    src = text.encode()
    body = class_body(parser("php").parse(src).root_node)
    word = lambda n: src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
    out: dict[str, str] = {}
    for m in body.named_children if body is not None else ():
        name = m.child_by_field_name("name") if m.type == "method_declaration" else None
        if name is None or word(name).casefold() != "__construct":
            continue
        params = m.child_by_field_name("parameters")
        for p in params.named_children if params is not None else ():
            v = next((x for x in p.children if x.type == "variable_name"), None)
            if p.type == "property_promotion_parameter" and v is not None:
                out[word(v).lstrip("$")] = _visibility(src, p, "php")
    return out


def scope(ctx, path: str, text_er: str, text_after: str, er_texts: dict[str, str]) -> Scope | None:
    """`ctx` is a state.RepoContext (HEAD parents and ancestors), `er_texts` the files the ER tree changed; None when
    the target file has no class (Ruby, or a broken file)."""
    from .outline import parse
    lang = ctx.language
    o_after, o_er = parse(lang, text_after), parse(lang, text_er)
    if o_after is None or o_er is None:
        return None
    names = lambda o, private=True: {n for m in o.members if private or m.visibility != "private" for n in m.names}
    available, project = names(o_after), names(o_er)
    # PHP names are case-insensitive: `__CONSTRUCT` promotes its parameters too; and the casefolded METHOD names, which conditions 5 and 6 match in any case (property names stay case-sensitive)
    meths = lambda o, private=True: {n.casefold() for m in o.members if m.kind in ("method", "constructor")
                                     and (private or m.visibility != "private") for n in m.names}
    methods, project_methods = (meths(o_after), meths(o_er)) if lang == "php" else (set(), set())
    if lang == "php":
        available |= set(php_promoted(text_after))
        project |= set(php_promoted(text_er))
    lombok = lang == "java" and LOMBOK_LOG.search(text_er) is not None and "log" not in available
    if lombok:  # Lombok's generated logger field
        available.add("log")
        project.add("log")
    open_all = o_after.kind == "trait"  # a trait uses the members of the classes that use it
    open_calls = False
    for p in [path] + [par.path for par, _ in ctx.ancestors(path)]:
        for par in ctx.parents(p):
            if par.status == "resolved":
                continue
            if par.relation in ("extends", "trait") or lang == "php" and par.relation != "implements":
                open_all = True
            elif lang != "php":
                open_calls = True
    for par, _ in ctx.ancestors(path):
        o = parse(lang, er_texts.get(par.path) or ctx.text(par.path) or "")
        if o is not None:
            inherited = names(o, private=par.relation == "trait")
            if lang == "php":
                promoted = php_promoted(er_texts.get(par.path) or ctx.text(par.path) or "")
                inherited |= {n for n, v in promoted.items() if par.relation == "trait" or v != "private"}
                methods |= meths(o, private=par.relation == "trait")
                project_methods |= meths(o, private=par.relation == "trait")
            available |= inherited
            project |= inherited
    if available & MAGIC:
        open_all = True
    # a static import counts only when the ER-tree file has it and its class resolves in the repo;
    # an external or output-added one is an open path (condition 6 cannot judge it; condition 5 never counts it)
    statics, open_statics, in_er = set(), set(), set(STATIC_IMPORT.findall(text_er))
    for cls, member in STATIC_IMPORT.findall(text_after):
        target = ctx.resolver.by_suffix.get(cls.replace(".", "/")) if lang == "java" else None
        o = parse(lang, ctx.text(target) or "") if target and (cls, member) in in_er else None
        if o is None:
            if member and member != "*":
                open_statics.add(member)
            else:
                open_calls = True
        elif member and member != "*":
            statics.add(member)
        else:
            statics |= {n for m in o.members for n in m.names}
    loggers = {n for n in available if LOGGER_NAME.match(n)} | logger_fields(lang, text_er)
    for par, _ in ctx.ancestors(path):
        loggers |= logger_fields(lang, er_texts.get(par.path) or ctx.text(par.path) or "")
    return Scope(frozenset(available), frozenset(project), frozenset(statics), open_all, open_calls, lombok,
                 frozenset(open_statics), frozenset(loggers), ROOT_MEMBERS.get(lang, frozenset()), frozenset(methods),
                 frozenset(project_methods))


def _any_case(r: Refs, s: Scope) -> set[str]:
    """The METHOD calls of `r` that resolve only case-insensitively, on a method of `s` (PHP method names ignore case,
    property names do not)."""
    return {n for n in r.calls if n.casefold() in s.methods}


def open_paths(ctx, o_after, r: Refs, s: Scope) -> set[str]:
    """The references that pass condition 6 only through an "open" path: (a) a name of an explicit static import that is
    external or added by the output; an ancestor outside the repo or a trait target (members nothing else resolves); a
    bare call left to an external interface or .* static import; `log` from Lombok; (b) a C# PascalCase name that is
    neither a member nor a class of the repo (never judged). The score row's `open_resolution` = condition 6 passed AND
    this set is not empty. PHP class references are never judged by condition 6 (§2.2), so they are not an open path."""
    strict = set(r.members) - (s.available - ({"log"} if s.lombok_log else set())) - s.statics - s.root
    strict -= _any_case(r, s)
    out = strict & s.open_statics
    if s.open_all:
        out |= strict
    elif s.open_calls:
        out |= strict & set(r.calls)
    if s.lombok_log and "log" in r.members:
        out.add("log")
    if ctx.language == "csharp":
        out |= {c for c in r.classes if c not in s.available | s.root
                and class_status(ctx, o_after, c) not in ("resolved", "global")}
    return out


def unresolved(r: Refs, s: Scope) -> set[str]:
    """BV condition 6: member references that the scored tree cannot resolve (the root object's members resolve; a PHP
    method call resolves in any case)."""
    if s.open_all:
        return set()
    out = set(r.members) - s.available - s.root - s.statics - s.open_statics - _any_case(r, s)
    return out - set(r.calls) if s.open_calls else out


def grounded(ctx, o_after, r: Refs, s: Scope) -> bool:
    """BV condition 5: the code uses at least one project member (of the ER tree, not one the output defines, not a
    logger) or an in-repo class. A capitalised name that `refs` files as a class counts when it is a member: C# members
    are PascalCase, and a Java UPPER_CASE field used as a receiver (`FORMATTER.format()`) looks like a class. Static
    imports count only through `statics` (in the ER tree, class in the repo). A PHP method call counts in any case when
    the ER tree has that method."""
    names = (set(r.members) | (set(r.classes) if ctx.language in ("csharp", "java") else set())) - s.loggers
    names = {n for n in names if not LOGGER_NAME.match(n)}
    if names & (s.project | s.statics) or any(n.casefold() in s.project_methods for n in set(r.calls) & names):
        return True
    return any(class_status(ctx, o_after, c) in ("resolved", "global") for c in r.classes)


_GLOBALS: dict[int, frozenset[str]] = {}


def class_status(ctx, o, name: str) -> str:
    """inherit.Resolver's status of a class reference, plus "global": a PHP class without a namespace counts as in the
    repo when a file <Name>.php exists outside vendor/ (PrestaShop's legacy classes/)."""
    st = ctx.resolver.resolve(o, "ref", name)
    if st.status != "resolved" and ctx.language == "php" and "\\" not in st.fqn:
        if id(ctx) not in _GLOBALS:
            _GLOBALS[id(ctx)] = frozenset(p.rsplit("/", 1)[-1][:-4] for p in ctx.tree
                                          if p.endswith(".php") and not p.startswith("vendor/") and "/vendor/" not in p)
        if st.fqn in _GLOBALS[id(ctx)]:
            return "global"
    return st.status


# ----- Ruby: the World and the application's constants ---------------------------------------------------------------

RUBY_DEF = re.compile(r"^\s*def\s+(?:self\.)?([a-z_]\w*[?!=]?)", re.M)
WORLD_DIRS = ("features/support/", "features/step_definitions/")


@dataclass(frozen=True)
class RubyWorld:
    methods: frozenset[str]  # `def` names under features/support/** and features/step_definitions/** at HEAD
    paths: frozenset[str]  # Zeitwerk paths of app/*/** and lib/**: "renalware/patients/patient" (files and dirs)
    roots: frozenset[str]  # first segments of those paths: the application's top-level namespaces


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def ruby_world(ctx) -> RubyWorld:
    methods, paths = set(), set()
    for p in ctx.tree:
        if not p.endswith(".rb"):
            continue
        if p.startswith(WORLD_DIRS):
            methods |= set(RUBY_DEF.findall(ctx.text(p) or ""))
        m = re.match(r"^(?:app/[^/]+|lib)/(.+)\.rb$", p)
        if m:
            parts = m.group(1).split("/")
            paths |= {"/".join(parts[:i]) for i in range(1, len(parts) + 1)}
    return RubyWorld(frozenset(methods), frozenset(paths), frozenset(p.split("/")[0] for p in paths))


def ruby_unresolved(r: Refs, w: RubyWorld) -> set[str]:
    """Constants under an application namespace that name no file or directory of app/*/ or lib/."""
    out = set()
    for c in r.classes:
        segs = [snake(s) for s in c.lstrip(":").split("::")]
        if segs[0] in w.roots and "/".join(segs) not in w.paths:
            out.add(c)
    return out


def ruby_grounded(r: Refs, w: RubyWorld) -> bool:
    if set(r.calls) & w.methods:
        return True
    return any("/".join(snake(s) for s in c.lstrip(":").split("::")) in w.paths for c in r.classes)
