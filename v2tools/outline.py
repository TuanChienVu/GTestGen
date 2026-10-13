"""Class outline of a step-definition source file (PHP, Java, C#) via tree-sitter: namespace, imports, parents and
members, each member with its signature text WITHOUT annotations, attributes or docblocks, and for a method or
constructor the line span of the whole member WITH them (rule 5 shows a function as it was before T from that span).
Ruby step files are not classes and have no outline. Used for ④(a) class skeletons, symbol masking and inheritance."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tree_sitter as ts

ANNOTATION_NODES = {"attribute_list", "marker_annotation", "annotation", "comment", "line_comment", "block_comment"}
COMMENT_NODES = {"comment", "line_comment", "block_comment"}


@dataclass(frozen=True)
class Member:
    names: tuple[str, ...]  # declared names (a Java/C# field may declare several); methods have one
    kind: str  # "method" | "constructor" | "property" | "const" | "promoted"
    visibility: str  # "public" | "protected" | "private" | "" (Java/C# default)
    line: int  # 1-based line where the signature text starts
    body_line: int | None  # 1-based line of the body's opening "{" (methods and constructors), else None
    text: str  # signature text: first non-annotation token to "{" inclusive (methods), else the whole declaration
    indent: str  # leading whitespace of the member's first signature line
    span: tuple[int, int] = (0, 0)  # methods and constructors: 1-based (first line with attributes and the docblock or
    # comment lines directly above, last line of the body); (0, 0) for other members


@dataclass(frozen=True)
class Outline:
    language: str
    namespace: str
    imports: dict  # short name -> fully qualified name
    kind: str  # "class" | "trait" | "interface" | "enum"
    name: str
    header: str  # class declaration text without annotations, up to and including "{"
    parents: tuple[tuple[str, str], ...]  # (relation, name as written): "extends" | "implements" | "trait"
    members: tuple[Member, ...]
    params: dict  # constructor Member.line -> tuple of (param text, promoted name or None, parameter name); keyed by
    # line, not name, so that overloaded Java/C# constructors keep their own parameters


@lru_cache(maxsize=None)
def _parser(language: str) -> ts.Parser:
    if language == "php":
        import tree_sitter_php as m
        return ts.Parser(ts.Language(m.language_php()))
    if language == "java":
        import tree_sitter_java as m
    elif language == "csharp":
        import tree_sitter_c_sharp as m
    else:
        raise ValueError(f"no outline for {language}")
    return ts.Parser(ts.Language(m.language()))


def has_syntax_error(language: str, source: str) -> bool:
    """True when tree-sitter reports an ERROR or MISSING node anywhere (BV condition 1)."""
    if language == "ruby":
        import tree_sitter_ruby as m
        tree = ts.Parser(ts.Language(m.language())).parse(source.encode())
    else:
        tree = _parser(language).parse(source.encode())
    return tree.root_node.has_error


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _sig_start(node):
    """First child that is not an annotation/attribute/comment; for Java/C# modifiers, the first non-annotation token."""
    for c in node.children:
        if c.type in ANNOTATION_NODES:
            continue
        if c.type == "modifiers":  # Java: annotations live inside modifiers
            for m in c.children:
                if m.type not in ANNOTATION_NODES:
                    return m
            continue
        return c
    return node


def _indent(src: bytes, node) -> str:
    line_start = src.rfind(b"\n", 0, node.start_byte) + 1
    prefix = src[line_start:node.start_byte].decode("utf-8", errors="replace")
    return prefix if not prefix.strip() else ""


def _span(src: bytes, node) -> tuple[int, int]:
    """1-based (first, last) line of a method or constructor: its attributes or annotations, and the docblock or comment
    lines directly above it (each starting its own line, no blank line between), to the end of its body."""
    first, prev = node.start_point[0], node.prev_sibling
    while prev is not None and prev.type in COMMENT_NODES and first - prev.end_point[0] <= 1 \
            and not src[src.rfind(b"\n", 0, prev.start_byte) + 1:prev.start_byte].strip():
        first, prev = prev.start_point[0], prev.prev_sibling
    return first + 1, node.end_point[0] + 1


def _visibility(src: bytes, node, language: str) -> str:
    words = set()
    for c in node.children:
        if c.type in ("visibility_modifier", "modifier", "modifiers"):
            words |= set(_text(src, c).split())
    for v in ("public", "protected", "private"):
        if v in words:
            return v
    return ""


def _body(node):
    for c in node.children:
        if c.type in ("compound_statement", "block", "constructor_body"):
            return c
    return None


def _signature(src: bytes, node) -> tuple[int, int | None, str]:
    start = _sig_start(node)
    body = _body(node)
    end = body.start_byte + 1 if body is not None else node.end_byte
    return start.start_point[0] + 1, (body.start_point[0] + 1 if body is not None else None), \
        src[start.start_byte:end].decode("utf-8", errors="replace")


def _names(src: bytes, node, child_types: tuple[str, ...]) -> tuple[str, ...]:
    out = []

    def walk(n):
        for c in n.children:
            if c.type in child_types:
                name = c.child_by_field_name("name") or next((x for x in c.children if x.type in ("identifier", "variable_name", "name")), None)
                if name is not None:
                    out.append(_text(src, name).lstrip("$"))
            elif c.type not in ANNOTATION_NODES:
                walk(c)
    walk(node)
    return tuple(out)


def _php_class_parts(src: bytes, cls):
    parents = []
    for c in cls.children:
        if c.type == "base_clause":
            parents += [("extends", _text(src, n)) for n in c.children if n.type in ("name", "qualified_name")]
        elif c.type == "class_interface_clause":
            parents += [("implements", _text(src, n)) for n in c.children if n.type in ("name", "qualified_name")]
    return parents


def parse_snippet(language: str, code: str) -> Outline | None:
    """Outline of a bare code block (the CREATE functions of an answer) wrapped in a class; None for Ruby."""
    if language == "ruby":
        return None
    return parse(language, ("<?php class X {\n" if language == "php" else "class X {\n") + code + "\n}")


def parse(language: str, source: str, name: str | None = None) -> Outline | None:
    """Outline of the class/trait/interface/enum named `name` in the file, else of the first one; None when there is no
    class, or for Ruby. A historical version may hold several classes: callers pass the HEAD class name."""
    if language == "ruby":
        return None
    src = source.encode("utf-8")
    root = _parser(language).parse(src).root_node
    namespace, imports, found = "", {}, []

    def visit(n):
        nonlocal namespace
        for c in n.children:
            t = c.type
            if t in ("namespace_definition", "package_declaration", "namespace_declaration", "file_scoped_namespace_declaration"):
                name = next((x for x in c.children if x.type in ("namespace_name", "scoped_identifier", "identifier", "qualified_name")), None)
                if name is not None:
                    namespace = _text(src, name)
                visit(c)
            elif t in ("namespace_use_declaration",):
                for clause in (x for x in c.children if x.type == "namespace_use_clause"):
                    parts = [x for x in clause.children if x.type in ("qualified_name", "name")]
                    if not parts:
                        continue
                    fqn = _text(src, parts[0]).lstrip("\\")
                    alias = _text(src, parts[1]) if len(parts) > 1 else fqn.rsplit("\\", 1)[-1]
                    imports[alias] = fqn
            elif t == "import_declaration":
                body = _text(src, c)[len("import"):].strip().rstrip(";").strip()
                if not body.startswith("static ") and not body.endswith(".*"):
                    imports[body.rsplit(".", 1)[-1]] = body
            elif t == "using_directive":
                name = next((x for x in c.children if x.type in ("qualified_name", "identifier")), None)
                if name is not None:
                    imports[_text(src, name).rsplit(".", 1)[-1]] = _text(src, name)
            elif t in ("declaration_list",):
                visit(c)
            elif t in ("class_declaration", "trait_declaration", "interface_declaration", "enum_declaration",
                       "record_declaration"):
                found.append(c)
    visit(root)
    if not found:
        return None

    def cls_name(c) -> str | None:
        nm = c.child_by_field_name("name") or next((x for x in c.children if x.type in ("name", "identifier")), None)
        return _text(src, nm) if nm is not None else None
    cls = next((c for c in found if name is not None and cls_name(c) == name), found[0])
    kind = cls.type.split("_")[0]
    name_node = cls.child_by_field_name("name") or next((x for x in cls.children if x.type in ("name", "identifier")), None)
    body = next((x for x in cls.children if x.type in ("declaration_list", "class_body", "interface_body", "enum_body")), None)
    if name_node is None or body is None:  # a broken historical version: no usable class
        return None
    head_start = _sig_start(cls)
    header = src[head_start.start_byte:body.start_byte + 1].decode("utf-8", errors="replace")
    if language == "php":
        parents = _php_class_parts(src, cls)
    else:
        parents = []
        for c in cls.children:
            if c.type == "superclass":
                parents += [("extends", _text(src, x)) for x in c.children if x.type not in ("extends",)]
            elif c.type in ("super_interfaces",):
                tl = next((x for x in c.children if x.type == "type_list"), None)
                if tl is not None:
                    parents += [("implements", _text(src, x)) for x in tl.children if x.type not in (",",)]
            elif c.type == "base_list":  # C#: first base may be a class or an interface; treat all as "extends"
                parents += [("extends", _text(src, x)) for x in c.children if x.type not in (":", ",")]
    members, params = [], {}
    for c in body.children:
        t = c.type
        if t == "use_declaration":  # PHP trait use inside the class body
            parents += [("trait", _text(src, x)) for x in c.children if x.type in ("name", "qualified_name")]
            continue
        if t in ("method_declaration", "constructor_declaration"):
            line, body_line, text = _signature(src, c)
            nm = c.child_by_field_name("name") or next((x for x in c.children if x.type in ("name", "identifier")), None)
            if nm is None:
                continue
            mname = _text(src, nm)
            is_ctor = t == "constructor_declaration" or mname == "__construct"
            members.append(Member((mname,), "constructor" if is_ctor else "method", _visibility(src, c, language),
                                  line, body_line, text, _indent(src, _sig_start(c)), _span(src, c)))
            ctor_line = line
            if is_ctor:
                fp = c.child_by_field_name("parameters") or next((x for x in c.children if x.type in ("formal_parameters", "parameter_list")), None)
                plist = []
                for p in (fp.children if fp is not None else []):
                    if p.type in ("(", ")", ","):
                        continue
                    promoted = None
                    pn = p.child_by_field_name("name") or next(
                        (x for x in p.children if x.type in ("variable_name", "identifier")), None)
                    pname = _text(src, pn).lstrip("$") if pn is not None else None
                    if p.type == "property_promotion_parameter":
                        vn = next((x for x in p.children if x.type == "variable_name"), None)
                        promoted = _text(src, vn).lstrip("$") if vn is not None else None
                        if promoted:
                            members.append(Member((promoted,), "promoted", _visibility(src, p, language),
                                                  p.start_point[0] + 1, None, _text(src, p), _indent(src, p)))
                    plist.append((_text(src, p), promoted, pname))
                params[ctor_line] = tuple(plist)
        elif t in ("property_declaration", "field_declaration", "const_declaration", "constant_declaration"):
            start = _sig_start(c)
            text = src[start.start_byte:c.end_byte].decode("utf-8", errors="replace")
            if language == "php":
                names = _names(src, c, ("property_element", "const_element"))
            else:
                names = _names(src, c, ("variable_declarator",))
                if not names and t == "property_declaration":  # C# auto-property
                    nm = c.child_by_field_name("name")
                    names = (_text(src, nm),) if nm is not None else ()
            kind_m = "const" if t in ("const_declaration", "constant_declaration") or " const " in f" {text} " else "property"
            members.append(Member(names, kind_m, _visibility(src, c, language), start.start_point[0] + 1, None, text,
                                  _indent(src, start)))
    return Outline(language, namespace, imports, kind, _text(src, name_node), header, tuple(parents), tuple(members), params)
