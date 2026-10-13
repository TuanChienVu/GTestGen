"""Pre-registered probe outputs for the agreement run of the two scorers, written from the controls run of one partition
(control_outputs.jsonl: the gold answer and stub-v3 of every record). Two families, per PHP/Java/C# record:
- the 9 model-like shapes of make_unusual.py, derived from the gold answer: U1 the whole class wrapped again, U2 one
block per function, U3 the first function twice, U4 an extra function re-binding the first REUSE step, U6 the last block
unclosed, U7 an extra bash block and a block for a file that is not a target, U8 a `./` path, U11 every method renamed,
U12 every body emptied;
- the probes of CA gaming (focused.py, probe_ca.py), built on stub-v3: the lines of the gold body in a comment (A), in
string literals (B), after `return;` (C), under `if (false)` (D), in a closure never called (E), in a helper never
called (F); a logger call alone (K); `assertThat(true)` through a static import the output adds (J, Java);
- further shapes (V): a field the output declares and uses (V1_field_added), `return` inside a switch case (V2), an
unknown member in an initializer (V3, Java/C#) or a C# argument (V6), a char literal (V4, Java/C#), nullsafe calls (V5,
PHP), `return` inside a PHP `:` block (V7), after a PHP label (V8), a PHP backtick string (V9), and the gold answer cut
off at 60% of its text, as when an output hits the token cap (V10);
- the dead-code and hidden-member probes (X and R, from the gold answer): `exit;` (X7, PHP) or `System.exit(0);` (X8,
Java) then the lines of the gold body, on stub-v3 like C; the gold methods inside an anonymous class in a field
initializer (X3, Java; only for gold written as methods: a cucumber-java8 lambda step is pulled out before the members
are read, so X3 would test nothing) or wrapped in a struct (X4, C#); the gold answer plus a field whose initializer uses
an unknown member, `private Object probeF = ghostFieldZ.make();` (X5, Java); the gold answer plus a logger field whose
initializer calls getClass() of java.lang.Object (X6, Java; it must score as gold); the gold method inside a class
nested in an anonymous class of a field (X9, Java) or inside an anonymous class of a property (X10, PHP): a member never
binds a step. Ruby records (aux, renalware) get only R1, `return` then the lines of the gold block's body at the end of
every stub-v3 block, and R2, the gold answer with an inner block `[1].each do |i| … end` at the end of every step block
(condition 4 must read the step's own |…|);
- the probes of the future methods (the expansion of helper calls), PHP/Java/C#: the gold methods' bodies moved into a
cycle of two helpers (H3: step → A → B → A, the gold lines in A; CA as gold) or a chain of four (H4: the gold lines in
the fourth helper, beyond depth 3; CA 0); stub-v3m and stub-v3all with each body moved into a helper the step calls (H6,
H7; CA as the stub's); and, when the controls run has them, gold-helpers (H1) and gold-helpers-renamed (H2) as output
rows, and gold-helpers with every helper's body replaced by a meaningless statement (H5: a future helper's name, another
body; only the output's calls count). A helper takes the parameters of the method it comes from;
- the fence shapes (PHP/Java/C#): U13, the gold's target files in ONE fence, each after its own FILE line (a single-file
record: the gold, then a second FILE line for the same file with an empty helper), status multi_file_fence; U14, the
gold in a four-backtick fence holding a three-backtick line in a comment, which stays code;
- the hidden step holders, for a gold made of methods only: PA2 (PHP, `private $probeX;` and probeInit(<the first gold
method's parameters>) assigning `$this->probeX = new class { <every gold method> };`), JB4 (Java,
probeHolder(<parameters>) { Object probeX = new Object() { <every gold method> }; }) and CS1 (C#, every gold method as a
local function keeping its attributes, inside ProbeHolder(<parameters>)); no framework registers such a step, so binding
fails;
- O1 (Java, a documented limit): the first step method of the gold with a body calls a new private helper probeH(),
whose body calls toString() on the gold body's first field receiver (else `this`), and a never-called overload
probeH(<the method's parameters>, int neverCalled) holds the gold body lines: calls resolve by name to the union of
overloads, so CA is as gold's. Variant of the battery's probeH(int neverCalled) (fixed 25/09/2026): the dead overload
keeps the step method's own parameters so a parameterised gold body's parameter names stay declared; for a gold
method without parameters it is exactly probeH(int neverCalled).
- SM (PHP/Java/C#, a documented limit), for a record whose CREATE steps are bound by two or more different gold
functions: ONE method replaces them all, carrying every one of their annotations or attributes (PHP: their attribute
groups stacked and their docblocks merged into one; Java: the annotations repeated; C#: the attribute lists), with the
name and parameters of the function bound to the first CREATE step and the union of their body lines, in step order.
Condition 4 may fail where the steps' arities differ; where it binds, CA is as gold's (CA matches per gold function,
through the union). Not built (stub_merge says why) when fewer than two gold functions bind CREATE steps, when a gold
function is not a method with a block body, or when the gold items do not match the record's CREATE functions.
Each row has the generation run's output format (model "probe", seed "s42", condition = the probe's name), so score_run
--outputs and score_independent read it like any run; the two scorers must agree on every row."""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.compose import COMMENT, FENCE  # noqa: E402
from v2tools.controls import _escape, body_span, param_names  # noqa: E402
from v2tools.ertree import parser  # noqa: E402
from v2tools.extract import extract  # noqa: E402
from v2tools.insert import split_code  # noqa: E402

T = "`" * 3
FIELDS = {"model": "probe", "seed": "s42", "variant": "main", "prompt_tokens": 0, "new_tokens": 0, "max_new_tokens": 6144,
          "stopped_at_eos": True, "prompts_sha256": "0" * 64, "prompt_sha256": "0" * 64}
LOGGER = {"php": "$this->logger->info('probe');", "java": 'log.info("probe");', "csharp": '_logger.LogInformation("probe");'}


def load(p: Path) -> list[dict]:
    """JSONL rows as 3A reads them: split on "\\n" only from the bytes, the final empty line dropped, every
    other line a JSON value (a blank line inside raises; U+2028, U+0085 or a lone "\\r" of a text stays in it)."""
    lines = p.read_bytes().decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    return [json.loads(line) for line in lines]


def fenced(lang: str, blocks: dict[str, str], prefix: str = "") -> str:
    return "\n\n".join(f"{T}{FENCE[lang]}\n{COMMENT[lang]} FILE: {prefix}{p}\n{c}\n{T}" for p, c in blocks.items())


def rebuild(lang: str, code: str, edit) -> str:
    """`code` with edit(item text) applied to every method, lambda or Ruby step block item, imports kept."""
    imports, items = split_code(lang, code)
    return "\n".join(imports + ([""] if imports else []) + ["\n\n".join(edit(it.text) if it.kind in ("method", "lambda", "block") else it.text
                                                                        for it in items)])


def enclose(lang: str, code: str, head: str, tail: str) -> str:
    """`code` without its import lines between `head` and `tail`, the imports kept in front."""
    imports, items = split_code(lang, code)
    return "\n".join(imports + ([""] if imports else []) + [head + "\n\n".join(it.text for it in items) + tail])


def ruby_add(lines: list[str]):
    """An edit that appends `lines` at the end of a Ruby step block's body (before its last line, the closing end or
    }); a one-line block is left as it is."""
    def edit(text: str) -> str:
        ls = text.split("\n")
        return text if len(ls) < 2 else "\n".join(ls[:-1] + [f"  {l}" for l in lines] + ls[-1:])
    return edit


def ruby_lines(gold: dict[str, str]) -> list[str]:
    """The body lines of the first gold step block (between its first and last line), dedented."""
    for code in gold.values():
        for it in split_code("ruby", code)[1]:
            if it.kind == "block":
                return [l for l in textwrap.dedent("\n".join(it.text.split("\n")[1:-1])).split("\n") if l.strip()]
    return []


def ruby_probes(gold: dict[str, str], stub: dict[str, str]) -> dict[str, str]:
    """R1 and R2, for a Ruby record."""
    out = {"R1_after_return": fenced("ruby", {p: rebuild("ruby", c, ruby_add(["return", *ruby_lines(gold)])) for p, c in stub.items()}),
           "R2_inner_block": fenced("ruby", {p: rebuild("ruby", c, ruby_add(["[1].each do |i|", "  i", "end"])) for p, c in gold.items()})}
    return {k: v for k, v in out.items() if v}


def add_lines(lang: str, lines: list[str]):
    """An edit that appends `lines` at the end of a body (before its closing brace)."""
    def edit(text: str) -> str:
        span = body_span(lang, text)
        if span is None:
            return text
        close = span[1] - 1
        return text[:close].rstrip() + "\n" + "".join(f"    {l}\n" for l in lines) + text[close:]
    return edit


def set_body(lang: str, lines: list[str]):
    def edit(text: str) -> str:
        span = body_span(lang, text)
        return text if span is None else text[:span[0] + 1] + "\n" + "".join(f"    {l}\n" for l in lines) + text[span[1] - 1:]
    return edit


DECOR = re.compile(r"^\s*(?:#\[|@|\[|\*|/\*\*|\*/|//)")
NOISE = {"php": "$probeX = 1;", "java": "int probeX = 1;", "csharp": "var probeX = 1;"}


def body_lines(lang: str, text: str) -> list[str]:
    span = body_span(lang, text)
    return [] if span is None else [l for l in textwrap.dedent(text[span[0] + 1:span[1] - 1]).split("\n") if l.strip()]


def call(lang: str, name: str, args: list[str]) -> str:
    return f"$this->{name}({', '.join('$' + a for a in args)});" if lang == "php" else f"{name}({', '.join(args)});"


def helper(lang: str, text: str, name: str, lines: list[str]) -> str:
    """The method `text` as a private helper `name` with the same parameters (its annotations, attributes and docblock
    dropped) whose body is `lines`."""
    ls = text.split("\n")
    k = next((i for i, l in enumerate(ls) if l.strip() and not DECOR.match(l)), 0)
    sig = "\n".join(ls[k:])
    sig = re.sub(r"(function\s+)\w+(\s*\()", rf"\g<1>{name}\2", sig, count=1) if lang == "php" else re.sub(r"\w+(\s*\()", name + r"\1", sig, count=1)
    sig = re.sub(r"^\s*(?:(?:public|protected|private|internal)\s+)*", "private ", sig, count=1)
    return set_body(lang, lines)(sig)


def helper_shape(lang: str, code: str, chain: list[str], bodies) -> str:
    """Every method of `code` (imports kept) with its body replaced by a call to the first helper of `chain`; the
    helpers take the method's parameters, and bodies(lines, calls) gives their bodies from the method's own body lines
    and the call statement of each helper."""
    imports, items = split_code(lang, code)
    out, extra = [], []
    for k, it in enumerate(items, 1):
        if it.kind != "method" or body_span(lang, it.text) is None:
            out.append(it.text)
            continue
        names = [(n[:1].upper() + n[1:] if lang == "csharp" else n) + ("" if k == 1 else f"_{k}") for n in chain]
        calls = [call(lang, n, list(param_names(lang, it.text))) for n in names]
        out.append(set_body(lang, [calls[0]])(it.text))
        extra += [helper(lang, it.text, n, b) for n, b in zip(names, bodies(body_lines(lang, it.text), calls))]
    return "\n".join(imports + ([""] if imports else []) + ["\n\n".join(out + extra)])


def first_params(lang: str, items) -> str | None:
    """The parameter list, "(…)" as written, of the first method item; None when there is none."""
    for it in items:
        if it.kind != "method":
            continue
        pre = "<?php class __X {\n" if lang == "php" else "class __X {\n"
        src = (pre + it.text + "\n}").encode()
        stack = [parser(lang).parse(src).root_node]
        while stack:
            n = stack.pop()
            if n.type == "method_declaration":
                ps = n.child_by_field_name("parameters")
                return src[ps.start_byte:ps.end_byte].decode("utf-8", "replace") if ps is not None else "()"
            stack.extend(reversed(n.children))
    return None


def pad(text: str, indent: str) -> str:
    return "\n".join(indent + l if l.strip() else "" for l in text.split("\n"))


def holders(lang: str, gold: dict[str, str]) -> dict[str, str]:
    """PA2, JB4 or CS1 (hidden step holders) for a gold made of methods only; nothing otherwise."""
    split = {p: split_code(lang, c) for p, c in gold.items()}
    if lang not in ("php", "java", "csharp") or not all(it.kind == "method" for _, its in split.values() for it in its):
        return {}
    joined = lambda its: "\n\n".join(it.text for it in its)

    def body(its) -> str | None:
        ps = first_params(lang, its)
        if ps is None:
            return None
        if lang == "php":
            return f"private $probeX;\n\npublic function probeInit{ps}: void\n{{\n    $this->probeX = new class {{\n{pad(joined(its), ' ' * 8)}\n    }};\n}}"
        if lang == "java":
            return f"public void probeHolder{ps} {{\n    Object probeX = new Object() {{\n{pad(joined(its), ' ' * 8)}\n    }};\n}}"
        local = [re.sub(r"^(\s*)(?:public|private|protected|internal)\s+(?:static\s+)?", r"\1", it.text, flags=re.M) for it in its]
        sep = "\n\n"
        return f"public void ProbeHolder{ps}\n{{\n{pad(sep.join(local), '    ')}\n}}"
    blocks = {}
    for p, (imports, its) in split.items():
        b = body(its)
        if b is None:
            return {}
        blocks[p] = "\n".join(imports + ([""] if imports else []) + [b])
    name = {"php": "PA2_method_anon_class", "java": "JB4_method_local_anon", "csharp": "CS1_local_function"}[lang]
    return {name: fenced(lang, blocks)}


def dead_overload(gold: dict[str, str]) -> str:
    """O1: the gold, its first step method with a body calling probeH() only; probeH() calls toString() on the first
    field receiver of that body (else `this`); probeH(<its parameters>, int neverCalled), never called, holds the body's
    lines. Empty when no Java method item has a body. Variant of the battery's probeH(int neverCalled), fixed
    25/09/2026: keeping the method's parameters leaves a parameterised body's names declared; without parameters the
    two are the same."""
    for p, code in gold.items():
        imports, items = split_code("java", code)
        k = next((i for i, it in enumerate(items) if it.kind == "method" and body_lines("java", it.text)), None)
        if k is None:
            continue
        lines, params = body_lines("java", items[k].text), first_params("java", [items[k]]) or "()"
        m = re.search(r"(?<![\w.])([a-z_]\w*)\s*\.\s*\w+\s*\(", "\n".join(lines))
        recv = m.group(1) if m and m.group(1) not in ("this", "super") else "this"
        dead = params[:-1] + (", " if params.strip("() ") else "") + "int neverCalled)"
        texts = [it.text for it in items]
        texts[k] = (set_body("java", ["probeH();"])(items[k].text) + f"\n\nprivate void probeH() {{\n    {recv}.toString();\n}}\n\n"
                    f"private void probeH{dead} {{\n" + "".join(f"    {l}\n" for l in lines) + "}")
        return fenced("java", {**gold, p: "\n".join(imports + ([""] if imports else []) + ["\n\n".join(texts)])})
    return ""


COMMENT_NODES = ("comment", "block_comment", "line_comment")


def method_parts(lang: str, text: str) -> tuple[list[str], list[str], str, bool] | None:
    """(the comments before the method, its annotations or attributes, the method without them, whether its body is a
    block) of a PHP, Java or C# method item, read from the tree: a PHP attribute group `#[…]`, a Java annotation, a C#
    attribute list `[…]`, each as written (several lines too); a C# `=> expr;` body or none (abstract) is no block
    (controls.body_span would still cut one out of `=> expr;`); None when the item holds no method."""
    pre = "<?php\nclass __X {\n" if lang == "php" else "class __X {\n"
    src = (pre + text + "\n}\n").encode()
    stack, m = [parser(lang).parse(src).root_node], None
    while stack and m is None:
        n = stack.pop()
        m = n if n.type == "method_declaration" else None
        stack.extend(reversed(n.children))
    if m is None:
        return None
    comments, c = [], m.prev_named_sibling
    while c is not None and c.type in COMMENT_NODES and c.start_byte >= len(pre):
        comments.insert(0, src[c.start_byte:c.end_byte].decode("utf-8", "replace"))
        c = c.prev_named_sibling
    if lang == "java":
        mods = next((x for x in m.children if x.type == "modifiers"), None)
        anns = [x for x in (mods.children if mods is not None else []) if x.type in ("annotation", "marker_annotation")]
    elif lang == "php":
        anns = [g for x in m.children if x.type == "attribute_list" for g in x.children if g.type == "attribute_group"]
    else:
        anns = [x for x in m.children if x.type == "attribute_list"]
    rest, pos = b"", m.start_byte
    for a in anns:
        rest, pos = rest + src[pos:a.start_byte], a.end_byte
    rest += src[pos:m.end_byte]
    body = m.child_by_field_name("body")
    return (comments, [src[a.start_byte:a.end_byte].decode("utf-8", "replace") for a in anns], rest.decode("utf-8", "replace").lstrip(),
            body is not None and body.type in ("block", "compound_statement"))


def merge_comments(lang: str, comments: list[str]) -> list[str]:
    """The comments of several methods for one: each once; in PHP their docblocks merged into ONE, last (Behat reads
    the docblock right before the method)."""
    flat = list(dict.fromkeys(comments))
    docs = [c for c in flat if c.startswith("/**")] if lang == "php" else []
    if len(docs) < 2:
        return flat
    inner = [re.sub(r"^\s*\*?\s?", "", l).rstrip() for c in docs for l in (c[3:-2] if c.endswith("*/") else c[3:]).split("\n")]
    return [c for c in flat if c not in docs] + ["/**\n" + "".join(f" * {l}\n" for l in dict.fromkeys(l for l in inner if l)) + " */"]


def stub_merge(lang: str, record: dict, gold: dict[str, str]) -> tuple[str, str]:
    """SM: (the probe's output text, "") or ("", why it is not built) for a record and its gold
    blocks {path: code}. The gold functions bound to the record's CREATE steps, in step order (record["step_fns"]),
    become one method: the first one's name and parameters, every one's comments (PHP docblocks merged) and
    annotations or attributes, the union of their body lines; it takes the first one's place and the others leave the
    answer (a file left with no item leaves the output)."""
    if lang not in ("php", "java", "csharp"):
        return "", "not PHP, Java or C#"
    if "step_fns" not in record:
        return "", "the record has no step_fns"
    fids = list(dict.fromkeys(f for f, k in zip(record["step_fns"], record["step_kinds"]) if k == "create"))
    if len(fids) < 2:
        return "", "fewer than 2 gold functions bind CREATE steps"
    split, where = {p: split_code(lang, c) for p, c in gold.items()}, {}
    for p in dict.fromkeys(c["path"] for c in record["create"]):
        ids = [c["fn_id"] for c in record["create"] if c["path"] == p]
        if p not in split or len(split[p][1]) != len(ids):
            return "", "the gold items do not match the CREATE functions"
        where.update({f: (p, i) for i, f in enumerate(ids)})
    if any(f not in where for f in fids):
        return "", "a CREATE step is bound by a function outside the answer"
    parts = []
    for f in fids:
        it = split[where[f][0]][1][where[f][1]]
        got = method_parts(lang, it.text) if it.kind == "method" else None
        if got is None:
            return "", "a gold function is not a method"
        if not got[3]:
            return "", "a gold function has no block body"
        parts.append((got, body_lines(lang, it.text)))
    head = merge_comments(lang, [c for (cs, _, _, _), _ in parts for c in cs]) + list(dict.fromkeys(a for (_, anns, _, _), _ in parts for a in anns))
    merged = "\n".join(head + [set_body(lang, [l for _, ls in parts for l in ls])(parts[0][0][2])])
    need = step_imports(lang, [a for (_, anns, _, _), _ in parts for a in anns])
    drop = {where[f] for f in fids[1:]}
    blocks = {}
    for p, (imports, items) in split.items():
        texts = [merged if (p, i) == where[fids[0]] else it.text for i, it in enumerate(items) if (p, i) not in drop]
        if p == where[fids[0]][0]:  # the merged method's file imports what its annotations name (a valid output's)
            imports = imports + [x for x in need if x not in imports]
        if texts:
            blocks[p] = "\n".join(imports + ([""] if imports else []) + ["\n\n".join(texts)])
    return fenced(lang, blocks), ""


def step_imports(lang: str, anns: list[str]) -> list[str]:
    """The imports that the step annotations `anns` (as written, each an attribute group or an annotation) need, as a
    valid output brings them (design specification §2.2 at 8e4405a: names resolve through the
    file's imports after insertion): a merged head may carry annotations of gold functions of another file, whose own
    imports do not follow them. PHP `use Behat\\Step\\K;` per short Given/When/Then, Java `import
    io.cucumber.java.en.K;` per short name, C# `using TechTalk.SpecFlow;`."""
    if lang == "php":
        ks = {m.group(1) for a in anns for m in re.finditer(r"(?:#\[|,)\s*(Given|When|Then)\s*\(", a)}
        return [f"use Behat\\Step\\{k};" for k in sorted(ks)]
    if lang == "java":
        ks = {m.group(1) for a in anns for m in re.finditer(r"@\s*(Given|When|Then|And|But)\s*\(", a)}
        return [f"import io.cucumber.java.en.{k};" for k in sorted(ks)]
    return ["using TechTalk.SpecFlow;"] if lang == "csharp" and anns else []


def replace_helpers(lang: str, code: str, gold: str) -> str:
    """`code` (gold-helpers) with the body of every method that the gold answer `gold` does not declare replaced by a
    meaningless statement."""
    own = {it.name for it in split_code(lang, gold)[1] if it.kind == "method"}
    return rebuild(lang, code, lambda t: t if next((it.name for it in split_code(lang, t)[1]), None) in own else set_body(lang, [NOISE[lang]])(t))


def gold_lines(lang: str, gold: dict[str, str]) -> list[str]:
    """The statements of the first gold function's body, dedented."""
    for code in gold.values():
        for it in split_code(lang, code)[1]:
            span = body_span(lang, it.text) if it.kind in ("method", "lambda") else None
            if span:
                return [l for l in textwrap.dedent(it.text[span[0] + 1:span[1] - 1]).split("\n") if l.strip()]
    return []


def quoted(lang: str, line: str, k: int) -> str:
    if lang == "php":
        return "$probe%d = '%s';" % (k, line.replace("\\", "\\\\").replace("'", "\\'"))
    lit = line.replace("\\", "\\\\").replace('"', '\\"')
    return f'String probe{k} = "{lit}";' if lang == "java" else f'var probe{k} = "{lit}";'


def reuse_item(lang: str, style: str, step: str) -> str:
    pat = "^" + _escape(step) + "$"
    if style == "php_attribute":
        return f"#[Given('/{pat}/')]\npublic function probeReuse(): void\n{{\n}}"
    if style.startswith("php"):
        return f"/**\n * @Given /{pat}/\n */\npublic function probeReuse(): void\n{{\n}}"
    if lang == "java":
        return '@Given("%s")\npublic void probeReuse() {\n}' % pat.replace("\\", "\\\\").replace('"', '\\"')
    return '[Given(@"%s")]\npublic void ProbeReuse()\n{\n}' % pat.replace('"', '""')


def reuse_import(lang: str, style: str) -> str | None:
    """The import of reuse_item's `Given` (design specification §2.2 at 8e4405a: names resolve through the file's
    imports after insertion): without it, in a target file that does not import `Given`, the extra function registers
    nothing and U4 no longer re-binds a REUSE step. None for a PHP docblock, which imports nothing."""
    if style == "php_attribute":
        return "use Behat\\Step\\Given;"
    if lang == "java":
        return "import io.cucumber.java.en.Given;"
    return "using TechTalk.SpecFlow;" if lang == "csharp" else None


def rename(lang: str, code: str) -> str:
    k = [0]

    def sub(m):
        k[0] += 1
        return f"{m.group(1)}probeRenamed{k[0]}{m.group(3)}"
    if lang == "php":
        return re.sub(r"(function\s+)(\w+)(\s*\()", sub, code)
    return re.sub(r"(\b(?:public|private|protected)\s+[\w<>\[\], ]+?\s+)(\w+)(\s*\()", sub, code)


def probes(record: dict, lang: str, style: str, gold: dict[str, str], stub: dict[str, str],
           more: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """probe name -> output text; `more` = the blocks of other controls (stub-v3m, stub-v3all, gold-helpers,
    gold-helpers-renamed) that the helper probes derive from."""
    more = more or {}
    if lang == "ruby":
        return ruby_probes(gold, stub)
    out: dict[str, str] = {}
    first = next(iter(gold))
    items = {p: split_code(lang, c)[1] for p, c in gold.items()}
    wrap = (lambda c: "<?php\n\nclass Wrapper\n{\n" + c + "\n}") if lang == "php" else (lambda c: "public class Wrapper {\n" + c + "\n}")
    out["U1_whole_class"] = fenced(lang, {p: wrap(c) for p, c in gold.items()})
    out["U2_block_per_fn"] = "\n\nSome prose.\n\n".join(fenced(lang, {p: it.text}) for p, its in items.items() for it in its)
    out["U3_dup_fn"] = fenced(lang, {**gold, first: gold[first] + "\n\n" + items[first][0].text}) if items[first] else ""
    reuse = [s for s, k in zip(record["steps"], record["step_kinds"]) if k != "create"]
    if reuse:  # the block opens with the import its `Given` needs (§2.2 at 8e4405a), as a valid output's would
        head = reuse_import(lang, style)
        out["U4_dup_reuse"] = fenced(lang, {**gold, first: (head + "\n\n" if head else "") + gold[first] + "\n\n"
                                            + reuse_item(lang, style, reuse[0])})
    out["U6_unclosed"] = fenced(lang, gold).rstrip()[:-3].rstrip() + "\n"
    other = "src/Other/NotATarget." + {"php": "php", "java": "java", "csharp": "cs"}[lang]
    out["U7_extra_blocks"] = fenced(lang, gold) + f"\n\nRun:\n{T}bash\nvendor/bin/behat\n{T}\n" + fenced(lang, {other: "foo();"})
    out["U8_dot_path"] = fenced(lang, gold, "./")
    out["U11_renamed"] = fenced(lang, {p: rename(lang, c) for p, c in gold.items()})
    out["U12_empty_bodies"] = fenced(lang, {p: rebuild(lang, c, set_body(lang, [])) for p, c in gold.items()})
    lines = gold_lines(lang, gold)
    cm = "//"
    lam = {"php": ("$probe = function () {", "};"), "java": ("Runnable probe = () -> {", "};"), "csharp": ("System.Action probe = () => {", "};")}[lang]
    bodies = {"A_comment": [f"{cm} {l}" for l in lines], "B_string": [quoted(lang, l, k) for k, l in enumerate(lines)],
              "C_after_return": ["return;", *lines], "D_if_false": ["if (false) {", *lines, "}"],
              "E_closure": [lam[0], *lines, lam[1]]}
    for name, extra in bodies.items():
        out[name] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, extra)) for p, c in stub.items()})
    head = "private function probeHelper(): void\n{\n" if lang == "php" else "private void probeHelper() {\n"
    helper = head + "".join(f"    {l}\n" for l in lines) + "}"
    out["F_helper_not_called"] = fenced(lang, {p: c + "\n\n" + helper for p, c in stub.items()})
    out["K_logger_only"] = fenced(lang, {p: rebuild(lang, c, set_body(lang, [LOGGER[lang]])) for p, c in stub.items()})
    if lang == "java":
        imp = "import static org.assertj.core.api.Assertions.assertThat;\n\n"
        out["J_static_import"] = fenced(lang, {p: imp + rebuild(lang, c, set_body(lang, ["assertThat(true).isTrue();"])) for p, c in stub.items()})
    field = {"php": ("private $probeState;", "$this->probeState = 1;"), "java": ("private int probeState;", "this.probeState = 1;"),
             "csharp": ("private int _probeState;", "_probeState = 1;")}[lang]
    out["V1_field_added"] = fenced(lang, {p: field[0] + "\n\n" + rebuild(lang, c, add_lines(lang, [field[1]])) for p, c in gold.items()})
    first = {"php": "$probeA = 1;", "java": "int probeA = 1;", "csharp": "var probeA = 1;"}[lang]
    case_end = [] if lang == "php" else ["    break;"]
    out["V2_switch_return"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["switch (1) {", "case 1:", "    " + first, "    return;",
                                                                                "default:", *["    " + l for l in lines], *case_end, "}"]))
                                            for p, c in stub.items()})
    if lang in ("java", "csharp"):
        ghost = "Object probeX = ghostFieldZ;" if lang == "java" else "var probeX = _ghostFieldZ;"
        out["V3_ghost_initializer"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, [ghost])) for p, c in gold.items()})
        out["V4_char_literal"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["char probeSep = ',';"])) for p, c in gold.items()})
    if lang == "csharp":
        out["V6_arg_ghost"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["System.Console.WriteLine(_ghostFieldZ);"])) for p, c in gold.items()})
    if lang == "php":
        nullsafe = [re.sub(r"(\$this->\w+)->", r"\1?->", l) for l in lines]
        out["V5_nullsafe"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, nullsafe)) for p, c in stub.items()})
        out["V7_colon_return"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["if (true):", "    $probeA = 1;", "    return;", "endif;", *lines]))
                                               for p, c in stub.items()})
        out["V8_label_return"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["probeEnd:", "return;", *lines])) for p, c in stub.items()})
        out["V9_backtick"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["$probe%d = `%s`;" % (k, l.replace("`", "")) for k, l in enumerate(lines)]))
                                           for p, c in stub.items()})
    whole = fenced(lang, gold)
    out["V10_cut_off"] = whole[:int(len(whole) * 0.6)]
    if lang == "php":  # a process exit
        out["X7_exit"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["exit;", *lines])) for p, c in stub.items()})
        out["X10_anon_class_prop"] = fenced(lang, {p: enclose(lang, c, "private $probeX = new class {\n", "\n};") for p, c in gold.items()})
    if lang == "java":  # a process exit, anonymous classes, field initializers
        out["X8_system_exit"] = fenced(lang, {p: rebuild(lang, c, add_lines(lang, ["System.exit(0);", *lines])) for p, c in stub.items()})
        methods = {p: c for p, c in gold.items() if not LAMBDA_STEP.search(c)}  # X3 needs a method to hide: no java8 lambda gold
        out["X3_anon_class_field"] = fenced(lang, {p: enclose(lang, c, "private Object probeX = new Object() {\n", "\n};")
                                                   for p, c in methods.items()}) if len(methods) == len(gold) else ""
        out["X5_field_ghost_init"] = fenced(lang, {p: enclose(lang, c, "private Object probeF = ghostFieldZ.make();\n\n", "") for p, c in gold.items()})
        out["X9_nested_class_field"] = fenced(lang, {p: enclose(lang, c, "private Object probeX = new Object() {\nclass In {\n",
                                                                     "\n}\n};") for p, c in methods.items()}) if len(methods) == len(gold) else ""
        out["X6_logger_field"] = fenced(lang, {p: enclose(lang, c, "private final org.slf4j.Logger probeLog = "
                                                          "org.slf4j.LoggerFactory.getLogger(getClass());\n\n", "") for p, c in gold.items()})
    if lang == "csharp":  # a declared type
        out["X4_struct_wrapper"] = fenced(lang, {p: enclose(lang, c, "public struct Wrapper\n{\n", "\n}") for p, c in gold.items()})
    out.update(holders(lang, gold))  # steps no framework registers
    if lang == "java":
        out["O1_dead_overload"] = dead_overload(gold)
    out["SM_stub_merge"] = stub_merge(lang, record, gold)[0]  # one method for every CREATE step
    # fences
    other = {"php": "private function probeOther(): void\n{\n}", "java": "private void probeOther() {\n}",
             "csharp": "private void ProbeOther()\n{\n}"}[lang]
    parts = [f"{COMMENT[lang]} FILE: {p}\n{c}" for p, c in gold.items()]
    if len(parts) == 1:
        parts.append(f"{COMMENT[lang]} FILE: {next(iter(gold))}\n{other}")
    out["U13_multi_file_fence"] = f"{T}{FENCE[lang]}\n" + "\n\n".join(parts) + f"\n{T}"
    out["U14_long_fence"] = "\n\n".join(f"{T}`{FENCE[lang]}\n{COMMENT[lang]} FILE: {p}\n{c}\n\n/*\n{T}\n*/\n{T}`" for p, c in gold.items())
    # helpers (future methods)
    out["H3_helper_cycle"] = fenced(lang, {p: helper_shape(lang, c, ["probeA", "probeB"], lambda ls, cs: [[cs[1], *ls], [cs[0]]])
                                           for p, c in gold.items()})
    out["H4_helper_chain4"] = fenced(lang, {p: helper_shape(lang, c, ["probeH1", "probeH2", "probeH3", "probeH4"],
                                                            lambda ls, cs: [[cs[1]], [cs[2]], [cs[3]], ls]) for p, c in gold.items()})
    for name, src in (("H6_v3m_helper", "stub-v3m"), ("H7_v3all_helper", "stub-v3all")):
        if more.get(src):
            out[name] = fenced(lang, {p: helper_shape(lang, c, ["probeHelper"], lambda ls, cs: [ls]) for p, c in more[src].items()})
    if more.get("gold-helpers"):
        out["H1_gold_helpers"] = fenced(lang, more["gold-helpers"])
        out["H5_helper_other_body"] = fenced(lang, {p: replace_helpers(lang, c, gold.get(p, "")) for p, c in more["gold-helpers"].items()})
    if more.get("gold-helpers-renamed"):
        out["H2_gold_helpers_renamed"] = fenced(lang, more["gold-helpers-renamed"])
    return {k: v for k, v in out.items() if v}


LAMBDA_STEP = re.compile(r"^\s*(?:Given|When|Then|And|But)\s*\(", re.M)  # a cucumber-java8 lambda step (X3 skips such gold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--partition", required=True, choices=("test", "dev", "aux_test"))
    ap.add_argument("--controls-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sources = {s["repo"]: s for s in json.loads(args.config.read_text())}
    records = {r["id"]: r for r in load(args.phase2 / "records" / f"{args.partition}.jsonl")}
    texts = {(x["record_id"], x["condition"]): x["text"] for x in load(args.controls_dir / "control_outputs.jsonl")}
    rows = []
    for rid, r in sorted(records.items()):
        lang, style = sources[r["repo"]]["language"], sources[r["repo"]]["styles"][0]
        if (rid, "gold") not in texts:
            continue
        targets = list(dict.fromkeys(c["path"] for c in r["create"]))
        gold = extract(texts[(rid, "gold")], targets).by_path()
        stub = extract(texts.get((rid, "stub-v3"), ""), targets).by_path()
        more = {c: extract(texts[(rid, c)], targets).by_path() for c in ("stub-v3m", "stub-v3all", "gold-helpers", "gold-helpers-renamed")
                if (rid, c) in texts}
        for name, text in probes(r, lang, style, gold, stub, more).items():
            rows.append({"record_id": rid, "condition": name, "text": text, **FIELDS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(json.dumps({"records": len({x["record_id"] for x in rows}), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
