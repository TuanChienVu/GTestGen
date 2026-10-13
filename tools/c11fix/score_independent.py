"""Re-score BV from the files alone and compare with the evaluate scores. Imports from v2tools only the Phase 1
primitives patterns, stepdefs and bind (the official pattern compiler, extractor and binder), plus tree-sitter; the ER
tree, the extractor, the class reader, parent resolution and all six conditions are written again here and never
imported from compose, evaluate, insert, extract, outline, state, inherit or symbols. Condition 1 is checked without
building the file: the ER-tree file parses and every inserted block parses as class members (Java lambdas: as
constructor statements; Ruby: as a file) and declares no type of its own (a C# struct included), nor a Java field whose
initializer holds an anonymous class. Calls and member uses are read from the text left by a hand-written scanner
(`clean`: comments and string literals out, and what follows an unconditional return or throw or a process exit), a
method different from evaluate's tree walk; Ruby: the lines after a return/raise/fail line up to the end of its block
are dead (`ruby_dead`). Condition 6 also reads the initializers of the fields, properties and constants of the output,
like a body (`init_texts`). Future methods: the methods ④(a) does not show (`visible`: the visibility rules re-derived
from the sealed symtimes rows and the file versions they list, never from state) are future; an output method with the
key of a future method of the target class replaces it; and every call set, gold or output, and condition 5, open
("self", f) calls to the output's methods or to future ones into their bodies, to depth 3 (`resolver`, `expand`).

This copy of scripts/score_independent.py (the checker pinned with the scoring code) carries one fix, in one place only:
in the member reader, a field access on this, super or base (`this.x`, `super.x`, C# `this.X`, `base.X`) is read before
the names of locals and parameters are skipped, so `this.x = x;` counts x as a member, as symbols.jvm_refs reads it. Run
it with PYTHONPATH set to the repository root, so that it imports this repository's v2tools (docs/REPRODUCE.md).

Besides BV, CA and the six conditions, the static columns of a score row are recomputed and compared when the row has
them: extract_status, insert_status, duplicate_reuse, unresolved, open_resolution, n_calls, call_precision, rouge_l,
exact, conv_style, conv_pattern, conv_naming. Not recomputed (a documented limit): hallucinated and members_used (the
④(a) re-derived here, `view`/`visible`, would allow it; left open), dose (it reads the prompts), the reader and PHP
columns (php_tag_stripped, php_docblock_steps, php_attribute_steps, unread_step_annotations, cuke_escaped_patterns,
cuke_custom_types: covered by the step reader's own tests only), ca_ref_* (the join of ca_ref.jsonl; the CA of the stub
rows it takes ARE compared), excl_bv, excl_ca and exclusions.jsonl (covered by their own tests), er* and the columns
copied from the output row (cap_hit, sha256): none of these is compared. --partition scores one partition's rows of
generation files that also hold its test/aux_test partner's (skipped and counted). Exit 1 when any (record, run) differs
(every differing cell is printed); exit 3 (`refusing: ...`) on an input it cannot compare."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import tree_sitter as ts

try:  # the parser of the `re` module (Python 3.11+; sre_parse before)
    from re import _parser as re_parser
except ImportError:  # pragma: no cover
    import sre_parse as re_parser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.bind import Library  # noqa: E402
from v2tools.patterns import _BEHAT_TOKEN, _CUKE_ARGS, _TOKEN, _ruby_to_python, compile_pattern, turnip_to_regex  # noqa: E402
from v2tools.stepdefs import KEYWORDS, extract as defs_of, file_imports  # noqa: E402

CONDS = ("syntax", "binding", "no_ambiguity", "arity", "grounded", "resolved")
ROW = ("extract_status", "insert_status", "duplicate_reuse", "unresolved", "open_resolution", "n_calls", "call_precision",
       "rouge_l", "exact", "conv_style", "conv_pattern", "conv_naming", "fence_truncated")
ENV = {"GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
KW = KEYWORDS.split("|")


# ----- git and parsing -----------------------------------------------------------------------------------------------

class Git:
    def __init__(self, bare: Path, commit: str):
        self.bare, self.commit, self._t = bare, commit, {}
        out = subprocess.run(["git", "-C", str(bare), "ls-tree", "-r", "--name-only", commit], check=True,
                             capture_output=True, text=True, env=ENV).stdout
        self.tree = out.splitlines()

    def text(self, path: str) -> str | None:
        if path not in self._t:
            r = subprocess.run(["git", "-C", str(self.bare), "cat-file", "blob", f"{self.commit}:{path}"], capture_output=True, env=ENV)
            self._t[path] = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        return self._t[path]

    def history(self, path: str) -> list[tuple[str, int, str]]:
        """(commit, author time, path there) of every commit that touched the file, newest first, following renames;
        a copy, or a rename scored under 60, is the file's first version (as the sealed symtimes rows are built)."""
        out = subprocess.run(["git", "-C", str(self.bare), "log", "--follow", "--format=%x00%H %at", "--name-status", self.commit,
                              "--", path], check=True, capture_output=True, env=ENV).stdout.decode("utf-8", "replace")
        res = []
        for chunk in out.split("\x00")[1:]:
            lines = [l for l in chunk.splitlines() if l.strip()]
            sha, at = lines[0].split()
            st = lines[1].split("\t") if len(lines) > 1 else ["M", path]
            if st[0].startswith("D"):
                continue
            res.append((sha, int(at), st[-1]))
            if st[0][:1] == "C" or st[0][:1] == "R" and int(st[0][1:] or 100) < 60:
                break
        return res

    def blob(self, commit: str, path: str) -> str:
        if (commit, path) not in self._t:
            r = subprocess.run(["git", "-C", str(self.bare), "cat-file", "blob", f"{commit}:{path}"], check=True, capture_output=True, env=ENV)
            self._t[(commit, path)] = r.stdout.decode("utf-8", "replace")
        return self._t[(commit, path)]


_P = {}


def ts_parse(lang: str, text: str):
    if lang not in _P:
        if lang == "php":
            import tree_sitter_php as m
            _P[lang] = ts.Parser(ts.Language(m.language_php()))
        else:
            _P[lang] = ts.Parser(ts.Language(__import__({"java": "tree_sitter_java", "csharp": "tree_sitter_c_sharp",
                                                         "ruby": "tree_sitter_ruby"}[lang]).language()))
    return _P[lang].parse(text.encode())


def nodes(n, types):
    stack = [n]
    while stack:
        x = stack.pop()
        if x.type in types:
            yield x
        stack.extend(reversed(x.children))


# ----- ER tree -------------------------------------------------------------------------------------------------------

def phantom_ids(lang: str, fns: dict, git: Git) -> set[str]:
    """Functions never cut: those the catalog flags dead, and those whose every annotation line starts in a
    comment of the HEAD file (a PHP docblock is code)."""
    bare = sorted(fid for fid, f in fns.items() if not f["annotations"])
    if bare:  # "every annotation in a comment" would hold vacuously (the ER tree fails closed)
        raise ValueError(f"catalog rows with no annotation: {bare[:3]}")
    out = {f["fn_id"] for f in fns.values() if f.get("dead")}
    by = defaultdict(list)
    for f in fns.values():
        by[f["path"]].append(f)
    for path, group in by.items():
        text = git.text(path)
        if text is None:  # its rows could not be checked (the ER tree fails closed)
            raise ValueError(f"not in HEAD: {path}")
        src = text.encode()
        spans = [(c.start_byte, c.end_byte) for c in nodes(ts_parse(lang, text).root_node, ("comment", "line_comment", "block_comment"))
                 if not (lang == "php" and src[c.start_byte:c.start_byte + 3] == b"/**")]
        starts = [0]
        for piece in src.split(b"\n")[:-1]:
            starts.append(starts[-1] + len(piece) + 1)
        for f in group:
            def inside(line):
                s = starts[line - 1]
                e = starts[line] if line < len(starts) else len(src)
                at = s + (len(src[s:e]) - len(src[s:e].lstrip()))
                return any(a <= at < b for a, b in spans)
            if all(inside(a["line"]) for a in f["annotations"]):
                out.add(f["fn_id"])
    return out


def er_tree(r: dict, fns: dict, ghosts: set[str], git: Git) -> tuple[dict[str, str], set[str]]:
    t, cdefs = r["t_vis"], set(r["create_def_ids"])
    early = lambda a: a["time"] is not None and a["time"] < t
    drop, gone = defaultdict(set), set()
    created = {c["fn_id"]: c for c in r["create"]}
    for fid, c in created.items():
        f = fns[fid]
        if c["preexisting"]:
            for a in f["annotations"]:
                if a["def_id"] in cdefs or not early(a):
                    drop[f["path"]].add(a["line"])
                    gone.add(a["def_id"])
        else:
            drop[f["path"]] |= set(range(f["first_line"], f["body_end"] + 1))
            gone |= {a["def_id"] for a in f["annotations"]}
    for fid, f in fns.items():
        if fid in created:
            continue
        if fid in ghosts:
            gone |= {a["def_id"] for a in f["annotations"]}
        elif not any(early(a) for a in f["annotations"]):
            drop[f["path"]] |= set(range(f["first_line"], f["body_end"] + 1))
            gone |= {a["def_id"] for a in f["annotations"]}
    missing = sorted(p for p in drop if git.text(p) is None)
    if missing:
        raise ValueError(f"not in HEAD: {missing[0]}")
    files = {p: "\n".join(l for i, l in enumerate(git.text(p).split("\n"), start=1) if i not in d) for p, d in drop.items()}
    return files, gone


# ----- output blocks -------------------------------------------------------------------------------------------------

FENCE_OPEN = re.compile(r"^\s*(`{3,})[^`]*$")  # three or more backticks, any info string without a backtick
FENCE_CLOSE = re.compile(r"^\s*(`{3,})\s*$")  # backticks only, at least as many as the fence opened with
FILE_LINE = re.compile(r"^\s*(?://|#)\s*FILE:\s*(\S+)\s*$")


def blocks_of(text: str, targets: list[str]) -> tuple[str, dict[str, str], bool]:
    """(status, code per target file, truncated) of an output's fenced blocks. Lines are split on "\\n" only, a trailing
    "\\r" dropped. A fence opens on three or more backticks with any info string without a backtick, and closes on a
    line of backticks only, at least as many. Every FILE line inside a fence starts a file; non-blank lines before the
    fence's first FILE line are a block without a path. Status: none (no fence), malformed (a block without a path or
    outside the targets), unclosed (the last block never closes), multi_file_fence (a fence with more than one FILE
    line), else ok, in that order. A PHP block (of a `.php` target) loses its own `<?php` / `?>` tags (php_block_tags)
    before the blocks of one file are joined (§1.4, per block, fixed 27/09/2026), and a block that is then empty is no
    code. Truncated: a fence opened by backticks alone whose first block has no path, right after a closed fence whose
    first block has one (a three-backtick line of the code closed that fence early)."""
    lines = [l[:-1] if l.endswith("\r") else l for l in text.split("\n")]
    found, i, multi, truncated, after_file = [], 0, False, False, False
    while i < len(lines):
        m = FENCE_OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, j = m.group(1), i + 1
        while j < len(lines) and not ((c := FENCE_CLOSE.match(lines[j])) and len(c.group(1)) >= len(ticks)):
            j += 1
        body, closed = lines[i + 1:j], j < len(lines)
        # output format of §1.4 at 0858bb1: a fence-open line right before a FILE line is a close and reopen, dropped;
        # before the first FILE line, a line of only `<?php` is blank
        body = [l for k, l in enumerate(body) if not (FENCE_OPEN.match(l) and k + 1 < len(body) and FILE_LINE.match(body[k + 1]))]
        starts = [k for k, l in enumerate(body) if FILE_LINE.match(l)]
        head = body[:starts[0]] if starts else body
        blank = lambda l: not l.strip() or (starts and re.fullmatch(r"\s*<\?php\s*", l, re.I))
        segs = [(None, "\n".join(head).rstrip())] if not starts or not all(blank(l) for l in head) else []
        segs += [(FILE_LINE.match(body[a]).group(1).removeprefix("./"), "\n".join(body[a + 1:b]).rstrip())
                 for a, b in zip(starts, starts[1:] + [len(body)])]
        multi |= len(starts) > 1
        truncated |= after_file and lines[i].strip() == ticks and segs[0][0] is None and (not closed or bool(starts))  # §1.4
        after_file = segs[0][0] is not None
        found += [(path, code, closed or k < len(segs) - 1) for k, (path, code) in enumerate(segs)]
        i = j + 1
    if not found:
        return "none", {}, False
    status = "malformed" if any(p not in targets for p, _, _ in found) else "unclosed" if not found[-1][2] else \
        "multi_file_fence" if multi else "ok"
    merged = defaultdict(list)
    for p, code, _ in found:
        code = php_block_tags(code) if p in targets and p.endswith(".php") else code  # §1.4: tags stripped per block
        if p in targets and code.strip():
            merged[p].append(code)
    return status, {p: "\n\n".join(v) for p, v in merged.items()}, truncated


IMPORTS = {"php": r"^\s*use\s+(?:function\s+|const\s+)?[\\\w]+(?:\\\{[^}]*\})?(?:\s+as\s+\w+)?\s*;\s*$",
           "java": r"^\s*import\s+(?:static\s+)?[\w.]+(?:\.\*)?\s*;\s*$",  # `.*` only at the end
           "csharp": r"^\s*using\s+(?:static\s+)?[\w.]+(?:\s*=\s*[\w.<>, ]+)?\s*;\s*$",
           "ruby": r"^\s*require(?:_relative)?\s*\(?\s*['\"][^'\"]+['\"]\s*\)?\s*$"}
LAMBDA_START = re.compile(r'^[ \t]*(?:Given|When|Then|And|But)\(\s*"(?:[^"\\]|\\.)*"\s*,\s*(\([^)]*\)|\w+)\s*->\s*\{', re.M)


def lambda_matches(text: str) -> list:
    """The java8 step registrations (LAMBDA_START matches) that are steps: a match whose line starts in
    code (not in a string, character literal or comment) at brace depth 0, or at depth 1 inside a `{` whose header (from
    the end of the previous depth-0 statement or member: the last `;` or `}` at brace and parenthesis depth 0) is one
    constructor declaration. A registration in a `static {}` or instance initializer, a method body, an anonymous
    class or a nested block stays where it is (and never binds)."""
    depth_at, header, stack, seg, parens, quote, esc, j = {}, {}, [], 0, 0, None, False, 0
    while j < len(text):
        c = text[j]
        if quote is None and (j == 0 or text[j - 1] == "\n"):
            depth_at[j] = (len(stack), stack[-1] if stack else None)
        if quote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
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
                header[j] = seg
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

    def ctor(h: str) -> bool:
        tree = ts_parse("java", "class __X {\n" + h + "{}\n}\n")
        body = next(nodes(tree.root_node, ("class_body",)), None)
        kids = [x for x in (body.named_children if body is not None else []) if "comment" not in x.type]
        return not tree.root_node.has_error and len(kids) == 1 and kids[0].type == "constructor_declaration"
    out = []
    for m in LAMBDA_START.finditer(text):
        depth, brace = depth_at.get(m.start(), (None, None))
        if depth == 0 or depth == 1 and ctor(text[header[brace]:brace]):
            out.append(m)
    return out


def close_brace(s: str, i: int) -> int:
    depth, q, esc, j = 0, None, False, i
    while j < len(s):
        c = s[j]
        if q:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == q:
                q = None
        elif s.startswith("//", j):
            j = s.find("\n", j)
            if j < 0:
                break
        elif s.startswith("/*", j):
            j = s.find("*/", j) + 1
            if j <= 0:
                break
        elif c in "\"'":
            q = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


QUOTES = {"php": "\"'`", "java": "\"'", "csharp": "\"'", "ruby": "\"'"}


PHP_OPEN_TAG = re.compile(r"<\?(?:php\b|=)?", re.I)
CS_RAW = re.compile(r'\$*("{3,})')  # the opening of a C# 11 raw string literal

# Own way (fixed 26/09/2026): a PHP output block's own `<?php` open tag (any case, the
# spaces/newline right after it, or the spaces before code on its line) and its trailing `?>` (the whitespace around
# it and one line break before it) are no code either: every reader of a raw output block strips them first, so a
# model that wraps its answer in a full `<?php ... ?>` file is read the same as one that does not.
# Aligned with `insert.php_open_tag` of v2tools (`PHP_OPEN`/`PHP_CLOSE`) so both scorers agree
# ("at the start"/"at the end" of a block is after/before ANY leading/trailing whitespace, blank lines
# included): a blank line (or several) before `<?php`, and after `?>`, is part of the tag that gets stripped, not
# left as dead code that trips the syntax check.
PHP_BLOCK_OPEN = re.compile(r"\A\s*<\?php(?:[ \t]*\n|[ \t]+|\Z)", re.I)
PHP_BLOCK_CLOSE = re.compile(r"\n?[ \t]*\?>\s*\Z")


def php_block_tags(code: str) -> str:
    """`code` without its own leading `<?php` tag and trailing `?>` (PHP_BLOCK_OPEN, PHP_BLOCK_CLOSE); every other
    language, and PHP code that has neither, is returned as is."""
    return PHP_BLOCK_CLOSE.sub("", PHP_BLOCK_OPEN.sub("", code, count=1), count=1)


def clean(lang: str, text: str) -> str:
    """The code that can run: comments become spaces, every string literal becomes an empty one (PHP heredocs, Java
    text blocks and C# verbatim, interpolated or raw strings included), PHP inline HTML (from `?>` to the next `<?php`,
    `<?` or `<?=`, whose echoed expression is code; a `//` or `#` comment ends at `?>`) becomes spaces, then
    `drop_dead`."""
    out, i, n = [], 0, len(text)
    while i < n:
        c, two = text[i], text[i:i + 2]
        if two == "//" or (lang == "php" and c == "#" and text[i + 1:i + 2] != "["):
            j = text.find("\n", i)
            j = n if j < 0 else j
            if lang == "php" and 0 <= text.find("?>", i, j):
                j = text.find("?>", i, j)  # PHP leaves code at `?>`, even inside a one-line comment
            out.append(" " * (j - i))
            i = j
        elif lang == "php" and two == "?>":
            m = PHP_OPEN_TAG.search(text, i + 2)
            j = n if m is None else m.end()
            out.append(re.sub(r"[^\n]", " ", text[i:j]))  # inline HTML, the closing and opening tags: no code
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif lang == "php" and text.startswith("<<<", i) and re.match(r"<<<[ \t]*(['\"]?)(\w+)\1\r?\n", text[i:]):
            ident = re.match(r"<<<[ \t]*(['\"]?)(\w+)", text[i:]).group(2)
            end = re.compile(rf"^[ \t]*{ident}\b", re.M).search(text, text.find("\n", i) + 1)
            out.append("''")
            i = n if end is None else end.end()
        elif lang == "java" and text.startswith('"""', i):
            j = text.find('"""', i + 3)
            out.append('""')
            i = n if j < 0 else j + 3
        elif lang == "csharp" and c in '$"' and CS_RAW.match(text, i):  # a raw string, interpolated or not:
            q = CS_RAW.match(text, i)  # it ends at as many quotes as open it
            j = text.find(q.group(1), q.end())
            out.append('""')
            i = n if j < 0 else j + len(q.group(1))
        elif lang == "csharp" and re.match(r'(?:\$@|@\$|@|\$)"', text[i:i + 3]):
            q = text.index('"', i)
            verbatim, j = "@" in text[i:q], q + 1
            while j < n:
                if verbatim and text[j:j + 2] == '""' or not verbatim and text[j] == "\\":
                    j += 2
                elif text[j] == '"':
                    break
                else:
                    j += 1
            out.append('""')
            i = j + 1
        elif c in QUOTES[lang]:
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append("'x'" if c == "'" and lang in ("java", "csharp") else c + c)  # a Java/C# char literal stays one
            i = j + 1
        else:
            out.append(c)
            i += 1
    return drop_dead("".join(out), lang)


CASE_STOP = re.compile(r"(?:case\b|default\s*(?::|->))", re.I)
ALT_OPEN = re.compile(r"(if|elseif|while|for|foreach|switch)\s*\(", re.I)
ALT_END = re.compile(r"end(if|while|for|foreach|switch)\b", re.I)


def closes_at_end(head: str, open_at: int) -> bool:
    """The parenthesis opened at `open_at` closes on the last character of `head`."""
    depth = 0
    for k in range(open_at, len(head)):
        depth += (head[k] == "(") - (head[k] == ")")
        if depth == 0:
            return k == len(head) - 1
    return False


def paren_colon(s: str, k: int) -> bool:
    """The parenthesis opened at `k` closes and is followed by a `:` that is not `::` (a PHP `:` block head)."""
    depth = 0
    for j in range(k, len(s)):
        depth += (s[j] == "(") - (s[j] == ")")
        if depth == 0:
            rest = s[j + 1:].lstrip()
            return rest[:1] == ":" and rest[1:2] != ":"
    return False


def alt_word(s: str, k: int) -> tuple[str, str] | None:
    """The PHP `:` block keyword starting at `k`: ("open", kw) for `if/while/for/foreach/switch (…):`, ("cont", kw) for
    `else:` and `elseif (…):`, ("end", kw) for `endif` and the other end… keywords; None otherwise (a braced
    `else {` or `if (…) {` included)."""
    m = ALT_END.match(s, k)
    if m:
        return "end", m.group(1).lower()
    m = re.compile(r"else\b", re.I).match(s, k)
    if m and not re.compile(r"\s*if\b", re.I).match(s, m.end()):
        rest = s[m.end():].lstrip()
        return ("cont", "else") if rest[:1] == ":" and rest[1:2] != ":" else None
    m = ALT_OPEN.match(s, k)
    if m and paren_colon(s, m.end() - 1):
        return ("cont" if m.group(1).lower() == "elseif" else "open"), m.group(1).lower()
    return None


def colon_kind(head: str, php: bool) -> str | None:
    """What a `:` ends, from the statement text before it: a case label, a PHP `:` block opener or its `else`/`elseif`
    continuation, a statement label, or nothing that starts a statement (a ternary, `assert`, a for-each, an argument)."""
    if re.match(r"(?:case\b|default$)", head, re.I if php else 0):
        return "case"
    if php and re.fullmatch(r"else", head, re.I):
        return "cont"
    m = ALT_OPEN.match(head) if php else None
    if m and closes_at_end(head, m.end() - 1):
        return "cont" if m.group(1).lower() == "elseif" else "alt"
    return "label" if re.fullmatch(r"[A-Za-z_]\w*", head) else None


EXITS = {"php": re.compile(r"(?:@\s*)*(?:exit|die)\b\s*(\(?)", re.I),  # also under any number of `@`
         "java": re.compile(r"(?:java\s*\.\s*lang\s*\.\s*)?(?:System\s*\.\s*exit|Runtime\s*\.\s*getRuntime\s*\(\s*\)\s*\.\s*(?:exit|halt))"
                            r"\s*(\()"),
         "csharp": re.compile(r"(?:(?:global\s*::\s*)?System\s*\.\s*)?Environment\s*\.\s*(?:Exit|FailFast)\s*(\()")}
SEMI = re.compile(r"\s*;")


def ends_flow_at(s: str, i: int, lang: str) -> bool:
    """The statement starting at `i` ends the flow: `return`, `throw`, or a statement that is only a
    process exit: PHP `exit` or `die`, any case, bare or with an argument list, also under `@`; Java `System.exit(…)`,
    `Runtime.getRuntime().exit(…)` or `.halt(…)`, each also after `java.lang.`; C# `Environment.Exit(…)` or
    `Environment.FailFast(…)`, also `System.Environment.…` and `global::System.Environment.…`."""
    if re.match(r"(?:return|throw)\b", s[i:i + 7], re.I if lang == "php" else 0):  # PHP keywords: any case
        return True
    m = EXITS[lang].match(s, i) if lang in EXITS else None
    if m is None:
        return False
    j, depth = m.end(), 1 if m.group(1) else 0
    while depth and j < len(s):  # the exit's own argument list, then nothing but the `;`
        depth += (s[j] == "(") - (s[j] == ")")
        j += 1
    return depth == 0 and SEMI.match(s, j) is not None


def drop_dead(s: str, lang: str = "") -> str:
    """Drop what follows a `return`, a `throw` or a process exit (`ends_flow_at`) that starts a statement, up to the end
    of its block: the brace that closes it; inside a switch case also the next `case`/`default` label; inside a PHP `:`
    block also its `else:`, `elseif (…):` or end… keyword. Inside the dropped part, a nested block (braces, or a PHP `:`
    block up to its end… keyword) is dropped whole. A statement starts after `{`, `;`, `}`, or the `:` of a case label,
    a statement label or a PHP `:` block; the `:` of a ternary does not start one, and neither a `;` nor a `:` inside
    parentheses (a `for (…; …; …)` head, an argument) ends or starts one. Strings and comments are already gone, so
    braces, colons and semicolons are code."""
    php = lang == "php"
    out, i, n, start, stmt = [], 0, len(s), True, 0
    # one frame per open brace: inside a case body; for each open PHP `:` block, the case flag it will restore at its end;
    # open parentheses; for a brace opened inside parentheses (a closure argument), the statement start to go back to
    frames: list[dict] = [{"case": False, "alt": [], "par": 0, "stmt": None}]
    while i < n:
        c = s[i]
        if start and ends_flow_at(s, i, lang):
            depth, j = 0, i
            while j < n and not (s[j] == ";" and depth == 0):
                depth += (s[j] in "([{") - (s[j] in ")]}")
                j += 1
            out.append(s[i:j + 1])
            fr, depth, nest, k = frames[-1], 0, 0, j + 1  # nest: PHP `:` blocks opened inside the dropped part
            while k < n and not (s[k] == "}" and depth == 0):
                if depth == 0 and not (s[k - 1].isalnum() or s[k - 1] in "_$>:"):
                    w = alt_word(s, k) if php else None
                    if nest == 0 and (fr["case"] and CASE_STOP.match(s, k) or fr["alt"] and w and w[0] != "open"):
                        break
                    nest = max(0, nest + (w is not None and w[0] == "open") - (w is not None and w[0] == "end"))
                depth += (s[k] == "{") - (s[k] == "}")
                k += 1
            i, start, stmt = k, True, k
            continue
        out.append(c)
        closed = None
        if c == "{":
            frames.append({"case": False, "alt": [], "par": 0, "stmt": stmt if frames[-1]["par"] else None})
        elif c == "}" and len(frames) > 1:
            closed = frames.pop()
        if c in "()":
            frames[-1]["par"] = max(0, frames[-1]["par"] + (c == "(") - (c == ")"))
        if closed is not None and closed["stmt"] is not None:  # a closure argument ends: back in the enclosing statement
            start, stmt = False, closed["stmt"]
        elif c in "{}":
            start, stmt = True, i + 1
        elif c in ";:" and frames[-1]["par"]:
            start = False
        elif c == ";":
            head = s[stmt:i].strip()
            if php and re.match(r"(?:case\b|default$)", head, re.I):  # PHP also accepts `case x;`
                frames[-1]["case"] = True
            m = ALT_END.fullmatch(head) if php else None
            if m and frames[-1]["alt"]:
                frames[-1]["case"] = frames[-1]["alt"].pop()
            start, stmt = True, i + 1
        elif c == ":" and s[i - 1:i] != ":" and s[i + 1:i + 2] != ":":
            kind = colon_kind(s[stmt:i].strip(), php)
            if kind == "case":
                frames[-1]["case"] = True
            elif kind == "alt":
                frames[-1]["alt"].append(frames[-1]["case"])
                frames[-1]["case"] = frames[-1]["case"] and not re.match(r"switch\b", s[stmt:i].strip(), re.I)
            if kind:
                start, stmt = True, i + 1
            else:
                start = False
        elif not c.isspace():
            start = False
        i += 1
    return "".join(out)


RB_BLANK = ("comment", "string", "heredoc_body")
RB_END = re.compile(r"\s*(?:return|raise|fail)\b")
RB_MODIFIER = re.compile(r" (?:if|unless|while|until) ")
RB_OPEN = re.compile(r"\s*(?:if|unless|while|until|case|begin|def|class|module)\b")
RB_STOP = re.compile(r"\s*(?:else|elsif|when|in|rescue|ensure)\b")
RB_NEST = re.compile(r"\b(?:do|end)\b|[{}]")


def ruby_dead(text: str, root) -> set[int]:
    """The rows (0-based) of dead lines, read by lines once comments, strings and heredocs are blanked. A
    line whose first token is `return`, `raise` or `fail` with no trailing modifier (` if `, ` unless `, ` while `,
    ` until ` after the keyword) ends the flow: every following line is dead up to the line holding the end/} that
    closes its innermost block, or a line starting with else/elsif/when/in/rescue/ensure at its depth. Nesting: `do`,
    `{`, and a line starting with if/unless/while/until/case/begin/def/class/module open; `end` and `}` close; the
    tokens after the keyword count too (a return line that closes its own block kills nothing)."""
    b, stack = bytearray(text.encode()), [root]
    while stack:
        x = stack.pop()
        if x.type in RB_BLANK:
            b[x.start_byte:x.end_byte] = re.sub(rb"[^\n]", b" ", bytes(b[x.start_byte:x.end_byte]))
        else:
            stack.extend(x.children)
    lines = b.decode("utf-8", "replace").split("\n")

    def nest(line: str) -> tuple[int, int]:  # (lowest running depth, capped at 0; net depth change) of one line
        run = low = int(RB_OPEN.match(line) is not None)
        for t in RB_NEST.findall(line):
            run += 1 if t in ("do", "{") else -1
            low = min(low, run)
        return min(low, 0), run
    dead, i = set(), 0
    while i < len(lines):
        m = RB_END.match(lines[i])
        i += 1
        if m is None or RB_MODIFIER.search(lines[i - 1], m.end()):
            continue
        low, depth = nest(lines[i - 1][m.end():])
        while low == 0 and i < len(lines) and not (depth == 0 and RB_STOP.match(lines[i])):
            low, net = nest(lines[i])
            low = min(0, depth + low)
            if low == 0:
                depth += net
                dead.add(i)
                i += 1
    return dead


LOGGER_NAME = re.compile(r"^(log|logger|LOG|LOGGER|_log|_logger)$")
EXPAND_DEPTH = 3  # helper bodies are read to depth 3; a cycle or a deeper call stays ("self", f)
# what every class inherits from its language's root object (java.lang.Object, System.Object): resolved
# for condition 6, never an open path, never a project member (condition 5) nor a call of the CA sets
ROOT = {"java": frozenset({"getClass", "hashCode", "equals", "toString", "clone", "finalize", "notify", "notifyAll", "wait"}),
        "csharp": frozenset({"GetType", "GetHashCode", "Equals", "ToString", "MemberwiseClone", "ReferenceEquals", "Finalize"})}


def loggers_of(lang: str, text: str) -> set[str]:
    """Logger fields by declared type: SLF4J, Log4j or JUL `Logger` (Java), PSR-3 `LoggerInterface` (PHP)."""
    if lang == "php":
        return set(re.findall(r"\b(?:private|protected|public|readonly|var)\s+(?:readonly\s+)?\??\\?(?:Psr\\Log\\)?LoggerInterface\s+\$(\w+)", text))
    if lang != "java":
        return set()
    known = ("org.slf4j.Logger", "org.apache.logging.log4j.Logger", "org.apache.log4j.Logger", "java.util.logging.Logger")
    imported = any(re.search(r"^\s*import\s+" + re.escape(k) + r"\s*;", text, re.M) for k in known)
    return {nm for t, nm in re.findall(r"\b((?:[\w.]+\.)?Logger)\s+(\w+)\s*[=;]", text) if t in known or (t == "Logger" and imported)}


PARAMS = {"simple_parameter", "variadic_parameter", "property_promotion_parameter", "formal_parameter", "spread_parameter",
          "parameter"}


def regex_args(regex: str, lang: str) -> int:
    """N of condition 4 for a regex step pattern, as each framework hands its groups over (design specification §2.2 at
    448165c, fixed 25/09/2026): Behat (PHP) and SpecFlow (C#) every capture group, nested ones included, a named group
    once; cucumber-jvm (Java) and cucumber (Ruby) the capture groups no other capture group holds, a non-capturing
    group, an alternation, a repeat, an atomic group, a lookaround or a conditional hiding nothing. Read from the parse
    tree of Python's `re` module (the pattern as the step library compiles it); a pattern it rejects hands over nothing
    (it never binds)."""
    return capture_groups(regex, nested=lang in ("php", "csharp"))


def capture_groups(regex: str, nested: bool) -> int:
    """The capture groups of a regex: all of them (`nested`), or those no other capture group holds."""
    def count(items) -> int:
        n = 0
        for op, av in items:
            kind = str(op)
            if kind == "SUBPATTERN":
                n += 1 + count(av[-1]) if av[0] is not None and nested else 1 if av[0] is not None else count(av[-1])
            elif kind == "BRANCH":
                n += sum(count(b) for b in av[1])
            elif kind in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
                n += count(av[2])
            elif kind == "ATOMIC_GROUP":
                n += count(av)
            elif kind in ("ASSERT", "ASSERT_NOT"):
                n += count(av[1])
            elif kind == "GROUPREF_EXISTS":
                n += count(av[1]) + (count(av[2]) if av[2] is not None else 0)
        return n
    try:
        return count(re_parser.parse(_ruby_to_python(regex)))
    except re.error:  # a pattern the parser rejects never binds (the step library cannot compile it): no argument,
        return 0  # and no hand scanning of parentheses (design specification §2.2 at 448165c)


def arity_ok(lang: str, u: dict, n: int, table: bool = False) -> bool:
    """Condition 4's count check per framework (design specification §2.2 at 448165c; the table parameter rule at
    fe73dc9): PHP required <= N, no upper bound, and a step with a table or doc string (`table`) needs a parameter that
    takes it, one type-hinted TableNode or PyStringNode, or N parameters in all; C# every declared parameter, optional
    ones and a `params` array (one) included, makes N; Java the declared parameters make N (varargs count one); Ruby the
    required block parameters make N, or N >= required when the block has a splat."""
    if lang == "php":
        return u["required"] <= n and (not table or u.get("takes", False) or u["total"] >= n)
    if lang == "ruby":
        return u["required"] == n or u["total"] == math.inf and n >= u["required"]
    return u["count"] == n


RUBY_STEP = {"Given", "When", "Then", "And", "But"}


def ruby_arity(text: str) -> tuple[int, int, float]:
    """(parameters, required, total) of a Ruby step block as Proc#arity reads them: a plain or destructured parameter is
    required, the required keywords together count one, a default, **kw or &blk does not count; a splat leaves no upper
    bound (total inf), else the total adds the defaults."""
    b = text.encode()
    kinds = [(c.type, c.child_by_field_name("value") is not None) for c in ruby_step_nodes(text)]
    req = sum(k in ("identifier", "destructured_parameter") for k, _ in kinds) + any(k == "keyword_parameter" and not v for k, v in kinds)
    total = math.inf if any(k == "splat_parameter" for k, _ in kinds) else req + sum(k == "optional_parameter" for k, _ in kinds)
    return len(kinds), req, total


def ruby_step_params(text: str) -> list[str]:
    """The parameters (texts) of a Ruby step."""
    b = text.encode()
    return [b[c.start_byte:c.end_byte].decode("utf-8", "replace") for c in ruby_step_nodes(text)]


def ruby_step_nodes(text: str) -> list:
    """The parameters of a Ruby step: the block parameters of the block attached to the first call to
    Given, When, Then, And or But, read from the tree (never a `do|` inside the step's regex; on the next line
    too; never an inner block's |i|)."""
    b = text.encode()
    T = lambda n: b[n.start_byte:n.end_byte].decode("utf-8", "replace")
    stack = [ts_parse("ruby", text).root_node]
    while stack:
        n = stack.pop()
        name = n.child_by_field_name("method") if n.type == "call" else None
        if name is not None and T(name) in RUBY_STEP:
            blk = n.child_by_field_name("block")
            ps = next((c for c in blk.named_children if c.type == "block_parameters"), None) if blk is not None else None
            return [c for i, c in enumerate(ps.children) if c.is_named and ps.field_name_for_child(i) != "locals"] \
                if ps is not None else []  # never a block-local variable after `;` (|n; tmp|; Proc#arity ignores it)
        stack.extend(reversed(n.children))
    return []


TABLE_NODES = {"tablenode", "pystringnode"}


def php_takes_table(typ: str) -> bool:
    """The table parameter rule on a PHP parameter's type text: one of its alternatives (the names between |, &, ?,
    parentheses and spaces: unions, intersections, DNF, nullable) is, after its last backslash, TableNode or
    PyStringNode, in any case (PHP class names ignore case). Any class of that short name counts (a documented
    limit)."""
    return any(name.split("\\")[-1].casefold() in TABLE_NODES for name in re.findall(r"[^\s|&?()]+", typ))


def lambda_params(stmt: str) -> int | None:
    """The parameters of the first lambda of a java8 step statement, read from its tree: no
    parameter list 0, a bare name 1, else the formal, spread or inferred parameters of its list, so the commas of a
    generic type count nothing. None when the statement holds no lambda."""
    root = ts_parse("java", "class __X { __X() {\n" + stmt + "\n} }").root_node
    lam = next(nodes(root, ("lambda_expression",)), None)
    if lam is None:
        return None
    ps = lam.child_by_field_name("parameters")
    if ps is None or ps.type == "identifier":
        return 0 if ps is None else 1
    return sum(1 for c in ps.named_children if c.type in ("formal_parameter", "spread_parameter", "identifier"))


def n_params(m) -> int:
    """The parameter count of a method key: declared parameters only (no comment, no C# `params` array)."""
    ps = m.child_by_field_name("parameters")
    return sum(p.type in PARAMS for p in (ps.named_children if ps is not None else []))


def units(lang: str, code: str) -> tuple[list[str], list[dict], bool]:
    """(import lines, units, parses) of one block. A unit is {kind, name, arity, required, total, text}; Ruby: the step
    blocks (one nested in another is part of it) and the lines left over, as one "raw" unit. PHP: read without a
    leading `<?php` or trailing `?>`."""
    if lang == "php":
        code = php_block_tags(code)
    lines = code.split("\n")
    k = 0
    while k < len(lines) and (not lines[k].strip() or re.match(IMPORTS[lang], lines[k])):
        k += 1
    imports = [l.strip() for l in lines[:k] if l.strip()]
    rest = "\n".join(lines[k:])
    out = []
    if lang == "ruby":
        ok = not ts_parse("ruby", rest).root_node.has_error
        rl, used = rest.split("\n"), set()
        for d in sorted(defs_or_none("", rest, "ruby_block"), key=lambda d: d.line):
            if d.line in used:
                continue
            used |= set(range(d.line, d.body_end + 1))
            text = "\n".join(rl[d.line - 1:d.body_end])
            n, req, total = ruby_arity(text)
            out.append({"kind": "block", "name": None, "arity": n, "required": req, "total": total, "text": text})
        left = "\n".join(l for i, l in enumerate(rl, start=1) if i not in used)
        if left.strip():
            out.append({"kind": "raw", "name": None, "arity": 0, "required": 0, "total": 0, "text": left.strip("\n")})
        return imports, out, ok
    ok = True
    if lang == "java":
        keep, pos = [], 0
        for m in lambda_matches(rest):
            if m.start() < pos:
                continue
            c = close_brace(rest, m.end() - 1)
            tail = re.match(r"\s*\)\s*;", rest[c + 1:]) if c >= 0 else None
            if tail is None:
                continue
            end = c + 1 + tail.end()
            ps = [p for p in m.group(1).strip("()").split(",") if p.strip()]  # the item's arity, as insert.split_code
            stmt = rest[m.start():end]
            ok &= not ts_parse("java", "class __X { __X() {\n" + stmt + "\n} }").root_node.has_error
            n = lambda_params(stmt)  # condition 4 reads the lambda's tree
            n = len(ps) if n is None else n
            out.append({"kind": "lambda", "name": None, "arity": len(ps), "required": n, "total": n, "count": n, "text": stmt})
            keep.append(rest[pos:m.start()])
            pos = end
        keep.append(rest[pos:])
        rest = "".join(keep)
    if rest.strip():
        pre = "<?php\nclass __X {\n" if lang == "php" else "class __X {\n"
        src = pre + rest + "\n}\n"
        tree = ts_parse(lang, src)
        ok &= not tree.root_node.has_error
        wrapper = next(nodes(tree.root_node, ("class_declaration",)), None)
        body = next((x for x in (wrapper.children if wrapper is not None else []) if x.type in ("declaration_list", "class_body")), None)
        ok &= not any(raw_member(lang, x) for x in (body.named_children if body is not None else []))
        b = src.encode()
        for m in nodes(tree.root_node, ("method_declaration", "constructor_declaration")):
            if m.parent is None or wrapper is None or m.parent.parent is None or m.parent.parent.id != wrapper.id:
                continue
            ps = m.child_by_field_name("parameters")
            plist = [p for p in (ps.named_children if ps is not None else []) if p.type != "comment"]
            texts = [b[p.start_byte:p.end_byte].decode("utf-8", "replace") for p in plist]
            var = any(c.type == "params" for c in ps.children) if ps is not None else False  # C# params T[] x
            var |= any(p.type in ("variadic_parameter", "spread_parameter") or t.lstrip().startswith("params ") for p, t in zip(plist, texts))
            ordinary = [(p, t) for p, t in zip(plist, texts) if p.type in ("simple_parameter", "property_promotion_parameter", "formal_parameter", "parameter")]
            name = m.child_by_field_name("name")
            out.append({"kind": "method", "name": b[name.start_byte:name.end_byte].decode(), "arity": n_params(m),
                        "ctor": m.type == "constructor_declaration" or b[name.start_byte:name.end_byte].lower() == b"__construct",
                        # optional only with a default in the tree (a direct `=`: PHP default_value, C#), never for
                        # a `=` inside an attribute; a Java parameter never
                        "required": sum(not any(c.type == "=" for c in q.children) for q, _ in ordinary),
                        "total": math.inf if var else len(ordinary), "text": b[span_of(b, m):m.end_byte].decode("utf-8", "replace"),
                        "count": sum(not t.lstrip().startswith("params ") for _, t in ordinary) + var,  # every declared one
                        "takes": any(php_takes_table(b[t.start_byte:t.end_byte].decode("utf-8", "replace"))
                                     for p, _ in ordinary if (t := p.child_by_field_name("type")) is not None)})  # PHP table
    return imports, out, ok


TYPES = ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "trait_declaration")
BODIES = ("declaration_list", "class_body", "enum_body", "interface_body")
RAW_TYPES = TYPES + ("struct_declaration", "record_struct_declaration")


def raw_member(lang: str, x) -> bool:
    """A class-body node of a block that condition 1 refuses: a declared type, a C# struct or record struct included,
    or a Java field whose initializer holds an anonymous class body `new T(…) { … }` that declares a method, which can
    never bind as a step of the class. A body without a method (`new TypeReference<…>() {}`, fields only) leaves the
    field an ordinary member."""
    return x.type in RAW_TYPES or lang == "java" and x.type == "field_declaration" and any(
        c.type == "class_body" and any(m.type == "method_declaration" for m in c.named_children)
        for n in nodes(x, ("object_creation_expression",)) for c in n.children)


def span_of(b: bytes, n) -> int:
    """Start of a member, a /** docblock right before it included."""
    prev = n.prev_named_sibling
    if prev is not None and prev.type in ("comment", "block_comment") and b[prev.start_byte:prev.start_byte + 3] == b"/**" \
            and not b[prev.end_byte:n.start_byte].strip():
        return prev.start_byte
    return n.start_byte


def items_of(lang: str, code: str) -> list[dict]:
    """What the insertion rule inserts from one block, {kind, name, arity, text}: Java lambda steps (up to the first
    whose braces never close), then the class members, each a "method", a "member", or "raw" for a declared type; the
    rest is one "raw" item when it does not parse as class members. Ruby: its units. PHP: read without a leading
    `<?php` or trailing `?>`."""
    if lang == "ruby":
        return units(lang, code)[1]
    if lang == "php":
        code = php_block_tags(code)
    lines = code.split("\n")
    k = 0
    while k < len(lines) and (not lines[k].strip() or re.match(IMPORTS[lang], lines[k])):
        k += 1
    rest, out = textwrap.dedent("\n".join(lines[k:])).strip("\n"), []
    if lang == "java":
        keep, pos = [], 0
        for m in lambda_matches(rest):
            if m.start() < pos:
                continue
            c = close_brace(rest, m.end() - 1)
            if c < 0:
                break
            tail = re.match(r"\s*\)\s*;", rest[c + 1:])
            if tail is None:
                continue
            end = c + 1 + tail.end()
            out.append({"kind": "lambda", "name": None, "arity": len([p for p in m.group(1).strip("()").split(",") if p.strip()]),
                        "text": textwrap.dedent(rest[m.start():end])})
            keep.append(rest[pos:m.start()])
            pos = end
        rest = textwrap.dedent("".join(keep) + rest[pos:]).strip("\n")
    if not rest.strip():
        return out
    src = ("<?php\nclass __X {\n" if lang == "php" else "class __X {\n") + rest + "\n}\n"
    tree, b = ts_parse(lang, src), src.encode()
    wrapper = next(nodes(tree.root_node, TYPES), None)
    body = next((x for x in (wrapper.children if wrapper is not None else []) if x.type in BODIES), None)
    if tree.root_node.has_error or body is None:
        return out + [{"kind": "raw", "name": None, "arity": None, "text": rest}]
    for x in body.named_children:
        if x.type in ("comment", "block_comment"):
            continue
        text = textwrap.dedent(b[b.rfind(b"\n", 0, span_of(b, x)) + 1:x.end_byte].decode("utf-8", "replace"))
        if x.type in ("method_declaration", "constructor_declaration"):
            nm = x.child_by_field_name("name")
            out.append({"kind": "method", "name": b[nm.start_byte:nm.end_byte].decode(), "arity": n_params(x), "text": text})
        else:
            out.append({"kind": "raw" if raw_member(lang, x) else "member", "name": None, "arity": None, "text": text})
    return out


class Method(NamedTuple):
    name: str
    arity: int
    text: str  # with its /** docblock, attributes and annotations
    ctor: bool  # a constructor (Java/C# constructor_declaration, PHP __construct)
    private: bool  # an explicit `private` modifier (a C# member without one is not private here)
    access: bool  # an explicit access modifier (public, protected, internal or private)
    body_line: int | None  # 1-based line of the body's opening brace (the line of a catalog fn_id)
    annotated: bool  # a step annotation, attribute or docblock tag before the body


def class_methods(lang: str, text: str | None) -> list[Method]:
    """Every method and constructor of the file's first class, in order."""
    if not text or lang == "ruby":
        return []
    b = text.encode()
    T = lambda n: b[n.start_byte:n.end_byte].decode("utf-8", "replace")
    cls = next(nodes(ts_parse(lang, text).root_node, TYPES), None)
    body = next((x for x in (cls.children if cls is not None else []) if x.type in BODIES), None)
    out = []
    for x in body.named_children if body is not None else []:
        if x.type in ("method_declaration", "constructor_declaration") and x.child_by_field_name("name") is not None:
            name, block = T(x.child_by_field_name("name")), x.child_by_field_name("body")
            words = {w for c in x.children if c.type in ("visibility_modifier", "modifier", "modifiers") for w in T(c).split()}
            head = b[span_of(b, x):block.start_byte if block is not None else x.end_byte].decode("utf-8", "replace")
            out.append(Method(name, n_params(x), b[span_of(b, x):x.end_byte].decode("utf-8", "replace"),
                              x.type == "constructor_declaration" or name.lower() == "__construct", "private" in words,
                              bool(words & {"public", "protected", "internal", "private"}),
                              block.start_point[0] + 1 if block is not None else None,
                              any(ANN_LINE[lang].match(l) for l in head.split("\n"))))
    return out


def body_start(lang: str, kind: str, text: str) -> int | None:
    """Index in `text` of what opens the body of a method unit (the first method or constructor it declares: its body
    `{`, or the `=>` of a C# expression body) or of a java8 lambda unit (the `{` of its own registration); None when
    there is none (a method without a body)."""
    if kind == "lambda":
        m = LAMBDA_START.search(text)
        return m.end() - 1 if m is not None else None
    pre = "<?php\nclass __X {\n" if lang == "php" else "class __X {\n"
    src = (pre + text + "\n}\n").encode()
    cls = next(nodes(ts_parse(lang, src.decode("utf-8", "replace")).root_node, TYPES), None)
    body = next((x for x in (cls.children if cls is not None else []) if x.type in BODIES), None)
    m = next((x for x in (body.named_children if body is not None else []) if x.type in ("method_declaration", "constructor_declaration")), None)
    open_ = m.child_by_field_name("body") if m is not None else None
    return None if open_ is None else len(src[:open_.start_byte].decode("utf-8", "replace")) - len(pre)


def unit_defs(lang: str, kind: str, path: str, text: str, styles: list[str], imports=None) -> list:
    """The step definitions a method, java8 lambda or Ruby block unit registers: those whose annotation
    starts before the `{` that opens the unit's body (`body_start`); for a Ruby block, the step call on its first line
    only. An annotation of a method of an anonymous or local class, of a local function, or a registration nested in a
    lambda or block body never binds. A style whose extraction fails gives nothing. `imports` (the output block's
    and the ER-tree file's, own way; §2.2) resolves annotation names as the language does."""
    if kind not in ("method", "lambda", "block"):
        return []
    opener = body_start(lang, kind, text) if kind != "block" else None
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    out = []
    for st in styles:
        try:
            found = defs_of(path, text, st, imports)
        except ValueError:
            continue
        out += [d for d in found if d.line == 1] if kind == "block" else \
            [d for d in found if opener is None or text.find(d.code, starts[d.line - 1]) < opener]
    return out


def csharp_private(lang: str, text: str | None, m: Method) -> bool:
    """A method that only its own class and the traits it uses can reach: an explicit `private`; in C# also no access
    modifier at all, in a class, struct or record, never in an interface, whose members are public. The type is the
    file's first class-like declaration, the one class_methods reads (insert.first_class: a struct is none, so an
    interface after a struct keeps its members public)."""
    if m.private or lang != "csharp" or m.access:
        return m.private
    cls = next(nodes(ts_parse(lang, text or "").root_node, TYPES), None)
    return cls is None or cls.type != "interface_declaration"


PHP_ATTR_STR = r"""(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")"""
PHP_ATTR_PART = re.compile(r"\s*(\\?[A-Za-z_]\w*(?:\\[A-Za-z_]\w*)*)\s*\(\s*(?:pattern\s*:\s*)?"
                           rf"({PHP_ATTR_STR}(?:\s*\.\s*{PHP_ATTR_STR})*)\s*,?\s*\)\s*", re.S)


def attr_groups(text: str):
    """Own way (§2.2): (start, end, [(start, end) of each attribute]) of every `#[ ... ]` of
    `text` that closes: a character scan that steps over quoted strings and counts ( ) [ ] nesting; the attributes are
    the pieces between commas at depth 0, each span without its surrounding blanks."""
    i = text.find("#[")
    while i >= 0:
        depth, j, piece, pieces = 0, i + 2, i + 2, []
        closed = False
        while j < len(text):
            c = text[j]
            if c in "'\"":
                q, j = c, j + 1
                while j < len(text) and text[j] != q:
                    j += 2 if text[j] == "\\" else 1
            elif c in "([":
                depth += 1
            elif c in ")]":
                if depth == 0:
                    closed = c == "]"
                    break
                depth -= 1
            elif c == "," and depth == 0:
                pieces.append((piece, j))
                piece = j + 1
            j += 1
        if closed:
            pieces.append((piece, j))
            trim = lambda a, b: (a + len(text[a:b]) - len(text[a:b].lstrip()), b - (len(text[a:b]) - len(text[a:b].rstrip())))
            yield i, j + 1, [trim(a, b) for a, b in pieces]
            i = text.find("#[", j + 1)
        else:
            i = text.find("#[", i + 2)


def php_step_attr(piece: str) -> str | None:
    """The pattern of one attribute text when it is a Behat step attribute (a short name Given/When/Then in any case,
    or \\Behat\\Step\\Given/When/Then as written; one line, no imports), each string literal with backslash + its
    quote and backslash + backslash undone and the literals joined; else None."""
    m = PHP_ATTR_PART.fullmatch(piece)
    if m is None:
        return None
    name = m.group(1)
    if not (name.lower() in ("given", "when", "then") or name in ("\\Behat\\Step\\Given", "\\Behat\\Step\\When",
                                                              "\\Behat\\Step\\Then")):
        return None
    return "".join(re.sub(r"\\([\\" + s.group(0)[0] + "])", r"\1", s.group(0)[1:-1])
                   for s in re.finditer(PHP_ATTR_STR, m.group(2)))


class PhpAnnLine:
    """ANN_LINE["php"], own way: a docblock step line, or one `#[...]` alone on the line holding a step attribute."""
    DOC = re.compile(r"^\s*\*\s*@(?i:given|when|then)\s+\S.*$")  # Behat 3's tags

    def match(self, line: str):
        if self.DOC.match(line):
            return True
        s = line.strip()
        g = next(attr_groups(s), None)
        return True if g and g[0] == 0 and g[1] == len(s) and any(php_step_attr(s[a:b]) is not None
                                                                    for a, b in g[2]) else None


ANN_LINE = {"php": PhpAnnLine(),  # the reader's forms (§2.2), own way
            "java": re.compile(r"^\s*@\s*(?:io\s*\.\s*cucumber\s*\.\s*java\s*\.\s*en\s*\.\s*)?(?:Given|When|Then|And|But)"
                               r"\s*\(.*\)\s*$"),
            "csharp": re.compile(r"^\s*\[\s*(?:TechTalk\s*\.\s*SpecFlow\s*\.\s*)?(?:Given|When|Then|StepDefinition)"
                                 r"(?:Attribute)?\s*\(.*\)\s*\]\s*$")}
DECOR_NODES = {"attribute_list", "annotation", "marker_annotation", "comment", "line_comment", "block_comment"}


def head_line(lang: str, text: str) -> int | None:
    """Own way: the 0-based line of `text` (one method) before which a kept HEAD
    annotation line goes: the line of the method's first token after its attributes/annotations (tree-sitter;
    attribute_list, annotation/marker_annotation, comments skipped) when only whitespace precedes that token on its
    line, else the method's own first line (before its whole attribute list); None when `text` holds no method."""
    pre = "<?php\nclass __X {\n" if lang == "php" else "class __X {\n"
    src = (pre + text + "\n}\n").encode()
    tree = ts_parse(lang, pre + text + "\n}\n")
    wrapper = next(nodes(tree.root_node, TYPES), None)
    body = next((x for x in (wrapper.children if wrapper is not None else []) if x.type in BODIES), None)
    m = next((x for x in (body.named_children if body is not None else [])
              if x.type in ("method_declaration", "constructor_declaration")), None)
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
    at = lambda i: len(src[:i].decode("utf-8", "replace")) - len(pre)
    i = at(tok.start_byte) if tok is not None else at(m.start_byte)
    if tok is None or text[text.rfind("\n", 0, i) + 1:i].strip():
        i = at(m.start_byte)
    return text.count("\n", 0, i)


PHP_STEP_DOC = re.compile(r"(?:^[ \t]*\*|/\*\*)[ \t]*@((?i:given|when|then))[ \t]+(.+?)[ \t]*(?=\*/|$)", re.M)  # Behat 3


def php_pattern_spans(text: str) -> list[tuple[int, int, str, str]]:
    """(start, end, source, pattern) of every PHP step annotation of `text` (one method) before its body: an attribute
    #[Kw('p')] / #[Kw("p")] anywhere in the head (the literal unescaped: backslash + quote and backslash + backslash),
    a docblock line `* @Kw p` (the whole line) or the `@Kw p` of a one-line `/** @Kw p */`. The pattern is what Behat
    registers: form and keyword do not matter (fixed 26/09/2026, own way)."""
    end = body_start("php", "method", text)
    head = text if end is None else text[:end]
    out = []
    for gs, ge, pieces in attr_groups(head):  # several attributes may share one `#[...]`
        steps = [(a, b, php_step_attr(head[a:b])) for a, b in pieces]
        steps = [(a, b, pat) for a, b, pat in steps if pat is not None]
        if len(pieces) == 1 and steps:  # a group of one: the whole `#[...]`
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


def _drop_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """`text` without the spans (and the blanks right after one), and without a line that one of them leaves blank."""
    for a, b in sorted(set(spans), reverse=True):
        inline = text[b - 1:b] != "\n"  # an inline attribute: the blanks after it go too; a whole line keeps the next
        text = text[:a] + (text[b:].lstrip(" \t") if inline else text[b:])
        ls, le = text.rfind("\n", 0, a) + 1, text.find("\n", a)
        le = len(text) if le < 0 else le
        if not text[ls:le].strip():
            text = text[:ls] + text[le + 1:]
    return text


def php_pattern_kept(new: str, old: str) -> tuple[str, list[str]]:
    """Own way (rule fixed 26/09/2026), since Behat 4.0.0-alpha1 registers attributes only and refuses a pattern
    registered twice: (new method text, HEAD annotation lines to keep) when the PHP method `new` replaces `old`. Each
    OLD step annotation line (the ER tree has dropped the CREATE ones, so these are other scenarios' steps) is matched
    with the output's annotations by PATTERN (php_pattern_spans): (a) the output does not write it: the HEAD line is
    kept, as before; the output writes it with the same line: nothing changes; else the HEAD line is kept as HEAD wrote
    it and the output's annotations of that pattern are dropped. (b) An output annotation whose pattern is no OLD one (a
    CREATE step) is never changed. So no pattern is registered twice on the method."""
    norm = lambda l: " ".join(l.split())
    heads, keep, drop = php_pattern_spans(new), [], []
    for line in old.split("\n"):
        if not ANN_LINE["php"].match(line):
            continue
        mine = php_pattern_spans(line.strip() + "\npublic function __x() {}")
        if not mine:  # not a pattern Behat reads from one line: compare the line itself, as before
            if all(norm(line) != norm(l) for l in new.split("\n")):
                keep.append(line.strip())
            continue
        same = [h for h in heads if h[3] in {p for _, _, _, p in mine}]  # every pattern of a group
        if not same or not any(norm(h[2]) == norm(line) for h in same):
            keep.append(line.strip())
        drop += [(a, b) for a, b, src, _ in same if norm(src) != norm(line)]
    return edited(new, drop), keep


def edited(text: str, drop: list[tuple[int, int]]) -> str:
    """`text` without the spans of `drop` (own way): a span that is one attribute of a `#[...]` of
    several rewrites that group with the others, `#[a, b]`, or removes it when none stays; any other span goes as
    _drop_spans drops it. The edits run from the end of the text back, so none moves another."""
    edits = []
    rest = set(drop)
    for gs, ge, pieces in attr_groups(text):
        gone = [s for s in pieces if s in rest]
        if len(pieces) < 2 or not gone:
            continue
        rest -= set(gone)
        stay = [text[a:b] for a, b in pieces if (a, b) not in gone]
        edits.append((gs, ge, "#[" + ", ".join(stay) + "]" if stay else None))
    edits += [(a, b, None) for a, b in rest]
    for a, b, new in sorted(set(edits), reverse=True):
        text = _drop_spans(text, [(a, b)]) if new is None else text[:a] + new + text[b:]
    return text



def with_old(lang: str, text: str, old: str) -> str:
    """A replacing method with the step annotation lines of the old one (`old`, the shell in the ER tree) that it lacks:
    attribute lines on their own lines before the head of the method (`head_line`: never inside a multi-line attribute,
    never between the signature and the body), docblock lines inside its leading docblock (a new one when it has none).
    PHP compares by the pattern Behat registers, dropping the output's own duplicate of a kept HEAD pattern
    (php_pattern_kept); Java/C# keep the normalised-line rule."""
    if lang == "php":
        text, lack = php_pattern_kept(text, old)
    else:
        norm = lambda l: " ".join(l.split())
        have = {norm(l) for l in text.split("\n") if ANN_LINE[lang].match(l)}
        lack = [l.strip() for l in old.split("\n") if ANN_LINE[lang].match(l) and norm(l) not in have]
    doc, other, ls = [" " + l for l in lack if l.startswith("*")], [l for l in lack if not l.startswith("*")], text.split("\n")
    if other:
        at = head_line(lang, text)
        if at is not None:
            ls[at:at] = other
        else:
            at, inside = len(ls) - 1, False
            for i, l in enumerate(ls):
                if inside or l.lstrip().startswith("/**"):
                    inside = "*/" not in l
                elif l.strip() and not re.match(r"\s*(?:#\[|@|\[|\*|/\*\*|//)", l):
                    at = i
                    break
            ls[at:at] = other
    if doc and ls[0].lstrip().startswith("/**"):
        end = next(i for i, l in enumerate(ls) if "*/" in l)
        if end == 0:
            inner = ls[0].strip()[3:].rsplit("*/", 1)[0].strip()
            ls[0:1] = ["/**"] + ([f" * {inner}"] if inner else []) + doc + [" */"]
        else:
            ls[end:end] = doc
    elif doc:
        ls[0:0] = ["/**", *doc, " */"]
    return "\n".join(ls)


# ----- descriptive measures ------------------------------------------------------------------------------------------

TOKEN = re.compile(r"\w+|[^\w\s]")


def lcs(a: list[str], b: list[str]) -> int:
    """Length of a longest common subsequence, bit-parallel over the positions of `a` (Hyyrö 2004)."""
    pos, full = {}, (1 << len(a)) - 1
    for i, x in enumerate(a):
        pos[x] = pos.get(x, 0) | 1 << i
    v = full
    for y in b:
        u = v & pos.get(y, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def naming(name: str) -> str:
    return "snake" if "_" in name.strip("_") else "camel" if name[:1].islower() else "pascal"


def majority(c: Counter):
    """The most frequent value when strictly more frequent than any other, else None."""
    top = c.most_common(2)
    return top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None


def method_names(lang: str, text: str) -> list[str]:
    """Method names (constructors aside) of the file's class, looked for at top level, in a namespace and in a C#
    namespace body only; none when that class has no name or body."""
    if lang == "ruby":
        return []
    tree, b = ts_parse(lang, text), text.encode()

    def find(n):
        for c in n.children:
            if c.type in TYPES:
                return c
            if c.type in ("namespace_definition", "package_declaration", "namespace_declaration", "file_scoped_namespace_declaration",
                          "declaration_list"):
                x = find(c)
                if x is not None:
                    return x
        return None
    cls = find(tree.root_node)
    name = None if cls is None else cls.child_by_field_name("name") or next((x for x in cls.children if x.type in ("name", "identifier")), None)
    body = None if cls is None else next((x for x in cls.children if x.type in BODIES), None)
    if name is None or body is None:
        return []
    out = []
    for c in body.children:
        nm = c.child_by_field_name("name") or next((x for x in c.children if x.type in ("name", "identifier")), None) \
            if c.type == "method_declaration" else None
        if nm is not None and b[nm.start_byte:nm.end_byte].decode("utf-8", "replace").lower() != "__construct":
            out.append(b[nm.start_byte:nm.end_byte].decode("utf-8", "replace"))
    return out


def defs_or_none(path: str, text: str, style: str, imports=None) -> list:
    try:
        return defs_of(path, text, style, imports)
    except ValueError:  # a step definition whose function or block never closes (a truncated output)
        return []


def members_src(lang: str, code: str) -> str:
    """One block as the body of a wrapper class, its import lines and Java lambda steps (constructor statements) out.
    PHP: read without a leading `<?php` or trailing `?>`."""
    if lang == "php":
        code = php_block_tags(code)
    lines = code.split("\n")
    k = 0
    while k < len(lines) and (not lines[k].strip() or re.match(IMPORTS[lang], lines[k])):
        k += 1
    rest = "\n".join(lines[k:])
    if lang == "java":  # lambda steps are constructor statements, not members
        for m in reversed(list(lambda_matches(rest))):
            c = close_brace(rest, m.end() - 1)
            tail = re.match(r"\s*\)\s*;", rest[c + 1:]) if c >= 0 else None
            if tail is not None:
                rest = rest[:m.start()] + rest[c + 1 + tail.end():]
    return ("<?php\nclass __X {\n" if lang == "php" else "class __X {\n") + rest + "\n}\n"


def declared(lang: str, code: str) -> set[str]:
    """Every member (method, field, property, constant) that one block of the output declares: condition 6 resolves it,
    condition 5 does not count it."""
    c = read_class(lang, members_src(lang, code))
    return set(c["members"]) if c else set()


LITERALS = ("comment", "line_comment", "block_comment", "string", "encapsed_string", "heredoc", "nowdoc", "string_literal",
            "shell_command_expression", "text_block", "character_literal", "verbatim_string_literal", "raw_string_literal",
            "interpolated_string_expression")
DECLS = ("field_declaration", "variable_declaration", "event_field_declaration", "property_declaration", "const_declaration",
         "constant_declaration")
DECLARATORS = ("variable_declarator", "property_element", "const_element", "property_declaration")


def init_texts(lang: str, code: str) -> list[str]:
    """Per field, property or constant of one block that has an initializer (what
    follows `=` in its declarators, literals aside; not an arrow body), a method that assigns those expressions to a
    local, so that member_uses reads them as it reads a body. Ruby: none."""
    if lang == "ruby":
        return []
    src = members_src(lang, code)
    tree, b = ts_parse(lang, src), src.encode()
    wrapper = next(nodes(tree.root_node, TYPES), None)
    body = next((x for x in (wrapper.children if wrapper is not None else []) if x.type in BODIES), None)

    def found(n) -> list[str]:
        eq = next((k for k, c in enumerate(n.children) if c.type == "="), None) if n.type in DECLARATORS else None
        if eq is not None:
            return [b[c.start_byte:c.end_byte].decode("utf-8", "replace") for c in n.children[eq + 1:] if c.is_named and c.type not in LITERALS]
        return [e for c in n.named_children for e in found(c)] if n.type in DECLS else []
    out = []
    for x in body.named_children if body is not None else []:
        exprs = found(x)
        if exprs and lang == "php":
            out.append("public function __init()\n{\n" + "".join(f"$__v = {e};\n" for e in exprs) + "}")
        elif exprs:
            out.append("void __init() {\n" + "".join(f"{'Object' if lang == 'java' else 'object'} __v = {e};\n" for e in exprs) + "}")
    return out


# ----- classes -------------------------------------------------------------------------------------------------------

def read_class(lang: str, text: str) -> dict | None:
    """Namespace, imports, the first class: kind, name, parents, members (name -> visibility, method arity)."""
    tree = ts_parse(lang, text)
    b = text.encode()
    T = lambda n: b[n.start_byte:n.end_byte].decode("utf-8", "replace")
    cls = next(nodes(tree.root_node, ("class_declaration", "trait_declaration", "interface_declaration", "enum_declaration", "record_declaration")), None)
    if cls is None:
        return None
    spaces = [n for n in nodes(tree.root_node, ("namespace_definition", "package_declaration", "namespace_declaration",
                                                 "file_scoped_namespace_declaration")) if n.start_byte < cls.start_byte]
    ns = T(spaces[-1].child_by_field_name("name") or spaces[-1].named_children[0]) if spaces else ""  # the last one before the class
    ns = re.sub(r"^package\s+|;$", "", ns).strip()
    imports, statics, static_all = {}, {}, []
    for n in nodes(tree.root_node, ("namespace_use_clause", "import_declaration", "using_directive")):
        t = T(n)
        if n.type == "namespace_use_clause" and n.parent is not None and n.parent.type == "namespace_use_group":
            continue  # `use A\{B};`: outline.parse reads no clause of a group, so nothing is imported
        if n.type == "namespace_use_clause":
            m = re.match(r"\\?([\w\\]+)(?:\s+as\s+(\w+))?", t.strip())
            if m:
                imports[m.group(2) or m.group(1).rsplit("\\", 1)[-1]] = m.group(1)
        else:
            m = re.match(r"(?:import|using)\s+(static\s+)?([\w.]+?)(\.\*)?\s*;", t.strip())
            if m and m.group(1):
                if m.group(3):
                    static_all.append(m.group(2))
                else:
                    statics[m.group(2).rsplit(".", 1)[-1]] = m.group(2).rsplit(".", 1)[0]
            elif m and not m.group(3):
                imports[m.group(2).rsplit(".", 1)[-1]] = m.group(2)
    parents = []
    for c in cls.children:
        if c.type in ("base_clause", "superclass", "base_list"):
            parents += [("extends", T(x)) for x in c.named_children if x.type not in ("comment",)]
        elif c.type in ("class_interface_clause", "super_interfaces"):
            for x in nodes(c, ("name", "qualified_name", "type_identifier", "scoped_type_identifier", "generic_type")):
                if x.parent is c or (x.parent is not None and x.parent.type == "type_list"):
                    parents.append(("implements", T(x)))
    members, fields, methods = {}, {}, {}  # fields: the non-method members; methods: methods and constructors
    body = next((c for c in cls.children if c.type in ("declaration_list", "class_body", "enum_body", "interface_body")), None)
    for c in (body.named_children if body is not None else []):
        vis = "private" if re.search(r"\bprivate\b", T(c).split("(")[0].split("{")[0]) else "open"
        if c.type in ("method_declaration", "constructor_declaration"):
            nm = c.child_by_field_name("name")
            members[T(nm)] = methods[T(nm)] = vis
            if T(nm).lower() == "__construct":
                for p in nodes(c, ("property_promotion_parameter",)):
                    v = next((x for x in p.children if x.type == "variable_name"), None)
                    if v is not None:
                        members[T(v).lstrip("$")] = fields[T(v).lstrip("$")] = "private" if "private" in T(p) else "open"
        elif c.type in ("property_declaration", "field_declaration", "const_declaration", "constant_declaration", "event_field_declaration"):
            for x in nodes(c, ("property_element", "const_element", "variable_declarator")):
                nm = x.child_by_field_name("name") or next((y for y in x.children if y.type in ("variable_name", "name", "identifier")), None)
                if nm is not None:
                    members[T(nm).lstrip("$")] = fields[T(nm).lstrip("$")] = vis
            if c.type == "property_declaration" and lang == "csharp" and c.child_by_field_name("name") is not None:
                members[T(c.child_by_field_name("name"))] = fields[T(c.child_by_field_name("name"))] = vis
        elif c.type == "use_declaration":
            parents += [("trait", T(x)) for x in c.named_children if x.type in ("name", "qualified_name")]
    lombok = lang == "java" and "log" not in members and \
        re.search(r"^\s*@(?:lombok\.extern\.\w+\.)?(?:Slf4j|XSlf4j|Log4j2?|Log|CommonsLog|JBossLog|Flogger|CustomLog)\b", text, re.M) is not None
    if lombok:  # the logger field Lombok generates, private to this class
        members["log"] = "open"
    return {"ns": ns, "imports": imports, "statics": statics, "static_all": static_all, "kind": cls.type.split("_")[0],
            "parents": parents, "members": members, "fields": fields, "methods": methods, "loggers": loggers_of(lang, text),
            "lombok": lombok}


class Parents:
    """In-repo resolution of parents and class references: PSR-4 (autoload + autoload-dev) and global PHP classes with a
    <Name>.php file outside vendor/; Java by package path; C# by unique file name."""

    def __init__(self, lang: str, git: Git):
        self.lang, self.git, self.tree = lang, git, set(git.tree)
        self.prefixes, self.globals = [], set()
        if lang == "php":
            data = json.loads(git.text("composer.json") or "{}")
            for sec in ("autoload", "autoload-dev"):
                for pre, dirs in ((data.get(sec) or {}).get("psr-4") or {}).items():
                    for d in [dirs] if isinstance(dirs, str) else dirs:
                        self.prefixes.append((pre, d.rstrip("/") + "/" if d else ""))
            self.prefixes.sort(key=lambda x: -len(x[0]))
            self.globals = {p.rsplit("/", 1)[-1][:-4] for p in self.tree if p.endswith(".php") and not p.startswith("vendor/") and "/vendor/" not in p}
        self.java = {}
        for p in self.tree:
            if p.endswith(".java"):
                parts = p[:-5].split("/")
                for i in range(len(parts)):
                    self.java.setdefault("/".join(parts[i:]), p)
        self.cs = defaultdict(list)
        for p in self.tree:
            if p.endswith(".cs"):
                self.cs[p.rsplit("/", 1)[-1][:-3]].append(p)

    def find(self, c: dict, written: str) -> tuple[str, str | None]:
        """("in" | "global" | "out", path)."""
        name = written.split("<", 1)[0].strip()
        if self.lang == "php":
            head, _, rest = name.lstrip("\\").partition("\\")
            if name.startswith("\\"):
                fqn = name[1:]
            elif head in c["imports"]:
                fqn = c["imports"][head] + ("\\" + rest if rest else "")
            else:
                fqn = f"{c['ns']}\\{name}" if c["ns"] else name
            for pre, d in self.prefixes:
                if fqn.startswith(pre):
                    path = d + fqn[len(pre):].replace("\\", "/") + ".php"
                    return ("in", path) if path in self.tree else ("out", None)
            return ("global", None) if "\\" not in fqn and fqn in self.globals else ("out", None)
        if self.lang == "java":
            fqn = name if "." in name else c["imports"].get(name, f"{c['ns']}.{name}" if c["ns"] else name)
            p = self.java.get(fqn.replace(".", "/"))
            return ("in", p) if p else ("out", None)
        ps = self.cs.get(name.rsplit(".", 1)[-1], [])
        return ("in", ps[0]) if len(ps) == 1 else ("out", None)


# ----- symbols -------------------------------------------------------------------------------------------------------

NAME_ONLY = {("parameter", "name"), ("formal_parameter", "name"), ("variable_declarator", "name"), ("catch_formal_parameter", "name"),
             ("enhanced_for_statement", "name"), ("foreach_statement", "left"), ("resource", "name"), ("argument", "name"),
             ("element_value_pair", "key"), ("attribute", "name"), ("declaration_expression", "name"),
             ("catch_declaration", "name"), ("local_function_statement", "name")}


def pattern_locals(root, T) -> set[str]:
    """Java pattern variables: the identifier of a type pattern (`case String s ->`) or of a record pattern component
    (`Entry(var k, var v)`), as symbols.DECLARES has them. `instanceof String s` (an instanceof_expression name) is a
    local too, read with the other declarations in member_uses."""
    return {T(x) for n in nodes(root, ("type_pattern", "record_pattern_component")) for x in n.named_children if x.type == "identifier"}


PHP_RELATIVE = r"(?<![\w$\\])(?i:self|static|parent)"  # PHP `self::`, `static::`, `parent::` in any case; never
# `$self::` (a variable's class) nor `\self::`
# the start of a class name before `::`: not inside a name or a variable, not the second name of `A::B::`, not after
# `->` (`=>X::`, `?:X::`, `case 1:X::`, `lbl:X::` are class calls)
PHP_CLASS_AT = r"(?<![\w\\$])(?<!::)(?<!->)"


def member_uses(lang: str, text: str) -> tuple[set[str], set[str], set[str]]:
    """(member names, of which bare calls, class names) that a function uses, on the cleaned text."""
    text = clean(lang, text)
    if lang == "php":
        body = text[text.find("{") + 1:] if "{" in text else text
        # PHP keywords in any case (SELF::, NEW, ::CLASS); `$this?->` is a member access too
        mem = set(re.findall(r"\$this\s*\??->\s*(\w+)", body)) | {x for x in re.findall(PHP_RELATIVE + r"\s*::\s*\$?(\w+)", body) if x.lower() != "class"}
        # a scoped METHOD call (self::x(, static::x(, parent::x() is also a call: it resolves and grounds in any case
        # like $this->x(
        calls = set(re.findall(r"\$this\s*\??->\s*(\w+)\s*\(", body)) \
            | set(re.findall(PHP_RELATIVE + r"\s*::\s*(\w+)\s*\(", body))
        cls = {a or b for a, b in re.findall(r"(?:\b(?i:new)\s+(\\?[A-Za-z_][\w\\]*)|" + PHP_CLASS_AT + r"(\\?[A-Z][\w\\]*)\s*::(?!\s*(?i:class)\b))", body)
               if (a or b).lower() not in ("self", "static", "parent")}
        return mem, calls, cls
    m = LAMBDA_START.search(text) if lang == "java" else None
    if m:
        c = close_brace(text, m.end() - 1)
        src = "class __X { void __m() { __f(" + text[m.start(1):c + 1] + "); } }"
    else:
        src = "class __X {\n" + text + "\n}"
    tree, b = ts_parse(lang, src), src.encode()
    T = lambda n: b[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def in_anonymous(n) -> bool:  # own way: under `new T(…) { … }`'s body
        p = n.parent
        while p is not None and not (p.type == "class_body" and p.parent is not None and p.parent.type == "object_creation_expression"):
            p = p.parent
        return p is not None
    local = set()
    for n in nodes(tree.root_node, ("formal_parameter", "parameter", "catch_formal_parameter", "variable_declarator",
                                    "enhanced_for_statement", "foreach_statement", "inferred_parameters", "lambda_expression",
                                    "implicit_parameter", "resource", "declaration_expression", "catch_declaration",
                                    "instanceof_expression", "declaration_pattern", "from_clause")):
        if n.type == "implicit_parameter":
            local.add(T(n))
        elif n.type == "inferred_parameters":
            local |= {T(x) for x in n.named_children if x.type == "identifier"}
        elif n.type == "lambda_expression":
            p = n.child_by_field_name("parameters")
            if p is not None and p.type == "identifier":
                local.add(T(p))
        elif n.type == "foreach_statement":
            p = n.child_by_field_name("left")
            if p is not None:
                local.add(T(p))
        else:
            nm = n.child_by_field_name("name")
            if nm is not None:
                local.add(T(nm))
    local |= pattern_locals(tree.root_node, T)
    # own way: C# `var (a, b)` names, LINQ `let x`, `join x`, `into x`
    local |= {T(x) for n in nodes(tree.root_node, ("tuple_pattern", "query_expression")) for x in n.named_children if x.type == "identifier"}
    local |= {T(next(x for x in n.named_children if x.type == "identifier")) for n in nodes(tree.root_node, ("let_clause", "join_clause"))
              if any(x.type == "identifier" for x in n.named_children)}
    mem, calls, cls = set(), set(), set()
    decl_names = {T(n.child_by_field_name("name")) for n in nodes(tree.root_node, ("method_declaration", "constructor_declaration", "class_declaration")) if n.child_by_field_name("name") is not None}
    for n in nodes(tree.root_node, ("identifier",)):
        p = n.parent
        idx = [c.id for c in p.children].index(n.id)
        field = p.field_name_for_child(idx)
        name = T(n)
        if (p.type == "method_invocation" and field == "name") or (p.type == "invocation_expression" and field == "function"):
            obj = p.child_by_field_name("object") if p.type == "method_invocation" else None
            # `this.f(` and `super.f(` are the class's calls too; a bare or `this.` call inside an
            # anonymous class body is that class's own
            if not name.startswith("__") and (obj is None or obj.type in ("this", "super")) \
                    and (obj is not None and obj.type == "super" or not in_anonymous(p)):
                calls.add(name)  # a call names a method, even when a local has the same name
            continue
        # Added 11/10/2026: `this.x` names a member even when a parameter or local is also named x (`this.x = x;`), as
        # symbols.jvm_refs reads it; so the field access is read before the locals are skipped
        if p.type in ("field_access", "member_access_expression") and field in ("field", "name"):
            obj = p.child_by_field_name("object") or p.child_by_field_name("expression")
            if obj is not None and obj.type in ("this", "this_expression", "super", "base") and not (obj.type == "this" and in_anonymous(p)):
                mem.add(name)  # `super.x` and C# `base.X` too
                if p.parent is not None and p.parent.type == "invocation_expression" and p.parent.child_by_field_name("function") == p:
                    calls.add(name)  # C# `this.F()`, `base.F()`
            continue
        if name in local or name.startswith("__") or name in decl_names and (p.type, field) in (("method_declaration", "name"), ("constructor_declaration", "name"), ("class_declaration", "name")):
            continue
        if p.type == "object_creation_expression":
            cls.add(name)
            continue
        if p.type in ("method_reference", "labeled_statement", "break_statement", "continue_statement", "scoped_identifier",
                      "marker_annotation", "annotation", "switch_label", "name_colon", "qualified_name", "generic_name",
                      "goto_statement", "type_pattern", "inferred_parameters") or (p.type, field) in NAME_ONLY:
            continue
        if name in {"java", "javax", "jakarta", "org", "com", "io", "net", "lombok", "System", "Microsoft"}:
            continue
        receiver = field in ("object", "expression")
        if name[0].isupper() and (receiver or not (name.isupper() and len(name) > 1)):
            cls.add(name)
        else:
            mem.add(name)
    for n in nodes(tree.root_node, ("object_creation_expression",)):
        t = n.child_by_field_name("type")
        if t is not None:
            cls.add(T(t).split("<", 1)[0])
    return mem | calls, calls, cls


# ----- the scorer ----------------------------------------------------------------------------------------------------

class Scorer:
    def __init__(self, repo: str, lang: str, styles: list[str], fns: dict, git: Git, suite_steps: dict,
                 symtimes: dict | None = None, forbidden: dict | None = None, infra=()):
        self.repo, self.lang, self.styles, self.fns, self.git, self.suite_steps = repo, lang, styles, fns, git, suite_steps
        self.symtimes = dict(symtimes or {})  # path -> sealed symtimes row (versions, vnames, vprivate)
        self.forbidden = forbidden or {}  # the repo's forbidden.json entry: the answers each partition hides
        self.infra = {tuple(x) for x in infra}  # 3A's masked answer infrastructure: (declaring path, name)
        self._seen = {}  # (record id, path, "visible" | "future") -> visible(), future() of that record and file
        self.ghosts = phantom_ids(lang, fns, git)
        self.def_fn = {a["def_id"]: fid for fid, f in fns.items() for a in f["annotations"]}
        self.parents = Parents(lang, git) if lang != "ruby" else None
        self._ident, self._world, self._gold, self._gold_blocks = {}, None, {}, {}

    def ident(self, fid: str):
        if fid not in self._ident:
            f = self.fns[fid]
            us = units(self.lang, f["code"])[1] if self.lang != "ruby" else []
            m = next((u for u in us if u["kind"] == "method"), None)
            self._ident[fid] = (f["path"], self.key(m["name"], m["arity"])) if m else (f["path"], fid)
        return self._ident[fid]

    def key(self, name, arity):
        return name.casefold() if self.lang == "php" else (name, arity)

    def world(self):
        if self._world is None:
            meths, paths = set(), set()
            for p in self.git.tree:
                if p.endswith(".rb") and p.startswith(("features/support/", "features/step_definitions/")):
                    meths |= set(re.findall(r"^\s*def\s+(?:self\.)?([a-z_]\w*[?!=]?)", self.git.text(p) or "", re.M))
                m = re.match(r"^(?:app/[^/]+|lib)/(.+)\.rb$", p)
                if m:
                    parts = m.group(1).split("/")
                    paths |= {"/".join(parts[:i]) for i in range(1, len(parts) + 1)}
            self._world = (meths, paths, {p.split("/")[0] for p in paths})
        return self._world

    def chain(self, path: str, texts: dict[str, str]):
        """(class info of the target, inherited members, open_all, open_calls, logger fields of the class and its
        in-repo ancestors, inherited METHOD names: an ancestor's non-private ones, a trait's all)."""
        c = read_class(self.lang, texts.get(path) or self.git.text(path) or "")
        if c is None:
            return None
        inherited, open_all, open_calls, inh_methods = {}, c["kind"] == "trait", False, set()
        loggers = set(c["loggers"])
        seen, todo = {path}, [(path, c, 0)]
        while todo:
            p, ci, depth = todo.pop(0)
            if depth >= 5:
                continue
            for rel, written in ci["parents"]:
                where, ppath = self.parents.find(ci, written)
                if where != "in" or ppath is None:
                    if rel in ("extends", "trait") or (self.lang == "php" and rel != "implements"):
                        open_all = True
                    elif self.lang != "php":
                        open_calls = True
                    continue
                if ppath in seen:
                    continue
                seen.add(ppath)
                pc = read_class(self.lang, texts.get(ppath) or self.git.text(ppath) or "")
                if pc is None:
                    continue
                for n, vis in pc["members"].items():
                    if (vis != "private" or rel == "trait") and not (n == "log" and pc["lombok"]):
                        inherited[n] = vis
                inh_methods |= {n for n, vis in pc["methods"].items() if vis != "private" or rel == "trait"}
                loggers |= pc["loggers"]
                todo.append((ppath, pc, depth + 1))
        return c, inherited, open_all, open_calls, loggers, inh_methods

    # ----- future methods and the expansion of helper calls -----

    def times(self, path: str) -> dict:
        """The symtimes row of `path` (the sealed input); without one, the same fields re-derived from git: the file's
        versions (newest first) and, per version, the member names its first class declares (plus `$this->x =` /
        `this.x =` assignments) and its private ones."""
        if path not in self.symtimes:
            vs, vn, vp = self.git.history(path), [], []
            assign = re.compile(r"\$this->(\w+)\s*=(?![=>])" if self.lang == "php" else r"\bthis\.(\w+)\s*=(?!=)")
            for sha, _, p in vs:
                text = self.git.blob(sha, p)
                c = read_class(self.lang, text)
                members = c["members"] if c else {}
                vn.append(sorted(set(members) | set(assign.findall(text))))
                vp.append(sorted(n for n, v in members.items() if v == "private"))
            self.symtimes[path] = {"versions": [[sha, at, p] for sha, at, p in vs], "vnames": vn, "vprivate": vp}
        return self.symtimes[path]

    def vindex(self, path: str, t: int) -> int | None:
        """The version of `path` as it was before t: the one just older than the oldest version authored at or after t;
        None when the file had no version before t."""
        vs = self.times(path)["versions"]
        later = [i for i, v in enumerate(vs) if v[1] >= t]
        i = later[-1] + 1 if later else 0
        return i if i < len(vs) else None

    def files(self, path: str, texts: dict, rels: dict | None = None) -> tuple[list[str], set[str]]:
        """(the class chain of `path`: the file, then its in-repo ancestors breadth first, each once; the files whose
        private members the class can use: itself and the traits it uses, directly or through traits). `rels`, when
        given, receives each ancestor's relation ("extends", "implements", "trait") on the edge that reached it."""
        read = lambda q: read_class(self.lang, texts.get(q) or self.git.text(q) or "")
        out, access, todo = [path], {path}, [(path, read(path), 0)]
        while todo:
            q, ci, depth = todo.pop(0)
            if ci is None or depth >= 5:
                continue
            for rel, written in ci["parents"]:
                where, pq = self.parents.find(ci, written)
                if where != "in" or pq is None or pq in out:
                    continue
                out.append(pq)
                if rels is not None:
                    rels[pq] = rel
                if rel == "trait" and q in access:
                    access.add(pq)
                todo.append((pq, read(pq), depth + 1))
        return out, access

    def pre_text(self, q: str, t: int) -> str | None:
        """The file `q` as it was before t, from git; None when it did not exist then."""
        i = self.vindex(q, t)
        return None if i is None else self.git.blob(self.times(q)["versions"][i][0], self.times(q)["versions"][i][2])

    def chain_before(self, path: str, t: int, chain: list[str]) -> list[str]:
        """The files of `chain` the class reached before t: breadth first over the parents declared in the versions
        before t, kept inside `chain` (a parent added at or after t is not part of V(r))."""
        out, todo = [path], [path]
        while todo:
            c = read_class(self.lang, self.pre_text(todo.pop(0), t) or "")
            for _, written in c["parents"] if c else ():
                where, pq = self.parents.find(c, written)
                if where == "in" and pq in chain and pq not in out:
                    out.append(pq)
                    todo.append(pq)
        return out

    def gone(self, r: dict) -> set[str]:
        """Step functions V(r) never shows: the record's new answers and what its partition hides (train: test, dev and
        aux answers; dev: test and aux answers)."""
        hide = {"train": ("test", "dev", "aux"), "dev": ("test", "aux")}.get(r["partition"], ())
        return {c["fn_id"] for c in r["create"] if not c["preexisting"]} | {f for k in hide for f in self.forbidden.get(k, ())}

    def masked(self, r: dict) -> set[tuple[str, str]]:
        """What 3A masks in the record's prompt (infra.json, Composer.masked): the repo's answer
        infrastructure for a train or dev record, nothing for test and aux_test."""
        return self.infra if r["partition"] in ("train", "dev") else set()

    def visible(self, path: str, r: dict, texts: dict) -> set[str]:
        """The member names 3A shows for the class of `path` in ④(a) (`view`). For a file new at t_vis (future by time,
        fixed 25/09/2026), the names shown for each in-repo ancestor, less an ancestor's constructor and a member
        private at HEAD, or in the declaration taken from before t, outside the file and the traits it uses: the
        ancestors' old helpers stay shown, the new file's own methods are future."""
        if (r["id"], path, "visible") in self._seen:
            return self._seen[(r["id"], path, "visible")]
        if self.lang == "ruby":
            return set()
        if self.vindex(path, r["t_vis"]) is not None:
            shown = {e[0] for e in self.view(path, r, texts)}
        else:
            chain, access = self.files(path, texts)
            shown = {name for a in chain[1:] for name, ctor, private, origin, pre_private, pre_origin in self.view(a, r, texts)
                     if not ctor and not (private and origin not in access) and not (pre_private and pre_origin not in access)}
        self._seen[(r["id"], path, "visible")] = shown
        return shown

    def view(self, path: str, r: dict, texts: dict) -> list[tuple[str, bool, bool, str, bool, str]]:
        """④(a) of `path` re-derived from the sealed symtimes rows and the file versions they list, as (name, the HEAD
        member is a constructor, it is private, its file, the declaration taken from before t is private, its file). A
        member of the class chain at HEAD (a private one only of the class and its traits, an ancestor's constructor
        never) is shown when a chain file's version before t declares its name (vnames), 3A does not
        mask it for the record (`masked`), and it is not a step function of `gone` (the record's answers and what the
        partition hides) nor one with no annotation before t; the member taken is the one the chain as it was before t
        declares with the same name and group, a method without a step annotation unless it is a step itself: the
        member's own file first, then the other files of that chain. A shown field name shows a method of that name too.
        Nothing for a file without a version before t (a new file)."""
        t, gone = r["t_vis"], self.gone(r)
        if self.vindex(path, t) is None:
            return []
        chain, access = self.files(path, texts)
        declared = set()
        for q in chain:
            i = self.vindex(q, t)
            if i is not None:
                tm = self.times(q)
                declared |= set(tm["vnames"][i]) - (set() if q in access else set(tm["vprivate"][i]))
        declared -= {n for q, n in self.masked(r) if q in chain}
        before = self.chain_before(path, t, chain)
        pre_m, pre_o = {}, {}  # file -> [(name, constructor?, step-annotated, private)]; file -> {field name: private}
        for q in before:
            text = self.pre_text(q, t) or ""
            c = read_class(self.lang, text)
            pre_m[q] = [(m.name, m.ctor, m.annotated, m.private) for m in class_methods(self.lang, text) if not m.private or q in access]
            pre_o[q] = {n: vis == "private" for n, vis in (c["fields"].items() if c else ()) if vis != "private" or q in access}
        order = lambda q: ([q] if q in before else []) + [x for x in before if x != q]
        steps = {(f["path"], int(fid.rsplit(":", 1)[1])): fid for fid, f in self.fns.items()}
        out = []
        for q in chain:
            head = self.git.text(q)
            ms, c = class_methods(self.lang, head), read_class(self.lang, head or "")
            for m in ms:
                if (m.ctor and q != path) or (m.private and q not in access) or m.name not in declared:
                    continue
                fid = steps.get((q, m.body_line))
                if fid is not None and (fid in gone or not any(a["time"] is not None and a["time"] < t for a in self.fns[fid]["annotations"])):
                    continue
                step = fid is not None or m.annotated
                pick = next(((f, pv) for f in order(q) for n, ctor, ann, pv in pre_m[f] if n == m.name and ctor == m.ctor and (step or not ann)), None)
                if pick is not None:
                    out.append((m.name, m.ctor, m.private, q, pick[1], pick[0]))
            for n, vis in (c["fields"].items() if c else ()):
                if (vis == "private" and q not in access) or n not in declared:
                    continue
                pick = next(((f, pre_o[f][n]) for f in order(q) if n in pre_o[f]), None)
                if pick is not None:
                    out.append((n, False, vis == "private", q, pick[1], pick[0]))
        return out

    def future(self, path: str, r: dict, texts: dict) -> dict[object, list[tuple[str, str, str, bool, bool]]]:
        """Method key -> [(name, declaring file, text, is a constructor, has a body)] of the methods and constructors of
        the class chain of `path` in the ER tree (`texts`, else HEAD) whose name the view does not show; a private one
        (C#: also one without an access modifier) only of the class itself or of a trait it uses, directly or through
        traits (never of a trait a parent uses); a key declared along the chain comes from the most derived file
        declaring it, every overload of that file with that key kept (Java/C# overloads of the same arity do not
        collapse)."""
        if self.lang == "ruby":
            return {}
        if (r["id"], path, "future") in self._seen:
            return self._seen[(r["id"], path, "future")]
        shown, out = self.visible(path, r, texts), {}
        chain, access = self.files(path, texts)
        for q in chain:
            here = defaultdict(list)
            text_q = texts.get(q) or self.git.text(q)
            for m in class_methods(self.lang, text_q):
                private = csharp_private(self.lang, text_q, m)  # a C# member without a modifier, never in an interface
                if m.name not in shown and not (private and q not in access):  # private methods only of the
                    here[self.key(m.name, m.arity)].append((m.name, q, m.text, m.ctor, m.body_line is not None))  # class, own traits
            for k, found in here.items():
                out.setdefault(k, found)
        self._seen[(r["id"], path, "future")] = out
        return out

    def fold(self, name: str) -> str:
        return name.casefold() if self.lang == "php" else name  # PHP method names are case-insensitive

    def resolver(self, path: str, visible: set[str], own: dict[str, list[str]], ctors: set[str], keys: set, future: dict,
                 shells: set = frozenset()):
        """f -> the bodies (declaring file, text) a ("self", f) call opens into, or None. A name V(r) shows is kept (a
        project member, even when the output rewrites it); else the union of the output's methods named f (PHP names in
        any case) and the future methods of the ER tree named f that the file keeps after insertion: for a key the
        output declares, never an ancestor's declaration (the target now declares it) and, in the target file, not the
        FIRST declaration of that key (the one the insertion replaced; the file's other declarations of the key stay). A
        constructor, a method without a body (abstract, interface), or a preexisting CREATE shell of the target file
        (D11: its ER-tree body is the answer) never opens from the ER tree. The shell rule only matters for a shell V(r)
        does not show: a visible name is kept first, so a visible shell stays ("self", f) even when the output rewrites
        it; the output's redefinition of a not-visible shell replaces it and opens the output's version."""
        mine, shown = defaultdict(list), {self.fold(n) for n in visible}
        for n, bodies in own.items():
            if n not in ctors:
                mine[self.fold(n)] += [(q, t) for q, t in bodies if body_start(self.lang, "method", t) is not None]
        for k, found in future.items():
            for i, (n, q, text, ctor, body) in enumerate(found):
                if k in keys and (q != path or i == 0):  # replaced (own file: its first declaration) or overridden
                    continue
                if q == path and k in shells:  # D11: a preexisting shell never opens from the ER tree
                    continue
                if not ctor and body:
                    mine[self.fold(n)].append((q, text))
        return lambda f: None if self.fold(f) in shown else list(dict.fromkeys(mine.get(self.fold(f), []))) or None

    def scope_of(self, q: str, tree: dict, blocks: dict):
        """(class info, available members, logger fields, project members) of file `q`, the Scope a body it declares is
        read with (1b287d0): the target file with the output's members and import lines, an ancestor as the ER
        tree has it; project members: its own, inherited and statically imported (in-repo) ones. None without a class."""
        ch = self.chain(q, tree)
        if ch is None:
            return None
        c, inherited, _, _, loggers, inh_methods = ch
        if q in blocks:  # the target file after insertion: the output's import lines join its imports
            lines = (php_block_tags(blocks[q]) if self.lang == "php" else blocks[q]).split("\n")  # PHP tags stripped
            k = next((i for i, l in enumerate(lines) if l.strip() and not re.match(IMPORTS[self.lang], l)), len(lines))
            added = read_class(self.lang, ("<?php\n" if self.lang == "php" else "") + "\n".join(lines[:k]) + "\nclass __X {}\n")
            c = {**c, "imports": {**c["imports"], **(added["imports"] if added else {})}}
        statics = {n for n, cls_name in c["statics"].items() if self.lang == "java" and self.parents.java.get(cls_name.replace(".", "/"))}
        for pkg in c["static_all"]:
            pq = self.parents.java.get(pkg.replace(".", "/")) if self.lang == "java" else None
            pc = read_class(self.lang, self.git.text(pq) or "") if pq else None
            statics |= set(pc["members"]) - ({"log"} if pc["lombok"] else set()) if pc else set()
        base = set(c["members"]) | set(inherited)
        methods = {self.fold(n) for n in set(c["methods"]) | inh_methods} if self.lang == "php" else set()  # the ER chain's
        return c, base | (declared(self.lang, blocks[q]) if q in blocks else set()), loggers, base | statics, methods

    def expand(self, text: str, origin: str, root: bool, resolve, scope):
        """(project calls, the helper bodies read, cuts) of one function of file `origin` (a method: `root` True, its
        own body starts the expansion path; a lambda: False): each ("self", f) call to a helper (resolve(f), `resolver`)
        is replaced by the calls of the helper's bodies, each read like a step body (dead code, loggers) with the Scope
        of the file that declares it (scope(file), `scope_of`; 1b287d0), recursively to depth EXPAND_DEPTH. The path
        holds BODIES (a step calling another overload of its own name opens it). Each body of a call is judged on its
        own: already on the path, a cycle; else at depth EXPAND_DEPTH, a depth cut; else it opens. When a body of the
        call is cut, the call stays ("self", f), and each counter counts that call once. The bodies read, (file, text)
        in order, are the ones condition 5 reads too."""
        cuts, read = {"depth": 0, "cycle": 0}, {}

        def walk(body: str, q: str, path: tuple) -> set:
            sc = scope(q)
            if sc is None:
                return set()
            c, avail, loggers = sc[:3]
            calls = set()
            for r0, f in sorted(self._calls(body, c, avail)):
                if r0 != "self" and (LOGGER_NAME.match(r0) or r0 in loggers):
                    continue
                bodies = resolve(f) if r0 == "self" else None
                if not bodies:
                    calls.add((r0, f))
                    continue
                cycle = [b for b in bodies if b in path]
                deep = [b for b in bodies if b not in path] if len(path) - root >= EXPAND_DEPTH else []
                cuts["cycle"] += bool(cycle)
                cuts["depth"] += bool(deep)
                if cycle or deep:
                    calls.add((r0, f))
                for b in bodies:
                    if b not in cycle and b not in deep:
                        read.setdefault(b, None)
                        calls |= walk(b[1], b[0], path + (b,))
            return calls
        calls = walk(text, origin, ((origin, text),) if root else ())
        return calls, list(read), cuts

    def grounds(self, text: str, q: str, scope, mine: set[str]) -> bool:
        """Condition 5 for one body of file `q`, read with that file's Scope (1b287d0): it uses a project member
        (not a logger; not a name in `mine`, the output's own methods) or an in-repo class."""
        sc = scope(q)
        if sc is None:
            return False
        c, _, loggers, project, methods = sc
        mem, calls, cls = member_uses(self.lang, text)
        names = {x for x in {x for x in mem if self.fold(x) not in mine} | (cls if self.lang in ("csharp", "java") else set())
                 if not LOGGER_NAME.match(x) and x not in loggers}
        # a PHP method call counts in any case when the ER-tree chain has that method
        return bool(names & project) or any(self.fold(n) in methods for n in calls & names) or \
            any(self.parents.find(c, x)[0] in ("in", "global") for x in cls)

    def score(self, r: dict, labels_row: dict, text: str) -> dict:
        out = dict.fromkeys(CONDS, 0)
        targets = list(dict.fromkeys(c["path"] for c in r["create"]))
        status, blocks, truncated = blocks_of(text, targets)
        tree, items = {}, []

        def stop(how: str) -> dict:  # no usable block, a collision, or condition 1 fails (conditions 2-6 are then 0)
            return {"bv": 0, **{f"bv_{c}": 0 for c in CONDS}, "extract_status": status, "insert_status": how, "binder": {},
                    "calls": {}, "cuts": {"depth": 0, "cycle": 0}, "duplicate_reuse": False, "unresolved": 0, "open_resolution": False,
                    "fence_truncated": truncated, **self.similarity(r, targets, blocks), **self.conventions(tree, items, blocks)}
        if not blocks:
            return stop("empty")
        tree, gone = er_tree(r, self.fns, self.ghosts, self.git)
        shells = self.shell_keys(r)
        found = {path: items_of(self.lang, code) for path, code in blocks.items()}
        # fixed 26/09/2026: conventions() reads the output's OWN items (its blocks, before any HEAD-kept-line merge, and
        # before insertion decides collision/replaceable), so every early `stop()` below (a collision included) sees
        # them too, like describe.row's own_items does; `found` already comes from the tag-stripped blocks
        items = [(path, i["kind"], i["text"], i["name"]) for path, its in found.items() for i in its]
        for its in found.values():  # the same method key twice in one output
            ks = [self.key(i["name"], i["arity"]) for i in its if i["kind"] == "method"]
            if len(ks) != len(set(ks)):
                return stop("collision")
        old = {path: self._methods(tree.get(path, self.git.text(path))) for path in blocks}
        fut = {path: self.future(path, r, tree) for path in blocks}
        # a method of the target class that is future for r is replaced like a shell; an ancestor's future
        # method is overridden (it is not in `old`)
        replaceable = {path: shells[path] | {k for k, found in fut[path].items() if found[0][1] == path} for path in blocks}
        for path, its in found.items():
            for i in its:
                k = self.key(i["name"], i["arity"]) if i["kind"] == "method" else None
                if k is not None and k in old[path] and k not in replaceable[path]:
                    return stop("collision")
        fix = lambda path, k, t: with_old(self.lang, t, old[path][k]) if k is not None and k in replaceable[path] and k in old[path] else t
        gen, idents, replaced, syntax, dfn = [], {}, set(), True, {}
        own_defs = defaultdict(lambda: defaultdict(list))  # path -> method name -> texts the output defines
        own_ctors = defaultdict(set)  # path -> names of the constructors the output defines
        own_keys = defaultdict(set)  # path -> method keys the output defines
        own_future = defaultdict(set)  # path -> folded names of the output's methods with a future method's key
        for path, code in blocks.items():
            base = tree.get(path, self.git.text(path))
            _, us, ok = units(self.lang, code)
            syntax &= ok and not ts_parse(self.lang, base).root_node.has_error
            for i, u in enumerate(us):
                k = self.key(u["name"], u["arity"]) if u["kind"] == "method" else None
                ident = (path, k) if k is not None else (path, f"gen:{i}")
                idents[ident] = (path, u)
                if k is not None and k in replaceable[path]:
                    replaced.add(ident)
                dfn[ident] = fix(path, k, u["text"])
                if u["kind"] == "method":
                    own_defs[path][u["name"]].append((path, dfn[ident]))
                    own_keys[path].add(k)
                    if u.get("ctor"):
                        own_ctors[path].add(u["name"])
                    if k in fut[path]:
                        own_future[path].add(self.fold(u["name"]))
                gen += [(d, ident) for d in unit_defs(self.lang, u["kind"], path, dfn[ident], self.styles,
                                                      file_imports(self.lang, code + "\n" + base))]  # §2.2 imports
        if not syntax:
            return stop("ok")
        lib = Library([d for d, _ in gen])
        of = {id(d): i for d, i in gen}
        m = lambda s: {of[id(d)] for d in lib.matches(s)}
        own = r["suite"]  # also every suite that a function of a target file serves
        served = {t: {own} | {su for f in self.fns.values() if f["path"] == t for su in (f.get("suites") or [])} for t in targets}
        bound = defaultdict(set)  # (suite, step) -> identities bound in the ER tree
        for su in set().union(*served.values()):
            for s, ids in self.suite_steps.get(su, []):
                bound[(su, s)] |= {self.ident(self.def_fn[i]) for i in ids if i not in gone and i in self.def_fn}
        for s, ids in zip(r["steps"], labels_row["step_def_ids"]):
            bound[(own, s)] |= {self.ident(self.def_fn[i]) for i in ids if i not in gone and i in self.def_fn}
        bound = {k: v for k, v in bound.items() if v}
        creates = list(dict.fromkeys(s for s, k in zip(r["steps"], r["step_kinds"]) if k == "create"))
        binder = {}
        for s in creates:
            b1 = {x for x in bound.get((own, s), set()) if x not in replaced} | m(s)
            if len(b1) == 1 and b1 <= set(idents):
                binder[s] = next(iter(b1))
        out["syntax"] = syntax
        out["binding"] = len(binder) == len(creates)
        changed = [s for (su, s), b0 in bound.items()
                   if ({x for x in b0 if x not in replaced} | {g for g in m(s) if su == own or su in served.get(g[0], ())}) != b0]
        out["no_ambiguity"] = not changed
        reuse = {s for s, k in zip(r["steps"], r["step_kinds"]) if k != "create"}
        args = self._args(r)
        ok4 = out["binding"]
        for s in creates:
            f = binder.get(s)
            if f is None:
                continue
            u = idents[f][1]
            good = False
            for d in lib.matches(s):
                if of[id(d)] == f:
                    want = self._placeholders(d, s) + args.get(s, False)
                    good |= arity_ok(self.lang, u, want, bool(args.get(s, False))) and not self._hardcoded(d, s)  # per framework
            ok4 &= good
        out["arity"] = ok4
        ok5, ok6, missing, opened = bool(binder), True, set(), set()
        chains = {}
        seen = {path: self.visible(path, r, tree) for path in blocks}
        hs = {path: self.resolver(path, seen[path], own_defs[path], own_ctors[path], own_keys[path], fut[path], shells[path])
              for path in blocks}
        root = lambda ident: ident in idents and idents[ident][1]["kind"] == "method"  # a method's own body starts the path
        scopes = {}
        scope = lambda q: scopes[q] if q in scopes else scopes.setdefault(q, self.scope_of(q, tree, blocks))
        judged = [(ident, path, dfn[ident]) for ident, (path, _) in idents.items()]
        # an initializer of a field, property or constant is read like a body, for condition 6 only
        judged += [(None, path, t) for path, code in blocks.items() for t in init_texts(self.lang, code)]
        for ident, path, text_u in judged:
            if self.lang == "ruby":
                meths, paths, roots = self.world()
                mem, calls, cls = self._ruby(text_u)
                snake = lambda x: re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", x).lower()
                segs = lambda c: [snake(x) for x in c.lstrip(":").split("::")]
                bad = {c for c in cls if segs(c)[0] in roots and "/".join(segs(c)) not in paths}
                ok6 &= not bad
                missing |= bad
                opened |= calls - meths  # a receiverless call outside the World is never judged
                if ident in binder.values():
                    ok5 &= bool(calls & meths) or any("/".join(segs(c)) in paths for c in cls)
                continue
            if path not in chains:
                chains[path] = self.chain(path, tree)
            ch = chains[path]
            mem, calls, cls = member_uses(self.lang, text_u)
            if ch is None:
                ok6 = False
                continue
            c, inherited, open_all, open_calls, loggers, inh_methods = ch
            own_er = set(c["members"])
            own_out = declared(self.lang, blocks[path]) if path in blocks else set()
            known = own_er | own_out | set(inherited) | ROOT.get(self.lang, frozenset())  # ROOT: not in `project`
            open_all = open_all or bool(known & {"__call", "__get", "__callStatic"})
            # a static import of the ER-tree file counts when its class is in the repo; an external one,
            # or one the output adds, only keeps condition 6 open and never grounds
            statics, open_names = set(), set()
            for name, cls_name in c["statics"].items():
                in_repo = self.lang == "java" and self.parents.java.get(cls_name.replace(".", "/"))
                (statics if in_repo else open_names).add(name)
            for pkg in c["static_all"]:
                p = self.parents.java.get(pkg.replace(".", "/")) if self.lang == "java" else None
                pc = read_class(self.lang, self.git.text(p) or "") if p else None
                if pc is None:
                    open_calls = True
                else:
                    statics |= set(pc["members"]) - ({"log"} if pc["lombok"] else set())
            er_pairs = {(v, k) for k, v in c["statics"].items()} | {(x, "*") for x in c["static_all"]}
            for b in blocks.values():
                for cls_name, member in re.findall(r"^\s*(?:import|using)\s+static\s+([\w.]+?)(?:\.(\w+|\*))?\s*;", b, re.M):
                    if (cls_name, member) in er_pairs:
                        continue
                    if member and member != "*":
                        open_names.add(member)
                    else:
                        open_calls = True
            avail = known | statics | open_names
            project = own_er | set(inherited) | statics
            # a PHP method call resolves in any case on a method of the class after
            # insertion or an inherited one (property names stay case-sensitive); such a call is no open path
            meths = {self.fold(n) for n in set(c["methods"]) | set(own_defs[path]) | inh_methods} if self.lang == "php" else set()
            any_case = {n for n in calls if self.fold(n) in meths}
            if not open_all:
                bad = mem - avail - any_case - (calls if open_calls else set())
                ok6 &= not bad
                missing |= bad
            # names that pass condition 6 only through an open path
            strict = mem - known - statics - any_case
            opened |= strict & open_names | (strict if open_all else strict & calls if open_calls else set())
            if c["lombok"] and "log" not in own_out and "log" in mem:
                opened.add("log")
            if self.lang == "csharp":  # a PascalCase name that is neither a member nor an in-repo class
                opened |= {x for x in cls if x not in known and self.parents.find(c, x)[0] not in ("in", "global")}
            if ident in binder.values():  # a logger never grounds
                # the step's body or any helper body its calls open into grounds (1b287d0), each
                # read with the Scope of its own file; the name of a NEW method of the output never grounds on its own:
                # not a name V(r) shows, nor one with the key of a future method, which grounds like
                # the member it replaces or overrides
                _, bodies, _ = self.expand(text_u, path, root(ident), hs[path], scope)  # the bodies read
                mine = {self.fold(n) for n in own_defs[path] if n not in own_ctors[path]} - {self.fold(n) for n in seen[path]} - own_future[path]
                ok5 &= any(self.grounds(b, q, scope, mine) for q, b in [(path, text_u)] + bodies)
        out["grounded"], out["resolved"] = ok5, ok6
        call_sets, cuts = {}, Counter(depth=0, cycle=0)
        for ident, (path, u) in idents.items():
            if self.lang == "ruby":
                call_sets[ident] = {("self", c) for c in self._ruby(dfn[ident])[1] if c in self.world()[0]}
                continue
            ch = chains.get(path) or self.chain(path, tree)
            if ch is None:
                call_sets[ident] = set()
                continue
            call_sets[ident], _, n = self.expand(dfn[ident], path, root(ident), hs[path], scope)  # helper calls opened
            cuts.update(n if ident in binder.values() else {})  # the cuts of the step-binding functions
        return {"bv": int(all(out.values())), **{f"bv_{k}": int(v) for k, v in out.items()}, "extract_status": status,
                "insert_status": "ok", "binder": binder, "calls": call_sets, "cuts": dict(cuts),
                "duplicate_reuse": any(s in reuse for s in changed),
                "unresolved": min(len(missing), 10), "open_resolution": ok6 and bool(opened),
                "fence_truncated": truncated, **self.similarity(r, targets, blocks), **self.conventions(tree, items, blocks)}

    def _methods(self, base: str | None) -> dict:
        """Method key -> text (a /** docblock right before it included) of the methods of a file's first class."""
        if self.lang == "ruby" or base is None:
            return {}
        b = base.encode()
        cls = next(nodes(ts_parse(self.lang, base).root_node, TYPES), None)
        body = next((x for x in (cls.children if cls is not None else []) if x.type in BODIES), None)
        out = {}
        for x in (body.named_children if body is not None else []):
            nm = x.child_by_field_name("name")
            if x.type in ("method_declaration", "constructor_declaration") and nm is not None:
                out.setdefault(self.key(b[nm.start_byte:nm.end_byte].decode("utf-8", "replace"), n_params(x)),
                               b[span_of(b, x):x.end_byte].decode("utf-8", "replace"))
        return out

    def similarity(self, r: dict, targets: list[str], blocks: dict[str, str]) -> dict:
        """rouge_l (LCS F1 over word and punctuation tokens, 4 decimals) and exact (same tokens up to whitespace) of the
        output's code against the gold answer's, target files in order."""
        if r["id"] not in self._gold_blocks:
            self._gold_blocks[r["id"]] = blocks_of(self.gold_text(r), targets)[1]
        g, o = ("\n".join(x.get(t, "") for t in targets) for x in (self._gold_blocks[r["id"]], blocks))
        a, b = TOKEN.findall(g), TOKEN.findall(o)
        n = lcs(a, b)
        f1 = 0.0 if n == 0 else 2 * (n / len(a)) * (n / len(b)) / (n / len(a) + n / len(b))
        return {"rouge_l": round(f1, 4), "exact": g.split() == o.split()}

    def conventions(self, tree: dict, items: list, blocks: dict[str, str]) -> dict:
        """Shares of the inserted items' step definitions that follow the majority annotation style and pattern kind of
        their target file in the ER tree, and of their methods that follow its majority method naming; None without a
        majority or without anything to judge. An item's annotation names resolve through the imports of its output
        block and of the ER-tree file (§2.2: as insertion merges them), not of the item alone."""
        got, want, bases = {"conv_style": [], "conv_pattern": [], "conv_naming": []}, {}, {}
        kind_of = lambda d: "regex" if d.kind.startswith("regex") else "placeholder"
        for path, kind, text, name in items:
            if path not in want:
                base = bases[path] = tree.get(path, self.git.text(path)) or ""
                styles, kinds = Counter(), Counter()
                for st in self.styles:
                    for d in defs_or_none(path, base, st, file_imports(self.lang, base)):
                        styles[st] += 1
                        kinds[kind_of(d)] += 1
                want[path] = (majority(styles), majority(kinds),
                              majority(Counter(naming(n) for n in method_names(self.lang, base) if not n.startswith("__"))))
            ws, wk, wn = want[path]
            imports = file_imports(self.lang, blocks.get(path, "") + "\n" + bases[path])
            for st in self.styles:
                for d in defs_or_none(path, text, st, imports):
                    if ws:
                        got["conv_style"].append(st == ws)
                    if wk:
                        got["conv_pattern"].append(kind_of(d) == wk)
            if kind == "method" and wn:
                got["conv_naming"].append(naming(name) == wn)
        return {k: round(sum(v) / len(v), 4) if v else None for k, v in got.items()}

    def _calls(self, text: str, c: dict, avail: set[str]) -> set[tuple[str, str]]:
        """Project calls (§2.2b), written again on the cleaned text: PHP by regex, Java/C# over the tree-sitter tree."""
        text = clean(self.lang, text)
        if self.lang == "php":
            body = text[text.find("{") + 1:] if "{" in text else text
            out = set(re.findall(r"\$this\s*\??->\s*(\w+)\s*\??->\s*(\w+)\s*\(", body))  # `?->` too
            out |= {("self", x) for x in re.findall(r"\$this\s*\??->\s*(\w+)\s*\(", body)}
            out |= {("self", x) for x in re.findall(PHP_RELATIVE + r"\s*::\s*(\w+)\s*\(", body)}
            for x, f in re.findall(PHP_CLASS_AT + r"(\\?[A-Z][\w\\]*)\s*::\s*(\w+)\s*\(", body):
                if x.lower() not in ("self", "static", "parent") and self.parents.find(c, x)[0] in ("in", "global"):
                    out.add((x.lstrip("\\"), f))
            return out
        m = LAMBDA_START.search(text) if self.lang == "java" else None
        src = ("class __X { void __m() { __f(" + text[m.start(1):close_brace(text, m.end() - 1) + 1] + "); } }") if m else "class __X {\n" + text + "\n}"
        tree, b = ts_parse(self.lang, src), src.encode()
        T = lambda n: b[n.start_byte:n.end_byte].decode("utf-8", "replace")
        local = set()
        for n in nodes(tree.root_node, ("formal_parameter", "parameter", "catch_formal_parameter", "variable_declarator",
                                        "enhanced_for_statement", "inferred_parameters", "implicit_parameter", "lambda_expression",
                                        "foreach_statement", "resource", "declaration_expression", "catch_declaration",
                                        # pattern and query locals on the call side, as in member_uses
                                        "instanceof_expression", "declaration_pattern", "from_clause")):
            if n.type == "implicit_parameter":
                local.add(T(n))
            elif n.type == "inferred_parameters":
                local |= {T(x) for x in n.named_children if x.type == "identifier"}
            elif n.type == "lambda_expression":
                q = n.child_by_field_name("parameters")
                if q is not None and q.type == "identifier":
                    local.add(T(q))
            elif n.type == "foreach_statement":
                q = n.child_by_field_name("left")
                if q is not None:
                    local.add(T(q))
            elif n.child_by_field_name("name") is not None:
                local.add(T(n.child_by_field_name("name")))
        local |= pattern_locals(tree.root_node, T)
        local |= {T(x) for n in nodes(tree.root_node, ("tuple_pattern", "query_expression"))
                  for x in n.named_children if x.type == "identifier"}
        local |= {T(next(x for x in n.named_children if x.type == "identifier"))
                  for n in nodes(tree.root_node, ("let_clause", "join_clause"))
                  if any(x.type == "identifier" for x in n.named_children)}
        this = lambda n: n is not None and n.type in ("this", "this_expression")

        def anon(n) -> bool:  # own way: under the body of `new T(…) { … }`, the anonymous class's own
            while n is not None:
                if n.type == "class_body" and n.parent is not None and n.parent.type == "object_creation_expression":
                    return True
                n = n.parent
            return False
        out = set()
        for n in nodes(tree.root_node, ("method_invocation",)):
            obj, name = n.child_by_field_name("object"), T(n.child_by_field_name("name"))
            if obj is None and name in avail or this(obj):
                if not anon(n):
                    out.add(("self", name))
            elif obj is not None and obj.type == "identifier" and T(obj) not in local and T(obj) in avail:
                out.add((T(obj), name))
            elif obj is not None and obj.type == "field_access" and this(obj.child_by_field_name("object")) and not anon(n):
                out.add((T(obj.child_by_field_name("field")), name))
        for n in nodes(tree.root_node, ("invocation_expression",)):
            f = n.child_by_field_name("function")
            if f is not None and f.type == "identifier" and T(f) in avail:
                out.add(("self", T(f)))
            elif f is not None and f.type == "member_access_expression":
                e, nm = f.child_by_field_name("expression"), f.child_by_field_name("name")
                if e is None or nm is None:
                    continue
                if this(e):
                    out.add(("self", T(nm)))
                elif e.type == "identifier" and T(e) not in local and T(e) in avail:
                    out.add((T(e), T(nm)))
                elif e.type == "member_access_expression" and this(e.child_by_field_name("expression")):
                    out.add((T(e.child_by_field_name("name")), T(nm)))
        return out

    def gold_text(self, r: dict) -> str:
        """The gold answer, rendered again: a new function's Phase 2 code; a preexisting one as V(r) shows it at t_vis
        plus the record's CREATE annotations; one fenced block per target file."""
        t, cdefs, by = r["t_vis"], set(r["create_def_ids"]), {}
        for c in r["create"]:
            code = c["code"]
            if c["preexisting"]:
                f = self.fns[c["fn_id"]]
                keep = {a["line"] for a in f["annotations"] if a["def_id"] in cdefs or (a["time"] is not None and a["time"] < t)}
                drop = {a["line"] for a in f["annotations"]} - keep
                code = "\n".join(l for i, l in enumerate(f["code"].splitlines(), start=f["first_line"]) if i not in drop)
            by.setdefault(c["path"], []).append(code)
        fence = {"php": "php", "java": "java", "ruby": "ruby", "csharp": "csharp"}[self.lang]
        mark = "#" if self.lang == "ruby" else "//"
        ticks = "`" * 3
        return "\n\n".join(f"{ticks}{fence}\n{mark} FILE: {p}\n" + "\n\n".join(v) + f"\n{ticks}" for p, v in by.items())

    def shell_keys(self, r: dict) -> dict:
        """path -> method keys of the record's preexisting CREATE items (the shells) in that file."""
        shells = defaultdict(set)
        for c in r["create"]:
            if c["preexisting"]:
                for u in units(self.lang, self.fns[c["fn_id"]]["code"])[1]:
                    if u["kind"] == "method":
                        shells[c["path"]].add(self.key(u["name"], u["arity"]))
        return shells

    def gold(self, r: dict, labels_row: dict) -> dict:
        """The gold answer scored as an output (the gold control row)."""
        if r["id"] not in self._gold:
            self._gold[r["id"]] = self.score(r, labels_row, self.gold_text(r))
        return self._gold[r["id"]]

    def reference(self, r: dict, labels_row: dict) -> dict:
        """The gold of CA and call_precision: `gold` whose call set of a binder (a function binding a CREATE step) that
        is one of the record's preexisting shells in its file (shell_keys), whose name, as the gold answer writes it,
        the view of that file shows (`visible`; PHP in any case) and whose call set is not empty, also holds ("self",
        that name): the gold only adds an annotation to a function the model is shown, and a new step calling it does
        the same at run time (D11 (e), fixed 26/09/2026). An empty call set gains nothing, so CA's definition never
        changes. Ruby: never."""
        if ("ref", r["id"]) in self._gold:
            return self._gold[("ref", r["id"])]
        g, ref = self.gold(r, labels_row), None
        if self.lang != "ruby" and g.get("binder"):
            shells = self.shell_keys(r)
            targets = list(dict.fromkeys(c["path"] for c in r["create"]))
            written = {(path, self.key(u["name"], u["arity"])): u["name"] for path, code in blocks_of(self.gold_text(r), targets)[1].items()
                       for u in units(self.lang, code)[1] if u["kind"] == "method"}
            tree, calls = er_tree(r, self.fns, self.ghosts, self.git)[0], dict(g["calls"])
            for ident in dict.fromkeys(g["binder"].values()):
                path, k = ident
                name = written.get(ident)
                if k in shells.get(path, ()) and calls.get(ident) and name is not None \
                        and self.fold(name) in {self.fold(n) for n in self.visible(path, r, tree)}:
                    calls[ident] = set(calls[ident]) | {("self", name)}
            ref = {**g, "calls": calls} if calls != g["calls"] else None
        self._gold[("ref", r["id"])] = ref or g
        return self._gold[("ref", r["id"])]

    def ca(self, r: dict, labels_row: dict, out: dict) -> tuple[bool, int]:
        g = self.reference(r, labels_row)
        steps_of = defaultdict(set)
        for step, ident in g.get("binder", {}).items():
            steps_of[ident].add(step)
        defined, agree = False, True
        for ident, steps in steps_of.items():
            key = lambda cs: {(r0, self.fold(f)) for r0, f in cs}  # PHP method names in any case; receivers as written
            gc = key(g["calls"].get(ident, set()))
            if not gc:
                continue
            defined = True
            oc = key(set().union(*(out.get("calls", {}).get(out.get("binder", {}).get(st), set()) for st in steps)))
            agree &= bool(oc & gc)
        return defined, int(defined and out["bv"] == 1 and agree)

    def overlap(self, r: dict, labels_row: dict, out: dict) -> dict:
        """n_calls = |O| and call_precision = |O & G| / |O| (4 decimals, None when O is empty), O and G the
        project calls of the output's and the gold answer's functions that bind CREATE steps (the gold's from
        `reference`, with D11 (e))."""
        g = self.reference(r, labels_row)
        key = lambda cs: {(r0, self.fold(f)) for r0, f in cs}  # PHP method names in any case; receivers as written
        o = key(set().union(*(out["calls"].get(f, set()) for f in set(out["binder"].values()))))
        gold = key(set().union(*(g["calls"].get(f, set()) for f in set(g["binder"].values()))))
        return {"n_calls": len(o), "call_precision": round(len(o & gold) / len(o), 4) if o else None}

    def _args(self, r):
        text = "\n".join(x for x in (r["background"], r["scenario"]) if x).split("\n")
        idx = [i for i, l in enumerate(text) if re.match(r"^\s*(?:Given|When|Then|And|But|\*)\s+\S", l)]
        flags = [next((l.strip() for l in text[i + 1:] if l.strip() and not l.strip().startswith("#")), "").startswith(("|", '"""', "`" * 3)) for i in idx]
        if len(flags) != len(r["steps"]):
            flags = [False] * len(r["steps"])
        out = {}
        for s, f in zip(r["steps"], flags):
            out[s] = out.get(s, False) or f
        return out

    def _placeholders(self, d, step: str | None = None):
        """N of condition 4: a Turnip :token or a Cucumber expression {…} is one; a regex, per framework (regex_args).
        Condition 4 (§2.2) at 25ba873: Behat gets what PHP's preg_match returns for the step, whose array
        ends at the last group that matched; so for PHP, N is the step's groups with the unmatched ones at the END
        dropped (a named group once, as it is numbered once), when the step matches the compiled pattern."""
        if d.kind == "turnip":
            return len(re.findall(r":\w+", d.pattern))
        if d.kind == "cucumber":
            return sum(1 for m in _TOKEN.finditer(d.pattern) if m.group(0).startswith("{"))
        if self.lang == "php" and step is not None:
            c = compile_pattern(d.kind, d.pattern)
            hit = c and (c[0].fullmatch(step) if c[1] == "fullmatch" else c[0].search(step))
            if hit:
                groups = list(hit.groups())
                while groups and groups[-1] is None:
                    groups.pop()
                return len(groups)
        return regex_args(d.pattern, self.lang)

    @staticmethod
    def _hardcoded(d, step):
        try:
            if d.kind == "turnip":
                rx, mode = re.compile(turnip_to_regex(d.pattern).replace(_BEHAT_TOKEN, "(" + _BEHAT_TOKEN + ")"), re.I), "full"
            elif d.kind == "cucumber":
                parts, pos = [], 0
                for m in _TOKEN.finditer(d.pattern):
                    parts.append(re.escape(d.pattern[pos:m.start()]))
                    tok = m.group(0)
                    parts.append("(" + _CUKE_ARGS.get(m.group(1), r".*?") + ")" if tok.startswith("{") else
                                 f"(?:{re.escape(m.group(2))})?" if m.group(2) is not None else
                                 "(?:" + "|".join(re.escape(x) for x in tok.split("/")) + ")" if "/" in tok else re.escape(tok))
                    pos = m.end()
                parts.append(re.escape(d.pattern[pos:]))
                rx, mode = re.compile("^" + "".join(parts) + "$"), "full"
            else:
                c = compile_pattern(d.kind, d.pattern)
                if c is None:  # a pattern `re` rejects (e.g. (?'n'…), \p{N}): hard-coded, as argcheck.hardcoded says
                    return True
                rx, mode = c[0], ("full" if c[1] == "fullmatch" else "search")
        except re.error:
            return True
        mm = rx.fullmatch(step) if mode == "full" else rx.search(step)
        if mm is None:
            return True
        caps = [mm.span(i) for i in range(1, rx.groups + 1) if mm.span(i) != (-1, -1)]
        plain = re.sub(r"\\(.)", r"\1", d.pattern)
        q = [(x.start(), x.end()) for x in re.finditer(r'"[^"]*"', step)]
        vals = [(a + 1, b - 1) for a, b in q if b - a > 2]
        vals += [(x.start(), x.end()) for x in re.finditer(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?![\w])", step) if not any(a <= x.start() < b for a, b in q)]
        for a, b in vals:
            if any(a < e and s < b for s, e in caps):
                continue
            v = step[a:b]
            pat = rf"(?<![\w.]){re.escape(v)}(?![\w])" if v[:1].isalnum() or v[-1:].isalnum() else re.escape(v)
            if re.search(pat, plain):
                return True
        return False

    @staticmethod
    def _ruby(text):
        tree = ts_parse("ruby", text)
        b = text.encode()
        T = lambda n: b[n.start_byte:n.end_byte].decode("utf-8", "replace")
        local, calls, cls = set(), set(), set()
        dead = ruby_dead(text, tree.root_node)

        def quoted(n):  # inside a string (interpolation included), a comment or a dead line: not code that runs
            while n is not None:
                if n.type in ("string", "comment") or n.start_point[0] in dead:
                    return True
                n = n.parent
            return False
        params = ("block_parameters", "method_parameters", "lambda_parameters")  # a helper `def` left over declares some too
        for n in nodes(tree.root_node, params + ("assignment", "call", "identifier", "constant", "scope_resolution")):
            if quoted(n):
                continue
            if n.type in params:
                local |= {T(x) for x in n.named_children if x.type == "identifier"}
            elif n.type == "assignment" and n.child_by_field_name("left") is not None and n.child_by_field_name("left").type == "identifier":
                local.add(T(n.child_by_field_name("left")))
            elif n.type == "call" and n.child_by_field_name("receiver") is None and n.child_by_field_name("method") is not None:
                calls.add(T(n.child_by_field_name("method")))
            elif n.type == "identifier" and n.parent is not None and n.parent.type in ("argument_list", "body_statement", "block_body", "do_block"):
                calls.add(T(n))
            elif n.type in ("constant", "scope_resolution") and (n.parent is None or n.parent.type != "scope_resolution"):
                cls.add(T(n))
        calls -= local | set(KW)
        return calls, calls, {c for c in cls if c not in KW}


def load_jsonl(p: Path) -> list[dict]:
    """The rows of a JSONL file as 3A reads them: split on "\\n" only from the decoded bytes, the final empty
    line dropped, every other line a JSON value, so a blank line inside the file raises and a lone "\\r" is no
    separator. str.splitlines() would also break at U+2028, U+2029, U+0085, \\x0b-\\x0c and \\x1c-\\x1e, which
    json.dumps(ensure_ascii=False) leaves raw inside an output's text."""
    lines = p.read_bytes().decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    return [json.loads(line) for line in lines]


class InputError(Exception):
    """An input the comparison cannot read or trust: main() refuses it with exit 3, as score_run does."""


SHAPE = {"text": str, "step_def_ids": list, "steps": list, "styles": list}  # the columns that are not scalar
SCALAR = (str, int, float, bool, type(None))


def need(row, cols: tuple[str, ...], where: str, optional: tuple[str, ...] = ()) -> dict:
    """A row of an input file: a JSON object holding `cols` (and `optional` ones, when it has them) of the right type:
    else a refusal."""
    if not isinstance(row, dict):
        raise InputError(f"{where}: not a JSON object")
    for c in cols + tuple(c for c in optional if c in row):
        if c not in row:
            raise InputError(f"{where}: no `{c}`")
        if not isinstance(row[c], SHAPE.get(c, SCALAR)):
            raise InputError(f"{where}: `{c}` is not a {SHAPE[c].__name__ if c in SHAPE else 'JSON scalar'}")
    return row


def load(p: Path, cols: tuple[str, ...] = (), optional: tuple[str, ...] = ()) -> list[dict]:
    """An input that is missing, not JSONL or whose rows lack `cols`: refuse, as score_run does."""
    if not p.exists():
        raise InputError(f"missing input {p}")
    try:
        rows = load_jsonl(p)
    except ValueError as e:
        raise InputError(f"{p}: not JSONL ({e})") from e
    except OSError as e:
        raise InputError(f"{p}: {e.strerror or e}") from e
    return [need(x, cols, f"{p}: row {i}", optional) for i, x in enumerate(rows, 1)]


def read_json(p: Path, kind: type):
    if not p.exists():
        raise InputError(f"missing input {p}")
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise InputError(f"{p}: not JSON ({e})") from e
    if not isinstance(x, kind):
        raise InputError(f"{p}: not a JSON {kind.__name__}")
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--full-root", type=Path, default=Path("repos-full"))
    ap.add_argument("--outputs", type=Path, nargs="+", required=True)
    ap.add_argument("--scores", type=Path, required=True, help="evaluate's score rows to compare with")
    ap.add_argument("--infra", type=Path, help="3A's infra.json (next to the prompts): the answer "
                                                 "infrastructure masked for train and dev records, "
                                                 "{repo: [[declaring path, name], ...]}; without it, none")
    ap.add_argument("--symtimes", type=Path, help="the sealed symtimes.jsonl (what ④(a) shows); without it, each "
                                                    "file's rows are re-derived from git history")
    ap.add_argument("--partition", choices=("test", "dev", "aux_test"),
                    help="score only this partition's rows; its test/aux_test partner's are skipped")
    args = ap.parse_args()
    try:
        return compare_files(args)
    except InputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3


def compare_files(args) -> int:
    key_cols = ("repo", "feature_path", "scenario_line", "suite")
    key = lambda x: (x["repo"], x["feature_path"], x["scenario_line"], x["suite"])
    records, part_of = {}, {}
    for part in ("test", "dev", "aux_test"):
        for r in load(args.phase2 / "records" / f"{part}.jsonl", ("id", "commit") + key_cols):
            records[r["id"]], part_of[r["id"]] = r, part
    labels = {key(l): l for l in load(args.phase2 / "labels.jsonl", key_cols + ("step_def_ids",))}
    pairs = {key(p): p for p in load(args.pairs, key_cols + ("steps",))}
    cat = defaultdict(dict)
    for f in load(args.phase2 / "catalog.jsonl", ("repo", "fn_id")):
        cat[f["repo"]][f["fn_id"]] = f
    suites = defaultdict(set)
    for k, l in labels.items():
        if k not in pairs:
            raise InputError(f"{args.pairs}: no pair for the label {k}")
        for s, ids in zip(pairs[k]["steps"], l["step_def_ids"]):
            if ids:
                suites[(k[0], k[3])].add((s, tuple(ids)))
    srcs = {s["repo"]: s for s in read_json(args.config, list) if isinstance(s, dict) and "repo" in s}
    forbidden = read_json(args.phase2 / "forbidden.json", dict)
    infra = read_json(args.infra, dict) if args.infra else {}
    sym = defaultdict(dict)
    for row in (load(args.symtimes, ("repo", "path")) if args.symtimes else []):
        sym[row["repo"]][row["path"]] = row
    # what makes the comparison unsafe is refused before any scoring:
    # an output of an unknown record or of one without a label, a row of --scores that no output matches or a key twice
    # in it, and a repository whose scored records are at more than one commit (one Scorer is built per repository)
    out_cols = ("record_id", "model", "seed", "condition", "text")
    outputs = [(f, i, o) for f in args.outputs for i, o in enumerate(load(f, out_cols, ("variant",)), 1)]
    okey = lambda o: (o["record_id"], o["model"], o["seed"], o["condition"], o.get("variant"))
    skipped = 0
    # the generation run writes the test and aux_test rows of a run into one workdir;
    # as score_run, the partner's rows are skipped and counted, any other partition refused
    if args.partition:
        kept = []
        for f, i, o in outputs:
            other = part_of.get(o["record_id"])
            if other is None or other == args.partition:
                kept.append((f, i, o))
            elif {other, args.partition} == {"test", "aux_test"}:
                skipped += 1
            else:
                raise InputError(f"{f}: row {i}: record {o['record_id']} is a {other} record, not {args.partition} "
                                 "(only test and aux_test share a generation workdir)")
        outputs = kept
    commits = defaultdict(set)
    for f, i, o in outputs:
        r = records.get(o["record_id"])
        if r is None:
            raise InputError(f"{f}: row {i}: unknown record_id {o['record_id']}")
        if key(r) not in labels:
            raise InputError(f"{f}: row {i}: no label for the record {o['record_id']}")
        if r["repo"] not in srcs:
            raise InputError(f"{f}: row {i}: the repository {r['repo']} is not in {args.config}")
        commits[r["repo"]].add(r["commit"])
    for repo, shas in commits.items():
        if len(shas) > 1:
            at = ", ".join(sorted(map(str, shas)))
            raise InputError(f"{repo}: its scored records are at more than one commit ({at}); "
                             "one Scorer is built per repository, on the first record's commit")
    theirs = {}
    for i, x in enumerate(load(args.scores), 1):
        ctl = isinstance(x, dict) and "control" in x
        cols = ("record_id", "control") if ctl else ("record_id", "model", "seed", "condition")
        need(x, cols, f"{args.scores}: row {i}", ("variant",))
        if "control" in x:
            k = (x["record_id"], "control", "-", x["control"], "control")
        else:
            k = (x["record_id"], x["model"], x["seed"], x["condition"], x.get("variant"))
        if k in theirs:
            raise InputError(f"{args.scores}: row {i}: duplicate key {k}")
        theirs[k] = x
    scored = {okey(o) for _, _, o in outputs}
    unmatched = [k for k in theirs if k not in scored]
    if unmatched:
        raise InputError(f"{len(unmatched)} rows of --scores {args.scores} have no output, the first {unmatched[0]}")
    scorers = {}
    diff, compared, n = Counter(), Counter(), 0
    for f, i, o in outputs:
        r = records[o["record_id"]]
        if r["repo"] not in scorers:
            s = need(srcs[r["repo"]], ("dir", "language", "styles"), f"{args.config}: {r['repo']}")
            try:
                suite_steps = {su: sorted(v) for (rp, su), v in suites.items() if rp == r["repo"]}
                git = Git(args.full_root / f"{s['dir']}.git", r["commit"])
                scorers[r["repo"]] = Scorer(r["repo"], s["language"], s["styles"], cat[r["repo"]], git, suite_steps,
                                            sym[r["repo"]], forbidden.get(r["repo"], {}), infra.get(r["repo"], ()))
            except ValueError as e:  # the catalog does not fit HEAD: refuse, as score_run does (exit 3)
                raise InputError(f"{r['repo']}: {e}") from e
        sc = scorers[r["repo"]]
        mine = sc.score(r, labels[key(r)], o["text"])
        mine["ca_defined"], mine["ca"] = sc.ca(r, labels[key(r)], mine)
        mine.update(sc.overlap(r, labels[key(r)], mine))
        other = theirs.get(okey(o))
        n += 1
        who = {"record_id": o["record_id"], "condition": o["condition"], "model": o["model"], "seed": o["seed"],
               "variant": o.get("variant")}
        if other is None:
            diff["missing_in_scores"] += 1
            print(json.dumps({"differs": "missing_in_scores", **who}, ensure_ascii=False))
            continue
        # the ROW columns the row carries: a controls.jsonl row has open_resolution, n_calls and call_precision
        # (D11); a scores.jsonl row has them all; ca_ref.jsonl (D11: with ca_ref_v3names) is not read here
        for k in ["bv", "ca", "ca_defined"] + [f"bv_{c}" for c in CONDS] + [c for c in ROW if c in other]:
            compared[k] += 1
            if mine[k] != other[k]:
                diff[k] += 1  # every differing cell is printed, not the first three of a column
                cell = {"differs": k, **who, "independent": mine[k], "evaluate": other[k]}
                print(json.dumps(cell, ensure_ascii=False))
    summary = {"scored": n, "compared": dict(compared), "differences": dict(diff)}
    if args.partition:
        summary["skipped_other_partition"] = skipped
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
