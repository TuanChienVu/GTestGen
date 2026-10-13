"""The one code extractor of Phase 3 (§1.4), shared by the gold answer, the stubs and every condition: an output is a
series of fenced blocks, one per target file, whose first non-blank line is `// FILE: <path>` (`# FILE:` for Ruby).
Status: ok; unclosed (the last block never closes, its code is kept); none (no fenced block); malformed (some block has
no FILE line, or names a file outside the record's targets); multi_file_fence (a fence holds more than one FILE line,
otherwise ok). Malformed blocks are dropped and the others are kept. Precedence: none > malformed > unclosed >
multi_file_fence > ok.

Fence rules (scripts/score_independent.py mirrors every rule):
- lines are split on "\n" only, a trailing "\r" dropped (CRLF), never str.splitlines(): U+2028, U+2029, U+0085,
  \x0b, \x0c and \x1c-\x1e stay inside the code;
- a fence opens on three or more backticks followed by any info string without a backtick (CommonMark):
  "php", " php", "php:src/A.php", "php {1,3}" or nothing after the backticks;
- it closes on a line of backticks only (surrounding whitespace allowed), at least as many as it opened with; a fence
  of four keeps a three-backtick line as code;
- a FILE line (FILE_LINE, as on a fence's first line) inside a fence starts a new file; the lines before it belong to
  the previous file, or, before the fence's first FILE line, to a block without a path (malformed) when not blank;
- a three-backtick line inside a three-backtick fence still closes it early (a documented limit), which
  `Extraction.truncated` detects: a fence opened by backticks only (no info string) whose first non-blank line is not
  a FILE line, coming right after (as the next fence) a closed fence whose first non-blank line is a FILE line.

Rules at the edges of a fence (§1.4, fixed 28/09/2026, before any output; scripts/score_independent.py mirrors them):
- head tag line: at the head of a fence, before its first FILE line, a line holding only `<?php` counts as blank and
  as a stripped tag (Block.tag_before, php_tag_stripped): the block is still the FILE line's;
- reopen: inside an open fence, a fence-open line RIGHT before a FILE line is a close and a reopen: it is dropped, not
  the previous file's code, and the status stays multi_file_fence;
- narrowed truncation: the truncation rule above is narrowed, not replaced: such a bare fence counts only when it
  never closes or holds a FILE line further down, so a closed bare fence of a shell command after the code is none."""
from __future__ import annotations

import re
from dataclasses import dataclass

OPEN = re.compile(r"^\s*(`{3,})[^`]*$")  # CommonMark: any info string without a backtick
CLOSE = re.compile(r"^\s*(`{3,})\s*$")  # a closing fence: backticks only, at least as many as the opening
FILE_LINE = re.compile(r"^\s*(?://|#)\s*FILE:\s*(\S+)\s*$")
PHP_OPEN = re.compile(r"\A\s*<\?php(?:[ \t]*\n|[ \t]+|\Z)", re.I)  # tolerated in every condition, any case
PHP_CLOSE = re.compile(r"\n?[ \t]*\?>\s*\Z")
PHP_TAG_LINE = re.compile(r"^\s*<\?php\s*$", re.I)  # §1.4: a head line holding only the open tag


def php_open_tag(code: str) -> tuple[str, bool]:
    """(code, stripped) of one PHP block without a leading `<?php` open tag (any case, after optional whitespace; with
    the spaces after it and one line break, or the spaces before code on its line) and without a trailing `?>` (with the
    whitespace around it and one line break before it); `stripped` = either was there. Tolerant for every condition
    (26/09/2026), and per BLOCK (§1.4, 27/09/2026), so Extraction.by_path applies it to every PHP block before it joins
    the blocks of one file."""
    out = PHP_CLOSE.sub("", PHP_OPEN.sub("", code, count=1), count=1)
    return (out, True) if out != code else (code, False)


@dataclass(frozen=True)
class Block:
    path: str | None  # None when the first non-blank line is not a FILE line
    code: str  # the lines after the FILE line (the whole segment when there is none), trailing blank lines removed
    closed: bool
    usable: bool  # a FILE line that names one of the record's target files
    tag_before: bool = False  # §1.4: a `<?php` line stood before the fence's first FILE line, this block's


@dataclass(frozen=True)
class Extraction:
    status: str
    blocks: tuple[Block, ...]
    truncated: bool = False  # a fence detectably closed early by a backtick line of its code

    def by_path(self) -> dict[str, str]:
        """Code of the usable blocks, merged per target file in output order (two blocks for one file are joined). A PHP
        block (of a `.php` target) loses its own `<?php` / `?>` tags first (php_open_tag; §1.4: per block, so an output
        that opens, or closes, every fence with a tag reads as one that writes one block); a block that is then empty is
        no code."""
        out: dict[str, list[str]] = {}
        for b in self.blocks:
            code = php_open_tag(b.code)[0] if b.usable and b.path.endswith(".php") else b.code
            if b.usable and code.strip():
                out.setdefault(b.path, []).append(code)
        return {p: "\n\n".join(cs) for p, cs in out.items()}

    @property
    def php_tag_stripped(self) -> bool:
        """A usable PHP block had its own `<?php` / `?>` tag, which by_path dropped (describe's php_tag_stripped)."""
        return any(b.usable and b.path.endswith(".php") and (b.tag_before or php_open_tag(b.code)[1]) for b in self.blocks)


def _norm_path(p: str) -> str:
    return p.removeprefix("./")


def _closes(line: str, ticks: str) -> bool:
    m = CLOSE.match(line)
    return m is not None and len(m.group(1)) >= len(ticks)


def extract(text: str, targets: list[str]) -> Extraction:
    lines = [l[:-1] if l.endswith("\r") else l for l in text.split("\n")]  # never splitlines()
    blocks: list[Block] = []
    i, multi, truncated, after_file = 0, False, False, False
    while i < len(lines):
        m = OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, body, j = m.group(1), [], i + 1
        while j < len(lines) and not _closes(lines[j], ticks):
            body.append(lines[j])
            j += 1
        closed = j < len(lines)
        body = [l for n, l in enumerate(body)  # §1.4, reopen: a fence-open line right before a FILE line is dropped
                if not (OPEN.match(l) and n + 1 < len(body) and FILE_LINE.match(body[n + 1]))]
        starts = [n for n, l in enumerate(body) if FILE_LINE.match(l)]  # every FILE line starts a file
        head = body[:starts[0]] if starts else body
        tag = bool(starts) and any(PHP_TAG_LINE.match(l) for l in head)  # §1.4, head tag line
        segs = [(None, "\n".join(head).rstrip())] if not starts or any(l.strip() and not (tag and PHP_TAG_LINE.match(l))
                                                                         for l in head) else []
        segs += [(_norm_path(FILE_LINE.match(body[a]).group(1)), "\n".join(body[a + 1:b]).rstrip())
                 for a, b in zip(starts, starts[1:] + [len(body)])]
        multi |= len(starts) > 1
        truncated |= (after_file and lines[i].strip() == ticks and segs[0][0] is None
                      and (not closed or bool(starts)))  # §1.4, narrowed truncation: unclosed or a FILE line below
        after_file = segs[0][0] is not None  # a fence that never closes ends the text: no fence follows it
        blocks += [Block(path, code, closed or n < len(segs) - 1, path in targets,  # a FILE line closes the segment
                         tag and path is not None and path == segs[0][0] and n == 0)
                   for n, (path, code) in enumerate(segs)]
        i = j + 1
    if not blocks:
        status = "none"
    elif any(not b.usable for b in blocks):
        status = "malformed"
    elif not blocks[-1].closed:
        status = "unclosed"
    else:
        status = "multi_file_fence" if multi else "ok"
    return Extraction(status, tuple(blocks), truncated)
