"""The repo state that scoring runs on (the ER tree: two subtractions, with the phantom rule of phantoms()): HEAD minus
(1) the answer, i.e. a new CREATE function
(preexisting=False) whole and, for a preexisting one, the lines of the record's CREATE annotations and of every
annotation not strictly before t_vis; and (2) every catalog step function of the REPO with no annotation strictly
before t_vis. Nothing is masked: the ER tree is never shown to a model. The same tree serves the gold answer, the stubs,
BV, CA and the ER overlays. Inputs the tree cannot be built from fail closed: a catalog row with no annotation
or a file absent from HEAD raises ValueError."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import tree_sitter as ts


@lru_cache(maxsize=None)
def parser(language: str) -> ts.Parser:
    if language == "php":
        import tree_sitter_php as m
        return ts.Parser(ts.Language(m.language_php()))
    mod = {"java": "tree_sitter_java", "csharp": "tree_sitter_c_sharp", "ruby": "tree_sitter_ruby"}[language]
    return ts.Parser(ts.Language(__import__(mod).language()))


@dataclass(frozen=True)
class Removal:
    lines: dict[str, frozenset[int]]  # path -> 1-based HEAD line numbers deleted
    defs: frozenset[str]  # def ids (annotation lines) that no longer exist in the ER tree
    answer_fns: int
    answer_annotations: int
    future_fns: int
    phantom_fns: int  # commented-out catalog functions: no lines cut, def ids counted as gone


def phantoms(language: str, fns: dict[str, dict], text_of: Callable[[str], str | None]) -> frozenset[str]:
    """Catalog functions whose every annotation starts inside a comment of the HEAD file (a PHP docblock does not count),
    plus every row the catalog flags `dead` (commented out, or an annotation stack with no body line):
    Phase 1's extractor also read commented-out step definitions, and its brace matcher then ran from a "{" inside the
    comment into real code, so their body_end is wrong. Their lines are never cut; their def ids count as gone.
    When the catalog carries the `dead` flags, every commented-out function and every `contains_fn` row must also be
    flagged `dead`; a disagreement stops scoring. So do a row with no annotation, which the test "every
    annotation starts inside a comment" would hold for vacuously, and a catalog file absent from HEAD, whose rows could
    not be checked: all three raise ValueError. A row without its `annotations` key counts as one with none (a
    ValueError, not a bare KeyError)."""
    bare = sorted(fid for fid, fn in fns.items() if not fn.get("annotations"))
    if bare:
        repo = fns[bare[0]].get("repo")
        raise ValueError(f"{len(bare)} catalog rows with no annotation" + (f" (repo {repo})" if repo else "")
                         + f": {bare[:3]}")
    by_path: dict[str, list[dict]] = {}
    for fn in fns.values():
        by_path.setdefault(fn["path"], []).append(fn)
    out = {fid for fid, fn in fns.items() if fn.get("dead")}
    commented_out: set[str] = set()
    for path, group in sorted(by_path.items()):
        text = text_of(path)
        if text is None:
            repo = group[0].get("repo")
            raise ValueError(f"not in HEAD: {path}" + (f" (repo {repo})" if repo else ""))
        src = text.encode()
        spans = []

        def walk(n):
            if n.type in ("comment", "block_comment", "line_comment"):
                if not (language == "php" and src[n.start_byte:n.start_byte + 3] == b"/**"):
                    spans.append((n.start_byte, n.end_byte))
                return
            for c in n.children:
                walk(c)
        walk(parser(language).parse(src).root_node)
        starts = [0]
        for line in src.split(b"\n")[:-1]:
            starts.append(starts[-1] + len(line) + 1)

        def commented(line_no: int) -> bool:
            s0 = starts[line_no - 1]
            e0 = starts[line_no] if line_no < len(starts) else len(src)
            at = s0 + len(src[s0:e0]) - len(src[s0:e0].lstrip())
            return any(a <= at < b for a, b in spans)
        commented_out |= {fn["fn_id"] for fn in group if all(commented(a["line"]) for a in fn["annotations"])}
    if any("dead" in fn for fn in fns.values()):
        dead = {fid for fid, fn in fns.items() if fn.get("dead")}
        odd = sorted((commented_out | {fid for fid, fn in fns.items() if fn.get("contains_fn")}) - dead)
        if odd:
            raise ValueError(f"phantoms disagree with the catalog's dead flag: {odd[:3]}")
    return frozenset(out | commented_out)


def before(a: dict, t: int) -> bool:
    return a["time"] is not None and a["time"] < t


def removal(record: dict, fns: dict[str, dict], ghost: frozenset[str] = frozenset()) -> Removal:
    """Subtractions 1 and 2 (see the module docstring) for one record, over the catalog rows of its repo
    (fn_id -> row); `ghost` is phantoms() of the repo. A record without a t_vis raises ValueError, a t_vis None and a
    record with no t_vis key alike (the key is read with .get: no bare TypeError or KeyError)."""
    t, cdefs = record.get("t_vis"), set(record["create_def_ids"])
    if t is None:
        raise ValueError(f"{record.get('id')}: no t_vis")
    lines: dict[str, set[int]] = {}
    defs: set[str] = set()
    n_fns = n_anns = n_future = n_phantom = 0
    create = {c["fn_id"]: c for c in record["create"]}
    for fid, c in create.items():
        fn = fns[fid]
        if c["preexisting"]:
            gone = [a for a in fn["annotations"] if a["def_id"] in cdefs or not before(a, t)]
            lines.setdefault(fn["path"], set()).update(a["line"] for a in gone)
            defs |= {a["def_id"] for a in gone}
            n_anns += len(gone)
        else:
            lines.setdefault(fn["path"], set()).update(range(fn["first_line"], fn["body_end"] + 1))
            defs |= {a["def_id"] for a in fn["annotations"]}
            n_fns += 1
    for fid, fn in fns.items():
        if fid in create:
            continue
        if fid in ghost:
            defs |= {a["def_id"] for a in fn["annotations"]}
            n_phantom += 1
        elif not any(before(a, t) for a in fn["annotations"]):
            lines.setdefault(fn["path"], set()).update(range(fn["first_line"], fn["body_end"] + 1))
            defs |= {a["def_id"] for a in fn["annotations"]}
            n_future += 1
    return Removal({p: frozenset(v) for p, v in sorted(lines.items())}, frozenset(defs), n_fns, n_anns, n_future, n_phantom)


def cut(text: str, drop: frozenset[int]) -> str:
    """`text` without the given 1-based lines (numbered by "\\n", as the catalog numbers them)."""
    return "\n".join(l for i, l in enumerate(text.split("\n"), start=1) if i not in drop)


def er_tree(rem: Removal, text_of: Callable[[str], str | None]) -> dict[str, str]:
    """Every file the removal changes, as full text. A path absent from HEAD raises ValueError, as in phantoms()."""
    out = {}
    for path, drop in rem.lines.items():
        text = text_of(path)
        if text is None:
            raise ValueError(f"not in HEAD: {path}")
        out[path] = cut(text, drop)
    return out
