"""body_key() and body_key_nosig(): normal forms of a step function's body, so that two functions with different ids
(another file or another line) but the same code are recognised as twins (fixed 24/09/2026).
Two functions are twins when either key is equal. near_duplicates() finds copies that are not equal but close:
difflib ratio of the nosig texts >= NEAR_RATIO. containments() finds functions that hold most of a long
answer inside a longer body, which the ratio misses because it penalises the length difference."""
from __future__ import annotations

import difflib
import re
from collections import Counter

MIN_CHARS = 60
NEAR_RATIO = 0.90
CONTAIN_MIN_CHARS = 200
CONTAIN_SHARE = 0.90
_COMMENT = re.compile(r"^\s*(/\*\*|\*|\*/|//|#(?!\[))")  # docblock and comment lines; `#[` (PHP attribute) is kept


def _text(fn: dict) -> str:
    """The catalog function's code (HEAD) without (i) its annotation lines, by line number counted from `first_line`,
    and (ii) docblock/comment lines, joined with spaces and whitespace collapsed."""
    ann = {a["line"] for a in fn["annotations"]}
    kept = [l for i, l in enumerate(fn["code"].splitlines(), start=fn["first_line"]) if i not in ann and not _COMMENT.match(l)]
    return " ".join(" ".join(kept).split())


def body_key(fn: dict) -> str | None:
    """_text() with the method signature kept, so the key catches copies of one function, not short idioms: None when
    fewer than MIN_CHARS characters remain."""
    s = _text(fn)
    return s if len(s) >= MIN_CHARS else None


def nosig_text(fn: dict, language: str) -> str:
    """_text() without the declaration: outside Ruby, everything up to the FIRST `{` goes when that prefix
    holds a `(` and no `;` (a multi-line PHP signature goes whole, a Java lambda's parameter list too). In Ruby the
    declaration line is the annotation line, already dropped."""
    s = _text(fn)
    if language != "ruby":
        j = s.find("{")
        if j >= 0 and "(" in s[:j] and ";" not in s[:j]:
            s = s[j + 1:].strip()
    return s


def body_key_nosig(fn: dict, language: str) -> str | None:
    """nosig_text(), so a copy under another method name is a twin too: None when fewer than MIN_CHARS characters."""
    s = nosig_text(fn, language)
    return s if len(s) >= MIN_CHARS else None


def near_duplicates(answers: dict[str, str], candidates: dict[str, str]) -> dict[str, tuple[str, float]]:
    """For every candidate text that reaches NEAR_RATIO with some answer text, the best answer id and its ratio
    (difflib.SequenceMatcher(None, candidate, answer, autojunk=False).ratio()); texts under MIN_CHARS are ignored.
    real_quick_ratio() and quick_ratio() are upper bounds of ratio(), so they only skip pairs that cannot qualify.
    Deterministic: answers and candidates are visited in id order and a tie keeps the answer met first."""
    out: dict[str, tuple[str, float]] = {}
    sm = difflib.SequenceMatcher(None, autojunk=False)
    for f, b in sorted(answers.items()):
        if len(b) < MIN_CHARS:
            continue
        sm.set_seq2(b)
        for g, a in sorted(candidates.items()):
            if len(a) < MIN_CHARS:
                continue
            sm.set_seq1(a)
            if sm.real_quick_ratio() < NEAR_RATIO or sm.quick_ratio() < NEAR_RATIO:
                continue
            ratio = sm.ratio()
            if ratio >= NEAR_RATIO and ratio > out.get(g, ("", 0.0))[1]:
                out[g] = (f, ratio)
    return out


def _masks(text: str) -> dict[str, int]:
    masks: dict[str, int] = {}
    for i, ch in enumerate(text):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    return masks


def _lcs_len(masks: dict[str, int], width: int, other: str) -> int:
    """Length of the longest common subsequence of the text behind `masks` (`width` characters) and `other`
    (bit-parallel, Allison-Dix/Hyyro). SequenceMatcher's matching blocks form a common subsequence, so their total
    never exceeds it: a sound filter that only skips pairs that cannot qualify."""
    full = (1 << width) - 1
    v = full
    for ch in other:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return width - bin(v).count("1")


def containments(answers: dict[str, str], candidates: dict[str, str]) -> dict[str, tuple[str, float]]:
    """For every candidate text holding at least CONTAIN_SHARE of some answer text of at least CONTAIN_MIN_CHARS
    characters, the answer with the largest share and that share. share = sum of the block sizes of
    difflib.SequenceMatcher(None, answer, candidate, autojunk=False).get_matching_blocks() / len(answer): a = the answer,
    b = the candidate. Only candidates at least CONTAIN_SHARE * len(answer) long are compared; the multiset of shared
    characters and then the longest common subsequence (both upper bounds of the matched characters) skip pairs that
    cannot qualify. Deterministic: candidates and answers are visited in id order and a tie keeps the answer met first."""
    long = sorted((f, a, Counter(a)) for f, a in answers.items() if len(a) >= CONTAIN_MIN_CHARS)
    out: dict[str, tuple[str, float]] = {}
    sm = difflib.SequenceMatcher(None, autojunk=False)
    for g, b in sorted(candidates.items()):
        counts_b, masks_b = None, None
        for f, a, counts_a in long:
            need = CONTAIN_SHARE * len(a)
            if len(b) < need:
                continue
            counts_b = counts_b or Counter(b)
            if sum((counts_a & counts_b).values()) < need:
                continue
            masks_b = masks_b or _masks(b)
            if _lcs_len(masks_b, len(b), a) < need:
                continue
            if sm.b is not b:
                sm.set_seq2(b)
            sm.set_seq1(a)
            share = sum(m.size for m in sm.get_matching_blocks()) / len(a)
            if share >= CONTAIN_SHARE and share > out.get(g, ("", 0.0))[1]:
                out[g] = (f, share)
    return out
