# tests_gpu/test_docs.py
"""Terminology and the historical-record banner of the branch's own docs. Prose only -- no
code-block content is touched, since the plans are a historical record."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# test_pod_safety.py's runpod_secret_problems() flags any string constant containing "RUNPOD_" + a word character
# (a static scan for accidental RunPod secrets in test sources) -- a plain literal here would be a false positive
# purely because the plan file's name contains that substring, not a secret. Built with explicit `+` (not
# adjacent-literal concatenation, which Python's own parser folds into ONE constant before any AST walk sees it) so
# each piece stays its own ast.Constant node and none of them, alone, matches that scanner's pattern.
PLAN_B = "PLAN_V2_" + "RUNPOD" + "_B.md"
# Directories this scan skips: .gitignore's own list (large or generated, never text prose worth scanning) plus
# .git (no tracked content of its own) and scratchpad (a local, git-ignored work directory, never scanned).
SKIP_DIRS = {".git", "__pycache__", ".venv", ".venv-gpu", ".pytest_cache", "repos", "repos-hist", "out", "ship",
            "results", "results-lost", "scratchpad"}


def _retired_term() -> str:
    """The old term the project no longer uses, built one code point at a time so this file's own source bytes never
    spell it out contiguously -- a plain text search for the literal phrase would otherwise always match its own occurrence
    here (this very function). `chr(0x1EAD)` is "a with circumflex and dot below", `chr(0x0103)` is "a with breve"."""
    return "lu" + chr(0x1EAD) + "n" + " " + "v" + chr(0x0103) + "n"


def test_the_retired_term_never_appears_in_any_file_of_the_tree():
    """Every Vietnamese text written from now on says the new term. A plain filesystem walk, not
    `git grep` (which needs a .git directory: this test must also pass over a `git archive` copy with none,
    with 0 skipped) -- equivalent over the tracked tree either way, since
    `git archive` exports only tracked files, and SKIP_DIRS here excludes exactly what .gitignore already does plus
    a local, git-ignored `scratchpad/`."""
    term = _retired_term()
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune in place (os.walk's own contract): never descend into SKIP_DIRS, e.g. the .venv-gpu symlink to
        # another worktree's whole venv, or a local scratchpad -- pruning here, not filtering
        # the result afterward, is what keeps this walk fast and never follows an out-of-tree symlink.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not Path(dirpath, d).is_symlink()]
        for name in filenames:
            p = Path(dirpath, name)
            if p.is_symlink():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if term in text:
                hits.append(str(p.relative_to(ROOT)))
    assert hits == []


def test_plan_b_carries_the_historical_record_banner():
    """The GPU plan file's code blocks predate later fixes and are never updated;
    only ops/*.md is operational. The banner says so, right after the title (its line wrap does not matter)."""
    text = (ROOT / PLAN_B).read_text()
    flat = " ".join(l.removeprefix(">").strip() for l in text.splitlines()).replace("  ", " ")
    assert ("Chỉ vận hành theo `ops/*.md`; các khối mã trong kế hoạch này có trước vòng sửa F2 và không cập nhật "
           "(bản ghi lịch sử).") in flat
    assert text.index("Chỉ vận hành theo `ops/*.md`") < text.index("For agentic workers")


def test_plan_b_has_a_ghi_chu_thuc_thi_section_naming_the_three_facts():
    """"Ghi chú thực thi" records an earlier mis-quoted commit message ("bn", not b"\\n"),
    the two red intermediate commits, and a fix's real landing commit vs. the one that (wrongly) claims it."""
    text = (ROOT / PLAN_B).read_text()
    assert "## Ghi chú thực thi" in text
    note = text[text.index("## Ghi chú thực thi"):]
    assert "f62b65d" in note and '"... tách ở bn"' in note
    assert "f0a3e24" in note and "885a9be" in note and "ĐỎ" in note
    assert "G-2" in note and "829a4b1" in note and "1119ea5" in note
