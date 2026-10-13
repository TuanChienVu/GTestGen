"""Offline reads from a full (not blob:none) bare clone. GIT_NO_LAZY_FETCH=1 makes a missing object an error instead of
a silent network fetch, so a result never depends on the network."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENV = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
MIN_RENAME_SCORE = 60  # a weaker rename that --follow reports is taken as the file's creation


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=ENV,
                          encoding="utf-8", errors="replace").stdout


def show(repo: Path, commit: str, path: str) -> str:
    """File content at commit. Every failure is an error (unknown commit, absent path, missing blob): callers decide
    from ls_tree() whether a path exists, so an absent object can never pass for an absent file."""
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"], capture_output=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"git cat-file {commit}:{path}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout.decode("utf-8", errors="replace")


def ls_tree(repo: Path, commit: str) -> list[str]:
    return git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()


def versions(repo: Path, commit: str, path: str) -> list[tuple[str, int, str]]:
    """(commit, author time, path at that commit) for every commit that touched the file, following renames, newest
    first. Deleting commits are skipped (no content). History stops, keeping that commit as the file's first version,
    at a COPY (the source file lived on, so this file was new; 68 of 546 histories ran into other files through
    copies) and at a rename whose similarity is below MIN_RENAME_SCORE (--follow then joins an unrelated file, e.g.
    fineract AccountingConstantsTest.java → a step file, R057)."""
    out = git(repo, "log", "--follow", "--format=%x00%H %at", "--name-status", commit, "--", path)
    res = []
    for chunk in out.split("\x00")[1:]:
        lines = [l for l in chunk.splitlines() if l.strip()]
        sha, at = lines[0].split()
        status = lines[1].split("\t") if len(lines) > 1 else ["M", path]
        if status[0].startswith("D"):
            continue
        res.append((sha, int(at), status[-1]))
        if status[0][:1] == "C" or (status[0][:1] == "R" and int(status[0][1:] or 100) < MIN_RENAME_SCORE):
            break
    return res


def cat_batch(repo: Path, specs: list[str]) -> list[str]:
    """Contents of many "<commit>:<path>" blobs through one `git cat-file --batch` process. A missing object is an
    error, never a skipped version."""
    p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env=ENV)
    p.stdin.write("".join(s + "\n" for s in specs).encode())
    p.stdin.close()
    res = []
    for spec in specs:
        header = p.stdout.readline().decode()
        if not header.strip() or header.endswith(" missing\n") or header.endswith(" ambiguous\n"):
            p.kill()
            raise RuntimeError(f"git cat-file --batch {spec}: {header.strip() or 'no output'}")
        size = int(header.split()[2])
        res.append(p.stdout.read(size).decode("utf-8", errors="replace"))
        p.stdout.read(1)
    if p.wait() != 0:
        raise RuntimeError("git cat-file --batch failed")
    return res
