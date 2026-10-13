#!/usr/bin/env python3
"""A stat manifest of one or more directory trees, for the bootstrap's step-5 tamper check (measured on the real EU-RO-1
MooseFS on 28/09: a /workspace there keeps no mode bits, so chmod go-w
cannot stop the unprivileged test user from changing the repo, the venv or the HF cache that root runs later).

python3 tree_manifest.py [--skip PATH]... DIR... -> one line per entry (the root itself included), sorted:
    <root>/<relative path> TAB kind(f|d|l) TAB mode TAB size TAB mtime_ns TAB ctime_ns TAB link target
--skip PATH leaves out the one line of that exact entry (a root's own line, or a file root itself writes while the
test user runs), never its children.

Every write, chmod, chown, rename, create or delete changes an entry's ctime (or adds or removes a line), and an
unprivileged user cannot set a ctime back, so comparing two manifests catches any change between them without hashing
gigabytes of weights over a network volume. Symlinks are listed, never followed. Standard library only: root runs a
copy of this file with the system python3, outside the test user's reach."""
from __future__ import annotations

import os
import sys


def lines(root: str, skip: frozenset = frozenset()) -> list[str]:
    out = []

    def one(path: str) -> None:
        if os.path.normpath(path) in skip:
            return
        st = os.lstat(path)
        kind = "l" if os.path.islink(path) else "d" if os.path.isdir(path) else "f"
        target = os.readlink(path) if kind == "l" else ""
        rel = os.path.relpath(path, root)
        out.append(f"{root}/{rel}\t{kind}\t{st.st_mode:o}\t{st.st_size}\t{st.st_mtime_ns}\t{st.st_ctime_ns}\t{target}")

    one(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            one(os.path.join(dirpath, name))
    return out


def main(argv: list[str]) -> int:
    skip, roots = set(), []
    while argv:
        if argv[0] == "--skip" and len(argv) > 1:
            skip.add(os.path.normpath(os.path.abspath(argv[1])))
            argv = argv[2:]
        else:
            roots.append(argv[0])
            argv = argv[1:]
    if not roots or any(r.startswith("--") for r in roots):
        print("usage: tree_manifest.py [--skip PATH]... DIR...", file=sys.stderr)
        return 2
    for line in sorted(line for root in roots for line in lines(os.path.abspath(root), frozenset(skip))):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
