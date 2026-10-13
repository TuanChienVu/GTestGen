"""Per-line author times from `git blame --line-porcelain`, and first-seen times of line texts from `git log -p`."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

_HEADER = re.compile(r"^[0-9a-f]{40} \d+ (\d+)")


def line_times(repo: Path, path: str) -> dict[int, int]:
    out = subprocess.run(["git", "blame", "--line-porcelain", "HEAD", "--", path], cwd=repo,
                         capture_output=True, text=True, check=True).stdout
    times: dict[int, int] = {}
    final_line, when = None, None
    for line in out.splitlines():
        header = _HEADER.match(line)
        if header:
            final_line = int(header.group(1))
        elif line.startswith("author-time "):
            when = int(line.split()[1])
        elif line.startswith("\t") and final_line is not None:
            times[final_line] = when
    return times


def min_time(times: dict[int, int], lines: Iterable[int]) -> int | None:
    values = [times[n] for n in lines if n in times]
    return min(values) if values else None


def first_seen(repo: Path, pathspecs: list[str]) -> dict[str, int]:
    """Earliest author time at which each stripped line text was added to any file matching `pathspecs`.

    Unlike blame this survives splitting, merging and moving content between files (blame dates such lines to the move)."""
    out = subprocess.run(["git", "log", "--no-renames", "--format=@@C %at", "-p", "--unified=0", "HEAD", "--", *pathspecs],
                         cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    seen: dict[str, int] = {}
    now = None
    for line in out.splitlines():
        if line.startswith("@@C "):
            now = int(line[4:])
        elif line.startswith("+") and not line.startswith("+++ ") and now is not None:
            text = line[1:].strip()
            if text:
                seen[text] = min(seen.get(text, now), now)
    return seen


def earliest(*times: int | None) -> int | None:
    values = [t for t in times if t is not None]
    return min(values) if values else None
