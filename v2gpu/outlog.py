"""Append-only JSONL output of a generation job: one line per (record, run).

Each line is written whole (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n"), flushed and fsynced before the
next generation starts, so a job that is killed loses at most the line it was writing. Opening an existing file is a
resume:
- every complete line (ending in "\\n") must be a JSON object that the job's `check` accepts, and no (record_id, model,
  seed, condition, variant) may appear twice: otherwise OutputLogError (a job never appends to a file it cannot account
  for);
- a last line without its "\\n" is the line being written when the job died: it is cut off (the file is truncated to the
  last complete line) and its length is counted in `truncated_tail`;
- `done` maps every written key to its exact line, so a resumed job skips those pairs and never writes one twice;
- the file is locked (flock) for as long as it is open, before anything is read or cut: a second writer (the same job
  started twice, by hand and by the queue) is refused instead of interleaving lines.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Callable

from .runs import row_key


class OutputLogError(RuntimeError):
    pass


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class OutputLog:
    def __init__(self, path: str | Path, check: Callable[[dict], list[str]] | None = None):
        self.path = Path(path)
        self.done: dict[tuple, bytes] = {}
        self.truncated_tail = 0
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a+b")             # appends only; read and truncate through the same descriptor
        try:
            fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._f.close()
            raise OutputLogError(f"{self.path} is being written by another process") from None
        try:
            self._resume(check)
        except BaseException:
            self._f.close()
            raise
        if new:
            _fsync_dir(self.path.parent)

    def _resume(self, check) -> None:
        self._f.seek(0)
        data = self._f.read()
        complete, _, tail = data.rpartition(b"\n")
        if tail:                                    # the last line has no newline: torn by a kill mid-write
            self.truncated_tail = len(tail)
            self._f.truncate(len(data) - len(tail))
            self._f.flush()
            os.fsync(self._f.fileno())
        for n, line in enumerate(complete.split(b"\n") if complete else [], start=1):
            try:
                row = json.loads(line)
            except ValueError as e:
                raise OutputLogError(f"{self.path.name} line {n}: not JSON ({e})") from None
            problems = ["not a JSON object"] if not isinstance(row, dict) else (check(row) if check else [])
            if problems:
                raise OutputLogError(f"{self.path.name} line {n}: {'; '.join(problems[:3])}")
            key = row_key(row)
            if key in self.done:
                raise OutputLogError(f"{self.path.name} line {n}: {key} written twice")
            self.done[key] = line + b"\n"

    def __contains__(self, key) -> bool:
        return key in self.done

    def append(self, row: dict) -> bytes:
        key = row_key(row)
        if key in self.done:
            raise OutputLogError(f"{key} is already in {self.path.name}")
        line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        self._f.write(line)
        self._f.flush()
        os.fsync(self._f.fileno())
        self.done[key] = line
        return line

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
