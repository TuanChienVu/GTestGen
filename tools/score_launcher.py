"""Run a Python script exactly as `python <script> <args>` would, with a higher recursion limit.

The scoring run used this launcher, with limit 20000, for every call of scripts/score_run.py and
scripts/score_independent.py; the scoring code itself is unchanged. One model output (deepseek, seed 99: a
`".*" + ".*" + ...` chain of about 1,500 operands) makes the pure-Python tree walk insert.first_class need about 1,530
frames, over Python's default limit of 1,000. This file sits outside v2tools/ and scripts/, so the scoring code's
code_sha256 (SCORES_META) and its clean-tree check do not change.

usage: <python> score_launcher.py <limit> </absolute/path/script.py> [script args...]

What differs from `python <script>`: only sys.getrecursionlimit(). sys.argv, sys.path (sys.path[0] is the script's
directory, links resolved), __file__ (absolute), __name__ == "__main__" and the exit status are those of a plain run.
"""
import os
import runpy
import sys

TESTED = (3, 12, 12)  # the interpreter of the run (.venv -> uv cpython-3.12.12); the limits were measured on it
if sys.implementation.name != "cpython" or sys.version_info[:3] != TESTED:
    sys.exit(f"STOP: score_launcher was tested on CPython {'.'.join(map(str, TESTED))} only; "
             f"this is {sys.implementation.name} {sys.version.split()[0]}")
if len(sys.argv) < 3 or not sys.argv[1].isdigit():
    sys.exit("usage: score_launcher.py <limit> <script.py> [args...]")
if not os.path.isabs(sys.argv[2]):  # runpy sets argv[0] to the path it runs: only an absolute path keeps argv[0] and
    sys.exit("STOP: give the script as an absolute path")  # __file__ both as a plain run sets them
limit = int(sys.argv[1])
if limit < 1000:
    sys.exit("STOP: limit below the default 1000")
sys.setrecursionlimit(limit)
sys.argv = sys.argv[2:]  # argv[0] is the script path as given, as with `python <script>`
sys.path[0] = os.path.dirname(os.path.realpath(sys.argv[0]))  # as with `python <script>`
runpy.run_path(sys.argv[0], run_name="__main__")
