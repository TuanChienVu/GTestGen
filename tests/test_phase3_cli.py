import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scripts_run_without_pythonpath():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for name in ("fetch_full", "scan_prompts", "state_report", "build_prompts"):
        r = subprocess.run([sys.executable, f"scripts/{name}.py", "--help"], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)

