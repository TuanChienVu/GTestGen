import os
import subprocess
import sys
from pathlib import Path

def test_script_runs_without_pythonpath_set():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "scripts/wrong_suite_control.py", "--help"],
                            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
    assert result.returncode == 0
