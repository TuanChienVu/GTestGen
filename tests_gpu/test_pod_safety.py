"""No test may reach the pod's own /etc/rp_environment, a real runpodctl, or a
real ssh or scp.

On a RunPod pod, /etc/rp_environment is written by the image's /start.sh and exports the container PATH, the pod's
RUNPOD_POD_ID and its RUNPOD_API_KEY. Two tests used to source it through the sheets' watchdog script, which dropped
their fake runpodctl, so the pod's real `runpodctl stop pod` ran during the KIND=gpu bootstrap gate.

There are two guards:
- the conftest (_pod_traps) puts traps for runpodctl, ssh and scp first on PATH and points RP_ENV at a file that keeps
  them first; a test that reaches one of them fails;
- the static scans below."""
import ast
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests_gpu.conftest import TRAPPED, _snapshot

ROOT = Path(__file__).resolve().parents[1]
POD_ENV = "/etc/rp_environment"
OVERRIDE = "${RP_ENV:-/etc/rp_environment}"


def pod_env_uses(source: str) -> list[tuple[int, str]]:
    """(line, text) of every string in a test module that names the pod's env file, other than:
    - as the default of the RP_ENV override;
    - in a text assertion (inside an assert, not inside a call);
    - in a docstring."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if POD_ENV not in node.value.replace(OVERRIDE, ""):
            continue                                           # absent, or only the override's default
        parent = parents.get(node)
        if isinstance(parent, ast.Expr) and isinstance(parents.get(parent), (ast.Module, ast.FunctionDef,
                                                                              ast.ClassDef)):
            continue                                           # a docstring
        up, text_check = node, False
        while up in parents:
            up = parents[up]
            if isinstance(up, ast.Call):
                break                                          # passed to something: it may be executed
            if isinstance(up, ast.Assert):
                text_check = True
                break
        if not text_check:
            out.append((node.lineno, node.value[:120]))
    return out


def test_the_scan_flags_an_executed_pod_env_and_accepts_the_override_a_text_check_and_a_docstring():
    bad = textwrap.dedent('''
        import subprocess
        SCRIPT = "sleep 1; source /etc/rp_environment; runpodctl stop pod x"
        subprocess.run(["bash", "-c", "source /etc/rp_environment; runpodctl stop pod x"])
        assert subprocess.run(["bash", "-c", "source /etc/rp_environment"]).returncode == 0
        ''')
    assert [line for line, _ in pod_env_uses(bad)] == [3, 4, 5]
    good = textwrap.dedent('''
        """A docstring may name /etc/rp_environment."""
        SCRIPT = 'RP_ENV=${RP_ENV:-/etc/rp_environment}; if [ -f "$RP_ENV" ]; then source "$RP_ENV"; fi'
        def test_x(lines):
            """Nor does this one: source /etc/rp_environment."""
            assert 'source /etc/rp_environment; runpodctl stop pod' in lines[2]
        ''')
    assert pod_env_uses(good) == []


def test_no_test_names_the_pods_env_file_outside_the_override():
    """Static scan: every module under tests_gpu but this one (its samples are parsed, never run; the only command it
    runs sources "$RP_ENV", the trap's file)."""
    found = {p.name: pod_env_uses(p.read_text()) for p in sorted((ROOT / "tests_gpu").glob("*.py"))
             if p.name != Path(__file__).name}
    assert len(found) >= 25 and "test_runsheets.py" in found and "conftest.py" in found
    assert {name: uses for name, uses in found.items() if uses} == {}


def test_the_sheets_watchdog_reads_its_env_file_through_rp_env():
    """Every sheet's on-pod watchdog, which the tests run here, sources "$RP_ENV" (default /etc/rp_environment), only
    if it exists. TRAIN_RUNSHEET.md is added now that its §4 writes the watchdog out in full
    (before that, the cross-sheet safety scans skipped TRAIN, because there were no lines to scan)."""
    for sheet in ("CANARY_RUNSHEET.md", "GEN_RUNSHEET.md", "TRAIN_RUNSHEET.md"):
        text = (ROOT / "ops" / sheet).read_text()
        line = next(l for l in text.splitlines() if l.startswith("setsid nohup bash -c ") and "V2GPU_WATCHDOG" in l)
        assert f'RP_ENV={OVERRIDE}; if [ -f "$RP_ENV" ]; then source "$RP_ENV"; fi;' in line, sheet
        assert f"source {POD_ENV}" not in line, sheet


def test_the_ops_scripts_name_the_pods_env_file_only_in_a_command_sent_over_ssh():
    """Of the ops scripts the tests run, only watch_pod.sh reads the file: its STOP, which only ever goes through
    "${SSH[@]}" (a fake in the tests; ssh is trapped anyway); the bootstrap only makes it root-only before the test step
    (a dry run in the tests prints it). Since 27/09 the bootstrap names the path only
    as RP_ENV_FILE's default and in the refusal of a moved path without V2GPU_BOOTSTRAP_TEST=1; its chmod lines use
    "$RP_ENV_FILE" (tests_gpu/test_ops_scripts.py pins them)."""
    uses = {s.name: [l.strip() for l in s.read_text().splitlines() if POD_ENV in l and not l.lstrip().startswith("#")]
            for s in sorted((ROOT / "ops").glob("*.sh"))}
    # pod_parity_gate.sh (the laptop gate) also plants the real path, but INSIDE its
    # throwaway container, never on this machine.
    assert {k: v for k, v in uses.items() if v} == {
        "watch_pod.sh": ["STOP='source /etc/rp_environment; runpodctl stop pod \"$RUNPOD_POD_ID\"'"],
        "pod_bootstrap.sh": [
            "RP_ENV_FILE=${RP_ENV_FILE:-/etc/rp_environment}",
            'if [ "$RP_ENV_FILE" != /etc/rp_environment ] || [ "$V2TEST_HOME" != /home/v2test ] || '
            '[ "$V2GPU_LOCAL" != /opt/v2 ]; then'],
        "pod_parity_gate.sh": [
            "printf 'export RUNPOD_API_KEY=FAKE-KEY\\nexport RUNPOD_POD_ID=fake-pod-$KIND\\n' > /etc/rp_environment",
            "chmod 644 /etc/rp_environment", "chown root:root /etc/rp_environment"]}
    wp = (ROOT / "ops" / "watch_pod.sh").read_text()
    stops = [l for l in wp.splitlines() if '"$STOP"' in l or "$STOP" in l.replace("$STOP_", "")]
    assert stops and all('"${SSH[@]}" "$STOP"' in l or l.lstrip().startswith("echo ") for l in stops), stops


def test_every_test_sees_the_traps_first():
    """runpodctl, ssh, scp, sftp, rsync, curl and wget resolve to the traps, and sourcing RP_ENV
    keeps them first (as the pod's own file would not)."""
    assert set(TRAPPED) == {"runpodctl", "ssh", "scp", "sftp", "rsync", "curl", "wget"}
    for tool in TRAPPED:
        assert "TRAP" in Path(shutil.which(tool)).read_text(), tool
    env_file = Path(os.environ["RP_ENV"])
    assert env_file.is_file() and str(env_file) != POD_ENV
    found = subprocess.run(["bash", "-c", 'PATH=/usr/bin:/bin; source "$RP_ENV"; command -v runpodctl'],
                           capture_output=True, text=True).stdout.strip()
    assert found == shutil.which("runpodctl")


@pytest.mark.parametrize("tool", TRAPPED)
def test_a_test_that_reaches_a_real_pod_command_fails(tmp_path, tool):
    """The trap: a test that runs one of these commands without a fake of its own fails, loudly. The nested run's PATH
    holds no real command: a broken trap fails this test, never reaches a real one."""
    probe = tmp_path / "test_reach.py"
    probe.write_text(f"import subprocess\n\ndef test_reach():\n    subprocess.run([{tool!r}, 'stop', 'pod', 'x'])\n")
    (tmp_path / "no-commands").mkdir()
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tests_gpu.conftest",
                          "--rootdir", str(tmp_path), str(probe)], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PATH": str(tmp_path / "no-commands")})
    assert run.returncode != 0 and re.search(rf"reached a real pod command: {tool} stop pod x", run.stdout), \
        run.stdout[-2000:]


# No test reads or passes on RUNPOD_API_KEY or any RUNPOD_* secret

PLANTED = {"RUNPOD_API_KEY": "planted-secret-key", "RUNPOD_POD_ID": "planted-secret-pod",
           "RUNPOD_SECRET_hf_token": "planted-secret-hf"}
BASH_PROBE = """if [ "${V2GPU_PROBE_ARMED:-}" = 1 ] && env | grep -q planted-secret; then
  echo "bash $0 $*" >> "$V2GPU_LEAK_LOG"
fi
"""
PY_PROBE = """import os, sys
if os.environ.get("V2GPU_PROBE_ARMED") == "1" and any("planted-secret" in v for v in os.environ.values()):
    with open(os.environ["V2GPU_LEAK_LOG"], "a") as f:
        f.write("python " + " ".join(sys.argv)[:200] + "\\n")
"""
ARM = "import os\n\n\ndef pytest_configure(config):\n    os.environ['V2GPU_PROBE_ARMED'] = '1'\n"
CONTROL = """import os, subprocess, sys


def test_control_leaks_on_purpose():
    env = {**os.environ, "RUNPOD_API_KEY": "planted-secret-control"}
    subprocess.run(["bash", "-c", "true"], env=env, check=True)
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)
"""
# every test that runs an ops script, a sheet's snippet, a watcher, the watchdog or a queue's self-stop
OPS_TESTS = ["tests_gpu/test_ops_scripts.py", "tests_gpu/test_runsheets.py", "tests_gpu/test_jobqueue.py",
             "tests_gpu/test_bootstrap_locks.py", "tests_gpu/test_genqueue.py::test_bootstrap_installs_the_pin_set_chosen_at_the_canary",
             "tests_gpu/test_genqueue.py::test_bootstrap_builds_the_generation_venv_on_request",
             "tests_gpu/test_genqueue.py::test_the_bootstrap_runs_the_suite_without_its_own_selectors",
             "tests_gpu/test_genqueue.py::test_the_training_bootstrap_leaves_out_the_nested_generation_simulation",
             "tests_gpu/test_genqueue.py::test_ship_copies_every_target_and_the_queue_then_checks_every_sha256_on_the_pod",
             "tests_gpu/test_pod_safety.py::test_no_runpod_variable_reaches_a_test"]


def test_no_runpod_variable_reaches_a_test():
    """The conftest removed every RUNPOD_* variable of the process before the first test."""
    assert not [k for k in os.environ if k.startswith("RUNPOD_")]


# RUNPOD_API_KEY's old exemption was a blanket regex, so it passed in ANY test
# module -- probe: os.environ.get("RUNPOD_API_KEY") in an unrelated module was flagged by one version and silently
# passed by a later one. Only one site (run_v2test_gate_fixture, whose value is always a parameter defaulting to
# "") legitimately needs the literal name; whitelist that one (file, function) pair instead.
RUNPOD_API_KEY_SITES = {("test_ops_scripts.py", "run_v2test_gate_fixture")}


def runpod_secret_problems(root: Path, exempt_file: str | None = None) -> dict:
    """Static scan: no string in a *.py module under root (other than exempt_file) names a RUNPOD_* variable outside
    RUNPOD_POD_ID (every test's own fake) and RUNPOD_API_KEY_SITES' one whitelisted (file, function) pair; docstrings
    aside."""
    found = {}
    for p in sorted(root.glob("*.py")):
        if exempt_file and p.name == exempt_file:
            continue
        tree = ast.parse(p.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        enclosing_func = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(fn):
                    enclosing_func.setdefault(child, fn.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and \
                    re.search(r"RUNPOD_(?!POD_ID\b)\w+", node.value) and not (
                    isinstance(parents.get(node), ast.Expr) and isinstance(
                        parents.get(parents.get(node)), (ast.Module, ast.FunctionDef, ast.ClassDef))):
                if node.value == "RUNPOD_API_KEY" and (p.name, enclosing_func.get(node)) in RUNPOD_API_KEY_SITES:
                    continue
                found.setdefault(p.name, []).append((node.lineno, node.value[:80]))
    return found


def test_no_test_names_a_runpod_secret():
    """Static scan: outside this module, no string in a test names a RUNPOD_* variable other than RUNPOD_POD_ID
    (every test's own fake) and the one whitelisted RUNPOD_API_KEY_SITES call site
    (since 27/09/2026: pinning that env -i strips a real RUNPOD_API_KEY needs the name itself, always paired with a
    fake value); docstrings aside."""
    assert runpod_secret_problems(ROOT / "tests_gpu", exempt_file=Path(__file__).name) == {}


def test_a_runpod_api_key_sighting_outside_the_whitelisted_site_is_flagged(tmp_path):
    """The exemption is scoped to RUNPOD_API_KEY_SITES' one (file, function)
    pair, not to every test module -- planted probe: os.environ.get("RUNPOD_API_KEY") in an unrelated module must
    still be flagged."""
    (tmp_path / "test_other.py").write_text('import os\n\n\ndef test_x():\n    os.environ.get("RUNPOD_API_KEY")\n')
    problems = runpod_secret_problems(tmp_path)
    assert "test_other.py" in problems and any("RUNPOD_API_KEY" in v for _, v in problems["test_other.py"])


def test_no_subprocess_of_the_ops_script_tests_sees_a_runpod_secret(tmp_path):
    """Pin: the ops-script tests run with planted RUNPOD_API_KEY, RUNPOD_POD_ID and RUNPOD_SECRET_* values. Every bash
    (BASH_ENV) and every Python (sitecustomize) they start records itself if it sees a planted value. Only the control
    test, which passes a planted key on purpose, may be recorded."""
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "bash_probe.sh").write_text(BASH_PROBE)
    (probe / "sitecustomize.py").write_text(PY_PROBE)
    (probe / "probe_arm.py").write_text(ARM)
    (tmp_path / "control" ).mkdir()
    (tmp_path / "control" / "test_control.py").write_text(CONTROL)
    log = tmp_path / "leaks.log"
    log.write_text("")
    outer = ("KIND", "GEN_SET", "FIRST_RUN", "V2GPU_PROBE_ARMED", "BASH_ENV", "PYTEST_ADDOPTS", "V2GPU_LEAK_LOG")
    env = {**{k: v for k, v in os.environ.items() if k not in outer}, **PLANTED,        # never an outer probe's arming
           "BASH_ENV": str(probe / "bash_probe.sh"), "V2GPU_LEAK_LOG": str(log),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(probe), str(ROOT), os.environ.get("PYTHONPATH")]))}
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "probe_arm",
                          "--basetemp", str(tmp_path / "bt"), *OPS_TESTS, str(tmp_path / "control" / "test_control.py")],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout[-3000:]
    leaks = log.read_text().splitlines()
    assert len(leaks) == 2 and leaks[0].startswith("bash ") and leaks[1].startswith("python "), leaks[:10]


# The watchdog, the self-stop and every read of the env file stay with root


def test_the_watchdog_the_queues_and_the_env_file_are_never_run_as_the_test_user():
    """Every sheet command that reaches the pod goes through an ssh (or scp) array to root@$IP; no sheet line switches
    user; the watchdog is launched over that root ssh; only the bootstrap's test step runs as v2test.
    TRAIN_RUNSHEET.md is added now that its §4/§6 write the watchdog and the queue launch out in full."""
    for sheet in ("CANARY_RUNSHEET.md", "GEN_RUNSHEET.md", "TRAIN_RUNSHEET.md"):
        text = (ROOT / "ops" / sheet).read_text()
        code = "\n".join(re.findall(r"```bash\n(.*?)```", text, re.S))
        assert not re.search(r"\b(runuser|setpriv|sudo|v2test)\b|\bsu\s", code), sheet
        arrays = re.findall(r"\bSSH=\((.*?)\)", code, re.S)
        assert arrays and all(a.rstrip().endswith('"root@$IP"') for a in arrays), (sheet, arrays)
        launch = next(l for l in code.splitlines() if "bash -s --" in l)
        assert launch.startswith('"${SSH[@]}" bash -s -- "$REM" <<\'EOF\''), (sheet, launch)
        for line in code.splitlines():
            if "V2GPU_WATCHDOG" in line or "v2gpu.jobqueue" in line or "source /etc/rp_environment" in line:
                assert "v2test" not in line and "runuser" not in line, (sheet, line)


# The v2test gate: find_spec("torch") can lie under a permission gap for
# v2test, and tests_gpu/conftest.py used to skip the whole directory silently in that case, keeping the bootstrap's
# gate green. V2GPU_REQUIRE_TORCH=1 (set by the bootstrap's test step for KIND=gpu and KIND=gen) must instead fail
# collection loudly.


def sim_dir(tmp_path: Path, fake_missing_torch: bool) -> Path:
    """An isolated copy of tests_gpu/conftest.py plus one trivial test, with sitecustomize.py making
    importlib.util.find_spec("torch") return None when fake_missing_torch (simulating the permission gap), so the
    test never depends on whether this venv actually has torch."""
    d = tmp_path / "sim"
    d.mkdir()
    shutil.copy2(ROOT / "tests_gpu" / "conftest.py", d / "conftest.py")
    (d / "test_trivial.py").write_text("def test_x():\n    assert 1 == 1\n")
    site = tmp_path / "site"
    site.mkdir()
    if fake_missing_torch:
        (site / "sitecustomize.py").write_text(
            "import importlib.util as u\n"
            "_real = u.find_spec\n"
            "def find_spec(name, *a, **k):\n"
            "    return None if name == 'torch' else _real(name, *a, **k)\n"
            "u.find_spec = find_spec\n")
    return d


def run_sim(d: Path, site: Path, **extra_env: str) -> subprocess.CompletedProcess:
    # Never the V2GPU_REQUIRE_TORCH of the process running this suite (on a pod the bootstrap's
    # test step sets it to 1 for the whole suite, which turned the "unset" case into a set one); only extra_env sets it
    env = {**{k: v for k, v in os.environ.items() if k != "V2GPU_REQUIRE_TORCH"},
           "PYTHONPATH": os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH")])), **extra_env}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(d)],
                          capture_output=True, text=True, env=env)


def test_v2gpu_require_torch_fails_collection_loudly_instead_of_skipping(tmp_path):
    """With V2GPU_REQUIRE_TORCH=1 and torch not importable (the permission-gap simulation), conftest must refuse to
    load (a real error, not the graceful 'no tests collected' of a plain skip)."""
    d = sim_dir(tmp_path, fake_missing_torch=True)
    run = run_sim(d, tmp_path / "site", V2GPU_REQUIRE_TORCH="1")
    assert run.returncode not in (0, 5), (run.stdout, run.stderr)
    assert "V2GPU_REQUIRE_TORCH" in run.stdout + run.stderr


def test_v2gpu_require_torch_unset_keeps_the_old_graceful_skip(tmp_path, monkeypatch):
    """Without V2GPU_REQUIRE_TORCH (the laptop's plain .venv, which has no torch on purpose), the directory is still
    skipped gracefully: the gate must not turn the laptop's own harmless case into a failure. The
    suite itself runs with V2GPU_REQUIRE_TORCH=1 on a pod (the bootstrap's test step), as set here; run_sim drops it
    from the nested run's environment, so "unset" is unset there too."""
    monkeypatch.setenv("V2GPU_REQUIRE_TORCH", "1")
    d = sim_dir(tmp_path, fake_missing_torch=True)
    run = run_sim(d, tmp_path / "site")
    assert run.returncode == 5, (run.stdout, run.stderr)                 # pytest: no tests were collected
    assert "V2GPU_REQUIRE_TORCH" not in run.stdout + run.stderr


def test_v2gpu_require_torch_set_but_torch_present_runs_normally(tmp_path):
    """When torch really is importable (this venv), V2GPU_REQUIRE_TORCH=1 changes nothing: the directory collects
    and runs as usual."""
    d = sim_dir(tmp_path, fake_missing_torch=False)
    run = run_sim(d, tmp_path / "site", V2GPU_REQUIRE_TORCH="1")
    assert run.returncode == 0 and "1 passed" in run.stdout, (run.stdout, run.stderr)


# decision 27/09: a test that runs the REAL bootstrap (not DRY_RUN) used to chmod/chown the
# machine's /etc/rp_environment and rm -rf its /home/v2test (on a pod as v2test: EPERM, a red gate; as root: the pod's
# own file and home). Such a run must move both into its tmp_path (RP_ENV_FILE, V2TEST_HOME); a static scan checks it.

BOOTSTRAP = "pod_bootstrap.sh"
KNOBS = ("RP_ENV_FILE", "V2TEST_HOME", "V2GPU_LOCAL")          # the container-disk venv root too
REAL_PATHS = ("/etc/rp_environment", "/home/v2test", "/opt/v2")
MUTATORS = ("update", "pop", "popitem", "setdefault", "clear", "__setitem__", "__delitem__", "__ior__")


def _is_runner(call: ast.Call, modules: dict) -> bool:
    """subprocess.* (any), os.system/popen/exec*/spawn*/posix_spawn*, pty.spawn, asyncio.create_subprocess_*, by the
    module's own import names (`import subprocess as sp` too)."""
    f = call.func
    if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
        return False
    mod = modules.get(f.value.id)
    return mod == "subprocess" or (mod == "os" and (f.attr in ("system", "popen") or f.attr.startswith(
        ("exec", "spawn", "posix_spawn")))) or (mod == "pty" and f.attr == "spawn") or (
        mod == "asyncio" and f.attr.startswith("create_subprocess"))


def _body_nodes(scope):
    """The nodes of a scope, without those of the functions, lambdas and classes nested in it."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(node))


def _bindings(name: str, scopes: list):
    """(values, unreadable): the values assigned to name in the innermost scope that binds it; unreadable when it is
    also bound otherwise (a parameter, a loop or with target, an import, +=) or mutated (x[k] = v, x.update(...))."""
    for scope in scopes:
        values, other = [], False
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = scope.args
            params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *filter(None, [a.vararg, a.kwarg])]
            other = any(p.arg == name for p in params)
        for node in _body_nodes(scope):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
                    values.append(node.value)
                elif any(isinstance(n, ast.Name) and n.id == name for t in targets for n in ast.walk(t)):
                    other = True                                   # a[...] = v, a.x = v, tuple unpacking
            elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Delete)):
                targets = [node.target] if isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor)) else (
                    node.targets if isinstance(node, ast.Delete) else [i.optional_vars for i in node.items])
                other |= any(isinstance(n, ast.Name) and n.id == name for t in targets if t for n in ast.walk(t))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                other |= any((a.asname or a.name.split(".")[0]) == name for a in node.names)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                other |= name in node.names
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    isinstance(node.func.value, ast.Name) and node.func.value.id == name and node.func.attr in MUTATORS:
                other = True
        if values or other:
            return values, other
    return [], False


PATH_CALLS = ("str", "fspath", "Path", "PurePath", "PosixPath", "join", "format", "resolve", "absolute", "as_posix",
              "read_text", "read_bytes")


def _names_bootstrap(node, scopes, funcs, seen=frozenset()) -> bool:
    """Whether an expression can name the bootstrap (its path, or its text): a string holding pod_bootstrap.sh,
    lockfree_checkout(...) (the bootstrap of a lock-free copy), or a list, f-string, path join, str()/Path()/read_text()
    of one, a name bound to one, or a call of a function of this module that returns one. A value taken out of its
    text (re.search(...).group(1), lines of a dry run) is not the bootstrap."""
    def again(n):
        return _names_bootstrap(n, scopes, funcs, seen)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and BOOTSTRAP in node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.JoinedStr)):
        return any(again(e) for e in node.elts if e is not None) if not isinstance(node, ast.JoinedStr) else any(
            again(v) for v in node.values)
    if isinstance(node, (ast.Starred, ast.FormattedValue)):
        return again(node.value)
    if isinstance(node, ast.BinOp):
        return again(node.left) or again(node.right)
    if isinstance(node, ast.IfExp):
        return again(node.body) or again(node.orelse)
    if isinstance(node, ast.Name):
        return node.id not in seen and any(_names_bootstrap(v, scopes, funcs, seen | {node.id})
                                           for v in _bindings(node.id, scopes)[0])
    if isinstance(node, ast.Call):
        callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if callee == "lockfree_checkout":
            return True
        if callee in PATH_CALLS:
            return any(again(a) for a in [*node.args, *(k.value for k in node.keywords)]) or (
                isinstance(node.func, ast.Attribute) and again(node.func.value))
        if callee in funcs and callee not in seen and isinstance(node.func, ast.Name):
            return any(_names_bootstrap(r.value, [funcs[callee], scopes[-1]], funcs, seen | {callee})
                       for r in _body_nodes(funcs[callee]) if isinstance(r, ast.Return) and r.value is not None)
    return False


def _env_keys(node, scopes, funcs, seen=frozenset()):
    """The keys a mapping expression surely holds, each with its value node (a later key wins over an earlier one); an
    unreadable part (os.environ, a parameter, a call) may hold any key, so it cancels every key before it. None: the
    mapping itself cannot be read."""
    if isinstance(node, ast.Dict):
        keys = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                inner = _env_keys(v, scopes, funcs, seen)
                keys = {**keys, **inner} if inner is not None else {}
            elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys[k.value] = v
            else:
                keys = {}
        return keys
    if isinstance(node, ast.Name) and node.id not in seen:
        values, other = _bindings(node.id, scopes)
        return _env_keys(values[0], scopes, funcs, seen | {node.id}) if len(values) == 1 and not other else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs and \
            node.func.id not in seen:
        f = funcs[node.func.id]
        found = [_env_keys(r.value, [f, scopes[-1]], funcs, seen | {node.func.id}) for r in _body_nodes(f)
                 if isinstance(r, ast.Return) and r.value is not None]
        if found and all(k is not None for k in found):
            return {k: v for k, v in found[0].items() if all(k in other for other in found[1:])}
    return None


def _bootstrap_calls(source: str):
    """(call, scopes, funcs) of every call in a module that runs ops/pod_bootstrap.sh (a subprocess/os/pty/asyncio call
    whose command can name it); scopes run from the innermost function out to the module."""
    tree = ast.parse(source)
    modules = {a.asname or a.name: a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and _is_runner(call, modules)):
            continue
        scopes, up = [], call
        while up in parents:
            up = parents[up]
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                scopes.append(up)
        command = call.args[:1] + [k.value for k in call.keywords if k.arg == "args"]
        if any(_names_bootstrap(c, scopes, funcs) for c in command):
            yield call, scopes, funcs


def _scope_name(scopes) -> str:
    return scopes[0].name if scopes and not isinstance(scopes[0], ast.Module) else "<module>"


def bootstrap_runs(source: str) -> list[tuple[str, int, str]]:
    """(function, line, problem) of every call in a module that runs ops/pod_bootstrap.sh (a subprocess/os/pty/asyncio
    call whose command can name it); problem is "" when its env is sure to hold DRY_RUN=1, or both RP_ENV_FILE and
    V2TEST_HOME set to something other than the real paths; otherwise it says why. Unreadable is a problem."""
    out = []
    for call, scopes, funcs in _bootstrap_calls(source):
        env = [k.value for k in call.keywords if k.arg == "env"]
        keys = _env_keys(env[0], scopes, funcs) if env else None
        dry = keys is not None and isinstance(keys.get("DRY_RUN"), ast.Constant) and keys["DRY_RUN"].value == "1"
        moved = keys is not None and all(k in keys and not (isinstance(keys[k], ast.Constant) and
                                                            keys[k].value in REAL_PATHS) for k in KNOBS)
        problem = "" if dry or moved else ("no env: it inherits this process's environment" if not env else
                                           "env unreadable" if keys is None else
                                           "a real run without both RP_ENV_FILE and V2TEST_HOME moved")
        out.append((_scope_name(scopes), call.lineno, problem))
    return out


def test_the_bootstrap_run_scan_flags_a_real_run_on_the_machines_own_paths():
    bad = textwrap.dedent('''
        import os
        import subprocess
        import subprocess as sp
        def a(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"])
        def b(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={**os.environ, "WS": str(tmp_path)})
        def c(tmp_path):
            env = {"DRY_RUN": "1"}
            env["DRY_RUN"] = "0"
            sp.run(["bash", str(tmp_path / "ops" / "pod_bootstrap.sh")], env=env)
        def d(tmp_path, extra):
            subprocess.Popen(["bash", str(lockfree_checkout(tmp_path))], env={"DRY_RUN": "1", **extra})
        def e(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={"RP_ENV_FILE": str(tmp_path / "x")})
        def f(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={"RP_ENV_FILE": "x", "V2TEST_HOME": "/home/v2test"})
        def g(tmp_path):
            script = tmp_path / "ops" / "pod_bootstrap.sh"
            os.system(f"bash {script}")
        def h(tmp_path):
            env = {"DRY_RUN": "1"}
            env.update(DRY_RUN="0")
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env=env)
        ''')
    assert [(name, bool(problem)) for name, _, problem in bootstrap_runs(bad)] == [
        (n, True) for n in "abcdefgh"], bootstrap_runs(bad)
    good = textwrap.dedent('''
        import os
        import subprocess
        def knobs(tmp_path):
            return {"RP_ENV_FILE": str(tmp_path / "e"), "V2TEST_HOME": str(tmp_path / "h"),
                    "V2GPU_LOCAL": str(tmp_path / "l"), "V2GPU_BOOTSTRAP_TEST": "1"}
        def a(tmp_path, extra):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={**extra, "DRY_RUN": "1"})
        def b(tmp_path):
            env = {**os.environ, "RP_ENV_FILE": str(tmp_path / "e"), "V2TEST_HOME": str(tmp_path / "h"),
                   "V2GPU_LOCAL": str(tmp_path / "l")}
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={**env, "KIND": "gen"})
        def c(tmp_path):
            subprocess.run(["bash", str(tmp_path / "ops" / "pod_bootstrap.sh")], env={**os.environ, **knobs(tmp_path)})
        def d(tmp_path):
            subprocess.run(["bash", "-c", "true"])
            (tmp_path / "pod_bootstrap.sh").read_text()
        ''')
    assert [(name, problem) for name, _, problem in bootstrap_runs(good)] == [("a", ""), ("b", ""), ("c", "")]


def test_no_test_runs_the_real_bootstrap_on_the_machines_env_file_or_v2test_home():
    """Static scan of every test module (tests_gpu, tests): a call that runs ops/pod_bootstrap.sh either is a dry run
    (DRY_RUN=1, last in its env) or moves both the env file and the v2test HOME (into its tmp_path; the bootstrap then
    also needs V2GPU_BOOTSTRAP_TEST=1, or it refuses)."""
    runs = {p.relative_to(ROOT).as_posix(): bootstrap_runs(p.read_text())
            for p in sorted([*(ROOT / "tests_gpu").glob("*.py"), *(ROOT / "tests").rglob("*.py")])}
    found = {(name, fn) for name, calls in runs.items() for fn, _, _ in calls}
    assert {("tests_gpu/lockfree.py", "dry_run"), ("tests_gpu/test_bootstrap_locks.py",
            "test_the_bootstrap_tests_pass_with_and_without_committed_locks"),
            ("tests_gpu/test_ops_scripts.py", "run_v2test_gate_fixture"),
            ("tests_gpu/test_ops_scripts.py", "test_a_broken_user_switch_stops_the_bootstrap_at_once_with_exit_2"),
            ("tests_gpu/test_ops_scripts.py",
             "test_the_bootstrap_stops_with_exit_2_when_the_first_run_lock_has_no_sha256")} <= found, sorted(found)
    assert {name: [c for c in calls if c[2]] for name, calls in runs.items() if any(c[2] for c in calls)} == {}


# /opt/v2 EXISTS where this suite also runs -- the
# gate container and a pod's own step 5, whose venv the suite itself runs from -- and a bootstrap that finds a venv there
# rightly skips building one. test_bootstrap_installs_uv_when_the_image_lacks_it kept the default venv root in a dry run
# and expected `uv venv`: green on the laptop, red in the gate (and it would have been red on the final smoke's step 5).
# Every bootstrap run of a test, dry or real, now moves the venv root into its tmp_path; only the tests that check the
# default itself keep it, each by name.

HERMETIC = ("hermetic_local", "bootstrap_test_knobs")   # both move V2GPU_LOCAL into tmp_path (lockfree, test_ops_scripts)
DEFAULT_ROOT_TESTS = {          # dry runs that check the default /opt/v2 itself (dry: nothing is written there)
    ("tests_gpu/test_ops_scripts.py",
     "test_the_test_step_runs_as_an_unprivileged_user_after_the_env_file_is_made_root_only"),
    ("tests_gpu/test_ops_scripts.py", "test_a_dry_run_puts_every_venv_under_opt_v2"),
    ("tests_gpu/test_ops_scripts.py", "test_moving_the_env_file_or_the_v2test_home_needs_the_test_flag"),
    ("tests_gpu/test_genqueue.py", "test_bootstrap_builds_the_generation_venv_on_request"),
    ("tests_gpu/test_ops_scripts.py", "test_the_dry_run_helpers_keep_the_venv_root_in_tmp_path_unless_told"),
}


def _moves_the_venv_root(node, scopes, funcs, seen=frozenset()) -> bool:
    """The env mapping surely moves V2GPU_LOCAL off the real /opt/v2: a literal V2GPU_LOCAL with any other value, or a
    spread of hermetic_local(...) or bootstrap_test_knobs(...), not undone by a later key or unreadable spread."""
    if isinstance(node, ast.Name) and node.id not in seen:
        values, other = _bindings(node.id, scopes)
        return len(values) == 1 and not other and _moves_the_venv_root(values[0], scopes, funcs, seen | {node.id})
    if not isinstance(node, ast.Dict):
        return False
    moved = False
    for k, v in zip(node.keys, node.values):
        if k is None:
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in HERMETIC:
                moved = True
            elif _moves_the_venv_root(v, scopes, funcs, seen):
                moved = True
            else:
                keys = _env_keys(v, scopes, funcs, seen)
                if keys is None:                                # unreadable: may put the default back
                    moved = False
                elif "V2GPU_LOCAL" in keys:
                    moved = not (isinstance(keys["V2GPU_LOCAL"], ast.Constant) and
                                 keys["V2GPU_LOCAL"].value in REAL_PATHS)
        elif isinstance(k, ast.Constant) and k.value == "V2GPU_LOCAL":
            moved = not (isinstance(v, ast.Constant) and v.value in REAL_PATHS)
    return moved


def default_root_runs(source: str) -> list[tuple[str, int]]:
    """(function, line) of every call that runs ops/pod_bootstrap.sh without surely moving V2GPU_LOCAL off /opt/v2."""
    return [(_scope_name(scopes), call.lineno) for call, scopes, funcs in _bootstrap_calls(source)
            if not any(_moves_the_venv_root(k.value, scopes, funcs) for k in call.keywords if k.arg == "env")]


def test_the_default_root_scan_flags_a_run_that_keeps_opt_v2():
    bad = textwrap.dedent('''
        import os
        import subprocess
        def a(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={"DRY_RUN": "1", "WS": str(tmp_path)})
        def b(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={"V2GPU_LOCAL": "/opt/v2", "DRY_RUN": "1"})
        def c(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={**hermetic_local(tmp_path), **os.environ})
        def d(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={**hermetic_local(tmp_path), "DRY_RUN": "1"})
        def e(tmp_path):
            env = {**bootstrap_test_knobs(tmp_path), "KIND": "gpu"}
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={**env, "FIRST_RUN": "1"})
        def f(tmp_path):
            subprocess.run(["bash", "ops/pod_bootstrap.sh"], env={"V2GPU_LOCAL": str(tmp_path / "l"), "DRY_RUN": "1"})
        ''')
    assert [name for name, _ in default_root_runs(bad)] == ["a", "b", "c"]


def test_every_bootstrap_run_of_a_test_moves_the_venv_root_off_the_real_one():
    """Every test module (tests_gpu, tests): a call that runs ops/pod_bootstrap.sh moves V2GPU_LOCAL into its tmp_path.
    The dry-run helpers of tests_gpu/lockfree.py do so themselves unless told otherwise (their own test, and the next
    one, pin that)."""
    runs = {p.relative_to(ROOT).as_posix(): default_root_runs(p.read_text())
            for p in sorted([*(ROOT / "tests_gpu").glob("*.py"), *(ROOT / "tests").rglob("*.py")])}
    assert {name: calls for name, calls in runs.items() if calls and name != "tests_gpu/lockfree.py"} == {}
    assert [f for f, _ in runs["tests_gpu/lockfree.py"]] == ["dry_run", "dry_run_no_gpu_path"]


def test_only_the_tests_that_check_the_default_venv_root_keep_it():
    """dry_run(..., default_local=True) keeps /opt/v2: only in DEFAULT_ROOT_TESTS, and none of them asserts on what a
    bootstrap does only when the venv is missing (uv venv, the container-disk check), which depends on the machine."""
    found, sources = set(), {}
    for p in sorted([*(ROOT / "tests_gpu").glob("*.py"), *(ROOT / "tests").rglob("*.py")]):
        tree = ast.parse(p.read_text())
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
                        n.func.id in ("dry_run", "dry_run_no_gpu_path") and \
                        any(k.arg == "default_local" and not (isinstance(k.value, ast.Constant) and
                                                               k.value.value is False) for k in n.keywords):
                    found.add((p.relative_to(ROOT).as_posix(), fn.name))
                    sources[(p.relative_to(ROOT).as_posix(), fn.name)] = ast.get_source_segment(p.read_text(), fn)
    assert found == DEFAULT_ROOT_TESTS
    for key, src in sources.items():
        assert "uv venv" not in src and "local_disk_check" not in src and "container disk" not in src, key


# RP_ENV_FILE, V2TEST_HOME and V2GPU_BOOTSTRAP_TEST are the tests' own; a pod never sets them

TEST_ONLY = ("RP_ENV_FILE", "V2TEST_HOME", "V2GPU_LOCAL", "V2GPU_BOOTSTRAP_TEST")
BOOTSTRAP_SETS = [          # the bootstrap's only lines that match NAME=: its three defaults and its refusal message
    "RP_ENV_FILE=${RP_ENV_FILE:-/etc/rp_environment}",
    "V2TEST_HOME=${V2TEST_HOME:-/home/v2test}",
    "V2GPU_LOCAL=${V2GPU_LOCAL:-/opt/v2}",
    '[ "${V2GPU_BOOTSTRAP_TEST:-0}" = 1 ] || { echo "RP_ENV_FILE=$RP_ENV_FILE V2TEST_HOME=$V2TEST_HOME V2GPU_LOCAL=$V2GPU_LOCAL:" \\',
    '"(V2GPU_BOOTSTRAP_TEST=1): refusing" >&2; exit 2; }']


def knob_problems(ops: Path) -> list[str]:
    """Every problem of ops/ (every file: scripts and both RUNSHEET files) with the test-only knobs: any file but the
    bootstrap that names one; any bootstrap line that sets one (NAME=, export, declare, read, ...) beyond
    BOOTSTRAP_SETS; the flag read anywhere but in the refusal."""
    problems = []
    for p in sorted(ops.rglob("*")):
        if not p.is_file():
            continue
        lines = [l.strip() for l in p.read_text().splitlines() if any(k in l for k in TEST_ONLY)]
        if p.name != "pod_bootstrap.sh":
            problems += [f"{p.name}: {l}" for l in lines]
            continue
        code = [l for l in lines if not l.startswith("#")]
        sets = [l for l in code if re.search(r"(?<![\w$])(RP_ENV_FILE|V2TEST_HOME|V2GPU_LOCAL|V2GPU_BOOTSTRAP_TEST)=", l) or
                re.search(r"\b(export|declare|typeset|local|readonly|read|mapfile|printf -v)\s", l)]
        if sets != BOOTSTRAP_SETS:
            problems.append(f"{p.name} sets: {sets}")
        problems += [f"{p.name} flag: {l}" for l in code
                     if "V2GPU_BOOTSTRAP_TEST" in l and l not in BOOTSTRAP_SETS[3:]]
    return problems


def test_ops_and_both_run_sheets_never_set_the_test_only_knobs(tmp_path):
    """(c): only the bootstrap names RP_ENV_FILE, V2TEST_HOME and V2GPU_BOOTSTRAP_TEST, and only to default them and to
    refuse a moved path without the flag; the RUNSHEET files, which start every pod's bootstrap, and the other ops scripts
    never name them. The scan itself flags a RUNSHEET file that would."""
    ops = ROOT / "ops"
    assert {"CANARY_RUNSHEET.md", "GEN_RUNSHEET.md", "pod_bootstrap.sh"} <= {p.name for p in ops.iterdir()}
    assert knob_problems(ops) == []
    planted = tmp_path / "ops"
    shutil.copytree(ops, planted)
    sheet = planted / "CANARY_RUNSHEET.md"
    launch = "setsid nohup bash -c"      # CANARY §5's gpu launch (the proven returning shape)
    sheet.write_text(sheet.read_text().replace(launch, "V2GPU_BOOTSTRAP_TEST=1 " + launch, 1))
    boot = planted / "pod_bootstrap.sh"
    boot.write_text(boot.read_text().replace("KIND=${KIND:-gpu}\n", "KIND=${KIND:-gpu}\nexport RP_ENV_FILE=/tmp/x\n"))
    assert "V2GPU_BOOTSTRAP_TEST=1 setsid nohup" in sheet.read_text() and "export RP_ENV_FILE" in boot.read_text()
    problems = knob_problems(planted)
    assert len(problems) == 2 and problems[0].startswith("CANARY_RUNSHEET.md: ") and "export RP_ENV_FILE" in \
        problems[1], problems


# Since 27/09/2026 GATE_ONLY needs the same kind of scan as the knobs above,
# but with a different allow-list -- pod_parity_gate.sh legitimately sets GATE_ONLY too (its docker INNER line)


def gate_problems(ops: Path) -> list[str]:
    """Only pod_bootstrap.sh (its default, its runtime guard) and pod_parity_gate.sh (the docker INNER line that
    invokes the bootstrap) may name GATE_ONLY; GATE_MARKER (its own test-only knob) is pod_bootstrap.sh's alone. Any
    other file of ops/, the RUNSHEET files included, naming either is a problem."""
    problems = []
    for p in sorted(ops.rglob("*")):
        if not p.is_file():
            continue
        text = p.read_text()
        if "GATE_ONLY" in text and p.name not in ("pod_bootstrap.sh", "pod_parity_gate.sh"):
            problems += [f"{p.name}: {l.strip()}" for l in text.splitlines() if "GATE_ONLY" in l]
        if "GATE_MARKER" in text and p.name != "pod_bootstrap.sh":
            problems += [f"{p.name}: {l.strip()}" for l in text.splitlines() if "GATE_MARKER" in l]
    return problems


# Since 27/09/2026 the backstop must ignore ordinary activity inside V2TEST_HOME (on a
# pod it IS the v2test suite's own $HOME) but still catch a chmod, chown or delete+recreate of the directory itself


def test_the_backstop_ignores_activity_inside_the_directory_but_catches_a_real_change(tmp_path):
    d = tmp_path / "home"
    d.mkdir()
    before = _snapshot(d)
    (d / ".cache").mkdir()                          # a benign top-level entry: used to change the directory's mtime
    (d / ".cache" / "x").write_text("y")
    assert _snapshot(d) == before, "an entry created inside the directory must not trip the backstop"
    d.chmod(0o700)
    assert _snapshot(d) != before, "a real chmod of the directory itself must still be caught"


def test_the_backstop_message_names_v2test_home():
    src = (ROOT / "tests_gpu" / "conftest.py").read_text()
    assert "override RP_ENV_FILE, V2TEST_HOME and V2GPU_LOCAL" in src and "override RP_ENV_FILE, TEST_HOME" not in src
