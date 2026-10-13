"""ER runner for Sylius @ c096ca3: one (record, condition) by hand, the single-run form of er.runner.

    .venv/bin/python er/er_run.py --record <id> --condition <c> [--overlay <dir>] [--lane 1|2|3] [--out <dir>]

Each run is a fresh container of gtestgen-er/sylius:c096ca3, so the clean tree is the image layer itself. The overlay
(full contents of the changed files, paths relative to the repo root) is mounted read-only and copied over /app.
Same isolation as er.runner: the lane's DML-only user (er.db.db_url, passed through the docker CLI's environment),
er.runner.HARDENING, the lane lock and the fixed container name per lane.
The runner never composes code and never applies the insertion rule. It prints one JSON line:
{record_id, condition, status, seconds, behat_exit, log_sha256, ...}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # also works as a top-level `er_run`
from er.db import NETWORK, db_url  # noqa: E402
from er.runner import HARDENING, container_name, lane_lock, remove_container  # noqa: E402

IMAGE = "gtestgen-er/sylius:c096ca3"
LANES = (1, 2, 3)
PHASE2 = Path("out/phase2")  # relative to the working directory: run from the repository root


def find_record(rid: str) -> dict:
    for part in ("test", "dev"):
        with (PHASE2 / "records" / f"{part}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] == rid:
                    return r
    raise SystemExit(f"record not found: {rid}")


def classify(out: Path, inner: dict, scenario_name: str) -> tuple[str, dict]:
    """Map behat exit + JUnit of exactly one scenario to PASS|FAIL|UNDEFINED|ERROR|TIMEOUT."""
    code = inner.get("behat_exit")
    if code in (124, 137):
        return "TIMEOUT", {}
    cases = []
    for x in sorted((out / "junit").glob("*.xml")) if (out / "junit").is_dir() else []:
        cases += ET.parse(x).getroot().iter("testcase")
    detail = {"n_scenarios": len(cases)}
    if len(cases) != 1:
        return "ERROR", detail  # 0 scenarios would otherwise exit 0 and look like a PASS
    case = cases[0]
    detail["junit_status"] = case.get("status")
    if case.get("name") != scenario_name:
        return "ERROR", {**detail, "ran": case.get("name")}
    if code not in (0, 1):
        return "ERROR", detail  # PHP fatal / uncaught exception in Behat itself
    if case.get("status") == "passed":
        return ("PASS" if code == 0 else "ERROR"), detail
    if case.get("status") == "undefined":
        return "UNDEFINED", detail
    return "FAIL", detail  # failed, pending, skipped


def run(rid: str, condition: str, overlay: Path | None, lane: int, out: Path, timeout: int, force_cold: bool,
        image: str = IMAGE, env: dict[str, str] | None = None) -> dict:
    rec = find_record(rid)
    out.mkdir(parents=True, exist_ok=True)
    name = container_name(lane)
    cmd = ["docker", "run", "--rm", "--name", name, "--network", NETWORK, *HARDENING, "-e", "DATABASE_URL",
           "-e", f"ER_TIMEOUT={timeout}", "-e", f"ER_FORCE_COLD={int(force_cold)}", "-v", f"{out.resolve()}:/out"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    if overlay is not None:
        cmd += ["-v", f"{overlay.resolve()}:/overlay:ro"]
    cmd += [image, "er-inner", rec["suite"], f"{rec['feature_path']}:{rec['scenario_line']}"]
    t0 = time.monotonic()
    with lane_lock(lane):
        remove_container(name)  # an orphan of a crashed run would still be using the lane database
        try:
            subprocess.run(cmd, check=True, timeout=timeout + 60, capture_output=True,
                           env={**os.environ, "DATABASE_URL": db_url(lane)})
        except subprocess.TimeoutExpired:
            remove_container(name)
            return {"record_id": rid, "condition": condition, "status": "TIMEOUT",
                    "seconds": round(time.monotonic() - t0, 3), "behat_exit": None, "log_sha256": None}
    seconds = round(time.monotonic() - t0, 3)
    inner = json.loads((out / "inner.json").read_text(encoding="utf-8"))
    status, detail = classify(out, inner, rec["scenario_name"])
    log = (out / "behat.log").read_bytes()
    return {"record_id": rid, "condition": condition, "status": status, "seconds": seconds,
            "behat_exit": inner.get("behat_exit"), "log_sha256": hashlib.sha256(log).hexdigest(),
            "behat_seconds": inner.get("behat_seconds"), "cold": inner.get("cold"),
            "mem_peak_bytes": inner.get("mem_peak_bytes"), **detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--overlay", type=Path)
    ap.add_argument("--lane", type=int, default=1, choices=LANES)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--force-cold", action="store_true")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the container")
    a = ap.parse_args()
    out = a.out or Path(tempfile.mkdtemp(prefix="er_"))
    res = run(a.record, a.condition, a.overlay, a.lane, out, a.timeout, a.force_cold, a.image,
              dict(kv.split("=", 1) for kv in a.env))
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
