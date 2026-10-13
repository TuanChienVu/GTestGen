"""Grid pack (05/10; in the repo since decision 05/10): merge several genqueue QUEUE.json files of ONE pod into
one queue (laptop only, standard library only).

    python3 ops/merge_gen_queues.py OUT.json QUEUE_1.json QUEUE_2.json ...

genqueue takes one --variant and every queue it writes ends with a copy window and self-stops the pod, so 6-7 grid
labels on one pod would stop the pod after each label (a restart wipes the container disk and may lose the card). This
merges them the way GEN_RUNSHEET §6 merges the cross-pod --recheck-only queues: every regular job in order (each
label's gen job, then its recheck; a --prepend-queue'd xpod recheck at the head of the first queue stays first), then
ONE copy window (the last queue's) and ONE self-stop. Refused (exit 2, nothing written) when:
- fewer than two queues, or an unreadable one;
- the queues differ in queue_dir, lock_path, self_stop_cmd or deadline_utc;
- a queue does not end with exactly one copy-window-* job, or has another copy window, or a job that is neither
  gen-* nor recheck-*;
- the generate commands differ in --pin-set or --gpu-name;
- two jobs share a name or a workdir;
- one ship target gets two different sources.
"""
import json
import sys
from pathlib import Path

SAME = ("queue_dir", "lock_path", "self_stop_cmd", "deadline_utc")


def flag(cmd: list, name: str):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def merge(paths: list[Path]) -> tuple[dict | None, list[str]]:
    bad = []
    if len(paths) < 2:
        return None, ["need at least two queues"]
    qs = []
    for p in paths:
        try:
            qs.append(json.loads(Path(p).read_text()))
        except (OSError, ValueError) as e:
            bad.append(f"{p}: unreadable ({e})")
    if bad:
        return None, bad
    for k in SAME:
        if len({json.dumps(q.get(k), sort_keys=True) for q in qs}) != 1:
            bad.append(f"the queues differ in {k}")
    jobs, names, workdirs, settings = [], set(), set(), set()
    for p, q in zip(paths, qs):
        js = q.get("jobs") or []
        windows = [j for j in js if j.get("name", "").startswith("copy-window-")]
        if not js or not js[-1].get("name", "").startswith("copy-window-") or len(windows) != 1:
            bad.append(f"{p}: does not end with exactly one copy-window-* job")
        for j in js[:-1]:
            n = j.get("name", "")
            if not n.startswith(("gen-", "recheck-")):
                bad.append(f"{p}: job {n!r} is neither gen-* nor recheck-*")
            cmd = j.get("cmd") or []
            if "v2gpu.generate" in cmd:
                settings.add((flag(cmd, "--pin-set"), flag(cmd, "--gpu-name")))
            jobs.append(j)
    for j in jobs:
        if j["name"] in names:
            bad.append(f"job name {j['name']!r} twice")
        if j["workdir"] in workdirs:
            bad.append(f"workdir {j['workdir']!r} twice")
        names.add(j["name"])
        workdirs.add(j["workdir"])
    if len(settings) > 1:
        bad.append(f"the generate commands differ in --pin-set/--gpu-name: {sorted(map(str, settings))}")
    ship: dict = {}
    for p, q in zip(paths, qs):
        for section, items in (q.get("ship") or {}).items():
            dst = ship.setdefault(section, {})
            for target, source in items.items():
                if target in dst and dst[target] != source:
                    bad.append(f"ship target {target!r} has two sources: {dst[target]!r} and {source!r}")
                dst[target] = source
    if bad:
        return None, bad
    window = qs[-1]["jobs"][-1]
    out = {k: qs[0][k] for k in SAME}
    out["jobs"] = jobs + [window]
    out["ship"] = ship
    return out, []


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out, bad = merge([Path(a) for a in argv[1:]])
    if bad:
        print("REFUSED: " + "; ".join(bad), file=sys.stderr)
        return 2
    Path(argv[0]).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"out": argv[0], "jobs": [j["name"] for j in out["jobs"]],
                      "sum_max_hours": round(sum(j["max_hours"] for j in out["jobs"]), 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
