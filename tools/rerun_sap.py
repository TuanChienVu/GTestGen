"""Rerun the registered analysis from this repository and the data packages, and compare with the published results.

usage: python3 tools/rerun_sap.py --data DIR --a3 DIR --work DIR [--python EXE]

--data   a directory holding the unpacked data packages at their repository paths: out/phase2/ (A1),
         out/phase3/prompts/test_v0/ (A2), out/phase3/scores/, out/phase3/sap/ and out/phase3/self_contained.jsonl (A4),
         out/phase3/er/controls/ and out/phase3/overlays/controls/ (A5). It may be the repository root itself.
--a3     the root of the unpacked generation package A3 (it holds gen-main-0482a76-20261008/results/).
--work   a new or empty directory; everything this script writes goes there.
--python the interpreter for scripts/sap_run.py and scripts/sap_dev_select.py (default: this one).

What it does:
1. Replay repository. scripts/sap_run.py reads the lock and each scores manifest (SCORES_META.json) as committed in
   git: the commits must be on a branch of `origin` and ancestors of HEAD, the scores commit must descend from the
   lock commit, and the lock's commit date must precede every scoring time. This repository has a new history, so the
   script builds <work>/replay with a local bare `origin`: commit L holds prereg/SAP_LOCK.public.json at
   out/phase3/sap/SAP_LOCK.json, then commit M the published test SCORES_META.json and commit A the published aux_test
   one. Author, dates and messages are those of the original commits, read from the `fuller` fields of the published
   SAP_RESULTS.json. In this rerun the date and ancestry checks are therefore a replay; the independent time anchor
   of the lock is its OSF registration (prereg/LOCK_EQUIVALENCE.md). The data files are linked into the replay
   checkout at their repository paths, so sap_run records the same relative paths as the original run.
2. scripts/sap_run.py for the test part and for the aux_test part (--descriptive-only), with the options of the
   original run, from the replay checkout.
3. scripts/sap_dev_select.py over the 27 dev score files; its output must equal prereg/DEV_SELECTION.json byte for
   byte.
4. Each rerun SAP_RESULTS.json is compared with the published one: every numeric field must be equal, and then the
   whole file byte for byte after a fixed substitution (SUBSTITUTION in RERUN_REPORT.json): the replay commits by
   the original commits, and the commit of the code that ran (tool_commit) by the published one. SAP_REPORT.md is
   compared the same way. Nothing else may differ.
Exit 0 when every check passes, 1 otherwise, 2 on bad input. The report is <work>/RERUN_REPORT.json.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = "A3/gen-main-0482a76-20261008/results"
PARTS = {"test": {"results": "out/phase3/sap/results", "extra": []},
         "aux_test": {"results": "out/phase3/sap/aux_test", "extra": ["--descriptive-only"]}}
DEV_V = "dev_pilot_v0_0482a76"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(repo: Path, *args: str, env: dict | None = None) -> str:
    base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env={**base, **(env or {})})
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def original_commit(fuller: str, committed: str) -> dict:
    """Author, author date, committer, commit date and message of a commit, from its `git show -s --format=fuller`
    text and its %cI date, as SAP_RESULTS.json records them."""
    m = re.match(r"commit ([0-9a-f]{40})\nAuthor: +(.+) <(.+)>\nAuthorDate: +(.+)\nCommit: +(.+) <(.+)>\nCommitDate: +(.+)\n"
                 r"\n((?: {4}.*\n?)+)$", fuller)
    if not m:
        sys.exit("a published fuller field is not in the expected form")
    adate = dt.datetime.strptime(m.group(4), "%a %b %d %H:%M:%S %Y %z").isoformat()
    message = "\n".join(line[4:] for line in m.group(8).rstrip("\n").split("\n")) + "\n"
    return {"sha": m.group(1), "author": (m.group(2), m.group(3)), "author_date": adate,
            "committer": (m.group(5), m.group(6)), "commit_date": committed, "message": message}


def commit(repo: Path, paths: list[str], c: dict) -> str:
    git(repo, "add", "--", *paths)
    env = {"GIT_AUTHOR_NAME": c["author"][0], "GIT_AUTHOR_EMAIL": c["author"][1], "GIT_AUTHOR_DATE": c["author_date"],
           "GIT_COMMITTER_NAME": c["committer"][0], "GIT_COMMITTER_EMAIL": c["committer"][1],
           "GIT_COMMITTER_DATE": c["commit_date"]}
    git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", c["message"].rstrip("\n"), env=env)
    return git(repo, "rev-parse", "HEAD")


def link(replay: Path, rel: str, target: Path) -> None:
    if not target.exists():
        sys.exit(f"missing input: {target}")
    p = replay / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.symlink_to(target)


def numbers(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from numbers(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from numbers(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, o


def main() -> int:
    ap = argparse.ArgumentParser(description="Rerun the registered analysis and compare with the published results.")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--a3", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    data, a3, work = a.data.resolve(), a.a3.resolve(), a.work.resolve()
    if work.exists() and any(work.iterdir()):
        print(f"--work {work} is not empty", file=sys.stderr)
        return 2
    published = {p: json.loads((data / v["results"] / "SAP_RESULTS.json").read_text(encoding="utf-8")) for p, v in PARTS.items()}
    orig = {"lock": original_commit(published["test"]["lock_commit"]["fuller"], published["test"]["lock_commit"]["committed"])}
    for p in PARTS:
        g = published[p]["generation"]["scores_meta_commit"]
        orig[p] = original_commit(g["fuller"], g["committed"])

    # 1. replay repository
    replay, origin = work / "replay", work / "origin.git"
    replay.mkdir(parents=True)
    git(work, "init", "-q", "--bare", str(origin))
    git(work, "init", "-q", "-b", "main", str(replay))
    lock_rel = "out/phase3/sap/SAP_LOCK.json"
    (replay / lock_rel).parent.mkdir(parents=True)
    (replay / lock_rel).write_bytes((ROOT / "prereg" / "SAP_LOCK.public.json").read_bytes())
    replay_sha = {"lock": commit(replay, [lock_rel], orig["lock"])}
    for p in PARTS:
        rel = f"out/phase3/scores/{p}/SCORES_META.json"
        (replay / rel).parent.mkdir(parents=True, exist_ok=True)
        (replay / rel).write_bytes((data / rel).read_bytes())
        replay_sha[p] = commit(replay, [rel], orig[p])
    git(replay, "remote", "add", "origin", str(origin))
    git(replay, "push", "-q", "origin", "main")
    git(replay, "fetch", "-q", "origin")
    # the data, linked at their repository paths (untracked)
    for name in ("MDE.json", "CALIBRATION.json", "DEV_SELECTION.json"):
        link(replay, f"out/phase3/sap/{name}", ROOT / "prereg" / name)
    for p in PARTS:
        for f in sorted((data / "out/phase3/scores" / p).iterdir()):
            if f.name not in ("SCORES_META.json", "SHA256SUMS"):
                link(replay, f"out/phase3/scores/{p}/{f.name}", f)
    scores = data / "out/phase3/scores"
    for d in sorted(scores.iterdir()):
        if d.name.startswith("dev_") or d.name in ("controls_test", "controls_aux_test", "controls_dev", "dose_test"):
            link(replay, f"out/phase3/scores/{d.name}", d)
    link(replay, "out/phase2", data / "out/phase2")
    link(replay, "out/phase3/prompts/test_v0", data / "out/phase3/prompts/test_v0")
    link(replay, "out/phase3/self_contained.jsonl", data / "out/phase3/self_contained.jsonl")
    link(replay, "out/phase3/er/controls", data / "out/phase3/er/controls")
    link(replay, "out/phase3/overlays/controls", data / "out/phase3/overlays/controls")
    link(replay, "evidence/pret_flags", ROOT / "evidence/pret_flags")
    link(replay, "config", ROOT / "config")
    link(replay, "A3", a3)

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    report = {"replay_commits": replay_sha, "original_commits": {k: v["sha"] for k, v in orig.items()}, "checks": {}}
    ok = True

    # 2. sap_run, test and aux_test
    common = ["--repo", str(replay), "--lock", lock_rel, "--lock-commit", replay_sha["lock"],
              "--mde", "out/phase3/sap/MDE.json", "--calibration", "out/phase3/sap/CALIBRATION.json",
              "--dev-selection", "out/phase3/sap/DEV_SELECTION.json",
              "--dev-scores", f"out/phase3/scores/dev_{DEV_V}/scores.jsonl",
              "--dev-controls-dir", "out/phase3/scores/controls_dev", "--controls-dir", "out/phase3/scores/controls_test",
              "--compose-excluded", "out/phase3/prompts/test_v0/excluded.jsonl",
              "--records", "out/phase2/records/test.jsonl", "--subsets", "out/phase2/sensitivity_subsets.json",
              "--self-contained", "out/phase3/self_contained.jsonl", "--phase2-sums", "out/phase2/PHASE2_SHA256SUMS",
              "--er-controls", "out/phase3/er/controls/er_results.jsonl",
              "--er-manifest", "out/phase3/overlays/controls/manifest.jsonl",
              "--aux-controls-dir", "out/phase3/scores/controls_aux_test",
              "--aux-records", "out/phase2/records/aux_test.jsonl",
              "--pre-t-flags", "evidence/pret_flags/FLAGGED_PRE_T.tsv", "--dose", "out/phase3/scores/dose_test/dose.jsonl",
              "--manifest", f"{GEN}/manifest.main.jsonl", "--coverage", f"{GEN}/coverage.main.json"]
    for p, v in PARTS.items():
        out = work / f"sap_run_{p}"
        cmd = [a.python, str(ROOT / "scripts/sap_run.py"), *common, "--scores", f"out/phase3/scores/{p}/scores.jsonl",
               "--scores-meta-commit", replay_sha[p], "--out", str(out), *v["extra"]]
        r = subprocess.run(cmd, cwd=replay, env=env, capture_output=True, text=True)
        (work / f"sap_run_{p}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        check = {"exit": r.returncode}
        if r.returncode == 0:
            check.update(compare(out, data / v["results"], replay_sha, orig, p, published[p]))
        check["pass"] = r.returncode == 0 and check.get("numbers_equal") and check.get("results_bytes_equal") \
            and check.get("report_bytes_equal")
        report["checks"][f"sap_run_{p}"] = check
        ok &= bool(check["pass"])

    # 3. sap_dev_select over the 27 dev score files
    dev = sorted(f"out/phase3/scores/{d.name}/scores.jsonl" for d in scores.iterdir() if d.name.startswith("dev_"))
    out = work / "DEV_SELECTION.json"
    r = subprocess.run([a.python, str(ROOT / "scripts/sap_dev_select.py"), "--dev-scores", *dev,
                        "--dev-controls-dir", "out/phase3/scores/controls_dev", "--out", str(out)],
                       cwd=replay, env=env, capture_output=True, text=True)
    (work / "sap_dev_select.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    same = r.returncode == 0 and out.exists() and out.read_bytes() == (ROOT / "prereg/DEV_SELECTION.json").read_bytes()
    report["checks"]["sap_dev_select"] = {"exit": r.returncode, "dev_score_files": len(dev),
                                          "equals_prereg_DEV_SELECTION": same, "pass": same}
    ok &= same

    report["pass"] = ok
    (work / "RERUN_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in report["checks"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}\t{k}\t" + json.dumps({x: y for x, y in v.items() if x != 'pass'},
                                                                        ensure_ascii=False)[:300])
    print("PASS" if ok else "FAIL", "-", work / "RERUN_REPORT.json")
    return 0 if ok else 1


def compare(out: Path, pub_dir: Path, replay_sha: dict, orig: dict, part: str, published: dict) -> dict:
    """Numbers of SAP_RESULTS.json, then SAP_RESULTS.json and SAP_REPORT.md byte for byte after the substitution."""
    got = json.loads((out / "SAP_RESULTS.json").read_text(encoding="utf-8"))
    a, b = dict(numbers(got)), dict(numbers(published))
    diff_num = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    subst = [(replay_sha["lock"], orig["lock"]["sha"]), (replay_sha[part], orig[part]["sha"])]
    tool = (json.dumps(got.get("tool_commit")), json.dumps(published.get("tool_commit")))
    res = {"numeric_fields": len(b), "numbers_equal": not diff_num, "numbers_differing": diff_num[:20],
           "substitution": [list(x) for x in subst] + [[f'"tool_commit": {tool[0]}', f'"tool_commit": {tool[1]}']]}

    def apply(text: str, report: bool) -> str:
        for x, y in subst:
            text = text.replace(x, y)
        if report:  # the report prints the tool commit as `<sha>` (sap_run's header line)
            return text.replace(f"công cụ `{got.get('tool_commit')}`", f"công cụ `{published.get('tool_commit')}`")
        return text.replace(*res["substitution"][-1], 1)

    gr = apply((out / "SAP_RESULTS.json").read_text(encoding="utf-8"), False)
    res["results_bytes_equal"] = gr.encode() == (pub_dir / "SAP_RESULTS.json").read_bytes()
    rr = apply((out / "SAP_REPORT.md").read_text(encoding="utf-8"), True)
    res["report_bytes_equal"] = rr.encode() == (pub_dir / "SAP_REPORT.md").read_bytes()
    if not res["report_bytes_equal"]:
        pub = (pub_dir / "SAP_REPORT.md").read_text(encoding="utf-8").split("\n")
        res["report_first_difference"] = next(([i + 1, x[:160], y[:160]] for i, (x, y) in
                                               enumerate(zip(rr.split("\n"), pub)) if x != y), "length differs")
    return res


if __name__ == "__main__":
    sys.exit(main())
