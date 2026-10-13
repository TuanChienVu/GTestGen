"""Calibrate the SAP's two tests before the lock (§5 of the design specification at 40bc09a, fixed 24/09/2026).
Simulates CA under the REAL test cluster structure with this data model (logit difficulty = logit(base rate) + u_cluster
~ N(0, 0.8) + e_record ~ N(0, 1); each trained adapter shifts its success rate by g ~ N(0, tau); a scenario's value is
the mean over its records) and reports, per comparison type ("base": adapter condition vs the one base run; "adapter":
two adapter conditions paired by seed) and per (tau, delta): the rejection rates of (1) the cluster sign-flip, (2) the
two-level bootstrap, and the "supported" / "contrary" rule (both), with their Monte Carlo SE; and, over the null data
sets, r = mean SD of the two-level distribution / mean SD of the single-seed cluster bootstrap, the factor of MDE(2) =
MDE(1) x r (MDE(1) is planned from one dev seed), with r_sign_flip = the same ratio for test (1) on the K-seed mean, so
that both MDEs can also be read on the same K-seed footing. The structure is that of the test's usable records (controls
minus the mechanical exclusions minus compose's excluded.jsonl); CALIBRATION.json's `inputs` hold the sha256 of the
records file, of controls.jsonl, of exclusions.jsonl and of the compose file, which sap_lock checks against the locked
test files and PHASE2_SHA256SUMS. A script, not a unit test; every simulation has its own seed, so the output does not
depend on --jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.sap import SapInputError, jsonl_rows, sha256_of, usable_records  # noqa: E402
from v2tools.stats import bootstrap, sign_flip_p, two_level, two_level_p  # noqa: E402

SETTINGS = [(mode, tau, delta) for mode in ("base", "adapter")
            for tau, delta in ((0.0, 0.0), (0.02, 0.0), (0.04, 0.0), (0.0, 0.06), (0.04, 0.06))]


def load(p: Path) -> list[dict]:
    return jsonl_rows(p)  # 3A's rule: "\n" only, a blank line inside the file refused


def structure(records: list[dict], keep: set[str]) -> list[list[int]]:
    """Cluster (.feature file within its repo) -> number of records of each of its scenarios, in sorted order."""
    by: dict = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r["id"] in keep:
            by[f"{r['repo']}|{r['feature_path']}"][f"{r['repo']}|{r['feature_path']}|{r['scenario_line']}"] += 1
    return [[by[c][s] for s in sorted(by[c])] for c in sorted(by)]


def clip(p: float) -> float:
    return min(1.0, max(0.0, p))


def simulate(task: tuple) -> tuple[float, float, float, float, float, float]:
    """One data set: (estimate, p of test 1, p of test 2, SD of the single-seed cluster bootstrap, SD of the two-level
    bootstrap, SD of the cluster bootstrap of the K-seed mean)."""
    struct, mode, tau, delta, k, seed, reps, base_rate = task
    rng = random.Random(seed)
    ga = [rng.gauss(0, tau) for _ in range(k)]
    gb = [rng.gauss(0, tau) for _ in range(k)]
    logit = math.log(base_rate / (1 - base_rate))
    per_seed: list[list[tuple[str, float]]] = [[] for _ in range(k)]
    for c, scenarios in enumerate(struct):
        u = rng.gauss(0, 0.8)
        for n in scenarios:
            ps = [1 / (1 + math.exp(-(logit + u + rng.gauss(0, 1.0)))) for _ in range(n)]
            base = sum(rng.random() < p for p in ps) / n if mode == "base" else None
            for s in range(k):
                a = sum(rng.random() < clip(p + delta + ga[s]) for p in ps) / n
                b = base if mode == "base" else sum(rng.random() < clip(p + gb[s]) for p in ps) / n
                per_seed[s].append((f"c{c}", a - b))
    pooled = [(c, sum(per_seed[s][j][1] for s in range(k)) / k) for j, (c, _) in enumerate(per_seed[0])]
    dist = two_level({f"s{s}": per_seed[s] for s in range(k)}, reps, seed + 2)
    return (sum(d for _, d in pooled) / len(pooled), sign_flip_p(pooled, reps, seed + 1), two_level_p(dist),
            statistics.stdev(bootstrap(per_seed[0], reps, seed + 3)), statistics.stdev(dist),
            statistics.stdev(bootstrap(pooled, reps, seed + 4)))


def summarise(res: list[tuple], alphas: list[float]) -> dict:
    n = len(res)
    rate = lambda f: (lambda x: {"rate": x, "mc_se": math.sqrt(x * (1 - x) / n)})(sum(map(f, res)) / n)
    out = {"n": n}
    for a in alphas:
        out[f"alpha={a:.4g}"] = {"sign_flip": rate(lambda r: r[1] < a), "two_level": rate(lambda r: r[2] < a),
                                 "supported": rate(lambda r: r[1] < a and r[2] < a and r[0] > 0),
                                 "contrary": rate(lambda r: r[1] < a and r[2] < a and r[0] < 0)}
    out["sd_single_seed"] = statistics.fmean(r[3] for r in res)
    out["sd_two_level"] = statistics.fmean(r[4] for r in res)
    out["sd_k_seed_mean"] = statistics.fmean(r[5] for r in res)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sap.json"))
    ap.add_argument("--records", type=Path, required=True, help="out/phase2/records/test.jsonl (the sealed test records)")
    ap.add_argument("--controls-dir", type=Path, required=True, help="test controls run: keep the usable records only")
    ap.add_argument("--compose-excluded", type=Path, required=True, help="compose's excluded.jsonl of the test prompts")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=999, help="draws per test inside one simulation")
    ap.add_argument("--base-rate", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    k = len(cfg["seeds"][cfg["anchor_model"]])
    test_files = {"controls": args.controls_dir / "controls.jsonl", "exclusions": args.controls_dir / "exclusions.jsonl",
                  "compose_excluded": args.compose_excluded}
    keep = usable_records(*(load(p) for p in test_files.values()), cfg["anchor_model"])
    struct = structure(load(args.records), keep)
    tasks = [(struct, mode, tau, delta, k, cfg["seed"] + 1_000_000 * j + 10 * i, args.reps, args.base_rate)
             for j, (mode, tau, delta) in enumerate(SETTINGS) for i in range(args.sims)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        res = list(ex.map(simulate, tasks, chunksize=max(1, len(tasks) // (20 * args.jobs))))
    alphas = [cfg["alpha"], cfg["alpha"] / max(len(p) for p in cfg["families"].values())]
    table = {f"{mode}|tau={tau:g}|delta={delta:g}": summarise(res[j * args.sims:(j + 1) * args.sims], alphas)
             for j, (mode, tau, delta) in enumerate(SETTINGS)}
    ratio = lambda num: {mode: {f"tau_{tau:g}": table[f"{mode}|tau={tau:g}|delta=0"][num] / table[f"{mode}|tau={tau:g}|delta=0"]["sd_single_seed"]
                                for tau in (0.0, 0.04)} for mode in ("base", "adapter")}
    r, r1 = ratio("sd_two_level"), ratio("sd_k_seed_mean")
    out = {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
           "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
           "inputs": {"records": sha256_of(args.records), **{name: sha256_of(p) for name, p in test_files.items()}},
           "clusters": len(struct), "scenarios": sum(map(len, struct)), "records": sum(map(sum, struct)), "seeds": k,
           "sims": args.sims, "reps": args.reps, "base_rate": args.base_rate, "results": table, "r": r, "r_sign_flip": r1}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    a = f"alpha={cfg['alpha']:.4g}"
    print(f"{out['clusters']} clusters, {out['scenarios']} scenarios, {out['records']} records, K = {k}, {args.sims} sims")
    print("| type | tau | delta | (1) sign-flip | (2) two-level | supported | contrary |")
    for name, v in table.items():
        mode, tau, delta = (x.split("=")[-1] for x in name.split("|"))
        print(f"| {mode} | {tau} | {delta} | " + " | ".join(f"{v[a][t]['rate']:.3f}" for t in ("sign_flip", "two_level", "supported", "contrary")) + " |")
    for name, rr in (("r", r), ("r_sign_flip", r1)):
        print(f"{name}:", json.dumps({m: {t: round(x, 3) for t, x in v.items()} for m, v in rr.items()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SapInputError as e:  # an input that is not JSONL (sap.jsonl_rows), wherever it is read: bad input
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(2)
