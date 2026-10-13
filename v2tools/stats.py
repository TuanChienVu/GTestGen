"""Statistics of the Phase 3 SAP (§5 of the design specification at 40bc09a), pure Python and deterministic (every draw
comes from random.Random(seed)). The unit is the scenario and the cluster is its .feature file: a paired comparison is a
list of (cluster, d) with d the scenario's difference between two conditions. The estimate is the scenario-weighted mean
of d; its 95% interval is a percentile cluster bootstrap. Test (1): sign-flipping whole clusters of the seed-averaged d
(conditional on the adapters trained). Test (2): the two-level bootstrap (seeds, then clusters), which generalises over
training runs. Holm adjusts within one RQ family, separately for each test; the MDE is planned from dev-pilot
differences."""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from statistics import NormalDist

Pairs = list[tuple[str, float]]


def clusters(pairs: Pairs) -> list[tuple[float, int]]:
    """(sum of d, number of scenarios) per cluster, in sorted cluster order."""
    acc: dict[str, list] = {}
    for c, d in pairs:
        s = acc.setdefault(c, [0.0, 0])
        s[0] += d
        s[1] += 1
    return [tuple(acc[c]) for c in sorted(acc)]


def mean(pairs: Pairs) -> float:
    return sum(d for _, d in pairs) / len(pairs) if pairs else float("nan")


def bootstrap(pairs: Pairs, reps: int, seed: int) -> list[float]:
    """Cluster bootstrap distribution of the scenario-weighted mean of d."""
    cs = clusters(pairs)
    rng, out = random.Random(seed), []
    for _ in range(reps):
        tot = n = 0.0
        for _ in range(len(cs)):
            s, k = cs[rng.randrange(len(cs))]
            tot += s
            n += k
        out.append(tot / n)
    return out


def percentile_ci(dist: list[float], level: float = 0.95) -> tuple[float, float]:
    xs = sorted(dist)
    lo = xs[int(math.floor((1 - level) / 2 * (len(xs) - 1)))]
    hi = xs[int(math.ceil((1 + level) / 2 * (len(xs) - 1)))]
    return lo, hi


def sign_flip_p(pairs: Pairs, reps: int, seed: int) -> float:
    """Two-sided p-value of mean d = 0 by flipping the sign of whole clusters: exact when 2^clusters <= reps, else
    (1 + #{|T*| >= |T|}) / (1 + reps)."""
    cs = clusters(pairs)
    if not cs:
        return float("nan")
    n = sum(k for _, k in cs)
    t = abs(sum(s for s, _ in cs)) / n
    eps = 1e-12
    if 2 ** len(cs) <= reps:
        hits = sum(abs(sum(s * g for (s, _), g in zip(cs, signs))) / n >= t - eps
                   for signs in itertools.product((1, -1), repeat=len(cs)))
        return hits / 2 ** len(cs)
    rng = random.Random(seed)
    hits = sum(abs(sum(s if rng.random() < 0.5 else -s for s, _ in cs)) / n >= t - eps for _ in range(reps))
    return (1 + hits) / (1 + reps)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


@dataclass(frozen=True)
class Comparison:
    n_scenarios: int
    n_clusters: int
    estimate: float
    ci: tuple[float, float]
    p: float


def compare(pairs: Pairs, reps_boot: int, reps_perm: int, seed: int, with_p: bool = True) -> Comparison:
    ci = percentile_ci(bootstrap(pairs, reps_boot, seed)) if pairs else (float("nan"), float("nan"))
    return Comparison(len(pairs), len(clusters(pairs)), mean(pairs), ci,
                      sign_flip_p(pairs, reps_perm, seed + 1) if with_p and pairs else float("nan"))


def two_level(per_seed: dict[str, Pairs], reps: int, seed: int) -> list[float]:
    """Two-level bootstrap (§5 of the design specification, test (2)): resample the seeds with replacement, then the
    clusters; the statistic is the scenario-weighted mean over the drawn clusters of the mean over the drawn seeds.
    Every seed must cover the same scenarios (the completeness check guarantees it for CA)."""
    names = sorted(per_seed)
    cl = sorted({c for c, _ in per_seed[names[0]]})
    idx = {c: i for i, c in enumerate(cl)}
    sums = [[0.0] * len(cl) for _ in names]
    count = [0] * len(cl)
    for j, s in enumerate(names):
        for c, d in per_seed[s]:
            sums[j][idx[c]] += d
    for c, _ in per_seed[names[0]]:
        count[idx[c]] += 1
    rng, out, k = random.Random(seed), [], len(names)
    for _ in range(reps):
        w = [0] * k
        for _ in range(k):
            w[rng.randrange(k)] += 1
        mixed = [sum(x) / k for x in zip(*([v * n for v in sums[j]] for j, n in enumerate(w) if n))]
        tot = n = 0.0
        for _ in cl:
            i = rng.randrange(len(cl))
            tot += mixed[i]
            n += count[i]
        out.append(tot / n)
    return out


def two_level_p(dist: list[float]) -> float:
    """Two-sided p of test (2) from the bootstrap distribution (fixed 24/09/2026):
    min(1, 2 min(1 + #{t <= 0}, 1 + #{t >= 0}) / (B + 1)), never exactly 0,
    like the sign-flip's (1 + hits) / (1 + reps)."""
    le = sum(t <= 0 for t in dist)
    ge = sum(t >= 0 for t in dist)
    return min(1.0, 2 * min(1 + le, 1 + ge) / (len(dist) + 1))


def mde(dev_pairs: Pairs, n_clusters_target: int, n_scenarios_target: int, alpha: float, power: float, reps: int,
        seed: int) -> dict:
    """Minimum detectable effect of a paired comparison on the test set, planned from dev-pilot differences (one seed):
    the cluster-bootstrap SE of the dev mean, scaled by sqrt(clusters_dev / clusters_test) (primary: the bootstrap
    resamples clusters) and, for reference, by sqrt(scenarios_dev / scenarios_test), times z(1 - alpha/2) + z(power).
    Use the family's strictest Holm level as alpha (0.05/3 for RQ1)."""
    dist = bootstrap(dev_pairs, reps, seed)
    m = sum(dist) / len(dist)
    se_dev = math.sqrt(sum((x - m) ** 2 for x in dist) / (len(dist) - 1))
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    k_dev = len(clusters(dev_pairs))
    return {"n_dev": len(dev_pairs), "clusters_dev": k_dev, "se_dev": se_dev, "clusters_target": n_clusters_target,
            "n_target": n_scenarios_target, "alpha": alpha, "power": power,
            "mde": z * se_dev * math.sqrt(k_dev / n_clusters_target),
            "mde_by_scenarios": z * se_dev * math.sqrt(len(dev_pairs) / n_scenarios_target),
            "degenerate": se_dev == 0}  # no variation on dev: report, not trust


def kappa(pairs: list[tuple[int, int]]) -> dict:
    """Cohen's kappa and raw agreement of two binary ratings."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    agree = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agreement": agree, "kappa": None if pe == 1 else (agree - pe) / (1 - pe),
            "table": {f"{a}{b}": sum(x == a and y == b for x, y in pairs) for a in (0, 1) for b in (0, 1)}}
