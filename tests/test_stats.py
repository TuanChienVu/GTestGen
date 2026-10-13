import random

import pytest

from v2tools.stats import bootstrap, clusters, compare, holm, kappa, mde, percentile_ci, sign_flip_p, two_level, two_level_p


def synthetic(delta: float, seed: int, n_clusters: int = 60) -> list[tuple[str, float]]:
    """Paired scenario differences with a cluster effect: d = delta + u_cluster + e, clipped to [-1, 1]."""
    rng = random.Random(seed)
    out = []
    for c in range(n_clusters):
        u = rng.gauss(0, 0.1)
        for _ in range(rng.randint(1, 8)):
            out.append((f"f{c:03d}.feature", max(-1.0, min(1.0, delta + u + rng.gauss(0, 0.4)))))
    return out


def test_known_effect_is_detected_and_covered():
    hits = covered = 0
    for s in range(20):
        c = compare(synthetic(0.15, s), reps_boot=1000, reps_perm=999, seed=s)
        hits += c.p < 0.05
        covered += c.ci[0] <= 0.15 <= c.ci[1]
    assert hits >= 18 and covered >= 16


def test_type_one_error_is_near_alpha():
    rejections = sum(sign_flip_p(synthetic(0.0, 1000 + s), 499, s) < 0.05 for s in range(200))
    assert 0.015 <= rejections / 200 <= 0.09


def test_exact_sign_flip_with_few_clusters():
    pairs = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
    assert sign_flip_p(pairs, reps=10_000, seed=0) == pytest.approx(2 / 8)


def test_bootstrap_is_deterministic_and_clusters_move_together():
    pairs = [("a", 1.0), ("a", 1.0), ("b", 0.0)]
    assert bootstrap(pairs, 50, 7) == bootstrap(pairs, 50, 7)
    assert set(bootstrap(pairs, 200, 7)) <= {0.0, 2 / 3, 1.0, 0.5}
    assert clusters(pairs) == [(2.0, 2), (0.0, 1)]
    assert percentile_ci([float(i) for i in range(101)]) == (2.0, 98.0)


def test_holm():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"a": 1.0}) == {"a": 1.0}


def test_mde_scales_with_target_clusters():
    dev = synthetic(0.0, 3, n_clusters=40)
    a = mde(dev, 100, 400, 0.05 / 3, 0.8, 2000, 1)
    b = mde(dev, 400, 1600, 0.05 / 3, 0.8, 2000, 1)
    assert b["mde"] == pytest.approx(a["mde"] / 2, rel=1e-9) and 0 < a["mde"] < 1
    assert a["clusters_dev"] == 40 and a["mde_by_scenarios"] > 0
    flat = mde([("f1", 1.0), ("f2", 1.0)], 100, 100, 0.05, 0.8, 100, 1)
    assert flat["mde"] == 0 and flat["degenerate"] is True


def test_two_level_bootstrap_widens_with_seed_disagreement():
    base = synthetic(0.0, 9, n_clusters=30)
    same = {s: [(c, d + 0.1) for c, d in base] for s in "abcde"}
    spread = {s: [(c, d + off) for c, d in base] for s, off in zip("abcde", (-0.2, -0.1, 0.1, 0.3, 0.4))}
    lo1, hi1 = percentile_ci(two_level(same, 1000, 1))
    lo2, hi2 = percentile_ci(two_level(spread, 1000, 1))
    assert hi2 - lo2 > 2 * (hi1 - lo1) and two_level(same, 50, 3) == two_level(same, 50, 3)


def test_two_level_p_is_two_sided_and_never_zero():
    assert two_level_p([0.1] * 99) == pytest.approx(2 / 100)
    assert two_level_p([-0.1] * 99) == pytest.approx(2 / 100)
    assert two_level_p([-1.0] * 30 + [1.0] * 70) == pytest.approx(2 * 31 / 101)
    assert two_level_p([0.0] * 10) == 1.0


def test_kappa():
    assert kappa([(1, 1), (0, 0), (1, 1), (0, 0)])["kappa"] == 1.0
    k = kappa([(1, 1), (1, 0), (0, 1), (0, 0)])
    assert k["agreement"] == 0.5 and k["kappa"] == 0.0 and k["table"] == {"00": 1, "01": 1, "10": 1, "11": 1}
    assert kappa([])["kappa"] is None
