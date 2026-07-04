"""Statistics tests (task §10): BH + permutation known-answer fixtures, plus
Cliff's delta, bootstrap and Wilson sanity."""

from __future__ import annotations

import numpy as np

from failure_mech import stats


def test_bh_known_answer():
    # m=5, alpha=0.05. Threshold p_(k) <= (k/m)*alpha:
    #   k=1: 0.001<=0.01 ok; k=2: 0.008<=0.02 ok; k=3: 0.039<=0.03 FAILS.
    # Largest passing k=2 -> reject the two smallest only.
    p = [0.001, 0.008, 0.039, 0.2, 0.5]
    out = stats.bh_correct(p, 0.05)
    assert out["rejected"].tolist() == [True, True, False, False, False]
    # q-values are monotone non-decreasing in p order
    q = out["qvalues"]
    assert q[0] <= q[1] <= q[2] <= q[3] <= q[4]


def test_bh_nothing_rejected():
    out = stats.bh_correct([0.9, 0.8, 0.7], 0.05)
    assert out["rejected"].tolist() == [False, False, False]


def test_bh_handles_nan():
    out = stats.bh_correct([0.001, float("nan"), 0.5], 0.05)
    assert out["rejected"][0] == True  # noqa: E712
    assert out["rejected"][1] == False  # noqa: E712
    assert np.isnan(out["qvalues"][1])


def test_permutation_exact_all_positive():
    # 5 all-positive diffs: only the observed sign pattern is >= observed (two
    # tails) -> p = 2/32 = 0.0625 exactly (enumerated).
    out = stats.paired_permutation([0.2, 0.3, 0.1, 0.25, 0.4], n_perm=10000, two_sided=True)
    assert out["exact"] is True
    assert abs(out["p"] - 2 / 32) < 1e-9


def test_permutation_symmetric_is_nonsignificant():
    out = stats.paired_permutation([0.1, -0.1, 0.1, -0.1], n_perm=10000, two_sided=True)
    assert out["p"] > 0.5


def test_cliffs_delta_extremes():
    assert abs(stats.cliffs_delta([5, 6, 7], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(stats.cliffs_delta([1, 2, 3], [5, 6, 7]) + 1.0) < 1e-9
    assert abs(stats.cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-9


def test_bootstrap_ci_brackets_mean():
    data = list(np.random.default_rng(0).normal(5.0, 1.0, 200))
    ci = stats.bootstrap_ci(lambda x: float(x.mean()), data, n=2000, level=0.95, seed=0)
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert ci["lo"] < 5.0 < ci["hi"]


def test_wilson_ci_bounds():
    w = stats.wilson_ci(8, 10, 0.95)
    assert 0.0 <= w["lo"] < w["p"] < w["hi"] <= 1.0
    edge = stats.wilson_ci(0, 10, 0.95)
    assert edge["lo"] == 0.0
