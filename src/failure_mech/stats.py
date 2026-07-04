"""
Statistics primitives (task §4.6).

Paired sign-flip permutation, Benjamini-Hochberg, Cliff's delta, bootstrap CI,
plus the Wilson interval E1 reports for cell accuracy. Statistical parameters
(alpha, effect-size choice, CI level, permutation/bootstrap counts) are passed
in EXPLICITLY by the caller from prereg ``statistics`` / ``causal_criteria`` —
this module never reaches into prereg and has no hidden thresholds (§4.6).

Determinism (§1.7): every Monte-Carlo routine takes an explicit ``seed``; the
same inputs + seed give the same bytes. Exact enumeration is used when the
number of sign patterns is small enough, making those cases seed-independent.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Paired permutation (sign-flip)
# ---------------------------------------------------------------------------

def paired_permutation(
    diffs: Sequence[float],
    n_perm: int = 10000,
    two_sided: bool = True,
    *,
    seed: int = 0,
    statistic: str = "mean",
) -> dict[str, float]:
    """Sign-flip permutation test on paired differences.

    Under H0 the sign of each pair difference is exchangeable. The observed
    statistic is ``mean(diffs)`` (or ``sum``). When ``2**n <= n_perm`` all sign
    patterns are enumerated exactly (seed-independent); otherwise ``n_perm``
    random sign vectors are drawn with ``seed``.

    Returns ``{"p": ..., "observed": ..., "n": ..., "exact": bool, "n_perm": ...}``.
    Zeros contribute no sign information but are kept in the mean, matching a
    two-sided test of the null that the mean difference is 0.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n == 0:
        return {"p": float("nan"), "observed": float("nan"), "n": 0,
                "exact": False, "n_perm": 0}

    agg = (np.mean if statistic == "mean" else np.sum)
    observed = float(agg(d))

    def _stat(signs: np.ndarray) -> np.ndarray:
        # signs: (P, n) of +-1 -> aggregate over axis 1
        vals = signs * d[None, :]
        return np.mean(vals, axis=1) if statistic == "mean" else np.sum(vals, axis=1)

    exact = (2 ** n) <= n_perm
    if exact:
        patterns = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        stats = _stat(patterns)
        total = patterns.shape[0]
    else:
        rng = np.random.default_rng(seed)
        patterns = rng.choice(np.array([1.0, -1.0]), size=(n_perm, n))
        stats = _stat(patterns)
        total = n_perm

    if two_sided:
        extreme = np.sum(np.abs(stats) >= abs(observed) - 1e-12)
    else:
        extreme = np.sum(stats >= observed - 1e-12)

    if exact:
        p = float(extreme / total)
    else:
        # +1 correction (Monte-Carlo p-value never 0): count the observed too.
        p = float((extreme + 1) / (total + 1))
    return {"p": p, "observed": observed, "n": int(n), "exact": bool(exact),
            "n_perm": int(total)}


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------

def bh_correct(pvals: Sequence[float], alpha: float) -> dict[str, object]:
    """Benjamini-Hochberg FDR correction at level ``alpha``.

    Returns ``{"rejected": bool ndarray, "qvalues": ndarray, "alpha": alpha,
    "n": m}`` in the INPUT order. NaN p-values are carried through as
    non-rejected with NaN q-value and excluded from the ranking (they are not
    valid tests).
    """
    p = np.asarray(pvals, dtype=np.float64)
    m_total = p.size
    valid = ~np.isnan(p)
    idx_valid = np.where(valid)[0]
    m = idx_valid.size

    rejected = np.zeros(m_total, dtype=bool)
    qvalues = np.full(m_total, np.nan, dtype=np.float64)
    if m == 0:
        return {"rejected": rejected, "qvalues": qvalues, "alpha": float(alpha), "n": 0}

    pv = p[idx_valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    ranks = np.arange(1, m + 1)

    # BH-adjusted p-values (step-up, monotone via reverse cumulative min).
    q_ranked = ranked * m / ranks
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    # Threshold: largest k with p_(k) <= (k/m) * alpha.
    below = ranked <= (ranks / m) * alpha
    k = np.max(np.where(below)[0]) + 1 if np.any(below) else 0
    rej_ranked = np.zeros(m, dtype=bool)
    if k > 0:
        rej_ranked[:k] = True

    # Un-sort back to valid-index order, then scatter to full input order.
    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    qvalues[idx_valid] = q_ranked[inv]
    rejected[idx_valid] = rej_ranked[inv]
    return {"rejected": rejected, "qvalues": qvalues, "alpha": float(alpha), "n": int(m)}


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta effect size in [-1, 1].

    delta = P(X > Y) - P(X < Y) over all pairs. +1: every x exceeds every y;
    0: stochastic equality. Ties contribute 0. O(nx log nx) via sorting.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    nx, ny = xa.size, ya.size
    if nx == 0 or ny == 0:
        return float("nan")
    ys = np.sort(ya)
    # For each x: count y strictly less than x, and y strictly greater than x.
    less = np.searchsorted(ys, xa, side="left")            # per x: #(y < x) == #(x > y)
    greater = ny - np.searchsorted(ys, xa, side="right")   # per x: #(y > x) == #(x < y)
    # Cliff's delta = P(x > y) - P(x < y).
    delta = (np.sum(less) - np.sum(greater)) / (nx * ny)
    return float(delta)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    data: Sequence[float],
    n: int = 10000,
    level: float = 0.95,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for ``stat_fn`` over ``data`` (1-D resampling).

    Returns ``{"point": stat_fn(data), "lo": ..., "hi": ..., "level": level,
    "n_boot": n}``. Rows of ``data`` are resampled with replacement; for paired
    statistics pass the per-pair values and let ``stat_fn`` aggregate.
    """
    arr = np.asarray(data, dtype=np.float64)
    m = arr.size
    if m == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "level": float(level), "n_boot": 0}
    point = float(stat_fn(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    boot = np.array([stat_fn(arr[i]) for i in idx], dtype=np.float64)
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    lo, hi = np.nanpercentile(boot, [100 * lo_q, 100 * hi_q])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "level": float(level), "n_boot": int(n)}


# ---------------------------------------------------------------------------
# Wilson interval (E1 accuracy CI, §5)
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, level: float = 0.95) -> dict[str, float]:
    """Wilson score interval for a binomial proportion (E1 cell accuracy)."""
    if n == 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "level": float(level)}
    z = _z_for_level(level)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return {"p": float(phat), "lo": float(max(0.0, center - half)),
            "hi": float(min(1.0, center + half)), "n": int(n), "level": float(level)}


def _z_for_level(level: float) -> float:
    """Two-sided normal quantile for a confidence level (via scipy; erfinv
    fallback keeps this dependency-light and deterministic)."""
    alpha = 1.0 - level
    return float(math.sqrt(2.0) * _erfinv(1.0 - alpha))


def _erfinv(y: float) -> float:
    try:
        from scipy.special import erfinv  # local import: scipy optional at call
        return float(erfinv(y))
    except Exception:  # noqa: BLE001
        # Winitzki approximation (adequate for CI z-scores when scipy absent).
        a = 0.147
        ln = math.log(1 - y * y)
        t1 = 2 / (math.pi * a) + ln / 2
        return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)
