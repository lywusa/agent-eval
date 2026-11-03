"""Statistics used by the analyses. Bootstraps resample *questions* (clusters),
because seeds and configs on the same question are not independent."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

DEFAULT_B = 1000


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_bootstrap(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    cluster: str = "question_id",
    n_boot: int = DEFAULT_B,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of `stat`, resampling clusters with replacement."""
    df = df.reset_index(drop=True)
    point = stat(df)
    groups = list(df.groupby(cluster, sort=True).indices.values())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in draw])
        v = stat(df.iloc[idx])
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return point, math.nan, math.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return math.nan
    t = sps.kendalltau(a, b).statistic
    return float(t) if t is not None else math.nan


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return math.nan
    return float(sps.spearmanr(a[m], b[m]).statistic)


def spearman_ci(a: Sequence[float], b: Sequence[float], n_boot: int = DEFAULT_B, seed: int = 0) -> tuple[float, float, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    point = spearman(a, b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        v = spearman(a[idx], b[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return point, math.nan, math.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def cohen_kappa(a: Sequence, b: Sequence, weights: str | None = None) -> float:
    """Cohen's κ; weights='linear' or 'quadratic' for ordinal labels."""
    a, b = list(a), list(b)
    if not a:
        return math.nan
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k == 1:
        return 1.0
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights is None:
        w = 1 - np.eye(k)
    else:
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
        w = d if weights == "linear" else d**2
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else math.nan


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Paired exact McNemar test for two configs on identical items."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    p = float(sps.binomtest(n10, n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_only": n10, "b_only": n01, "p_value": p, "diff": float(a.mean() - b.mean())}


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Configs not dominated in (lower cost, higher accuracy)."""
    front = []
    for k, (c, a) in points.items():
        dominated = any(
            (c2 <= c and a2 >= a) and (c2 < c or a2 > a) for k2, (c2, a2) in points.items() if k2 != k
        )
        if not dominated:
            front.append(k)
    return sorted(front, key=lambda k: points[k][0])
