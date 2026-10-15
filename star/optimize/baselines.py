"""
Baselines the recommended portfolio has to beat before anyone should pay for it.

Equal weight is a famously hard benchmark for optimised portfolios (DeMiguel,
Garlappi & Uppal 2009): estimation error in expected returns often costs more
than optimisation gains. If the mandate portfolio cannot be shown to beat 1/N,
the brief says so - that is rule `ALPHA-01` in the guidance engine, and it is one
of the reasons this rebuild exists.

Three alternatives are provided, none of which needs an expected-return estimate:

* **equal weight** - the null hypothesis of portfolio construction;
* **equal risk contribution** (Maillard, Roncalli & Teiletche 2010) - every name
  contributes the same share of portfolio variance, solved through the convex
  log-barrier form (Spinu 2013) so the answer is unique;
* **hierarchical risk parity** (Lopez de Prado 2016) - clusters correlated names,
  then allocates between and within clusters by inverse variance. It never
  inverts the covariance matrix, which is why it is stable when assets outnumber
  observations.

ERC and HRP ignore the mandate's box and sector caps by construction, so their
weights are **checked** against the mandate and any breach is reported rather
than clipped away. They are diagnostics and comparisons, not the shipped
portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from ..core.errors import OptimizationError
from ..core.types import CovarianceEstimate, Provenance
from .mean_variance import Portfolio


def _annual(cov: CovarianceEstimate, tickers=None):
    m = cov.annualise().matrix
    if tickers is not None:
        m = m.reindex(index=list(tickers), columns=list(tickers))
    if m.isna().to_numpy().any():
        raise OptimizationError("covariance has missing entries for the requested names")
    return m


def _wrap(w: pd.Series, objective: str, method: str, cov: CovarianceEstimate) -> Portfolio:
    return Portfolio(weights=w.rename("weight"), objective=objective,
                     provenance=Provenance(method=method, window=f"{len(w)} names",
                                           as_of=cov.provenance.as_of, source=cov.provenance.method),
                     diagnostics={"mandate_aware": False})


def equal_weight(cov: CovarianceEstimate, tickers=None) -> Portfolio:
    names = list(tickers) if tickers is not None else list(cov.tickers)
    if not names:
        raise OptimizationError("equal weight needs at least one name")
    return _wrap(pd.Series(1.0 / len(names), index=names), "equal_weight",
                 "equal weight (1/N) - the benchmark optimisation must beat", cov)


def risk_contributions(weights: pd.Series, cov: CovarianceEstimate) -> pd.Series:
    """Euler decomposition: each name's share of portfolio variance. Sums to 1."""
    S = _annual(cov, weights.index).to_numpy()
    w = weights.to_numpy()
    var = float(w @ S @ w)
    if var <= 0:
        raise OptimizationError("portfolio variance is not positive")
    return pd.Series(w * (S @ w) / var, index=weights.index, name="risk_contribution")


def equal_risk_contribution(cov: CovarianceEstimate, tickers=None, *, tol: float = 1e-12) -> Portfolio:
    """
    Every holding contributes the same share of risk.

    Solved in the convex form  min 1/2 x' Sigma x - (1/n) sum log x_i  over x > 0,
    whose unique solution has equal risk contributions once rescaled to sum to one.
    Solving the equal-contribution equations directly is non-convex and can land on
    the wrong root.

    Minimised by cyclical coordinate descent (Griveau-Billion, Richard & Roncalli
    2013): each coordinate has a closed-form update - the positive root of
    sigma_ii x_i^2 + (Sigma x)_{-i} x_i - 1/n = 0 - so the answer reaches machine
    precision instead of stopping at a general-purpose optimiser's tolerance.
    """
    S = _annual(cov, tickers).to_numpy()
    names = list(tickers) if tickers is not None else list(cov.tickers)
    n = len(names)
    if n < 2:
        raise OptimizationError("equal risk contribution needs at least two names")
    vol = np.sqrt(np.diag(S))
    if not np.all(vol > 0):
        raise OptimizationError("a zero-variance name cannot take an equal share of risk")
    x = (1.0 / vol) / np.sum(1.0 / vol)
    for _ in range(10_000):
        prev = x.copy()
        for i in range(n):
            beta = float(S[i] @ x) - S[i, i] * x[i]
            x[i] = (-beta + np.sqrt(beta ** 2 + 4.0 * S[i, i] / n)) / (2.0 * S[i, i])
        if np.max(np.abs(x - prev)) < tol * max(np.max(x), 1e-12):
            break
    w = pd.Series(x / x.sum(), index=names)
    rc = risk_contributions(w, cov)
    spread = float(rc.max() - rc.min())
    if spread > 1e-9:
        raise OptimizationError(f"equal risk contribution did not converge (spread {spread:.2e})")
    return _wrap(w, "equal_risk_contribution",
                 "equal risk contribution (Maillard et al. 2010, convex form)", cov)


def _quasi_diagonal_order(link: np.ndarray, n: int) -> list[int]:
    """Leaf order from the linkage tree: correlated names end up adjacent."""
    tree = to_tree(link, rd=False)
    return list(tree.pre_order(lambda x: x.id))


def hierarchical_risk_parity(cov: CovarianceEstimate, tickers=None) -> Portfolio:
    """
    Lopez de Prado (2016): cluster, order, then split capital by inverse variance
    down the tree. No matrix inversion, so it stays sane when the covariance is
    poorly conditioned.
    """
    m = _annual(cov, tickers)
    names = list(m.columns)
    n = len(names)
    if n < 2:
        raise OptimizationError("hierarchical risk parity needs at least two names")
    S = m.to_numpy()
    sd = np.sqrt(np.diag(S))
    corr = np.clip(S / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diagonal_order(link, n)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            var_l = _cluster_variance(S, left)
            var_r = _cluster_variance(S, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    out = pd.Series(w / w.sum(), index=names)
    return _wrap(out, "hierarchical_risk_parity",
                 "hierarchical risk parity (Lopez de Prado 2016, single linkage)", cov)


def _cluster_variance(S: np.ndarray, idx: list[int]) -> float:
    sub = S[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    iv = iv / iv.sum()
    return float(iv @ sub @ iv)
