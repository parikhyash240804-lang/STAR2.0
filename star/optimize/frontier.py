"""
The efficient frontier, solved point by point.

The legacy drew a "frontier" by interpolating between two solved portfolios,
which is not a frontier: with box and sector constraints the efficient set is
piecewise, and the straight line between two efficient portfolios is generally
**infeasible or inefficient**. Every point here is its own constrained solve.

The maximum-Sharpe point is found the same way - by searching solved points and
refining between the best two - rather than by the tangency formula, which is
only valid when nothing binds. Under a mandate, something almost always binds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..core.errors import OptimizationError
from ..core.types import CovarianceEstimate, Provenance, ReturnEstimate
from .constraints import ConstraintSet
from .mean_variance import Portfolio, _annual_inputs, _solve, min_variance, target_return


@dataclass(frozen=True)
class Frontier:
    points: pd.DataFrame            # expected_return, volatility, sharpe, per solved point
    weights: pd.DataFrame           # points x tickers
    provenance: Provenance

    def max_sharpe(self) -> pd.Series:
        return self.points.loc[self.points["sharpe"].idxmax()]

    def max_sharpe_weights(self) -> pd.Series:
        return self.weights.loc[self.points["sharpe"].idxmax()].rename("weight")

    def at_volatility(self, target: float) -> pd.Series:
        """Weights of the nearest solved point - never an interpolation between two."""
        idx = (self.points["volatility"] - float(target)).abs().idxmin()
        return self.weights.loc[idx].rename("weight")


def max_return_portfolio(mu: ReturnEstimate, cov: CovarianceEstimate,
                         constraints: ConstraintSet) -> Portfolio:
    """The highest expected return the mandate allows: the frontier's right-hand end."""
    names = list(constraints.tickers)
    m, S = _annual_inputs(mu, cov, names)
    ridge = 1e-10 * float(np.trace(S)) / len(names)      # keeps the solve strictly convex
    w, sol = _solve(constraints, ridge * np.eye(len(names)), -m, what="maximum-return weights")
    prov = Provenance(method="maximum expected return under mandate constraints",
                      window=f"{len(names)} names; solver {sol.solver}",
                      as_of=cov.provenance.as_of, source=mu.provenance.method)
    return Portfolio(weights=w, objective="max_return", provenance=prov,
                     prev_weights=constraints.prev_weights, diagnostics={"solver": sol.solver})


def efficient_frontier(mu: ReturnEstimate, cov: CovarianceEstimate, constraints: ConstraintSet, *,
                       n_points: int = 15, refine_max_sharpe: bool = True) -> Frontier:
    """
    `n_points` constrained solves between the minimum-variance and maximum-return
    ends, plus an optional refinement around the best Sharpe ratio.
    """
    if n_points < 3:
        raise OptimizationError(f"a frontier needs at least 3 points, got {n_points}")
    lo, hi = min_variance(cov, constraints), max_return_portfolio(mu, cov, constraints)
    r_lo, r_hi = lo.expected_return(mu), hi.expected_return(mu)
    if r_hi <= r_lo + 1e-12:
        raise OptimizationError("expected returns are flat across the universe: no frontier to draw")

    rows, weights = [], []

    def add(label, w: pd.Series, target=None):
        r = float(w.to_numpy() @ mu.annualise().values.reindex(w.index).to_numpy())
        c = cov.annualise().matrix.reindex(index=w.index, columns=w.index).to_numpy()
        v = float(np.sqrt(w.to_numpy() @ c @ w.to_numpy()))
        rows.append({"label": label, "target_return": target, "expected_return": r,
                     "volatility": v, "sharpe": r / v if v > 0 else np.nan})
        weights.append(w)

    add("min_variance", lo.weights)
    for t in np.linspace(r_lo, r_hi, n_points)[1:-1]:
        add("target", target_return(mu, cov, constraints, float(t)).weights, float(t))
    add("max_return", hi.weights, r_hi)

    if refine_max_sharpe and n_points >= 4:
        pts = pd.DataFrame(rows)
        k = int(pts["sharpe"].idxmax())
        left = pts["expected_return"].iloc[max(k - 1, 0)]
        right = pts["expected_return"].iloc[min(k + 1, len(pts) - 1)]
        if right > left + 1e-12:
            def neg_sharpe(t):
                w = target_return(mu, cov, constraints, float(t)).weights
                c = cov.annualise().matrix.reindex(index=w.index, columns=w.index).to_numpy()
                v = float(np.sqrt(w.to_numpy() @ c @ w.to_numpy()))
                r = float(w.to_numpy() @ mu.annualise().values.reindex(w.index).to_numpy())
                return -r / v if v > 0 else np.inf
            best = minimize_scalar(neg_sharpe, bounds=(left, right), method="bounded",
                                   options={"xatol": 1e-6})
            add("max_sharpe", target_return(mu, cov, constraints, float(best.x)).weights, float(best.x))

    points = pd.DataFrame(rows)
    points.index.name = "point"
    prov = Provenance(
        method="constrained efficient frontier, each point solved (never interpolated)",
        window=f"{len(points)} solves from {r_lo:.2%} to {r_hi:.2%} expected excess return",
        as_of=cov.provenance.as_of, source=mu.provenance.method,
        degradations=mu.provenance.degradations)
    return Frontier(points=points,
                    weights=pd.DataFrame(weights, index=points.index).fillna(0.0),
                    provenance=prov)
