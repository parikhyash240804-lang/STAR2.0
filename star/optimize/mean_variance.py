"""
Mean-variance construction, with the three things that make it usable in
practice: constraints the solver actually receives, costs of trading priced into
the objective, and controls on how far the portfolio is allowed to move.

    maximise   mu' w  -  (delta / 2) w' Sigma w  -  lambda * sum |w - w_prev|

Legacy defects addressed (docs/AUDIT_legacy.md):

* **M3-01** - maximising the Sharpe ratio on unshrunk sample means is the most
  noise-sensitive thing one can do with the least reliable input. The default
  here is maximum utility on Black-Litterman inputs, at a stated risk aversion.
* **M3-02 / M3-06** - caps enforced by clipping and renormalising, and a maximum
  holdings limit that was never enforced. Constraints now go to the solver and
  are verified on the result.
* **Nothing at all** on turnover: the legacy re-derived weights from scratch each
  period, so small changes in the inputs produced large reallocations. Three
  controls address it, all in weight space where they are visible and tunable:
  a trading penalty in the objective, a no-trade band, and smoothing towards the
  current portfolio. Damping the *return estimate* to get the same effect is
  rejected: it hides a forecast bias that nobody can inspect.

Returns and covariance must be annual and excess; the risk-free asset has no
variance, so mixing bases here would silently distort the risk trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..core.errors import OptimizationError
from ..core.types import Basis, CovarianceEstimate, Provenance, ReturnEstimate
from .constraints import ConstraintSet
from .qp import solve_qp


@dataclass(frozen=True, slots=True)
class OptimiserSettings:
    """How far the portfolio may move, and what that movement is assumed to cost."""

    turnover_penalty: float = 0.0     # cost of one unit of one-way turnover, in annual return units
    no_trade_band: float = 0.0        # ignore proposed moves smaller than this
    smoothing: float = 1.0            # 1.0 takes the target in full; 0.3 moves 30% of the way
    min_position: float = 0.0         # positions below this are not worth holding (applied by cardinality)

    def __post_init__(self) -> None:
        if self.turnover_penalty < 0:
            raise OptimizationError("turnover penalty cannot be negative")
        if not 0.0 <= self.smoothing <= 1.0:
            raise OptimizationError(f"smoothing must be in [0, 1], got {self.smoothing}")
        if self.no_trade_band < 0:
            raise OptimizationError("no-trade band cannot be negative")


@dataclass(frozen=True)
class Portfolio:
    """Solved weights, what they imply, and how they were produced."""

    weights: pd.Series
    objective: str
    provenance: Provenance
    prev_weights: pd.Series | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.index)

    def trades(self) -> pd.Series:
        if self.prev_weights is None:
            return self.weights.rename("trade")
        return (self.weights - self.prev_weights.reindex(self.weights.index).fillna(0.0)).rename("trade")

    def turnover(self) -> float:
        """One-way turnover: the share of the book that changes hands."""
        return float(np.abs(self.trades().to_numpy()).sum() / 2.0) if self.prev_weights is not None else 1.0

    def holdings(self, tol: float = 1e-6) -> list[str]:
        """
        Names held at `tol` or more. The comparison is tolerance-aware: a position
        placed exactly on a floor by the solver is held, and a strict `>` would
        report it as absent by a rounding error.
        """
        return [t for t, w in self.weights.items() if w >= tol - 1e-9]

    def expected_return(self, mu: ReturnEstimate) -> float:
        m = mu.annualise().values.reindex(self.weights.index)
        return float(self.weights.to_numpy() @ m.to_numpy())

    def volatility(self, cov: CovarianceEstimate) -> float:
        c = cov.annualise().matrix.reindex(index=self.weights.index, columns=self.weights.index)
        return float(np.sqrt(self.weights.to_numpy() @ c.to_numpy() @ self.weights.to_numpy()))


def _annual_inputs(mu: ReturnEstimate | None, cov: CovarianceEstimate, names: list[str]):
    c = cov.annualise().matrix.reindex(index=names, columns=names)
    if c.isna().to_numpy().any():
        raise OptimizationError(f"covariance is missing entries for some of: {names[:5]}")
    m = None
    if mu is not None:
        if mu.basis is not Basis.EXCESS:
            raise OptimizationError("the optimiser works in excess returns; convert with .to_excess(rf)")
        m = mu.annualise().values.reindex(names)
        if m.isna().any():
            raise OptimizationError(f"expected returns missing for: {list(m.index[m.isna()][:5])}")
        m = m.to_numpy()
    return m, c.to_numpy()


def _matrices(cs: ConstraintSet, H_w: np.ndarray, c_w: np.ndarray, *,
              turnover_penalty: float, extra_ub: tuple | None = None):
    """
    Assemble the QP. When turnover is priced or capped, |w - w_prev| enters through
    auxiliary variables t >= |w - w_prev| - the standard exact linearisation, not a
    smooth approximation of an absolute value.
    """
    n = cs.n
    use_t = turnover_penalty > 0 or cs.max_turnover is not None
    lb, ub = cs.lower.to_numpy().copy(), cs.upper.to_numpy().copy()
    A_ub, b_ub = [], []

    groups = cs.sector_groups()
    for _, idx in groups.items():
        row = np.zeros(n)
        row[idx] = 1.0
        A_ub.append(row)
        b_ub.append(cs.max_sector_weight)
    if extra_ub is not None:
        for row, rhs in zip(*extra_ub):
            A_ub.append(np.asarray(row, dtype=float))
            b_ub.append(float(rhs))

    if not use_t:
        A_ub_m = np.array(A_ub) if A_ub else None
        return (H_w, c_w, lb, ub, np.ones((1, n)), np.array([cs.budget]),
                A_ub_m, np.array(b_ub) if A_ub else None, n)

    w0 = cs.prev_weights.to_numpy() if cs.prev_weights is not None else np.zeros(n)
    N = 2 * n
    H = np.zeros((N, N))
    H[:n, :n] = H_w
    c = np.concatenate([c_w, np.full(n, float(turnover_penalty))])
    lb = np.concatenate([lb, np.zeros(n)])
    ub = np.concatenate([ub, np.full(n, float(np.max(cs.upper.to_numpy()) + np.max(np.abs(w0)) + 1.0))])
    rows = [np.concatenate([r, np.zeros(n)]) for r in A_ub]
    rhs = list(b_ub)
    eye = np.eye(n)
    rows += [np.concatenate([eye[i], -eye[i]]) for i in range(n)]      # w - t <= w0
    rhs += list(w0)
    rows += [np.concatenate([-eye[i], -eye[i]]) for i in range(n)]     # -w - t <= -w0
    rhs += list(-w0)
    if cs.max_turnover is not None:
        rows.append(np.concatenate([np.zeros(n), np.ones(n)]))
        rhs.append(2.0 * float(cs.max_turnover))          # the limit is one-way; sum|dw| is two-way
    A_eq = np.zeros((1, N))
    A_eq[0, :n] = 1.0
    return H, c, lb, ub, A_eq, np.array([cs.budget]), np.array(rows), np.array(rhs), n


def _solve(cs: ConstraintSet, H_w, c_w, *, turnover_penalty=0.0, extra_ub=None, what="portfolio"):
    problems = cs.infeasibilities()
    if problems:
        raise OptimizationError(f"the mandate cannot be satisfied as stated: {'; '.join(problems)}")
    H, c, lb, ub, A_eq, b_eq, A_ub, b_ub, n = _matrices(cs, H_w, c_w,
                                                        turnover_penalty=turnover_penalty,
                                                        extra_ub=extra_ub)
    sol = solve_qp(H, c, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
    w = pd.Series(sol.x[:n], index=list(cs.tickers), name="weight")
    cs.assert_satisfied(w, what=what)
    return w, sol


def _apply_band(cs: ConstraintSet, w: pd.Series, band: float):
    """Names that barely move are held still, and the rest re-solved around them."""
    if band <= 0 or cs.prev_weights is None:
        return w, cs, 0
    w0 = cs.prev_weights.reindex(w.index).fillna(0.0)
    small = (w - w0).abs() < band
    if not small.any() or small.all():
        return w, cs, int(small.sum())
    return None, cs.pinned(w0[small]), int(small.sum())


def optimise(mu: ReturnEstimate, cov: CovarianceEstimate, constraints: ConstraintSet, *,
             risk_aversion: float, settings: OptimiserSettings = OptimiserSettings()) -> Portfolio:
    """
    Maximum utility under the mandate, with turnover priced and damped.

    The order is deliberate: solve, then hold still the names that barely moved
    and re-solve around them, then optionally take only part of the step. Each
    stage ends with the constraints verified, so no stage can restore the budget
    by breaking a cap.
    """
    names = list(constraints.tickers)
    m, S = _annual_inputs(mu, cov, names)
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise OptimizationError(f"risk aversion must be positive, got {risk_aversion}")

    H_w, c_w = risk_aversion * S, -m
    w, sol = _solve(constraints, H_w, c_w, turnover_penalty=settings.turnover_penalty,
                    what="optimised weights")

    active = constraints
    banded, cs2, pinned = _apply_band(constraints, w, settings.no_trade_band)
    if banded is None:
        active = cs2
        w, sol = _solve(cs2, H_w, c_w, turnover_penalty=settings.turnover_penalty,
                        what="weights after the no-trade band")

    smoothed, smoothing_floor = False, False
    if settings.smoothing < 1.0 and constraints.prev_weights is not None:
        # Smoothing is a *trading budget*, not a blend. Blending with the current
        # portfolio would inherit its breaches - and a portfolio is usually being
        # rebalanced precisely because it is out of compliance.
        w0 = constraints.prev_weights.reindex(names).fillna(0.0)
        full_move = float(np.abs(w.to_numpy() - w0.to_numpy()).sum()) / 2.0
        allowed = settings.smoothing * full_move
        floor = active.min_turnover_required()
        if floor > allowed:
            allowed, smoothing_floor = floor * 1.0000001 + 1e-9, True
        if allowed < full_move - 1e-12:
            capped = replace(active, max_turnover=min(allowed, active.max_turnover)
                             if active.max_turnover is not None else allowed)
            w, sol = _solve(capped, H_w, c_w, turnover_penalty=settings.turnover_penalty,
                            what="weights after smoothing")
            active = capped
            smoothed = True

    at_cap = [t for t in names if w[t] > constraints.upper[t] - 1e-7]
    sector_at_cap = []
    if constraints.sectors is not None:
        for sec, idx in constraints.sector_groups().items():
            if float(w.iloc[idx].sum()) > constraints.max_sector_weight - 1e-7:
                sector_at_cap.append(sec)
    w0 = None if constraints.prev_weights is None else constraints.prev_weights.reindex(names).fillna(0.0)
    turnover = None if w0 is None else float(np.abs(w.to_numpy() - w0.to_numpy()).sum() / 2.0)

    prov = Provenance(
        method=f"maximum utility (delta {risk_aversion:.2f}) under mandate constraints",
        window=(f"{len(names)} names; solver {sol.solver}; "
                f"turnover penalty {settings.turnover_penalty:g}, band {settings.no_trade_band:g}, "
                f"smoothing {settings.smoothing:g}"),
        as_of=cov.provenance.as_of,
        source=mu.provenance.method,
        degradations=mu.provenance.degradations + ((sol.note,) if sol.note else ()),
    )
    return Portfolio(weights=w, objective="max_utility", provenance=prov, prev_weights=w0,
                     diagnostics={"solver": sol.solver, "iterations": sol.iterations,
                                  "max_constraint_violation": sol.max_constraint_violation,
                                  "names_at_cap": at_cap, "sectors_at_cap": sector_at_cap,
                                  "names_held_by_band": pinned, "smoothed": smoothed,
                                  "smoothing_limited_by_compliance": smoothing_floor,
                                  "turnover": turnover,
                                  "binding_cap": None if constraints.binding_cap is None
                                  else constraints.binding_cap[at_cap].to_dict()})


def min_variance(cov: CovarianceEstimate, constraints: ConstraintSet) -> Portfolio:
    """The lowest-risk portfolio the mandate allows. A baseline, and the frontier's left end."""
    names = list(constraints.tickers)
    _, S = _annual_inputs(None, cov, names)
    w, sol = _solve(constraints, S, np.zeros(len(names)), what="minimum-variance weights")
    prov = Provenance(method="minimum variance under mandate constraints",
                      window=f"{len(names)} names; solver {sol.solver}",
                      as_of=cov.provenance.as_of, source=cov.provenance.method)
    return Portfolio(weights=w, objective="min_variance", provenance=prov,
                     prev_weights=constraints.prev_weights,
                     diagnostics={"solver": sol.solver, "volatility": None})


def target_return(mu: ReturnEstimate, cov: CovarianceEstimate, constraints: ConstraintSet,
                  target: float) -> Portfolio:
    """
    Least risk for at least `target` expected excess return - the frontier point,
    solved in its own right rather than interpolated between two other portfolios.
    """
    names = list(constraints.tickers)
    m, S = _annual_inputs(mu, cov, names)
    w, sol = _solve(constraints, S, np.zeros(len(names)),
                    extra_ub=([-m], [-float(target)]), what="frontier weights")
    prov = Provenance(method=f"minimum variance at expected excess return >= {target:.2%}",
                      window=f"{len(names)} names; solver {sol.solver}",
                      as_of=cov.provenance.as_of, source=mu.provenance.method)
    return Portfolio(weights=w, objective="target_return", provenance=prov,
                     prev_weights=constraints.prev_weights,
                     diagnostics={"solver": sol.solver, "target_return": float(target)})
