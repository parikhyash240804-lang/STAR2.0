"""
Black-Litterman posterior: the equilibrium prior, moved by views, each view
moved only as far as its stated confidence allows.

    M       = [ (tau Sigma)^-1 + P' Omega^-1 P ]^-1
    mu_BL   = M [ (tau Sigma)^-1 Pi + P' Omega^-1 q ]
    Sigma_BL = Sigma + M

Four legacy defects are structurally impossible here (all verified in
docs/AUDIT_legacy.md):

* **M2B-11** - Omega proportional to tau*Sigma makes tau cancel, so the "view
  weight" control did nothing and views carried ~91% of the posterior instead of
  the advertised 40%. Here Omega is *solved for* from a stated confidence
  (Idzorek 2005), so 50% confidence really does move the posterior half way.
* **M2B-01** - an absolute, bullish-only view on every asset is a market call and
  a stock-selection call mixed together. Default views are **relative**.
* **M2B-05** - mapping a score onto a fixed return band ignores how predictive
  the signal is; view sizes come from Grinold (see `views.py`).
* **M2-01** - the posterior is an EXCESS return and carries that in its type, so
  subtracting the risk-free rate twice raises rather than silently mis-stating.

The posterior **covariance** Sigma + M is kept, not discarded: it is the estimate
that tells the optimiser how uncertain these returns are.

See PAPERS/notes/02_black_litterman.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..core.errors import EstimationError
from ..core.types import Basis, Compounding, CovarianceEstimate, Periodicity, Provenance, ReturnEstimate

DEFAULT_TAU = 0.05


@dataclass(frozen=True)
class View:
    """
    One statement about returns.

    `picks` maps ticker -> coefficient. A relative view sums to zero
    ("A beats B by 3%"); an absolute view has a single positive coefficient
    ("A returns 8%"). `q` is the expected **annual excess** return of that
    combination. `confidence` is Idzorek's 0-1 scale: the fraction of the way to
    the fully-certain tilt that the posterior should move. None means classical
    Black-Litterman uncertainty, Omega = diag(P tau Sigma P').
    """

    name: str
    picks: dict[str, float]
    q: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.picks:
            raise EstimationError(f"view '{self.name}' picks no assets")
        if not np.isfinite(self.q):
            raise EstimationError(f"view '{self.name}' has a non-finite size")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise EstimationError(f"view '{self.name}': confidence must be in [0, 1], got {self.confidence}")
        if not np.isfinite(list(self.picks.values())).all():
            raise EstimationError(f"view '{self.name}' has a non-finite coefficient")

    @property
    def is_relative(self) -> bool:
        return abs(sum(self.picks.values())) < 1e-9


@dataclass(frozen=True)
class BLResult:
    mu: ReturnEstimate                  # posterior, ANNUAL SIMPLE EXCESS
    covariance: CovarianceEstimate      # Sigma + M, annual
    prior: ReturnEstimate
    P: pd.DataFrame
    q: pd.Series
    omega: pd.DataFrame
    tau: float
    delta: float
    view_shift: pd.Series = field(default_factory=pd.Series)   # posterior - prior, per asset

    def active_view_weight(self) -> float:
        """How much of the posterior came from views, as a share of the total move."""
        denom = float(np.abs(self.mu.values - self.prior.values).sum())
        return 0.0 if denom == 0 else float(np.abs(self.view_shift).sum() / denom)


def build_view_matrices(views: list[View], tickers) -> tuple[pd.DataFrame, pd.Series]:
    """P (views x assets) and q (views), in the asset order given."""
    names = list(tickers)
    if not views:
        return pd.DataFrame(index=pd.Index([], name="view"), columns=names, dtype=float), pd.Series(dtype=float)
    seen = set()
    for v in views:
        if v.name in seen:
            raise EstimationError(f"two views share the name '{v.name}'")
        seen.add(v.name)
        missing = [t for t in v.picks if t not in names]
        if missing:
            raise EstimationError(f"view '{v.name}' names assets outside the universe: {missing[:5]}")
    P = pd.DataFrame(0.0, index=pd.Index([v.name for v in views], name="view"), columns=names)
    for v in views:
        for t, c in v.picks.items():
            P.loc[v.name, t] += float(c)
    q = pd.Series([float(v.q) for v in views], index=P.index, name="q")
    return P, q


def _posterior_arrays(sigma: np.ndarray, pi: np.ndarray, P: np.ndarray, q: np.ndarray,
                      omega: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu, M). Solved rather than inverted where it matters numerically."""
    ts_inv = np.linalg.inv(tau * sigma)
    if P.size == 0:
        M = np.linalg.inv(ts_inv)
        return pi.copy(), M
    om_inv = np.linalg.inv(omega)
    M = np.linalg.inv(ts_inv + P.T @ om_inv @ P)
    mu = M @ (ts_inv @ pi + P.T @ om_inv @ q)
    return mu, M


def _certain_tilt(sigma: np.ndarray, pi: np.ndarray, p_row: np.ndarray, q_k: float,
                  tau: float, delta: float) -> np.ndarray:
    """
    Weights implied by treating one view as certain (Idzorek's w_100).

    mu_100 = Pi + tau*Sigma*P' (P tau Sigma P')^-1 (q - P Pi), the limit of the
    posterior as Omega -> 0 for that view.
    """
    ts = tau * sigma
    denom = float(p_row @ ts @ p_row)              # numpy 2 refuses float() on a (1,1) array
    if denom <= 0:
        raise EstimationError("view portfolio has no variance; it cannot be informative")
    mu_100 = pi + (ts @ p_row) * (q_k - float(p_row @ pi)) / denom
    return np.linalg.solve(delta * sigma, mu_100)


def idzorek_omega(sigma: np.ndarray, pi: np.ndarray, P: np.ndarray, q: np.ndarray,
                  confidences, tau: float, delta: float) -> np.ndarray:
    """
    Omega calibrated so that stated confidence means what it says (Idzorek 2005).

    For each view k, confidence c_k should move the unconstrained weights c_k of
    the way from equilibrium to the tilt that a *certain* view would produce.
    Each omega_k is found by a one-dimensional search, view by view, as Idzorek
    prescribes - there is no closed form for the exact version.
    """
    w_ref = np.linalg.solve(delta * sigma, pi)
    omegas = np.empty(len(q))
    for k, c in enumerate(confidences):
        p_row = P[k]
        scale = float(p_row @ (tau * sigma) @ p_row)
        if c is None:
            omegas[k] = scale                      # classical Black-Litterman
            continue
        if c <= 1e-9:
            omegas[k] = scale * 1e12               # a view with no confidence must not move anything
            continue
        if c >= 1.0 - 1e-9:
            omegas[k] = scale * 1e-10
            continue
        target = w_ref + c * (_certain_tilt(sigma, pi, p_row, float(q[k]), tau, delta) - w_ref)

        def distance(log_omega, p_row=p_row, k=k, target=target):
            om = np.array([[np.exp(log_omega)]])
            mu, _ = _posterior_arrays(sigma, pi, p_row.reshape(1, -1), np.array([q[k]]), om, tau)
            w = np.linalg.solve(delta * sigma, mu)
            return float(np.sum((w - target) ** 2))

        base = np.log(max(scale, 1e-300))
        best = minimize_scalar(distance, bounds=(base - 25.0, base + 25.0), method="bounded",
                               options={"xatol": 1e-10})
        omegas[k] = float(np.exp(best.x))
    return np.diag(omegas)


def black_litterman(prior: ReturnEstimate, cov: CovarianceEstimate, views: list[View], *,
                    delta: float, tau: float = DEFAULT_TAU) -> BLResult:
    """
    Posterior expected returns and covariance.

    `prior` must be an EXCESS estimate (use `equilibrium_prior`); `cov` is
    annualised internally. With no views the posterior returns the prior exactly,
    which is the property that makes "no opinion" a safe default.
    """
    if prior.basis is not Basis.EXCESS:
        raise EstimationError("Black-Litterman works in excess returns; the prior is not EXCESS")
    if not 0 < tau <= 1:
        raise EstimationError(f"tau must be in (0, 1], got {tau}")
    if not np.isfinite(delta) or delta <= 0:
        raise EstimationError(f"delta must be positive, got {delta}")

    annual_cov = cov.annualise()
    names = list(annual_cov.tickers)
    pi_series = prior.annualise().values.reindex(names)
    if pi_series.isna().any():
        raise EstimationError(f"prior missing for: {list(pi_series.index[pi_series.isna()][:8])}")

    P_df, q_s = build_view_matrices(views, names)
    sigma = annual_cov.matrix.to_numpy()
    pi = pi_series.to_numpy()
    P = P_df.to_numpy() if len(views) else np.zeros((0, len(names)))
    q = q_s.to_numpy() if len(views) else np.zeros(0)

    if len(views):
        omega = idzorek_omega(sigma, pi, P, q, [v.confidence for v in views], tau, delta)
    else:
        omega = np.zeros((0, 0))
    mu, M = _posterior_arrays(sigma, pi, P, q, omega, tau)

    stated = ", ".join(f"{v.name} {'rel' if v.is_relative else 'abs'} "
                       f"{v.q:+.2%}@{'BL' if v.confidence is None else f'{v.confidence:.0%}'}"
                       for v in views) or "none"
    prov = Provenance(
        method="Black-Litterman posterior with Idzorek-calibrated Omega",
        window=f"tau {tau:g}, delta {delta:.2f}, {len(views)} view(s): {stated}",
        as_of=annual_cov.provenance.as_of,
        source=prior.provenance.method,
        degradations=prior.provenance.degradations,
    )
    mu_est = ReturnEstimate(values=pd.Series(mu, index=names), basis=Basis.EXCESS,
                            compounding=Compounding.SIMPLE, periodicity=Periodicity.ANNUAL,
                            provenance=prov, periods_per_year=cov.periods_per_year)
    post_cov = CovarianceEstimate(matrix=pd.DataFrame(sigma + M, index=names, columns=names),
                                  periodicity=Periodicity.ANNUAL, provenance=prov,
                                  periods_per_year=cov.periods_per_year,
                                  vol_method=annual_cov.vol_method,
                                  corr_method=f"{annual_cov.corr_method}+bl_parameter_risk")
    return BLResult(mu=mu_est, covariance=post_cov, prior=prior.annualise(), P=P_df, q=q_s,
                    omega=pd.DataFrame(omega, index=P_df.index, columns=P_df.index),
                    tau=tau, delta=delta,
                    view_shift=pd.Series(mu - pi, index=names, name="view_shift"))
