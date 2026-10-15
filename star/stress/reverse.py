"""
Reverse stress testing: what would it take to lose the amount you cannot afford?

Forward stress asks "what does 2008 cost us". Reverse stress asks the question a
risk committee actually needs answered: *our drawdown tolerance is 20% - what has
to happen for that to occur, and is it plausible?* Supervisors have pushed this
way round since 2018 (BCBS) precisely because the forward version only ever
explores scenarios somebody thought of.

The smallest move that causes a given loss has a closed form. Among all return
vectors r with w'r = -L, the one of least Mahalanobis distance is

    r* = -L * Sigma w / (w' Sigma w),      distance = L / sqrt(w' Sigma w)

so the answer comes out in units everybody already understands: **the loss
expressed in portfolio standard deviations**. A 20% loss on a book with 10%
volatility is a two-sigma move - uncomfortable but ordinary. The same loss on a
4% book is five sigma, which either will not happen or means the risk model is
wrong. Both readings are useful, and the second is the one worth saying out loud.

The scenario itself is also informative: r* is proportional to Sigma w, so it
names the holdings whose moves would do the damage - which are not the largest
positions but the largest *risk* positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import EstimationError
from ..core.types import CovarianceEstimate, Provenance


@dataclass(frozen=True)
class ReverseStress:
    target_loss: float
    horizon_days: int
    sigmas: float                     # how many portfolio standard deviations the loss is
    probability_if_normal: float      # indicative only; real tails are fatter
    scenario: pd.Series               # the least-unlikely return vector that does it
    contributions: pd.Series
    portfolio_volatility: float       # at the stated horizon
    provenance: Provenance

    @property
    def headline(self) -> str:
        worst = self.contributions.nsmallest(3)
        names = ", ".join(f"{t} {self.scenario[t]:+.0%}" for t in worst.index)
        period = "day" if self.horizon_days == 1 else f"{self.horizon_days} days"
        return (f"Losing {self.target_loss:.0%} over {period} takes a {self.sigmas:.1f}-sigma move "
                f"({names}).")

    @property
    def plausibility(self) -> str:
        if self.sigmas < 1.5:
            return "well within normal variation - this should be treated as likely, not as a stress"
        if self.sigmas < 3.0:
            return "an ordinary bad period; expect to see this"
        if self.sigmas < 5.0:
            return "a genuine crisis, of the kind that has happened several times in Indian markets"
        return ("beyond anything in the historical record - if the limit can only be breached by such "
                "a move, the risk model is the more likely explanation")


def reverse_stress(weights: pd.Series, cov: CovarianceEstimate, target_loss: float, *,
                   horizon_days: int = 252) -> ReverseStress:
    """
    The least-unlikely way to lose `target_loss` (0.20 for 20% of capital).

    `probability_if_normal` is reported for scale only, and is deliberately
    labelled: real return distributions have much fatter tails, so it understates
    how often such a move occurs - by a lot at four sigma and beyond.
    """
    w = pd.Series(weights, dtype=float)
    if target_loss <= 0:
        raise EstimationError(f"target loss must be positive (0.20 for 20%), got {target_loss}")
    if horizon_days < 1:
        raise EstimationError(f"horizon must be at least 1 day, got {horizon_days}")
    names = list(w.index)
    annual = cov.annualise()
    S = annual.matrix.reindex(index=names, columns=names)
    if S.isna().to_numpy().any():
        raise EstimationError("covariance is missing entries for some holdings")
    scale = horizon_days / float(annual.periods_per_year)
    S_h = S.to_numpy() * scale                                  # covariance over the horizon
    wv = w.to_numpy()
    Sw = S_h @ wv
    var = float(wv @ Sw)
    if var <= 0:
        raise EstimationError("portfolio variance is not positive")
    vol = float(np.sqrt(var))
    scenario = pd.Series(-target_loss * Sw / var, index=names, name="scenario_return")
    sigmas = float(target_loss / vol)
    prov = Provenance(
        method="reverse stress: minimum-Mahalanobis return vector achieving the target loss",
        window=f"{horizon_days}-day horizon; portfolio volatility {vol:.1%}",
        as_of=cov.provenance.as_of, source=cov.provenance.method,
        degradations=("the probability shown assumes normal returns and therefore understates how "
                      "often such moves happen",))
    return ReverseStress(target_loss=float(target_loss), horizon_days=int(horizon_days),
                         sigmas=sigmas, probability_if_normal=float(stats.norm.cdf(-sigmas)),
                         scenario=scenario, contributions=(w * scenario).rename("contribution"),
                         portfolio_volatility=vol, provenance=prov)


def loss_at_sigmas(weights: pd.Series, cov: CovarianceEstimate, sigmas: float,
                   horizon_days: int = 252) -> float:
    """The inverse question: what does a k-sigma move cost? Used for the limits table."""
    w = pd.Series(weights, dtype=float)
    names = list(w.index)
    annual = cov.annualise()
    S = annual.matrix.reindex(index=names, columns=names).to_numpy()
    scale = horizon_days / float(annual.periods_per_year)
    vol = float(np.sqrt(w.to_numpy() @ (S * scale) @ w.to_numpy()))
    return float(sigmas * vol)
