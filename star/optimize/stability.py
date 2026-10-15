"""
How much of this portfolio is signal, and how much is estimation noise?

Mean-variance weights are notoriously sensitive to expected returns. A portfolio
that changes completely when the return estimates are nudged inside their own
error bars is not a recommendation - it is one draw from a distribution of
defensible answers, and saying so is the honest output.

So the optimisation is re-run over draws of mu from its **own** uncertainty (the
Black-Litterman posterior covariance where available, otherwise tau * Sigma), and
the dispersion of the resulting weights is reported. This is what feeds guidance
rule `DATA-01`: when weights move more than the stated threshold, the brief says
"treat these as one of many defensible answers" instead of presenting three
decimal places of false precision.

Michaud (1989) made the point; this measures it for the portfolio actually
proposed, rather than citing it in general.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import OptimizationError
from ..core.types import CovarianceEstimate, ReturnEstimate
from .constraints import ConstraintSet
from .mean_variance import OptimiserSettings, Portfolio, optimise


@dataclass(frozen=True)
class StabilityReport:
    base_weights: pd.Series
    mean_abs_change: float          # average across draws of the mean |dw| per name
    max_name_change: float          # worst single-name move seen in any draw
    mean_turnover: float            # average one-way distance from the base portfolio
    draws: int
    dispersion: pd.Series           # per-name standard deviation of weight across draws
    headline: str

    def is_stable(self, threshold: float = 0.15) -> bool:
        return self.mean_turnover <= threshold


def perturbation_sensitivity(mu: ReturnEstimate, cov: CovarianceEstimate,
                             constraints: ConstraintSet, *, risk_aversion: float,
                             mu_uncertainty: CovarianceEstimate | None = None,
                             tau: float = 0.05, draws: int = 40, seed: int = 20260920,
                             settings: OptimiserSettings = OptimiserSettings(),
                             base: Portfolio | None = None) -> StabilityReport:
    """
    Re-solve under `draws` samples of mu drawn from its estimation uncertainty.

    Deterministic for a given seed, so the number in the report is reproducible -
    a run that cannot be repeated cannot be audited.
    """
    if draws < 2:
        raise OptimizationError(f"stability needs at least 2 draws, got {draws}")
    names = list(constraints.tickers)
    base = base or optimise(mu, cov, constraints, risk_aversion=risk_aversion, settings=settings)
    w0 = base.weights.reindex(names)

    annual = cov.annualise().matrix.reindex(index=names, columns=names).to_numpy()
    if mu_uncertainty is not None:
        sigma_mu = mu_uncertainty.annualise().matrix.reindex(index=names, columns=names).to_numpy()
    else:
        sigma_mu = tau * annual
    m = mu.annualise().values.reindex(names).to_numpy()
    rng = np.random.default_rng(seed)
    draws_w = []
    for _ in range(draws):
        shocked = mu.annualise()
        shocked = type(shocked)(values=pd.Series(rng.multivariate_normal(m, sigma_mu), index=names),
                                basis=shocked.basis, compounding=shocked.compounding,
                                periodicity=shocked.periodicity, provenance=shocked.provenance,
                                periods_per_year=shocked.periods_per_year)
        p = optimise(shocked, cov, constraints, risk_aversion=risk_aversion, settings=settings)
        constraints.assert_satisfied(p.weights, what="perturbed weights")
        draws_w.append(p.weights.reindex(names))

    W = pd.DataFrame(draws_w)
    diff = W.sub(w0, axis=1)
    mean_abs = float(diff.abs().mean(axis=1).mean())
    max_name = float(diff.abs().to_numpy().max())
    mean_turnover = float(diff.abs().sum(axis=1).mean() / 2.0)
    stable = mean_turnover <= 0.15
    headline = (f"Re-estimating expected returns inside their error bars moves "
                f"{mean_turnover:.0%} of the portfolio on average"
                + ("; the allocation is stable." if stable else
                   " - treat these weights as one of several defensible answers."))
    return StabilityReport(base_weights=w0, mean_abs_change=mean_abs, max_name_change=max_name,
                           mean_turnover=mean_turnover, draws=draws,
                           dispersion=W.std(ddof=1).rename("weight_sd"), headline=headline)
