"""
"What if Financials fall 20%?" - with the rest of the book responding.

A scenario that shocks three holdings and leaves the other thirty-seven at zero
is not a scenario; it is an arithmetic exercise. Holdings move together, and the
covariance matrix already says by how much. The conditional expectation of the
unshocked names given the shocked ones is

    E[r_u | r_s = f] = Sigma_us Sigma_ss^-1 f

which is the same algebra as a regression of each unshocked name on the shocked
ones. A shock to Financials therefore reaches Industrials through their measured
co-movement and barely touches an uncorrelated name, without anybody specifying
that relationship by hand.

This is also what makes the custom-scenario control in the brief possible: the
whole calculation is one small matrix multiplied by the shock vector, so an
arbitrary shock can be recomputed in the page, offline, in a fraction of a
millisecond.

Two honest limits. The conditional expectation is a *mean*, not a worst case - it
answers "what typically happens to the rest", not "what could happen". And it
uses normal-times co-movement; in a crisis correlations rise, so a shock of this
kind understates contagion. Where the scenario is a real crisis, use the
historical replay instead, which needs no such assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.types import CovarianceEstimate, Provenance


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    portfolio_return: float
    returns: pd.Series               # every holding, shocked and implied
    shocked: tuple[str, ...]
    contributions: pd.Series
    provenance: Provenance

    @property
    def headline(self) -> str:
        direct = float(self.contributions[list(self.shocked)].sum())
        spill = self.portfolio_return - direct
        return (f"{self.name}: {self.portfolio_return:+.1%} overall - {direct:+.1%} from the names "
                f"shocked directly and {spill:+.1%} from everything that moves with them.")


def conditional_returns(cov: CovarianceEstimate, shocks: dict[str, float], *,
                        tickers=None, ridge: float = 1e-10) -> pd.Series:
    """
    Expected return of every name given the shocked ones move as specified.

    Shocks are simple returns over the scenario's (unstated) horizon; the same
    horizon applies to the implied moves, since covariance enters only through
    ratios.
    """
    names = list(tickers) if tickers is not None else list(cov.tickers)
    shocked = [t for t in shocks]
    missing = [t for t in shocked if t not in names]
    if missing:
        raise EstimationError(f"shocked names are not in the universe: {missing[:5]}")
    if not shocked:
        raise EstimationError("a scenario must shock at least one name")
    rest = [t for t in names if t not in shocks]
    S = cov.annualise().matrix.reindex(index=names, columns=names)
    if S.isna().to_numpy().any():
        raise EstimationError("covariance is missing entries for the requested names")

    f = np.array([float(shocks[t]) for t in shocked])
    if not np.all(np.isfinite(f)):
        raise EstimationError("shock sizes must be finite")
    out = pd.Series(0.0, index=names, dtype=float)
    out[shocked] = f
    if rest:
        Sss = S.loc[shocked, shocked].to_numpy()
        Sus = S.loc[rest, shocked].to_numpy()
        Sss = Sss + ridge * np.eye(len(shocked)) * max(np.trace(Sss) / len(shocked), 1e-12)
        out[rest] = Sus @ np.linalg.solve(Sss, f)
    return out.rename("scenario_return")


def apply_scenario(weights: pd.Series, cov: CovarianceEstimate, shocks: dict[str, float], *,
                   name: str = "custom scenario") -> ScenarioResult:
    """Portfolio impact of a shock, including everything that moves with it."""
    w = pd.Series(weights, dtype=float)
    r = conditional_returns(cov, shocks, tickers=list(w.index))
    contributions = (w * r).rename("contribution")
    prov = Provenance(
        method="conditional scenario E[r_unshocked | r_shocked] from the covariance matrix",
        window=f"{len(shocks)} name(s) shocked directly, {len(w) - len(shocks)} implied",
        as_of=cov.provenance.as_of, source=cov.provenance.method,
        degradations=("uses normal-times co-movement; correlations rise in a crisis, so contagion "
                      "here is a lower bound",))
    return ScenarioResult(name=name, portfolio_return=float(contributions.sum()), returns=r,
                          shocked=tuple(shocks), contributions=contributions, provenance=prov)


def sector_shock(weights: pd.Series, cov: CovarianceEstimate, sectors: pd.Series,
                 sector: str, move: float, *, name: str | None = None) -> ScenarioResult:
    """
    Shock a whole sector at once - the form a manager actually states a view in.

    Every holding in the sector moves by `move`; everything else responds through
    its co-movement with them.
    """
    members = [t for t in weights.index if str(sectors.get(t, "")) == str(sector)]
    if not members:
        raise EstimationError(f"no holding is in sector {sector!r}")
    return apply_scenario(weights, cov, {t: float(move) for t in members},
                          name=name or f"{sector} {move:+.0%}")
