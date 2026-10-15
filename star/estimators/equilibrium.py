"""
The equilibrium prior: what returns would have to be true for the reference
portfolio to be the right one.

Mean-variance optimisation is far more sensitive to expected returns than to
risk, and expected returns are the least reliably estimated input there is.
Rather than feed it sample means - which is what the legacy did, and which
produces a portfolio concentrated in whatever happened to go up (M3-01) - we
start from the returns implied by a reference portfolio and move away only as
far as the evidence justifies.

    Pi = delta * Sigma * w_ref        (annual, simple, EXCESS of the risk-free rate)

`delta` (risk aversion) is **derived, not typed in**: delta = SR_target / sigma_mkt,
so the knob a user turns is a target market Sharpe ratio (default 0.35), which is
a quantity one can argue about with evidence. The legacy hard-coded delta = 0.5,
which on a 15% market implies an equity risk premium near 1% - so its prior was
effectively inert and the views did all the work (M2-05).

Reference weights are free-float market caps where available, then shares
outstanding, then equal weight, and **every fallback is recorded**. In a backtest
today's market caps are look-ahead, so equal weight is used as-of and said so.
They are labelled "universe reference portfolio" throughout, because forty NSE
stocks are not the market.

See PAPERS/notes/02_black_litterman.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.types import Basis, Compounding, CovarianceEstimate, Periodicity, Provenance, ReturnEstimate


@dataclass(frozen=True, slots=True)
class EquilibriumSettings:
    target_market_sharpe: float = 0.35   # the ERP assumption, in the unit people can argue about
    min_delta: float = 0.5               # guards against an inert or absurd prior
    max_delta: float = 15.0


def risk_aversion(market_volatility: float, settings: EquilibriumSettings = EquilibriumSettings()) -> float:
    """
    delta = SR_target / sigma_mkt, both annual.

    At sigma = 15% and SR = 0.35 this gives delta = 2.33 and an implied equity
    risk premium of delta * sigma^2 = 5.25% - in the range the evidence supports
    for Indian equities, and four times the legacy's implied premium.
    """
    if not np.isfinite(market_volatility) or market_volatility <= 0:
        raise EstimationError(f"market volatility must be positive, got {market_volatility}")
    if not np.isfinite(settings.target_market_sharpe) or settings.target_market_sharpe <= 0:
        raise EstimationError(f"target market Sharpe must be positive, got {settings.target_market_sharpe}")
    delta = settings.target_market_sharpe / market_volatility
    return float(np.clip(delta, settings.min_delta, settings.max_delta))


def reference_weights(tickers, *, market_caps: pd.Series | None = None,
                      shares_outstanding: pd.Series | None = None,
                      price: pd.Series | None = None,
                      allow_market_cap: bool = True) -> tuple[pd.Series, tuple[str, ...]]:
    """
    Cap weights within the universe, with an explicit fallback ladder.

    Returns (weights, degradations). `allow_market_cap=False` is the backtest
    setting: a market cap read today did not exist at the as-of date.
    """
    names = list(tickers)
    if not names:
        raise EstimationError("reference portfolio needs at least one ticker")
    notes: list[str] = []

    caps = None
    if not allow_market_cap:
        notes.append("market caps are point-in-time unavailable (backtest); using equal weight")
    elif market_caps is not None:
        caps = pd.Series(market_caps, dtype=float).reindex(names)
    elif shares_outstanding is not None and price is not None:
        caps = pd.Series(shares_outstanding, dtype=float).reindex(names) * pd.Series(price, dtype=float).reindex(names)
        notes.append("market cap built from shares outstanding x price (free float unavailable)")

    if caps is not None:
        bad = caps[~np.isfinite(caps) | (caps <= 0)]
        if len(bad):
            notes.append(f"no usable market cap for {len(bad)} of {len(names)} names; "
                         f"equal-weighted within the universe instead: {list(bad.index[:5])}")
            caps = None

    if caps is None:
        w = pd.Series(1.0 / len(names), index=names)
    else:
        w = caps / caps.sum()
    return w.rename("w_ref"), tuple(notes)


def market_volatility(cov: CovarianceEstimate, w_ref: pd.Series) -> float:
    """Annualised volatility of the reference portfolio."""
    annual = cov.annualise()
    w = pd.Series(w_ref, dtype=float).reindex(annual.tickers)
    if w.isna().any():
        raise EstimationError(f"reference weights missing for: {list(w.index[w.isna()][:8])}")
    var = float(w.to_numpy() @ annual.matrix.to_numpy() @ w.to_numpy())
    if not np.isfinite(var) or var <= 0:
        raise EstimationError("reference portfolio variance is not positive")
    return float(np.sqrt(var))


def equilibrium_prior(cov: CovarianceEstimate, w_ref: pd.Series, *,
                      settings: EquilibriumSettings = EquilibriumSettings(),
                      degradations: tuple[str, ...] = ()) -> tuple[ReturnEstimate, float]:
    """
    Pi = delta * Sigma * w_ref, returned as an ANNUAL SIMPLE **EXCESS** estimate
    together with the delta that produced it.

    Reverse-optimisation identity, asserted by test: feeding Pi straight back
    into an unconstrained mean-variance solve returns `w_ref` exactly.
    """
    annual = cov.annualise()
    w = pd.Series(w_ref, dtype=float).reindex(annual.tickers)
    if w.isna().any():
        raise EstimationError(f"reference weights missing for: {list(w.index[w.isna()][:8])}")
    sigma_mkt = market_volatility(annual, w)
    delta = risk_aversion(sigma_mkt, settings)
    pi = pd.Series(delta * (annual.matrix.to_numpy() @ w.to_numpy()), index=annual.tickers)
    erp = float(delta * sigma_mkt ** 2)
    prov = Provenance(
        method="equilibrium prior Pi = delta * Sigma * w_ref (Black-Litterman 1992)",
        window=(f"delta {delta:.2f} from target market Sharpe {settings.target_market_sharpe:.2f} "
                f"at reference volatility {sigma_mkt:.1%}; implied risk premium {erp:.2%}"),
        as_of=annual.provenance.as_of,
        source="universe reference portfolio",
        degradations=tuple(degradations),
    )
    return ReturnEstimate(values=pi, basis=Basis.EXCESS, compounding=Compounding.SIMPLE,
                          periodicity=Periodicity.ANNUAL, provenance=prov,
                          periods_per_year=cov.periods_per_year), delta


def implied_weights(mu_excess: ReturnEstimate, cov: CovarianceEstimate, delta: float) -> pd.Series:
    """Unconstrained mean-variance weights w = (delta * Sigma)^-1 mu. The inverse of the prior."""
    if mu_excess.basis is not Basis.EXCESS:
        raise EstimationError("implied weights need EXCESS returns; call .to_excess(rf) first")
    annual = cov.annualise()
    mu = mu_excess.annualise().values.reindex(annual.tickers)
    if mu.isna().any():
        raise EstimationError(f"expected returns missing for: {list(mu.index[mu.isna()][:8])}")
    w = np.linalg.solve(delta * annual.matrix.to_numpy(), mu.to_numpy())
    return pd.Series(w, index=annual.tickers, name="w_implied")
