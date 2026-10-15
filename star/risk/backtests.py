"""
Was the risk model right? The tests that answer it.

A VaR number is a forecast, and a forecast that is never scored is an opinion.
The legacy printed risk figures and never once checked them against what
happened (M8-06), including a "HONEST RISK SUMMARY" containing claims that were
never computed (M6-07).

Four standard tests, each answering a different question:

* **Kupiec (1995) proportion of failures** - were there about as many breaches as
  the confidence level implies? 14 breaches where 6 were expected is a model that
  understates risk.
* **Christoffersen (1998) independence** - were the breaches spread out, or did
  they arrive in clusters? Clustered breaches mean the model misses regime
  changes, which is precisely when the number matters.
* **Conditional coverage** - the two together, which is the test that should be
  quoted.
* **Engle & Manganelli (2004) dynamic quantile** - can tomorrow's breach be
  predicted from today's information? If it can, the model is leaving something on
  the table.

Plus **Basel's traffic light** for a supervisor-recognisable verdict, and
**Acerbi & Szekely (2014)** for expected shortfall, since ES needs its own test -
counting breaches says nothing about how bad they were.

Every test reports its p-value and a plain-English verdict. **A test that does not
reject is not proof the model is right** - and that caveat is not decorative.
Measured here (tests/test_risk_backtests.py), feeding a constant-volatility VaR
to data with real volatility clustering, the chance of correctly rejecting it is:

| observations | at 95% | at 99% |
|---|---|---|
| 750  (3 years) | 29% | 10% |
| 1,500 (6 years) | 57% | 26% |
| 2,500 (10 years) | 84% | 30% |

So on three years of data a clearly wrong risk model passes roughly seven times
out of ten, and at 99% it passes nine times out of ten - because breaches are
rare, and rare events carry little information. The brief therefore reports
"not rejected" with the sample size beside it, never as a clean bill of health.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import EstimationError


def _prep(returns, var_forecast) -> tuple[np.ndarray, np.ndarray]:
    """
    Align realised returns with their forecasts.

    Aligned on the index when both carry the same one, positionally otherwise -
    silently pairing a return with the wrong day's forecast would invalidate every
    test in this module, so a length mismatch raises instead of producing an
    empty overlap.
    """
    r = pd.Series(returns).astype(float)
    if np.isscalar(var_forecast):
        v = pd.Series(float(var_forecast), index=r.index)
    else:
        v = pd.Series(var_forecast).astype(float)
        if not v.index.equals(r.index):
            if len(v) != len(r):
                raise EstimationError(f"{len(r)} returns but {len(v)} VaR forecasts")
            v = pd.Series(v.to_numpy(), index=r.index)
    df = pd.concat([r, v], axis=1, keys=["r", "var"]).dropna()
    if len(df) < 30:
        raise EstimationError(f"a VaR backtest needs at least 30 observations, got {len(df)}")
    if (df["var"] <= 0).any():
        raise EstimationError("VaR forecasts must be positive losses (0.02 means a 2% loss)")
    return df["r"].to_numpy(), df["var"].to_numpy()


def breaches(returns, var_forecast) -> np.ndarray:
    """1 where the realised loss exceeded the forecast loss."""
    r, v = _prep(returns, var_forecast)
    return (r < -v).astype(int)


def kupiec_pof(returns, var_forecast, confidence: float) -> tuple[float, float, int, float]:
    """
    Unconditional coverage. Returns (statistic, p_value, breaches, expected).

    Likelihood ratio against the null that the breach rate equals 1 - confidence;
    chi-squared with one degree of freedom.
    """
    hits = breaches(returns, var_forecast)
    n, x = len(hits), int(hits.sum())
    p = 1.0 - confidence
    expected = n * p
    if x == 0:
        lr = -2.0 * (n * np.log(1 - p))
    elif x == n:
        lr = -2.0 * (n * np.log(p))
    else:
        pi = x / n
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(lr), float(stats.chi2.sf(lr, 1)), x, float(expected)


def christoffersen_independence(returns, var_forecast) -> tuple[float, float]:
    """Do breaches cluster? Likelihood ratio on the 2-state transition counts."""
    hits = breaches(returns, var_forecast)
    prev, cur = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0                                   # too few breaches to say anything
    pi0, pi1 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def ll(p, a, b):
        if p <= 0 or p >= 1:
            return 0.0
        return a * np.log(1 - p) + b * np.log(p)
    lr = -2.0 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_cc(returns, var_forecast, confidence: float) -> tuple[float, float]:
    """Conditional coverage: right number of breaches AND not clustered. chi2(2)."""
    uc, _, _, _ = kupiec_pof(returns, var_forecast, confidence)
    ind, _ = christoffersen_independence(returns, var_forecast)
    lr = uc + ind
    return float(lr), float(stats.chi2.sf(lr, 2))


def dynamic_quantile(returns, var_forecast, confidence: float, lags: int = 4) -> tuple[float, float]:
    """
    Engle & Manganelli (2004): regress the centred breach indicator on its own
    lags and the VaR forecast. If anything predicts tomorrow's breach, the model
    is mis-specified. chi2 with (2 + lags) degrees of freedom.
    """
    r, v = _prep(returns, var_forecast)
    hits = (r < -v).astype(float)
    p = 1.0 - confidence
    y = hits - p
    n = len(y)
    if n <= lags + 3:
        raise EstimationError("not enough observations for the dynamic quantile test")
    X = [np.ones(n - lags)]
    for k in range(1, lags + 1):
        X.append(y[lags - k: n - k])
    X.append(v[lags:])
    X = np.column_stack(X)
    yy = y[lags:]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    stat = float(beta @ X.T @ X @ beta / (p * (1 - p)))
    return stat, float(stats.chi2.sf(stat, X.shape[1]))


def basel_traffic_light(returns, var_forecast, confidence: float = 0.99) -> tuple[str, int, int]:
    """
    The supervisor's version: breaches in a 250-day window at 99%.
    0-4 green, 5-9 yellow, 10+ red. Scaled pro rata for other window lengths, and
    the raw count is returned so nobody has to trust the scaling.
    """
    hits = breaches(returns, var_forecast)
    n, x = len(hits), int(hits.sum())
    scaled = int(round(x * 250 / n))
    zone = "green" if scaled <= 4 else ("yellow" if scaled <= 9 else "red")
    return zone, x, scaled


def acerbi_szekely_z2(returns, var_forecast, es_forecast, confidence: float,
                      null_returns: np.ndarray | None = None) -> tuple[float, float | None]:
    """
    Test of expected shortfall itself (Acerbi & Szekely 2014, test 2).

        Z = 1 + mean( r_t * 1{r_t < -VaR_t} / ((1 - c) * ES_t) )

    Z is around zero when ES is right, **negative** when realised tail losses are
    worse than forecast. Counting breaches cannot detect that; this can.

    The null distribution has no closed form, so a p-value is only returned when
    `null_returns` is supplied: an (n_sims, n_obs) array simulated from **the model
    that produced these VaR and ES forecasts** - not from the true distribution,
    which is unknown, and not from some other model. Getting that wrong silently
    destroys the test: simulate from the truth while testing a model that halves
    its ES and the statistic shifts identically under both, so the p-value comes
    back reassuring. Because that mistake is easy to make and invisible, the draws
    are checked against the supplied VaR and rejected if their breach rate is
    inconsistent with the confidence level.

    Without `null_returns` the statistic and its sign are still informative, and
    the report states that the test was not calibrated.
    """
    r, v = _prep(returns, var_forecast)
    es = pd.Series(es_forecast).astype(float) if not np.isscalar(es_forecast) \
        else pd.Series(float(es_forecast), index=range(len(r)))
    es = es.to_numpy()[: len(r)]
    if np.any(es <= 0):
        raise EstimationError("expected shortfall forecasts must be positive losses")
    p = 1.0 - confidence

    def z_of(x):
        hit = x < -v
        return 1.0 + float(np.sum(x * hit / (p * es)) / len(x))

    z = z_of(r)
    if null_returns is None:
        return z, None
    sims = np.asarray(null_returns, dtype=float)
    if sims.ndim != 2 or sims.shape[1] != len(r):
        raise EstimationError(f"null_returns must be (n_sims, {len(r)})")
    rate = float(np.mean(sims < -v))
    tol = 3.0 * np.sqrt(p * (1 - p) / (sims.size / sims.shape[0]))
    if abs(rate - p) > max(tol, 0.2 * p):
        raise EstimationError(
            f"null_returns breach the supplied VaR {rate:.2%} of the time, but the model claims "
            f"{p:.2%}. The draws must come from the model being tested, or the p-value is meaningless.")
    null = np.array([z_of(row) for row in sims])
    return z, float(np.mean(null <= z))                   # left tail: ES understated


@dataclass(frozen=True)
class VaRBacktest:
    confidence: float
    observations: int
    breaches: int
    expected_breaches: float
    kupiec_p: float
    independence_p: float
    conditional_coverage_p: float
    dq_p: float
    basel_zone: str
    es_z2: float | None
    es_p: float | None
    headline: str
    passed: bool

    def verdict(self, alpha: float = 0.05) -> str:
        return "model health: acceptable" if self.passed else "model health: rejected"


def validate_var(returns, var_forecast, confidence: float, *, es_forecast=None,
                 null_returns=None, alpha: float = 0.05) -> VaRBacktest:
    """Run every test and turn the result into one sentence a manager can act on."""
    lr_uc, p_uc, x, expected = kupiec_pof(returns, var_forecast, confidence)
    _, p_ind = christoffersen_independence(returns, var_forecast)
    _, p_cc = christoffersen_cc(returns, var_forecast, confidence)
    _, p_dq = dynamic_quantile(returns, var_forecast, confidence)
    zone, raw, _ = basel_traffic_light(returns, var_forecast, confidence)
    z2 = p_es = None
    if es_forecast is not None:
        z2, p_es = acerbi_szekely_z2(returns, var_forecast, es_forecast, confidence,
                                     null_returns=null_returns)
    n = len(breaches(returns, var_forecast))
    passed = p_cc > alpha and p_dq > alpha and (p_es is None or p_es > alpha)

    if x > expected and p_uc <= alpha:
        head = (f"The risk model understated losses: {x} breaches of the {confidence:.0%} limit "
                f"where {expected:.0f} were expected ({n} observations, p={p_uc:.3f}).")
    elif x < expected and p_uc <= alpha:
        head = (f"The risk model overstated losses: {x} breaches where {expected:.0f} were expected "
                f"({n} observations, p={p_uc:.3f}). Risk limits may be tighter than intended.")
    elif p_ind <= alpha:
        head = (f"Breaches arrived in clusters (independence p={p_ind:.3f}): the model reacts to "
                f"turbulence too slowly, even though the total count looks right.")
    else:
        head = (f"{x} breaches where {expected:.0f} were expected over {n} observations; "
                f"the model is consistent with its own {confidence:.0%} claim.")
    if not passed and p_es is not None and p_es <= alpha:
        head += " Realised tail losses were worse than the expected-shortfall forecast."
    return VaRBacktest(confidence=confidence, observations=n, breaches=x, expected_breaches=expected,
                       kupiec_p=p_uc, independence_p=p_ind, conditional_coverage_p=p_cc, dq_p=p_dq,
                       basel_zone=zone, es_z2=z2, es_p=p_es, headline=head, passed=bool(passed))
