"""
Is the track record real, or is it the best of many tries?

A Sharpe ratio without a standard error is not evidence. On three years of
quarterly rebalancing the standard error is large enough that most apparently
good strategies cannot be distinguished from equal weight - and saying so is the
single most valuable output this rebuild produces. The legacy printed Sharpe
ratios with no inference of any kind (M8-06).

What is here:

* **Lo (2002) standard error** - how precise the Sharpe ratio is, adjusted for
  autocorrelation, which inflates a naive Sharpe when returns are smoothed.
* **Ledoit & Wolf (2008) difference test** - is this strategy's Sharpe genuinely
  different from equal weight, or from the index? The correct test, because the
  two series are the same periods and are correlated.
* **Probabilistic and Deflated Sharpe** (Bailey & Lopez de Prado 2014) - the
  probability that the true Sharpe exceeds a threshold, and the same after
  penalising how many configurations were tried. Twenty variations tried on the
  same data will produce an apparently good one by chance; the deflated version
  prices that in.
* **Minimum track record length** - how long a record would have to be before
  this Sharpe could be called significant. Usually the most sobering number.
* **PBO via CSCV** (Bailey et al. 2017) - the probability that the configuration
  which looked best in-sample is below median out-of-sample. Above 0.5 means the
  selection process is worse than choosing at random.
* **Stationary bootstrap intervals** - confidence intervals that respect serial
  dependence instead of assuming it away.

Skewness and kurtosis enter these formulas deliberately: a strategy that makes
small gains and occasional large losses has a flattering Sharpe, and the
probabilistic version discounts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import EstimationError


def _clean(returns) -> np.ndarray:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 20:
        raise EstimationError(f"inference needs at least 20 observations, got {len(r)}")
    return r


def sharpe_ratio(returns, *, risk_free=0.0, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe of excess returns. `risk_free` is per period, not annual."""
    r = _clean(returns)
    rf = np.asarray(pd.Series(risk_free).reindex(pd.Series(returns).dropna().index).fillna(0.0)
                    if hasattr(risk_free, "__len__") else np.full(len(r), float(risk_free)))
    ex = r - rf
    sd = ex.std(ddof=1)
    if sd <= 0:
        raise EstimationError("returns have no variance, so a Sharpe ratio is undefined")
    return float(ex.mean() / sd * np.sqrt(periods_per_year))


def autocorrelation_factor(returns, lags: int = 4) -> float:
    """
    Lo's (2002) adjustment for serial correlation.

    Positively autocorrelated returns - smoothed marks, illiquid holdings - make
    the naive annualised Sharpe too high. The factor is below 1 in that case.
    """
    r = _clean(returns)
    q = int(min(lags, len(r) // 4))
    if q < 1:
        return 1.0
    rho = [float(pd.Series(r).autocorr(lag=k)) for k in range(1, q + 1)]
    rho = [0.0 if not np.isfinite(x) else x for x in rho]
    denom = q + 1 + 2 * sum((q + 1 - k) * rho[k - 1] for k in range(1, q + 1))
    if denom <= 0:
        return 1.0
    return float((q + 1) / np.sqrt(denom) / np.sqrt(q + 1))


def sharpe_standard_error(returns, *, periods_per_year: float = 252.0,
                          adjust_autocorrelation: bool = True) -> float:
    """
    Lo (2002): SE(SR) = sqrt((1 + SR^2 / 2) / n), annualised, with an optional
    correction for serial correlation.
    """
    r = _clean(returns)
    n = len(r)
    sr_period = float(r.mean() / r.std(ddof=1))
    se = np.sqrt((1.0 + 0.5 * sr_period ** 2) / n) * np.sqrt(periods_per_year)
    if adjust_autocorrelation:
        se = se / max(autocorrelation_factor(r), 1e-6)
    return float(se)


def sharpe_confidence_interval(returns, *, confidence: float = 0.95,
                               periods_per_year: float = 252.0) -> tuple[float, float, float]:
    """(Sharpe, lower, upper). The width is usually the point of the exercise."""
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    se = sharpe_standard_error(returns, periods_per_year=periods_per_year)
    z = stats.norm.ppf(0.5 + confidence / 2)
    return sr, sr - z * se, sr + z * se


def probabilistic_sharpe_ratio(returns, *, threshold: float = 0.0,
                               periods_per_year: float = 252.0) -> float:
    """
    P(true Sharpe > threshold), accounting for skewness and kurtosis
    (Bailey & Lopez de Prado 2014). Threshold 0 asks "is there any edge at all?".
    """
    r = _clean(returns)
    n = len(r)
    sr = float(r.mean() / r.std(ddof=1))                         # per period
    target = float(threshold) / np.sqrt(periods_per_year)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - target) * np.sqrt(n - 1) / denom))


def minimum_track_record_length(returns, *, threshold: float = 0.0, confidence: float = 0.95,
                                periods_per_year: float = 252.0) -> float:
    """
    How many observations would be needed for this Sharpe to be significant.

    Returns the length in **periods**; divide by `periods_per_year` for years.
    Infinite when the observed Sharpe is below the threshold - no amount of data
    makes a losing record significant.
    """
    r = _clean(returns)
    sr = float(r.mean() / r.std(ddof=1))
    target = float(threshold) / np.sqrt(periods_per_year)
    if sr <= target:
        return float("inf")
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    z = stats.norm.ppf(confidence)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / (sr - target)) ** 2)


def deflated_sharpe_ratio(returns, *, n_trials: int, trial_sharpe_variance: float | None = None,
                          periods_per_year: float = 252.0) -> float:
    """
    The probabilistic Sharpe against the threshold a *lucky* strategy would clear
    after `n_trials` attempts (Bailey & Lopez de Prado 2014).

    With twenty configurations tried, the best one is expected to show a positive
    Sharpe even with no skill; this deflates by that expectation. `n_trials` must
    be the honest count - the pipeline records it, and where it is unknown the
    report says so rather than assuming one.
    """
    if n_trials < 1:
        raise EstimationError(f"n_trials must be at least 1, got {n_trials}")
    r = _clean(returns)
    var = (float(trial_sharpe_variance) if trial_sharpe_variance is not None
           else float(np.var([r.mean() / r.std(ddof=1)], ddof=0)) or (1.0 / len(r)))
    if n_trials == 1:
        return probabilistic_sharpe_ratio(r, threshold=0.0, periods_per_year=periods_per_year)
    euler = 0.5772156649015329
    e1 = stats.norm.ppf(1 - 1.0 / n_trials)
    e2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    expected_max = np.sqrt(var) * ((1 - euler) * e1 + euler * e2)
    return probabilistic_sharpe_ratio(r, threshold=expected_max * np.sqrt(periods_per_year),
                                      periods_per_year=periods_per_year)


@dataclass(frozen=True)
class SharpeDifference:
    difference: float
    standard_error: float
    t_statistic: float
    p_value: float
    headline: str


def sharpe_difference_test(a, b, *, periods_per_year: float = 252.0,
                           lags: int | None = None) -> SharpeDifference:
    """
    Ledoit & Wolf (2008): is the Sharpe of `a` different from that of `b`?

    Both series must cover the same periods. The variance of the difference comes
    from a HAC (Newey-West) estimate of the four moments the two Sharpe ratios
    depend on, so serial correlation and the correlation between the two series
    are both handled - a naive two-sample test gets this badly wrong, because the
    two series share the same market.
    """
    df = pd.concat([pd.Series(a), pd.Series(b)], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 20:
        raise EstimationError(f"comparing Sharpe ratios needs at least 20 aligned periods, got {len(df)}")
    x, y = df["a"].to_numpy(), df["b"].to_numpy()
    n = len(x)
    mu1, mu2 = x.mean(), y.mean()
    g1, g2 = (x ** 2).mean(), (y ** 2).mean()
    s1, s2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    if s1 <= 0 or s2 <= 0:
        raise EstimationError("one of the series has no variance")
    diff = mu1 / np.sqrt(s1) - mu2 / np.sqrt(s2)

    grad = np.array([g1 / s1 ** 1.5, -g2 / s2 ** 1.5, -0.5 * mu1 / s1 ** 1.5, 0.5 * mu2 / s2 ** 1.5])
    v = np.column_stack([x - mu1, y - mu2, x ** 2 - g1, y ** 2 - g2])
    q = int(lags if lags is not None else max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1))
    omega = v.T @ v / n
    for k in range(1, q + 1):
        gamma = v[k:].T @ v[:-k] / n
        w = 1.0 - k / (q + 1)
        omega += w * (gamma + gamma.T)
    var = float(grad @ omega @ grad) / n
    scale = max(abs(mu1 / np.sqrt(s1)), abs(mu2 / np.sqrt(s2)), 1e-12)
    if var <= 0:
        if abs(diff) <= 1e-10 * scale:
            # the two series are the same (a strategy compared with itself, or one
            # that exactly replicates its benchmark): zero difference, and zero
            # uncertainty about it. That is an answer, not a failure.
            return SharpeDifference(difference=0.0, standard_error=0.0, t_statistic=0.0,
                                    p_value=1.0,
                                    headline="The two series are identical: there is no difference "
                                             "in Sharpe ratio to test.")
        raise EstimationError("the estimated variance of the Sharpe difference is not positive, "
                              f"yet the difference is {diff:.4f}: the inputs are too short or "
                              f"too nearly identical for this test")
    se = np.sqrt(var)
    t = diff / se
    p = float(2 * stats.norm.sf(abs(t)))
    annual = np.sqrt(periods_per_year)
    if p > 0.10:
        head = (f"The Sharpe difference of {diff * annual:+.2f} is within estimation error "
                f"(p={p:.2f}): this strategy cannot be shown to beat the comparison.")
    else:
        head = (f"Sharpe difference {diff * annual:+.2f}, p={p:.3f}: the gap is larger than "
                f"estimation error over this period.")
    return SharpeDifference(difference=float(diff * annual), standard_error=float(se * annual),
                            t_statistic=float(t), p_value=p, headline=head)


@dataclass(frozen=True)
class OverfittingResult:
    pbo: float
    n_splits: int
    n_configurations: int
    median_oos_rank: float
    headline: str


def probability_of_backtest_overfitting(returns: pd.DataFrame, *, n_splits: int = 8) -> OverfittingResult:
    """
    CSCV (Bailey et al. 2017): how often the configuration that looked best
    in-sample lands below median out-of-sample.

    `returns` is one column per configuration tried, over the same periods. A PBO
    above 0.5 means the selection procedure is worse than picking at random, which
    is the honest description of most backtest-driven parameter choices.
    """
    R = pd.DataFrame(returns).dropna()
    n_config = R.shape[1]
    if n_config < 2:
        raise EstimationError("overfitting can only be assessed across two or more configurations")
    if n_splits % 2 or n_splits < 4:
        raise EstimationError(f"n_splits must be even and at least 4, got {n_splits}")
    if len(R) < n_splits * 10:
        raise EstimationError(f"{len(R)} observations is too few for {n_splits} splits")

    chunks = np.array_split(R.to_numpy(), n_splits)
    logits, ranks = [], []
    for pick in combinations(range(n_splits), n_splits // 2):
        is_idx = list(pick)
        oos_idx = [i for i in range(n_splits) if i not in pick]
        is_data = np.vstack([chunks[i] for i in is_idx])
        oos_data = np.vstack([chunks[i] for i in oos_idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            is_sr = is_data.mean(axis=0) / is_data.std(axis=0, ddof=1)
            oos_sr = oos_data.mean(axis=0) / oos_data.std(axis=0, ddof=1)
        is_sr = np.nan_to_num(is_sr, nan=-np.inf)
        oos_sr = np.nan_to_num(oos_sr, nan=-np.inf)
        best = int(np.argmax(is_sr))
        rank = float((oos_sr <= oos_sr[best]).sum()) / n_config          # 1 = best out of sample
        rank = min(max(rank, 1.0 / (n_config + 1)), n_config / (n_config + 1.0))
        ranks.append(rank)
        logits.append(np.log(rank / (1 - rank)))
    pbo = float(np.mean(np.asarray(logits) <= 0))
    median_rank = float(np.median(ranks))
    if pbo > 0.5:
        head = (f"Probability of backtest overfitting {pbo:.0%}: the configuration that looked best "
                f"in-sample was below median out-of-sample more often than not. Selection is adding "
                f"nothing.")
    else:
        head = (f"Probability of backtest overfitting {pbo:.0%}: in-sample selection carried over "
                f"out-of-sample more often than not.")
    return OverfittingResult(pbo=pbo, n_splits=n_splits, n_configurations=n_config,
                             median_oos_rank=median_rank, headline=head)


def stationary_bootstrap_interval(returns, statistic, *, confidence: float = 0.95,
                                  n_resamples: int = 1000, mean_block: float = 10.0,
                                  seed: int = 20260920) -> tuple[float, float, float]:
    """
    (point estimate, lower, upper) from a stationary block bootstrap, which keeps
    the serial dependence that a naive i.i.d. bootstrap would destroy.
    """
    r = _clean(returns)
    rng = np.random.default_rng(seed)
    n = len(r)
    p_new = 1.0 / mean_block
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(0, n)
        for t in range(1, n):
            idx[t] = rng.integers(0, n) if rng.random() < p_new else (idx[t - 1] + 1) % n
        draws[b] = statistic(r[idx])
    lo = float(np.percentile(draws, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(draws, 100 * (1 + confidence) / 2))
    return float(statistic(r)), lo, hi
