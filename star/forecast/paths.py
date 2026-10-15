"""
Path generators: three ways to produce plausible futures, each with a different
thing it is good at.

* **Stationary block bootstrap** (Politis & Romano 1994) - resample *blocks of
  consecutive dates* from history. Because whole dates are drawn, the
  cross-sectional pattern of a day survives untouched: the way holdings moved
  together on 24 February 2022 is in the sample, with no copula to estimate and
  no assumption about tail dependence. Random block lengths keep the resampled
  series stationary, which fixed-length blocks do not.
* **Filtered historical simulation** (Barone-Adesi, Giannopoulos & Vosper 1999) -
  standardise each asset's returns by its own GARCH volatility, resample the
  standardised residuals *by date*, then rescale by volatility that evolves along
  the path. This is the one that keeps tails fat at long horizons, because
  volatility clusters as the path unfolds rather than being fixed at today's
  level.
* **Multivariate Student-t** - a parametric reference. Its value is that its
  answers are known in closed form, so it is what the engine is tested against.

Every generator returns **portfolio-level log returns**, with the benchmark
generated on the same draws. Simulating the benchmark independently would make
"probability of underperforming the index" meaningless - easy to miss, and fatal
if missed.

The asset-level cube (days x paths x holdings) is never materialised. Where a
generator needs per-asset state, as filtered historical simulation does, only the
*current* state is carried - one number per path per asset - so memory stays flat
in the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import EstimationError


def stationary_block_indices(n_obs: int, horizon: int, n_paths: int, *, mean_block: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Row indices for a stationary block bootstrap: (horizon, n_paths).

    Each step either continues the previous block (probability 1 - 1/mean_block)
    or starts a new one at a uniformly random date. Blocks wrap around the end of
    the sample, which is what keeps the resampled series stationary.
    """
    if n_obs < 2:
        raise EstimationError("the bootstrap needs at least two historical observations")
    if mean_block < 1:
        raise EstimationError(f"mean block length must be at least 1, got {mean_block}")
    p_new = 1.0 / float(mean_block)
    idx = np.empty((horizon, n_paths), dtype=np.int32)
    idx[0] = rng.integers(0, n_obs, n_paths)
    for t in range(1, horizon):
        cont = rng.random(n_paths) >= p_new
        idx[t] = np.where(cont, (idx[t - 1] + 1) % n_obs, rng.integers(0, n_obs, n_paths))
    return idx


def iid_indices(n_obs: int, horizon: int, n_paths: int, *, rng: np.random.Generator,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Independent draws - the i.i.d. reference, and the basis of regime conditioning."""
    if weights is None:
        return rng.integers(0, n_obs, (horizon, n_paths)).astype(np.int32)
    p = np.asarray(weights, dtype=float)
    if len(p) != n_obs or np.any(p < 0) or p.sum() <= 0:
        raise EstimationError("resampling weights must be non-negative, sum to more than zero, "
                              "and match the history length")
    return rng.choice(n_obs, size=(horizon, n_paths), p=p / p.sum()).astype(np.int32)


@dataclass
class BootstrapGenerator:
    """
    Resamples historical dates. `portfolio` and `benchmark` are daily **log**
    returns on the same index.

    `regime_weights` (optional) biases the draw towards dates that resemble the
    current regime - the regime-conditional variant. Dates are still whole dates,
    so cross-sectional structure survives.
    """

    portfolio: np.ndarray
    benchmark: np.ndarray | None = None
    mean_block: float = 10.0
    regime_weights: np.ndarray | None = None
    name: str = "stationary block bootstrap"

    def __post_init__(self) -> None:
        self.portfolio = np.asarray(self.portfolio, dtype=float)
        if self.portfolio.ndim != 1 or len(self.portfolio) < 60:
            raise EstimationError("the bootstrap needs at least 60 daily portfolio returns")
        if self.benchmark is not None:
            self.benchmark = np.asarray(self.benchmark, dtype=float)
            if len(self.benchmark) != len(self.portfolio):
                raise EstimationError("portfolio and benchmark histories must be the same length")
        if self.regime_weights is not None:
            self.regime_weights = np.asarray(self.regime_weights, dtype=float)
            self.name = "regime-conditional bootstrap"

    @property
    def description(self) -> str:
        if self.regime_weights is not None:
            return f"{self.name}, weighted towards dates resembling the current regime"
        return f"{self.name}, mean block {self.mean_block:g} sessions"

    def generate(self, horizon: int, n_paths: int, rng: np.random.Generator):
        n = len(self.portfolio)
        if self.regime_weights is not None:
            idx = iid_indices(n, horizon, n_paths, rng=rng, weights=self.regime_weights)
        else:
            idx = stationary_block_indices(n, horizon, n_paths, mean_block=self.mean_block, rng=rng)
        port = self.portfolio[idx]
        bench = self.benchmark[idx] if self.benchmark is not None else None
        return port, bench


@dataclass
class StudentTGenerator:
    """
    Parametric reference: portfolio and benchmark drawn from a bivariate Student-t
    with the given daily covariance and tail index. Used to test the engine
    against answers that are known in closed form.
    """

    cov: np.ndarray                      # 1x1 or 2x2 daily log-return covariance
    nu: float = 6.0
    name: str = "multivariate Student-t"

    def __post_init__(self) -> None:
        self.cov = np.atleast_2d(np.asarray(self.cov, dtype=float))
        if self.cov.shape[0] != self.cov.shape[1] or self.cov.shape[0] > 2:
            raise EstimationError("the parametric generator handles the portfolio, optionally with "
                                  "a benchmark: a 1x1 or 2x2 covariance")
        if self.nu <= 2:
            raise EstimationError(f"tail index must exceed 2 for a finite variance, got {self.nu}")

    @property
    def description(self) -> str:
        return f"{self.name} (nu={self.nu:g})"

    def generate(self, horizon: int, n_paths: int, rng: np.random.Generator):
        k = self.cov.shape[0]
        scale = self.cov * (self.nu - 2.0) / self.nu           # so the covariance is as given
        L = np.linalg.cholesky(scale + 1e-18 * np.eye(k))
        z = rng.standard_normal((horizon, n_paths, k))
        g = rng.chisquare(self.nu, (horizon, n_paths, 1)) / self.nu
        draws = (z @ L.T) / np.sqrt(g)
        port = draws[:, :, 0]
        bench = draws[:, :, 1] if k == 2 else None
        return port, bench


@dataclass
class FilteredHistoricalGenerator:
    """
    Filtered historical simulation at the asset level.

    `residuals` is (T x n) of standardised residuals - each asset's log return
    divided by its own conditional volatility - so resampling a **row** keeps the
    cross-sectional pattern of that day. `omega/alpha/gamma/beta` are the fitted
    GJR-GARCH parameters per asset, and `sigma0` the current conditional
    volatility, both in log-return units.

    Volatility evolves along each path, so a bad day makes the next day more
    volatile, which is why the tails stay fat as the horizon lengthens. Only the
    current variance is carried - shape (paths, assets) - never the full cube.
    """

    residuals: np.ndarray
    weights: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    sigma0: np.ndarray
    mu: np.ndarray | None = None
    benchmark_residuals: np.ndarray | None = None
    benchmark_sigma: float | None = None
    name: str = "filtered historical simulation"
    _n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.residuals = np.asarray(self.residuals, dtype=float)
        if self.residuals.ndim != 2 or len(self.residuals) < 60:
            raise EstimationError("filtered historical simulation needs at least 60 residual rows")
        self._n = self.residuals.shape[1]
        for nm in ("weights", "omega", "alpha", "gamma", "beta", "sigma0"):
            v = np.asarray(getattr(self, nm), dtype=float)
            if v.shape != (self._n,):
                raise EstimationError(f"{nm} must have one entry per asset ({self._n})")
            setattr(self, nm, v)
        self.mu = np.zeros(self._n) if self.mu is None else np.asarray(self.mu, dtype=float)
        if np.any(self.sigma0 <= 0):
            raise EstimationError("current volatilities must be positive")

    @property
    def description(self) -> str:
        return f"{self.name} over {self._n} holdings, volatility propagated along each path"

    def generate(self, horizon: int, n_paths: int, rng: np.random.Generator):
        T = len(self.residuals)
        var = np.tile(self.sigma0 ** 2, (n_paths, 1))            # (paths, assets) - the only state
        port = np.empty((horizon, n_paths))
        bench = None if self.benchmark_residuals is None else np.empty((horizon, n_paths))
        w = self.weights
        for t in range(horizon):
            rows = rng.integers(0, T, n_paths)
            z = self.residuals[rows]                             # (paths, assets), a whole date each
            sd = np.sqrt(var)
            e = sd * z
            r = self.mu + e
            simple = np.expm1(r) @ w
            port[t] = np.log1p(np.clip(simple, -0.999999, None))
            if bench is not None:
                bench[t] = self.benchmark_residuals[rows] * (self.benchmark_sigma or 1.0)
            var = self.omega + (self.alpha + self.gamma * (e < 0)) * e ** 2 + self.beta * var
        return port, bench


def regime_similarity_weights(stress_probability: pd.Series, current: float, *,
                              sharpness: float = 8.0, floor: float = 0.05) -> np.ndarray:
    """
    Weights for the regime-conditional bootstrap: dates whose regime resembled
    today's get drawn more often.

    `floor` keeps every date reachable, so a calm forecast still contains some
    turbulent days - a simulation that can only produce the current regime would
    understate risk exactly when it matters.
    """
    p = np.asarray(pd.Series(stress_probability).dropna(), dtype=float)
    if p.size == 0:
        raise EstimationError("no regime probabilities supplied")
    if not 0.0 <= current <= 1.0:
        raise EstimationError(f"current stress probability must be in [0, 1], got {current}")
    w = np.exp(-sharpness * np.abs(p - current))
    w = w / w.max()
    return (floor + (1.0 - floor) * w) / (floor + (1.0 - floor) * w).sum()
