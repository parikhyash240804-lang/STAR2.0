"""
One simulation, every horizon.

The output is a `HorizonTermStructure`: every statistic stored **per day** out to
the maximum horizon, so asking for 2 days, 37 days or a year is a lookup. That is
what makes the horizon control in the brief instant, and it is also the structural
cure for the legacy's contradictory figures - a 1-day number and a 1-year number
that could not both be right came from different code paths (M6-01/02). Here they
are two rows of the same array, and a test asserts the fan chart and the table
read the same cell.

Three design points worth stating plainly:

**Drift is an assumption, and it is the largest one.** The legacy fed the
optimiser's own expected returns into the simulation, so the forecast confirmed
the view it had been given (M5-03). Here the default assumes **no alpha**: the
portfolio is expected to earn the risk-free rate plus a shrunk equity risk
premium, nothing more. Any other choice is named in the output, and
`drift_sensitivity` shows how the probability of loss moves across the range, so
a reader can see how much of the conclusion rests on it.

**Drawdown is computed along the path**, day by day, not sliced out of the
endpoint distribution. A portfolio that ends a year flat can still have fallen 30%
in the middle, and that is the number that gets people fired.

**Memory is budgeted, not hoped for.** Only the portfolio value matrix
(days x paths, float32) and one chunk of generated returns are ever held. The
asset-level cube is never built. A run whose matrix would exceed the budget is
refused with the arithmetic, rather than discovering it as a swap storm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.horizons import HorizonGrid
from ..core.types import Provenance

MAX_HORIZON = 756                     # ~3 years; beyond this a 5-year sample is extrapolation
LONG_HORIZON_WARNING = 504            # ~2 years
DEFAULT_QUANTILES = (1, 5, 25, 50, 75, 95, 99)
DEFAULT_ES = (95, 99)
DEFAULT_DAR = (50, 95)
CHUNK_PATHS = 2_000                   # fixed, so the answer never depends on the machine


@dataclass(frozen=True)
class DriftAssumption:
    """What the simulation assumes about expected return, stated in one line."""

    kind: str = "rf_plus_erp"          # rf_plus_erp | risk_free | zero | historical | custom
    risk_free: float = 0.06            # annual simple
    equity_risk_premium: float = 0.05
    shrinkage: float = 0.5             # how much of the premium is assumed earned
    custom_annual_return: float | None = None

    def annual_simple(self) -> float | None:
        if self.kind == "historical":
            return None                 # keep whatever the history says
        if self.kind == "zero":
            return 0.0
        if self.kind == "risk_free":
            return self.risk_free
        if self.kind == "rf_plus_erp":
            return self.risk_free + self.shrinkage * self.equity_risk_premium
        if self.kind == "custom":
            if self.custom_annual_return is None:
                raise EstimationError("a custom drift needs custom_annual_return")
            return float(self.custom_annual_return)
        raise EstimationError(f"unknown drift kind {self.kind!r}")

    def describe(self) -> str:
        if self.kind == "historical":
            return ("drift: the sample's own average return (assumes the past repeats - the legacy's "
                    "implicit assumption)")
        target = self.annual_simple()
        if self.kind == "rf_plus_erp":
            return (f"drift: no alpha assumed - risk-free {self.risk_free:.1%} plus "
                    f"{self.shrinkage:.0%} of a {self.equity_risk_premium:.1%} equity risk premium "
                    f"= {target:.1%} a year")
        return f"drift: {self.kind.replace('_', ' ')}, {target:.1%} a year"


@dataclass(frozen=True)
class HorizonTermStructure:
    """Every forward statistic, per day, from one run."""

    days: np.ndarray
    quantiles: dict[int, np.ndarray]          # percentile -> portfolio value per day
    es: dict[int, np.ndarray]                 # confidence -> expected shortfall (loss fraction)
    p_loss: np.ndarray
    p_below_rf: np.ndarray
    p_below_benchmark: np.ndarray | None
    dar: dict[int, np.ndarray]                # percentile -> drawdown-at-risk per day
    initial_value: float
    drift: DriftAssumption
    n_paths: int
    grid: HorizonGrid
    provenance: Provenance
    warnings: tuple[str, ...] = ()
    generator: str = ""

    @property
    def max_days(self) -> int:
        return int(self.days[-1])

    def _row(self, days: int) -> int:
        d = int(days)
        if not 1 <= d <= self.max_days:
            raise EstimationError(f"horizon {d} is outside this run (1..{self.max_days}); "
                                  f"re-run with a longer horizon rather than extrapolating")
        return d - 1

    def at(self, days: int) -> dict:
        """Everything about one horizon. The single source every view reads."""
        i = self._row(days)
        v0 = self.initial_value
        return {
            "days": int(days),
            "label": self.grid.label_for(int(days)) if int(days) in self.grid.days else f"{days}D",
            "median": float(self.quantiles[50][i]),
            "median_return": float(self.quantiles[50][i] / v0 - 1.0),
            "quantiles": {q: float(self.quantiles[q][i]) for q in self.quantiles},
            "returns": {q: float(self.quantiles[q][i] / v0 - 1.0) for q in self.quantiles},
            "expected_shortfall": {c: float(self.es[c][i]) for c in self.es},
            "p_loss": float(self.p_loss[i]),
            "p_below_rf": float(self.p_below_rf[i]),
            "p_below_benchmark": (None if self.p_below_benchmark is None
                                  else float(self.p_below_benchmark[i])),
            "drawdown_at_risk": {q: float(self.dar[q][i]) for q in self.dar},
        }

    def band(self, lo: int = 5, hi: int = 95) -> pd.DataFrame:
        """The fan chart's data: median and an outer band, per day."""
        if lo not in self.quantiles or hi not in self.quantiles:
            raise EstimationError(f"quantiles {lo} and {hi} were not stored; available: "
                                  f"{sorted(self.quantiles)}")
        return pd.DataFrame({"day": self.days, "low": self.quantiles[lo],
                             "median": self.quantiles[50], "high": self.quantiles[hi]}).set_index("day")

    def table(self) -> pd.DataFrame:
        """One row per labelled horizon - what the Excel summary and the brief show."""
        rows = [self.at(d) for d in self.grid.days if d <= self.max_days]
        return pd.DataFrame([{
            "label": r["label"], "days": r["days"], "median_return": r["median_return"],
            "p5_return": r["returns"][5], "p95_return": r["returns"][95],
            "p_loss": r["p_loss"], "p_below_rf": r["p_below_rf"],
            "p_below_benchmark": r["p_below_benchmark"],
            "expected_shortfall_95": r["expected_shortfall"][95],
            "drawdown_at_risk_95": r["drawdown_at_risk"][95],
        } for r in rows]).set_index("label")

    def headline(self, days: int) -> str:
        r = self.at(days)
        period = "day" if days == 1 else f"{days} days"
        return (f"Over {period}, the middle half of outcomes lies between "
                f"{r['returns'][25]:+.1%} and {r['returns'][75]:+.1%}, with a "
                f"{r['p_loss']:.0%} chance of ending below where it started.")


def _plan_chunks(horizon: int, n_paths: int, max_bytes: float,
                 chunk_paths: int = CHUNK_PATHS) -> int:
    """
    Paths generated at a time, checked against an explicit memory budget.

    The chunk size is **fixed rather than derived from the budget**, so the same
    seed gives the same answer on a laptop and on a server; a chunk size that
    floated with available memory would make runs unreproducible across machines,
    which is worse than using a little more memory.

    The value matrix is (horizon + 1) x n_paths float32 and is unavoidable - it is
    what the quantiles are computed from. Generation needs a handful of temporary
    arrays of (horizon x chunk) on top, and the budget has to cover both.
    """
    per_path = (horizon + 1) * 4
    matrix_bytes = per_path * n_paths
    chunk = int(min(max(chunk_paths, 1), n_paths))
    working_bytes = 5 * horizon * chunk * 8
    if matrix_bytes + working_bytes > max_bytes:
        affordable = int(max((max_bytes - working_bytes) / per_path, 0))
        raise EstimationError(
            f"{n_paths:,} paths over {horizon} days needs {matrix_bytes / 1e6:.0f} MB for the value "
            f"matrix plus {working_bytes / 1e6:.0f} MB to generate, above the "
            f"{max_bytes / 1e6:.0f} MB budget. Reduce paths to about {affordable:,}, "
            f"shorten the horizon, or raise max_bytes deliberately.")
    return chunk


def simulate(generator, *, horizon: int, n_paths: int = 20_000, initial_value: float = 1.0,
             drift: DriftAssumption = DriftAssumption(), grid: HorizonGrid | None = None,
             seed: int = 20260920, max_bytes: float = 512e6, chunk_paths: int = CHUNK_PATHS,
             periods_per_year: float = 252.0) -> HorizonTermStructure:
    """
    Run one simulation and return every horizon at once.

    `generator.generate(horizon, n_paths, rng)` must return portfolio daily **log**
    returns of shape (horizon, n_paths), and either `None` or benchmark log returns
    on the same draws.
    """
    if not 1 <= int(horizon) <= MAX_HORIZON:
        raise EstimationError(f"horizon must be between 1 and {MAX_HORIZON} days, got {horizon}")
    if n_paths < 1_000:
        raise EstimationError(f"{n_paths} paths is too few for a 99th percentile; use at least 1,000")
    horizon = int(horizon)
    grid = grid or HorizonGrid.default(periods_per_year=periods_per_year)
    chunk = _plan_chunks(horizon, n_paths, max_bytes, chunk_paths)
    rng = np.random.default_rng(seed)

    values = np.empty((horizon + 1, n_paths), dtype=np.float32)
    values[0] = initial_value
    max_dd = np.empty((horizon, n_paths), dtype=np.float32)
    bench_values = None
    target = drift.annual_simple()

    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        port, bench = generator.generate(horizon, m, rng)
        port = np.asarray(port, dtype=float)
        if port.shape != (horizon, m):
            raise EstimationError(f"generator returned {port.shape}, expected {(horizon, m)}")
        if target is not None:
            # centre the sample, then impose the stated expectation. The half-variance
            # term makes E[value] hit the target, not the median - the distinction that
            # separates an average return from a typical one.
            daily_var = float(np.var(port))
            mu_log = (np.log1p(target) - 0.5 * daily_var * periods_per_year) / periods_per_year
            port = port - port.mean() + mu_log
        cum = np.cumsum(port, axis=0)
        block = initial_value * np.exp(cum)
        values[1:, done:done + m] = block.astype(np.float32)
        running_max = np.maximum.accumulate(np.vstack([np.full((1, m), initial_value), block]), axis=0)
        dd = 1.0 - np.vstack([np.full((1, m), initial_value), block]) / running_max
        max_dd[:, done:done + m] = np.maximum.accumulate(dd, axis=0)[1:].astype(np.float32)
        if bench is not None:
            if bench_values is None:
                bench_values = np.empty((horizon, n_paths), dtype=np.float32)
            bench_values[:, done:done + m] = np.exp(np.cumsum(np.asarray(bench, float), axis=0)).astype(np.float32)
        done += m

    body = values[1:]
    quantiles = {q: np.percentile(body, q, axis=1).astype(float) for q in DEFAULT_QUANTILES}
    es = {}
    for c in DEFAULT_ES:
        k = max(int(round((1 - c / 100) * n_paths)), 1)
        worst = np.partition(body, k - 1, axis=1)[:, :k]
        es[c] = (1.0 - worst.mean(axis=1) / initial_value).astype(float)
    p_loss = (body < initial_value).mean(axis=1).astype(float)

    days = np.arange(1, horizon + 1)
    rf_path = initial_value * (1.0 + drift.risk_free) ** (days / periods_per_year)
    p_below_rf = (body < rf_path[:, None]).mean(axis=1).astype(float)
    p_below_benchmark = None if bench_values is None else \
        (body / initial_value < bench_values).mean(axis=1).astype(float)
    dar = {q: np.percentile(max_dd, q, axis=1).astype(float) for q in DEFAULT_DAR}

    warnings = []
    if horizon > LONG_HORIZON_WARNING:
        warnings.append(f"beyond about two years a five-year sample contains only a handful of "
                        f"independent {horizon}-day periods: treat the {horizon}-day figures as "
                        f"extrapolation, not measurement")
    if bench_values is None:
        warnings.append("no benchmark was simulated, so the chance of underperforming the index "
                        "is not available")
    prov = Provenance(
        method=f"forward distribution from {getattr(generator, 'description', type(generator).__name__)}",
        window=f"{n_paths:,} paths x {horizon} days (seed {seed}, chunk {chunk}), {drift.describe()}",
        source="simulated", degradations=tuple(warnings))
    return HorizonTermStructure(days=days, quantiles=quantiles, es=es, p_loss=p_loss,
                                p_below_rf=p_below_rf, p_below_benchmark=p_below_benchmark,
                                dar=dar, initial_value=float(initial_value), drift=drift,
                                n_paths=int(n_paths), grid=grid, provenance=prov,
                                warnings=tuple(warnings),
                                generator=getattr(generator, "description", ""))


def drift_sensitivity(generator, *, horizon: int, drifts: tuple[DriftAssumption, ...],
                      n_paths: int = 5_000, **kwargs) -> pd.DataFrame:
    """
    The same simulation under each drift assumption - the strip that shows how much
    of the conclusion is the data and how much is the assumption.
    """
    rows = []
    for d in drifts:
        ts = simulate(generator, horizon=horizon, n_paths=n_paths, drift=d, **kwargs)
        r = ts.at(horizon)
        rows.append({"drift": d.describe(), "kind": d.kind,
                     "annual_assumed": d.annual_simple(), "median_return": r["median_return"],
                     "p_loss": r["p_loss"], "p5_return": r["returns"][5],
                     "expected_shortfall_95": r["expected_shortfall"][95]})
    return pd.DataFrame(rows).set_index("kind")
