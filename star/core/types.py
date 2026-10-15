"""
Units as types.

Three of the legacy engine's seven critical defects were unit confusions:

  M2-01  the Black-Litterman posterior is an EXCESS return, but downstream code
         treated it as TOTAL, so the risk-free rate was subtracted twice.
  M2-04  series named ``arith_*`` actually held GEOMETRIC means (exp of a mean
         log return) and were mixed with arithmetic ones.
  M5-04  a per-stock Ito correction was applied to an input that had already
         been converted, taking a second volatility haircut.

Bugs like these are invisible in code review because a return is just a float.
So a return estimate here is never a bare Series: it is a `ReturnEstimate` that
knows its basis (excess/total), compounding (log/simple) and periodicity
(daily/annual), and the conversions are the only way to change them. Converting
a total return to a total return raises rather than quietly succeeding, which is
precisely the operation that produced M2-01.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
import pandas as pd

from .errors import ConventionError


class Basis(StrEnum):
    """Is the risk-free rate included?"""

    TOTAL = "total"     # the whole return
    EXCESS = "excess"   # return minus the risk-free rate


class Compounding(StrEnum):
    """How the return is expressed."""

    SIMPLE = "simple"   # (P1 - P0) / P0
    LOG = "log"         # ln(P1 / P0)


class Periodicity(StrEnum):
    """Over what period."""

    DAILY = "daily"
    ANNUAL = "annual"


@dataclass(frozen=True, slots=True)
class Provenance:
    """
    Where a number came from, so the report can always answer "says who?".

    ``degradations`` records fallbacks that fired (a missing market cap, a
    truncated history, an unavailable field). The legacy printed such warnings
    to a stdout the GUI discarded, so a silently degraded run looked identical
    to a clean one.
    """

    method: str
    window: str | None = None
    as_of: pd.Timestamp | None = None
    source: str | None = None
    degradations: tuple[str, ...] = ()

    def degraded(self, note: str) -> "Provenance":
        return replace(self, degradations=self.degradations + (note,))

    def describe(self) -> str:
        bits = [self.method]
        if self.window:
            bits.append(f"window={self.window}")
        if self.as_of is not None:
            bits.append(f"as of {pd.Timestamp(self.as_of).date()}")
        if self.source:
            bits.append(f"source={self.source}")
        if self.degradations:
            bits.append(f"{len(self.degradations)} fallback(s)")
        return " · ".join(bits)


def _rf_in(
    rf_annual_simple: float | pd.Series,
    *,
    compounding: Compounding,
    periodicity: Periodicity,
    periods_per_year: float,
) -> float | pd.Series:
    """
    Convert an annual simple risk-free rate into the units of an estimate.

    The risk-free rate is *always* supplied as an annual simple rate (the way
    RBI quotes a T-bill); this is the one place that convention is interpreted.
    The legacy divided an annual simple rate by 252 to get a "daily" rate and
    also used it as a log-space intercept, which are not the same number.
    """
    rf = rf_annual_simple
    if periodicity is Periodicity.DAILY:
        # de-annualise by compounding, not by dividing
        rf = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
    if compounding is Compounding.LOG:
        rf = np.log1p(rf)
    return rf


@dataclass(frozen=True, slots=True)
class ReturnEstimate:
    """
    A vector of expected returns that knows its own units.

    Attributes
    ----------
    values : pd.Series indexed by ticker
    basis, compounding, periodicity : the units
    periods_per_year : annualisation factor from the trading calendar (never a
        hard-coded 252 - the legacy hard-coded it in 40 places)
    provenance : how this estimate was produced
    """

    values: pd.Series
    basis: Basis
    compounding: Compounding
    periodicity: Periodicity
    provenance: Provenance
    periods_per_year: float = 252.0

    def __post_init__(self) -> None:
        if not isinstance(self.values, pd.Series):
            raise ConventionError("ReturnEstimate.values must be a pandas Series")
        if self.values.index.has_duplicates:
            raise ConventionError("ReturnEstimate has duplicate tickers in its index")

    # ---- units in the description, so a mislabelled chart is impossible ----
    @property
    def units(self) -> str:
        return f"{self.periodicity.value} {self.compounding.value} {self.basis.value}"

    def _with(self, values: pd.Series, **kw) -> "ReturnEstimate":
        return replace(self, values=values, **kw)

    # ---- basis ----
    #
    # Excess return is defined MULTIPLICATIVELY: (1 + R_total) = (1 + R_excess)(1 + rf),
    # i.e. additively in log space. That is the only definition under which every
    # conversion path (basis, compounding, periodicity, in any order) gives the same
    # number. The arithmetic definition R_total - rf differs from it by rf * R_excess:
    # ~1e-8 for daily rates, ~0.4% a year at annual rates - which is exactly the
    # path-dependence a units-as-types layer exists to remove.

    def _rf_checked(self, rf_annual_simple):
        if isinstance(rf_annual_simple, pd.Series) and not rf_annual_simple.index.equals(self.values.index):
            raise ConventionError(
                "a per-asset risk-free Series must be indexed by the same tickers as the "
                "estimate; for a dated rate series pass the rate for the estimate's date")
        return _rf_in(rf_annual_simple, compounding=self.compounding,
                      periodicity=self.periodicity, periods_per_year=self.periods_per_year)

    def to_total(self, rf_annual_simple: float | pd.Series) -> "ReturnEstimate":
        """Add the risk-free rate. Raises if this estimate is already total."""
        if self.basis is Basis.TOTAL:
            raise ConventionError(
                "to_total() called on an estimate that is already TOTAL - this is "
                "the double-counting that produced legacy defect M2-01. Use "
                "ensure_total() if you do not know the basis."
            )
        rf = self._rf_checked(rf_annual_simple)
        if self.compounding is Compounding.LOG:
            return self._with(self.values + rf, basis=Basis.TOTAL)
        return self._with((1.0 + self.values) * (1.0 + rf) - 1.0, basis=Basis.TOTAL)

    def to_excess(self, rf_annual_simple: float | pd.Series) -> "ReturnEstimate":
        """Remove the risk-free rate. Raises if this estimate is already excess."""
        if self.basis is Basis.EXCESS:
            raise ConventionError(
                "to_excess() called on an estimate that is already EXCESS - the "
                "risk-free rate would be subtracted twice (legacy defect M2-01). "
                "Use ensure_excess() if you do not know the basis."
            )
        rf = self._rf_checked(rf_annual_simple)
        if self.compounding is Compounding.LOG:
            return self._with(self.values - rf, basis=Basis.EXCESS)
        return self._with((1.0 + self.values) / (1.0 + rf) - 1.0, basis=Basis.EXCESS)

    def ensure_total(self, rf_annual_simple: float | pd.Series) -> "ReturnEstimate":
        """Idempotent to_total, for boundaries where the basis is not known."""
        return self if self.basis is Basis.TOTAL else self.to_total(rf_annual_simple)

    def ensure_excess(self, rf_annual_simple: float | pd.Series) -> "ReturnEstimate":
        """Idempotent to_excess, for boundaries where the basis is not known."""
        return self if self.basis is Basis.EXCESS else self.to_excess(rf_annual_simple)

    # ---- compounding ----
    #
    # These are conversions of EXPECTED returns, so they need the variance.
    # exp(E[log R]) - 1 is the median (geometric) return, not the mean; the mean is
    # exp(mu + var/2) - 1 under lognormality. Converting without the variance is the
    # legacy's M2-04 bug (a geometric rate fed to mean-variance), so it is refused.
    # Pass variance=0.0 explicitly to ask for the median on purpose.

    def _variance_in_own_units(self, variance) -> float | pd.Series:
        if variance is None:
            raise ConventionError(
                "converting an expected return between log and simple needs the "
                "log-return variance (E[R] = exp(mu + var/2) - 1). Pass variance=... "
                "(a CovarianceEstimate, a per-ticker Series, or 0.0 for the median).")
        if isinstance(variance, CovarianceEstimate):
            cov = variance.annualise() if self.periodicity is Periodicity.ANNUAL else variance.deannualise()
            var = pd.Series(np.diag(cov.matrix.values), index=cov.tickers)
            missing = self.values.index.difference(var.index)
            if len(missing):
                raise ConventionError(f"covariance lacks variances for: {list(missing)[:8]}")
            return var.reindex(self.values.index)
        if isinstance(variance, pd.Series):
            if not variance.index.equals(self.values.index):
                raise ConventionError("variance Series must be indexed exactly like the estimate")
            return variance
        return float(variance)

    def to_simple(self, *, variance=None) -> "ReturnEstimate":
        """Expected log return -> expected simple return: exp(mu + var/2) - 1."""
        if self.compounding is Compounding.SIMPLE:
            return self
        var = self._variance_in_own_units(variance)
        return self._with(np.expm1(self.values + 0.5 * var), compounding=Compounding.SIMPLE)

    def to_log(self, *, variance=None) -> "ReturnEstimate":
        """Expected simple return -> expected log return: log(1 + m) - var/2."""
        if self.compounding is Compounding.LOG:
            return self
        if (self.values <= -1.0).any():
            raise ConventionError("cannot take log of a return <= -100%")
        var = self._variance_in_own_units(variance)
        return self._with(np.log1p(self.values) - 0.5 * var, compounding=Compounding.LOG)

    # ---- periodicity ----
    def annualise(self) -> "ReturnEstimate":
        """
        Scale to annual. Log returns scale linearly; simple returns compound.
        Doing this the other way round is the M2-04 bug class.
        """
        if self.periodicity is Periodicity.ANNUAL:
            return self
        if self.compounding is Compounding.LOG:
            return self._with(self.values * self.periods_per_year,
                              periodicity=Periodicity.ANNUAL)
        scaled = (1.0 + self.values) ** self.periods_per_year - 1.0
        return self._with(scaled, periodicity=Periodicity.ANNUAL)

    def deannualise(self) -> "ReturnEstimate":
        """Scale to daily, mirroring `annualise`."""
        if self.periodicity is Periodicity.DAILY:
            return self
        if self.compounding is Compounding.LOG:
            return self._with(self.values / self.periods_per_year,
                              periodicity=Periodicity.DAILY)
        scaled = (1.0 + self.values) ** (1.0 / self.periods_per_year) - 1.0
        return self._with(scaled, periodicity=Periodicity.DAILY)

    # ---- convenience for the optimiser, which needs one exact convention ----
    def as_annual_simple_total(self, rf_annual_simple: float | pd.Series, *, variance=None) -> pd.Series:
        """
        The single convention the optimiser consumes: annual, simple, total.

        Mean-variance needs *arithmetic* (simple) expected returns; feeding it a
        geometric mean understates every expected return, which the legacy did
        on its default path (M2-04, M3-03, M8-13). A LOG estimate therefore needs
        `variance` (a CovarianceEstimate, or a Series/float in this estimate's own
        periodicity); a SIMPLE one does not.
        """
        est = self.ensure_total(rf_annual_simple)
        if est.compounding is Compounding.SIMPLE:
            return est.annualise().values
        if not isinstance(variance, CovarianceEstimate) and variance is not None \
                and est.periodicity is Periodicity.DAILY:
            variance = variance * est.periods_per_year      # log variance scales linearly
        return est.annualise().to_simple(variance=variance).values

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n = len(self.values)
        return (f"ReturnEstimate(n={n}, {self.units}, "
                f"mean={float(self.values.mean()):+.4f}, {self.provenance.method})")


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    """
    A covariance matrix that knows its periodicity, plus how it was built.

    `matrix` is always in the stated periodicity and in LOG-return space (the
    space where scaling by the number of periods is correct).
    """

    matrix: pd.DataFrame
    periodicity: Periodicity
    provenance: Provenance
    periods_per_year: float = 252.0
    vol_method: str = "sample"
    corr_method: str = "sample"

    def __post_init__(self) -> None:
        m = self.matrix
        if m.shape[0] != m.shape[1]:
            raise ConventionError(f"covariance must be square, got {m.shape}")
        if not m.index.equals(m.columns):
            raise ConventionError("covariance index and columns must match exactly")

    @property
    def tickers(self) -> pd.Index:
        return self.matrix.index

    def annualise(self) -> "CovarianceEstimate":
        if self.periodicity is Periodicity.ANNUAL:
            return self
        return replace(self, matrix=self.matrix * self.periods_per_year,
                       periodicity=Periodicity.ANNUAL)

    def deannualise(self) -> "CovarianceEstimate":
        if self.periodicity is Periodicity.DAILY:
            return self
        return replace(self, matrix=self.matrix / self.periods_per_year,
                       periodicity=Periodicity.DAILY)

    def volatilities(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.matrix.values)), index=self.tickers)

    def correlation(self) -> pd.DataFrame:
        sd = np.sqrt(np.diag(self.matrix.values))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = self.matrix.values / np.outer(sd, sd)
        return pd.DataFrame(corr, index=self.tickers, columns=self.tickers)

    def portfolio_variance(self, weights: pd.Series) -> float:
        extra = weights.index.difference(self.tickers)
        if len(extra):
            from .errors import AlignmentError
            raise AlignmentError(f"weights name tickers the covariance does not have: {list(extra)[:8]}")
        w = weights.reindex(self.tickers)
        if w.isna().any():
            missing = list(w.index[w.isna()])
            raise AlignmentErrorLazy(missing)
        return float(w.values @ self.matrix.values @ w.values)

    def min_eigenvalue(self) -> float:
        return float(np.linalg.eigvalsh(self.matrix.values).min())

    def is_psd(self, tol: float = -1e-10) -> bool:
        return self.min_eigenvalue() >= tol


def AlignmentErrorLazy(missing):  # noqa: N802 - tiny helper to keep imports flat
    from .errors import AlignmentError
    return AlignmentError(f"weights are missing covariance entries for: {missing}")
