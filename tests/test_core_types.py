"""
Units-as-types tests.

The headline test is `test_M2_01_*`: the legacy engine subtracted the risk-free
rate twice because an excess return was indistinguishable from a total one. Here
that operation raises.
"""
import numpy as np
import pandas as pd
import pytest

from star.core import (
    Basis, Compounding, CovarianceEstimate, Periodicity, Provenance, ReturnEstimate,
)
from star.core.errors import AlignmentError, ConventionError

TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS"]
RF = 0.065


def est(values=(0.10, 0.15, 0.20), basis=Basis.TOTAL,
        compounding=Compounding.SIMPLE, periodicity=Periodicity.ANNUAL):
    return ReturnEstimate(
        values=pd.Series(values, index=TICKERS),
        basis=basis, compounding=compounding, periodicity=periodicity,
        provenance=Provenance(method="test"),
    )


# --------------------------------------------------------------------------
# the defect this module exists to prevent
# --------------------------------------------------------------------------
def test_M2_01_double_rf_subtraction_is_impossible():
    excess = est(basis=Basis.EXCESS)
    with pytest.raises(ConventionError, match="already EXCESS"):
        excess.to_excess(RF)


def test_M2_01_double_rf_addition_is_impossible():
    total = est(basis=Basis.TOTAL)
    with pytest.raises(ConventionError, match="already TOTAL"):
        total.to_total(RF)


def test_ensure_variants_are_idempotent_at_boundaries():
    total = est(basis=Basis.TOTAL)
    assert total.ensure_total(RF) is total
    excess = total.ensure_excess(RF)
    assert excess.basis is Basis.EXCESS
    assert excess.ensure_excess(RF) is excess


def test_basis_round_trip_recovers_the_original():
    total = est(basis=Basis.TOTAL)
    back = total.to_excess(RF).to_total(RF)
    pd.testing.assert_series_equal(back.values, total.values)
    assert back.basis is Basis.TOTAL


# --------------------------------------------------------------------------
# risk-free conversion: the rf is always an ANNUAL SIMPLE rate
# --------------------------------------------------------------------------
def test_rf_is_compounded_not_divided_for_daily_simple():
    daily = est(values=(0.0, 0.0, 0.0), basis=Basis.EXCESS,
                compounding=Compounding.SIMPLE, periodicity=Periodicity.DAILY)
    got = daily.to_total(RF).values.iloc[0]
    assert got == pytest.approx((1 + RF) ** (1 / 252) - 1)
    # the legacy's rf/252 is a different number, and bigger
    assert got < RF / 252


def test_rf_enters_log_space_as_log1p():
    daily_log = est(values=(0.0, 0.0, 0.0), basis=Basis.EXCESS,
                    compounding=Compounding.LOG, periodicity=Periodicity.DAILY)
    expected = np.log1p((1 + RF) ** (1 / 252) - 1)
    assert daily_log.to_total(RF).values.iloc[0] == pytest.approx(expected)


# --------------------------------------------------------------------------
# compounding and periodicity
# --------------------------------------------------------------------------
def test_log_returns_annualise_linearly_simple_returns_compound():
    mu_d = 0.0005
    log_daily = est(values=(mu_d,) * 3, compounding=Compounding.LOG,
                    periodicity=Periodicity.DAILY)
    assert log_daily.annualise().values.iloc[0] == pytest.approx(mu_d * 252)

    simple_daily = est(values=(mu_d,) * 3, compounding=Compounding.SIMPLE,
                       periodicity=Periodicity.DAILY)
    assert simple_daily.annualise().values.iloc[0] == pytest.approx((1 + mu_d) ** 252 - 1)


@pytest.mark.parametrize("compounding", [Compounding.LOG, Compounding.SIMPLE])
def test_annualise_deannualise_round_trip(compounding):
    e = est(values=(0.0004, 0.0006, 0.0008), compounding=compounding,
            periodicity=Periodicity.DAILY)
    back = e.annualise().deannualise()
    pd.testing.assert_series_equal(back.values, e.values)


def test_M2_04_geometric_is_not_silently_used_as_arithmetic():
    """
    A log-daily mean annualised and exponentiated is a GEOMETRIC (CAGR) return.
    The legacy called such series `arith_*` and fed them to mean-variance, which
    needs arithmetic returns. The two differ, and the difference must be visible.
    """
    mu_d, sigma_d = 0.0005, 0.012
    log_daily = est(values=(mu_d,) * 3, basis=Basis.TOTAL,
                    compounding=Compounding.LOG, periodicity=Periodicity.DAILY)
    # converting an EXPECTED log return without its variance is refused outright
    with pytest.raises(ConventionError, match="variance"):
        log_daily.annualise().to_simple()
    # asking for the median explicitly gives the geometric rate...
    geometric = log_daily.annualise().to_simple(variance=0.0).values.iloc[0]
    assert geometric == pytest.approx(np.expm1(mu_d * 252))
    # ...and supplying the variance gives the arithmetic mean the optimiser needs
    arithmetic = log_daily.annualise().to_simple(variance=sigma_d ** 2 * 252).values.iloc[0]
    assert arithmetic == pytest.approx(np.expm1(mu_d * 252 + 0.5 * sigma_d ** 2 * 252))
    assert arithmetic > geometric


def test_units_string_names_every_convention():
    assert est(basis=Basis.EXCESS, compounding=Compounding.LOG,
               periodicity=Periodicity.DAILY).units == "daily log excess"


def test_duplicate_tickers_are_rejected():
    with pytest.raises(ConventionError, match="duplicate"):
        ReturnEstimate(values=pd.Series([0.1, 0.2], index=["A", "A"]),
                       basis=Basis.TOTAL, compounding=Compounding.SIMPLE,
                       periodicity=Periodicity.ANNUAL, provenance=Provenance("t"))


# --------------------------------------------------------------------------
# covariance
# --------------------------------------------------------------------------
def cov_est(periodicity=Periodicity.DAILY):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 3))
    m = np.cov(a, rowvar=False)
    return CovarianceEstimate(
        matrix=pd.DataFrame(m, index=TICKERS, columns=TICKERS),
        periodicity=periodicity, provenance=Provenance(method="sample"),
    )


def test_covariance_annualisation_scales_by_periods_per_year():
    daily = cov_est()
    annual = daily.annualise()
    assert annual.matrix.iloc[0, 0] == pytest.approx(daily.matrix.iloc[0, 0] * 252)
    pd.testing.assert_frame_equal(annual.deannualise().matrix, daily.matrix)


def test_correlation_diagonal_is_one_and_psd_holds():
    c = cov_est()
    assert np.allclose(np.diag(c.correlation().values), 1.0)
    assert c.is_psd()
    assert c.min_eigenvalue() > 0


def test_portfolio_variance_matches_quadratic_form():
    c = cov_est()
    w = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    assert c.portfolio_variance(w) == pytest.approx(float(w.values @ c.matrix.values @ w.values))


def test_portfolio_variance_rejects_unknown_tickers():
    c = cov_est()
    with pytest.raises(AlignmentError):
        c.portfolio_variance(pd.Series([1.0], index=["ZZZ.NS"]))


def test_non_square_covariance_is_rejected():
    with pytest.raises(ConventionError, match="square"):
        CovarianceEstimate(matrix=pd.DataFrame(np.zeros((2, 3))),
                           periodicity=Periodicity.DAILY,
                           provenance=Provenance("t"))
