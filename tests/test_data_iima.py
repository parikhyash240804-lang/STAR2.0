"""
IIMA factor library: four factors plus the RBI 91-day T-bill rate.

The yield checks pin known Indian money-market history, so a misreading of the
file's units (percent, holding-period returns over calendar days) cannot pass.
"""
import pandas as pd
import pytest

from star.core.errors import DataError
from star.data.iima import FACTOR_COLUMNS, IimaFactors


@pytest.fixture(scope="module")
def iima():
    try:
        return IimaFactors.load()
    except DataError:
        pytest.skip("bundled IIMA file not present")


def test_it_is_the_survivorship_adjusted_four_factor_release(iima):
    assert iima.survivorship_adjusted
    assert tuple(iima.factors.columns) == FACTOR_COLUMNS          # Carhart four, not FF5
    assert iima.release == "2025-12"


@pytest.mark.parametrize("date, low, high", [
    ("2013-08-30", 0.10, 0.125),     # taper tantrum: RBI squeezed short rates
    ("2020-05-29", 0.025, 0.040),    # COVID easing
    ("2021-01-15", 0.025, 0.040),    # the legacy charged a flat 6.5% here
    ("2025-12-31", 0.045, 0.060),
])
def test_tbill_yields_match_known_indian_history(iima, date, low, high):
    y = iima.tbill_yield.loc[:date].iloc[-1]
    assert low < y < high


def test_weekend_rows_are_not_mistaken_for_triple_rates(iima):
    """RF is a holding-period return: Monday's row covers three calendar days."""
    y = iima.tbill_yield.loc["2025-12-01":"2025-12-31"]
    assert y.max() - y.min() < 0.005


def test_risk_free_curve_is_effective_and_as_of(iima):
    c = iima.risk_free_curve()
    simple = iima.tbill_yield.loc[:"2021-01-31"].iloc[-1]
    assert c.rate("2021-02-15") == pytest.approx((1 + simple * 91 / 365) ** (365 / 91) - 1, abs=1e-9)
    assert any("91-day" in n for n in c.notes)
    with pytest.raises(DataError):
        c.rate("1990-01-01")


def test_carrying_the_rate_forward_is_reported(iima):
    assert iima.staleness_note("2026-01-10") is None
    note = iima.staleness_note("2026-09-11")
    assert note and "carried forward" in note
    assert iima.provenance("2026-09-11").degradations


def test_factor_returns_are_decimals_not_percent(iima):
    daily = iima.factors.loc["2016":"2025"]
    assert daily.abs().max().max() < 0.2                              # a 20% daily factor move is not plausible
    assert 0.03 < daily["MF"].mean() * 250 < 0.20                      # market premium of a few % a year
