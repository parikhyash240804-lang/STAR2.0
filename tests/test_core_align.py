"""
Alignment is a mechanism, not a habit.

The property that matters is permutation invariance: permuting the order of the
inputs must permute the outputs identically and never change their values. That
single property covers the whole legacy bug class where weights belonged to
different stocks than the risk numbers they were multiplied by.
"""
import numpy as np
import pandas as pd
import pytest

from star.core import CanonicalIndex, aligned
from star.core.errors import AlignmentError

TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"]


def make_cov(tickers, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(500, len(tickers)))
    return pd.DataFrame(np.cov(a, rowvar=False), index=tickers, columns=tickers)


def test_canonical_index_rejects_duplicates_and_emptiness():
    with pytest.raises(AlignmentError, match="duplicate"):
        CanonicalIndex.of(["A", "A"])
    with pytest.raises(AlignmentError, match="empty"):
        CanonicalIndex.of([])


def test_align_reorders_a_shuffled_series():
    canon = CanonicalIndex.of(TICKERS)
    shuffled = pd.Series([4.0, 2.0, 1.0, 3.0], index=["DDD.NS", "BBB.NS", "AAA.NS", "CCC.NS"])
    out = canon.align(shuffled, name="mu")
    assert list(out.index) == TICKERS
    assert list(out.values) == [1.0, 2.0, 3.0, 4.0]


def test_align_reorders_both_axes_of_a_covariance_matrix():
    canon = CanonicalIndex.of(TICKERS)
    shuffled_order = ["CCC.NS", "AAA.NS", "DDD.NS", "BBB.NS"]
    cov = make_cov(shuffled_order)
    out = canon.align(cov, name="cov")
    assert list(out.index) == TICKERS and list(out.columns) == TICKERS
    # the entry for (AAA, DDD) must survive the reordering unchanged
    assert out.loc["AAA.NS", "DDD.NS"] == pytest.approx(cov.loc["AAA.NS", "DDD.NS"])


def test_a_missing_ticker_is_an_error_not_a_nan():
    canon = CanonicalIndex.of(TICKERS)
    with pytest.raises(AlignmentError, match="missing"):
        canon.align(pd.Series([1.0, 2.0], index=["AAA.NS", "BBB.NS"]), name="mu")


def test_validate_rejects_an_unaligned_object_without_repairing_it():
    canon = CanonicalIndex.of(TICKERS)
    s = pd.Series(range(4), index=list(reversed(TICKERS)), dtype=float)
    with pytest.raises(AlignmentError, match="canonical order"):
        canon.validate(s, name="mu")


# --------------------------------------------------------------------------
# the decorator, and the property it buys
# --------------------------------------------------------------------------
@aligned("mu", "cov")
def portfolio_stats(index: CanonicalIndex, mu: pd.Series, cov: pd.DataFrame, w: pd.Series):
    w = index.align(w, name="w")
    return float(w @ mu), float(w.values @ cov.values @ w.values)


def test_decorator_aligns_arguments_so_results_are_permutation_invariant():
    canon = CanonicalIndex.of(TICKERS)
    mu = pd.Series([0.10, 0.12, 0.14, 0.16], index=TICKERS)
    cov = make_cov(TICKERS)
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=TICKERS)

    expected = portfolio_stats(canon, mu, cov, w)

    rng = np.random.default_rng(7)
    for _ in range(20):
        order = list(rng.permutation(TICKERS))
        got = portfolio_stats(canon, mu.reindex(order), cov.reindex(index=order, columns=order),
                              w.reindex(order))
        assert got[0] == pytest.approx(expected[0])
        assert got[1] == pytest.approx(expected[1])


def test_misaligned_inputs_would_otherwise_change_the_answer():
    """Sanity check that the property above is not vacuous."""
    canon = CanonicalIndex.of(TICKERS)
    mu = pd.Series([0.10, 0.12, 0.14, 0.16], index=TICKERS)
    cov = make_cov(TICKERS)
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=TICKERS)
    aligned_ret, _ = portfolio_stats(canon, mu, cov, w)
    # the same numbers paired with the wrong tickers (what the legacy risked)
    naive_ret = float(np.asarray(w.values) @ np.asarray(mu.reindex(reversed(TICKERS)).values))
    assert naive_ret != pytest.approx(aligned_ret)


def test_decorator_requires_a_canonical_index_argument():
    @aligned("mu")
    def f(mu: pd.Series):
        return mu

    with pytest.raises(AlignmentError, match="no CanonicalIndex"):
        f(pd.Series([1.0], index=["A"]))


def test_decorator_rejects_unknown_parameter_names_at_definition_time():
    with pytest.raises(AlignmentError, match="does not take"):
        @aligned("nope")
        def f(index: CanonicalIndex):
            return index


def test_decorated_function_advertises_what_it_aligns():
    assert portfolio_stats.__aligned_params__ == ("mu", "cov")
