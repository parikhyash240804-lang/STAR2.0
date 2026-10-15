"""Risk measures at a stated horizon, and risk split by holding."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from star.core.errors import EstimationError
from star.core.types import CovarianceEstimate, Periodicity, Provenance
from star.data.fixture import FixtureProvider
from star.risk import metrics as M
from star.risk.metrics import (HorizonSample, beta_and_tracking_error, diversification_ratio,
                               tail_weights,
                               drawdown_at_risk, expected_shortfall, horizon_sample_from_daily,
                               risk_report, shortfall_contributions, tail_reliability,
                               value_at_risk, volatility_contributions)


def _normal_sample(n=200_000, sd=0.01, seed=3, horizon=1):
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(0, sd, n), index=pd.bdate_range("2000-01-03", periods=n))
    return HorizonSample(returns=r, horizon_days=horizon, provenance=Provenance(method="simulated"))


def test_var_and_es_match_the_closed_form_for_a_known_distribution():
    sd = 0.01
    s = _normal_sample(sd=sd)
    for c in (0.95, 0.975, 0.99):
        z = stats.norm.ppf(c)
        assert value_at_risk(s, c) == pytest.approx(sd * z, rel=0.02)
        # ES of a normal = sigma * phi(z) / (1 - c)
        assert expected_shortfall(s, c) == pytest.approx(sd * stats.norm.pdf(z) / (1 - c), rel=0.02)


def test_tail_weights_select_exactly_the_worst_fraction():
    r = np.arange(100.0)                      # 0 is the worst outcome, 99 the best
    w = tail_weights(r, 0.95)
    assert w.sum() == pytest.approx(1.0)
    assert np.count_nonzero(w) == 5 and np.allclose(w[:5], 0.2)
    assert w[5:].max() == 0.0
    # a fractional tail takes part of the next observation rather than rounding
    w = tail_weights(r, 0.975)
    assert w.sum() == pytest.approx(1.0)
    assert np.count_nonzero(w) == 3 and w[2] == pytest.approx(0.5 / 2.5)


def test_expected_shortfall_is_never_smaller_than_value_at_risk():
    s = _normal_sample(n=5000)
    for c in (0.90, 0.95, 0.99):
        assert expected_shortfall(s, c) >= value_at_risk(s, c)


def test_expected_shortfall_is_coherent_where_value_at_risk_is_not():
    """
    The textbook reason ES is the headline: two independent bonds, each defaulting
    with 4% probability. At 95%, VaR says the diversified pair is *riskier* than
    the sum of its parts. ES does not.
    """
    rng = np.random.default_rng(1)
    n = 400_000
    a = np.where(rng.random(n) < 0.04, -1.0, 0.02)
    b = np.where(rng.random(n) < 0.04, -1.0, 0.02)
    combined = pd.Series(a + b)
    va, vb, vab = (value_at_risk(pd.Series(x), 0.95) for x in (a, b, combined))
    assert vab > va + vb                       # VaR is not subadditive
    ea, eb, eab = (expected_shortfall(pd.Series(x), 0.95) for x in (a, b, combined))
    assert eab <= ea + eb + 1e-9               # ES always is
    # and the quantile here sits on a repeated value, which is where a naive
    # "average everything below the quantile" estimator collapses
    assert ea > 0.5


def test_there_is_no_way_to_scale_a_one_day_figure_to_another_horizon():
    """Legacy M6-02 turned a 10% annual loss into 159% with a sqrt(252)."""
    public = [n for n in dir(M) if not n.startswith("_")]
    assert not [n for n in public if "scale" in n.lower() or "annualise_var" in n.lower()]


def test_the_square_root_rule_would_have_been_materially_wrong_on_real_data():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    s = p.benchmark("^NSEI").dropna()
    daily = (s / s.shift(1) - 1).dropna()
    one = horizon_sample_from_daily(daily, 1)
    ten = horizon_sample_from_daily(daily, 10)
    honest = expected_shortfall(ten, 0.975)
    shortcut = expected_shortfall(one, 0.975) * np.sqrt(10)
    assert honest > 0 and shortcut > 0
    assert abs(honest - shortcut) / honest > 0.05      # the shortcut is not a rounding difference


def test_horizon_samples_compound_actual_days():
    idx = pd.bdate_range("2024-01-01", periods=300)
    r = pd.Series(0.01, index=idx)                      # 1% every day
    s = horizon_sample_from_daily(r, 5)
    assert s.horizon_days == 5
    assert s.returns.iloc[-1] == pytest.approx(1.01 ** 5 - 1)      # compounded, not 5%
    assert s.overlapping and s.effective_observations == pytest.approx(len(s.returns) / 5)
    one = horizon_sample_from_daily(r, 1)
    assert not one.overlapping and one.effective_observations == len(one.returns)


def test_non_overlapping_windows_are_available_and_independent():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 2000), index=pd.bdate_range("2015-01-01", periods=2000))
    s = horizon_sample_from_daily(r, 21, overlapping=False)
    assert not s.overlapping
    assert s.effective_observations == len(s.returns) == 2000 // 21
    with pytest.raises(EstimationError, match="non-overlapping"):
        horizon_sample_from_daily(r.iloc[:400], 21, overlapping=False)


def test_a_thin_tail_is_flagged_rather_than_quoted_precisely():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300))
    s = horizon_sample_from_daily(r, 10)                # ~30 effective observations
    ok, note = tail_reliability(s, 0.99)
    assert not ok and "indicative" in note
    assert tail_reliability(horizon_sample_from_daily(r, 1), 0.95)[0]
    rep = risk_report(s, confidences=(0.95, 0.99))
    assert rep.warnings and rep.provenance.degradations


def test_risk_report_states_its_horizon_and_never_hides_it():
    s = _normal_sample(n=5000, horizon=1)
    rep = risk_report(s)
    assert rep.horizon_days == 1
    assert "Over day" in rep.headline or "Over 1 day" in rep.headline or "day" in rep.headline
    assert set(rep.expected_shortfall) == {0.95, 0.975, 0.99}
    assert rep.expected_shortfall[0.99] > rep.expected_shortfall[0.95]
    assert "no sqrt-time scaling" in rep.provenance.method
    assert rep.annual_volatility == pytest.approx(rep.volatility * np.sqrt(252), rel=1e-9)


# ------------------------------------------------------------ attribution ---
def _cov(n=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 2))
    m = (f @ f.T / 2 + np.diag(rng.uniform(0.02, 0.06, n))) * 0.05
    names = [f"S{i}" for i in range(n)]
    return CovarianceEstimate(matrix=pd.DataFrame(m, index=names, columns=names),
                              periodicity=Periodicity.ANNUAL, provenance=Provenance(method="t"))


def test_volatility_contributions_add_up_to_the_portfolio_volatility():
    cov = _cov()
    w = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1], index=cov.matrix.columns)
    c = volatility_contributions(w, cov)
    total = float(np.sqrt(w.to_numpy() @ cov.matrix.to_numpy() @ w.to_numpy()))
    assert c["contribution"].sum() == pytest.approx(total)
    assert c["share"].sum() == pytest.approx(1.0)
    assert c.loc["S0", "share"] > c.loc["S4", "share"]          # the big position carries more


def test_shortfall_contributions_add_up_to_expected_shortfall():
    rng = np.random.default_rng(2)
    names = [f"S{i}" for i in range(4)]
    R = pd.DataFrame(rng.multivariate_normal(np.zeros(4), np.eye(4) * 0.0004
                                             + 0.0002, 4000), columns=names)
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=names)
    c = shortfall_contributions(R, w, 0.975)
    port = pd.Series(R.to_numpy() @ w.to_numpy())
    assert c["contribution"].sum() == pytest.approx(expected_shortfall(port, 0.975), rel=1e-9)
    assert c["share"].sum() == pytest.approx(1.0)
    assert c.index[0] == c["contribution"].idxmax()             # sorted worst first


def test_diversification_ratio_behaves_as_correlation_rises():
    names = ["A", "B", "C"]
    def cov_at(rho):
        sd = np.array([0.2, 0.2, 0.2])
        c = np.full((3, 3), rho)
        np.fill_diagonal(c, 1.0)
        return CovarianceEstimate(matrix=pd.DataFrame(c * np.outer(sd, sd), index=names, columns=names),
                                  periodicity=Periodicity.ANNUAL, provenance=Provenance(method="t"))
    w = pd.Series(1 / 3, index=names)
    assert diversification_ratio(w, cov_at(1.0)) == pytest.approx(1.0)      # no benefit at all
    assert diversification_ratio(w, cov_at(0.5)) > 1.0
    assert diversification_ratio(w, cov_at(0.0)) > diversification_ratio(w, cov_at(0.5))


def test_drawdown_at_risk_is_measured_along_the_path():
    rising = np.cumprod(np.full((100, 50), 1.001), axis=0)
    dar, cdar = drawdown_at_risk(rising, 0.95)
    assert dar == pytest.approx(0.0, abs=1e-12)
    path = np.ones((5, 1))
    path[:, 0] = [100, 120, 60, 80, 90]                          # a 50% fall from the peak
    dar, cdar = drawdown_at_risk(np.repeat(path, 40, axis=1), 0.95)
    assert dar == pytest.approx(0.5)
    assert cdar >= dar
    with pytest.raises(EstimationError, match="at least two steps"):
        drawdown_at_risk(np.ones((1, 10)))


def test_beta_and_tracking_error_are_what_they_claim():
    rng = np.random.default_rng(4)
    b = pd.Series(rng.normal(0, 0.01, 600))
    assert beta_and_tracking_error(b, b) == pytest.approx((1.0, 0.0), abs=1e-12)
    beta, te = beta_and_tracking_error(2 * b, b)
    assert beta == pytest.approx(2.0)
    assert te == pytest.approx(float(b.std(ddof=1) * np.sqrt(252)), rel=1e-9)
    with pytest.raises(EstimationError, match="at least 30"):
        beta_and_tracking_error(b.iloc[:10], b.iloc[:10])


def test_measures_refuse_impossible_confidences():
    s = _normal_sample(n=1000)
    for bad in (0.4, 1.0, 1.5):
        with pytest.raises(EstimationError, match="confidence"):
            value_at_risk(s, bad)
    with pytest.raises(EstimationError, match="at least 30 observations"):
        HorizonSample(returns=pd.Series([0.01] * 10), horizon_days=1,
                      provenance=Provenance(method="t"))
