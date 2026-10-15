"""The walk-forward loop: no look-ahead, real drift, real costs, honest warnings."""
import numpy as np
import pandas as pd
import pytest

from star.backtest.engine import BacktestSettings, rebalance_dates, run_backtest
from star.core.errors import DataError
from star.data.fixture import FixtureProvider
from star.data.provider import PricePanel
from star.data.quality import assess


def _panel(n_names=6, n_days=900, seed=0):
    """A real PricePanel, so the engine is exercised through the type it ships with."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    names = [f"S{i}" for i in range(n_names)]
    r = rng.normal(0.0004, 0.012, (n_days, n_names))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=names)
    return PricePanel(close=close, close_raw=close, high=close * 1.005, low=close * 0.995,
                      open_=close, volume=pd.DataFrame(1e7, index=idx, columns=names),
                      source="synthetic", as_of=idx[-1])


def equal_weight_strategy(view, as_of, prev_weights, universe):
    return pd.Series(1.0 / len(universe), index=list(universe))


def test_rebalance_dates_respect_the_warm_up_and_the_frequency():
    p = _panel(n_days=900)
    q = rebalance_dates(p.sessions, "Q", warmup=504)
    m = rebalance_dates(p.sessions, "M", warmup=504)
    assert q[0] >= p.sessions[504]
    assert len(m) > len(q) > 2
    assert all(d in p.sessions for d in q)
    with pytest.raises(DataError, match="rebalance must be one of"):
        rebalance_dates(p.sessions, "fortnightly", warmup=100)
    with pytest.raises(DataError, match="not enough for a"):
        rebalance_dates(p.sessions[:100], "Q", warmup=504)


def test_the_strategy_never_sees_a_price_after_the_rebalance_date():
    """The structural guarantee: a look-ahead would have to get past the DataView."""
    p = _panel()
    seen = []

    def peeking_strategy(view, as_of, prev_weights, universe):
        close = view.ohlc()["close"]
        seen.append((as_of, close.index.max()))
        return equal_weight_strategy(view, as_of, prev_weights, universe)

    run_backtest(p, peeking_strategy, settings=BacktestSettings(warmup_sessions=300))
    assert seen
    for as_of, latest in seen:
        assert latest <= as_of


def test_weights_drift_between_rebalances_and_that_drift_costs_money():
    """The legacy rebalanced to target every day for free (M8-09)."""
    p = _panel(seed=3)
    res = run_backtest(p, equal_weight_strategy, settings=BacktestSettings(warmup_sessions=300))
    first, second = res.targets.index[0], res.targets.index[1]
    day_before = res.weights.loc[:second].index[-2]
    drifted = res.weights.loc[day_before]
    assert np.abs(drifted - res.targets.loc[first]).max() > 1e-4      # it really drifted
    assert res.turnover.loc[second] > 0                               # so rebalancing costs something
    assert res.costs.loc[second, "total"] > 0


def test_gross_returns_match_the_strategy_exactly_when_costs_are_off():
    p = _panel(seed=5)
    res = run_backtest(p, equal_weight_strategy,
                       settings=BacktestSettings(warmup_sessions=300, charge_costs=False))
    assert np.allclose(res.returns_net.to_numpy(), res.returns_gross.to_numpy())
    assert any("gross-of-cost" in w for w in res.warnings)
    # and the first day's return is the equal-weighted return of the held names
    day = res.returns_net.index[1]
    held = res.weights.loc[res.weights.index[res.weights.index.get_loc(day) - 1]]
    expected = float((held * p.close.pct_change().loc[day].reindex(held.index).fillna(0)).sum())
    assert res.returns_gross.loc[day] == pytest.approx(expected, abs=1e-12)


def test_costs_reduce_the_return_and_are_itemised_per_rebalance():
    p = _panel(seed=7)
    gross = run_backtest(p, equal_weight_strategy,
                         settings=BacktestSettings(warmup_sessions=300, charge_costs=False))
    net = run_backtest(p, equal_weight_strategy, settings=BacktestSettings(warmup_sessions=300))
    assert net.returns_net.sum() < gross.returns_net.sum()
    assert set(net.costs.columns) >= {"brokerage", "stt", "stamp_duty", "gst", "dp_charges",
                                      "impact", "total", "cost_fraction"}
    assert net.cost_drag_annual() > 0
    assert net.annual_turnover() > 0


def test_a_static_universe_is_flagged_as_survivorship_bias():
    """The one bias the loop cannot remove - so it is disclosed (M8-02)."""
    p = _panel()
    res = run_backtest(p, equal_weight_strategy, settings=BacktestSettings(warmup_sessions=300))
    assert any("survivorship" in w for w in res.warnings)
    assert res.diagnostics["distinct_universes"] == 1

    changing = [list(p.close.columns[:5]), list(p.close.columns)]   # both above min_names
    res2 = run_backtest(p, equal_weight_strategy, settings=BacktestSettings(warmup_sessions=300),
                        universe_at=lambda d: changing[0] if d < p.sessions[600] else changing[1])
    assert res2.diagnostics["distinct_universes"] == 2
    assert not any("survivorship" in w for w in res2.warnings)


def test_a_strategy_that_returns_bad_weights_fails_loudly():
    p = _panel()

    def unnormalised(view, as_of, prev_weights, universe):
        return pd.Series(1.0, index=list(universe))

    with pytest.raises(DataError, match="sum to"):
        run_backtest(p, unnormalised, settings=BacktestSettings(warmup_sessions=300))

    def broken(view, as_of, prev_weights, universe):
        s = pd.Series(1.0 / len(universe), index=list(universe))
        s.iloc[0] = np.nan
        return s

    with pytest.raises(DataError, match="no usable weights"):
        run_backtest(p, broken, settings=BacktestSettings(warmup_sessions=300))


def test_the_previous_weights_are_handed_to_the_strategy():
    """So a strategy can control its own turnover - impossible if it cannot see them."""
    p = _panel()
    received = []

    def remembering(view, as_of, prev_weights, universe):
        received.append(prev_weights.copy())
        return equal_weight_strategy(view, as_of, prev_weights, universe)

    run_backtest(p, remembering, settings=BacktestSettings(warmup_sessions=300))
    assert received[0].empty                        # nothing held at the first rebalance
    assert not received[1].empty
    assert received[1].sum() == pytest.approx(1.0)


def test_missing_liquidity_data_is_disclosed_rather_than_assumed_free():
    p = _panel()
    res = run_backtest(p, equal_weight_strategy, settings=BacktestSettings(warmup_sessions=300))
    assert any("market impact is excluded" in w for w in res.warnings)
    adv = pd.DataFrame(5e6, index=p.sessions, columns=p.close.columns)
    with_impact = run_backtest(p, equal_weight_strategy,
                               settings=BacktestSettings(warmup_sessions=300), adv_inr=adv)
    assert with_impact.costs["impact"].sum() > 0
    assert with_impact.returns_net.sum() < res.returns_net.sum()


def test_the_equity_curve_and_provenance_describe_the_run():
    p = _panel(seed=11)
    bench = p.close.pct_change().mean(axis=1)
    res = run_backtest(p, equal_weight_strategy, settings=BacktestSettings(warmup_sessions=300),
                       benchmark=bench)
    curve = res.equity_curve()
    assert list(curve.columns) == ["strategy", "benchmark"]
    assert curve["strategy"].iloc[-1] > 0
    assert "walk-forward backtest" in res.provenance.method
    assert "rebalances" in res.provenance.window
    assert res.rebalances == len(res.targets)


def test_it_runs_on_the_real_universe():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept

    def screened_equal_weight(view, as_of, prev_weights, universe):
        names = [t for t in universe if t in kept][:20]
        return pd.Series(1.0 / len(names), index=names)

    bench = p.benchmark("^NSEI").pct_change()
    res = run_backtest(panel, screened_equal_weight,
                       settings=BacktestSettings(warmup_sessions=504, rebalance="Q"),
                       benchmark=bench)
    assert res.rebalances >= 3
    assert len(res.returns_net) > 200
    assert res.returns_net.abs().max() < 0.25              # no absurd daily moves
    assert 0 < res.annual_turnover() < 5
    assert res.cost_drag_annual() > 0
