"""Indian trading costs, to the paisa."""
import numpy as np
import pandas as pd
import pytest

from star.backtest.costs import rebalance_cost, trade_costs, tranche_schedule
from star.config.schema import CostModel
from star.core.errors import DataError

M = CostModel()


def test_a_single_buy_is_costed_line_by_line():
    """Hand-computed: a Rs 1,00,000 delivery buy at the FY2025-26 rates."""
    v = 100_000.0
    c = trade_costs(pd.Series({"RELIANCE.NS": v}), nav_inr=1e7)
    brokerage = min(0.0003 * v, 20.0)                      # 30 -> capped at 20
    exchange, sebi = 0.0000297 * v, 0.000001 * v
    assert c.brokerage == pytest.approx(20.0)
    assert c.stt == pytest.approx(100.0)                   # 0.1%
    assert c.exchange == pytest.approx(2.97)
    assert c.sebi == pytest.approx(0.10)
    assert c.stamp_duty == pytest.approx(15.0)             # 0.015%, buy only
    assert c.gst == pytest.approx(0.18 * (brokerage + exchange + sebi))
    assert c.dp_charges == 0.0                             # sells only
    assert c.impact == 0.0                                 # no ADV supplied
    assert c.total == pytest.approx(20.0 + 100.0 + 2.97 + 0.10 + 15.0 + 0.18 * 23.07)
    assert c.cost_bps_of_traded == pytest.approx(1e4 * c.total / v)


def test_a_sell_pays_the_per_scrip_dp_charge_and_no_stamp_duty():
    c = trade_costs(pd.Series({"TCS.NS": -100_000.0}), nav_inr=1e7)
    assert c.stamp_duty == 0.0
    assert c.dp_charges == pytest.approx(M.dp_charge_inr_per_scrip_sell)
    assert c.stt == pytest.approx(100.0)                   # STT is charged on both legs


def test_the_dp_charge_is_per_scrip_not_per_rupee():
    """The cost the legacy has no way to represent, and it dominates at small NAV."""
    small = trade_costs(pd.Series({f"S{i}.NS": -5_000.0 for i in range(25)}), nav_inr=1_000_000)
    assert small.dp_charges == pytest.approx(25 * M.dp_charge_inr_per_scrip_sell)
    # 25 names x Rs 15.93 on a Rs 10 lakh book is about 4 basis points on its own
    assert small.dp_charges / 1_000_000 > 0.0003
    one_big = trade_costs(pd.Series({"S0.NS": -125_000.0}), nav_inr=1_000_000)
    assert one_big.dp_charges == pytest.approx(M.dp_charge_inr_per_scrip_sell)
    assert small.dp_charges > 20 * one_big.dp_charges


def test_brokerage_is_capped_and_the_cap_binds_above_the_threshold():
    small = trade_costs(pd.Series({"A": 50_000.0}), nav_inr=1e7)
    big = trade_costs(pd.Series({"A": 5_000_000.0}), nav_inr=1e7)
    assert small.brokerage == pytest.approx(0.0003 * 50_000)   # 15, under the cap
    assert big.brokerage == pytest.approx(20.0)                # capped
    assert big.stt == pytest.approx(0.001 * 5_000_000)         # but STT scales


def test_gst_is_charged_only_on_the_fees_it_applies_to():
    c = trade_costs(pd.Series({"A": 1_000_000.0}), nav_inr=1e7)
    assert c.gst == pytest.approx(0.18 * (c.brokerage + c.exchange + c.sebi))
    assert c.gst < 0.18 * (c.brokerage + c.exchange + c.sebi + c.stt + c.stamp_duty)


def test_impact_follows_the_square_root_law_and_needs_liquidity_data():
    adv = pd.Series({"A": 10_000_000.0})
    vol = pd.Series({"A": 0.02})
    small = trade_costs(pd.Series({"A": 1_000_000.0}), nav_inr=1e8, adv_inr=adv, daily_volatility=vol)
    big = trade_costs(pd.Series({"A": 4_000_000.0}), nav_inr=1e8, adv_inr=adv, daily_volatility=vol)
    # four times the size is twice the impact *rate*, so eight times the impact cost
    assert big.impact == pytest.approx(8 * small.impact, rel=1e-9)
    assert small.impact == pytest.approx(0.02 * np.sqrt(0.1) * 1_000_000)
    assert trade_costs(pd.Series({"A": 1_000_000.0}), nav_inr=1e8).impact == 0.0


def test_both_legs_are_charged():
    """The legacy charged one side after the first window, halving every cost (M8-05)."""
    cur = pd.Series({"A": 0.5, "B": 0.5})
    tgt = pd.Series({"A": 0.3, "B": 0.7})
    c = rebalance_cost(cur, tgt, nav_inr=1e7)
    assert set(c.per_scrip["side"]) == {"buy", "sell"}
    assert c.traded_value == pytest.approx(0.4 * 1e7)      # 20% sold and 20% bought
    assert c.dp_charges == pytest.approx(M.dp_charge_inr_per_scrip_sell)   # one sell leg


def test_costs_are_measured_against_what_is_actually_held():
    """Drift is not free: turnover is versus current weights, not last period's target."""
    drifted = pd.Series({"A": 0.62, "B": 0.38})            # drifted away from 50/50
    target = pd.Series({"A": 0.50, "B": 0.50})
    c = rebalance_cost(drifted, target, nav_inr=1e7)
    assert c.traded_value == pytest.approx(0.24 * 1e7)
    assert c.total > 0


def test_a_rebalance_that_changes_nothing_costs_nothing():
    w = pd.Series({"A": 0.5, "B": 0.5})
    c = rebalance_cost(w, w, nav_inr=1e7)
    assert c.total == 0.0 and c.traded_value == 0.0
    assert c.per_scrip.empty


def test_tranching_trades_impact_against_fixed_fees():
    adv = pd.Series({"A": 5_000_000.0})
    trade = pd.Series({"A": -2_000_000.0})
    t = tranche_schedule(trade, nav_inr=1e8, adv_inr=adv, daily_volatility=pd.Series({"A": 0.02}))
    assert list(t.index) == ["all at once", "day 1", "day 2", "day 3", "split total"]
    assert t.loc["split total", "impact"] < t.loc["all at once", "impact"]
    # but the per-scrip DP charge is paid on every selling tranche
    assert t.loc["split total", "fees"] > t.loc["all at once", "fees"]
    assert t.loc["split total", "traded_value"] == pytest.approx(t.loc["all at once", "traded_value"])
    with pytest.raises(DataError, match="sum to 1"):
        tranche_schedule(trade, nav_inr=1e8, fractions=(0.5, 0.2))


def test_costs_refuse_impossible_inputs():
    with pytest.raises(DataError, match="NAV must be positive"):
        trade_costs(pd.Series({"A": 1000.0}), nav_inr=0.0)


def test_a_realistic_rebalance_lands_in_the_expected_range():
    """About 12 basis points one-way on a Rs 10 lakh buy, per the plan's estimate."""
    c = trade_costs(pd.Series({"A": 1_000_000.0}), nav_inr=1e7)
    assert 10 <= c.cost_bps_of_traded <= 14
