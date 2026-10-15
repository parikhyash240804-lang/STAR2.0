"""
What a trade actually costs on an Indian exchange, itemised to the paisa.

Delivery equity, FY2025-26 rates. Every component is charged where it is really
charged, which matters more than it sounds:

| Component | Buy | Sell | Base |
|---|---|---|---|
| Brokerage | yes | yes | min(0.03% of value, Rs 20) per order |
| STT | yes | yes | 0.1% of value |
| Exchange transaction | yes | yes | 0.00297% of value |
| SEBI turnover fee | yes | yes | 0.0001% of value |
| Stamp duty | yes | **no** | 0.015% of value |
| GST | yes | yes | 18% of brokerage + exchange + SEBI |
| DP charge | **no** | yes | ~Rs 15.93 **per scrip**, regardless of size |
| Market impact | yes | yes | square-root law on participation |

Two of these decide whether a strategy survives contact with reality, and the
legacy had neither right.

**The DP charge is per scrip, not per rupee.** Selling Rs 5,000 of one stock costs
the same Rs 15.93 as selling Rs 5,00,000 of it. On a Rs 10 lakh book holding 25
names, a full rebalance pays around Rs 400 in DP charges alone - about 4 basis
points that no percentage-based model can see. It is absent from the legacy
entirely, and it dominates at small NAV with many holdings.

**Impact is not linear.** Trading 10% of a stock's daily volume moves the price
against you; the square-root law (Almgren et al. 2005) says roughly
`k * sigma * sqrt(value / ADV)`. Without depth data this is an approximation, and
it is labelled as one - but ignoring it entirely, as the legacy did, makes any
high-turnover strategy look profitable.

Costs are charged on **both** legs. The legacy charged one side after the first
window, halving every subsequent cost (M8-05).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.schema import CostModel
from ..core.errors import DataError


@dataclass(frozen=True)
class TradeCosts:
    """Itemised cost of one rebalance, in rupees."""

    per_scrip: pd.DataFrame            # one row per traded name
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp_duty: float
    gst: float
    dp_charges: float
    impact: float
    traded_value: float
    nav: float

    @property
    def total(self) -> float:
        return float(self.brokerage + self.stt + self.exchange + self.sebi
                     + self.stamp_duty + self.gst + self.dp_charges + self.impact)

    @property
    def cost_fraction(self) -> float:
        """Cost as a share of the portfolio, which is what the return loses."""
        return self.total / self.nav if self.nav > 0 else 0.0

    @property
    def cost_bps_of_traded(self) -> float:
        return 1e4 * self.total / self.traded_value if self.traded_value > 0 else 0.0

    def itemised(self) -> pd.Series:
        return pd.Series({"brokerage": self.brokerage, "stt": self.stt, "exchange": self.exchange,
                          "sebi": self.sebi, "stamp_duty": self.stamp_duty, "gst": self.gst,
                          "dp_charges": self.dp_charges, "impact": self.impact,
                          "total": self.total}, name="rupees")

    def headline(self) -> str:
        return (f"Trading {self.traded_value:,.0f} rupees costs {self.total:,.0f} "
                f"({self.cost_bps_of_traded:.0f} bp of traded value, "
                f"{self.cost_fraction:.2%} of the portfolio).")


def trade_costs(trade_value_inr: pd.Series, *, nav_inr: float, model: CostModel = CostModel(),
                adv_inr: pd.Series | None = None, daily_volatility: pd.Series | None = None,
                default_volatility: float = 0.02) -> TradeCosts:
    """
    Cost of a set of trades. Positive values are buys, negative are sells.

    `adv_inr` and `daily_volatility` drive the impact estimate; without them impact
    is zero and the caller is expected to say so, since a backtest with no impact
    flatters every high-turnover strategy.
    """
    t = pd.Series(trade_value_inr, dtype=float).dropna()
    t = t[np.abs(t) > 1e-9]
    if nav_inr <= 0:
        raise DataError(f"NAV must be positive, got {nav_inr}")
    if t.empty:
        empty = pd.DataFrame(columns=["value", "side", "brokerage", "stt", "exchange", "sebi",
                                      "stamp_duty", "gst", "dp", "impact", "total"])
        return TradeCosts(empty, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, float(nav_inr))

    value = t.abs()
    is_buy = t > 0
    brokerage = np.minimum(model.brokerage_rate * value, model.brokerage_cap_inr)
    stt = model.stt_delivery * value
    exchange = model.exchange_txn * value
    sebi = model.sebi_fee * value
    stamp = np.where(is_buy, model.stamp_duty_buy * value, 0.0)
    gst = model.gst_rate * (brokerage + exchange + sebi)
    dp = np.where(is_buy, 0.0, model.dp_charge_inr_per_scrip_sell)

    if adv_inr is not None:
        adv = pd.Series(adv_inr, dtype=float).reindex(t.index)
        vol = (pd.Series(daily_volatility, dtype=float).reindex(t.index)
               if daily_volatility is not None else pd.Series(default_volatility, index=t.index))
        vol = vol.fillna(default_volatility)
        participation = np.where((adv > 0) & np.isfinite(adv), value / adv.replace(0, np.nan), np.nan)
        impact_rate = model.impact_coefficient * vol.to_numpy() * np.sqrt(np.nan_to_num(participation))
        impact = impact_rate * value.to_numpy()
    else:
        impact = np.zeros(len(t))

    per = pd.DataFrame({
        "value": t, "side": np.where(is_buy, "buy", "sell"), "brokerage": brokerage, "stt": stt,
        "exchange": exchange, "sebi": sebi, "stamp_duty": stamp, "gst": gst, "dp": dp,
        "impact": impact})
    per["total"] = per[["brokerage", "stt", "exchange", "sebi", "stamp_duty", "gst", "dp",
                        "impact"]].sum(axis=1)
    return TradeCosts(per_scrip=per, brokerage=float(brokerage.sum()), stt=float(stt.sum()),
                      exchange=float(exchange.sum()), sebi=float(sebi.sum()),
                      stamp_duty=float(np.sum(stamp)), gst=float(gst.sum()),
                      dp_charges=float(np.sum(dp)), impact=float(np.sum(impact)),
                      traded_value=float(value.sum()), nav=float(nav_inr))


def rebalance_cost(current_weights: pd.Series, target_weights: pd.Series, *, nav_inr: float,
                   model: CostModel = CostModel(), adv_inr: pd.Series | None = None,
                   daily_volatility: pd.Series | None = None) -> TradeCosts:
    """
    Cost of moving from one weight vector to another - **both legs**, on the
    weights actually held rather than on last period's target.

    The legacy measured turnover against the previous *target*, so drift between
    rebalances was free and a buy-and-hold strategy appeared to trade nothing
    (M8-05, M8-09).
    """
    names = sorted(set(current_weights.index) | set(target_weights.index))
    cur = pd.Series(current_weights, dtype=float).reindex(names).fillna(0.0)
    tgt = pd.Series(target_weights, dtype=float).reindex(names).fillna(0.0)
    return trade_costs((tgt - cur) * nav_inr, nav_inr=nav_inr, model=model, adv_inr=adv_inr,
                       daily_volatility=daily_volatility)


def tranche_schedule(trade_value_inr: pd.Series, *, nav_inr: float, fractions=(0.6, 0.3, 0.1),
                     model: CostModel = CostModel(), adv_inr: pd.Series | None = None,
                     daily_volatility: pd.Series | None = None) -> pd.DataFrame:
    """
    Split the trade across several days and price each tranche.

    Impact grows with the square root of size, so three tranches of a third cost
    `sqrt(1/3)` each in impact - about 58% of doing it at once. Fixed fees do not
    shrink, and the per-scrip DP charge is paid on **every** selling tranche, so
    splitting a small sell order costs more than it saves. The table shows both
    effects rather than assuming tranching always wins.
    """
    f = np.asarray(fractions, dtype=float)
    if np.any(f <= 0) or abs(f.sum() - 1.0) > 1e-9:
        raise DataError(f"tranche fractions must be positive and sum to 1, got {fractions}")
    one_go = trade_costs(trade_value_inr, nav_inr=nav_inr, model=model, adv_inr=adv_inr,
                         daily_volatility=daily_volatility)
    rows = [{"tranche": "all at once", "fraction": 1.0, "traded_value": one_go.traded_value,
             "impact": one_go.impact, "fees": one_go.total - one_go.impact, "total": one_go.total}]
    split_total = 0.0
    for i, frac in enumerate(f, start=1):
        c = trade_costs(pd.Series(trade_value_inr) * frac, nav_inr=nav_inr, model=model,
                        adv_inr=adv_inr, daily_volatility=daily_volatility)
        split_total += c.total
        rows.append({"tranche": f"day {i}", "fraction": float(frac), "traded_value": c.traded_value,
                     "impact": c.impact, "fees": c.total - c.impact, "total": c.total})
    rows.append({"tranche": "split total", "fraction": 1.0, "traded_value": one_go.traded_value,
                 "impact": sum(r["impact"] for r in rows[1:]), "fees": split_total - sum(
                     r["impact"] for r in rows[1:]), "total": split_total})
    return pd.DataFrame(rows).set_index("tranche")
