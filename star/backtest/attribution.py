"""
Where the return came from, and how the individual decisions worked out.

**Brinson-Fachler attribution** splits performance against a benchmark into three
parts, so "we beat the index by 2.1%" becomes a sentence about decisions:

* **allocation** - holding more or less of a sector than the benchmark, when that
  sector beat or lagged the market;
* **selection** - picking better or worse names *within* a sector;
* **interaction** - the part that needs both to be true at once.

The three sum exactly to the difference in return, which is the property that
makes the decomposition worth trusting. Measured against equal weight by default,
because free NIFTY constituent weights do not exist (M8-07) - and a comparison
against a benchmark whose weights had to be invented is worse than no comparison.

**Trade statistics** answer the questions a manager asks about the decisions
themselves: how often was a position profitable, how large were the wins against
the losses, and how far under water did a holding go before it recovered. Selected
by the owner from `docs/INDUSTRY_METRICS_ASSESSMENT.md`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import DataError


@dataclass(frozen=True)
class BrinsonResult:
    by_group: pd.DataFrame             # allocation, selection, interaction, total per group
    allocation: float
    selection: float
    interaction: float
    total: float
    portfolio_return: float
    benchmark_return: float

    @property
    def headline(self) -> str:
        parts = {"stock selection": self.selection, "sector allocation": self.allocation,
                 "the interaction of the two": self.interaction}
        driver = max(parts, key=lambda k: abs(parts[k]))
        return (f"{self.total:+.2%} against the benchmark, mostly from {driver} "
                f"({parts[driver]:+.2%}).")


def brinson(portfolio_weights: pd.Series, portfolio_returns: pd.Series,
            benchmark_weights: pd.Series, benchmark_returns: pd.Series,
            groups: pd.Series) -> BrinsonResult:
    """
    Brinson-Fachler decomposition over one period.

    Weights and returns are per holding; `groups` maps each holding to a sector.
    Benchmark-only holdings are included, since not holding something the
    benchmark holds is itself an allocation decision.
    """
    names = sorted(set(portfolio_weights.index) | set(benchmark_weights.index))
    wp = pd.Series(portfolio_weights, dtype=float).reindex(names).fillna(0.0)
    wb = pd.Series(benchmark_weights, dtype=float).reindex(names).fillna(0.0)
    rp = pd.Series(portfolio_returns, dtype=float).reindex(names)
    rb = pd.Series(benchmark_returns, dtype=float).reindex(names)
    g = pd.Series(groups).reindex(names)
    if g.isna().any():
        raise DataError(f"no sector for: {list(g.index[g.isna()][:5])}")
    if (wp > 0).any() and rp[wp > 0].isna().any():
        raise DataError("a held name has no return")
    rp = rp.fillna(0.0)
    rb = rb.fillna(0.0)

    total_b = float((wb * rb).sum())
    rows = []
    for sector, idx in g.groupby(g).groups.items():
        idx = list(idx)
        wpg, wbg = float(wp[idx].sum()), float(wb[idx].sum())
        rpg = float((wp[idx] * rp[idx]).sum() / wpg) if wpg > 0 else 0.0
        rbg = float((wb[idx] * rb[idx]).sum() / wbg) if wbg > 0 else 0.0
        allocation = (wpg - wbg) * (rbg - total_b)
        selection = wbg * (rpg - rbg)
        interaction = (wpg - wbg) * (rpg - rbg)
        rows.append({"group": sector, "portfolio_weight": wpg, "benchmark_weight": wbg,
                     "portfolio_return": rpg, "benchmark_return": rbg,
                     "allocation": allocation, "selection": selection,
                     "interaction": interaction, "total": allocation + selection + interaction})
    by_group = pd.DataFrame(rows).set_index("group").sort_values("total", ascending=False)
    total_p = float((wp * rp).sum())
    return BrinsonResult(by_group=by_group, allocation=float(by_group["allocation"].sum()),
                         selection=float(by_group["selection"].sum()),
                         interaction=float(by_group["interaction"].sum()),
                         total=total_p - total_b, portfolio_return=total_p,
                         benchmark_return=total_b)


@dataclass(frozen=True)
class TradeStats:
    trades: pd.DataFrame               # one row per completed holding period
    hit_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    worst_drawdown_while_held: float
    open_positions: int

    @property
    def headline(self) -> str:
        if len(self.trades) == 0:
            return "No completed positions yet."
        return (f"{self.hit_rate:.0%} of {len(self.trades)} closed positions were profitable; "
                f"wins averaged {self.average_win:+.1%} against {self.average_loss:+.1%} for losses "
                f"(profit factor {self.profit_factor:.2f}).")


def trade_statistics(weights: pd.DataFrame, prices: pd.DataFrame, *,
                     min_weight: float = 1e-6) -> TradeStats:
    """
    Turn a weight history into completed holding periods and score them.

    A position opens when its weight rises above `min_weight` from nothing and
    closes when it returns to nothing. Positions still open at the end are counted
    separately rather than closed at the last price, which would flatter a
    strategy holding losers.
    """
    if weights.empty:
        raise DataError("no weight history to analyse")
    held = weights.reindex(columns=prices.columns.intersection(weights.columns)).fillna(0.0)
    if held.empty:
        raise DataError("no overlap between the weight history and the price history")
    px = prices.reindex(index=held.index, columns=held.columns).ffill()

    rows, open_count = [], 0
    for name in held.columns:
        w = held[name].to_numpy()
        p = px[name].to_numpy()
        active = w > min_weight
        i = 0
        while i < len(active):
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(active) and active[j + 1]:
                j += 1
            if j == len(active) - 1 and active[-1]:
                open_count += 1
                break
            entry, exit_ = p[i], p[j]
            if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
                i = j + 1
                continue
            path = p[i:j + 1]
            peak = np.maximum.accumulate(path)
            worst = float(np.max(1.0 - path / np.where(peak > 0, peak, np.nan)))
            rows.append({"ticker": name, "entry": held.index[i], "exit": held.index[j],
                         "sessions": j - i + 1, "return": float(exit_ / entry - 1.0),
                         "average_weight": float(w[i:j + 1].mean()),
                         "worst_drawdown_while_held": worst})
            i = j + 1

    trades = pd.DataFrame(rows)
    if trades.empty:
        return TradeStats(trades, 0.0, 0.0, 0.0, 0.0, 0.0, open_count)
    wins = trades[trades["return"] > 0]["return"]
    losses = trades[trades["return"] <= 0]["return"]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return TradeStats(
        trades=trades.sort_values("return"),
        hit_rate=float(len(wins) / len(trades)),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        average_win=float(wins.mean()) if len(wins) else 0.0,
        average_loss=float(losses.mean()) if len(losses) else 0.0,
        worst_drawdown_while_held=float(trades["worst_drawdown_while_held"].max()),
        open_positions=open_count)
