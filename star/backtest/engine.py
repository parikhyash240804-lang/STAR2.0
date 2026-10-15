"""
Walk-forward backtest that runs the strategy that ships.

The strategy is supplied as **one callable**, and the live pipeline calls the same
one. That is the only structural way to stop a backtest from testing a close
cousin of the real thing, which is what the legacy did: same return estimator but
a different covariance, a different post-processing floor and a different number
of optimiser restarts (M8-01).

Everything the strategy sees arrives through a `DataView` truncated at the
rebalance date, so a look-ahead is an exception rather than a silent advantage.

Between rebalances **weights drift with prices**, as a real portfolio does. The
legacy rebalanced back to target every day for free (M8-09), which both removed
the drift and made turnover - and therefore cost - fictional.

Costs are charged on both legs, against the weights actually held. The risk-free
rate is the rate of that period, not today's (M8-10).

What this engine cannot fix: if the universe handed to it is today's list of
survivors, the result is optimistic no matter how clean the loop is (M8-02). So it
takes a `universe_at` callable, defaults to "everything with enough history as of
that date", and **records a warning whenever the universe could not vary** - the
one bias that code cannot remove, only disclose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.schema import CostModel
from ..core.errors import DataError
from ..core.types import Provenance
from ..data.view import DataView
from .costs import rebalance_cost

REBALANCE_RULES = {"M": "ME", "2M": "2ME", "Q": "QE", "2Q": "2QE", "Y": "YE", "W": "W-FRI"}


@dataclass(frozen=True)
class BacktestSettings:
    rebalance: str = "Q"               # M | 2M | Q | 2Q | Y | W
    warmup_sessions: int = 504         # history the first decision may use
    nav_inr: float = 1e7
    charge_costs: bool = True
    include_impact: bool = True
    min_names: int = 5


@dataclass(frozen=True)
class BacktestResult:
    returns_net: pd.Series             # daily portfolio return after costs
    returns_gross: pd.Series
    benchmark: pd.Series | None
    risk_free: pd.Series | None
    weights: pd.DataFrame              # held weights at each session (drifted)
    targets: pd.DataFrame              # target weights set at each rebalance
    costs: pd.DataFrame                # one row per rebalance, itemised
    turnover: pd.Series                # one-way, per rebalance
    provenance: Provenance
    warnings: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def rebalances(self) -> int:
        return len(self.targets)

    def equity_curve(self, initial: float = 1.0) -> pd.DataFrame:
        out = pd.DataFrame({"strategy": initial * (1 + self.returns_net).cumprod()})
        if self.benchmark is not None:
            out["benchmark"] = initial * (1 + self.benchmark.reindex(out.index).fillna(0)).cumprod()
        return out

    def annual_turnover(self) -> float:
        if self.turnover.empty:
            return 0.0
        years = max(len(self.returns_net) / 252.0, 1e-9)
        return float(self.turnover.sum() / years)

    def total_cost_fraction(self) -> float:
        return float(self.costs["cost_fraction"].sum()) if len(self.costs) else 0.0

    def cost_drag_annual(self) -> float:
        years = max(len(self.returns_net) / 252.0, 1e-9)
        return self.total_cost_fraction() / years


def rebalance_dates(sessions: pd.DatetimeIndex, rule: str, *, warmup: int) -> pd.DatetimeIndex:
    """Last trading session of each period, after the warm-up."""
    if rule not in REBALANCE_RULES:
        raise DataError(f"rebalance must be one of {sorted(REBALANCE_RULES)}, got {rule!r}")
    if len(sessions) <= warmup:
        raise DataError(f"{len(sessions)} sessions is not enough for a {warmup}-session warm-up")
    eligible = pd.DatetimeIndex(sessions[warmup:])
    marks = pd.Series(eligible, index=eligible).resample(REBALANCE_RULES[rule]).last().dropna()
    return pd.DatetimeIndex(sorted(set(marks.to_numpy()))).intersection(eligible)


def run_backtest(panel, strategy, *, settings: BacktestSettings = BacktestSettings(),
                 benchmark: pd.Series | None = None, risk_free: pd.Series | None = None,
                 cost_model: CostModel = CostModel(), universe_at=None,
                 adv_inr: pd.DataFrame | None = None) -> BacktestResult:
    """
    Run the walk-forward.

    `strategy(view, as_of, prev_weights, universe) -> pd.Series` must return target
    weights summing to one. It is handed a `DataView` truncated at `as_of` and
    nothing else, so it cannot see the future even by accident.
    """
    close = panel.close
    sessions = pd.DatetimeIndex(panel.sessions)
    dates = rebalance_dates(sessions, settings.rebalance, warmup=settings.warmup_sessions)
    if len(dates) < 2:
        raise DataError(f"only {len(dates)} rebalance dates available; lengthen the history or "
                        f"use a more frequent rebalance than {settings.rebalance!r}")

    daily = close.pct_change()
    warnings: list[str] = []
    universes_seen: set[tuple] = set()

    held = pd.Series(dtype=float)
    rows_w, rows_t, cost_rows, turnover = {}, {}, [], {}
    gross, net = {}, {}
    start = sessions.get_loc(dates[0])
    nav = float(settings.nav_inr)

    for i in range(start, len(sessions)):
        day = sessions[i]
        r = daily.loc[day]

        if not held.empty:
            names = held.index
            rr = r.reindex(names).fillna(0.0)
            day_gross = float((held * rr).sum())
            grown = held * (1.0 + rr)
            total = grown.sum()
            held = grown / total if total > 0 else grown
            nav *= (1.0 + day_gross)
        else:
            day_gross = 0.0
        day_net = day_gross

        if day in dates:
            view = DataView.build(panel, day)
            universe = (list(universe_at(day)) if universe_at is not None
                        else _default_universe(close, day, settings))
            if len(universe) < settings.min_names:
                warnings.append(f"{day.date()}: only {len(universe)} names passed the universe "
                                f"screen; rebalance skipped")
            else:
                universes_seen.add(tuple(sorted(universe)))
                target = pd.Series(strategy(view, day, held.copy(), universe), dtype=float)
                if target.empty or not np.isfinite(target.to_numpy()).all():
                    raise DataError(f"{day.date()}: the strategy returned no usable weights")
                if abs(float(target.sum()) - 1.0) > 1e-6:
                    raise DataError(f"{day.date()}: target weights sum to {target.sum():.6f}, not 1")

                one_way = float(np.abs(
                    target.reindex(sorted(set(target.index) | set(held.index))).fillna(0.0)
                    - held.reindex(sorted(set(target.index) | set(held.index))).fillna(0.0)).sum() / 2)
                turnover[day] = one_way
                if settings.charge_costs:
                    adv_row = None
                    if settings.include_impact and adv_inr is not None and day in adv_inr.index:
                        adv_row = adv_inr.loc[day]
                    c = rebalance_cost(held, target, nav_inr=nav, model=cost_model,
                                       adv_inr=adv_row)
                    day_net -= c.cost_fraction
                    nav *= (1.0 - c.cost_fraction)
                    cost_rows.append({"date": day, **c.itemised().to_dict(),
                                      "traded_value": c.traded_value,
                                      "cost_fraction": c.cost_fraction,
                                      "cost_bps_of_traded": c.cost_bps_of_traded})
                held = target.copy()
                rows_t[day] = target

        gross[day] = day_gross
        net[day] = day_net
        rows_w[day] = held.copy()

    if len(universes_seen) <= 1:
        warnings.append("the universe never changed across rebalances: results include survivorship "
                        "bias, because only names that exist today were ever eligible")
    if not settings.charge_costs:
        warnings.append("costs were not charged: this is a gross-of-cost result and is not achievable")
    if settings.include_impact and adv_inr is None:
        warnings.append("no traded-value data was supplied, so market impact is excluded and costs "
                        "are understated for large or illiquid trades")

    returns_net = pd.Series(net, name="net").sort_index()
    returns_gross = pd.Series(gross, name="gross").sort_index()
    bench = None if benchmark is None else pd.Series(benchmark).reindex(returns_net.index)
    rf = None if risk_free is None else pd.Series(risk_free).reindex(returns_net.index)
    prov = Provenance(
        method=f"walk-forward backtest, {settings.rebalance} rebalance, costs "
               f"{'charged' if settings.charge_costs else 'EXCLUDED'}",
        window=f"{returns_net.index[0].date()} to {returns_net.index[-1].date()}, "
               f"{len(rows_t)} rebalances, {settings.warmup_sessions}-session warm-up",
        as_of=pd.Timestamp(returns_net.index[-1]), source="point-in-time DataView per rebalance",
        degradations=tuple(warnings))
    return BacktestResult(
        returns_net=returns_net, returns_gross=returns_gross, benchmark=bench, risk_free=rf,
        weights=pd.DataFrame(rows_w).T.fillna(0.0).sort_index(),
        targets=pd.DataFrame(rows_t).T.fillna(0.0).sort_index(),
        costs=pd.DataFrame(cost_rows).set_index("date") if cost_rows else pd.DataFrame(),
        turnover=pd.Series(turnover, name="turnover").sort_index(),
        provenance=prov, warnings=tuple(warnings),
        diagnostics={"rebalances": len(rows_t), "distinct_universes": len(universes_seen),
                     "sessions": len(returns_net), "final_nav": nav})


def _default_universe(close: pd.DataFrame, as_of, settings: BacktestSettings) -> list[str]:
    """Names with enough history and a price **as of that date** - no future knowledge."""
    upto = close.loc[:as_of]
    enough = upto.notna().sum() >= settings.warmup_sessions
    priced = upto.iloc[-1].notna()
    return list(close.columns[enough & priced])
