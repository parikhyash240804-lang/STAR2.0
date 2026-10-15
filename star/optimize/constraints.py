"""
The mandate, expressed as constraints a solver must satisfy - not as a cleanup
step applied to a solution that ignored them.

The legacy imposed its per-stock cap by clipping weights and renormalising the
result (M3-02). Renormalising restores the budget by scaling every weight back
up, which puts the capped names straight back over the cap; its `MAX_ASSETS`
limit was never enforced at all (M3-06). So a portfolio could be presented as
respecting a 10% limit while holding 12%.

Here a constraint set is built once, handed to the solver, and then **verified on
the returned weights**. If the constraints cannot all hold at the same time, that
is reported as an infeasibility with the arithmetic that proves it, rather than
silently resolved in favour of whichever one the code happened to apply last.

Three caps stack multiplicatively on each name, and the binding one is named in
the output:

* the mandate's per-stock cap;
* a **liquidity** cap - a position one cannot exit inside the mandate's exit
  window is not a position, it is a commitment;
* a **conviction** cap - where the evidence for a name is weak, it may not be a
  large holding. This is the owner-selected addition from
  `docs/INDUSTRY_METRICS_ASSESSMENT.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..config.schema import Mandate
from ..core.errors import OptimizationError

TOL = 1e-9


def conviction_cap(base_cap: float, evidence: pd.Series, *, floor_fraction: float = 0.4) -> pd.Series:
    """
    Scale the per-stock cap by the strength of the evidence behind each name.

    `evidence` is in [0, 1] - in practice the confidence derived from a measured
    information coefficient (see `estimators/views.py`), never a typed-in number.
    Full evidence earns the full cap; no evidence still earns `floor_fraction` of
    it, because a name can be held for diversification without a view.
    """
    e = pd.Series(evidence, dtype=float)
    if not ((e >= -1e-12) & (e <= 1 + 1e-12)).all():
        raise OptimizationError(f"evidence must be in [0, 1]; out of range: "
                                f"{list(e[(e < 0) | (e > 1)].index[:5])}")
    if not 0 <= floor_fraction <= 1:
        raise OptimizationError(f"floor_fraction must be in [0, 1], got {floor_fraction}")
    return (base_cap * (floor_fraction + (1.0 - floor_fraction) * e.clip(0.0, 1.0))).rename("cap")


def liquidity_cap(adv_inr: pd.Series, nav_inr: float, mandate: Mandate) -> pd.Series:
    """
    The largest weight that can still be exited inside the mandate's window.

    weight <= days_to_exit * participation * ADV / NAV. With a 5-day window and
    20% participation, a stock trading Rs 1 crore a day supports Rs 1 crore of
    position - so on a Rs 10 crore book, 10%.
    """
    adv = pd.Series(adv_inr, dtype=float)
    if nav_inr <= 0:
        raise OptimizationError(f"NAV must be positive, got {nav_inr}")
    cap = mandate.max_days_to_exit * mandate.max_adv_participation * adv / nav_inr
    return cap.clip(lower=0.0).rename("cap")


@dataclass(frozen=True)
class ConstraintSet:
    """Box, budget, sector and turnover limits over a fixed ticker order."""

    tickers: tuple[str, ...]
    lower: pd.Series
    upper: pd.Series
    budget: float = 1.0
    sectors: pd.Series | None = None
    max_sector_weight: float | None = None
    max_turnover: float | None = None          # ONE-WAY: half of sum |w - w_prev|, as in Mandate
    prev_weights: pd.Series | None = None
    binding_cap: pd.Series | None = None       # which cap set each upper bound, for the report

    def __post_init__(self) -> None:
        names = list(self.tickers)
        for label, s in (("lower", self.lower), ("upper", self.upper)):
            if list(s.index) != names:
                raise OptimizationError(f"{label} bounds must be indexed by the ticker order")
        if (self.lower > self.upper + TOL).any():
            bad = self.lower.index[self.lower > self.upper + TOL]
            raise OptimizationError(f"lower bound exceeds upper bound for: {list(bad[:5])}")
        if self.max_turnover is not None and self.prev_weights is None:
            raise OptimizationError("a turnover limit needs the current weights to measure against")
        if self.sectors is not None and self.max_sector_weight is None:
            raise OptimizationError("sector labels were given without a sector cap")

    @property
    def n(self) -> int:
        return len(self.tickers)

    def sector_groups(self) -> dict[str, list[int]]:
        if self.sectors is None:
            return {}
        s = self.sectors.reindex(list(self.tickers))
        if s.isna().any():
            raise OptimizationError(f"no sector for: {list(s.index[s.isna()][:5])}")
        pos = {t: i for i, t in enumerate(self.tickers)}
        groups: dict[str, list[int]] = {}
        for t, sec in s.items():
            groups.setdefault(str(sec), []).append(pos[t])
        return groups

    # ---------------------------------------------------------------- checks --
    def min_turnover_required(self) -> float:
        """
        A lower bound on the **one-way** trading needed to become compliant.

        Money sold out of an over-weight name has to go somewhere, so a breach of
        size D costs D of selling *and* D of buying unless the budget absorbs it.
        That symmetry is the part a naive bound misses, and it is usually the
        larger half. A sector breach and a name breach inside it overlap, so each
        sector counts the larger of the two rather than their sum.

        Still a bound, not the exact minimum - the solver settles the rest - but
        close enough to say "you cannot rebalance this little" with a number.
        """
        if self.prev_weights is None:
            return 0.0
        w0 = self.prev_weights.reindex(list(self.tickers)).fillna(0.0)
        over = np.maximum(w0 - self.upper, 0.0)
        must_buy = float(np.maximum(self.lower - w0, 0.0).sum())
        grouped = set()
        must_sell = 0.0
        if self.sectors is not None and self.max_sector_weight is not None:
            for _, idx in self.sector_groups().items():
                grouped.update(idx)
                sector_excess = max(float(w0.iloc[idx].sum()) - self.max_sector_weight, 0.0)
                must_sell += max(sector_excess, float(over.iloc[idx].sum()))
        must_sell += float(over.iloc[[i for i in range(self.n) if i not in grouped]].sum())
        drift = float(w0.sum()) - self.budget            # positive = too much invested today
        sells = max(must_sell, must_buy + drift, drift)
        return max(sells - drift / 2.0, 0.0)

    def infeasibilities(self) -> tuple[str, ...]:
        """
        Every reason the constraints cannot hold together, with the arithmetic.

        Checked before solving, so the failure names the conflict instead of
        arriving as an opaque solver error.
        """
        problems: list[str] = []
        lo, hi = float(self.lower.sum()), float(self.upper.sum())
        if hi < self.budget - TOL:
            problems.append(f"position caps sum to {hi:.1%}, below the {self.budget:.0%} budget: "
                            f"raise the per-stock cap or add names")
        if lo > self.budget + TOL:
            problems.append(f"minimum holdings sum to {lo:.1%}, above the {self.budget:.0%} budget")
        if self.sectors is not None and self.max_sector_weight is not None:
            reach = sum(min(self.max_sector_weight, float(self.upper.iloc[idx].sum()))
                        for idx in self.sector_groups().values())
            if reach < self.budget - TOL:
                problems.append(f"sector caps allow at most {reach:.1%} invested, below the "
                                f"{self.budget:.0%} budget: {len(self.sector_groups())} sector(s) at "
                                f"{self.max_sector_weight:.0%}")
        if self.max_turnover is not None:
            need = self.min_turnover_required()
            if need > self.max_turnover + TOL:
                problems.append(f"one-way turnover limit {self.max_turnover:.0%} is below the {need:.1%} "
                                f"needed just to bring the current portfolio into compliance")
        return tuple(problems)

    def violations(self, weights: pd.Series, *, tol: float = 1e-8) -> dict[str, float]:
        """Every breach in the given weights, with its size. Empty means compliant."""
        w = pd.Series(weights, dtype=float).reindex(list(self.tickers))
        if w.isna().any():
            raise OptimizationError(f"weights missing for: {list(w.index[w.isna()][:5])}")
        out: dict[str, float] = {}
        budget_gap = float(w.sum()) - self.budget
        if abs(budget_gap) > tol:
            out["budget"] = budget_gap
        for t in self.tickers:
            if w[t] < self.lower[t] - tol:
                out[f"below floor: {t}"] = float(w[t] - self.lower[t])
            if w[t] > self.upper[t] + tol:
                out[f"over cap: {t}"] = float(w[t] - self.upper[t])
        if self.sectors is not None and self.max_sector_weight is not None:
            for sec, idx in self.sector_groups().items():
                total = float(w.iloc[idx].sum())
                if total > self.max_sector_weight + tol:
                    out[f"over sector cap: {sec}"] = total - self.max_sector_weight
        if self.max_turnover is not None:
            w0 = self.prev_weights.reindex(list(self.tickers)).fillna(0.0)
            turnover = float(np.abs(w.to_numpy() - w0.to_numpy()).sum()) / 2.0
            if turnover > self.max_turnover + tol:
                out["over turnover limit"] = turnover - self.max_turnover
        return out

    def assert_satisfied(self, weights: pd.Series, *, what: str = "weights", tol: float = 1e-8) -> None:
        bad = self.violations(weights, tol=tol)
        if bad:
            detail = "; ".join(f"{k} by {v:+.4%}" for k, v in list(bad.items())[:6])
            raise OptimizationError(f"{what} breach the mandate: {detail}")

    # ----------------------------------------------------------------- edits --
    def pinned(self, fixed: pd.Series) -> "ConstraintSet":
        """A copy with some names fixed to given weights (used by the no-trade band)."""
        lower, upper = self.lower.copy(), self.upper.copy()
        for t, v in pd.Series(fixed, dtype=float).items():
            if t not in lower.index:
                raise OptimizationError(f"cannot pin {t}: not in the universe")
            v = float(np.clip(v, self.lower[t], self.upper[t]))
            lower[t] = upper[t] = v
        return replace(self, lower=lower, upper=upper)

    def subset(self, keep) -> "ConstraintSet":
        """A copy over a subset of names, for the cardinality stage."""
        names = [t for t in self.tickers if t in set(keep)]
        if not names:
            raise OptimizationError("cannot restrict the universe to nothing")
        return replace(self, tickers=tuple(names), lower=self.lower[names], upper=self.upper[names],
                       sectors=None if self.sectors is None else self.sectors.reindex(names),
                       prev_weights=None if self.prev_weights is None
                       else self.prev_weights.reindex(names).fillna(0.0),
                       binding_cap=None if self.binding_cap is None else self.binding_cap[names])


def build_constraints(tickers, mandate: Mandate, *, sectors: pd.Series | None = None,
                      evidence: pd.Series | None = None, adv_inr: pd.Series | None = None,
                      nav_inr: float | None = None, prev_weights: pd.Series | None = None,
                      max_turnover: float | None = None, min_weight: float = 0.0,
                      budget: float = 1.0, floor_fraction: float = 0.4) -> ConstraintSet:
    """
    Assemble the constraint set, recording **which** cap binds each name.

    The upper bound is the tightest of the mandate cap, the conviction cap and the
    liquidity cap, so the report can say "trimmed for liquidity" rather than just
    showing a smaller number.
    """
    names = list(tickers)
    if not names:
        raise OptimizationError("cannot optimise an empty universe")
    caps = pd.DataFrame({"mandate": pd.Series(mandate.max_stock_weight, index=names)})
    if evidence is not None:
        caps["conviction"] = conviction_cap(mandate.max_stock_weight,
                                           pd.Series(evidence, dtype=float).reindex(names).fillna(0.0),
                                           floor_fraction=floor_fraction)
    if adv_inr is not None:
        if nav_inr is None:
            raise OptimizationError("a liquidity cap needs the portfolio NAV")
        caps["liquidity"] = liquidity_cap(pd.Series(adv_inr, dtype=float).reindex(names).fillna(0.0),
                                         nav_inr, mandate)
    upper = caps.min(axis=1)
    binding = caps.idxmin(axis=1)
    return ConstraintSet(tickers=tuple(names), lower=pd.Series(float(min_weight), index=names),
                         upper=upper, budget=float(budget),
                         sectors=None if sectors is None else pd.Series(sectors).reindex(names),
                         max_sector_weight=None if sectors is None else mandate.max_sector_weight,
                         max_turnover=max_turnover,
                         prev_weights=None if prev_weights is None
                         else pd.Series(prev_weights, dtype=float).reindex(names).fillna(0.0),
                         binding_cap=binding.rename("binding_cap"))
