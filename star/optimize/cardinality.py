"""
Holding-count limits that are actually enforced.

The legacy exposed a `MAX_ASSETS` setting and never applied it (M3-06): runs
routinely returned more names than the limit allowed, and a "concentrated"
portfolio silently became a diversified one - with the transaction costs and
monitoring burden that implies.

Choosing the best K of N holdings is combinatorial, so this uses the standard
two-stage approach rather than pretending to solve it exactly:

1. solve the relaxed problem with every name available;
2. keep the K names contributing most to utility - return earned net of the risk
   they add, not raw weight, so a large position in something the portfolio
   already owns plenty of is ranked below a smaller diversifying one;
3. re-solve on those K alone, with a floor that stops the solver quietly
   reintroducing a name at 0.1%.

Stage 3 can drop names again (the floor makes that visible), so the count is
verified at the end and the shortfall repaired from the next-best candidates.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..core.errors import OptimizationError
from ..core.types import CovarianceEstimate, ReturnEstimate
from .constraints import ConstraintSet
from .mean_variance import OptimiserSettings, Portfolio, _annual_inputs, optimise


def utility_contribution(weights: pd.Series, mu: ReturnEstimate, cov: CovarianceEstimate,
                         risk_aversion: float) -> pd.Series:
    """
    Each name's share of the objective: w_i * mu_i - (delta/2) * w_i * (Sigma w)_i.

    The second term is the Euler risk contribution, so a name is judged on what
    it adds *given everything else held*, not in isolation.
    """
    names = list(weights.index)
    m, S = _annual_inputs(mu, cov, names)
    w = weights.to_numpy()
    return pd.Series(w * m - 0.5 * risk_aversion * w * (S @ w), index=names,
                     name="utility_contribution")


def limit_holdings(mu: ReturnEstimate, cov: CovarianceEstimate, constraints: ConstraintSet, *,
                   risk_aversion: float, max_holdings: int, min_holdings: int = 0,
                   min_position: float = 0.0,
                   settings: OptimiserSettings = OptimiserSettings()) -> Portfolio:
    """
    Maximum utility subject to holding no more than `max_holdings` names (and, if
    asked, no fewer than `min_holdings`).
    """
    names = list(constraints.tickers)
    if max_holdings < 1:
        raise OptimizationError(f"max_holdings must be at least 1, got {max_holdings}")
    if min_holdings > max_holdings:
        raise OptimizationError(f"min_holdings {min_holdings} exceeds max_holdings {max_holdings}")
    if max_holdings > len(names):
        max_holdings = len(names)
    reachable = float(constraints.upper.nlargest(max_holdings).sum())
    if reachable < constraints.budget - 1e-12:
        raise OptimizationError(
            f"{max_holdings} holdings cannot reach the {constraints.budget:.0%} budget under the "
            f"position caps: the largest {max_holdings} caps sum to {reachable:.1%}. "
            f"Raise max_holdings to at least {int(np.ceil(constraints.budget / float(constraints.upper.max())))} "
            f"or raise the per-stock cap")
    if min_holdings * min_position > constraints.budget + 1e-12:
        raise OptimizationError(
            f"{min_holdings} holdings of at least {min_position:.1%} each need "
            f"{min_holdings * min_position:.0%}, above the {constraints.budget:.0%} budget")

    relaxed = optimise(mu, cov, constraints, risk_aversion=risk_aversion, settings=settings)
    ranked = utility_contribution(relaxed.weights, mu, cov, risk_aversion).sort_values(ascending=False)
    if len(relaxed.holdings(min_position or 1e-6)) <= max_holdings and min_holdings == 0:
        relaxed.diagnostics["cardinality"] = "not binding"
        return relaxed

    chosen, notes = list(ranked.index[:max_holdings]), []
    for attempt in range(3):
        cs = constraints.subset(chosen)
        if min_position > 0:
            floor = pd.Series(min(min_position, float(cs.upper.min())), index=list(cs.tickers))
            cs = replace(cs, lower=floor)
        problems = cs.infeasibilities()
        if problems:
            raise OptimizationError(f"the selected {len(chosen)} names cannot satisfy the mandate: "
                                    f"{'; '.join(problems)}")
        p = optimise(mu, cov, cs, risk_aversion=risk_aversion, settings=settings)
        held = p.holdings(max(min_position, 1e-6))
        if len(held) >= min_holdings or len(chosen) >= len(names):
            weights = p.weights.reindex(names).fillna(0.0)
            constraints.assert_satisfied(weights, what="weights after the holdings limit")
            p.diagnostics.update({"cardinality": f"{len(held)} of at most {max_holdings} names",
                                  "candidates_considered": len(names),
                                  "selection": "top utility contribution from the relaxed solve",
                                  "cardinality_passes": attempt + 1,
                                  "cardinality_notes": notes})
            return Portfolio(weights=weights, objective="max_utility_cardinality",
                             provenance=p.provenance, prev_weights=relaxed.prev_weights,
                             diagnostics=p.diagnostics)
        extra = [t for t in ranked.index if t not in chosen][: min_holdings - len(held)]
        if not extra:
            break
        notes.append(f"pass {attempt + 1} held only {len(held)} names; added {len(extra)} more")
        chosen = (held + extra)[:max_holdings] if len(held) + len(extra) > max_holdings else held + extra
    raise OptimizationError(f"could not reach {min_holdings} holdings within {max_holdings} names "
                            f"under the position caps")
