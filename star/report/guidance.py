"""
Rules that turn the report model into decisions.

Three questions the owner named, and every rule serves one of them:

1. **Approve the rebalance?**
2. **Are we inside our limits?**
3. **Add or withdraw capital?**

Two properties make the output auditable rather than assertive.

**Rules may not compute new statistics.** They read fields that other modules
produced and tested. If a rule wants a p-value, a p-value must already exist. A
rule that could calculate its own evidence could also calculate evidence that
suits its conclusion, and nothing downstream would catch it. The legacy's
"HONEST RISK SUMMARY" contained hard-coded claims that were never computed at all
(M6-07).

**Confidence is derived, never chosen.** It comes from the strength of the test
behind the claim, so a recommendation resting on p = 0.34 is labelled
`INSUFFICIENT_EVIDENCE` and rendered in weaker language automatically. Every
recommendation that says ACT must also state its **falsifier**: what would have
to be true for it to be wrong.

Two rules exist specifically to deliver unwelcome news, and they are the reason
this rebuild is worth having:

* `SKILL-01` - "we cannot show this does better than simply splitting the money evenly."
* `VALID-01` - "the risk model understated losses, so every figure here is too small."

`SKILL-01` was three separate boxes until the owner read them: one on the Sharpe
difference, one on how long a record would have to be, one on the trial-and-error
adjustment, each in its own vocabulary. Three weak tellings of one message read as
noise. They are one recommendation with three supporting lines now, and no
measurement was dropped to get there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from ..config.schema import Mandate


def rupees(amount: float) -> str:
    """
    Indian grouping, because the reader is Indian and 3,691,029 is a foreign number.

    Above a lakh the unit does more work than the digits: "36.9 lakh" is read at a
    glance where "36,91,029" has to be counted.
    """
    if amount is None or not np.isfinite(amount):
        return "--"
    sign = "-" if amount < 0 else ""
    a = abs(amount)
    if a >= 1e7:
        return f"{sign}{a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{sign}{a / 1e5:.2f} lakh"
    whole = f"{int(round(a)):d}"
    if len(whole) > 3:                     # 12345 -> 12,345 ; 123456 -> 1,23,456
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts) + "," + tail
    return f"{sign}{whole}"


class Severity(IntEnum):
    """Sorting order for the brief: the reader sees what matters first."""

    ACT = 4
    WATCH = 3
    INSUFFICIENT_EVIDENCE = 2
    INFO = 1

    @property
    def label(self) -> str:
        return {4: "Act", 3: "Watch", 2: "Not established", 1: "Note"}[int(self)]


class Decision(IntEnum):
    REBALANCE = 1
    LIMITS = 2
    CAPITAL = 3

    @property
    def label(self) -> str:
        return {1: "Approve the rebalance?", 2: "Are we within limits?",
                3: "Add or withdraw capital?"}[int(self)]


@dataclass(frozen=True)
class Recommendation:
    id: str
    decision: Decision
    severity: Severity
    headline: str
    detail: str
    evidence: tuple[str, ...] = ()
    falsifier: str | None = None
    confidence: str = "derived"

    def __post_init__(self) -> None:
        if self.severity is Severity.ACT and not self.falsifier:
            raise ValueError(f"{self.id}: an ACT recommendation must state what would "
                             f"prove it wrong")


def confidence_from_p(p: float | None) -> tuple[Severity, str]:
    """
    The only place a p-value becomes a severity, so the mapping is one decision
    rather than a judgement repeated in every rule.
    """
    if p is None or not np.isfinite(p):
        return Severity.INSUFFICIENT_EVIDENCE, "no test available"
    if p <= 0.01:
        return Severity.ACT, f"strong evidence (p={p:.3f})"
    if p <= 0.05:
        return Severity.ACT, f"evidence (p={p:.3f})"
    if p <= 0.10:
        return Severity.WATCH, f"weak evidence (p={p:.2f})"
    return Severity.INSUFFICIENT_EVIDENCE, f"not distinguishable from chance (p={p:.2f})"


@dataclass(frozen=True)
class BriefInputs:
    """
    Everything the rules may read. All optional: a rule whose inputs are missing
    simply does not fire, rather than guessing.
    """

    as_of: pd.Timestamp
    mandate: Mandate = field(default_factory=Mandate)
    nav_inr: float = 0.0
    weights: pd.Series | None = None
    previous_weights: pd.Series | None = None
    violations: dict | None = None
    binding_caps: dict | None = None
    turnover: float | None = None
    traded_value_inr: float | None = None
    cost_fraction: float | None = None
    expected_return: float | None = None
    expected_return_error: float | None = None
    portfolio_volatility: float | None = None
    risk_contributions: pd.DataFrame | None = None
    # Deliberately untyped by class: these hold objects from other modules, and
    # annotating them here would make `report` import half the package. They must
    # still carry an annotation, or dataclasses treats them as class attributes
    # rather than constructor arguments - a silent and very confusing failure.
    term_structure: object | None = None
    var_backtest: object | None = None
    backtest: object | None = None
    sharpe_vs_equal_weight: object | None = None
    deflated_sharpe: float | None = None
    pbo: float | None = None
    min_track_record_years: float | None = None
    track_record_years: float | None = None
    stability: object | None = None
    regime: object | None = None
    stress_table: pd.DataFrame | None = None
    days_to_exit: pd.Series | None = None
    benchmark_gap: float | None = None
    benchmark_gap_measured: bool = False
    signals_available: bool = False


# --------------------------------------------------------------- the rules ---
def limit_breaches(x: BriefInputs) -> list[Recommendation]:
    """LIMIT-01: the mandate is not advisory. Arithmetic, so no p-value applies."""
    if not x.violations:
        if x.weights is not None:
            return [Recommendation(
                id="LIMIT-00", decision=Decision.LIMITS, severity=Severity.INFO,
                headline="Every limit is satisfied.",
                detail="Position sizes, sector totals, the number of holdings and how quickly "
                       "each could be sold all sit inside the mandate. The table above shows "
                       "how much room is left on each.",
                confidence="arithmetic, not statistical")]
        return []
    worst = sorted(x.violations.items(), key=lambda kv: -abs(kv[1]))
    detail = "; ".join(f"{k} by {abs(v):.2%}" for k, v in worst[:4])
    return [Recommendation(
        id="LIMIT-01", decision=Decision.LIMITS, severity=Severity.ACT,
        headline=f"{len(x.violations)} limit(s) breached - fix these before anything else.",
        detail=f"Over the line by: {detail}.",
        evidence=tuple(f"{k}: {v:+.2%}" for k, v in worst[:6]),
        falsifier="the limits themselves are wrong, in which case change them formally "
                  "rather than ignoring them",
        confidence="arithmetic, not statistical")]


def rebalance_size(x: BriefInputs) -> list[Recommendation]:
    """
    REBAL-00/01: what is actually being proposed.

    When the turnover controls conclude that no trade is worth making, that is
    the answer to "approve the rebalance?" - not a footnote. A brief that leads
    with a caveat while the real recommendation is "do nothing" has buried it.
    """
    if x.turnover is None:
        return []
    if x.turnover < 1e-6:
        return [Recommendation(
            id="REBAL-00", decision=Decision.REBALANCE, severity=Severity.INFO,
            headline="No trades proposed: what we would gain does not cover what it costs "
                     "to get there.",
            detail="The portfolio stays exactly as it is. The trading rules priced the move "
                   "and found it was not worth making.",
            confidence="turnover penalty and no-trade band")]
    cost_bit = ""
    if x.cost_fraction is not None and x.nav_inr:
        cost_bit = (f" and cost about {rupees(x.cost_fraction * x.nav_inr)} rupees, "
                    f"{x.cost_fraction:.2%} of the portfolio")
    elif x.cost_fraction is not None:
        cost_bit = f" at about {x.cost_fraction:.2%} of the portfolio"
    # turnover is one-way; the traded value counts both legs. Saying so stops a reader
    # doing the arithmetic themselves and concluding the page contradicts itself.
    traded = ("" if x.traded_value_inr is None else
              f"That means about {rupees(x.traded_value_inr)} rupees changes hands, counting "
              f"buys and sells together. ")
    return [Recommendation(
        id="REBAL-01", decision=Decision.REBALANCE, severity=Severity.INFO,
        headline=f"The proposed trades move {x.turnover:.0%} of the portfolio{cost_bit}.",
        detail=f"{traded}Every individual trade is listed in the workbook.",
        confidence="itemised Indian delivery costs")]


def skill_is_unproven(x: BriefInputs) -> list[Recommendation]:
    """
    SKILL-01: the rule this rebuild exists for, and the one the reader must not be
    able to skim past.

    It used to be three separate boxes - one on the Sharpe difference, one on how
    long the record would have to be, one on the trial-and-error adjustment - each
    in its own vocabulary. Three weak tellings of one message read as noise; the
    owner said so. They are one recommendation now, with the three measurements as
    supporting lines underneath, and nothing has been dropped.
    """
    d = x.sharpe_vs_equal_weight
    strands: list[str] = []

    # a genuine, measurable edge is reported on its own terms
    if d is not None:
        severity, conf = confidence_from_p(d.p_value)
        if severity is Severity.ACT and d.difference > 0:
            return [Recommendation(
                id="SKILL-02", decision=Decision.CAPITAL, severity=Severity.WATCH,
                headline=("This portfolio did better than an equal split, by more than the "
                          "measurement error."),
                detail=("It earned more per unit of risk taken. That is evidence over this "
                        "stretch of history - not proof it will keep happening."),
                evidence=(d.headline,), confidence=conf)]
        strands.append(f"return per unit of risk was {d.difference:+.2f} against an equal split, "
                       f"which is smaller than the measurement error")

    need, have = x.min_track_record_years, x.track_record_years
    if need is not None and have is not None and np.isfinite(need) and need > have:
        strands.append(f"settling this either way needs about {need:.1f} years of record and "
                       f"only {have:.1f} exist, so {need - have:.1f} more years are required")
    if x.deflated_sharpe is not None and x.deflated_sharpe < 0.5:
        strands.append("once you allow for how many versions were tried before this one, no "
                       "sign of skill is left")
    if x.pbo is not None and x.pbo > 0.5:
        strands.append(f"the settings that looked best on half the history were below average "
                       f"on the other half {x.pbo:.0%} of the time")

    if not strands:
        return []
    return [Recommendation(
        id="SKILL-01", decision=Decision.CAPITAL, severity=Severity.INSUFFICIENT_EVIDENCE,
        headline="We cannot show this does better than simply splitting the money evenly.",
        detail=("So do not add money on the strength of the track record, and do not pay for "
                "the tilts away from an equal split. This is not a claim that the approach is "
                "bad - it is a statement that the evidence is not there either way."),
        evidence=tuple(strands),
        confidence="measured, not assumed")]


def losing_record(x: BriefInputs) -> list[Recommendation]:
    """INSUF-02: a losing record is a different message from a short one."""
    need, have = x.min_track_record_years, x.track_record_years
    if need is None or have is None or np.isfinite(need):
        return []
    return [Recommendation(
        id="INSUF-02", decision=Decision.CAPITAL, severity=Severity.INSUFFICIENT_EVIDENCE,
        headline="The record loses money, so waiting for more of it would not help.",
        detail=f"{have:.1f} years observed. More time cannot rescue a losing record.",
        confidence="no test can rescue a losing record")]


def costs_versus_evidence(x: BriefInputs) -> list[Recommendation]:
    """COST-01: paying a known cost for an unproven gain is the easy call to get wrong."""
    if x.cost_fraction is None or x.turnover is None:
        return []
    cost = x.cost_fraction
    gain = x.expected_return
    err = x.expected_return_error
    unproven = (x.sharpe_vs_equal_weight is not None
                and x.sharpe_vs_equal_weight.p_value > 0.10)
    if cost <= 0:
        return []
    if unproven and cost > 0.0005:
        return [Recommendation(
            id="COST-01", decision=Decision.REBALANCE, severity=Severity.ACT,
            headline=(f"These trades cost {cost:.2%} of the portfolio to chase a gain we "
                      f"have not been able to demonstrate."),
            detail=(f"Moving {x.turnover:.0%} of the portfolio costs "
                    f"{rupees(cost * x.nav_inr)} rupees if the NAV is right. That money leaves "
                    f"for certain. The benefit does not."),
            evidence=(f"moves {x.turnover:.0%} of the portfolio",
                      f"costs {cost:.3%} of the portfolio"),
            falsifier="the trades are needed to fix a limit breach, or the reasoning behind "
                      "them is later shown to actually predict returns",
            confidence="cost is measured; benefit is untested")]
    if gain is not None and err is not None and gain - 2 * err < cost:
        return [Recommendation(
            id="COST-02", decision=Decision.REBALANCE, severity=Severity.WATCH,
            headline="The gain we expect is too close to the cost to be sure it is a gain.",
            detail=(f"Expected gain {gain:.2%}, give or take {err:.2%}; the cost is {cost:.2%}. "
                    f"Consider trading only the positions that breach a limit."),
            evidence=(f"expected gain {gain:.2%}", f"margin of error {err:.2%}",
                      f"cost {cost:.2%}"),
            confidence="two standard errors")]
    return []


def risk_model_health(x: BriefInputs) -> list[Recommendation]:
    """VALID-01: was the risk model right? A failed model makes every other number soft."""
    b = x.var_backtest
    if b is None:
        return []
    if b.passed:
        return [Recommendation(
            id="VALID-00", decision=Decision.LIMITS, severity=Severity.INFO,
            headline="The risk model is telling the truth about itself.",
            detail=f"It said a daily loss would pass a stated size on about "
                   f"{b.expected_breaches:.0f} of the last {b.observations} trading days. It "
                   f"happened on {b.breaches}. Worth knowing: a check on this much history "
                   f"would still miss a model that was moderately wrong.",
            confidence=f"Kupiec p={b.kupiec_p:.2f}")]
    severity, conf = confidence_from_p(min(b.kupiec_p, b.conditional_coverage_p))
    return [Recommendation(
        id="VALID-01", decision=Decision.LIMITS,
        severity=Severity.ACT if severity is Severity.ACT else Severity.WATCH,
        headline=b.headline,
        detail="Every risk number on this page comes out of this model, so treat all of "
               "them as rough until it has been rebuilt.",
        evidence=(f"losses passed the stated size on {b.breaches} days where "
                  f"{b.expected_breaches:.0f} were expected",),
        falsifier="the excess losses all came from one event that the current estimate "
                  "already reflects",
        confidence=conf)]


def weights_are_unstable(x: BriefInputs) -> list[Recommendation]:
    """DATA-01: if the answer moves when the inputs wobble, say so."""
    s = x.stability
    if s is None:
        return []
    if s.is_stable():
        return []
    return [Recommendation(
        id="DATA-01", decision=Decision.REBALANCE, severity=Severity.WATCH,
        headline="Approve the shape of this portfolio, not the exact percentages.",
        detail=(f"Nudge the return forecasts anywhere inside their own margin of error and "
                f"{s.mean_turnover:.0%} of the portfolio moves. Which shares to hold, and "
                f"roughly how much, is a solid answer. The precise percentages are not."),
        evidence=(f"the biggest change to any one holding was {s.max_name_change:.1%}",
                  f"tested over {s.draws} independent re-runs"),
        confidence=f"{s.draws} perturbation draws")]


def concentration(x: BriefInputs) -> list[Recommendation]:
    """CONC-01: how much of the risk sits in how few names."""
    rc = x.risk_contributions
    if rc is None or "share" not in rc or rc.empty:
        return []
    top = rc.nlargest(3, "share")
    share = float(top["share"].sum())
    names = ", ".join(top.index[:3])
    if share < 0.40:
        return [Recommendation(
            id="CONC-00", decision=Decision.LIMITS, severity=Severity.INFO,
            headline=f"Risk is well spread: the three biggest sources carry {share:.0%} of it.",
            detail=f"Most of it comes from {names}.", confidence="Euler decomposition")]
    return [Recommendation(
        id="CONC-01", decision=Decision.LIMITS, severity=Severity.WATCH,
        headline=f"Three holdings carry {share:.0%} of the risk in this portfolio.",
        detail=f"{names}. You cannot see this from position sizes alone: a small holding in a "
               f"jumpy share that moves with everything else carries more risk than a large "
               f"holding in a steady one.",
        evidence=tuple(f"{t} holds {r['weight']:.1%} of the money but {r['share']:.0%} of the risk"
                       for t, r in top.iterrows()),
        confidence="Euler decomposition")]


def liquidity(x: BriefInputs) -> list[Recommendation]:
    """LIQ-01: a position that cannot be exited is a commitment, not a position."""
    d = x.days_to_exit
    if d is None or d.empty:
        return []
    limit = x.mandate.max_days_to_exit
    slow = d[d > limit].sort_values(ascending=False)
    if slow.empty:
        return []
    worst = slow.index[0]
    return [Recommendation(
        id="LIQ-01", decision=Decision.LIMITS, severity=Severity.WATCH,
        headline=f"{len(slow)} holding(s) would take more than {limit:.0f} days to sell out of.",
        detail=f"{worst} needs about {slow.iloc[0]:.0f} days without pushing the price around. "
               f"In a falling market that estimate is optimistic, because everyone sells at once.",
        evidence=tuple(f"{t} would take about {v:.0f} days" for t, v in slow.head(5).items()),
        confidence="from average traded value")]


def regime_state(x: BriefInputs) -> list[Recommendation]:
    """REGIME-01: what the market is doing, and whether the gauges agree."""
    r = x.regime
    if r is None:
        return []
    severity = Severity.WATCH if r.hmm_says == "stress" or not r.agree else Severity.INFO
    return [Recommendation(
        id="REGIME-01", decision=Decision.CAPITAL, severity=severity,
        headline=r.headline,
        detail=("This shapes the scenarios below. On its own it is not a reason to change "
                "anything."),
        # the duration is the length of a STRESSED spell. Printed bare under a "calm"
        # headline it read as "the calm lasts 18 days" - the opposite of its meaning.
        evidence=(f"chance the market is in a stressed state right now: "
                  f"{r.stress_probability:.0%}",
                  f"when stress does arrive, spells have typically lasted about "
                  f"{r.expected_duration_days:.0f} days"),
        confidence="read only from past moves, no future data used")]


def benchmark_comparison(x: BriefInputs) -> list[Recommendation]:
    """BENCH-01: outperformance inside the dividend gap is not outperformance."""
    if x.benchmark_gap is None:
        return []
    return [Recommendation(
        id="BENCH-01", decision=Decision.CAPITAL, severity=Severity.INFO,
        headline=(f"Beating the index by less than {x.benchmark_gap:.2%} a year is not "
                  f"really beating it."),
        detail=("Our returns count the dividends the holdings pay out. The NIFTY 50 index does "
                "not count the dividends its constituents pay. So the index starts every "
                "comparison behind by roughly one year's dividend yield."),
        confidence=("measured from a dividend-paying index fund" if x.benchmark_gap_measured
                    else "assumed, not measured - the index fund data was unavailable"))]


def no_signals(x: BriefInputs) -> list[Recommendation]:
    """The honest disclosure that the return engine has no inputs."""
    if x.signals_available:
        return []
    return [Recommendation(
        id="SIGNAL-00", decision=Decision.REBALANCE, severity=Severity.INFO,
        headline="This portfolio does not try to pick winners.",
        detail=("It is built from risk and the mandate alone. It holds no opinion about which "
                "shares will do better than others. Where it differs from an equal split, that "
                "comes from controlling risk and from not over-trading."),
        confidence="configuration, not estimation")]


RULES = (limit_breaches, rebalance_size, skill_is_unproven, losing_record,
         costs_versus_evidence, risk_model_health, weights_are_unstable, concentration,
         liquidity, regime_state, benchmark_comparison, no_signals)


def evaluate(inputs: BriefInputs) -> list[Recommendation]:
    """Every rule that fires, most important first."""
    out: list[Recommendation] = []
    for rule in RULES:
        out.extend(rule(inputs))
    return sorted(out, key=lambda r: (-int(r.severity), r.decision, r.id))


def headline_recommendation(recommendations: list[Recommendation]) -> str:
    """
    The one sentence at the top of the brief.

    A limit breach outranks everything, because it is arithmetic rather than
    inference. Otherwise the most severe item speaks, and if nothing is pressing
    the brief says so plainly rather than manufacturing urgency.
    """
    if not recommendations:
        return "Nothing requires a decision today."
    breaches = [r for r in recommendations if r.id == "LIMIT-01"]
    if breaches:
        return breaches[0].headline
    acts = [r for r in recommendations if r.severity is Severity.ACT]
    if acts:
        return acts[0].headline
    # "there is nothing to approve" answers the first question outright, so it
    # outranks a caveat about weights that are not going to change anyway
    nothing = [r for r in recommendations if r.id == "REBAL-00"]
    if nothing:
        return nothing[0].headline
    watch = [r for r in recommendations if r.severity is Severity.WATCH]
    if watch:
        return watch[0].headline
    return "No action needed: the portfolio is within its limits and nothing has changed materially."


def by_decision(recommendations: list[Recommendation]) -> dict:
    """Grouped the way the brief lays them out - one block per question."""
    return {d: [r for r in recommendations if r.decision is d] for d in Decision}
