"""
The guidance rules.

The test that matters most is the one asserting the engine says the unwelcome
thing when the evidence is weak - that is the behaviour the rebuild exists for.
"""
import numpy as np
import pandas as pd
import pytest

from star.backtest.stats import SharpeDifference
from star.config.schema import Mandate
from star.report.guidance import (BriefInputs, Decision, Recommendation, Severity, by_decision,
                                  confidence_from_p, evaluate, headline_recommendation, rupees)
from star.risk.backtests import VaRBacktest

AS_OF = pd.Timestamp("2026-09-11")


def _inputs(**kw) -> BriefInputs:
    base = dict(as_of=AS_OF, mandate=Mandate(), nav_inr=1e7,
                weights=pd.Series({"A": 0.4, "B": 0.35, "C": 0.25}))
    base.update(kw)
    return BriefInputs(**base)


def _sharpe_diff(diff, p):
    return SharpeDifference(difference=diff, standard_error=abs(diff) / max(p, 0.01),
                            t_statistic=1.0, p_value=p, headline=f"Sharpe {diff:+.2f}, p={p:.2f}")


def _var_backtest(passed, breaches=8, expected=7.3, p=0.7):
    return VaRBacktest(confidence=0.99, observations=730, breaches=breaches,
                       expected_breaches=expected, kupiec_p=p, independence_p=p,
                       conditional_coverage_p=p, dq_p=p,
                       basel_zone="green" if passed else "red", es_z2=None, es_p=None,
                       headline="the model is consistent" if passed else
                               "The risk model understated losses: 29 breaches where 7 were expected.",
                       passed=passed)


# ------------------------------------------------------------- the mapping ---
def test_confidence_comes_from_the_p_value_not_from_a_choice():
    assert confidence_from_p(0.001)[0] is Severity.ACT
    assert confidence_from_p(0.04)[0] is Severity.ACT
    assert confidence_from_p(0.08)[0] is Severity.WATCH
    assert confidence_from_p(0.34)[0] is Severity.INSUFFICIENT_EVIDENCE
    assert confidence_from_p(None)[0] is Severity.INSUFFICIENT_EVIDENCE
    assert "p=0.34" in confidence_from_p(0.34)[1]


def test_an_act_recommendation_must_say_what_would_prove_it_wrong():
    with pytest.raises(ValueError, match="prove it wrong"):
        Recommendation(id="X", decision=Decision.REBALANCE, severity=Severity.ACT,
                       headline="do something", detail="because")
    ok = Recommendation(id="X", decision=Decision.REBALANCE, severity=Severity.ACT,
                        headline="do something", detail="because", falsifier="unless this")
    assert ok.falsifier == "unless this"


# --------------------------------------------------------------- the rules ---
def test_a_limit_breach_outranks_everything_and_is_not_a_statistical_claim():
    recs = evaluate(_inputs(violations={"over cap: RELIANCE.NS": 0.042,
                                        "over sector cap: Financials": 0.031}))
    breach = next(r for r in recs if r.id == "LIMIT-01")
    assert breach.severity is Severity.ACT
    assert "2 limit(s) breached" in breach.headline
    assert "RELIANCE.NS" in breach.detail
    assert breach.falsifier
    assert "arithmetic" in breach.confidence
    assert headline_recommendation(recs) == breach.headline


def test_a_compliant_portfolio_says_so_without_manufacturing_urgency():
    recs = evaluate(_inputs(violations={}))
    assert any(r.id == "LIMIT-00" and r.severity is Severity.INFO for r in recs)
    assert "No action needed" in headline_recommendation(recs)


def test_the_engine_says_the_unwelcome_thing_when_the_evidence_is_weak():
    """SKILL-01: the rule this rebuild exists for."""
    recs = evaluate(_inputs(sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60)))
    skill = next(r for r in recs if r.id == "SKILL-01")
    assert skill.severity is Severity.INSUFFICIENT_EVIDENCE
    assert "cannot show this does better than simply splitting the money evenly" in skill.headline
    assert "do not add money" in skill.detail
    # it must not read as a verdict against the approach - only against the evidence
    assert "not a claim that the approach is bad" in skill.detail


def test_a_genuine_edge_is_reported_without_overclaiming():
    recs = evaluate(_inputs(sharpe_vs_equal_weight=_sharpe_diff(0.45, 0.02)))
    assert not any(r.id == "SKILL-01" for r in recs)
    good = next(r for r in recs if r.id == "SKILL-02")
    assert good.severity is Severity.WATCH          # evidence, not proof
    assert "not proof it will keep happening" in good.detail


def test_one_message_is_told_once_with_every_measurement_under_it():
    """
    This was three boxes - the Sharpe gap, the record length, the trial-and-error
    adjustment - each in its own vocabulary. The owner read them and could not tell
    they were one point. Merging must not lose a measurement.
    """
    recs = evaluate(_inputs(sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60),
                            min_track_record_years=4.4, track_record_years=2.9,
                            deflated_sharpe=0.0, pbo=0.62))
    skill = [r for r in recs if r.id.startswith("SKILL")]
    assert len(skill) == 1, "one message, one box"
    ev = " | ".join(skill[0].evidence)
    assert "4.4 years" in ev and "2.9" in ev and "1.5 more years" in ev   # the arithmetic
    assert "how many versions were tried" in ev                           # the deflation
    assert "62%" in ev                                                    # the overfitting
    assert "-0.02" in ev                                                  # the Sharpe gap
    assert not any(r.id in {"ALPHA-01", "INSUF-01", "INSUF-03", "INSUF-04"} for r in recs)


def test_a_losing_record_is_distinguished_from_a_short_one():
    recs = evaluate(_inputs(min_track_record_years=float("inf"), track_record_years=2.9))
    r = next(x for x in recs if x.id == "INSUF-02")
    assert "waiting for more of it would not help" in r.headline


def test_paying_a_certain_cost_for_an_unproven_gain_is_flagged():
    recs = evaluate(_inputs(turnover=0.18, cost_fraction=0.0024,
                            sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60)))
    r = next(x for x in recs if x.id == "COST-01")
    assert r.severity is Severity.ACT
    assert "0.24%" in r.headline
    assert "24,000 rupees" in r.detail          # Indian grouping, and the money named
    assert "That money leaves for certain" in r.detail
    assert "limit breach" in r.falsifier


def test_a_gain_inside_its_own_error_bar_is_flagged_more_gently():
    recs = evaluate(_inputs(turnover=0.18, cost_fraction=0.0024,
                            expected_return=0.004, expected_return_error=0.021,
                            sharpe_vs_equal_weight=_sharpe_diff(0.30, 0.03)))
    r = next(x for x in recs if x.id == "COST-02")
    assert r.severity is Severity.WATCH
    assert "too close to the cost to be sure it is a gain" in r.headline


def test_a_failed_risk_model_makes_every_other_number_provisional():
    recs = evaluate(_inputs(var_backtest=_var_backtest(passed=False, breaches=29, p=0.0001)))
    r = next(x for x in recs if x.id == "VALID-01")
    assert r.severity is Severity.ACT
    assert "understated losses" in r.headline
    assert "treat all of them as rough" in r.detail


def test_a_passing_risk_model_still_states_the_limits_of_the_test():
    recs = evaluate(_inputs(var_backtest=_var_backtest(passed=True)))
    r = next(x for x in recs if x.id == "VALID-00")
    assert r.severity is Severity.INFO
    assert "would still miss a model that was moderately wrong" in r.detail


def test_unstable_weights_are_disclosed_as_one_of_several_answers():
    class _Stability:
        mean_turnover = 0.63
        max_name_change = 0.08
        draws = 40
        headline = "Re-estimating expected returns moves 63% of the portfolio on average"

        def is_stable(self, threshold=0.15):
            return False

    recs = evaluate(_inputs(stability=_Stability()))
    r = next(x for x in recs if x.id == "DATA-01")
    assert "Approve the shape of this portfolio, not the exact percentages" in r.headline
    assert "63%" in r.detail
    assert "precise percentages are not" in r.detail


def test_concentration_is_measured_in_risk_not_in_weight():
    rc = pd.DataFrame({"weight": [0.10, 0.09, 0.08, 0.05],
                       "share": [0.22, 0.14, 0.09, 0.05]},
                      index=["A", "B", "C", "D"])
    recs = evaluate(_inputs(risk_contributions=rc))
    r = next(x for x in recs if x.id == "CONC-01")
    assert "45%" in r.headline
    assert "cannot see this from position sizes alone" in r.detail
    # the contrast that makes the point land, in words rather than in a formula
    assert "jumpy share" in r.detail and "steady one" in r.detail
    spread = pd.DataFrame({"weight": [0.1] * 4, "share": [0.1, 0.09, 0.08, 0.07]},
                          index=list("ABCD"))
    assert any(x.id == "CONC-00" for x in evaluate(_inputs(risk_contributions=spread)))


def test_illiquid_positions_are_named():
    recs = evaluate(_inputs(days_to_exit=pd.Series({"THIN": 11.0, "OK": 2.0})))
    r = next(x for x in recs if x.id == "LIQ-01")
    assert "THIN" in r.detail and "11 days" in r.detail
    assert "everyone sells at once" in r.detail


def test_the_absence_of_signals_is_disclosed_rather_than_hidden():
    recs = evaluate(_inputs())
    r = next(x for x in recs if x.id == "SIGNAL-00")
    assert "does not try to pick winners" in r.headline
    assert not any(x.id == "SIGNAL-00" for x in evaluate(_inputs(signals_available=True)))


def test_the_benchmark_handicap_says_whether_it_was_measured_or_assumed():
    """It used to claim it was measured while the number was typed into the tool."""
    assumed = evaluate(_inputs(benchmark_gap=0.0114))
    r = next(x for x in assumed if x.id == "BENCH-01")
    assert "1.14% a year is not really beating it" in r.headline
    assert "assumed, not measured" in r.confidence

    measured = evaluate(_inputs(benchmark_gap=0.0114, benchmark_gap_measured=True))
    r2 = next(x for x in measured if x.id == "BENCH-01")
    assert "measured from a dividend-paying index fund" in r2.confidence


# ------------------------------------------------------------ the ordering ---
def test_recommendations_come_out_most_important_first():
    recs = evaluate(_inputs(violations={"over cap: A": 0.04},
                            sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60),
                            days_to_exit=pd.Series({"THIN": 11.0}),
                            benchmark_gap=0.012))
    severities = [int(r.severity) for r in recs]
    assert severities == sorted(severities, reverse=True)
    assert recs[0].id == "LIMIT-01"


def test_recommendations_group_under_the_three_questions():
    recs = evaluate(_inputs(violations={"over cap: A": 0.04},
                            sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60),
                            turnover=0.2, cost_fraction=0.002))
    grouped = by_decision(recs)
    assert set(grouped) == set(Decision)
    assert any(r.id == "LIMIT-01" for r in grouped[Decision.LIMITS])
    assert any(r.id == "SKILL-01" for r in grouped[Decision.CAPITAL])
    assert any(r.id == "COST-01" for r in grouped[Decision.REBALANCE])
    assert Decision.LIMITS.label == "Are we within limits?"


def test_rules_that_have_no_data_stay_silent_rather_than_guessing():
    recs = evaluate(BriefInputs(as_of=AS_OF))
    ids = {r.id for r in recs}
    assert ids == {"SIGNAL-00"}                      # only the one that needs no inputs
    assert headline_recommendation([]) == "Nothing requires a decision today."


def test_doing_nothing_is_reported_as_the_answer_not_as_a_footnote():
    """When the turnover controls say don't trade, that IS the recommendation."""
    recs = evaluate(_inputs(turnover=0.0, cost_fraction=0.0,
                            stability=_Unstable()))
    r = next(x for x in recs if x.id == "REBAL-00")
    assert "No trades proposed" in r.headline
    # it outranks a WATCH-level caveat about weights that will not change
    assert headline_recommendation(recs) == r.headline


def test_a_real_rebalance_states_its_size_and_cost():
    recs = evaluate(_inputs(turnover=0.18, cost_fraction=0.0024, nav_inr=1e7,
                            traded_value_inr=3_691_029))
    r = next(x for x in recs if x.id == "REBAL-01")
    assert "move 18% of the portfolio" in r.headline
    assert "0.24%" in r.headline and "24,000 rupees" in r.headline
    assert "workbook" in r.detail
    # turnover counts one side, traded value counts both. A reader who does the
    # arithmetic must not conclude the page contradicts itself.
    assert "36.91 lakh" in r.detail and "counting buys and sells together" in r.detail


def test_rupees_are_written_the_way_an_indian_reader_reads_them():
    assert rupees(3_691_029) == "36.91 lakh"
    assert rupees(12_345_678) == "1.23 crore"
    assert rupees(24_000) == "24,000"
    assert rupees(4_701) == "4,701"
    assert rupees(-250_000) == "-2.50 lakh"
    assert rupees(float("nan")) == "--"


def test_a_breach_still_outranks_doing_nothing():
    recs = evaluate(_inputs(turnover=0.0, violations={"over cap: A": 0.04}))
    assert "limit(s) breached" in headline_recommendation(recs)


# ------------------------------------------------------- the plain-English rule ---
JARGON = [
    "Sharpe difference", "estimation error", "perturbation", "Euler", "Kupiec", "CSCV",
    "deflated", "in-sample", "out-of-sample", "go/no-go", "active tilt", "basis point",
    "standard error", "information coefficient", "the book", "indicative", "defensible",
    "configuration", "stochastic", "heteroskedastic", "quantile", "ex-ante", "ex-post",
]


def test_no_recommendation_speaks_in_jargon():
    """
    The owner's complaint, turned into a test: "a lot of things you said are in a
    language too difficult to understand".

    Only the reader-facing fields are checked. `confidence` is allowed to name a
    method, because it goes to the workbook and never to the page.
    """
    recs = evaluate(_inputs(
        violations={"over cap: A": 0.04}, turnover=0.18, traded_value_inr=3.7e6,
        cost_fraction=0.0024, sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60),
        min_track_record_years=4.4, track_record_years=2.9, deflated_sharpe=0.0, pbo=0.62,
        var_backtest=_var_backtest(passed=True), stability=_Unstable(),
        days_to_exit=pd.Series({"THIN": 11.0}), benchmark_gap=0.0114,
        risk_contributions=pd.DataFrame({"weight": [0.4, 0.35, 0.25],
                                         "share": [0.5, 0.3, 0.2]}, index=list("ABC"))))
    assert recs
    for r in recs:
        text = " ".join([r.headline, r.detail, *r.evidence, r.falsifier or ""]).lower()
        for word in JARGON:
            assert word.lower() not in text, f"{r.id} says {word!r}: {text[:120]}"


def test_the_page_is_not_a_wall_of_notes():
    """Ten boxes was too many. The merge has to actually reduce the count."""
    recs = evaluate(_inputs(
        turnover=0.18, traded_value_inr=3.7e6, cost_fraction=0.0005,
        sharpe_vs_equal_weight=_sharpe_diff(-0.02, 0.60),
        min_track_record_years=4.4, track_record_years=2.9, deflated_sharpe=0.0, pbo=0.62,
        var_backtest=_var_backtest(passed=True), stability=_Unstable(),
        benchmark_gap=0.0114, risk_contributions=pd.DataFrame(
            {"weight": [0.4, 0.35, 0.25], "share": [0.2, 0.2, 0.2]}, index=list("ABC"))))
    assert len(recs) <= 8, [r.id for r in recs]
    # and the unwelcome one is still there - brevity must not come out of the bad news
    assert any(r.id == "SKILL-01" for r in recs)


class _Unstable:
    mean_turnover = 0.63
    max_name_change = 0.08
    draws = 40
    headline = "moves 63% of the portfolio on average"

    def is_stable(self, threshold=0.15):
        return False
