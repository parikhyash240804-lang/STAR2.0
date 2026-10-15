"""
The HTML brief.

Two rules from docs/PERSONAS.md are enforced here rather than left to discipline:
it must work with no internet, and it must not contain the technical detail that
belongs in the workbook.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from star.forecast.engine import DriftAssumption, simulate
from star.forecast.paths import BootstrapGenerator
from star.report.brief import build_brief_data, confidence_badge, render_brief, write_brief
from star.report.guidance import BriefInputs, evaluate

AS_OF = pd.Timestamp("2026-09-11")


@pytest.fixture(scope="module")
def term_structure():
    rng = np.random.default_rng(4)
    port = rng.normal(0.0004, 0.011, 1500)
    bench = port * 0.8 + rng.normal(0.0002, 0.006, 1500)
    return simulate(BootstrapGenerator(portfolio=port, benchmark=bench, mean_block=10),
                    horizon=252, n_paths=4000, seed=3, drift=DriftAssumption())


def _data(**kw):
    base = dict(as_of=AS_OF, nav_inr=1.0e7, recommendations=evaluate(BriefInputs(as_of=AS_OF)))
    base.update(kw)
    return build_brief_data(**base)


# ------------------------------------------------------------- confidence ---
def test_confidence_badges_follow_the_stated_rule():
    good = confidence_badge(21, risk_model_passed=True, tail_reliable=True, regime_settled=True)
    assert good["level"] == "Good"

    year = confidence_badge(252, risk_model_passed=True, tail_reliable=True, regime_settled=True)
    assert year["level"] == "Fair"
    assert "few independent periods" in year["why"]

    far = confidence_badge(600, risk_model_passed=True, tail_reliable=True, regime_settled=True)
    assert far["level"] == "Weak" and "extrapolates" in far["why"]

    broken = confidence_badge(21, risk_model_passed=False, tail_reliable=True, regime_settled=True)
    assert broken["level"] in {"Fair", "Weak"} and "failed its breach test" in broken["why"]

    unsettled = confidence_badge(252, risk_model_passed=False, tail_reliable=False,
                                 regime_settled=False)
    assert unsettled["level"] == "Weak"
    assert unsettled["why"].count(";") >= 2          # every reason is listed, not just the first


def test_no_invented_score_out_of_a_hundred():
    badge = confidence_badge(21, risk_model_passed=True)
    assert set(badge) == {"level", "why", "basis"}
    assert badge["level"] in {"Good", "Fair", "Weak"}
    assert not re.search(r"\b\d{1,3}\s*/\s*100\b", json.dumps(badge))
    assert "score" not in json.dumps(badge).lower()


def test_the_badge_shows_the_checks_behind_the_word():
    """"Fair" on its own invites the question; the basis is the answer."""
    badge = confidence_badge(252, risk_model_passed=True, regime_settled=False,
                             checks=[{"name": "Volatility model",
                                      "value": "explains 38% of the variation",
                                      "verdict": "fair"}])
    names = [b["name"] for b in badge["basis"]]
    assert names[0] == "How far ahead"
    assert "Does the risk model hold up" in names and "Volatility model" in names
    assert {b["verdict"] for b in badge["basis"]} <= {"pass", "fair", "fail"}
    # the horizon line carries the arithmetic, not an adjective
    assert "independent periods" in badge["basis"][0]["value"]
    assert "252 sessions" in badge["basis"][0]["value"]


def test_a_failed_risk_model_says_what_it_means_for_the_page():
    basis = confidence_badge(21, risk_model_passed=False)["basis"]
    line = next(b for b in basis if b["name"] == "Does the risk model hold up")
    assert line["verdict"] == "fail"
    assert "understates the risk" in line["value"]


# ---------------------------------------------------------------- payload ---
def test_every_horizon_reads_the_same_cell_the_engine_stored(term_structure):
    data = _data(term_structure=term_structure)
    assert [h["days"] for h in data["horizons"]] == [1, 5, 21, 63, 126, 252]
    for h in data["horizons"]:
        at = term_structure.at(h["days"])
        assert h["median"] == pytest.approx(at["returns"][50], abs=1e-6)
        assert h["p5"] == pytest.approx(at["returns"][5], abs=1e-6)
        assert h["p_loss"] == pytest.approx(at["p_loss"], abs=1e-6)


def test_the_payload_is_valid_json_with_no_missing_numbers(term_structure):
    data = _data(term_structure=term_structure,
                 risk_contributors=pd.DataFrame({"weight": [0.1, np.nan], "share": [0.3, 0.2]},
                                                index=["A", "B"]))
    text = json.dumps(data, allow_nan=False)          # raises if a NaN survived
    assert "NaN" not in text and "Infinity" not in text


def test_a_horizon_beyond_the_run_is_left_out_rather_than_extrapolated(term_structure):
    data = _data(term_structure=term_structure, horizons=(1, 21, 252, 504))
    assert [h["days"] for h in data["horizons"]] == [1, 21, 252]


def test_the_fan_chart_is_subsampled_but_faithful(term_structure):
    data = _data(term_structure=term_structure)
    fan = data["fan"]
    assert 100 <= len(fan["days"]) <= 180              # small payload, smooth curve
    assert len(fan["median"]) == len(fan["days"])
    v0 = term_structure.initial_value
    for i, d in enumerate(fan["days"][:5]):
        # day d is stored at index d - 1, because day numbering starts at one
        assert fan["median"][i] == pytest.approx(
            term_structure.quantiles[50][d - 1] / v0 - 1, abs=1e-6)
    assert all(a <= b for a, b in zip(fan["p5"], fan["p25"]))
    assert all(a <= b for a, b in zip(fan["p75"], fan["p95"]))


def test_allocation_shows_the_biggest_moves_sorted():
    alloc = pd.DataFrame({"current": [0.10, 0.05, 0.08], "proposed": [0.04, 0.12, 0.08]},
                         index=["CUT", "ADD", "SAME"])
    alloc["change"] = alloc["proposed"] - alloc["current"]
    data = _data(allocation=alloc)
    assert [a["ticker"] for a in data["allocation"]] == ["CUT", "SAME", "ADD"]
    assert data["allocation"][0]["change"] == pytest.approx(-0.06)


# ----------------------------------------------------------------- render ---
def test_the_page_needs_no_internet():
    """Persona A reads it on a plane."""
    page = render_brief(_data())
    for forbidden in ("http://", "https://", "cdn.", "<link", "@import", "src="):
        assert forbidden not in page, forbidden


def test_the_page_does_not_leak_the_technical_detail(term_structure):
    """
    Persona A never sees the covariance matrix. This is a content rule, so it is
    tested rather than trusted to review.
    """
    page = render_brief(_data(term_structure=term_structure))
    payload = re.search(r"const D = (\{.*?\});\n", page, re.S).group(1).lower()
    for forbidden in ("covariance", "ledoit", "shrinkage", "eigenvalue", "garch",
                      "information coefficient", "p-value", "eigen"):
        assert forbidden not in payload, forbidden
    # the footer may *mention* what was left out - that is the point of it
    assert "covariance" in page.lower()


def test_the_page_carries_its_own_data_and_renders_the_headline():
    recs = evaluate(BriefInputs(as_of=AS_OF, violations={"over cap: RELIANCE.NS": 0.04}))
    data = _data(recommendations=recs)
    page = render_brief(data)
    assert "const D = {" in page
    assert data["headline"] in page or "mandate limit" in page
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert payload["headline"] == data["headline"]
    limits_block = next(g for g in payload["recommendations"]
                        if g["question"] == "Are we within limits?")
    assert limits_block["items"][0]["id"] == "LIMIT-01"


def test_dangerous_text_cannot_execute():
    alloc = pd.DataFrame({"current": [0.1], "proposed": [0.2], "change": [0.1]},
                         index=["<script>alert(1)</script>"])
    page = render_brief(_data(allocation=alloc))
    # the script block must not be closeable from inside the data
    body = page.split("const D = ")[1]
    assert "</script>" not in body.split("\n")[0]
    assert "\\u003c/script\\u003e" in page          # it survives as escaped data
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert payload["allocation"][0]["ticker"] == "<script>alert(1)</script>"


def test_it_writes_a_file_that_opens(tmp_path, term_structure):
    out = write_brief(tmp_path / "runs" / "brief.html", _data(term_structure=term_structure))
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>") and text.rstrip().endswith("</html>")
    assert len(text) < 400_000                      # a page, not a payload dump


def test_sections_with_no_data_hide_themselves():
    page = render_brief(_data())
    assert 'el("scenarios-section").style.display = "none"' in page
    data = _data(limits=[{"name": "Max position", "value": 0.07, "limit": 0.10,
                          "headroom": 0.03, "status": "ok"}])
    assert data["limits"][0]["status"] == "ok"


def test_both_light_and_dark_themes_are_defined():
    page = render_brief(_data())
    assert "prefers-color-scheme: dark" in page
    assert '[data-theme="dark"]' in page
    assert "--accent" in page and "--up" in page and "--down" in page


# ------------------------------------------------- the time-series charts ---
def _equity(n=300):
    idx = pd.bdate_range("2023-09-01", periods=n)
    rng = np.random.default_rng(2)
    strat = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)
    bench = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.009, n)), index=idx)
    return pd.DataFrame({"strategy": strat, "benchmark": bench})


def test_the_equity_curve_carries_its_drawdown_and_rebalance_marks():
    eq = _equity()
    marks = list(eq.index[::60])
    data = _data(equity=eq, rebalance_dates=marks)
    e = data["equity"]
    assert len(e["dates"]) == len(e["strategy"]) == len(e["drawdown"])
    assert e["strategy"][0] == pytest.approx(1.0)      # rebased, so the shape is comparable
    assert e["benchmark"][0] == pytest.approx(1.0)
    assert min(e["drawdown"]) < 0 and max(e["drawdown"]) <= 1e-9
    assert len(e["rebalances"]) >= 3
    page = render_brief(data)
    assert 'aria-label="Track record after costs"' in page
    assert 'el("equity-section").style.display = "none"' in page   # hides when absent


def test_the_regime_timeline_is_a_probability_over_time():
    idx = pd.bdate_range("2024-01-01", periods=500)
    stress = pd.Series(np.clip(np.abs(np.sin(np.linspace(0, 12, 500))), 0, 1), index=idx)
    data = _data(regime_history=stress, regime_headline="Markets are calm: 91% likely")
    r = data["regime_history"]
    assert all(0 <= v <= 1 for v in r["stress"])
    assert len(r["dates"]) <= 400                      # subsampled
    assert r["headline"].startswith("Markets are calm")
    assert 'aria-label="Probability the market is in a stress regime"' in render_brief(data)


def test_the_diversification_monitor_is_shown_as_a_trend():
    idx = pd.bdate_range("2024-01-01", periods=400)
    corr = pd.Series(np.linspace(0.1, 0.4, 400), index=idx)
    data = _data(monitor=corr, monitor_headline="Holdings are moving together 0.40")
    m = data["monitor"]
    assert m["values"][0] < m["values"][-1]
    assert "moving together" in m["headline"]


def test_the_evidence_panel_shows_the_error_bar_not_just_the_point():
    data = _data(evidence={"sharpe": 0.79, "sharpe_low": -0.41, "sharpe_high": 2.00,
                           "deflated_sharpe": 0.0, "pbo": 0.42,
                           "min_track_record_years": 4.4, "track_record_years": 2.9,
                           "var_zone": "green", "var_note": "13 breaches where 12 were expected"})
    page = render_brief(data)
    assert data["evidence"]["sharpe_low"] < 0 < data["evidence"]["sharpe_high"]
    assert "zero means no edge at all" in page
    assert "below 0.5: no evidence of skill" in page
    assert "worse than choosing at random" in page


def test_charts_with_no_data_do_not_render_empty_frames():
    page = render_brief(_data())
    for section in ("equity-section", "regime-section", "monitor-section", "evidence-section"):
        assert f'el("{section}").style.display = "none"' in page


# ------------------------------------------------- the reader's footholds ---
def test_every_term_with_a_marker_has_a_definition_behind_it():
    """A superscript that opens nothing is worse than no superscript."""
    page = render_brief(_data(evidence={"sharpe": 0.8, "sharpe_low": 0.1, "sharpe_high": 1.5},
                              limits=[{"name": "Max position", "value": 0.07, "limit": 0.10,
                                       "headroom": 0.03, "status": "ok"}]))
    defined = set(re.findall(r"^  (\w+): \[", page, re.M))
    assert {"drawdown", "sharpe", "volatility", "turnover", "regime"} <= defined
    used = set(re.findall(r'info\("(\w+)"\)', page)) | set(
        re.findall(r'data-info="(\w+)"', page))
    assert used, "no markers were placed at all"
    assert used <= defined, used - defined


def test_the_definitions_are_written_for_someone_who_does_not_know_the_word():
    page = render_brief(_data())
    body = page[page.index("const G = {"):page.index("const info =")]
    assert "highest point" in body                      # drawdown, in plain words
    assert "1 day in 100" in body                       # the risk model check
    # the Sharpe entry has to carry a scale, or the number is unreadable
    assert "0.5 to 1 is respectable" in body and "below 0 is losing money" in body


def test_the_sharpe_number_is_given_a_verdict_and_not_left_bare():
    weak = render_brief(_data(evidence={"sharpe": 0.79, "sharpe_low": -0.41,
                                        "sharpe_high": 2.00}))
    assert "sharpeVerdict" in weak
    assert "respectable for an equity portfolio" in weak
    assert "having no edge at all still fits this record" in weak
    assert "the whole range stays on one side of zero" in weak


# ------------------------------------------------ the board-level summary ---
def test_the_summary_strip_reuses_the_figures_already_on_the_page(term_structure):
    data = _data(term_structure=term_structure, nav_inr=1.0e7,
                 cost_inr=24_000.0, cost_fraction=0.0024, annual_volatility=0.148)
    k = data["key_metrics"]
    labels = [m["label"] for m in k["items"]]
    assert labels[0] == "Portfolio value"
    assert "Most likely outcome" in labels and "A bad run (1 in 20)" in labels
    assert "Cost of acting on this" in labels and "Year-to-year swing" in labels
    longest = data["horizons"][-1]
    assert k["horizon"] == longest["label"]

    median = next(m for m in k["items"] if m["label"] == "Most likely outcome")
    assert median["value"] == pytest.approx(longest["median"])
    assert median["value2"] == pytest.approx(longest["median"] * 1.0e7, rel=1e-6)
    bad = next(m for m in k["items"] if m["label"] == "A bad run (1 in 20)")
    assert bad["value"] == pytest.approx(longest["p5"]) and bad["tone"] == "down"

    page = render_brief(data)
    assert "The numbers that matter" in page
    assert json.dumps(k, allow_nan=False)


def test_the_summary_picks_up_the_worst_scenario_and_the_realised_fall():
    data = _data(scenarios=[{"name": "Covid 2020", "impact": -0.31, "value": -3.1e6},
                            {"name": "Rates 2022", "impact": -0.12, "value": -1.2e6}],
                 equity=_equity())
    labels = {m["label"]: m for m in data["key_metrics"]["items"]}
    assert labels["Worst scenario tested"]["value"] == pytest.approx(-0.31)
    assert labels["Worst scenario tested"]["sub"] == "Covid 2020"
    assert labels["Deepest fall actually suffered"]["value"] < 0


def test_the_summary_hides_itself_when_there_is_nothing_but_the_nav():
    assert _data()["key_metrics"] is None
    assert 'el("metrics-section").style.display = "none"' in render_brief(_data())


# --------------------------------------------------------------- the notes ---
def test_the_notes_are_one_closed_dropdown_rather_than_a_wall():
    data = _data(notes=["cost note", "model note", "a third"],
                 provenance=["how the estimate was made"])
    page = render_brief(data)
    assert page.count("<details") == 1
    assert "<details open" not in page                 # closed until asked for
    assert page.count("<summary") == 1
    # every note survives - collapsing is not deleting
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert payload["notes"] == ["cost note", "model note", "a third"]
    assert payload["provenance"] == ["how the estimate was made"]
    assert "Assumptions, data notes and what is not on this page" in page


def test_the_summary_strip_sits_above_the_notes_dropdown():
    """The order is the argument: the numbers, then the caveats."""
    page = render_brief(_data(cost_inr=1.0, cost_fraction=0.0001))
    assert page.index('id="metrics-section"') < page.index("<details")


def test_the_scenario_table_adds_up_to_the_headline():
    """The owner tried to add the rows and they did not reach the total. Now they do."""
    sc = {"id": "x", "name": "2022 rate shock", "impact": -0.1241, "value": -1.24e6,
          "coverage": 1.0, "note": "Inflation and the fastest tightening cycle in decades.",
          "held": 12,
          "top": [{"ticker": "A.NS", "impact": -0.027, "move": -0.327, "weight": 0.0833},
                  {"ticker": "B.NS", "impact": -0.021, "move": -0.252, "weight": 0.0833},
                  {"ticker": "C.NS", "impact": -0.017, "move": -0.204, "weight": 0.0833},
                  {"ticker": "D.NS", "impact": -0.015, "move": -0.180, "weight": 0.0833}]}
    shown = sum(r["impact"] for r in sc["top"])
    sc["other"] = sc["impact"] - shown          # exactly how make_brief derives it
    assert shown + sc["other"] == pytest.approx(sc["impact"], abs=1e-9)

    page = render_brief(_data(scenarios=[sc]))
    assert "How far it fell" in page and "Cost to the portfolio" in page
    assert "The other ${s.held - top.length} holdings" in page   # the remainder is drawn
    assert "<b>Total</b>" in page
    # the two numbers must be distinguishable, or a fall is read as a portfolio loss
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    row = payload["scenarios"][0]["top"][0]
    assert row["move"] < row["impact"] < 0        # the share fell much further than it cost
