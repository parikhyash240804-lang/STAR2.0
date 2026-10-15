"""
The HTML brief: one page, one decision, no dependencies.

Built for Persona A in docs/PERSONAS.md - the decision-maker with ten minutes,
who is comfortable with Kite and has no use for a covariance matrix. Everything
that does not change a decision she is about to make lives in the workbook
instead.

**Why hand-built SVG rather than a charting library.** The brief has to open from
a file with no internet, respond to every toggle in the same frame, and look like
a trading terminal rather than a template. Embedding a plotting library costs
several megabytes, brings a house style that has to be fought, and renders after
a delay. Charts here are SVG drawn by a few hundred lines of plain JavaScript
from data embedded as JSON: about 60 KB in total, instant, and styled exactly as
intended.

**Everything the page can show is computed before it is written.** The horizon
control and the scenario buttons read a table that is already in the file, so
they are lookups rather than recalculations - which is what makes them feel
immediate, and what guarantees the page can never disagree with the engine.

The template is a module constant rather than a data file so the frozen
executable has nothing to find at runtime.
"""
from __future__ import annotations

import html
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .guidance import Decision, Recommendation, Severity, by_decision, headline_recommendation


# ------------------------------------------------------------ confidence ---
def _horizon_basis(days: int) -> tuple[str, str]:
    """
    What the horizon itself says about how far the answer can be trusted.

    Five years of sessions is about 1,250 observations. A 252-day statement
    therefore rests on roughly five independent years, not on 1,250 facts, and
    saying so is the honest part of the badge.
    """
    independent = max(1, int(1250 // max(days, 1)))
    span = "1 session" if days == 1 else f"{days} sessions"
    if days == 1:
        return ("pass", "1 session - tomorrow, read straight off about 1,250 observed daily "
                        "moves rather than assumed")
    if days <= 126:
        return ("pass", f"{span} - about {independent:,} independent periods of this length in "
                        "the sample, so the range comes from observed moves rather than assumption")
    if days <= 504:
        return ("fair", f"{span} - only about {independent} independent periods of this length in "
                        "five years of history, so the range itself is loosely measured")
    return ("fail", f"{days} sessions - beyond two years this extrapolates from five years of "
                    "history rather than measuring anything")


def confidence_badge(days: int, *, risk_model_passed: bool | None = None,
                     tail_reliable: bool | None = None,
                     regime_settled: bool | None = None,
                     checks: list[dict] | None = None) -> dict:
    """
    The badge rule from docs/PERSONAS.md, stated rather than invented.

    No score out of 100: there is no honest way to compute one, and a made-up
    number reads as more precise than the thing it describes. What the badge does
    carry is `basis` - the individual checks behind the word, each with the
    measured figure that decided it, so the reader can see why it says Fair
    instead of having to take the word for it.
    """
    reasons = []
    if days > 504:
        reasons.append("beyond two years this extrapolates from five years of history")
    if risk_model_passed is False:
        reasons.append("the risk model failed its breach test")
    if tail_reliable is False:
        reasons.append("few independent observations in the tail at this horizon")
    if regime_settled is False:
        reasons.append("the market regime is unsettled")
    if days > 126 and not reasons:
        reasons.append("the sample holds few independent periods this long")

    if not reasons:
        level, why = "Good", "short horizon, calibrated risk model, enough data in the tail"
    elif len(reasons) == 1 and days <= 504 and risk_model_passed is not False:
        level, why = "Fair", reasons[0]
    else:
        level, why = "Weak", "; ".join(reasons)

    verdict, note = _horizon_basis(days)
    basis = [{"name": "How far ahead", "value": note, "verdict": verdict}]
    if risk_model_passed is not None:
        basis.append({
            "name": "Does the risk model hold up",
            "value": ("yes - over the last few years, losses worse than the stated limit arrived "
                      "about as often as the model predicted")
                     if risk_model_passed else
                     ("no - losses worse than the stated limit arrived more often than the model "
                      "predicted, so every figure here understates the risk"),
            "verdict": "pass" if risk_model_passed else "fail"})
    if tail_reliable is not None:
        basis.append({
            "name": "Data behind the bad cases",
            "value": ("enough separate bad stretches in the history to describe the downside")
                     if tail_reliable else
                     ("too few separate bad stretches at this horizon, so the worst cases are "
                      "the shakiest numbers on the page"),
            "verdict": "pass" if tail_reliable else "fail"})
    if regime_settled is not None:
        basis.append({
            "name": "Is the market settled",
            "value": ("yes - conditions today look like one recognisable state, so history is a "
                      "fair guide")
                     if regime_settled else
                     ("no - conditions sit between a calm and a stressed market, and the two "
                      "behave differently enough that the blend is a guess"),
            "verdict": "pass" if regime_settled else "fair"})
    basis.extend(checks or [])
    return {"level": level, "why": why, "basis": basis}


# --------------------------------------------------------- data assembly ---
def _clean(x):
    """JSON cannot hold NaN, and a silent NaN renders as 'null' in the page."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, pd.Series):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%d %b %Y")
    return x


def recommendation_payload(recs: list[Recommendation]) -> list[dict]:
    grouped = by_decision(recs)
    out = []
    for decision in Decision:
        items = grouped.get(decision, [])
        if not items:
            continue
        out.append({
            "question": decision.label,
            # `confidence` is deliberately not carried into the page. It is never
            # rendered, and it is where the method names live - "Euler decomposition",
            # "Kupiec p=0.85", "CSCV". Shipping invisible jargon inside a file meant
            # for a non-technical reader is all cost and no benefit; the field stays on
            # the object, and the workbook prints it.
            "items": [{"id": r.id, "severity": r.severity.label,
                       "rank": int(r.severity), "headline": r.headline, "detail": r.detail,
                       "evidence": list(r.evidence), "falsifier": r.falsifier}
                      for r in items],
        })
    return out


def build_brief_data(*, as_of, nav_inr: float, recommendations: list[Recommendation],
                     term_structure=None, horizons=(1, 5, 21, 63, 126, 252),
                     scenarios: list[dict] | None = None, limits: list[dict] | None = None,
                     risk_contributors: pd.DataFrame | None = None,
                     allocation: pd.DataFrame | None = None,
                     evidence: dict | None = None, provenance: list[str] | None = None,
                     notes: list[str] | None = None,
                     risk_model_passed: bool | None = None,
                     regime_settled: bool | None = None,
                     equity: pd.DataFrame | None = None,
                     rebalance_dates=None,
                     regime_history: pd.Series | None = None,
                     regime_headline: str | None = None,
                     monitor: pd.Series | None = None,
                     monitor_headline: str | None = None,
                     confidence_checks: list[dict] | None = None,
                     cost_inr: float | None = None, cost_fraction: float | None = None,
                     annual_volatility: float | None = None,
                     benchmark_label: str = "NIFTY 50") -> dict:
    """
    Assemble everything the page needs. Pure data: no HTML, no formatting, and
    every figure a number so the page can render it however it likes.
    """
    data = {
        "as_of": _clean(pd.Timestamp(as_of)),
        "nav": _clean(nav_inr),
        "benchmark": benchmark_label,
        "headline": headline_recommendation(recommendations),
        "recommendations": recommendation_payload(recommendations),
        "horizons": [],
        "fan": None,
        "scenarios": scenarios or [],
        "limits": limits or [],
        "risk_contributors": [],
        "allocation": [],
        "evidence": _clean(evidence or {}),
        "provenance": provenance or [],
        "notes": notes or [],
        "equity": None,
        "regime_history": None,
        "monitor": None,
    }

    if term_structure is not None:
        ts = term_structure
        max_days = int(ts.max_days)
        for days in horizons:
            if days > max_days:
                continue
            at = ts.at(days)
            tail_ok = None
            badge = confidence_badge(days, risk_model_passed=risk_model_passed,
                                     tail_reliable=tail_ok, regime_settled=regime_settled,
                                     checks=confidence_checks)
            data["horizons"].append({
                "label": at["label"], "days": days,
                "median": _clean(at["returns"][50]),
                "p5": _clean(at["returns"][5]), "p25": _clean(at["returns"][25]),
                "p75": _clean(at["returns"][75]), "p95": _clean(at["returns"][95]),
                "p_loss": _clean(at["p_loss"]),
                "p_below_benchmark": _clean(at["p_below_benchmark"]),
                "shortfall": _clean(at["expected_shortfall"].get(95)),
                "drawdown": _clean(at["drawdown_at_risk"].get(95)),
                "confidence": badge,
            })
        v0 = ts.initial_value
        step = max(1, -(-max_days // 180))          # at most ~180 points, so the payload stays small
        idx = list(range(0, max_days, step))
        data["fan"] = {
            "days": [int(ts.days[i]) for i in idx],
            "p5": [_clean(ts.quantiles[5][i] / v0 - 1) for i in idx],
            "p25": [_clean(ts.quantiles[25][i] / v0 - 1) for i in idx],
            "median": [_clean(ts.quantiles[50][i] / v0 - 1) for i in idx],
            "p75": [_clean(ts.quantiles[75][i] / v0 - 1) for i in idx],
            "p95": [_clean(ts.quantiles[95][i] / v0 - 1) for i in idx],
        }
        data["drift"] = ts.drift.describe()

    if equity is not None and not equity.empty:
        # subsample to keep the payload small; the shape is what is read, not each day
        e = equity.dropna(how="all")
        step = max(1, -(-len(e) // 400))
        e = e.iloc[::step]
        curve = e["strategy"] / float(e["strategy"].iloc[0])
        peak = curve.cummax()
        data["equity"] = {
            "dates": [d.strftime("%b %Y") for d in e.index],
            "strategy": [_clean(v) for v in curve],
            "benchmark": ([_clean(v) for v in e["benchmark"] / float(e["benchmark"].iloc[0])]
                          if "benchmark" in e else None),
            "drawdown": [_clean(v) for v in (curve / peak - 1.0)],
            "rebalances": [int(i) for i, d in enumerate(e.index)
                           if rebalance_dates is not None and d in set(rebalance_dates)],
        }

    if regime_history is not None and len(regime_history):
        r = pd.Series(regime_history).dropna()
        step = max(1, -(-len(r) // 400))
        r = r.iloc[::step]
        data["regime_history"] = {
            "dates": [d.strftime("%b %Y") for d in r.index],
            "stress": [_clean(v) for v in r],
            "headline": regime_headline,
        }

    if monitor is not None and len(monitor):
        m = pd.Series(monitor).dropna()
        step = max(1, -(-len(m) // 300))
        m = m.iloc[::step]
        data["monitor"] = {
            "dates": [d.strftime("%b %Y") for d in m.index],
            "values": [_clean(v) for v in m],
            "headline": monitor_headline,
        }

    if risk_contributors is not None and not risk_contributors.empty:
        rc = risk_contributors.nlargest(8, "share")
        data["risk_contributors"] = [
            {"ticker": str(t), "weight": _clean(r.get("weight")), "share": _clean(r.get("share"))}
            for t, r in rc.iterrows()]

    if allocation is not None and not allocation.empty:
        a = allocation.copy()
        a["abs_change"] = a["change"].abs()
        a = a.nlargest(10, "abs_change")
        data["allocation"] = [
            {"ticker": str(t), "current": _clean(r.get("current")),
             "proposed": _clean(r.get("proposed")), "change": _clean(r.get("change"))}
            for t, r in a.sort_values("change").iterrows()]

    data["key_metrics"] = _key_metrics(data, nav_inr, cost_inr=cost_inr,
                                       cost_fraction=cost_fraction,
                                       annual_volatility=annual_volatility)
    return data


def _key_metrics(data: dict, nav: float, *, cost_inr=None, cost_fraction=None,
                 annual_volatility=None) -> dict | None:
    """
    The handful of figures a board acts on, gathered into one strip.

    Every entry is read back out of what the page already shows rather than
    recomputed, so the strip can never disagree with the chart above it. A metric
    whose input is missing is left out rather than filled with a placeholder.
    """
    hs = data.get("horizons") or []
    h = hs[-1] if hs else None
    items = [{"label": "Portfolio value", "value": _clean(nav), "format": "inr",
              "sub": "as of " + str(data.get("as_of", "")), "g": "nav"}]

    if h is not None:
        items.append({"label": "Most likely outcome", "value": h["median"], "format": "signed",
                      "value2": _clean((h["median"] or 0.0) * nav), "format2": "inr",
                      "tone": "auto", "g": "median"})
        items.append({"label": "A bad run (1 in 20)", "value": h["p5"], "format": "signed",
                      "value2": _clean((h["p5"] or 0.0) * nav), "format2": "inr",
                      "tone": "down", "g": "badyear"})
        if h.get("drawdown") is not None:
            items.append({"label": "Deepest fall to plan for", "value": -abs(h["drawdown"]),
                          "format": "pct0", "sub": "peak to trough along the way",
                          "tone": "down", "g": "drawdown"})
        if h.get("p_loss") is not None:
            items.append({"label": "Chance of ending down", "value": h["p_loss"], "format": "pct0",
                          "sub": "roughly " + _in_n(h["p_loss"]), "g": "ploss"})
        if h.get("p_below_benchmark") is not None:
            items.append({"label": "Chance of trailing " + str(data.get("benchmark", "the index")),
                          "value": h["p_below_benchmark"], "format": "pct0", "g": "relative"})

    worst = None
    for sc in data.get("scenarios") or []:
        if sc.get("impact") is not None and (worst is None or sc["impact"] < worst["impact"]):
            worst = sc
    if worst is not None:
        items.append({"label": "Worst scenario tested", "value": _clean(worst["impact"]),
                      "format": "signed", "value2": _clean(worst.get("value")), "format2": "inr",
                      "sub": str(worst.get("name", "")), "tone": "down", "g": "scenario"})

    eq = data.get("equity")
    if eq and eq.get("drawdown"):
        items.append({"label": "Deepest fall actually suffered", "value": min(eq["drawdown"]),
                      "format": "pct0", "sub": "over the record that was tested",
                      "tone": "down", "g": "realised_dd"})

    if annual_volatility is not None:
        items.append({"label": "Year-to-year swing", "value": _clean(annual_volatility),
                      "format": "pct0", "sub": "typical distance from the average year",
                      "g": "volatility"})

    if cost_inr is not None:
        items.append({"label": "Cost of acting on this", "value": _clean(cost_inr), "format": "inr",
                      "value2": _clean(cost_fraction), "format2": "pct2",
                      "sub": "certain, unlike the gain", "g": "cost"})

    horizon_label = h["label"] if h is not None else None
    return {"horizon": horizon_label, "items": items} if len(items) > 1 else None


def _in_n(p: float | None) -> str:
    """'40%' means little at a board table; 'about 2 times in 5' is read correctly."""
    if p is None or not np.isfinite(p) or p <= 0:
        return ""
    n = int(round(1 / p)) if p < 0.5 else 0
    if n >= 2:
        return f"1 time in {n}"
    return f"{round(p * 10)} times in 10"


def write_brief(path, data: dict) -> Path:
    """Render and save. Returns the path, so a caller can open it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_brief(data), encoding="utf-8")
    return out


def render_brief(data: dict) -> str:
    """
    Render the page with its data embedded.

    The JSON is escaped before it goes into the `<script>` block. A ticker or a
    note containing `</script>` would otherwise close the tag early, and whatever
    followed would be parsed as markup - the classic way a data-driven page turns
    into an injection. Escaping `<`, `>` and `&` as unicode escapes is still valid
    JSON and parses back to the identical string, so nothing downstream changes.
    U+2028 and U+2029 are escaped too: legal in JSON, but line terminators in
    JavaScript, where they would be a syntax error.
    """
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"))
    for raw, safe in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                      (" ", "\\u2028"), (" ", "\\u2029")):
        payload = payload.replace(raw, safe)
    return _TEMPLATE.replace("__TITLE__", html.escape(f"Portfolio brief - {data.get('as_of', '')}")) \
                    .replace("__DATA__", payload)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#ffffff; --panel:#ffffff; --ink:#111418; --muted:#6b7280; --line:#e8eaed;
  --accent:#2f6fed; --up:#0f9d58; --down:#d93025; --warn:#b26a00; --chip:#f3f4f6;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0f1115; --panel:#151922; --ink:#e8eaed; --muted:#9aa4b2; --line:#232937;
    --accent:#6ea0ff; --up:#35c47a; --down:#ff6b5e; --warn:#e0a640; --chip:#1c2230;
  }
}
:root[data-theme="dark"]{
  --bg:#0f1115; --panel:#151922; --ink:#e8eaed; --muted:#9aa4b2; --line:#232937;
  --accent:#6ea0ff; --up:#35c47a; --down:#ff6b5e; --warn:#e0a640; --chip:#1c2230;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
h1,h2,h3{font-weight:600;margin:0}
.eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}
header .meta{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:14px}
header h1{font-size:21px;line-height:1.35;max-width:46em;letter-spacing:-.01em}
section{margin:34px 0}
section>.eyebrow{margin-bottom:12px}
.num{font-size:26px;font-weight:600;letter-spacing:-.02em}
.num.sm{font-size:18px}
.up{color:var(--up)} .down{color:var(--down)}
.muted{color:var(--muted)}
.row{display:flex;gap:28px;flex-wrap:wrap}
.stat{min-width:120px}
.stat .k{font-size:11px;color:var(--muted);margin-bottom:3px}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:18px}
button{font:inherit;color:var(--muted);background:transparent;border:1px solid var(--line);
  border-radius:6px;padding:6px 13px;cursor:pointer;transition:none}
button:hover{color:var(--ink);border-color:var(--muted)}
button[aria-pressed="true"]{color:#fff;background:var(--accent);border-color:var(--accent)}
:root[data-theme="dark"] button[aria-pressed="true"],
@media (prefers-color-scheme:dark){button[aria-pressed="true"]{color:#0f1115}}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;background:var(--chip);
  color:var(--muted);cursor:help;border:1px solid var(--line)}
.badge.good{color:var(--up)} .badge.fair{color:var(--warn)} .badge.weak{color:var(--down)}
/* the definition marker: small enough to ignore, large enough to hit on a phone */
sup.i{display:inline-block;width:13px;height:13px;line-height:13px;margin-left:3px;
  border:1px solid var(--line);border-radius:50%;background:var(--chip);color:var(--muted);
  font-size:9px;font-weight:600;font-style:italic;text-align:center;cursor:help;
  vertical-align:super;user-select:none;position:relative;top:-1px}
sup.i:hover,sup.i:focus{color:var(--accent);border-color:var(--accent);outline:none}
.tip b{display:block;margin-bottom:3px}
.tip .g{white-space:normal;max-width:320px;display:block;color:var(--ink)}
.tip .why{white-space:normal;max-width:340px;display:block}
.tip .why div{margin-top:5px;padding-left:12px;position:relative}
.tip .why div::before{position:absolute;left:0;top:0}
.tip .why .pass::before{content:"✓";color:var(--up)}
.tip .why .fair::before{content:"–";color:var(--warn)}
.tip .why .fail::before{content:"✕";color:var(--down)}
.tip .why .n{color:var(--ink);font-weight:500}
details{border-top:1px solid var(--line);padding-top:14px}
details summary{cursor:pointer;color:var(--muted);list-style:none;font-size:13px;
  display:flex;align-items:center;gap:8px}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"›";display:inline-block;transition:transform .12s;
  font-size:16px;line-height:1}
details[open] summary::before{transform:rotate(90deg)}
details summary:hover{color:var(--ink)}
details .body{margin-top:14px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:22px 28px}
.metrics .m .k{font-size:11px;color:var(--muted);margin-bottom:3px}
.metrics .m .v{font-size:21px;font-weight:600;letter-spacing:-.02em}
.metrics .m .s{font-size:11px;color:var(--muted);margin-top:2px}
@media (max-width:820px){.metrics{grid-template-columns:repeat(2,1fr)}}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:right;font-weight:500;color:var(--muted);font-size:11px;padding:6px 0;
  border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.05em}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:7px 0;border-bottom:1px solid var(--line)}
/* columns need air between them, or narrow screens run the headings together */
th+th,td+td{padding-left:14px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
tr:last-child td{border-bottom:none}
.bar{height:6px;border-radius:3px;background:var(--accent);display:inline-block;vertical-align:middle}
.rec{padding:14px 0;border-bottom:1px solid var(--line)}
.rec:last-child{border-bottom:none}
.rec .h{font-weight:500;margin-bottom:4px}
.rec .d{color:var(--muted);font-size:13px}
.rec .ev{color:var(--muted);font-size:12px;margin-top:6px}
.tag{font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:2px 7px;border-radius:4px;
  background:var(--chip);color:var(--muted);margin-right:8px;white-space:nowrap}
.tag.act{color:var(--down)} .tag.watch{color:var(--warn)} .tag.none{color:var(--muted)}
.light{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}
.light.ok{background:var(--up)} .light.near{background:var(--warn)} .light.bad{background:var(--down)}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:8px 10px;font-size:12px;opacity:0;transition:opacity .08s;z-index:10;
  box-shadow:0 1px 3px rgba(0,0,0,.08);white-space:nowrap}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
footer li{margin:4px 0}
@media (max-width:640px){
  .wrap{padding:20px 16px 48px}.num{font-size:22px}.num.sm{font-size:16px}
  header h1{font-size:18px}table{font-size:12px}th{font-size:10px}
  .row{gap:18px}.stat{min-width:104px}.metrics .m .v{font-size:18px}
}
@media print{details{display:block}details[open] .body,details .body{display:block}}
@media print{button{display:none}body{background:#fff}.wrap{max-width:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="meta">
      <span id="asof"></span><span id="nav"></span><span id="drift"></span>
    </div>
    <div class="eyebrow">Recommendation</div>
    <h1 id="headline"></h1>
  </header>

  <section id="outlook">
    <div class="eyebrow" data-info="fan">Expected outcome</div>
    <div class="controls" id="horizons"></div>
    <div class="row" id="outcome"></div>
    <div id="fan" style="margin-top:22px"></div>
  </section>

  <section id="scenarios-section">
    <div class="eyebrow" data-info="scenario">If this happened</div>
    <div class="controls" id="scenarios"></div>
    <div id="scenario-detail"></div>
  </section>

  <section id="limits-section">
    <div class="eyebrow" data-info="headroom">Limits</div>
    <div id="limits"></div>
  </section>

  <section id="risk-section">
    <div class="eyebrow" data-info="riskshare">Where the risk sits</div>
    <div id="risk"></div>
  </section>

  <section id="alloc-section">
    <div class="eyebrow" data-info="proposed">Proposed changes</div>
    <div id="alloc"></div>
  </section>

  <section id="equity-section">
    <div class="eyebrow" data-info="equitycurve">Track record, after costs</div>
    <div id="equity"></div>
  </section>

  <section id="regime-section">
    <div class="eyebrow" data-info="regime">Market conditions</div>
    <div class="muted" id="regime-head" style="margin-bottom:12px"></div>
    <div id="regime"></div>
  </section>

  <section id="monitor-section">
    <div class="eyebrow" data-info="correlation">Diversification</div>
    <div class="muted" id="monitor-head" style="margin-bottom:12px"></div>
    <div id="monitor"></div>
  </section>

  <section id="evidence-section">
    <div class="eyebrow" data-info="evidence">How strong is the evidence</div>
    <div id="evidence"></div>
  </section>

  <section id="recs">
    <div class="eyebrow">What to decide</div>
    <div id="recommendations"></div>
  </section>

  <section id="metrics-section">
    <div class="eyebrow">The numbers that matter</div>
    <div class="muted" id="metrics-head" style="margin-bottom:16px"></div>
    <div class="metrics" id="metrics"></div>
  </section>

  <footer>
    <details id="notes-block">
      <summary id="notes-summary">Assumptions, data notes and what is not on this page</summary>
      <div class="body">
        <div id="notes"></div>
        <div id="prov" style="margin-top:12px"></div>
        <div style="margin-top:12px">Detail deliberately left out of this page - the covariance
          matrix, model parameters, full statistics and per-holding history - is in the
          accompanying workbook.</div>
      </div>
    </details>
  </footer>
</div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const pct = (v, dp) => v === null || v === undefined ? "--" : (v * 100).toFixed(dp === undefined ? 1 : dp) + "%";
const signed = (v, dp) => v === null || v === undefined ? "--" : (v >= 0 ? "+" : "") + (v * 100).toFixed(dp === undefined ? 1 : dp) + "%";
const inr = v => v === null || v === undefined ? "--"
  : (v < 0 ? "-₹" : "₹") + Math.round(Math.abs(v)).toLocaleString("en-IN");
const cls = v => v === null || v === undefined ? "" : (v >= 0 ? "up" : "down");
const el = id => document.getElementById(id);
const esc = s => String(s === null || s === undefined ? "" : s).replace(/[&<>"]/g, c => (
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* ---------- glossary ----------
   Every piece of jargon on this page has an entry here, reached from the small
   "i" beside the label. The definitions live in the template rather than in the
   data because they are identical on every run, and because the payload is held
   to a no-jargon rule that this text would otherwise break. */
const G = {
  nav: ["Portfolio value",
    "What the holdings are worth today at the last traded prices. Everything else on this page is a percentage of this number."],
  median: ["Most likely outcome",
    "The middle of every simulated future: half the simulations end better than this, half end worse. It is not a target and not a promise. It is the centre of a wide range, and the range matters more than the centre."],
  range: ["The 5% to 95% range",
    "In 19 simulations out of 20 the outcome lands between these two numbers. One in 20 lands outside, and outside can be worse than the lower figure shown."],
  badyear: ["A bad run (1 in 20)",
    "The outcome that only 5% of simulations were worse than. A reasonable figure to hold capital against, but not the worst case, because 1 year in 20 is worse still."],
  ploss: ["Chance of ending down",
    "The share of simulated futures that finish below where the portfolio starts. It says nothing about how far down, so read it alongside the bad-run figure."],
  relative: ["Chance of trailing the index",
    "How often the portfolio finishes behind the benchmark. A portfolio can make money and still lose on this measure, which is the one a board is usually judged on."],
  drawdown: ["Drawdown",
    "The fall from the highest point reached to the lowest point after it, before a new high. It is what an investor actually lives through: a portfolio that ends the year up 10% may have been down 25% in the middle of it. This figure is the depth that only 5% of simulated paths exceeded."],
  realised_dd: ["Deepest fall actually suffered",
    "The worst peak-to-trough fall in the tested track record. Something that happened, not something simulated."],
  volatility: ["Volatility",
    "How far returns typically stray from their average, stated as a yearly figure. Roughly two years in three land within one volatility of the average; the other one in three is what the stress tests are for."],
  confidence: ["What the confidence word means",
    "Good, Fair or Weak, and not a score. It is the result of a handful of checks, each listed with the number behind it when you hover the badge itself. There is no honest way to compress those checks into a mark out of 100, so none is shown."],
  sharpe: ["Sharpe ratio",
    "Return earned per unit of risk taken. As a rough scale for a long-only equity book: below 0 is losing money for the trouble, 0 to 0.5 is weak, 0.5 to 1 is respectable, above 1 is strong, and above 2 is rare enough to be worth double-checking. The bracketed range is the error bar: if it crosses zero, the record cannot yet rule out having no skill at all."],
  dsr: ["Sharpe after allowing for trial and error",
    "Test enough strategies and one of them looks good by luck alone. This adjusts the Sharpe ratio for how many were tried before this one was picked. Below 0.5 there is no evidence of genuine skill left once that allowance is made."],
  pbo: ["Chance the choice was luck",
    "The record is repeatedly cut in half: fitted on one half, checked on the other. This is how often the configuration that looked best on the first half turned out to be below average on the second. Above 50% is worse than picking at random."],
  mintrl: ["Track record needed",
    "How many years of results it would take before a Sharpe ratio this size could be called real rather than chance. Compare it with how many years actually exist."],
  varmodel: ["Risk model check",
    "The model states a daily loss that should only be exceeded 1 day in 100. Counting how often it actually was exceeded over the last few years tests whether the model is honest. Green means the count matched. Red means losses arrived more often than stated, and every risk figure on this page is then too small."],
  correlation: ["How much holdings move together",
    "The average correlation between the holdings. Near 0 they move independently and diversification is working. Near 1 they move as one, the portfolio is effectively a single bet, and it is exactly during a crisis that this number rises."],
  regime: ["Market regime",
    "Markets alternate between calm stretches and stressed ones that behave differently: bigger moves, and holdings falling together. This is the estimated probability that today sits in the stressed state, read from the pattern of recent moves."],
  riskshare: ["Share of risk",
    "How much of the total portfolio risk a single holding accounts for. It is not the same as its weight: a 5% position in a volatile name that moves with everything else can carry more risk than a 15% position in a steady one."],
  turnover: ["Turnover",
    "The fraction of the portfolio bought or sold at this rebalance. Every unit of it costs brokerage, stamp duty, taxes and the price moved by the order itself, and those costs are certain while the benefit is not."],
  headroom: ["Headroom",
    "The distance between where a limit stands today and where it is allowed to go. Negative headroom is a breach."],
  scenario: ["Scenario test",
    "What this portfolio would lose if a past crisis, or a stated shock, happened again to the holdings as they stand today. It is not a forecast and carries no probability: it is a sizing exercise for whether the loss would be survivable."],
  equitycurve: ["Track record, after costs",
    "The value of the portfolio through the tested period with every trading cost deducted, against the benchmark. The red band underneath is how far below its previous high it was at each point."],
  cost: ["Cost of acting",
    "Brokerage, exchange and regulatory fees, stamp duty, securities transaction tax, depository charges and the price moved by the order itself. This money leaves whether or not the change turns out to be right."],
  fan: ["The range of outcomes",
    "Each shaded band is a share of the simulated futures: the darker inner band holds the middle half, the lighter outer band holds 19 in 20. The bands widen with time because uncertainty compounds."],
  hardest: ["Hardest hit",
    "The few holdings that lost the most in this scenario, worst first. It is a shortlist, not the whole portfolio, so the rows shown do not add up to the total on their own - the holdings not listed are totalled on the line beneath them."],
  ownmove: ["How far it fell",
    "What happened to the share price itself in this scenario. A stock can fall a long way and still barely dent the portfolio if the position in it is small."],
  contrib: ["Cost to the portfolio",
    "How many percentage points this single holding took off the whole portfolio: how far it fell, multiplied by how much of the portfolio sits in it. These add up, across every holding, to the total impact."],
  proposed: ["Proposed changes",
    "What the model would hold, set against what is held now. Nothing is executed from this page: it is the list a dealer would work from."],
  evidence: ["How strong is the evidence",
    "Whether the track record is long enough and clean enough to support a decision at all. These are the numbers that say when the answer is that we cannot tell yet, which is a legitimate answer and the one most often left out."],
};
const info = k => G[k] ? `<sup class="i" tabindex="0" data-g="${k}">i</sup>` : "";
function gTip(e, k) {
  const g = G[k];
  if (!g) return;
  showTip(e, "<b>" + esc(g[0]) + "</b><span class='g'>" + esc(g[1]) + "</span>");
}
const marker = e => (e.target && e.target.closest) ? e.target.closest("sup.i") : null;
document.addEventListener("mousemove", e => { const t = marker(e); if (t) gTip(e, t.dataset.g); });
document.addEventListener("mouseout", e => { if (marker(e)) hideTip(); });
document.addEventListener("click", e => {      /* touch: tap to read, tap away to dismiss */
  const t = marker(e);
  if (t) gTip(e, t.dataset.g); else hideTip();
});

/* ---------- header ---------- */
el("asof").textContent = "As of " + D.as_of;
if (D.nav) el("nav").textContent = "Portfolio " + inr(D.nav);
if (D.drift) el("drift").textContent = D.drift;
el("headline").textContent = D.headline;

/* ---------- tooltip ---------- */
const tip = el("tip");
function showTip(e, html_) {
  tip.innerHTML = html_;
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY - h - pad;
  if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
  if (y < 8) y = e.clientY + pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => tip.style.opacity = 0;

/* ---------- the confidence badge, opened up ----------
   The word on its own invites the question "on what basis". The answer is the
   list of checks the badge is made of, each with the measured figure behind it,
   so the reader can disagree with a specific line rather than with a label. */
function badgeTip(c) {
  const rows = (c.basis || []).map(b =>
    `<div class="${esc(b.verdict)}"><span class="n">${esc(b.name)}:</span> ${esc(b.value)}</div>`).join("");
  return "<b>Confidence: " + esc(c.level) + "</b><span class='why'>" +
    (rows ? rows : esc(c.why)) + "</span>";
}

/* ---------- horizon selector ---------- */
let current = 0;
function drawHorizons() {
  const box = el("horizons");
  if (!D.horizons.length) { el("outlook").style.display = "none"; return; }
  box.innerHTML = "";
  D.horizons.forEach((h, i) => {
    const b = document.createElement("button");
    b.textContent = h.label;
    b.setAttribute("aria-pressed", i === current);
    b.onclick = () => { current = i; drawHorizons(); drawOutcome(); drawFan(); };
    box.appendChild(b);
  });
}
function drawOutcome() {
  const h = D.horizons[current];
  const c = h.confidence || {level: "", why: ""};
  const stat = (k, v, klass) => `<div class="stat"><div class="k">${k}</div>
    <div class="num sm ${klass || ""}">${v}</div></div>`;
  el("outcome").innerHTML =
    `<div class="stat"><div class="k">Median outcome${info("median")}</div>
       <div class="num ${cls(h.median)}">${signed(h.median)}</div></div>` +
    stat("Range (5&ndash;95%)" + info("range"), signed(h.p5) + " to " + signed(h.p95)) +
    stat("Chance of a loss" + info("ploss"), pct(h.p_loss, 0)) +
    (h.p_below_benchmark === null ? "" :
      stat("Chance of trailing " + esc(D.benchmark) + info("relative"),
           pct(h.p_below_benchmark, 0))) +
    stat("Worst drawdown (95%)" + info("drawdown"),
         pct(-Math.abs(h.drawdown), 0), "down") +
    `<div class="stat"><div class="k">Confidence${info("confidence")}</div>
      <div style="margin-top:6px"><span class="badge ${c.level.toLowerCase()}">${esc(c.level)}</span></div></div>`;
  const badge = el("outcome").querySelector(".badge");
  if (badge) {
    badge.onmousemove = e => showTip(e, badgeTip(c));
    badge.onmouseleave = hideTip;
  }
}

/* ---------- fan chart ----------
   The whole horizon is always drawn, so the shape of the outcome range is
   visible at a glance. Choosing a horizon marks that point rather than cutting
   the chart down to it - a 1-day selection would otherwise draw a single dot. */
function drawFan() {
  if (!D.fan) { el("fan").innerHTML = ""; return; }
  const f = D.fan, W = 940, H = 260, L = 8, R = 58, T = 12, B = 26;
  const days = f.days, n = days.length;
  const S = ["p5", "p25", "median", "p75", "p95"].reduce((o, k) => (o[k] = f[k], o), {});
  const lo = Math.min(...S.p5), hi = Math.max(...S.p95);
  const pad = (hi - lo) * 0.08 || 0.01;
  const x = i => L + (W - L - R) * (n < 2 ? 0 : i / (n - 1));
  const y = v => T + (H - T - B) * (1 - (v - lo + pad) / (hi - lo + 2 * pad));
  const line = (k, upto) => S[k].slice(0, upto).map((v, i) =>
    (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
  const band = (a, b, upto) => {
    const fwd = S[a].slice(0, upto).map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1));
    const back = S[b].slice(0, upto).map((v, i) => [x(i), y(v)]).reverse()
      .map(p => "L" + p[0].toFixed(1) + " " + p[1].toFixed(1));
    return fwd.concat(back).join(" ") + " Z";
  };
  /* the selected horizon, as an index into the sampled days */
  const sel = D.horizons[current].days;
  let k = days.findIndex(d => d >= sel);
  if (k < 0) k = n - 1;
  const kx = x(k);

  const ticks = [lo, (lo + hi) / 2, hi].map(v =>
    `<line x1="${L}" y1="${y(v)}" x2="${W - R}" y2="${y(v)}" stroke="var(--line)"/>
     <text x="${W - R + 8}" y="${y(v) + 4}" fill="var(--muted)" font-size="11">${signed(v, 0)}</text>`).join("");
  const zero = (lo <= 0 && hi >= 0) ?
    `<line x1="${L}" y1="${y(0)}" x2="${W - R}" y2="${y(0)}" stroke="var(--muted)" stroke-dasharray="3 3"/>` : "";
  el("fan").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Range of outcomes over time">
      ${ticks}${zero}
      <g opacity="0.45">
        <path d="${band("p5","p95",n)}" fill="var(--accent)" opacity="0.10"/>
        <path d="${band("p25","p75",n)}" fill="var(--accent)" opacity="0.18"/>
        <path d="${line("median",n)}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
      </g>
      <path d="${band("p5","p95",k+1)}" fill="var(--accent)" opacity="0.12"/>
      <path d="${band("p25","p75",k+1)}" fill="var(--accent)" opacity="0.22"/>
      <path d="${line("median",k+1)}" fill="none" stroke="var(--accent)" stroke-width="2"/>
      <line x1="${kx}" y1="${T}" x2="${kx}" y2="${H-B}" stroke="var(--accent)" stroke-width="1"
            stroke-dasharray="3 3" opacity="0.6"/>
      <circle cx="${kx}" cy="${y(S.median[k])}" r="4" fill="var(--accent)"/>
      <circle id="dot" r="3.5" fill="var(--ink)" opacity="0"/>
      <rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" id="hit"/>
      <text x="${L}" y="${H - 6}" fill="var(--muted)" font-size="11">today</text>
      <text x="${kx}" y="${H - 6}" fill="var(--accent)" font-size="11" text-anchor="middle">
        ${esc(D.horizons[current].label)}</text>
      <text x="${W - R}" y="${H - 6}" fill="var(--muted)" font-size="11" text-anchor="end">
        ${days[n-1]} sessions</text>
    </svg>`;
  const svg = el("fan").querySelector("svg"), dot = el("dot");
  el("hit").onmousemove = e => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width * W;
    let i = Math.round((px - L) / (W - L - R) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(S.median[i]));
    dot.setAttribute("opacity", 1);
    showTip(e, `<b>Day ${days[i]}</b><br>
      median ${signed(S.median[i])}<br>
      middle half ${signed(S.p25[i])} to ${signed(S.p75[i])}<br>
      <span class="muted">5&ndash;95%: ${signed(S.p5[i])} to ${signed(S.p95[i])}</span>`);
  };
  el("hit").onmouseleave = () => { hideTip(); dot.setAttribute("opacity", 0); };
}

/* ---------- scenarios ---------- */
let scenario = 0;
function drawScenarios() {
  if (!D.scenarios.length) { el("scenarios-section").style.display = "none"; return; }
  const box = el("scenarios");
  box.innerHTML = "";
  D.scenarios.forEach((s, i) => {
    const b = document.createElement("button");
    b.textContent = s.name;
    b.setAttribute("aria-pressed", i === scenario);
    b.onclick = () => { scenario = i; drawScenarios(); };
    box.appendChild(b);
  });
  const s = D.scenarios[scenario];
  /* Two different numbers, and conflating them is the easiest way to lose a reader:
     how far the share itself fell, and how many percentage points of the whole
     portfolio that cost. The rows shown are only the worst few, so the holdings not
     listed get a line of their own and the column is totalled - otherwise the figures
     visibly fail to add up to the headline and the page looks wrong. */
  const top = s.top || [];
  const rows = top.map(t => {
    const w = Math.abs(t.impact) / Math.max(...(top.map(z => Math.abs(z.impact))), 1e-9) * 100;
    return `<tr><td>${esc(t.ticker)}</td>
      <td class="muted">${t.move === null || t.move === undefined ? "--" : signed(t.move, 0)}</td>
      <td><span class="bar" style="width:${w.toFixed(0)}px;background:var(--down)"></span></td>
      <td class="${cls(t.impact)}">${signed(t.impact, 1)}</td></tr>`;
  }).join("") +
  (s.other === null || s.other === undefined || !s.held ? "" :
    `<tr><td class="muted">The other ${s.held - top.length} holdings</td>
       <td></td><td></td><td class="muted">${signed(s.other, 1)}</td></tr>`) +
  (top.length ? `<tr><td><b>Total</b></td><td></td><td></td>
       <td class="${cls(s.impact)}"><b>${signed(s.impact, 1)}</b></td></tr>` : "");
  el("scenario-detail").innerHTML =
    `<div class="row" style="margin-bottom:14px">
       <div class="stat"><div class="k">Portfolio impact${info("scenario")}</div>
         <div class="num ${cls(s.impact)}">${signed(s.impact)}</div></div>
       ${s.value === null || s.value === undefined ? "" :
         `<div class="stat"><div class="k">In rupees</div>
            <div class="num sm ${cls(s.impact)}">${inr(s.value)}</div></div>`}
       ${s.coverage === null || s.coverage === undefined ? "" :
         `<div class="stat"><div class="k">Holdings with real data</div>
            <div class="num sm">${pct(s.coverage, 0)}</div></div>`}
     </div>
     ${s.note ? `<div class="muted" style="margin-bottom:12px">${esc(s.note)}</div>` : ""}
     ${rows ? `<div class="tw"><table><thead><tr>
         <th>Hardest hit${info("hardest")}</th><th>How far it fell${info("ownmove")}</th><th></th>
         <th>Cost to the portfolio${info("contrib")}</th></tr></thead>
       <tbody>${rows}</tbody></table></div>` : ""}`;
}

/* ---------- limits ---------- */
function drawLimits() {
  if (!D.limits.length) { el("limits-section").style.display = "none"; return; }
  el("limits").innerHTML =
    `<div class="tw"><table><thead><tr><th>Limit</th><th>Now</th><th>Allowed</th><th>Headroom${info("headroom")}</th></tr></thead><tbody>` +
    D.limits.map(l => {
      const state = l.status === "breach" ? "bad" : (l.status === "near" ? "near" : "ok");
      const fmt = l.format === "number" ? (v => v === null ? "--" : v.toFixed(0)) : (v => pct(v, 1));
      return `<tr><td><span class="light ${state}"></span>${esc(l.name)}</td>
        <td>${fmt(l.value)}</td><td class="muted">${fmt(l.limit)}</td>
        <td class="${l.status === "breach" ? "down" : ""}">${fmt(l.headroom)}</td></tr>`;
    }).join("") + `</tbody></table></div>`;
}

/* ---------- risk contributors ---------- */
function drawRisk() {
  if (!D.risk_contributors.length) { el("risk-section").style.display = "none"; return; }
  const max = Math.max(...D.risk_contributors.map(r => r.share || 0), 1e-9);
  el("risk").innerHTML =
    `<div class="tw"><table><thead><tr><th>Holding</th><th>Weight</th><th></th><th>Share of risk${info("riskshare")}</th></tr></thead><tbody>` +
    D.risk_contributors.map(r => `<tr><td>${esc(r.ticker)}</td>
      <td class="muted">${pct(r.weight, 1)}</td>
      <td style="width:180px"><span class="bar" style="width:${(r.share / max * 160).toFixed(0)}px"></span></td>
      <td>${pct(r.share, 0)}</td></tr>`).join("") + `</tbody></table></div>`;
}

/* ---------- allocation changes ---------- */
function drawAlloc() {
  if (!D.allocation.length) { el("alloc-section").style.display = "none"; return; }
  const max = Math.max(...D.allocation.map(a => Math.abs(a.change || 0)), 1e-9);
  el("alloc").innerHTML =
    `<div class="tw"><table><thead><tr><th>Holding</th><th>Now</th><th>Proposed</th><th></th><th>Change</th></tr></thead><tbody>` +
    D.allocation.map(a => {
      const w = Math.abs(a.change) / max * 80;
      const left = a.change < 0 ? 80 - w : 80;
      return `<tr><td>${esc(a.ticker)}</td><td class="muted">${pct(a.current, 1)}</td>
        <td>${pct(a.proposed, 1)}</td>
        <td style="width:170px"><span style="display:inline-block;width:160px">
          <span class="bar" style="margin-left:${left.toFixed(0)}px;width:${w.toFixed(0)}px;
            background:${a.change < 0 ? "var(--down)" : "var(--up)"}"></span></span></td>
        <td class="${cls(a.change)}">${signed(a.change, 1)}</td></tr>`;
    }).join("") + `</tbody></table></div>`;
}

/* ---------- recommendations ---------- */
function drawRecs() {
  if (!D.recommendations.length) { el("recs").style.display = "none"; return; }
  el("recommendations").innerHTML = D.recommendations.map(g =>
    `<div style="margin-bottom:26px">
      <h3 style="font-size:14px;margin-bottom:6px">${esc(g.question)}</h3>` +
      g.items.map(r => {
        const t = r.severity === "Act" ? "act" : (r.severity === "Watch" ? "watch" : "none");
        return `<div class="rec">
          <div class="h"><span class="tag ${t}">${esc(r.severity)}</span>${esc(r.headline)}</div>
          <div class="d">${esc(r.detail)}</div>
          ${r.evidence && r.evidence.length ?
            `<div class="ev">${r.evidence.map(esc).join(" &middot; ")}</div>` : ""}
          ${r.falsifier ? `<div class="ev">Would be wrong if: ${esc(r.falsifier)}</div>` : ""}
        </div>`;
      }).join("") + `</div>`).join("");
}

/* ---------- the numbers that matter ----------
   Everything here is read back out of the payload the charts above already use,
   so the strip cannot drift from them. It sits last because it is the summary a
   reader takes away, not the working. */
function drawMetrics() {
  const K = D.key_metrics;
  if (!K || !K.items || !K.items.length) { el("metrics-section").style.display = "none"; return; }
  const LONG = {"1D": "day", "1W": "week", "2W": "fortnight", "1M": "month",
                "2M": "two months", "3M": "three months", "6M": "six months",
                "9M": "nine months", "1Y": "year", "2Y": "two years", "3Y": "three years"};
  if (K.horizon) el("metrics-head").textContent =
    "Over the next " + (LONG[K.horizon] || K.horizon) + ", on the holdings as they stand today.";
  const f = (v, kind) => kind === "inr" ? inr(v)
    : kind === "signed" ? signed(v)
    : kind === "pct0" ? pct(v, 0)
    : kind === "pct2" ? pct(v, 2)
    : esc(v);
  el("metrics").innerHTML = K.items.map(m => {
    const tone = m.tone === "auto" ? cls(m.value) : (m.tone || "");
    const parts = [];
    if (m.value2 !== undefined && m.value2 !== null) parts.push(f(m.value2, m.format2));
    if (m.sub) parts.push(esc(m.sub));
    return `<div class="m"><div class="k">${esc(m.label)}${info(m.g)}</div>
      <div class="v ${tone}">${f(m.value, m.format)}</div>
      ${parts.length ? `<div class="s">${parts.join(" &middot; ")}</div>` : ""}</div>`;
  }).join("");
}

/* ---------- footer ----------
   One dropdown, closed. The notes are real and none of them is deleted, but a
   page that opens with fourteen caveats reads as a disclaimer rather than as a
   recommendation. */
function drawFooter() {
  const list = xs => "<ul style='margin:0;padding-left:18px'>" +
    xs.map(n => "<li>" + esc(n) + "</li>").join("") + "</ul>";
  if (D.notes.length) el("notes").innerHTML = list(D.notes);
  if (D.provenance.length) el("prov").innerHTML = list(D.provenance);
  const n = D.notes.length + D.provenance.length;
  if (n) el("notes-summary").textContent =
    "Assumptions, data notes and what is not on this page (" + n + ")";
  if (!n) el("notes-block").querySelector("#notes").remove();
}

/* ---------- shared helpers for the small time-series charts ---------- */
function sparkFrame(W, H, L, R, T, B, lo, hi, n) {
  const pad = (hi - lo) * 0.08 || 0.01;
  return {
    x: i => L + (W - L - R) * (n < 2 ? 0 : i / (n - 1)),
    y: v => T + (H - T - B) * (1 - (v - lo + pad) / (hi - lo + 2 * pad)),
  };
}
function attachHover(svgHost, W, L, R, n, describe) {
  const svg = svgHost.querySelector("svg"), hit = svgHost.querySelector(".hit");
  if (!hit) return;
  hit.onmousemove = e => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width * W;
    let i = Math.round((px - L) / (W - L - R) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    const cur = svgHost.querySelector(".cursor");
    if (cur) { cur.setAttribute("x1", (L + (W-L-R) * (n<2?0:i/(n-1))).toFixed(1));
               cur.setAttribute("x2", (L + (W-L-R) * (n<2?0:i/(n-1))).toFixed(1));
               cur.setAttribute("opacity", 0.6); }
    showTip(e, describe(i));
  };
  hit.onmouseleave = () => {
    hideTip();
    const cur = svgHost.querySelector(".cursor");
    if (cur) cur.setAttribute("opacity", 0);
  };
}

/* ---------- equity curve, after costs, with a drawdown underlay ---------- */
function drawEquity() {
  const e = D.equity;
  if (!e) { el("equity-section").style.display = "none"; return; }
  const W = 940, H = 240, L = 8, R = 58, T = 12, B = 48, DD = 52;
  const n = e.dates.length;
  const all = e.benchmark ? e.strategy.concat(e.benchmark) : e.strategy;
  const lo = Math.min(...all), hi = Math.max(...all);
  const F = sparkFrame(W, H - DD, L, R, T, 6, lo, hi, n);
  const line = arr => arr.map((v, i) => (i ? "L" : "M") + F.x(i).toFixed(1) + " " + F.y(v).toFixed(1)).join(" ");
  const ddLo = Math.min(...e.drawdown);
  const ddY = v => (H - DD + 8) + (DD - 16) * (v / (ddLo || -0.01));
  const ddPath = e.drawdown.map((v, i) => (i ? "L" : "M") + F.x(i).toFixed(1) + " " + ddY(v).toFixed(1)).join(" ")
    + ` L${F.x(n-1).toFixed(1)} ${(H-DD+8).toFixed(1)} L${F.x(0).toFixed(1)} ${(H-DD+8).toFixed(1)} Z`;
  const marks = (e.rebalances || []).map(i =>
    `<line x1="${F.x(i).toFixed(1)}" y1="${T}" x2="${F.x(i).toFixed(1)}" y2="${H-DD}"
      stroke="var(--muted)" opacity="0.18" stroke-width="1"/>`).join("");
  const ticks = [lo, hi].map(v =>
    `<line x1="${L}" y1="${F.y(v)}" x2="${W-R}" y2="${F.y(v)}" stroke="var(--line)"/>
     <text x="${W-R+8}" y="${F.y(v)+4}" fill="var(--muted)" font-size="11">${signed(v-1,0)}</text>`).join("");
  el("equity").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Track record after costs">
      ${ticks}${marks}
      ${e.benchmark ? `<path d="${line(e.benchmark)}" fill="none" stroke="var(--muted)"
        stroke-width="1.5" stroke-dasharray="4 3"/>` : ""}
      <path d="${line(e.strategy)}" fill="none" stroke="var(--accent)" stroke-width="2"/>
      <path d="${ddPath}" fill="var(--down)" opacity="0.16"/>
      <text x="${L}" y="${H-DD+6}" fill="var(--muted)" font-size="10">drawdown</text>
      <text x="${W-R+8}" y="${H-8}" fill="var(--muted)" font-size="11">${pct(ddLo,0)}</text>
      <line class="cursor" x1="0" y1="${T}" x2="0" y2="${H-DD}" stroke="var(--ink)"
            stroke-width="1" opacity="0"/>
      <rect class="hit" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B+20}" fill="transparent"/>
      <text x="${L}" y="${H-6}" fill="var(--muted)" font-size="11">${esc(e.dates[0])}</text>
      <text x="${W-R}" y="${H-6}" fill="var(--muted)" font-size="11" text-anchor="end">
        ${esc(e.dates[n-1])}</text>
      ${e.benchmark ? `<text x="${W/2}" y="${H-6}" fill="var(--muted)" font-size="11"
        text-anchor="middle">solid: portfolio &middot; dashed: ${esc(D.benchmark)}</text>` : ""}
    </svg>`;
  attachHover(el("equity"), W, L, R, n, i =>
    `<b>${esc(e.dates[i])}</b><br>portfolio ${signed(e.strategy[i]-1)}` +
    (e.benchmark ? `<br><span class="muted">${esc(D.benchmark)} ${signed(e.benchmark[i]-1)}</span>` : "") +
    `<br><span class="muted">drawdown ${pct(e.drawdown[i],1)}</span>`);
}

/* ---------- regime timeline ---------- */
function drawRegime() {
  const r = D.regime_history;
  if (!r) { el("regime-section").style.display = "none"; return; }
  if (r.headline) el("regime-head").textContent = r.headline;
  const W = 940, H = 130, L = 8, R = 58, T = 10, B = 26;
  const n = r.dates.length;
  const F = sparkFrame(W, H, L, R, T, B, 0, 1, n);
  const area = r.stress.map((v, i) => (i ? "L" : "M") + F.x(i).toFixed(1) + " " + F.y(v).toFixed(1)).join(" ")
    + ` L${F.x(n-1).toFixed(1)} ${F.y(0).toFixed(1)} L${F.x(0).toFixed(1)} ${F.y(0).toFixed(1)} Z`;
  const half = `<line x1="${L}" y1="${F.y(0.5)}" x2="${W-R}" y2="${F.y(0.5)}"
    stroke="var(--muted)" stroke-dasharray="3 3" opacity="0.5"/>`;
  el("regime").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Probability the market is in a stress regime">
      ${half}
      <path d="${area}" fill="var(--down)" opacity="0.18"/>
      <path d="${r.stress.map((v,i)=>(i?"L":"M")+F.x(i).toFixed(1)+" "+F.y(v).toFixed(1)).join(" ")}"
        fill="none" stroke="var(--down)" stroke-width="1.5"/>
      <text x="${W-R+8}" y="${F.y(1)+4}" fill="var(--muted)" font-size="11">stress</text>
      <text x="${W-R+8}" y="${F.y(0)+4}" fill="var(--muted)" font-size="11">calm</text>
      <line class="cursor" x1="0" y1="${T}" x2="0" y2="${H-B}" stroke="var(--ink)"
            stroke-width="1" opacity="0"/>
      <rect class="hit" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>
      <text x="${L}" y="${H-6}" fill="var(--muted)" font-size="11">${esc(r.dates[0])}</text>
      <text x="${W-R}" y="${H-6}" fill="var(--accent)" font-size="11" text-anchor="end">today</text>
    </svg>`;
  attachHover(el("regime"), W, L, R, n, i =>
    `<b>${esc(r.dates[i])}</b><br>chance of a stress regime ${pct(r.stress[i],0)}`);
}

/* ---------- diversification monitor ---------- */
function drawMonitor() {
  const m = D.monitor;
  if (!m) { el("monitor-section").style.display = "none"; return; }
  if (m.headline) el("monitor-head").textContent = m.headline;
  const W = 940, H = 110, L = 8, R = 58, T = 10, B = 26;
  const n = m.dates.length;
  const lo = Math.min(...m.values), hi = Math.max(...m.values);
  const F = sparkFrame(W, H, L, R, T, B, lo, hi, n);
  el("monitor").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="How much holdings move together">
      <line x1="${L}" y1="${F.y(hi)}" x2="${W-R}" y2="${F.y(hi)}" stroke="var(--line)"/>
      <line x1="${L}" y1="${F.y(lo)}" x2="${W-R}" y2="${F.y(lo)}" stroke="var(--line)"/>
      <text x="${W-R+8}" y="${F.y(hi)+4}" fill="var(--muted)" font-size="11">${hi.toFixed(2)}</text>
      <text x="${W-R+8}" y="${F.y(lo)+4}" fill="var(--muted)" font-size="11">${lo.toFixed(2)}</text>
      <path d="${m.values.map((v,i)=>(i?"L":"M")+F.x(i).toFixed(1)+" "+F.y(v).toFixed(1)).join(" ")}"
        fill="none" stroke="var(--accent)" stroke-width="1.5"/>
      <line class="cursor" x1="0" y1="${T}" x2="0" y2="${H-B}" stroke="var(--ink)"
            stroke-width="1" opacity="0"/>
      <rect class="hit" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>
      <text x="${L}" y="${H-6}" fill="var(--muted)" font-size="11">${esc(m.dates[0])}</text>
      <text x="${W-R}" y="${H-6}" fill="var(--muted)" font-size="11" text-anchor="end">today</text>
    </svg>`;
  attachHover(el("monitor"), W, L, R, n, i =>
    `<b>${esc(m.dates[i])}</b><br>average correlation between holdings ${m.values[i].toFixed(2)}`);
}

/* ---------- evidence panel ----------
   A Sharpe ratio means nothing to a reader who does not already know the scale,
   so the scale is stated next to the number rather than assumed. */
function sharpeVerdict(sr, lo, hi) {
  const band = sr < 0 ? "below zero: the risk taken was not rewarded"
    : sr < 0.5 ? "weak for an equity portfolio"
    : sr < 1.0 ? "respectable for an equity portfolio"
    : sr < 2.0 ? "strong"
    : "unusually strong, which is itself worth questioning";
  const crosses = lo !== undefined && lo !== null && hi !== undefined && hi !== null
                  && lo < 0 && hi > 0;
  return band + (crosses
    ? ", but the range crosses zero, so having no edge at all still fits this record"
    : ", and the whole range stays on one side of zero");
}

/* ---------- evidence panel ---------- */
function drawEvidence() {
  const e = D.evidence || {};
  if (!Object.keys(e).length) { el("evidence-section").style.display = "none"; return; }
  const items = [];
  if (e.sharpe !== undefined && e.sharpe !== null) {
    const lo = e.sharpe_low, hi = e.sharpe_high;
    const span = Math.max(Math.abs(lo), Math.abs(hi), 1);
    const toX = v => 50 + (v / (2 * span)) * 100;
    items.push(`<div class="stat" style="min-width:300px">
      <div class="k">Sharpe ratio, with its error bar${info("sharpe")}</div>
      <div class="num sm">${e.sharpe.toFixed(2)}
        <span class="muted" style="font-size:12px">(${lo.toFixed(2)} to ${hi.toFixed(2)})</span></div>
      <svg viewBox="0 0 100 18" style="width:200px;height:18px;margin-top:4px">
        <line x1="50" y1="2" x2="50" y2="16" stroke="var(--muted)" stroke-dasharray="2 2"/>
        <line x1="${toX(lo)}" y1="9" x2="${toX(hi)}" y2="9" stroke="var(--accent)" stroke-width="3"/>
        <circle cx="${toX(e.sharpe)}" cy="9" r="3" fill="var(--accent)"/>
      </svg>
      <div class="muted" style="font-size:11px;white-space:normal;max-width:300px">
        ${sharpeVerdict(e.sharpe, lo, hi)}<br>zero means no edge at all</div></div>`);
  }
  const chip = (k, v, note) => `<div class="stat"><div class="k">${k}</div>
    <div class="num sm">${v}</div>
    ${note ? `<div class="muted" style="font-size:11px">${esc(note)}</div>` : ""}</div>`;
  if (e.deflated_sharpe !== undefined && e.deflated_sharpe !== null)
    items.push(chip("After allowing for trial and error" + info("dsr"),
                    e.deflated_sharpe.toFixed(2),
                    e.deflated_sharpe < 0.5 ? "below 0.5: no evidence of skill" : "above 0.5"));
  if (e.pbo !== undefined && e.pbo !== null)
    items.push(chip("Chance the choice was luck" + info("pbo"), pct(e.pbo, 0),
                    e.pbo > 0.5 ? "worse than choosing at random" : "selection carried over"));
  if (e.min_track_record_years !== undefined && e.min_track_record_years !== null)
    items.push(chip("Track record needed" + info("mintrl"),
                    e.min_track_record_years.toFixed(1) + " yrs",
                    e.track_record_years ? "have " + e.track_record_years.toFixed(1) : null));
  if (e.var_zone)
    items.push(`<div class="stat"><div class="k">Risk model${info("varmodel")}</div>
      <div class="num sm"><span class="light ${e.var_zone === "green" ? "ok" :
        (e.var_zone === "yellow" ? "near" : "bad")}"></span>${esc(e.var_zone)}</div>
      ${e.var_note ? `<div class="muted" style="font-size:11px">${esc(e.var_note)}</div>` : ""}</div>`);
  el("evidence").innerHTML = `<div class="row">${items.join("")}</div>`;
}

drawHorizons(); if (D.horizons.length) { drawOutcome(); drawFan(); }
drawScenarios(); drawLimits(); drawRisk(); drawAlloc();
drawEquity(); drawRegime(); drawMonitor(); drawEvidence();
drawRecs(); drawMetrics(); drawFooter();
document.querySelectorAll("[data-info]").forEach(n => n.insertAdjacentHTML("beforeend",
                                                                          info(n.dataset.info)));
window.addEventListener("resize", () => { if (D.horizons.length) drawFan(); });
</script>
</body>
</html>
"""
