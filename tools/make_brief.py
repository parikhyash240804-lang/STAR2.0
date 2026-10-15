"""
Run the whole chain and write the HTML brief.

This is the end-to-end path the CLI will take in P9: data, estimates, portfolio,
risk, forecast, stress, backtest, guidance, page. Everything the brief shows is
produced here and nothing is invented by the template.

Run: python tools/make_brief.py [--out runs/brief.html] [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from star.backtest.costs import rebalance_cost                              # noqa: E402
from star.backtest.engine import BacktestSettings, run_backtest             # noqa: E402
from star.backtest.stats import (deflated_sharpe_ratio,                     # noqa: E402
                                 minimum_track_record_length,
                                 sharpe_difference_test)
from star.config.schema import Mandate                                      # noqa: E402
from star.data.benchmark import build_benchmark                            # noqa: E402
from star.data.fixture import FixtureProvider                               # noqa: E402
from star.data.quality import assess                                        # noqa: E402
from star.data.view import DataView                                         # noqa: E402
from star.estimators.covariance import estimate_covariance                  # noqa: E402
from star.estimators.har import MEASURED as VOL_SCORECARD                    # noqa: E402
from star.estimators.equilibrium import equilibrium_prior, reference_weights  # noqa: E402
from star.estimators.regime import combine, fit_regimes, vix_bands          # noqa: E402
from star.forecast.engine import DriftAssumption, simulate                  # noqa: E402
from star.forecast.paths import BootstrapGenerator                          # noqa: E402
from star.optimize.constraints import build_constraints                     # noqa: E402
from star.optimize.mean_variance import OptimiserSettings, optimise         # noqa: E402
from star.optimize.stability import perturbation_sensitivity                # noqa: E402
from star.report.brief import build_brief_data, write_brief                 # noqa: E402
from star.report.guidance import BriefInputs, evaluate                      # noqa: E402
from star.risk.backtests import validate_var                                # noqa: E402
from star.risk.metrics import volatility_contributions                      # noqa: E402
from star.risk.monitors import correlation_monitor                          # noqa: E402
from star.backtest.stats import sharpe_confidence_interval                  # noqa: E402
from star.stress.historical import EPISODE_BY_KEY, replay                   # noqa: E402
from star.stress.hypothetical import sector_shock                           # noqa: E402

warnings.filterwarnings("ignore")
SECTORS = ["Financials", "IT", "Energy", "Pharma", "Industrials"]


def _drift_sentence(drift) -> str:
    """The drift assumption in words. `describe()` is written for the workbook."""
    annual = drift.annual_simple()
    if annual is None:
        return ("While simulating the future we let history set the average return, and assumed "
                "no skill on top of it.")
    if drift.kind == "rf_plus_erp":
        return (f"While simulating the future we assumed the portfolio grows {annual:.1%} a year "
                f"before any skill: {drift.risk_free:.1%} from safe government rates, plus "
                f"{drift.shrinkage:.0%} of the {drift.equity_risk_premium:.1%} that shares have "
                f"historically paid above them. We assumed no skill on top of that.")
    return (f"While simulating the future we assumed the portfolio grows {annual:.1%} a year, "
            f"with no skill assumed on top.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/brief.html")
    ap.add_argument("--quick", action="store_true", help="skip the walk-forward backtest")
    args = ap.parse_args()

    print("loading data...")
    p = FixtureProvider.default()
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept
    as_of = panel.sessions.max()
    view = DataView.build(panel, as_of)
    nav = 1.0e7
    vix = p.benchmark("^INDIAVIX")
    nifty = p.benchmark("^NSEI")
    # the dividend handicap is measurable from a dividend-paying index fund, so measure
    # it. It used to be a literal typed into this file while the brief called it
    # "measured against a total-return proxy" - precisely the kind of unearned claim
    # this rebuild exists to remove.
    try:
        tri_proxy = p.benchmark("NIFTYBEES.NS")
    except Exception:
        tri_proxy = None
    bench = build_benchmark(nifty, tri_proxy, as_of=as_of)
    sectors = pd.Series({t: SECTORS[i % len(SECTORS)] for i, t in enumerate(kept)})

    print("estimating risk...")
    cov = estimate_covariance(view.ohlc(), tickers=kept, as_of=as_of, implied=vix)
    w_ref, notes = reference_weights(kept, allow_market_cap=False)
    prior, delta = equilibrium_prior(cov, w_ref, degradations=notes)

    print("building the portfolio...")
    mandate = Mandate(max_stock_weight=0.10, max_sector_weight=0.30)
    current = pd.Series(0.0, index=kept)
    current.iloc[:12] = 1.0 / 12                       # a plausible existing book
    cs = build_constraints(kept, mandate, sectors=sectors, prev_weights=current)
    port = optimise(prior, cov, cs, risk_aversion=delta,
                    settings=OptimiserSettings(turnover_penalty=0.005, no_trade_band=0.005))
    weights = port.weights
    costs = rebalance_cost(current, weights, nav_inr=nav)

    print("measuring risk...")
    contributions = volatility_contributions(weights, cov)
    close = panel.close[kept]
    log_ret = np.log(close / close.shift(1)).dropna()
    port_daily = (log_ret @ weights).dropna()
    bench_daily = np.log(nifty / nifty.shift(1)).reindex(port_daily.index).fillna(0.0)

    rolling_sd = port_daily.rolling(250).std().bfill()
    z = (port_daily / rolling_sd).dropna()
    q99 = -np.percentile(z, 1)
    var_bt = validate_var(port_daily, rolling_sd * q99, 0.99)

    print("simulating the forward distribution...")
    ts = simulate(BootstrapGenerator(portfolio=port_daily.to_numpy(),
                                     benchmark=bench_daily.to_numpy(), mean_block=10),
                  horizon=252, n_paths=20_000, seed=7, drift=DriftAssumption())

    print("stress testing...")
    scenarios = []
    for key in ("covid_2020", "rates_2022", "election_2024"):
        try:
            res = replay(key, panel.close, weights, benchmark=nifty)
        except Exception:
            continue
        ph = res.per_holding
        held = ph[ph["weight"] > 1e-9]
        # the headline divides by the weight that had data, so put the rows on the same
        # base - otherwise the column cannot add up to the number above it
        total_contrib = float(held["contribution"].sum())
        scale = (float(res.portfolio_return) / total_contrib) if abs(total_contrib) > 1e-12 else 1.0
        worst = res.worst(4)
        rows = [{"ticker": t, "impact": float(r["contribution"]) * scale,
                 "move": float(r["return"]), "weight": float(r["weight"])}
                for t, r in worst.iterrows()]
        scenarios.append({
            "id": key, "name": EPISODE_BY_KEY[key].name,
            "impact": float(res.portfolio_return), "value": float(res.portfolio_return * nav),
            "coverage": float(res.coverage),
            "note": EPISODE_BY_KEY[key].what_happened,
            "top": rows, "held": int(len(held)),
            "other": float(res.portfolio_return) - sum(r["impact"] for r in rows),
        })
    for sector, move in (("Financials", -0.20), ("IT", -0.15)):
        try:
            sc = sector_shock(weights, cov, sectors, sector, move)
        except Exception:
            continue
        worst = sc.contributions.nsmallest(4)
        rows = [{"ticker": t, "impact": float(v), "move": float(sc.returns.get(t, float("nan"))),
                 "weight": float(weights.get(t, 0.0))} for t, v in worst.items()]
        scenarios.append({
            "id": f"{sector}_{int(move*100)}", "name": f"{sector} {move:+.0%}",
            "impact": float(sc.portfolio_return), "value": float(sc.portfolio_return * nav),
            "coverage": None, "note": sc.headline,
            "top": rows, "held": int((weights > 1e-9).sum()),
            "other": float(sc.portfolio_return) - sum(r["impact"] for r in rows),
        })

    print("reading the regime...")
    regime_fit = fit_regimes(port_daily)
    opinion = combine(regime_fit, vix_bands(vix, as_of=as_of))

    print("checking stability...")
    stability = perturbation_sensitivity(prior, cov, cs, risk_aversion=delta, draws=12,
                                         base=port)

    print("watching diversification...")
    monitor = correlation_monitor(log_ret[kept], window=60)

    sharpe_diff = dsr = mintrl = track_years = None
    equity = rebalances = None
    evidence = {}
    if not args.quick:
        print("running the walk-forward backtest (slow)...")
        kept_set = set(kept)

        def strategy(v, d, prev, universe):
            names = [t for t in universe if t in kept_set]
            c = estimate_covariance(v.ohlc(), tickers=names, as_of=d, implied=vix)
            wr, nts = reference_weights(names, allow_market_cap=False)
            pr, dl = equilibrium_prior(c, wr, degradations=nts)
            k = build_constraints(names, mandate,
                                  sectors=sectors.reindex(names).fillna("Other"),
                                  prev_weights=prev if len(prev) else None)
            return optimise(pr, c, k, risk_aversion=dl,
                            settings=OptimiserSettings(turnover_penalty=0.005,
                                                       no_trade_band=0.005)).weights

        def equal_weight(v, d, prev, universe):
            names = [t for t in universe if t in kept_set]
            return pd.Series(1.0 / len(names), index=names)

        settings = BacktestSettings(warmup_sessions=504, rebalance="Q", nav_inr=nav)
        bt = run_backtest(panel, strategy, settings=settings)
        ew = run_backtest(panel, equal_weight, settings=settings)
        sharpe_diff = sharpe_difference_test(bt.returns_net, ew.returns_net)
        dsr = deflated_sharpe_ratio(bt.returns_net, n_trials=10, trial_sharpe_variance=0.05)
        mintrl = minimum_track_record_length(bt.returns_net) / 252
        track_years = len(bt.returns_net) / 252
        bench_bt = np.exp(np.log1p(bench_daily).cumsum()).reindex(bt.returns_net.index).ffill()
        equity = bt.equity_curve()
        equity["benchmark"] = bench_bt / float(bench_bt.iloc[0])
        rebalances = list(bt.targets.index)
        sr, lo, hi = sharpe_confidence_interval(bt.returns_net)
        evidence = {"sharpe": sr, "sharpe_low": lo, "sharpe_high": hi,
                    "deflated_sharpe": dsr, "min_track_record_years": mintrl,
                    "track_record_years": track_years}

    print("writing the brief...")
    top = weights.nlargest(1)
    sector_totals = weights.groupby(sectors).sum()
    holdings = int((weights > 1e-6).sum())
    limits = [
        {"name": "Largest position", "value": float(top.iloc[0]), "limit": mandate.max_stock_weight,
         "headroom": float(mandate.max_stock_weight - top.iloc[0]),
         "status": "breach" if top.iloc[0] > mandate.max_stock_weight + 1e-9 else
                   ("near" if top.iloc[0] > mandate.max_stock_weight * 0.9 else "ok")},
        {"name": f"Largest sector ({sector_totals.idxmax()})", "value": float(sector_totals.max()),
         "limit": mandate.max_sector_weight,
         "headroom": float(mandate.max_sector_weight - sector_totals.max()),
         "status": "breach" if sector_totals.max() > mandate.max_sector_weight + 1e-9 else
                   ("near" if sector_totals.max() > mandate.max_sector_weight * 0.9 else "ok")},
        {"name": "Holdings", "value": holdings, "limit": mandate.max_holdings,
         "headroom": mandate.max_holdings - holdings, "status": "ok", "format": "number"},
        {"name": "Turnover this rebalance", "value": float(port.turnover()),
         "limit": mandate.max_annual_turnover / 4,
         "headroom": float(mandate.max_annual_turnover / 4 - port.turnover()),
         "status": "ok" if port.turnover() <= mandate.max_annual_turnover / 4 else "breach"},
    ]

    alloc = pd.DataFrame({"current": current, "proposed": weights})
    alloc["change"] = alloc["proposed"] - alloc["current"]

    inputs = BriefInputs(
        as_of=as_of, mandate=mandate, nav_inr=nav, weights=weights, previous_weights=current,
        violations=cs.violations(weights), turnover=port.turnover(),
        cost_fraction=costs.cost_fraction, risk_contributions=contributions,
        portfolio_volatility=port.volatility(cov), var_backtest=var_bt, stability=stability,
        regime=opinion, sharpe_vs_equal_weight=sharpe_diff, deflated_sharpe=dsr,
        min_track_record_years=mintrl, track_record_years=track_years,
        benchmark_gap=bench.dividend_accrual_annual,
        benchmark_gap_measured=bench.calibrated,
        traded_value_inr=costs.traded_value, signals_available=False)
    recs = evaluate(inputs)

    # what the confidence badge is actually resting on, measured rather than asserted
    years = len(port_daily) / 252
    checks = [
        {"name": "History behind it",
         "value": f"{len(port_daily):,} trading sessions, about {years:.1f} years, "
                  f"across {len(kept)} holdings",
         "verdict": "pass" if years >= 5 else "fair"},
        {"name": "Volatility model",
         "value": (f"the range-based model chosen by an out-of-sample bake-off over "
                   f"{VOL_SCORECARD['forecasts']:,} forecasts; it explains "
                   f"{VOL_SCORECARD['r2']:.0%} of the variation in how much a stock moves, "
                   f"against {VOL_SCORECARD['baseline_r2']:.0%} for the usual method. Noise in "
                   f"the target caps any model at about {VOL_SCORECARD['attainable_r2']:.0%}, so "
                   f"this reaches roughly "
                   f"{VOL_SCORECARD['r2'] / VOL_SCORECARD['attainable_r2']:.0%} of what is "
                   f"reachable - good for sizing risk, useless for timing it"),
         "verdict": "fair"},
        {"name": "How the futures were drawn",
         "value": "20,000 simulated futures, each stitched from real blocks of past sessions, so "
                  "crashes stay clustered instead of being averaged away",
         "verdict": "pass"},
    ]

    evidence["var_zone"] = var_bt.basel_zone
    evidence["var_note"] = (f"{var_bt.breaches} breaches where "
                            f"{var_bt.expected_breaches:.0f} were expected")

    data = build_brief_data(
        as_of=as_of, nav_inr=nav, recommendations=recs, term_structure=ts,
        scenarios=scenarios, limits=limits, risk_contributors=contributions,
        allocation=alloc, risk_model_passed=var_bt.passed,
        regime_settled=opinion.hmm_says != "mixed",
        equity=equity, rebalance_dates=rebalances,
        regime_history=regime_fit.filtered["stress"], regime_headline=opinion.headline,
        monitor=monitor.series, monitor_headline=monitor.headline,
        evidence=evidence, confidence_checks=checks,
        cost_inr=costs.total, cost_fraction=costs.cost_fraction,
        annual_volatility=port.volatility(cov),
        # Three dense machine-readable lines became one sentence about the only
        # assumption in them that changes a decision: the return we assumed. The full
        # provenance of every input is in the workbook, per-field.
        provenance=[_drift_sentence(ts.drift),
                    (f"We compare against the NIFTY 50 plus the dividends its companies pay. "
                     f"That dividend top-up is {bench.dividend_accrual_annual:.2%} a year, "
                     + (f"measured from NIFTYBEES over "
                        f"{bench.calibration_start.date()} to {bench.calibration_end.date()}."
                        if bench.calibrated else
                        "assumed, because the index-fund data needed to measure it was "
                        "unavailable.")),
                    "How each number was produced, field by field, is recorded in the "
                    "workbook."],
        # The two notes that used to sit here repeated REBAL-01 and VALID-00 word for
        # word, and the risk-model one dropped the caveat, so the footer sounded more
        # confident than the page. Only genuine warnings survive.
        notes=list(ts.warnings),
        )
    out = write_brief(args.out, data)
    size = out.stat().st_size / 1024
    print(f"\n{out}  ({size:.0f} KB)")
    print(f"headline: {data['headline']}")
    for g in data["recommendations"]:
        print(f"\n  {g['question']}")
        for r in g["items"]:
            print(f"    [{r['severity']:>16}] {r['headline']}")


if __name__ == "__main__":
    main()
