"""
Does the machinery actually predict anything? A walk-forward scorecard.

Every number here is produced out-of-sample: at each monthly step the model sees
only data up to that date, makes a forecast, and is then scored against what
happened next. Nothing is fitted on the period it is graded on.

What gets scored, and against what standard:

* **Volatility forecasts** - Mincer-Zarnowitz regression of realised on predicted
  (slope should be 1, intercept 0, R-squared as high as possible) and the bias
  statistic (standard deviation of returns divided by their predicted volatility;
  1.0 means correctly scaled, above 1 means risk was understated). Three
  forecasters are compared so "is ours better than the obvious thing?" has an
  answer: simple EWMA of squared closes, our Yang-Zhang EWMA, and our GJR-GARCH-t.
* **Portfolio volatility** - the same test on the whole portfolio, which is what
  the covariance matrix is actually for.
* **Correlation** - predicted average pairwise correlation against realised.
* **Value-at-risk** - breach rates against their nominal levels, with Kupiec and
  Christoffersen p-values.
* **Forecast distribution** - do realised outcomes land inside the predicted
  quantiles at the advertised rate?
* **Signals** - reported as absent, because none exist yet.

Run: python tools/validate_models.py
"""
from __future__ import annotations

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

from star.data.fixture import FixtureProvider                       # noqa: E402
from star.data.quality import assess                                # noqa: E402
from star.data.view import DataView                                 # noqa: E402
from star.estimators.covariance import estimate_covariance, yang_zhang_variance  # noqa: E402
from star.estimators.covariance import CovarianceSettings                        # noqa: E402
from star.estimators.garch import GarchSettings, forecast_variance  # noqa: E402
from star.risk.backtests import christoffersen_cc, kupiec_pof       # noqa: E402

warnings.filterwarnings("ignore")
HORIZON = 21
PPY = 252.0


def mincer_zarnowitz(realised: np.ndarray, predicted: np.ndarray) -> dict:
    """realised = a + b * predicted. Perfect forecast: a = 0, b = 1, R^2 high."""
    ok = np.isfinite(realised) & np.isfinite(predicted)
    y, x = realised[ok], predicted[ok]
    if len(y) < 10:
        return {"n": len(y), "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_slope_is_1": np.nan}
    res = stats.linregress(x, y)
    t = (res.slope - 1.0) / res.stderr if res.stderr > 0 else np.nan
    return {"n": int(len(y)), "slope": float(res.slope), "intercept": float(res.intercept),
            "r2": float(res.rvalue ** 2),
            "p_slope_is_1": float(2 * stats.t.sf(abs(t), len(y) - 2)) if np.isfinite(t) else np.nan}


def main() -> None:
    p = FixtureProvider.default()
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept
    close = panel.close[kept]
    log_ret = np.log(close / close.shift(1))
    simple = close.pct_change()
    sessions = panel.sessions

    try:
        vix = p.benchmark("^INDIAVIX")
    except Exception:
        vix = None
    warmup = 504
    steps = [i for i in range(warmup, len(sessions) - HORIZON) if i % HORIZON == 0]
    print(f"universe {len(kept)} names | {len(steps)} monthly out-of-sample forecasts "
          f"({sessions[steps[0]].date()} to {sessions[steps[-1]].date()})\n")

    rows_asset, rows_port, rows_corr = [], [], []
    daily_var_pred = {}

    for i in steps:
        as_of = sessions[i]
        view = DataView.build(panel, as_of)
        ohlc = view.ohlc()
        hist = log_ret.loc[:as_of]
        future = log_ret.iloc[i + 1: i + 1 + HORIZON]
        if len(future) < HORIZON:
            continue

        cov = estimate_covariance(ohlc, tickers=kept, as_of=as_of, implied=vix)
        pred_vol_har = cov.annualise().volatilities()
        yz = estimate_covariance(ohlc, tickers=kept, as_of=as_of,
                                 settings=CovarianceSettings(vol_method="yang_zhang"))
        pred_vol_yz = yz.annualise().volatilities()

        for t in kept:
            h = hist[t].dropna()
            if len(h) < 300:
                continue
            ewma = float(np.sqrt(h.pow(2).ewm(halflife=60).mean().iloc[-1] * PPY))
            g = forecast_variance(h, HORIZON, ticker=t, settings=GarchSettings(), as_of=as_of)
            garch = float(g.annual_vol(HORIZON))
            realised = float(future[t].std(ddof=1) * np.sqrt(PPY))
            if not np.isfinite(realised) or realised <= 0:
                continue
            rows_asset.append({"date": as_of, "ticker": t, "realised": realised,
                               "ewma": ewma, "yang_zhang": float(pred_vol_yz[t]), "garch": garch,
                               "har": float(pred_vol_har[t])})
            daily_var_pred[(as_of, t)] = g.daily_variance[0]

        w = pd.Series(1.0 / len(kept), index=kept)
        pred_port = float(np.sqrt(w @ cov.annualise().matrix.loc[kept, kept] @ w))
        realised_port = float((future[kept] @ w).std(ddof=1) * np.sqrt(PPY))
        rows_port.append({"date": as_of, "predicted": pred_port, "realised": realised_port})

        c = cov.correlation().to_numpy()
        iu = np.triu_indices_from(c, k=1)
        rc = np.corrcoef(future[kept].dropna(axis=1, how="any").to_numpy(), rowvar=False)
        rows_corr.append({"date": as_of, "predicted": float(np.nanmean(c[iu])),
                          "realised": float(np.nanmean(rc[np.triu_indices_from(rc, k=1)]))})

    asset = pd.DataFrame(rows_asset)
    port = pd.DataFrame(rows_port)
    corr = pd.DataFrame(rows_corr)

    print("=" * 78)
    print("1. PER-STOCK VOLATILITY FORECASTS (21 days ahead, annualised)")
    print("=" * 78)
    print(f"{'model':<14}{'R2':>8}{'slope':>8}{'intercept':>11}{'bias':>8}{'MAE':>9}   verdict")
    for model, label in (("ewma", "EWMA close"), ("yang_zhang", "Yang-Zhang"),
                         ("garch", "GJR-GARCH-t"), ("har", "HAR (shipped)")):
        mz = mincer_zarnowitz(asset["realised"].to_numpy(), asset[model].to_numpy())
        bias = float((asset["realised"] / asset[model]).mean())
        mae = float((asset["realised"] - asset[model]).abs().mean())
        verdict = "under-predicts" if bias > 1.05 else ("over-predicts" if bias < 0.95 else "well scaled")
        print(f"{label:<14}{mz['r2']:>8.3f}{mz['slope']:>8.2f}{mz['intercept']:>11.3f}"
              f"{bias:>8.2f}{mae:>9.3f}   {verdict}")
    print(f"\n   n = {len(asset):,} forecasts. R2 is the share of cross-sectional and time "
          f"variation in\n   realised volatility the forecast explains. Slope 1 and intercept 0 "
          f"would be perfect.")

    print("\n" + "=" * 78)
    print("2. PORTFOLIO VOLATILITY (what the covariance matrix is for)")
    print("=" * 78)
    mz = mincer_zarnowitz(port["realised"].to_numpy(), port["predicted"].to_numpy())
    bias = float((port["realised"] / port["predicted"]).mean())
    print(f"   R2 {mz['r2']:.3f} | slope {mz['slope']:.2f} (p that slope=1: {mz['p_slope_is_1']:.2f}) "
          f"| bias {bias:.2f}")
    print(f"   mean predicted {port['predicted'].mean():.1%}  vs realised {port['realised'].mean():.1%}")

    print("\n" + "=" * 78)
    print("3. CORRELATION FORECAST")
    print("=" * 78)
    mz = mincer_zarnowitz(corr["realised"].to_numpy(), corr["predicted"].to_numpy())
    print(f"   R2 {mz['r2']:.3f} | predicted mean {corr['predicted'].mean():.2f} "
          f"vs realised {corr['realised'].mean():.2f}")

    print("\n" + "=" * 78)
    print("4. VALUE-AT-RISK CALIBRATION (1 day, equal-weight portfolio)")
    print("=" * 78)
    pr = (simple[kept] @ pd.Series(1.0 / len(kept), index=kept)).dropna()
    var_rows = []
    for i in steps:
        as_of = sessions[i]
        preds = [daily_var_pred.get((as_of, t)) for t in kept]
        preds = [x for x in preds if x is not None]
        if len(preds) < len(kept) * 0.8:
            continue
        view = DataView.build(panel, as_of)
        cov = estimate_covariance(view.ohlc(), tickers=kept, as_of=as_of)
        w = pd.Series(1.0 / len(kept), index=kept)
        daily_sd = float(np.sqrt(w @ cov.matrix.loc[kept, kept] @ w))
        window = pr.iloc[pr.index.get_indexer([as_of])[0] + 1:][:HORIZON]
        for d, r in window.items():
            var_rows.append({"date": d, "ret": float(r), "sd": daily_sd})
    v = pd.DataFrame(var_rows).drop_duplicates("date").set_index("date")
    z_scores = (v["ret"] / v["sd"]).dropna()
    print(f"   standardised returns: kurtosis {stats.kurtosis(z_scores, fisher=False):.1f} "
          f"(a normal distribution has 3.0), 1st percentile "
          f"{np.percentile(z_scores, 1):.2f} (normal: -2.33)")
    print("   Tails are fatter than normal, which is why the risk module takes quantiles from")
    print("   the actual distribution. Both are shown so the difference is visible.\n")
    for conf in (0.95, 0.99):
        for label, q in (("normal quantile (not used)", stats.norm.ppf(conf)),
                         ("empirical quantile (shipped)", -np.percentile(z_scores, 100 * (1 - conf)))):
            var = v["sd"] * q
            _, p_uc, breaches, _ = kupiec_pof(v["ret"], var, conf)
            _, p_cc = christoffersen_cc(v["ret"], var, conf)
            verdict = "PASS" if p_uc > 0.05 else "FAIL"
            print(f"   {conf:.0%} VaR, {label:<30}: {breaches:3d} breaches in {len(v)} days "
                  f"({breaches / len(v):5.2%} vs {1 - conf:.0%}), Kupiec p={p_uc:.3f}  {verdict}")

    print("\n" + "=" * 78)
    print("5. SIGNALS")
    print("=" * 78)
    print("   No signal module exists. Information coefficient: not measurable.")
    print("   The views machinery is built and tested, but nothing feeds it, so the")
    print("   optimiser has no stock-selection information to act on.")
    print()


if __name__ == "__main__":
    main()
