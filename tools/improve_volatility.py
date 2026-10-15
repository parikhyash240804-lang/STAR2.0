"""
Can the volatility forecast be made better than R^2 = 0.27?

Six candidates, all scored walk-forward against the same target (the realised
21-day close-to-close volatility that followed), pooled across stocks:

1. EWMA of squared closes - the incumbent baseline.
2. Yang-Zhang EWMA - what the covariance module ships.
3. HAR (Corsi 2009) - daily, weekly and monthly volatility components in one
   regression. The standard workhorse of the volatility-forecasting literature.
4. HAR on range-based inputs - the same, but fed Parkinson daily volatility,
   which uses the high and low and so measures each session far more precisely
   than a single close-to-close number.
5. HAR + India VIX - adding the market's own forward-looking estimate, which is
   information no backward-looking model contains.
6. Combination - the average of the models, because forecast combination is one
   of the most reliable results in forecasting.

Everything is modelled in **logs**: volatility is right-skewed and strictly
positive, so a regression on levels is fighting its own error distribution.

Regression coefficients are fitted on an expanding window of data strictly
before each forecast date, so no model is graded on anything it was fitted on.

A **noise ceiling** is estimated too: the target is itself a 21-observation
sample estimate, and no forecast can explain the part of it that is measurement
error. Comparing against that ceiling is the difference between "our model is
weak" and "the target is noisy".

Run: python tools/improve_volatility.py
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

from star.data.fixture import FixtureProvider          # noqa: E402
from star.data.quality import assess                   # noqa: E402

warnings.filterwarnings("ignore")
PPY = 252.0
HORIZON = 21
STEP = 5                      # weekly forecasts, for enough observations to say anything


def parkinson_daily(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Per-session variance from the high-low range - far more precise than one close."""
    rng = np.log(high / low)
    return rng.pow(2) / (4.0 * np.log(2.0))


def out_of_sample_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """1 - SSE/SST against the mean of the realised values. Can go negative."""
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    p = FixtureProvider.default()
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept
    close, high, low = panel.close[kept], panel.high[kept], panel.low[kept]
    sessions = panel.sessions
    lr = np.log(close / close.shift(1))
    park = parkinson_daily(high, low)

    try:
        vix = p.benchmark("^INDIAVIX").reindex(sessions).ffill()
    except Exception:
        vix = None

    ewma_var = lr.pow(2).ewm(halflife=60).mean()
    park_d = park.rolling(5).mean()
    park_w = park.rolling(22).mean()
    park_m = park.rolling(66).mean()
    close_d = lr.pow(2).rolling(5).mean()
    close_w = lr.pow(2).rolling(22).mean()
    close_m = lr.pow(2).rolling(66).mean()

    warmup = 504
    steps = list(range(warmup, len(sessions) - HORIZON, STEP))
    rows = []
    for i in steps:
        as_of = sessions[i]
        fut = lr.iloc[i + 1: i + 1 + HORIZON]
        realised = fut.std(ddof=1) * np.sqrt(PPY)
        half1 = fut.iloc[:10].std(ddof=1) * np.sqrt(PPY)
        half2 = fut.iloc[11:].std(ddof=1) * np.sqrt(PPY)
        for t in kept:
            v = realised.get(t, np.nan)
            if not np.isfinite(v) or v <= 0:
                continue
            feats = {
                "date": as_of, "ticker": t, "y": np.log(v),
                "ewma": np.log(max(np.sqrt(ewma_var[t].iloc[i] * PPY), 1e-6)),
                "cd": np.log(max(np.sqrt(close_d[t].iloc[i] * PPY), 1e-6)),
                "cw": np.log(max(np.sqrt(close_w[t].iloc[i] * PPY), 1e-6)),
                "cm": np.log(max(np.sqrt(close_m[t].iloc[i] * PPY), 1e-6)),
                "pd_": np.log(max(np.sqrt(park_d[t].iloc[i] * PPY), 1e-6)),
                "pw": np.log(max(np.sqrt(park_w[t].iloc[i] * PPY), 1e-6)),
                "pm": np.log(max(np.sqrt(park_m[t].iloc[i] * PPY), 1e-6)),
                "half1": np.log(half1.get(t, np.nan)) if half1.get(t, 0) > 0 else np.nan,
                "half2": np.log(half2.get(t, np.nan)) if half2.get(t, 0) > 0 else np.nan,
            }
            if vix is not None and np.isfinite(vix.iloc[i]):
                feats["vix"] = np.log(float(vix.iloc[i]) / 100.0)
            rows.append(feats)

    df = pd.DataFrame(rows).dropna(subset=["y", "ewma", "cd", "cw", "cm", "pd_", "pw", "pm"])
    df["xs_mean"] = df.groupby("date")["ewma"].transform("mean")
    has_vix = "vix" in df.columns and df["vix"].notna().mean() > 0.9
    print(f"{len(df):,} forecasts | {df['date'].nunique()} weekly dates | {df['ticker'].nunique()} names"
          f" | India VIX {'available' if has_vix else 'MISSING'}\n")

    models = {
        "EWMA close (baseline)": ["ewma"],
        "Yang-Zhang-style range": ["pw"],
        "HAR, close-to-close": ["cd", "cw", "cm"],
        "HAR, range-based": ["pd_", "pw", "pm"],
        "HAR range + market level": ["pd_", "pw", "pm", "xs_mean"],
    }
    if has_vix:
        models["HAR range + market + VIX"] = ["pd_", "pw", "pm", "xs_mean", "vix"]

    dates = sorted(df["date"].unique())
    split = dates[len(dates) // 3]                        # first third trains the earliest forecast
    preds = {}
    for name, cols in models.items():
        out = []
        for d in dates:
            if d <= split:
                continue
            train = df[df["date"] < d]
            test = df[df["date"] == d]
            if len(train) < 500 or test.empty:
                continue
            X = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in cols])
            beta, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
            Xt = np.column_stack([np.ones(len(test))] + [test[c].to_numpy() for c in cols])
            out.append(pd.DataFrame({"date": d, "ticker": test["ticker"].to_numpy(),
                                     "y": test["y"].to_numpy(), "yhat": Xt @ beta}))
        preds[name] = pd.concat(out, ignore_index=True)

    ref = preds["EWMA close (baseline)"]
    combo = ref[["date", "ticker", "y"]].copy()
    combo["yhat"] = np.mean([preds[m]["yhat"].to_numpy() for m in preds], axis=0)
    preds["Combination (average)"] = combo

    print(f"{'model':<28}{'R2 (log)':>10}{'R2 (level)':>12}{'bias':>8}{'gain':>8}")
    base_r2 = None
    for name, pr in preds.items():
        r2_log = out_of_sample_r2(pr["y"].to_numpy(), pr["yhat"].to_numpy())
        lvl_y, lvl_hat = np.exp(pr["y"].to_numpy()), np.exp(pr["yhat"].to_numpy())
        r2_lvl = out_of_sample_r2(lvl_y, lvl_hat)
        bias = float(np.mean(lvl_y / lvl_hat))
        if base_r2 is None:
            base_r2 = r2_log
        gain = (r2_log - base_r2) / abs(base_r2) if base_r2 else np.nan
        print(f"{name:<28}{r2_log:>10.3f}{r2_lvl:>12.3f}{bias:>8.2f}{gain:>+8.0%}")

    ceiling = df.dropna(subset=["half1", "half2"])
    # Two halves of the same window measure the same underlying volatility with
    # independent noise, so their correlation is the reliability of a 10-observation
    # estimate. Correcting for that noise and re-applying it to the 21-day target
    # gives the ceiling any forecast could reach - the correction for attenuation.
    r = float(np.corrcoef(ceiling["half1"], ceiling["half2"])[0, 1])
    noise_10 = 1.0 / (2 * (10 - 1))                  # variance of a log sample sd, n = 10
    noise_21 = 1.0 / (2 * (HORIZON - 1))
    signal = r * noise_10 / max(1 - r, 1e-9)
    ceiling_r2 = signal / (signal + noise_21)
    best = max(out_of_sample_r2(pr["y"].to_numpy(), pr["yhat"].to_numpy()) for pr in preds.values())
    print(f"\n   Two halves of the same window correlate {r:.2f}. Correcting for the noise in")
    print(f"   a 10-day estimate, only about {ceiling_r2:.0%} of the 21-day target is predictable")
    print(f"   at all - the rest is measurement error in the target itself.")
    print(f"   Best model reaches {best:.3f}: about {best / ceiling_r2:.0%} of what is attainable.")


if __name__ == "__main__":
    main()
