# Does it actually predict anything? — out-of-sample scorecard

Run `python tools/validate_models.py`. Every figure below is produced
walk-forward: at each monthly step the model sees only data up to that date,
makes a forecast, and is scored against what happened next. Nothing is fitted on
the period it is graded on.

**Universe:** 37 NSE names · **35 monthly forecasts** · 22 Sep 2023 → 12 Aug 2026.

---

## The bug this exercise found

Before the fix, the covariance model predicted **6.2%** portfolio volatility
against **13.9%** realised — a factor of 2.25. Average predicted correlation was
**0.01** against a sample value of **0.21**.

Cause: correlations are estimated from returns standardised by each asset's own
lagged EWMA volatility. At the start of a window that EWMA has almost no data
behind it, so a stock whose first observed move was near zero produced a
standardised return of **137**. Ledoit-Wolf shrinkage responds to the estimated
variance of the sample covariance, so that single observation drove the shrinkage
intensity from **0.03 to 0.90** — shrinking essentially all correlation away and
making the portfolio look twice as diversified as it was.

Fix: floor the standardising volatility at 25% of the asset's own median, and
winsorise standardised returns at ±8. Either guard alone is sufficient; both are
applied. Regression tests in `tests/test_estimators_covariance.py` reproduce the
pathology and assert it stays fixed.

**Why the existing tests missed it:** they checked that the matrix was positive
semi-definite, had a unit diagonal, matched `sklearn` exactly, and produced
plausible volatilities. Every one of those passed. Nothing checked the
*correlation level*. A well-formed matrix can still be badly wrong.

Measured with the Yang-Zhang volatility that was current at the time, so the
columns isolate this one fix (HAR came afterwards and changed the level again):

| | Before | After the fix |
|---|---|---|
| Predicted portfolio volatility | 6.2% | **14.7%** |
| Realised | 13.9% | 13.9% |
| Bias (realised / predicted) | 2.25 | **0.95** |
| Predicted average correlation | 0.01 | **0.18** |
| Realised | 0.21 | 0.21 |
| 95% VaR breach rate | 16.6% (p<0.001) | **4.7%** (p=0.67) |

---

## 1. Per-stock volatility forecasts (21 days ahead)

| Model | R² | Slope | Bias | MAE |
|---|---|---|---|---|
| EWMA of squared closes | 0.272 | 0.75 | 0.98 | 0.078 |
| Yang-Zhang EWMA | 0.274 | 0.63 | 0.87 | 0.094 |
| GJR-GARCH-t | 0.246 | 0.61 | 0.92 | 0.088 |
| **HAR (shipped)** | **0.299** | **0.97** | **1.02** | **0.072** |

n = 1,295 forecasts. HAR wins on every column, and the slope matters as much as
the R²: at 0.97 the forecast is correctly scaled across its whole range, where
the others (0.61–0.75) systematically over-respond to their own signal.

HAR was chosen by a bake-off (`tools/improve_volatility.py`, 5,184 weekly
forecasts) rather than by preference:

| Model | R² |
|---|---|
| EWMA of squared closes | 0.311 |
| range-based only | 0.362 |
| HAR on close-to-close | 0.323 |
| HAR on range-based inputs | 0.376 |
| **HAR + market + India VIX** | **0.385** |

The largest single gain (+17%) came from reading volatility off the day's
**high-low range** rather than its close. HAR's three timescales add more, and
India VIX a little beyond that.

**How good is 0.385?** Two halves of the same forecast window correlate 0.38;
correcting that for the noise in a 10-day estimate implies only about **58% of
the target is predictable at all**. So the model reaches roughly **66% of what is
attainable**, up from 54%. The remainder is measurement error in the target.

**Yang-Zhang was not wrong, just mis-scored.** Close-to-close realised volatility
ignores intraday movement; against a range-based realised measure its bias is
0.98, not 0.87. It remains the automatic fallback when there is too little
history to fit the HAR regression, and that fallback is always recorded.

## 2. Portfolio volatility

Mean predicted **12.3%** vs realised **13.9%**: the model **understates portfolio
volatility by about 9–12%**. R² across the 35 monthly observations rose from
0.008 to **0.105** when HAR replaced the EWMA.

The residual level bias is a correlation effect, not a volatility one — per-stock
volatility is well scaled (bias 1.02), but predicted average correlation is 0.19
against 0.21 realised, and portfolio variance is dominated by the correlation term
at 37 holdings. Three candidate fixes were measured and **none of them worked**:

| Change | Bias |
|---|---|
| current (shrink towards identity) | 1.12 |
| shrink towards constant correlation | 1.10 |
| correlation on raw rather than standardised returns | 1.09 |
| shorter correlation window (252–756 sessions) | 1.13–1.18 (worse) |

A 1.12 → 1.09 improvement over 35 observations is noise, not evidence, so the
method was **left alone** rather than tuned to this sample — which is the
behaviour this project is supposed to have. The bias is documented, pinned by a
test, and does **not** affect reported VaR or expected shortfall, which are taken
from the empirical distribution rather than from the covariance matrix.

## 3. Correlation

Predicted **0.18** against realised **0.21** — correctly scaled, slightly shrunk,
which is the intended behaviour. R² for month-to-month variation is ~0, the same
caveat as above.

## 4. Value-at-risk calibration (1 day, equal weight, 730 days)

Standardised returns have **kurtosis 6.7** (a normal distribution has 3.0) and a
1st percentile of **−3.16** against the normal's −2.33. Indian equity tails are
fat, and that decides which method works:

| Confidence | Method | Breaches | Rate | Kupiec p | |
|---|---|---|---|---|---|
| 95% | normal quantile | 37 | 5.03% | 0.91 | pass |
| 95% | **empirical quantile (shipped)** | 37 | 5.03% | 0.91 | **pass** |
| 99% | normal quantile | 23 | 3.13% | <0.001 | **fail** |
| 99% | **empirical quantile (shipped)** | 8 | 1.09% | 0.79 | **pass** |

A Gaussian assumption produces **three times** the expected number of 99%
breaches. `star/risk/metrics.py` takes quantiles from the actual distribution and
has no function that scales one horizon to another, which is exactly why it
passes where the parametric shortcut fails.

## 5. Regime detection

F1 **0.77** (precision 0.81, recall 0.74, ROC AUC 0.93) over five independent
2,500-session simulations, using filtered probabilities only. Measured on
simulated data because real markets do not come labelled.

## 6. Signals — **absent**

There is no signal module. No momentum, value, quality or revision signals exist,
so no information coefficient can be measured. The Black-Litterman views
machinery is built and tested, but nothing feeds it.

**Consequence:** the optimiser currently has no stock-selection information. Run
with the equilibrium prior and no views, it reduces mathematically to the
reference portfolio, so it cannot outperform equal weight on return — and the
backtest confirms exactly that (p = 0.60 against 1/N).

---

## Summary

| Component | Verdict |
|---|---|
| Per-stock volatility | **R² 0.30, slope 0.97, bias 1.02** — best of four models tested |
| Portfolio volatility level | **Understates by ~9–12%**, documented and pinned |
| Correlation level | 0.19 predicted vs 0.21 realised |
| VaR 95% / 99%, empirical quantiles | **Passes** Kupiec at both |
| VaR with a normal assumption | **Fails** at 99% — three times the expected breaches |
| Volatility *timing* (month to month) | Weak: R² 0.105, up from 0.008 |
| Regime detection | **F1 0.77** on simulated data |
| Return forecasting / stock selection | **Does not exist** |
| Optimiser return advantage | **Not demonstrated**, and untestable until signals exist |
| Optimiser turnover advantage | **Demonstrated** — 19%/yr vs 39%/yr at half the cost |

The risk engine is sound and now verified, with one documented level bias. The
return engine has no inputs.
