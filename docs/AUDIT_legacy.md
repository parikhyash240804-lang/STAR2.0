# Stardust 1.x — Model Validation Report

**Subject:** `stardust_core.py` (6,161 lines, one function) and `stardust_app.py` (801 lines)
**Date:** 2026-09-13 · **Revised after adversarial verification:** 2026-09-16
**Purpose:** establish what the legacy engine actually computes, so STAR2.0 fixes causes rather than symptoms, and so every fix has a test named after the defect it closes.

## Status and review method

This report went through two stages.

1. **Review.** Six independent model-validation reviews covered M0–M1 (config + data), M2A (covariance, factor model, Black-Litterman), M2B (fundamentals, signals, views), M3–M4 (optimiser, frontier), M5–M6 (simulation, risk, stress) and M8 (walk-forward backtest). A seventh review of M7 (the Excel report) and the GUI was replaced by a direct measured inventory of the generated workbook.
2. **Adversarial verification.** A separate set of skeptics was given each module's findings and instructed to *refute* them: re-read the cited code, reproduce the numbers where possible (several ran the mechanism numerically, or live against the installed yfinance 1.5.1 / pandas 3.0.3), check which configuration path actually triggers each defect, and correct severity. Module M2B was re-verified at every severity; the other five modules were re-verified for their critical and high findings only.

**What "Verified" means in the tables:**

- **confirmed** — mechanism, impact and severity stand (sometimes with a line-reference fix, noted in the text).
- **partially** — the mechanism is real, but the magnitude, direction, severity, triggering path or line references were wrong. The finding text shown is the verifier's correction, not the original claim.
- **refuted** — the claim did not hold. No finding was refuted outright; several *sub-claims* were, and they are listed under [What verification changed](#what-verification-changed).
- **not re-verified** — medium/low findings outside M2B. They are reviewer-reported with line references and have not been challenged; treat them accordingly.

Items marked 🔬 were additionally reproduced during the original session (by reading the cited code or measuring the generated workbook).

Severity means: **critical** = the reported number is wrong or the conclusion it supports is unsupported; **high** = materially misleading or methodologically indefensible; **medium** = wrong but bounded, or right for the wrong reason; **low** = latent. Severities below are the **verified** severities; where verification changed one, the original is shown as "(was …)".

**"Default run?"** says whether the defect affects the shipped default configuration: `RETURN_METHOD='ewma'` (core line 99, `stardust_app.py:92`), `SHRINK_FACTOR=0.00`, `T_DIST_DF=6`, `LOOKBACK_Y=5`.

- **Yes** — fires in the default run. **Yes, partly** — part of the finding fires by default, the rest is path-conditional (the text says which).
- **Only bl/hybrid**, **Only factor/hybrid**, **Only bl/factor/hybrid** — fires only for those return methods. An unrecognised `RETURN_METHOD` is silently rebound to `bl` in the live run (1822), so "bl" includes that case. Under the default, the BL posterior still appears in the report as the "Black-Litterman" column of Sheet 3 (5088-5092), but never reaches the optimiser, simulation or risk figures.
- **Conditional** — depends on data, run date or a non-default setting (stated).
- **No** — latent; no current code path is affected.

For rows marked *not re-verified*, the "Default run?" entry is read from the cited code path and was not checked by a skeptic.

---

## Executive summary — what survived verification

Ordered by verified severity, with defects that affect the shipped default run first. Only three findings remain critical, and all three hit the default run.

**Critical — default run**

1. **A one-year loss is labelled "1-day" and then multiplied by √252.** (M6-02, critical, confirmed 🔬) The simulation VaR is a 1-year figure, but Sheet 1 and Sheet 7 label it "1-day", and Sheet 7's "Annual Equivalent" multiplies it by √252 = 15.87. A 10%-of-capital annual loss is shown as ~159% of capital. The Excel build is unconditional, so every run is affected.
2. **"Probability of gain" largely restates an assumption.** (M5-03, critical, confirmed) The simulation drift is the optimiser's own in-sample expected return with no out-of-sample haircut (`SHRINK_FACTOR=0.00` by default; only the 30% `RETURN_CAP` binds). An illustrative sweep shows how much the answer depends on that input: P(loss) is 26.7% at μ=12%, 13.8% at 20%, 8.3% at 25% and 4.7% at 30% (consistent with ~17% portfolio volatility). This demonstrates sensitivity to μ; the size of the upward bias in μ itself was not measured.
3. **The backtest tests only today's survivors.** (M8-02, critical, confirmed; same bug as M1-06) The universe is today's hand-picked ticker list, downloaded once and filtered on missing data over the *whole* sample, then sliced at every past rebalance. There is no point-in-time universe. The in-sample/out-of-sample date split inside the loop is itself clean; the look-ahead comes from universe selection. Magnitude not quantified.

**High — default run**

4. **The optimiser maximises estimation error.** (M3-01 / M2-03, high, confirmed) Max-Sharpe runs directly on EWMA means (halflife 252) with shrinkage switched off. A 30%-vol stock's annual mean has a standard error of about 17-18%, far larger than the true spread between stocks, so weights mostly follow noise. The per-stock cap of min(40%, 4/N) limits concentration but does not stop the chase.
5. **The backtest runs a close cousin of the live strategy, and its "IS → OOS drop" verdict is invalid.** (M8-01 high, M8-04 high) Under the default `ewma`, the backtest uses the *same* return estimator as the live run, but a raw sample covariance instead of the factor-structured/Ledoit-Wolf matrix, a 1% polish floor the live weights never get, and 8 optimiser restarts instead of 50. Its "in-sample Sharpe" is the live optimiser's ex-ante objective value, compared with a realised OOS Sharpe and labelled "(same model)", which is false. (Only under bl/factor/hybrid does the backtest become a genuinely different model; see below.)
6. **The backtest's VaR breach rate is always NaN.** (M8-03, high, confirmed) The completed-window counter is used to index the full rebalance calendar; because the first window is always skipped, every breach-window slice is empty. The NaN flows to the console, Sheet 11 and the calibration block.
7. **The backtest reports no uncertainty.** (M8-06, high, partially) No Sharpe standard error or confidence interval, no test against 1/N or the benchmark. At defaults the OOS span is about 3.2 years (~13 quarterly windows), and each ~63-day per-window Sharpe has a standard error of about 2.
8. **The benchmark is handicapped.** (M8-07 / M0-02, high) Stocks are dividend-adjusted but `^NSEI` is a price index, so outperformance against Nifty is overstated by roughly the index dividend yield, ~1.2-1.5%/yr. A large-cap benchmark for a mid/small-cap-tilted universe is a further mismatch.
9. **The fat-tail setting has almost no effect on any reported figure.** (M5-02 high, M5-01 medium) Shocks are i.i.d. over 252 days with a constant covariance, and the "Student-t" draws are independent t marginals mixed by Cholesky, not a multivariate t (daily portfolio excess kurtosis ~0.8 vs ~2.9). Since the app reports only 1-year terminal figures, the central limit theorem removes the difference either way — while the user-facing text claims the t-distribution "captures fat tails better than normal GBM".
10. **Return conventions are mixed, and the simulated drift is biased down.** (M2-04 high, M5-04 high) The default `arith_ewma` is a geometric rate (`exp(mean log) − 1`) treated as arithmetic, and M5 applies a per-stock σ²/2 haircut before aggregating log returns. The drift is understated by 0.5(Σwᵢσᵢ² − σ_p²) under every method (−2.23%/yr is a plausible size), partly offset under `ewma` by exp convexity (~+1%/yr at m=15%). This bias is *conservative* and runs opposite to M5-03.
11. **In-sample numbers are presented as "Expected".** (M3-02, high, partially) The summary sheet shows the optimiser-inflated return and Sharpe as "Expected Return p.a." with no caveat, beside a "Better" column that is true by construction for Sharpe and volatility. (M8 does label `port_sharpe` as in-sample.)
12. **The simulation holds 2-4 GB of RAM.** (M5-05, high, partially) Four (252, n, N) float64 arrays, including zero-weight stocks, are never released: 1.94 GB at n=24, 2.42 GB at n=15, ~4.0 GB at n=50 or n=100.

**High — only on non-default return methods (bl / hybrid / factor)**

13. **The risk-free rate is subtracted twice on the Black-Litterman path.** (M2-01 = M2B-02 = M3-04, high) The BL posterior is an excess return treated as total. Measured with the code's own bounds: the as-coded optimum differs from the correct one by 97% of the book, runs at 25.5% vol instead of 15.5%, and reports Sharpe 0.119 against a true 0.456. Critical on that path; the default `ewma` run is unaffected.
14. **Generated views are a uniform bullish market call.** (M2B-01 high, M2B-05 high) Every view is forced into +1.5% to +15.5% excess with a cross-sectional mean pinned at +8.5%, a +6.9pp level shift over the δ=0.5 prior. The measured posterior has mean +7.8% and only 0.29pp of cross-sectional dispersion — level swamps selection ~21:1. The score→view map uses a ±3 domain the composite never reaches (measured max |composite| 1.51), so only ~29% of the view band is used. The gap is jointly caused by `BL_DELTA=0.5` (at δ=2.5 it vanishes).
15. **`BL_TAU` does nothing and "40% confidence" is mis-scaled.** (M2B-11 high) τ cancels exactly (posterior bit-identical at τ = 0.01, 0.10, 1.0), and views carry ~91% of the posterior level, not 40%.
16. **Sector macro tilts produce no net tilt.** (M2B-04 high) `SECTOR_MULTIPLIERS` scale a sector-relative composite that is already mean-zero within each sector: a "bullish" 1.30× changes the sector's mean view by exactly 0.00pp and pushes below-average names further *down* (−0.42pp in test). The shipped template populates tilts from 0.88 to 1.15. Views reach the optimiser only on bl/hybrid.
17. **The signal/view section cannot be audited or backtested.** (M2B-10 high) No point-in-time capability, no as-of stamps, signals and views destroyed on return, and M8 never sees `BL_VIEWS`.

**Measured output defects (M7/GUI — direct measurement, not part of the skeptic pass)** 🔬 The workbook is 12 sheets ≈ 31 A4 pages with **no page setup on any sheet** (Excel prints portrait, where all 12 sheets are too wide), **zero formulas** across 1,856 cells with most numbers stored as pre-formatted *strings*, and a "Summary" whose first block reports the covariance method and simulation path count rather than a recommendation. The GUI's **"Download Report" button does nothing** — it opens a save dialog, then only sets a status message (`stardust_app.py:664-680`).

---

## What verification changed

Every headline claim that was downgraded, corrected or re-scoped, and why.

**Downgraded from critical**

- **TLS disabling (M0-01): critical → low.** Line 17 replaces stdlib `ssl`'s default context, which is read only by `urllib`/`http.client`. yfinance 1.5.1 sends traffic through curl_cffi (libcurl/BoringSSL), and its fallback `requests`/urllib3 builds its own SSLContext; neither reads the hook, and no repo code calls `urllib` or `http.client`. It is a bad latent practice to remove, not a live exposure of the price feed. The earlier "Two more worth stating plainly" line has been dropped from the summary.
- **"The model structurally cannot dislike a stock" (M0-04): critical → medium, bl/hybrid only.** The quoted +8% to +22% is the *total*-return band; views are +1.5% to +15.5% *excess*. Stocks are still ranked against each other (the optimiser responds to that spread); what cannot be expressed is an absolute bearish view. The real defect is the uniform bullish market view, which is M2B-01.
- **Uniform market view (M2B-01): critical → high, bl/hybrid only.** The ~+7pp figure was confirmed (+6.9pp), but it is inseparable from `BL_DELTA=0.5` (M2-05/M0-03) and never reaches the optimiser under the default `ewma`.
- **Double risk-free subtraction (M2-01 / M2B-02 / M3-04): critical → high.** Mechanism and impact confirmed and measured (97% of the book, 25.5% vs 15.5% vol), but it fires only for bl, hybrid (half strength) or an unrecognised method.
- **MOM factor ≡ 0 (M2-02): critical → medium.** Confirmed exactly (verifier simulation max|MOM| = 1.7e-16), but pinv sets β_MOM to 0 with no numerical failure. The consequence is a thinner covariance model on the default path — the same impact already filed as M2-08.
- **Dead EPS-revision signal (M2B-03): critical → medium.** Confirmed live: `quarterly_earnings` returns `None`. But losing a 0.10 weight cuts composite dispersion by only ~3% (variance contribution 0.01 of 0.22), and the composite mean is unchanged. The failure *is* printed to stdout; it is invisible to users only because the Tk app never shows stdout.
- **Sector multipliers "flip sign" (M2B-04): critical → high, mechanism refuted.** A positive multiplier cannot flip a sign. Because the composite is sector-relative and already mean-zero within each sector, the multiplier produces **zero net tilt** and only scales within-sector dispersion (1.30× exactly), inverting the intended direction for below-average names. M2-09 had described this correctly.
- **Price-index benchmark (M0-02): critical → high, direction corrected.** On the factor path the MKT premium is biased *down*, not up (then floored at 3%), and alpha is never used. The upward bias is in the backtest's relative outperformance (same issue as M8-07).
- **BL_DELTA=0.50 (M0-03): critical → medium, bl/hybrid only.** The ~1% implied premium was confirmed; with views on every stock supplying ~86-91% of the posterior level, the prior error moves μ_BL only modestly.
- **"The backtest does not test the strategy the app recommends" (M8-01): critical → high for the default run.** Under `ewma` the backtest uses the same EWMA(252) + shrink + cap return estimator, and the code prints "Walk-forward return method matches main run". It differs in covariance (sample vs factor-structured/Ledoit-Wolf), the 1% polish floor, and restarts (8 vs 50) — a close cousin, not a different model. BL views and the SIZE scalar are absent only because the default run doesn't use them either. On bl/factor/hybrid the original critical claim stands (and an unrecognised method falls back to `bl` live but `equal` in the backtest).

**Retained as critical, with qualifications**

- **M5-03:** the P(loss) figures (26.7% → 4.7%) are an illustrative sweep over μ, not output from a default run and not a measurement of the bias itself. The "units change with `RETURN_METHOD`" sub-claim overlaps M5-04.
- **M6-02:** confirmed as stated. The defect sits in the M7 Excel builder (4981-4986, 5388-5405) and shares a root cause with M6-01.
- **M8-02:** confirmed; the cited lines were corrected to the mechanism's source (core 592-655, `stardust_app.py:72-81`). The in-loop date slicing is clean.

**Downgraded from high**

- **Fat tails "cosmetic" (M5-01): high → medium.** Not a multivariate t (confirmed), but the app reports no 1-day tail metric, so the "99% 1-day ES understated 11%" never reaches a displayed number; the practical impact is contained in M5-02.
- **Stress clip "decides the 2008 number" (M6-04): high → medium.** The −80% floor binds only for stocks with β > 1.026 in 2008 (β > 1.62 for COVID, β > 3.08 for the default −25% custom scenario). It decides the figure only for high-beta holdings. The uniform ×1.3 (a beta-1 portfolio losing −78% in the −60% scenario) is confirmed but is a documented heuristic — a calibration issue.
- **Historical replay (M6-05): high → medium.** The displayed-but-unused market shock is confirmed and fires by default (Russia-Ukraine shown at −15% while Nifty fell ~9% in that window). Silent truncation is real but does not fire at today's run date and lookback. Added: the replay uses weights optimised on data that includes the same windows.
- **Transaction costs (M8-05): high → medium.** Costs after the first window are understated by exactly half, but the likely impact is ~0.4-0.7%/yr (~0.02-0.05 of Sharpe). No strategy pays drift costs — not only 1/N — which shares a root cause with M8-09.
- **`POLISH_FLOOR` "never applied" (M3-06): high → medium, sub-claim refuted.** It is applied, in the M8 walk-forward (4074, 4078), just not to the reported M3 weights. `MAX_ASSETS` genuinely is never enforced while every report says "Sparse (MAX_ASSETS=25)".
- **Geometric `arith_returns` (M3-03): high → medium.** Confirmed, but cited lines were wrong (mechanism at 1065/1071, false comment at 2071-2074), and the ~σ²/2 bias is an order of magnitude smaller than M3-01's estimation noise.
- **Ill-posed max-Sharpe (M3-05): high → medium.** The box-constrained problem always has a maximiser; the real risk is volatility-seeking local optima reported as success when no feasible portfolio beats rf. Data-dependent; frequency not confirmed.
- **Constraints (M3-07): high → medium.** A scope/design limitation, not a calculation error.
- **Random-Dirichlet frontier (M4-01): high → medium.** It does not trace the frontier and the "unbounded" label is wrong (it is long-only), but nothing downstream uses it — the impact is one chart and the M4 printout.
- **Backtest/estimator mismatch (M0-08, M2-12): high → medium.** On the default path the production *mean* estimator is validated out of sample; only the covariance model is not. Both duplicate M8-01.
- **BL prior, τ/Ω, factor premia (M2-05, M2-06, M2-07): high → medium**, each path-conditional. "40% confidence is fiction" was softened: each diagonal weight is ~0.35-0.4, but the aggregate pull toward views is ~85-91%.
- **Data handling (M1-05): high → medium.** Truncation is capped at ~5% of the window by `MISSING_THRESHOLD`, and a deleted row does not lose its return (the next kept row carries a multi-day return). Late listings are dropped with a printed message.
- **SIZE regime and floor (M0-07, M0-09): high → low**, factor/hybrid only. **Factor index symbol (M0-10): high → low** — the `'SC'` selector is dead code with no effect on the result, and the zero-SIZE degradation prints a warning. **Universe checks (M1-12): high → low** — the failures are loud, not silent. **Constant rf (M0-11): high → medium** — the error largely cancels in the MKT factor and hits all strategies alike in backtest Sharpe.
- **M2B signals:** M2B-07 high → low (sector-neutral momentum/low-vol is a defensible, contested design choice; the verifiable defect is that the in-code rationale describes a different signal). M2B-08 high → medium (weight 0.20; also discards PE > 200). M2B-09 high → low (a design choice, ~+0.14pp of μ_BL per +1σ, inert under `ewma`). M2B-12 high → medium, and the dominant failure mode is an empty `.info` dict that still counts as "Successful", not an exception. M2B-11: "BL collapses to a fixed per-asset blend" **refuted** (the blending matrix is non-diagonal), τ cancellation confirmed exactly.
- **Other qualifications on retained highs:** M8-06 "single-regime" softened (the OOS span includes the Sep-2024 to early-2025 Indian correction; "one short, largely bullish period"). M8-07's lost-benchmark-days effect is likely a handful of days, and the hard-coded "Nifty 50" label is wrong only if `BENCHMARK` is changed. M5-05's "2.9 GB at n=24, N=20k" used a path count the code never picks for n=24 and counted three arrays instead of four. M2-04's "mixed with arithmetic ones" applies only to hybrid.

**Raised**

- **Sequential fundamentals fetches (M2B-19): low → medium.** 3N uncached network round-trips are plausibly the dominant runtime of the whole run, and one of the three loops (`quarterly_earnings`) buys nothing.
- **M2B-14** (confirmed, medium) is wider than stated: `filterwarnings('ignore')` is process-global, persists after the run, and suppresses the yfinance DeprecationWarning that would have exposed M2B-03.

**Duplicate IDs describing one bug**

| Bug | IDs |
|---|---|
| BL posterior excess treated as total (double rf) | M2-01 = M2B-02 = M3-04 (M2-01 partly overlaps M2-11) |
| Uniformly bullish, level-shifting views | M0-04 = M2B-01 |
| Sector multiplier has no net tilt | M2B-04 = M2-09 |
| Survivorship / full-sample universe filter | M1-06 = M8-02 |
| Price-index benchmark vs dividend-adjusted stocks | M0-02 = M8-07 |
| Backtest vs live model mismatch | M8-01 ≈ M0-08 ≈ M2-12; M8-04 partly |
| BL_DELTA too low | M0-03 = M2-05 (prior part) |
| τ cancels / views dominate posterior | M2-06 ≈ M2B-11 |
| Dead EPS signal | M2B-03 = M2-14 (hidden by M2B-14) |
| Thin risk model from zero MOM factor | M2-02 ≈ M2-08 |
| Unshrunk EWMA means | M2-03 ≈ M3-01 |
| Geometric returns treated as arithmetic | M2-04 ≈ M3-03 ≈ M5-04 (units part of M5-03); M8-13 related |
| Annual VaR treated as 1-day | M6-02 shares root cause with M6-01 |
| Fat tails ineffective | M5-01 contained in M5-02 |
| Turnover measured against targets, not drifted holdings | M8-05 (drift part) shares root cause with M8-09 |

---

## M0 — Configuration, and M1 — Data

| ID | Sev | Verified | Default run? | Finding | Lines |
|---|---|---|---|---|---|
| M0-01 🔬 | low (was critical) | partially | No | Line 17 (`ssl._create_default_https_context = ssl._create_unverified_context`) disables certificate checks process-wide, but only for stdlib `urllib`/`http.client`. yfinance (curl_cffi, or `requests` as fallback) is not affected, and no repo code uses `urllib`/`http.client`. A bad latent practice to remove, not a live exposure of the price feed | 16-17 |
| M0-02 | high (was critical) | partially | Yes, partly — backtest benchmark; premium effect only factor/hybrid | Stocks are dividend-adjusted (`auto_adjust=True`) but `^NSEI` is a price index. (1) The backtest's outperformance against Nifty is overstated by roughly the index dividend yield, ~1.2-1.5%/yr (duplicate of M8-07). (2) On the factor/hybrid path the MKT premium is understated, not overstated, then floored at 3%. The regression intercept is biased but never used | 596, 706-707, 844, 1090, 4203-4212 |
| M0-03 | medium (was critical) | partially | Only bl/hybrid | `BL_DELTA` defaults to 0.50, below the 2.0 the code's own comments recommend (the GUI template also ships 0.50). That implies an equilibrium premium of ~1.0-1.3%, so generated views swamp the prior. High on that path, medium overall | 105, 246, 1726 |
| M0-04 🔬 | medium (was critical) | partially | Only bl/hybrid | Because `VIEW_FLOOR` (0.08) is above `RISK_FREE` (0.065), every generated view is a positive excess return (+1.5% to +15.5%), injecting a uniform bullish market view on a weak prior. Stocks are still ranked relative to each other, but no absolute bearish view can be expressed (with `RISK_FREE` above 0.08, floor views would turn negative). Duplicate of M2B-01. High on that path, medium overall | 93-94, 1593-1599 |
| M1-05 | medium (was high) | partially | Conditional — any ticker with short gaps or a listing inside the first ~5% of the window | `ffill(limit=2)` + `dropna()` creates zero-return days for gaps of up to 2 sessions. Longer gaps delete that date for all stocks, but the return lands on the next kept row as a multi-day return. Late-listing truncation is capped at ~5% of the window (~63 days) by `MISSING_THRESHOLD`; later listings (e.g. HYUNDAI.NS) are dropped from the universe with a printed message | 612-626, 654 |
| M1-06 | high | confirmed | Yes | The universe is conditioned on surviving and trading *today* and filtered on full-window missingness; the backtest slices that same panel at every historical rebalance — survivorship plus a look-ahead filter. Duplicate of M8-02 | 91, 612-614, 4049-4052 |
| M0-07 | low (was high) | partially | Only factor/hybrid | The SIZE regime scalar is a 5-bucket step function of one noisy 126-day midcap-minus-Nifty return spread. Its noise (~6%) is about as wide as the inner buckets (5%, 6%, 5%), so the regime label is close to arbitrary. Medium on that path, low overall | 319-377 |
| M0-08 | medium (was high) | partially | Yes, partly — covariance only | On the default `ewma` path, live and backtest use the same return estimator. They differ in covariance: live uses the factor-structured or Ledoit-Wolf matrix, the backtest a raw sample covariance. Only on bl/factor/hybrid does the backtest also drop the BL views, factor returns and SIZE scalar. Duplicate of M8-01 | 1023-1026, 3785-3805, 3851 |
| M0-09 | low (was high) | partially | Only factor/hybrid | `SIZE_FLOOR=0` clamps negative SIZE premia to zero before the regime scalar is applied, so the premium is one-sided and the scalar does nothing whenever the raw premium is negative. Medium on that path, low overall | 293, 1104-1109 |
| M0-10 | low (was high) | partially | Conditional — only if the midcap index download fails | The SIZE factor uses Nifty Midcap 100, not the smallcap index the comments describe. The `'SC'` selector test is dead code but does not change the result. If the download fails, SIZE is set to zero *with a printed warning*, dropping one term from the factor covariance | 288-290, 820, 851-866 |
| M0-11 | medium (was high) | partially | Yes, partly — walk-forward Sharpe/Sortino; 840/1127 only factor/hybrid | One constant `RISK_FREE` (6.5%) stands in for historical rf (RBI repo was 4% until May 2022). In the MKT factor the error largely cancels to (1−β) × the rf error (factor/hybrid only). In walk-forward Sharpe/Sortino, early-window absolute Sharpe is understated by up to ~2.5pp of rf divided by vol, but all strategies are hit alike, so comparisons are mostly unaffected | 97, 840, 1127, 4133-4134, 4249-4256 |
| M1-12 | low (was high) | partially | Conditional — too few survivors, or a symbol not echoed back unchanged | Universe min/max checks run before the missing-data filter and are not repeated, so fewer than 3 survivors are not caught by a clear message; a symbol yfinance renames raises `KeyError`. Both failures are loud, not silent | 128-138, 612-623 |
| M1-13 | medium | not re-verified | Yes | `VOLUME_MIN` is 10,000 *shares*/day — wrong unit for a liquidity screen, and it only prints a warning | 87, 636-639 |
| M1-14 | medium | not re-verified | Yes | No positivity or outlier validation before `np.log`; the diagnostic chart that would reveal it is discarded | 654-655, 668-677 |
| M0-15 | medium | not re-verified | Yes | `END_DATE = now()` makes every run irreproducible while the banner advertises reproducibility | 146-147 |
| M0-16 | medium | not re-verified | Yes | No bounds or unit validation on any parameter; the GUI swallows bad input silently | 77-115 |
| M0-17 | medium | not re-verified | Yes | `assert` used as user-facing validation — removed under `-O`, and surfaces as a stack trace | 131-138 |
| M1-18 | medium | not re-verified | Yes (see M2-04) | The return convention is never fixed at the data boundary, so log-space premia are later added to a simple rate | 654-655 |

## M2A — Covariance, factor model, Black-Litterman

| ID | Sev | Verified | Default run? | Finding | Lines |
|---|---|---|---|---|---|
| M2-01 | high (was critical) | partially | Only bl/hybrid | When `RETURN_METHOD` is bl, hybrid or unrecognised, the BL posterior is an excess return but is used as a total return: the Sharpe objective subtracts `RISK_FREE` again and M5 simulates it as a total drift (~6.5%/yr understated). Critical on that path; the default `ewma` run is clean. Same bug as M2B-02/M3-04; partly overlaps M2-11 | 1599, 1726, 1754-1783, 1822, 2086-2107, 2828-2830 |
| M2-02 🔬 | medium (was critical) | partially | Yes, partly — thinner covariance; MOM premium only factor/hybrid | The MOM factor is identically zero (a mean of cross-sectional z-scores; verifier simulation max\|MOM\| = 1.7e-16), so the "3-factor" model is really MKT+SIZE. It fails silently and harmlessly: pinv sets β_MOM to 0. The consequence is a thinner covariance model on the default path, not a numerical failure; overlaps M2-08 | 876-902 |
| M2-03 | high | confirmed | Yes | Default estimator is the EWMA mean of raw log returns (halflife 252, so the last year carries ~50% of the weight) with `SHRINK_FACTOR=0.00`, making the James-Stein step a no-op; the only moderation is a one-sided 30% `RETURN_CAP`. Max-Sharpe loads on recent winners. The walk-forward uses the same estimator, so its OOS results do test this | 1067-1071, 1899-1915 |
| M2-04 | high | partially | Yes (ewma/equal); mixing only hybrid | The `equal` and `ewma` "arith_*" series are geometric (`exp(mean log) − 1`) but treated as arithmetic. M3 under-rewards volatile names, and M5 subtracts σ²/2 again, so the default run's simulated drift is biased down by ~σᵢ²/2 per stock (~4.5%/yr at 30% vol, partly offset — see M5-04). `arith_factor` is also log-based; mixing estimator types only happens in hybrid (M2-11) | 1064-1071, 2828-2847 |
| M2-05 | medium (was high) | partially | Only bl/hybrid | δ=0.5 gives an implied market ERP of ~1-2% (1.7% in a realistic-Σ simulation), and `w_mkt` uses the current full market cap (not free-float, a look-ahead, 1.0 on fetch failure). Views on every stock supply ~86% of the posterior level, so the prior error is diluted | 105, 1711-1726 |
| M2-06 | medium (was high) | partially | Only bl/hybrid | Ω ∝ τ, so `BL_TAU` has no effect on the posterior (identical view-weight matrices at τ = 0.05, 0.1, 1.0). Each diagonal view weight is ~c = 0.4, but with views on all 40 names the aggregate pull toward views is ~85-90%, so the "40% confidence" label understates view dominance. The 8-22% band is an M2B choice | 1735-1783 |
| M2-07 | medium (was high) | partially | Only factor/hybrid | Factor premia are unweighted in-sample means (despite the "EWMA-weighted" comment) with lower-only floors (MKT ≥ 3%, SIZE ≥ 0%) and a live, non-as-of 6-month regime scalar in [0.1, 1.0] on SIZE | 1086-1109, 319-377 |
| M2-12 | medium (was high) | partially | Yes, partly — covariance only | M2 reads live data (market caps, fundamentals, the regime scalar) that cannot be replayed as-of, so bl/factor/hybrid estimators are never tested out of sample (the walk-forward substitutes EWMA). On the default run the walk-forward uses the same mean estimator but swaps the production factor-structured/LW covariance for a sample covariance, so only the covariance model is unvalidated. Overlaps M8-01 | whole module; 3785-3851 |
| M2-08 | medium | not re-verified | Yes | Risk model too thin: two live factors plus diagonal residuals for 40 names | 941-1035 |
| M2-09 | medium | not re-verified (see M2B-04) | Only bl/hybrid | Sector multipliers scale sector-*relative* z-scores, so the macro tilt partly cancels (M2B-04's verifier measured it cancels completely) | 1541-1581 |
| M2-10 | medium | not re-verified | Yes | "James-Stein" shrinkage is a fixed intensity set to 0; `RETURN_CAP` is a one-sided clip | 1887-1921 |
| M2-11 | medium | not re-verified | Only hybrid | `hybrid` averages estimators with incompatible units | 1793-1805 |
| M2-13 🔬 | medium | not re-verified | Conditional — bl/hybrid, when the covariance fallback fires | The covariance fallback runs *after* BL has already used a different Σ (surfaced independently by the split tool as a mid-pipeline rebinding of `cov_annual`/`cov_daily`) | 1855-1873 |
| M2-14 🔬 | medium | not re-verified (M2B-03 confirmed) | Yes, partly — signal dead in every run; reaches optimiser only bl/hybrid | yfinance 1.5.1 removed `quarterly_earnings` (it returns `None`), so the EPS-revision signal is dead | 1253 |
| M2-15 | medium | not re-verified | Only factor/hybrid (per M0-11) | A constant 6.5% policy-rate proxy is used as five years of daily risk-free rate | 840 |
| M2-16 | medium | not re-verified | Conditional — non-default configuration | Configuration edge cases crash or silently switch the model (`RETURN_METHOD` is rebound to `bl` for unknown values) | 1822 |

## M2B — Fundamentals, signals, view generation

All M2B findings were re-verified at every severity. Signals and views are computed in every run, but they reach the optimiser, simulation and risk figures only through the BL posterior, i.e. on bl/hybrid.

| ID | Sev | Verified | Default run? | Finding | Lines |
|---|---|---|---|---|---|
| M2B-01 | high (was critical) | confirmed | Only bl/hybrid (Sheet 3 column by default) | The score→view map emits an absolute expected-return level, not a cross-sectional alpha. Every view is forced into [+1.5%, +15.5%] excess, so no negative view is possible and the view vector's mean is pinned at +8.5% excess whatever the signals say. Against the δ=0.5 prior (Π ≈ +1.7%) that is a uniform +6.9pp market-level view; the measured posterior is +7.8% mean with only 0.29pp cross-sectional dispersion — level swamps selection ~21:1. The gap depends on `BL_DELTA` and is inseparable from M2-05 | 1593-1624 |
| M2B-02 | high (was critical) | confirmed | Only bl/hybrid | `arith_bl` is excess (the view map subtracts rf at 1599; Π is already a risk premium) while `negative_sharpe` subtracts rf again. Measured under bl: the optimiser's solution differs from the correct one by 97% of the book and runs at 25.5% vol instead of 15.5% (reported Sharpe 0.119 vs true 0.456). Duplicate of M2-01/M3-04 | 1599, 1726, 2081-2086 |
| M2B-03 🔬 | medium (was critical) | partially | Only bl/hybrid | `eps_revision` is dead on yfinance 1.5.1: `quarterly_earnings` returns `None`, so every ticker falls to NaN and the fill chain resolves to the hard-coded 0.0. The 0.10 weight is never renormalised (effective weights sum to 0.90). The dispersion cost is ~3%, not 10% (variance contribution 0.01 of 0.22), and the composite mean is unaffected. Coverage (0%) is printed to stdout, which never reaches the Tk app's user; the DeprecationWarning that would have flagged it is suppressed by M2B-14 | 1248-1293, 1333 |
| M2B-04 | high (was critical) | partially | Only bl/hybrid (tilts ship in the default template) | `SECTOR_MULTIPLIERS` multiply a sector-relative composite that is already mean-zero within each sector, so they produce **no net sector tilt**: a 1.30× "bullish" multiplier changes the sector's mean view by exactly 0.00pp and only scales within-sector dispersion by 1.30×. No sign is flipped, but the direction is inverted for every below-average name, which is pushed further down (up to −0.42pp in test). The shipped template carries tilts from 0.88 to 1.15. Same defect as M2-09 | 1541-1581 |
| M2B-05 | high | confirmed | Only bl/hybrid | The score→view map uses a fixed ±3 domain the composite cannot reach (theoretical max \|composite\| 2.5, 2.25 with EPS dead). Over 300 simulated 40-name universes: composite std 0.41, max \|composite\| 1.51; the per-portfolio view range has median 4.10pp against the nominal 14.0pp band (29% used) and a view std of 0.96pp around the fixed +8.5% mean | 1593-1599 |
| M2B-06 | high | confirmed | Only bl/hybrid | Sector-relative z-scores are capped by the sample-std identity max\|z\| = (n−1)/√n, not by `WINSOR_CAP`: 1.15 for a 3-stock sector, 1.50 at n=4, 2.04 at n=6, 2.47 at n=8. The ±2.5 winsorisation binds only for sectors of 9+ stocks or the universe fallback (6.17 at n=40), so a stock's maximum score depends on how many peers it has — scores are not comparable across sectors | 1476-1540 |
| M2B-07 | low (was high) | partially | Only bl/hybrid | Momentum and low-volatility are z-scored inside sector peer groups, making both sector-neutral. Whether that helps or hurts is a live research question, not a defect. The verifiable defect is documentation: line 1369 calls momentum "the Carhart UMD factor in stock form" and 1376-1379 justify low-vol with a cross-sectional argument, neither of which describes the signal actually used | 1369-1379, 1495-1540 |
| M2B-08 | medium (was high) | confirmed | Only bl/hybrid | The PE filter (0, 200) discards loss-making companies *and* extremely expensive ones, making both indistinguishable from missing data; the fill chain then assigns the sector median, so a loss-maker and a 300×-PE name both score at sector average on value instead of at the bottom. No record of why a value was rejected. Weight 0.20, so modest magnitude | 1208-1246, 1308-1358 |
| M2B-09 | low (was high) | partially | No — inert under ewma | The low-volatility signal raises expected return for names the covariance already favours, so the tilt is expressed twice in max-Sharpe. This is how a low-risk factor is implemented in BL — a design choice, not an error — and the measured magnitude is ~+0.14pp of μ_BL per +1σ of low-vol z after shrinkage, against a 10% per-name cap | 1412-1422 |
| M2B-10 | high | confirmed | Only bl/hybrid | No point-in-time capability (zero as-of hits in the file; live `.info`/`.fast_info` joined to five years of history), signals/z-scores/composite/views are function-locals destroyed on return (the "exposed for inspection" comment is false), and `_report_context` carries no `BL_VIEWS` — M8 never sees them and the report cannot show which view produced which weight | whole section; 1665-1672, 4810-4862 |
| M2B-11 | high | partially | Only bl/hybrid | With one absolute view per asset, P is a permutation and Ω = ((1−c)/c)·τ·diag(Σ), so τ cancels exactly: μ_BL is bit-identical at τ = 0.01, 0.10, 1.0 — `BL_TAU` is a dead control. The posterior does **not** collapse to a per-asset blend (Σ is full, so the blending matrix is non-diagonal). The real scaling defect: views carry ~91% of the posterior level, not the 40% `BL_VIEW_CONFIDENCE` advertises | 1735-1783 |
| M2B-12 | medium (was high) | partially | Conditional — on a fundamentals fetch failure; effect only bl/hybrid | When fundamentals are unavailable, sectors collapse to one "Unknown" group and PE/ROE resolve to the constants 20.0 and 0.12, so three of five signals (0.50 of weight) become zero and view dispersion falls ~35% (view std 0.80pp → 0.52pp). The dominant failure mode is an empty or partial `.info` dict, which raises nothing and still counts as "Successful .info calls: N/N". Printed to stdout, invisible in the Tk app | 1218-1246, 1287-1293, 1328-1333 |
| M2B-13 🔬 | medium | confirmed | Only bl/hybrid | After line 1624 `BL_VIEWS` holds two conventions — systematic entries excess (1599), manual entries totals corrected at 1753 — with nothing in the dict to tell them apart. Disambiguation relies on the function-local set `_manual_overrides`, defined ~140 lines earlier (a plain local, not a closure variable) | 1599, 1610-1624, 1751-1754 |
| M2B-14 | medium | confirmed | Yes | `warnings.filterwarnings('ignore')` at line 1174 installs a *process-global* filter that persists after `run_stardust` returns (re-applied at 3747, 4694). It suppresses the yfinance DeprecationWarning raised on line 1253 — the warning that would have revealed M2B-03 | 1174, 3747, 4694 |
| M2B-15 | low (was medium) | partially | Yes (cosmetic) | The fetched fundamentals columns (trailingPE, returnOnEquity, sector, industry) and derived earnings_yield are object dtype from `pd.DataFrame(dict-of-dicts).T`; eps_revision and the two price-based signals are float64, so "all-object" overstates it. On pandas 3.0.3 every downstream operation still works; the cost over ≤100 rows is immaterial | 1274-1275, 1352 |
| M2B-16 | medium | confirmed | Only bl/hybrid | ROE 5th/95th-percentile winsorisation is computed on the post-imputation series, so every median-filled value pulls the bounds inward and clips genuine observations harder; never leverage-adjusted | 1342-1348 |
| M2B-17 | medium | confirmed | Only bl/hybrid | Signals combined by asserted weights: no estimated IC, no orthogonalisation, no correlation adjustment. (The weights *are* validated to sum to 1.0 at 432-437; the defect is absence of estimation) | 1554-1558 |
| M2B-18 | medium | confirmed | Conditional — only if misconfigured; bl/hybrid | No validation that `VIEW_FLOOR < VIEW_CEILING`; swapped values silently invert every view. Also unchecked: `VIEW_FLOOR ≥ 0` and `VIEW_FLOOR` against `RISK_FREE`, which is what makes negative views impossible (M2B-01) | 93-94, 1584-1598 |
| M2B-19 | medium (was low) | confirmed | Yes | Three sequential per-ticker loops (`.info`, `.quarterly_earnings`, `.fast_info`) = 3N uncached yfinance round-trips, a fresh `Ticker` each time. At N=40 plausibly the dominant runtime of the run, and the middle loop is pure waste because `quarterly_earnings` returns `None` | 1214, 1251, 1712 |
| M2B-20 | low | confirmed | Yes | The dependency docblock lists four variables never used in 2B.5-2B.7 and omits `SCORE_FLOOR`, `SCORE_CEILING`, `WINSOR_CAP`; fill diagnostics report constant default-fills as universe-median fills (for eps_revision, "N from universe median" when the truth is "N from the constant 0.0") | 1157-1167, 1335-1340 |

## M3 — Optimiser, M4 — "Efficient frontier"

| ID | Sev | Verified | Default run? | Finding | Lines |
|---|---|---|---|---|---|
| M3-01 | high (was critical) | confirmed | Yes | Max-Sharpe runs on EWMA means with James-Stein shrinkage switched off (`SHRINK_FACTOR=0.00`; the code exists at 1899-1901). Per-stock estimation error (~17-18% SE for a 30%-vol name) far exceeds the true cross-sectional spread, so weights mostly follow noise. The min(40%, 4/N) cap and 30% return cap limit concentration but do not stop the chase — hence high, not critical | 2079-2133 |
| M3-02 | high | partially | Yes | The M3 return and Sharpe are in-sample, optimiser-inflated figures, labelled "Expected" in the printout and as "Expected Return p.a." in the Excel summary with no caveat. The "Better" column is true by construction for Sharpe and volatility. (M8 does label `port_sharpe` as in-sample and shows the IS→OOS drop) | 2129-2160, 4953-4961 |
| M3-03 | medium (was high) | confirmed | Yes (ewma/equal) | On the default and `equal` paths, `arith_returns` is `exp(annualised mean log) − 1`, a geometric rate; the M3 comment wrongly calls it arithmetic. The ~σᵢ²/2 downward bias tilts toward low-vol names but is an order of magnitude smaller than M3-01's estimation error | 1065, 1071, 2071-2074 |
| M3-04 | high | confirmed | Only bl/hybrid | Risk-free subtracted twice under `bl` (and unknown methods), by half under `hybrid`: the Sharpe hurdle becomes 2rf / 1.5rf and expected returns are understated by rf (6.5%) under bl. Duplicate of M2-01 | 1599, 1726, 1807-1822, 2086, 2107 |
| M3-05 | medium (was high) | partially | Conditional — data-dependent; more likely under bl | When no feasible portfolio beats rf (roughly, the top ⌈N/4⌉ capped names cannot reach μ > rf), max-Sharpe becomes volatility-seeking and non-quasi-concave; SLSQP may report success on a meaningless local optimum with no warning. The box constraints keep the problem well-defined (not unbounded). Frequency on real data not confirmed | 2079-2125 |
| M3-06 🔬 | medium (was high) | partially | Yes | `MAX_ASSETS` is never enforced, yet the summary always says "Sparse (MAX_ASSETS=25)" because the app always passes an int. `POLISH_FLOOR` is not applied to M3's reported weights but **is** applied in the M8 walk-forward (4074, 4078), so the backtest trades differently-processed weights from the ones reported | 108, 2001, 4074-4078, 4873-4874 |
| M3-07 | medium (was high) | partially | Yes | The per-stock cap depends only on universe size (min(40%, 4/N)), with a zero floor and a budget constraint; no sector, liquidity, turnover or cost constraints (M8 applies costs only after the fact), and no assertion that n·WEIGHT_MAX ≥ 1. A scope/design limitation, not a calculation error | 1999-2025 |
| M4-01 | medium (was high) | partially | Yes | The "efficient frontier" is a random Dirichlet cloud over the long-only simplex; in high dimensions it undersamples the edges and does not trace the true frontier. It is mislabelled "unbounded / full opportunity set" (it is long-only). Impact limited to one chart and the M4 validation printout — nothing downstream uses it | 2387-2412 |
| M3-08 | medium | not re-verified | Yes | The multi-start rationale is wrong (the problems are convex) | 2025-2040 |
| M4-02 | medium | not re-verified | Yes | The "optimizer validation" is statistically invalid (false-positive by construction) | 2459-2503 |
| M4-03 | medium | not re-verified | Yes | M4 costs up to ~2.8 GB and ~40 s to produce one chart | 2336-2412 |
| M3-09 | medium | not re-verified | Yes | M3/M4 guidance and warnings never reach the user (stdout is discarded) | throughout |
| M3-10 | medium | not re-verified | Yes | Hidden global state: legacy `np.random.seed` plus process-wide thread pinning | 170-171 |

## M5 — Forward simulation, M6 — Risk and stress

| ID | Sev | Verified | Default run? | Finding | Lines |
|---|---|---|---|---|---|
| M5-03 | critical | confirmed | Yes | The drift is the in-sample, optimiser-selected expected return with no out-of-sample haircut (`SHRINK_FACTOR=0`; only the 30% `RETURN_CAP` binds), and it flows into P(loss), VaR, CVaR and M6. Illustrative sweep: P(loss) 26.7% at μ=12%, 13.8% at 20%, 8.3% at 25%, 4.7% at 30% (consistent with ~17% portfolio vol) — this shows sensitivity to μ, not a measured bias. The "units change with `RETURN_METHOD`" sub-claim overlaps M5-04 | 2828-2832, 2948-2956 |
| M6-02 🔬 | critical | confirmed | Yes | A 1-year VaR is labelled "1-day" (Summary and Risk sheets) and multiplied by √252 for an "Annual Equivalent": a 10%-of-capital loss is shown as ~159%. The same sheet then shows the same quantity again as "Annual VaR 95% (simulation)". The defect is in the M7 builder; shares a root cause with M6-01 | 2954, 4981-4986, 5388-5405 |
| M5-01 | medium (was high) | partially | Yes (T_DIST_DF=6) | Shocks are independent t marginals correlated through Cholesky, not a multivariate t, so portfolio fat tails are diluted (daily excess kurtosis ~0.8 vs ~2.9). None of the app's outputs are 1-day tail figures, and over 1 year the CLT removes the difference either way, so reported VaR, CVaR and P(loss) barely change. Practical impact contained in M5-02 | 2887-2923 |
| M5-02 | high | confirmed | Yes | i.i.d. daily shocks with one constant covariance: no volatility clustering, no crisis correlation, and the 1-year terminal distribution is close to lognormal by the CLT. The code comments (2816-2825) document this while the user-facing text (2982) claims the t-distribution captures fat tails better than normal GBM | 2787-2927, 2982 |
| M5-04 | high | partially | Yes | The drift is understated by 0.5(Σwᵢσᵢ² − σ_p²) under every method because a per-stock Itô term is applied before aggregating log returns (−2.23%/yr is plausible, e.g. ~28% average stock vol vs ~18% portfolio). Under the default `ewma` the input is already compounded, so the per-stock σᵢ²/2 haircut is largely redundant, only partly offset by exp convexity (~+1%/yr at m=15%); the net understatement can reach several %/yr. Conservative — runs opposite to M5-03. The gap check at 2843-2847 will usually print a misleading warning | 2828-2847, 2925-2926 |
| M5-05 🔬 | high | partially | Yes | Four (252, n, N) float64 arrays (`Z_raw`, `Z`, `corr_shocks`, `daily_log`) stay alive for the rest of `run_stardust` and include zero-weight stocks: 1.94 GB at n=24 (N=10k), 2.42 GB at n=15 (N=20k), ~4.0 GB at n=50 (N=10k) or n=100 (N=5k) | 2747-2752, 2889-2927 |
| M6-04 | medium (was high) | partially | Yes (2008 always; COVID at today's run date; custom scenario) | Beta-scaled stress applies a uniform ×1.3, so a beta-1 portfolio loses 1.3× the scenario's market move (−78% in the −60% 2008 scenario). The [−80%, 0] clip binds only for β > ~1.03 in 2008 (β > 1.62 for COVID), so it decides the 2008 figure only for high-beta holdings; it also zeroes gains for negative-beta names. A documented heuristic — a calibration issue | 3394-3415 |
| M6-05 | medium (was high) | partially | Yes, partly — displayed shock mismatch; truncation run-date dependent | For scenarios replayed from history, the configured market drop is displayed but not used; the loss is the portfolio's actual endpoint return (Russia-Ukraine shown as −15% while Nifty fell ~9% in that window). Windows only partly covered by data are accepted with ≥ 5 rows (not active at today's defaults). The replay also uses in-sample weights | 3359-3434 |
| M6-01 | medium | not re-verified | Yes | The M5-vs-M6 "1-day" comparison divides a drift-inclusive annual VaR by √252, overstating agreement by 2×–8.8× | 3224-3238 |
| M6-03 | medium | not re-verified | Yes | Historical VaR applies in-sample weights to a calm 5-year window (no 2008, no March 2020), at 95% and 1 day only | 3193-3202 |
| M6-06 | medium | not re-verified | Yes | Betas re-downloaded over a different window than the covariance, ignoring `LOOKBACK_Y` | 3305-3331 |
| M5-06 | medium | not re-verified | Yes | VaR/CVaR include drift, are 95% only, fixed at one year, and can go negative (a "gain at risk" shown in red) | 2944-2980 |
| M5-07 | medium | not re-verified | Yes | Probability of gain/loss given as a precise point estimate, with no benchmark or risk-free comparison | 2948-2977 |
| M6-07 | medium | not re-verified | Yes | The "HONEST RISK SUMMARY" contains hard-coded claims that were never computed | 3571-3595 |
| M5-08 | low | not re-verified | Conditional — only if `T_DIST_DF` ≤ 2 | `T_DIST_DF` unvalidated: df = 2 raises `ZeroDivisionError` after all upstream work; df < 2 silently yields NaN everywhere | 101-103, 2887-2900 |

## M8 — Walk-forward backtest

| ID | Sev | Verified | Default run? | Finding | Lines |
|---|---|---|---|---|---|
| M8-01 | high (was critical; critical on bl/factor/hybrid) | partially | Yes, partly — covariance, polish floor, restarts; μ/views/SIZE only bl/factor/hybrid | In the default `ewma` configuration the backtest runs the same EWMA(halflife 252) + shrink + cap return estimator as the live run. It differs in covariance (sample covariance vs factor-structured/Ledoit-Wolf), in the post-optimisation 1% polish floor, and in optimiser restarts (8 vs 50) — a close cousin of the live strategy, not the strategy itself. Only under bl/factor/hybrid does it also drop BL views, factor returns and the SIZE scalar; an unknown method falls back to `bl` live but `equal` in the backtest | 1000-1071, 1999-2133, 3785-3851, 4061-4083 |
| M8-02 | critical | confirmed | Yes | Survivorship and look-ahead: the universe is today's hand-picked list (including recent winners and listings), downloaded once and filtered on full-sample missingness, then sliced at every past rebalance. No point-in-time universe. The in-loop IS/OOS date split is clean; the look-ahead comes from universe selection. Magnitude not quantified | core 592-655; `stardust_app.py:72-81`; 4049-4052 |
| M8-03 | high | confirmed | Yes | VaR breach rate is always NaN — the completed-window index is used to index the full rebalance calendar; because the first rebalance is always skipped, every breach slice is empty. Holds for ME/3ME/6ME at any lookback; the NaN reaches the console, Sheet 11 and the calibration block | 4281-4299 |
| M8-04 | high | confirmed | Yes | "IS → OOS Sharpe drop" compares `port_sharpe` — the live optimiser's ex-ante objective on full-sample μ and factor/LW covariance with unpolished weights — against a realised OOS Sharpe from sample-covariance, polished, 8-start portfolios. The console labels it "(same model)", false in the default run; the < 50% "acceptable" verdict has no basis. Partly duplicates M8-01 | 4380-4397, 5717-5754 |
| M8-05 | medium (was high) | partially | Yes | Per-rebalance costs after the first window are understated by exactly half (one side charged). No strategy pays drift costs, because turnover is measured against prior target weights (shared root cause with M8-09); 1/N's turnover is exactly zero. The cost summary averages the one-off 100% construction turnover into the annualised figure. Likely impact ~0.4-0.7%/yr (~0.02-0.05 Sharpe) | 3955, 4087-4107, 4424-4430 |
| M8-06 | high | partially | Yes | No statistical inference: no Sharpe standard error or CI, no test against 1/N or the benchmark, no multiple-testing adjustment. The default run gives ~3.2 years and ~13 quarterly OOS windows; each ~63-day per-window Sharpe has a standard error of ~2. One short, largely bullish period (it does include the Sep-2024 to early-2025 correction) | 4232-4332 |
| M8-07 | high | partially | Yes | The benchmark is a price index compared against dividend-adjusted stocks, a ~1.2-1.4%/yr handicap — the material part. Labels are hard-coded "Nifty 50" whatever `BENCHMARK` is set to. Benchmark days missing from the factor-index panel are silently dropped (real, likely small). A large-cap benchmark for a mid/small-cap-tilted universe is a further mismatch | 596, 725-731, 4195-4221, 4322, 4354 |
| M8-08 | medium | not re-verified | Yes | The VaR "calibration" tests the wrong quantity | 5835-5865 |
| M8-09 | medium | not re-verified | Yes | Daily constant-mix returns: weights never drift between rebalances | 4108-4131 |
| M8-10 | medium | not re-verified | Yes | Today's constant risk-free rate used in historical optimisations | 4072 |
| M8-11 | medium | not re-verified | Conditional — n ≳ 75 tickers or legacy frequency strings | Aborts the whole run for n ≳ 75 tickers or legacy frequency strings | 4055-4059 |
| M8-12 | medium | not re-verified | Conditional — all μ ≤ rf | Max-Sharpe solver maximises volatility when all μ ≤ rf | 3854-3895 |
| M8-13 | medium | not re-verified | Yes | "Expected return" is the geometric (median) return, not the arithmetic mean | 3831-3851 |
| M8-14 | medium | not re-verified | Yes | Rebalance calendar and data window depend on the run date, so results are not reproducible | 3763, 3995-4005 |
| M8-15 | medium | not re-verified | Yes | Output is a console dump plus noisy panels, with no decision content | 4400-4644 |
| M8-16 | medium | not re-verified | Yes | Sortino denominator is the std of the loss subset, not downside deviation | 4253-4256 |

## M7 and the GUI — measured output inventory

All figures below were measured on the generated `portfolio_report.xlsx` (2.0 MB, 12 sheets) and by reading the generator. 🔬 throughout. This section was a direct measurement and was not part of the adversarial pass; every item affects the default run.

| Finding | Verified | Default run? | Measurement |
|---|---|---|---|
| **No print setup exists at all** | measured directly | Yes | `page_setup.orientation`, `paperSize`, `fitToPage`, `print_area`, `print_title_rows`, `row_breaks`, `page_margins`, headers/footers: unset on **all 12 sheets**. Excel therefore prints **portrait**, where **all 12 sheets are too wide** (narrowest 7.35 in vs 6.77 in usable; widest 11.12 in) |
| **Workbook size** | measured directly | Yes | ≈ **31 A4 landscape pages**; `10. Charts` alone is 8–10 pages |
| **Nothing is interactive or recomputable** | measured directly | Yes | **0 formulas** in 1,856 cells, 0 data validations, 0 defined names, 0 native charts, 0 conditional formatting, 0 autofilters. Most numerics are pre-formatted **strings** (`Rs.1,234`, `12.34%`) via `rupee()`/`pct()`, so no cell can be recomputed or reformatted |
| **Charts are print-hostile** | measured directly | Yes | Four PNGs at 150 dpi with a dark `#0f0f1a` background baked in |
| **Image row bands 1.33× too tall** | measured directly | Yes | `rows_needed = int(img.height/15)` treats pixels as points (15 pt = 20 px) |
| **Diagnostics sheet loses its tail** | measured directly | Yes | The back-fill computes `last_row` *after* three more log entries were appended, so `late_entries` is empty: rows 53–54 blank, the last three entries never appear, and the banner reads "Total entries: 49" while 52 were generated |
| **Conflicting column widths** | measured directly | Yes | `11. Walk-Fwd` pushes four different width vectors at the same 8 columns; last write wins, so 32-character labels sit in an 18-character column |
| **Summary leads with engine internals** | measured directly | Yes | The first block reports covariance method, optimiser version and simulation path count; the risk block shows the mislabelled VaR (M6-02); there is no recommendation, regime, benchmark comparison or "what changed" |
| **"Download Report" does nothing** | measured directly | Yes | `stardust_app.py:664-680` opens a save dialog, then only sets a status string. No file is ever written |
| **Uninvited side effects** | measured directly | Yes | Every run copies the workbook over `~/Downloads/portfolio_report.xlsx` and auto-opens Excel (`os.startfile`) |

---

## Why the backtest is not trustworthy

Four verified reasons. The first is sufficient on its own to stop the track record being quoted as evidence; the others compound it.

1. **It tests only today's survivors** (M8-02 critical, M1-06). The universe is today's hand-picked ticker list, filtered on missing data over the whole sample, and that fixed panel is sliced at every past rebalance. The date split inside the loop is clean — the look-ahead is in *which stocks* are there. The size of the resulting bias has not been measured.
2. **It tests a close cousin of the live strategy, and misreports the comparison** (M8-01 high, M8-04 high). In the default `ewma` run the return estimator is the same, but the backtest uses a sample covariance instead of the factor-structured/Ledoit-Wolf matrix, adds a 1% polish floor, and uses 8 restarts instead of 50. Its "IS → OOS drop" then compares the live optimiser's ex-ante objective against realised OOS returns under a false "(same model)" label. Only a user who switches to bl/factor/hybrid gets a backtest of a genuinely different model (no BL views, factor returns or SIZE scalar).
3. **Its benchmark is handicapped and its costs are low** (M8-07 high, M8-05 medium). The Nifty price index omits a ~1.2-1.4%/yr dividend yield the portfolio receives, and is a large-cap yardstick for a mid/small-cap-tilted book. Costs after the first rebalance are exactly half what they should be, and no strategy pays drift costs (M8-09, not re-verified); the likely cost effect is modest, ~0.4-0.7%/yr.
4. **It reports no uncertainty** (M8-06 high). No Sharpe confidence interval and no test against 1/N or the benchmark, on ~3.2 years and ~13 quarterly windows where each per-window Sharpe has a standard error of about 2. The one column that would have been a genuine check — the VaR breach rate — is always NaN (M8-03 high).

The honest expectation for STAR2.0: once the universe is point-in-time and the benchmark and costs are corrected, **the measured Sharpe is expected to fall.** The cost correction alone is small (~0.02-0.05 of Sharpe); the survivorship correction is likely larger but is unquantified. Measuring and publishing the before/after gap is a deliverable.

## Why the Monte Carlo misleads

- Its headline answer is mostly an echo of its input (M5-03, critical): the drift is the optimiser's own in-sample estimate with shrinkage switched off. An illustrative sweep moves P(loss) from 26.7% to 4.7% as μ goes from 12% to 30%. The upward bias in μ itself has not been measured.
- Its risk figures are mislabelled in the report (M6-02, critical): the 1-year VaR appears as "1-day" and is inflated by √252 into an "Annual Equivalent" of ~159% of capital for a 10% loss.
- Its distributional sophistication buys nothing at the horizon it reports (M5-02 high, M5-01 medium): i.i.d. days with constant covariance, and "Student-t" shocks that are not a multivariate t. Because only 1-year figures are reported, the CLT makes the t setting almost irrelevant to VaR, CVaR and P(loss) — while the user is told it captures fat tails.
- Its drift has a *downward* bias running the other way (M5-04 high): 0.5(Σwᵢσᵢ² − σ_p²) is lost under every method (−2.23%/yr is plausible), and under `ewma` a redundant per-stock σ²/2 haircut is only partly offset by ~+1%/yr of exp convexity. Because this bias is conservative and M5-03's is optimistic, the net direction of the headline error is not established.
- It holds 2-4 GB of RAM (M5-05 high): 1.94 GB at n=24, ~4.0 GB at n=50.
- The separate "efficient frontier" Monte Carlo (M4-01, medium) samples random long-only weights, which undersample the edges and cannot trace the frontier; it affects one chart only.

## What is worth keeping

Not everything is wrong, and the rebuild should preserve these:

- The **structure of the pipeline** — config → data → estimation → optimise → simulate → risk → backtest → report — is the right decomposition, which is why the mechanical split into 11 modules worked without touching the logic.
- **Ledoit-Wolf shrinkage** is correctly delegated to scikit-learn, and the factor-structured covariance `B F Bᵀ + D` is correctly formed given its factors (one of which, MOM, is identically zero — harmlessly, per M2-02).
- The walk-forward's **in-loop date split is clean** (`< _rebal_dt` / `> _rebal_dt`); its look-ahead problem is the universe, not the slicing (M8-02).
- The **alignment discipline** — six hand-written `assert list(x.index) == list(y.columns)` checks — is the right instinct with the wrong mechanism; STAR2.0 replaces it with a decorator and a permutation-invariance test.
- The **diagnostics log** concept (a per-module health trail) is genuinely good; it just never reaches the user intact.
- The **Excel-in / Excel-out** workflow suits the audience and is retained.

## Environment and compatibility risks

- `yfinance` 1.5.1 removed `quarterly_earnings` (M2-14/M2B-03), and the DeprecationWarning it raises is swallowed by the process-global warnings filter (M2B-14). `yf.download` returns MultiIndex columns even for a single ticker, which M6's benchmark download does not handle defensively. yfinance's HTTP stack is curl_cffi (fallback `requests`), which is why the stdlib TLS override does not reach it (M0-01).
- pandas 3.0 changes string dtypes, copy-on-write and silent downcasting; the engine currently survives (object-dtype fundamentals were confirmed to work on pandas 3.0.3, M2B-15), but several idioms are fragile.
- matplotlib figures are never closed (`plt.show()` under the Agg backend warns and leaks) — M4/M5/M6 each hold multi-hundred-MB figures.
- The legacy depends on the *legacy* global `np.random.seed`, which makes reproducibility dependent on call order across modules.

## Appendix — module data contracts

The exact inter-stage contract (which variables each stage reads and publishes) is machine-generated at `legacy/stardust_v1/_split_report.json`, and each module states its own `INPUTS`/`OUTPUTS`. Three contract observations that corroborate findings above: `m2b` rebinds `BL_VIEWS` (M2B-13), `m2c` rebinds `cov_annual`/`cov_daily` after BL has consumed them (M2-13), and `m2c` rebinds `RETURN_METHOD` (M2-16).

Parity evidence that the modular port is faithful — 908/908 printed lines, 0/1,856 differing cells, 4/4 charts pixel-identical — is in `legacy/PARITY.md`.
