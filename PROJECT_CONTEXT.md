# STAR2.0 project context

Last reviewed: **15 October 2026, Asia/Calcutta**. Code baseline: `e34147f0aeea8507881afa6e82fa8d0450f94c77`. This review adds documentation only; application behavior, tests, data, and the committed HTML were not intentionally changed.

## Resume in a new chat

1. Read this file, `CHANGES.md`, and `docs/REVIEW_2026-10-15.md`.
2. Inspect the working tree and recent history before editing; this snapshot will become stale as work progresses.
3. Read the modules and tests affected by the next request. The review is a map, not a substitute for checking changed code.
4. Treat the latest owner message as authoritative. No new detailed product vision or configurator specification has yet been supplied in this chat.

Repository: https://github.com/parikhyash240804-lang/STAR2.0, remote `origin`, branch `main`.
Working folder at review: `C:\Users\MI\OneDrive\Desktop\STAR2.0`.
Chat name requested by owner: `1_Configurator`. This is a chat label, not evidence that a particular UI or feature set was approved.

## Product description

STAR2.0 is a Python portfolio-construction and risk-analysis engine for Indian equities, rebuilding Stardust 1.x. The management-facing concept is an offline decision brief answering three questions: approve the rebalance, check limits, and add/withdraw capital. The existing design also calls for a separate detailed Excel analyst archive.

The implemented product surface is a generated HTML file, not an interactive application for configuring or running portfolios. The core is substantial and tested. The current demonstration uses an equilibrium prior with no stock-selection signals, so allocation differences reflect risk constraints and trading controls rather than a measured view about winners. Return outperformance is not established.

## Actual architecture

```text
Core types / asset alignment / trading calendar / horizons
    -> settings schema and Excel input helpers
    -> price providers / quality / cache / dated rates / benchmark
    -> covariance, volatility, equilibrium, BL views, regime estimates
    -> constrained optimisation / holdings selection / baselines / stability
    -> risk and forward paths / stress / backtest and attribution
    -> guidance records -> embedded JSON + HTML/SVG/JavaScript brief
```

`tools/make_brief.py` manually assembles much of this chain. There is no shared configuration-driven pipeline, `star.app`, `star.pipeline`, signal-generation package, or new Excel report package. The package docstrings describe a broader intended layering; they are not an inventory of existing modules or an enforced architecture guarantee.

### Key modules and data contracts

| Area | Files and responsibilities |
|---|---|
| Core | `star/core/types.py`: `ReturnEstimate`, `CovarianceEstimate`, `Provenance`, excess/total, simple/log, daily/annual conventions. Log/simple expectation conversions require variance. Excess/total conversion is multiplicative in simple space. `align.py`: canonical label order and decorator. `calendar.py`: observed sessions and measured annualisation. `horizons.py`: standard and arbitrary trading-day horizons. |
| Configuration | `star/config/schema.py`: frozen dataclasses for run/data/returns/forecast/mandate/costs/backtest; collects validation issues. `excel_input.py`: stable hidden keys, dropdowns, input workbook creation/readback. These are not connected to the demo command. |
| Providers | `star/data/provider.py`: protocols, `PricePanel`, thread-local `GUARD`. `fixture.py`: committed parquet. `yahoo.py`: live yfinance adapter and current-only fundamentals. `view.py`: physically truncated `DataView`. `cache.py`: parquet + SHA-256 manifest. |
| Data checks | `quality.py`: inclusion/exclusion reasons, rupee-liquidity screen, conservative artifact repair. `universe.py`: as-of screening; cannot recover delisted history. `benchmark.py`: price index plus a dividend accrual calibrated against NIFTYBEES, with explicit fallback. |
| Dated rates/factors | `riskfree.py`: annual effective rate curve, CSV conversions and backward-looking lookup. `iima.py`: Indian Carhart four factors and T-bill rate recovery. Factor loading/attribution regression is not implemented in the current engine. |
| Risk estimates | `star/estimators/covariance.py`: HAR volatility by default, Yang-Zhang fallback, long-window standardised Ledoit-Wolf correlation, nearest-correlation repair. `har.py`: pooled range-volatility regression with completed forward training targets. `garch.py`: GJR-GARCH-t forecasts, fallback, cache, horizon covariance. |
| Expected returns | `equilibrium.py`: reference weights and derived risk aversion. `black_litterman.py`: relative/absolute views, Idzorek confidence, posterior return and covariance. `views.py`: IC measurement, Grinold scaling and basket construction; consumes supplied scores but generates no signals itself. |
| Regimes | `estimators/regime.py`: deterministic two-state Student-t HMM, filtered probabilities, expected stress duration, India VIX bands and disagreement. Historical filtered timelines still use parameters fitted on the whole displayed window; they are not historical live forecasts. |
| Construction | `optimize/qp.py`: SciPy trust-constr/SLSQP constrained solves and comparison. `constraints.py`: budget, box, sector, optional liquidity/conviction/turnover limits. `mean_variance.py`: max utility, min variance, target-return portfolio, turnover penalty/band/smoothing. |
| Other allocation functions | `cardinality.py`: heuristic holdings selection + re-solve, not an exact combinatorial optimum. `frontier.py`: independently solved frontier points. `baselines.py`: 1/N, ERC, HRP (baselines ignore mandate caps). `stability.py`: weight dispersion under uncertain returns. |
| Forward forecasts | `forecast/paths.py`: stationary bootstrap, regime-weighted resampling, multivariate-t reference, asset-level filtered historical generator. `engine.py`: chunked simulation, explicit drift, daily term structure, quantiles, ES, probabilities and path drawdowns. The default demo uses ordinary block bootstrap, not the planned regime-switching FHS chain. |
| Risk | `risk/metrics.py`: horizon samples, empirical VaR/ES, exact fractional tail weights, contributions, drawdown, beta/tracking error. `backtests.py`: Kupiec, Christoffersen, DQ, traffic light, optional ES test. `monitors.py`: correlation, diversification ratio and gold sleeve diagnostics. |
| Stress | `stress/historical.py`: dated replay with actual/proxied/missing coverage. `hypothetical.py`: conditional covariance-based propagation of holding/sector shocks. `reverse.py`: minimum-Mahalanobis loss scenario under a stated horizon. |
| Historical testing | `backtest/engine.py`: callable strategy, truncated views, daily weight drift, rebalance costs, warnings and histories. `costs.py`: itemised delivery-equity costs and tranches. `stats.py`: Sharpe uncertainty/difference, probabilistic/deflated Sharpe, PBO, MinTRL and bootstrap. `attribution.py`: Brinson and completed-position statistics. |
| Reporting | `report/guidance.py`: recommendations by decision/severity; action records require a falsifier. `report/brief.py`: JSON assembly and self-contained responsive HTML/SVG/JS, glossary, preset horizon/scenario toggles, charts and collapsed notes. No new output-workbook writer. |

## Current progress and boundaries

### Implemented and exercised

- Settings/schema and Excel input-template validation and round trips.
- Data handling, conventions, estimation, constraints, baselines, risk, forecasts, stress, backtesting, attribution and guidance have automated tests.
- All **455 collected tests passed** on the reviewed baseline; one SciPy solver warning. Both quick and full offline brief generation completed successfully.
- The committed `runs/brief.html` already contains a full demo with a backtest, four available stress scenarios, six visible horizons, and explanatory material.

### Demo-specific behavior

The script selects 37 usable names from the 39-name fixture, sets NAV to Rs. 1 crore, starts from equal weights in the first 12 names, and cycles through five placeholder sector names by position. It uses 10% stock and 30% sector caps, a trading penalty of 0.005, a 0.005 no-trade band, 20,000 bootstrap paths over 252 days, and seed 7 for the forward simulation.

The reviewed output proposed 24 holdings and about 18.46% one-way turnover, costing approximately Rs. 4,701 before market impact. This count satisfies the default maximum of 25 by coincidence of this run: the demo never calls the separate holdings-limit routine. It does not supply ADV-based liquidity caps or a hard turnover limit to the solve.

The full generated output displayed Sharpe about 0.793 with interval -0.414 to 2.000, deflated-Sharpe probability 0, and 2.905 years of tested returns. These are reproducible demo outputs, not certified performance evidence: benchmark, return-convention and evaluation problems remain. See the review before quoting them.

### Incomplete / not integrated

- Configurator/application UI, input loading into a real run, saved configurations, shared live/backtest strategy and run orchestration.
- New analyst workbook, archive/history persistence, complete order list, and per-field provenance export. Current prose refers to a workbook that is not generated.
- Signal generation and a measured stock-selection process. The BL and IC building blocks do exist.
- True sector metadata/current holdings/NAV inputs in the demo; complete mandate enforcement and honest unknown/not-checked states.
- Regime-conditioned GARCH/FHS orchestration; no current generator implements the full planned Markov regime-path switching chain.
- Connecting existing tranche, Brinson, trade-statistics, gold-sleeve and diversification-ratio functions into product outputs.
- Technical entry-timing overlay (RSI/MACD/Bollinger), current factor-attribution regressions, and Dimson beta.
- Dependency manifest/lockfile, standard package install/CLI entry point, CI and frozen application packaging.
- Paid point-in-time membership/delisted data. Interfaces support future providers; the bundled/static universe retains survivorship limitations.

## Data inventory

- Five stock OHLCV parquet frames: **1,241 sessions x 39 tickers**, 13 September 2021 to 11 September 2026, each with 924 missing cells. Adjusted closes are the primary series; the stock fixture has no raw-close panel.
- Quality screening excludes HYUNDAI and GATEWAY under the demo's full-window missingness rule, leaving 37 names. Do not reuse that full-window selection in historical decisions.
- Benchmark parquet: NSEI 1,236 rows (14 September 2021 onward), NIFTY Midcap 100 1,191 rows (ends 17 July 2026), NIFTYBEES 2,151 rows (1 January 2018 onward), GOLDBEES 1,240 rows, India VIX 1,231 rows. All end on 11 September 2026 except Midcap. Different calendars require explicit alignment.
- `data/reference/iima/2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv`: **8,004 rows**, 1 October 1993 to 31 December 2025; columns Date, SMB, HML, WML, MF, RF. One missing value per numeric column. It supplies four factors, not FF5. File-level survivorship adjustment does not fix the app's stock-universe survivorship bias.
- `IimaFactors.staleness_note()` exists. The demo does not load this dated curve and instead uses a fixed 6% forecast risk-free assumption; performance statistics are called with their zero-risk-free defaults.
- 106 compiled `.pyc` files and five pytest-cache files were already tracked. They were inventoried, not treated as executable source or current test evidence.

## Legacy and development history

`legacy/stardust_v1` mechanically splits the original 6,161-line `run_stardust` monolith into 11 stages, sharing a context dictionary. Execution order: M0 config -> M1 data -> M2A factor covariance -> M2B signals -> M2C BL/method selection -> M3 optimise -> M4 frontier -> M5 simulate -> M6 risk/stress -> M8 backtest -> M7 Excel.

It intentionally retains flawed methods documented in `docs/AUDIT_legacy.md`. It is a before/baseline implementation, not code to reuse indiscriminately. `star/` does not import it. In particular, its bootstrap installs a stdlib unverified-TLS context, sets numerical environment variables, and selects matplotlib Agg; its report copies to Downloads and opens a file unless the parity harness neutralises those effects.

`legacy/PARITY.md` records a September 2026 historical parity result: 908 stdout lines identical, 12 sheets/1,856 cells matching, four pixel-identical charts. **Not rerun here**: the original monolith and record/replay pickle are absent from this repository, and the harness requires a recording from the same calendar day. The original GUI is also absent.

Git history available before this review contains only three commits:

| Commit | Meaning |
|---|---|
| `71bb464` | Initial repository README |
| `64909a0` | Imported project source, docs, tests and data; added ignore rules |
| `e34147f` | Uploaded remaining caches/compiled files at explicit owner request |

Earlier implementation progress is described in imported documents and tests, not recoverable as detailed Git commits. Their P1/P7/P8/P9 phase references are historical context, not independently verified project milestones.

## Validation and environment

See `README.md` for commands and `docs/REVIEW_2026-10-15.md` for the evidence and open defects. Tested on Windows, Python 3.12.10, NumPy 2.5.0, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0, arch 8.0.0, openpyxl 3.1.5, pyarrow 25.0.1, pytest 9.1.1. yfinance 1.5.1 is installed; its live network path was not exercised.

Do not interpret historical `docs/VALIDATION.md` figures as newly verified. The scorecard utilities were read, and evaluation leakage was found. The numerical core has good test coverage, but a passing suite does not establish portfolio profitability, statistical calibration of the shipped demo, or a complete application.

## Continuing work

Record the owner's next vision/features in `CHANGES.md` with their source and status. Proposed sequencing, not yet an owner-approved roadmap: make the shared run/configuration contract concrete, resolve the high-priority integration/evaluation defects, then connect the configurator and outputs to that contract. Avoid building a UI that merely exposes settings the engine ignores.
