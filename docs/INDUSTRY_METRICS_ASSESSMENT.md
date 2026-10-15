# Assessment of `investment_metrics.md` for STAR2.0

**Source:** `Downloads/investment_metrics.md`, "AI-Powered Investment Platform — Complete Metrics Reference Document" (v1.0, June 2026).
**Assessed:** 2026-09-16, with free-data availability probed live against Yahoo Finance for NSE symbols.

The document covers a whole investment platform: single-company valuation (buy side), analyst research (sell side) and portfolio construction. STAR2.0 is the portfolio-construction and risk engine, so the items fall into four groups.

## Data availability (probed)

| Series / field | Available free? |
|---|---|
| India VIX `^INDIAVIX`, GOLDBEES, NIFTYBEES, NIFTY 500 `^CRSLDX`, DXY, US 10Y | Yes — 5 clean years, no gaps |
| Company statements (income, balance sheet, cash flow) | Yes — 5 annual, 5 quarterly; current restated values only |
| Analyst price targets, rating distribution, EPS revisions/trend, estimate range | Yes — today's snapshot only (not point-in-time) |
| Short interest, days to cover | No (`None`) |
| Institutional holders | Empty for NSE |
| Insider transactions | Sparse |
| Options / implied volatility history | No |
| Yahoo `beta` field | Present but unreliable (Reliance shown as 0.15) |

## 1. Already in the STAR2.0 plan

Sharpe, Sortino, Calmar, maximum drawdown; VaR and expected shortfall; efficient frontier (true, not random-weight); Black-Litterman; risk parity; resampled frontier; Fama-French 5 + momentum exposure; GARCH volatility; crisis correlation; beta; sector caps; liquidity from average daily value traded; total-return prices; information ratio, tracking error, alpha; probability of loss.

## 2. Worth adding (free data verified)

| Suggestion | Value | Caveat |
|---|---|---|
| India VIX regime bands | Model-free calm/elevated/stress gauge beside the Markov model | Percentiles of India VIX's own history, not US thresholds 25/40 |
| Rolling 60-day correlation monitor, diversification ratio | Early warning that diversification is disappearing | — |
| Gold allocation impact (GOLDBEES) | Marginal risk/return effect of a 5/10/15% gold sleeve | — |
| EPS revision momentum, estimate dispersion, analyst target upside, rating distribution | Street-consensus views for Black-Litterman | Live only (not backtestable); analyst targets are optimistic (Bradshaw, Brown & Huang 2013), so low measured confidence |
| Earnings-quality flags: accruals (Sloan 1996), FCF conversion, Debt/EBITDA, interest coverage, ROIC | Red flags on holdings | Not point-in-time |
| Brinson attribution | Stock selection vs sector allocation | No free NIFTY constituent weights: attribute against equal weight or a user-supplied weights file |
| Hit rate, profit factor, worst drawdown while held | Trade-level backtest honesty | — |
| Tranche execution schedule | Split trades over weeks with cost per tranche | — |
| RSI, MACD, Bollinger, moving averages | Entry-timing hints on the trades page | Overlay only; predictive value measured, not assumed |
| Conviction sets maximum position size | Weak evidence ⇒ smaller positions | — |

## 3. Right idea, different implementation

| Document says | STAR2.0 does | Why |
|---|---|---|
| Monte Carlo from triangular/lognormal draws with a covariance matrix | Regime-conditional filtered historical simulation | That approach is what made the legacy fat tails cosmetic (audit M5-01/02) |
| Reject if probability of loss > 35% | Keep the rule, with the drift assumption shown | Otherwise the number restates the assumed return (M5-03) |
| Target Sharpe > 1.0 | Report Sharpe with a confidence interval and Deflated Sharpe | Rarely sustained long-only in NSE; targeting it invites overfitting |
| Maximum drawdown as a hard optimiser constraint | Conditional drawdown-at-risk optimisation over scenarios | A single-period optimiser cannot constrain a path-dependent drawdown |
| Beta from 5-year monthly regression | Daily beta with Dimson thin-trading correction | Better for Indian mid/small caps; never use Yahoo's beta |
| Half-Kelly position sizing | Diagnostic only | Kelly over-bets when edges are estimated |
| DCF margin of safety | Lightweight reverse-DCF sanity flag (research mode) | A real DCF needs analyst judgement per company |

## 4. Not feasible or out of scope

- **Different product:** three-statement models, DCF/LBO/sum-of-parts/residual income, archetype driver trees, guidance tracking and surprise-triggered remodelling (single-company valuation — could feed STAR2.0 as views if built separately); news/event NLP classification.
- **No free NSE data:** implied volatility and option-based tail hedges, short interest, institutional ownership trends, insider clusters, ESG, alternative data.
- **US-centric macro:** ISM, US PMI, TIPS real yields. India yield curve and PMI can be supplied as CSV files.

## Status

Owner's selection (2026-09-19): **market & risk monitors** (India VIX bands, correlation monitor, diversification ratio, gold sleeve) and **execution & attribution** (tranche schedule, technical entry-timing overlay, Brinson attribution, hit rate / profit factor, conviction-based caps) are in the build plan. **Street-consensus signals** and **fundamental red flags** are not being built.
