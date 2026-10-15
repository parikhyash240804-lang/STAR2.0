# 04 · Backtesting: methodology and statistical validity

## Why it matters for STAR2.0

A backtest is evidence only if it (a) runs the strategy that will actually be used, (b) uses only information available at each decision date, (c) charges realistic costs against a fair benchmark, and (d) says how uncertain its result is. The legacy walk-forward failed on all four (verified): it traded today's surviving tickers at every past date (M8-02, critical), ran a close cousin of the live model (M8-01), benchmarked a dividend-adjusted portfolio against a price index (M8-07, ~1.2–1.4%/yr handicap), halved costs after the first rebalance (M8-05), reported a VaR breach rate that was always NaN (M8-03), and gave no confidence interval on a ~3-year sample where each per-window Sharpe had a standard error of about 2 (M8-06).

## Key papers

| Paper | Takeaway for STAR2.0 |
|---|---|
| Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36–52. https://doi.org/10.2469/faj.v58.n4.2453 | Sharpe ratios have large standard errors, and serial correlation distorts annualisation. Always report a confidence interval. |
| Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance* 15(5), 850–859. https://doi.org/10.1016/j.jempfin.2008.03.002 | Studentised bootstrap test for the *difference* of two Sharpe ratios — the right test for "did we beat 1/N?". |
| Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2), 3–44. https://doi.org/10.21314/JOR.2012.255 | Minimum Track Record Length: how many observations are needed before an observed Sharpe is significant. |
| Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management* 40(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094 | Corrects the Sharpe ratio for the number of configurations tried and for non-normal returns. |
| Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20(4), 39–69. https://doi.org/10.21314/JCF.2016.322 | Combinatorially symmetric cross-validation (CSCV) estimates how likely the chosen configuration is overfit. |
| Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13–28. https://doi.org/10.3905/jpm.2015.42.1.013 | Haircut Sharpe ratios for multiple testing. |
| Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies* 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059 | Hundreds of "discovered" factors; a t-statistic of 2 is not enough. |
| White, H. (2000). A reality check for data snooping. *Econometrica* 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152 | Bootstrap test for the best of many strategies. |
| Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics* 23(4), 365–380. https://doi.org/10.1198/073500105000000063 | A more powerful refinement of the reality check. |
| Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870 | Resampling with random block lengths; confidence intervals that respect serial dependence. |
| Arnott, R., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the era of machine learning. *Journal of Financial Data Science* 1(1), 64–74. https://doi.org/10.3905/jfds.2019.1.1.064 | A checklist: pre-register the hypothesis, count trials, avoid look-ahead, report everything tried. |
| Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct estimation of equity market impact. *Risk* 18(7), 57–62. | Market impact grows with the square root of participation. |
| Brinson, G. P., Hood, L. R., & Beebower, G. L. (1986). Determinants of portfolio performance. *Financial Analysts Journal* 42(4), 39–44. https://doi.org/10.2469/faj.v42.n4.39 | Attribution into allocation, selection and interaction. |

## What STAR2.0 implements

**One strategy object.** `walk_forward(strategy, …)` calls exactly the `target_weights(as_of, data)` used by the live run, so the two cannot drift apart (closes M8-01).

**Point-in-time data.** At each rebalance the provider guard is armed at the rebalance date; any request for later data raises `LookAheadError` (the "look-ahead canary"). The universe at each date contains only stocks with enough history *as of that date* (closes the full-sample filter part of M8-02). Survivorship itself cannot be fixed with free Yahoo data (no delisted-stock or index-membership history), so the report states it plainly; a paid point-in-time vendor plugs in through the provider interface.

**Accounting.** Weights drift daily with prices between rebalances; turnover and costs are charged against drifted holdings; the risk-free rate is a dated series, not today's constant.

**Indian costs, per leg (editable defaults):** brokerage min(₹20, 0.03%); STT 0.10% on delivery, both sides; NSE transaction charge ≈ 0.00297%; SEBI fee 0.0001%; stamp duty 0.015% on buys only; GST 18% on brokerage and exchange/SEBI fees; DP charge ≈ ₹13–18 per scrip per sell day; plus square-root impact k·σ_daily·√(Q/ADV).

**Benchmarks:** NIFTY total return (NIFTY price index plus a dividend accrual calibrated to NIFTYBEES — see `star/data/benchmark.py`), equal weight (1/N), minimum variance, and "hold the previous weights" (the honest comparison for a rebalance decision).

**Statistics shown with every backtest:** CAGR, volatility, Sharpe with Lo (2002) confidence interval, Ledoit–Wolf (2008) Sharpe-difference test against 1/N and the benchmark, Deflated Sharpe, PBO via CSCV, Minimum Track Record Length, stationary-bootstrap intervals for CAGR and maximum drawdown, Sortino (with a true downside deviation), Calmar, tracking error, information ratio, turnover, itemised cost drag, hit rate and profit factor.

**Fundamentals mode:** fundamental signals from Yahoo are restated current values, never point-in-time, so the default backtest is price-only; running both modes and reporting the difference gives an *upper bound* on what fundamentals could add, not an estimate.

## Pitfalls — and where the legacy went wrong

- Backtesting today's survivors (M8-02) inflates results by an amount that cannot be measured with free data — it must be disclosed.
- Comparing the optimiser's in-sample objective with realised out-of-sample returns and calling it "the same model" (M8-04).
- Charging one side of each rebalance (M8-05) and ignoring drift between rebalances (M8-09).
- A price-index benchmark against total-return holdings (M8-07).
- No inference (M8-06): on ~13 quarterly windows almost nothing is statistically established, and the report must say so.
- Every tuned knob is a trial; the Deflated Sharpe needs an honest trial count.

## Tests

1. **Look-ahead canary never trips** across a full walk-forward, and a deliberately planted future read makes it trip.
2. **Same callable** is used live and in the backtest (identity test).
3. **Weights drift** between rebalances, and turnover is measured against drifted weights.
4. **Cost arithmetic to the paisa:** a ₹10,00,000 buy costs brokerage ₹20.00, STT ₹1,000.00, NSE ₹29.70, SEBI ₹1.00, stamp ₹150.00, GST ₹9.13 — total ₹1,209.83 (12.10 bps); the sell leg omits stamp duty and adds the DP charge.
5. **Power:** a synthetic signal with known IC = 0.10 must produce a statistically significant Sharpe difference.
6. **Size:** pure-noise signals must be "significant" at close to the nominal 5% rate across seeds.
7. **Overfitting detector works:** a deliberately overfit configuration search gives PBO > 0.5.
8. **Lo (2002) standard error:** for i.i.d. normal returns it matches the empirical spread of 20,000 simulated Sharpe ratios within 5%.
