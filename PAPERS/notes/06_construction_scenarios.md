# 06 · Portfolio construction and forward scenarios

## Why it matters for STAR2.0

Two jobs sit here. **Construction** turns estimates into weights that survive estimation error, respect a mandate and do not churn. **Scenarios** say what could happen to those weights over the horizon management chooses (1 day to 1 year, or any custom length). The legacy engine's "efficient frontier" was a cloud of random weights that cannot trace a frontier (M4-01), `MAX_ASSETS` was never enforced while every report said "Sparse" (M3-06), and its Monte Carlo drifted at the optimiser's own in-sample return, so "probability of loss" largely restated that input (M5-03, critical, default run); its i.i.d. daily shocks let the central limit theorem erase fat tails at one year (M5-02).

## Key papers

| Paper | Takeaway for STAR2.0 |
|---|---|
| Markowitz, H. (1952). Portfolio selection. *Journal of Finance* 7(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x | Mean-variance framework; the frontier is traced by solving an optimisation, not by sampling random weights. |
| Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management* 36(4), 60–70. https://doi.org/10.3905/jpm.2010.36.4.060 | Risk parity: each holding contributes equally to risk; needs no return forecast. |
| López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management* 42(4), 59–69. https://doi.org/10.3905/jpm.2016.42.4.059 | Hierarchical risk parity: robust allocation from a clustered correlation structure, no matrix inversion. |
| Choueifaty, Y., & Coignard, Y. (2008). Toward maximum diversification. *Journal of Portfolio Management* 35(1), 40–51. https://doi.org/10.3905/JPM.2008.35.1.40 | Diversification ratio = weighted average volatility / portfolio volatility; > 1 means real diversification. |
| Michaud, R. O. (1998). *Efficient Asset Management.* Harvard Business School Press. | Resampled frontier: average weights over perturbed inputs; shows that "the" frontier is a band. |
| Scherer, B. (2002). Portfolio resampling: Review and critique. *Financial Analysts Journal* 58(6), 98–109. https://doi.org/10.2469/faj.v58.n6.2489 | Resampling reduces but does not remove estimation error and has no decision-theoretic basis; use it to illustrate uncertainty, not as the optimiser. |
| Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance* 68(6), 2309–2340. https://doi.org/10.1111/jofi.12080 | Trade partially toward an "aim" portfolio; the formal version of turnover control. Research mode. |
| MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010). Long-term capital growth: the good and bad properties of the Kelly and fractional Kelly capital growth criteria. *Quantitative Finance* 10(7), 681–687. https://doi.org/10.1080/14697688.2010.506108 | Kelly sizing is optimal only with known edges; with estimated edges it over-bets. Diagnostic only in STAR2.0. |
| Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870 | Resample historical blocks of random length: keeps fat tails, cross-sectional dependence and short-run autocorrelation without a distributional assumption. |
| Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets* 19(5), 583–602. | Filtered historical simulation: historical standardised residuals rescaled by today's volatility. |
| Britton, E., Fisher, P., & Whitley, J. (1998). The Inflation Report projections: understanding the fan chart. *Bank of England Quarterly Bulletin*, February 1998, 30–37. | The fan chart: communicate a forecast as a widening band of probabilities, not a point. |

## What STAR2.0 implements

**Construction.**

- Primary: maximise μᵀw − (λ/2) wᵀΣw − c·‖w − w_prev‖₁ subject to the mandate (per-stock cap, sector caps, liquidity cap as a multiple of average daily value traded, turnover budget, long-only). μ is always the equilibrium/Black-Litterman estimate, never sample means. λ is set from a volatility target. Solved with `cvxpy` (numpy projected-gradient fallback for the packaged app).
- Reported alongside: minimum variance, equal risk contribution, hierarchical risk parity, and 1/N. The spread between them tells management how much of the allocation is a bet on the return forecast.
- **Cardinality done properly:** greedy forward selection to `MAX_ASSETS`, re-optimise on the chosen names, and report the objective gap versus the unconstrained solution.
- **True efficient frontier:** minimise wᵀΣw subject to μᵀw = target over ~40 targets, with the resampled band as an overlay.
- **Stability in weight space, not by smoothing returns:** ℓ1 turnover penalty, a no-trade band (trade only names that drifted beyond a threshold), and optional exponential smoothing w = (1−φ)·w_prev + φ·w_target. The risk/return cost of that stability is reported.
- Diagnostics (not used for sizing): diversification ratio, fractional-Kelly exposure, marginal effect of a 5/10/15% gold sleeve (GOLDBEES).

**Forward scenarios (replaces the Monte Carlo).**

- Default engine: regime-conditional filtered historical simulation. Fit GJR-GARCH-t per stock; resample standardised residual **vectors by date** (all stocks from the same day), which preserves the real cross-sectional dependence and tail co-movement with no copula to estimate; propagate each GARCH recursion along the path so volatility clusters; draw the regime path from the Markov model starting at today's filtered probabilities.
- Robustness engine: stationary block bootstrap (mean block ≈ 10–20 sessions).
- Reference only: a proper multivariate Student-t (one shared mixing variable), for tests.
- **Drift is an explicit, visible assumption:** default = risk-free plus the equilibrium premium (not the optimiser's own forecast); alternatives shown in a sensitivity strip (zero, risk-free, equilibrium, views).
- **One simulation serves every horizon:** portfolio paths are stored per day up to H_max (default 252, capped at 756 with a warning), so a 2-day, 7-day, 1-month or any custom horizon is a lookup, not a re-run. The benchmark is simulated jointly on the same resampled dates, so "probability of underperforming NIFTY" is meaningful.
- Memory: paths generated in chunks and projected to portfolio space immediately; only (H_max+1) × N float32 values are kept (≈ 20 MB at 252 × 20,000), versus the legacy's 2–4 GB.

## Pitfalls — and where the legacy went wrong

- Random-weight clouds do not trace a frontier; they undersample exactly the edge portfolios (M4-01).
- Printing "Sparse (MAX_ASSETS=25)" without enforcing it (M3-06).
- Simulating at the optimiser's own in-sample return turns P(loss) into an echo of an assumption (M5-03).
- Independent t marginals mixed by Cholesky are not a multivariate t, and i.i.d. days erase fat tails by one year (M5-01, M5-02).
- Simulating the benchmark independently of the portfolio makes relative probabilities meaningless.
- Kelly sizing on estimated edges over-bets.

## Tests

1. **Constraints hold** for every solution (caps, sectors, budget, long-only) and weights sum to 1.
2. **Permutation invariance:** reordering assets reorders the weights identically.
3. **Frontier dominance:** no random-weight portfolio lies above the computed frontier.
4. **Cardinality enforced:** the number of non-zero weights ≤ `MAX_ASSETS`.
5. **Turnover monotonicity:** turnover falls as the penalty, band width or smoothing increases.
6. **Fat tails survive:** 252-day kurtosis of simulated portfolio returns exceeds the i.i.d.-normal benchmark on GARCH-driven data.
7. **Closed-form check:** with i.i.d. normal inputs the simulated quantiles match the analytic ones within Monte Carlo error.
8. **Horizon identity:** the fan chart's 5th percentile at day h equals the h-day quantile exactly.
9. **Memory ceiling:** peak memory stays under budget at n = 60, N = 20,000, H = 1,260.
