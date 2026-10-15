# 03 · Volatility and covariance estimation

## Why it matters for STAR2.0

Every risk number, every optimised weight and every scenario depends on the covariance matrix. Two problems dominate: volatility changes over time (clustering), and a sample covariance of ~40 stocks estimated from a few years of data is noisy enough that an optimiser will exploit its errors. Expected returns are even worse — their estimation error is roughly an order of magnitude larger — which is why STAR2.0 never feeds raw sample means to an optimiser.

The legacy engine's default path ran max-Sharpe on unshrunk EWMA means (audit M2-03/M3-01, high, default run), used a factor covariance whose momentum factor was identically zero (M2-02), switched covariance method depending on the data (irreproducible), and used close-to-close volatility only.

## Key papers

| Paper | Takeaway for STAR2.0 |
|---|---|
| J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th ed. | EWMA variance σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}; λ = 0.94 for daily risk. Simple, responsive, no fitting. |
| Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices. *Journal of Business* 73(3), 477–492. https://doi.org/10.1086/209650 | Range-based estimator using OHLC; drift-independent, handles opening jumps, several times more efficient than close-to-close. |
| Engle, R. F., & Bollerslev, T. (1986). Modelling the persistence of conditional variances. *Econometric Reviews* 5(1), 1–50. https://doi.org/10.1080/07474938608800095 | GARCH-family modelling of conditional volatility and its persistence. |
| Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance* 48(5), 1779–1801. https://doi.org/10.1111/j.1540-6261.1993.tb05128.x | GJR-GARCH: negative returns raise volatility more than positive ones (leverage effect). |
| Engle, R. (2002). Dynamic conditional correlation. *Journal of Business & Economic Statistics* 20(3), 339–350. https://doi.org/10.1198/073500102288618487 | DCC: time-varying correlations. Research mode here — heavy for 40 names. |
| Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4 | Linear shrinkage toward a structured target with an analytically optimal intensity. |
| Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics* 48(5), 3043–3065. https://doi.org/10.1214/19-AOS1921 | Eigenvalue-by-eigenvalue shrinkage; better when n/T is large. |
| Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters* 83(7), 1467–1470. https://doi.org/10.1103/PhysRevLett.83.1467 | Most eigenvalues of a sample correlation matrix are indistinguishable from noise (Marchenko–Pastur). |
| Higham, N. J. (2002). Computing the nearest correlation matrix — a problem from finance. *IMA Journal of Numerical Analysis* 22(3), 329–343. https://doi.org/10.1093/imanum/22.3.329 | Repairs a non-PSD matrix to the nearest valid correlation matrix. |
| Dimson, E. (1979). Risk measurement when shares are subject to infrequent trading. *Journal of Financial Economics* 7(2), 197–226. https://doi.org/10.1016/0304-405X(79)90013-8 | Sum lagged and leading betas to correct thin-trading bias — relevant for Indian mid/small caps. |
| Michaud, R. O. (1989). The Markowitz optimization enigma: Is 'optimized' optimal? *Financial Analysts Journal* 45(1), 31–42. https://doi.org/10.2469/faj.v45.n1.31 | Optimisers are "estimation-error maximisers". |
| DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies* 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075 | Sample-based mean-variance rarely beats 1/N out of sample; 1/N is a mandatory benchmark. |

## What STAR2.0 implements

**Volatility (per stock):** Yang–Zhang range estimator on OHLC (Yahoo supplies OHLC for every NSE stock; verified), smoothed with an EWMA:

- σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1−k)·σ²_Rogers–Satchell, with k = 0.34 / (1.34 + (m+1)/(m−1)).
- Default half-life: 60 sessions for the risk model; the forward engine uses GJR-GARCH(1,1) with Student-t innovations (via the `arch` package), fitted on the as-of window.

**Correlation:** Ledoit–Wolf shrinkage applied to the correlation of volatility-standardised returns (not to the raw covariance), over a longer window (default 1,260 sessions); switch to analytical nonlinear shrinkage when n/T > 0.30.

**Covariance:** Σ = D C D, with D = diag(σ_YZ) and C the shrunk correlation; repaired with Higham's nearest-correlation algorithm if needed. Never silently switches method based on the data.

**Beta:** daily regression with Dimson's lead/lag correction; Yahoo's `beta` field is not used (it reports 0.15 for Reliance, which is not credible).

**Factor model:** kept for **risk attribution** only — the Carhart four-factor model (market, size, value, momentum) from the IIMA Indian factor library, which publishes four factors rather than Fama-French 5 — never as a silently substituted risk matrix.

**Every estimate** is a `CovarianceEstimate` carrying periodicity, method and provenance.

## Pitfalls — and where the legacy went wrong

- Sample means into max-Sharpe (M2-03/M3-01): weights follow noise. STAR2.0's optimiser is fed only equilibrium/Black-Litterman returns.
- A factor built by averaging cross-sectional z-scores is identically zero (M2-02).
- Data-dependent method switching (fall back to Ledoit-Wolf when many R² are low) makes results irreproducible.
- EWMA half-life is a bias–variance trade-off: short half-lives react fast but are noisy; the risk model and the optimiser need different ones.
- Forward-filled prices create zero-return days that bias volatility down (M1-05).
- Yahoo data contains broken corporate-action adjustments (NIFTYBEES, December 2019: −89.9% then +897%); unrepaired, one such day dominates a volatility estimate. STAR2.0 repairs only physically implausible artifacts and keeps genuine crashes.

## Tests

1. **Positive semi-definiteness** after estimation and after repair; repair is a no-op on an already-valid matrix.
2. **Ledoit–Wolf cross-check:** matches `sklearn.covariance.LedoitWolf` to 1e-10 on identical input.
3. **Yang–Zhang efficiency:** on simulated geometric Brownian motion with known σ, its sampling variance is several times smaller than close-to-close.
4. **GARCH recovery:** simulate 5,000 days from known GJR-GARCH-t parameters; each estimate within 3 standard errors, persistence within 0.03.
5. **Estimator horse race:** out-of-sample realised volatility of the minimum-variance portfolio — sample vs Ledoit–Wolf vs EWMA×shrunk correlation — on the NSE fixture; the shrunk estimator must not lose to sample covariance.
6. **Frobenius error** against a known Σ on simulated data: nonlinear ≤ linear shrinkage ≤ sample when n/T is large.
