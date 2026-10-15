# 05 · Risk measurement, VaR/ES validation and stress testing

## Why it matters for STAR2.0

Management needs three answers: how much could we lose over the horizon we care about, is that estimate trustworthy, and what happens in a crisis like the ones India has already had. The legacy engine got each one wrong in a verified way: a one-year VaR was printed as "1-day" and multiplied by √252, showing a 10% loss as ~159% of capital (M6-02, critical, default run); its VaR "calibration" breach rate was always NaN (M8-03); beta-scaled stress applied a uniform ×1.3 with a −80% clip (M6-04); replayed scenarios displayed a market shock they did not use (M6-05).

## Key papers

| Paper | Takeaway for STAR2.0 |
|---|---|
| Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance* 9(3), 203–228. https://doi.org/10.1111/1467-9965.00068 | VaR is not subadditive (diversification can appear to increase it); expected shortfall is coherent. |
| Acerbi, C., & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance* 26(7), 1487–1503. https://doi.org/10.1016/S0378-4266(02)00283-2 | The correct definition and estimation of expected shortfall (ES). |
| Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives* 3(2), 73–84. https://doi.org/10.3905/jod.1995.407942 | Proportion-of-failures test: are VaR breaches as frequent as they should be? |
| Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review* 39(4), 841–862. https://doi.org/10.2307/2527341 | Tests whether breaches cluster (independence) and joint conditional coverage. |
| Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk by regression quantiles. *Journal of Business & Economic Statistics* 22(4), 367–381. https://doi.org/10.1198/073500104000000370 | Dynamic quantile test for breach predictability. |
| Acerbi, C., & Székely, B. (2014). Backtesting expected shortfall. *Risk*, December 2014. | ES is not elicitable, but it can still be backtested; practical test statistics. |
| Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach.* https://www.bis.org/publ/bcbsc223.htm | The green/yellow/red traffic light for VaR breaches. |
| Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets* 19(5), 583–602. | Filtered historical simulation: rescale historical residuals by current volatility. |
| McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance* 7(3–4), 271–300. https://doi.org/10.1016/S0927-5398(00)00012-8 | GARCH filter plus extreme value theory for the far tail. |
| Tasche, D. (2007). Capital allocation to business units and sub-portfolios: the Euler principle. arXiv:0708.2542. https://arxiv.org/abs/0708.2542 | Euler allocation: risk contributions that add up exactly to total risk. |
| Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). Drawdown measure in portfolio optimization. *International Journal of Theoretical and Applied Finance* 8(1), 13–58. https://doi.org/10.1142/S0219024905002767 | Conditional drawdown-at-risk, and how to optimise against it. |
| Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3), 21–41. https://doi.org/10.21314/JOR.2000.038 | ES/CVaR optimisation as a linear program over scenarios. |
| Basel Committee on Banking Supervision (2018). *Stress testing principles.* https://www.bis.org/bcbs/publ/d450.htm | Scenarios must be severe but plausible, cover the portfolio's actual vulnerabilities, and be documented. |
| Glasserman, P., Kang, C., & Kang, W. (2015). Stress scenario selection by empirical likelihood. *Quantitative Finance* 15 (2015); volume/pages to confirm. | Choosing the most plausible scenario that produces a given loss (reverse stress testing). |
| Breuer, T., Jandačka, M., Rheinberger, K., & Summer, M. (2009). How to find plausible, severe, and useful stress scenarios. *International Journal of Central Banking* 5(3) (2009). | Plausibility measured by Mahalanobis distance. |

## What STAR2.0 implements

**Horizon-aware VaR and ES, never √t-scaled.** Every figure is read from the simulated distribution *at that horizon*, so a 1-day, 1-month and 1-year number come from the same scenario set and can never contradict each other (closes M6-02). ES at 97.5% is the headline (coherent; the Basel FRTB choice); VaR at 95% and 99% is shown alongside for familiarity. The API cannot accept a 1-day number and a horizon, so the legacy mistake is unrepresentable.

**Risk contributions (Euler):** marginal MCR_i = (Σw)_i / σ_p, component CCR_i = w_i · MCR_i, with Σ CCR_i = σ_p exactly; ES contributions by conditional tail expectation. Drawdown-at-risk and conditional drawdown-at-risk from the simulated paths.

**Validation of the risk model itself**, on out-of-sample daily returns against the forecast made the day before: Kupiec, Christoffersen independence and conditional coverage, the dynamic quantile test, Basel traffic light (≤4 green, 5–9 yellow, ≥10 red breaches per 250 days at 99%), and an Acerbi–Székely ES test. Shown in the brief as a model-health light.

**Stress testing, three kinds:**

1. **Historical replay** of named Indian episodes using each stock's *actual* returns over the window: 2008 global financial crisis (Jan–Oct 2008), 2013 taper tantrum (May–Aug 2013), 2016 demonetisation (Nov–Dec 2016), 2018 IL&FS/NBFC stress (Aug–Oct 2018), March 2020 COVID, the 2021–22 rate shock, and the 4 June 2024 election-result day. A replay is labelled "replayed" only if the window's data covers ≥ 90% of current holdings; otherwise the missing stocks are propagated through the factor model and the scenario is labelled "propagated", with coverage printed.
2. **Hypothetical factor shocks** propagated by conditional expectation: E[r \| f] = Σ_rf Σ_ff⁻¹ f, with the conditional covariance Σ_rr − Σ_rf Σ_ff⁻¹ Σ_fr giving a band around the loss. Crisis betas come from the stress-state covariance of the regime model — estimated, not a hard-coded ×1.3. No clip.
3. **Reverse stress:** the most plausible factor shock that produces a target loss L. With factor exposures b, minimising fᵀ Σ_ff⁻¹ f subject to bᵀf = −L gives f* = −L · Σ_ff b / (bᵀ Σ_ff b). Plausibility is reported as a Mahalanobis distance converted to an approximate "1-in-X-year" frequency.

## Pitfalls — and where the legacy went wrong

- √t scaling assumes i.i.d. returns and ignores drift; with volatility clustering it misstates both short- and long-horizon tails (M6-01, M6-02).
- A 95% one-day historical VaR from a calm five-year window with in-sample weights says little about crisis losses (M6-03).
- Stress = β × market drop × 1.3, clipped, is a heuristic whose clip decides the answer for high-beta holdings (M6-04).
- Displaying a scenario shock that the calculation does not use misleads (M6-05).
- A VaR that is never backtested cannot be trusted; one whose backtest silently returns NaN is worse (M8-03).
- Hard-coded narrative claims ("HONEST RISK SUMMARY", M6-07): STAR2.0 forbids numeric literals in report templates, enforced by a test.

## Tests

1. **Kupiec critical values:** at T = 250 and p = 0.01 the likelihood-ratio statistic reproduces the published non-rejection band; Basel zone boundaries are exact.
2. **Christoffersen independence:** a clustered breach sequence is rejected; an i.i.d. sequence with the same count is not.
3. **Both directions:** a correctly specified model passes ~95% of the time; a normal model on GARCH data is rejected (power).
4. **Euler identity:** Σ CCR_i = σ_p to 1e-12; ES contributions sum to ES.
5. **Coherence properties:** ES ≥ VaR at every level; VaR(99%) ≥ VaR(95%).
6. **Horizon consistency:** the fan chart's 5th percentile at day h equals the h-day VaR input exactly.
7. **No clip:** no stress output equals exactly −80%.
8. **Reverse stress:** the closed form matches a numerical constrained minimisation.
