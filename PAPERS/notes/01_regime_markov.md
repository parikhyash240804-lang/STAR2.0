# 01 · Market regimes: Markov models, jump models and model-free gauges

## Why it matters for STAR2.0

Volatility and correlation are not constant: Indian equities spend long calm stretches punctuated by short stress episodes (2008, the 2013 taper tantrum, March 2020, the 2022 rate shock). A model estimated on "average" conditions understates risk exactly when it matters. The legacy engine had no regime model at all — its only "regime" was a five-bucket step function on a single 126-day midcap-minus-Nifty spread whose noise was as wide as its buckets (audit M0-07).

STAR2.0 uses regimes for three things only: conditioning the forward scenario engine, estimating crisis covariance for stress tests, and staging a rebalance. It does **not** switch the allocation on a regime call.

## Key papers

| Paper | Takeaway for STAR2.0 |
|---|---|
| Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2), 357–384. https://doi.org/10.2307/1912559 | The Markov-switching model and the filter that computes P(state \| data up to t). The filter is causal; this is what a backtest may use. |
| Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies* 15(4), 1137–1187. https://doi.org/10.1093/rfs/15.4.1137 | Correlations rise and returns fall in the high-volatility regime; regime-aware allocation helps mainly through risk, not return forecasting. |
| Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control* 31(11), 3503–3544. https://doi.org/10.1016/j.jedc.2006.12.004 | Regimes change optimal weights materially, but parameter uncertainty is large; more than a few states overfit. |
| Nystrup, P., Madsen, H., & Lindström, E. (2015). Regime-based versus static asset allocation: Letting the data speak. *Journal of Portfolio Management* 42(1), 103–109. https://doi.org/10.3905/jpm.2015.42.1.103 | Out-of-sample evidence that regime conditioning improves risk-adjusted outcomes, mainly by cutting drawdowns. |
| Nystrup, P., Lindström, E., & Madsen, H. (2020). Learning hidden Markov models with persistent states by penalizing jumps. *Expert Systems with Applications* 150, 113307. https://doi.org/10.1016/j.eswa.2020.113307 | Statistical jump model: clustering with a penalty on state switches. More persistent, fewer spurious flips than an HMM. |
| Aydınhan, A. O., Kolm, P. N., Mulvey, J. M., & Shu, Y. (2024). Identifying patterns in financial markets: extending the statistical jump model for regime identification. *Annals of Operations Research*. (continuous jump model; citation to be confirmed against the journal page) | Extends the jump model with continuous state probabilities; strong out-of-sample regime identification. |
| Kritzman, M., Page, S., & Turkington, D. (2012). Regime shifts: Implications for dynamic strategies. *Financial Analysts Journal* 68(3), 22–39. https://doi.org/10.2469/faj.v68.n3.3 | Practitioner framing: regimes are most useful for risk budgeting and defensive tilts. |
| Kritzman, M., & Li, Y. (2010). Skulls, financial turbulence, and risk management. *Financial Analysts Journal* 66(5), 30–41. https://doi.org/10.2469/faj.v66.n5.3 | Turbulence = Mahalanobis distance of today's returns from history; high turbulence precedes poor returns and higher risk. Model-free. |
| Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2011). Principal components as a measure of systemic risk. *Journal of Portfolio Management* 37(4), 112–126. https://doi.org/10.3905/jpm.2011.37.4.112 | Absorption ratio: the share of variance explained by the top principal components. Rising values mean diversification is disappearing. |

## What STAR2.0 implements

**Primary model — 2-state Student-t hidden Markov model** on the benchmark's daily log return, fitted by expectation–maximisation (Baum–Welch):

- Emissions: r_t \| s_t = k ~ Student-t(μ_k, σ_k, ν_k); states labelled deterministically by σ (state 0 = calm), which removes EM label switching.
- Transition matrix P with p_ij = P(s_{t+1}=j \| s_t=i). Expected duration of state i = 1 / (1 − p_ii) sessions.
- **Filtered** probabilities ξ_t = P(s_t \| r_1..r_t) via the Hamilton filter are the only probabilities used in any backtest or "current regime" statement.
- Defaults: K = 2 (K chosen by BIC and economic interpretability, never above 3), 20 EM restarts, occupancy guard (warn if a state holds < 10% of sessions).
- Cross-check: `statsmodels` `MarkovRegression(k_regimes=2, switching_variance=True)` on the same data.

**Robustness companions (shipped alongside, reported as a panel):**

- Statistical jump model: minimise Σ_t ℓ(x_t, θ_{s_t}) + λ · 1[s_t ≠ s_{t−1}] over state sequences (dynamic programming) and centroids (alternating), λ chosen by out-of-sample persistence.
- Turbulence: d_t = (r_t − μ)ᵀ Σ⁻¹ (r_t − μ), percentile against its own history.
- Absorption ratio: AR_t = Σ_{i≤k} λ_i / Σ_j λ_j on a rolling correlation matrix (k ≈ n/5).
- India VIX (`^INDIAVIX`, 5 years of clean daily data verified 2026-09-16) in percentile bands calibrated to its own history — **not** the US thresholds of 25/40 quoted in generic references.

When the gauges disagree, the brief reports the disagreement as uncertainty rather than picking one.

## Pitfalls — and where the legacy went wrong

- **Smoothed probabilities are look-ahead.** Kim-smoothed P(s_t \| r_1..r_T) uses future data. It is legitimate for describing history, never inside a backtest. Enforced by a lint test.
- **Regimes describe the past far better than they predict the next switch.** The one-step-ahead forecast is dominated by persistence p_ii. Do not ship a regime-triggered allocation switch.
- **Too many states overfit** on ~1,250 daily observations; a third state is usually an artefact.
- **Label switching** across EM restarts makes "state 1" meaningless unless states are ordered.
- Legacy M0-07: a step function on one noisy spread is not a regime model.

## Tests

1. **Recovery on simulated data:** simulate 3,000 days from a known 2-state model (σ_calm 0.8%, σ_stress 2.4%, p_00 0.98, p_11 0.94); require transition entries within ±0.03, per-state σ within 10%, expected durations within 15%, filtered classification accuracy > 85%.
2. **Smoothed ≥ filtered accuracy** on the same simulation (proves both exist) — plus an AST lint that nothing under `backtest/` or `scenarios/` references smoothed probabilities.
3. **Cross-check** against `statsmodels.MarkovRegression` on NIFTY data: state volatilities and transition probabilities agree within tolerance.
4. **Reality check:** on NIFTY history the stress state must cover March 2020 and the 2022 rate-shock drawdown.
5. **Determinism:** same seed ⇒ identical states; state 0 always the lower-σ state.
