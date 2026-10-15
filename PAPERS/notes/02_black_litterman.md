# 02 · Black-Litterman, and turning signals into views

## Why it matters for STAR2.0

Mean-variance optimisation is extremely sensitive to expected returns, and expected returns are the least reliably estimated input. Black-Litterman anchors the forecast to an equilibrium (the returns that would make a reference portfolio optimal) and moves away from it only as far as the evidence in each view justifies.

The legacy engine used Black-Litterman in a way that removed most of its benefit (all verified; see `docs/AUDIT_legacy.md`): δ = 0.5 implied a ~1% equity risk premium (M2-05); every stock got an absolute view forced into a bullish +1.5% to +15.5% excess band (M2B-01); Ω ∝ τ made `BL_TAU` a dead control and views carried ~91% of the posterior rather than the advertised 40% (M2B-11); the posterior, an excess return, was treated downstream as a total return (M2-01); sector multipliers produced zero net tilt (M2B-04). Most of these fire only on the `bl`/`hybrid` path, but STAR2.0 makes Black-Litterman the default, so all of them must be right.

## Key papers

| Paper | Takeaway for STAR2.0 |
|---|---|
| Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal* 48(5), 28–43. https://doi.org/10.2469/faj.v48.n5.28 | The framework: equilibrium prior plus views with stated uncertainty. |
| He, G., & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. Goldman Sachs Investment Management Research. https://doi.org/10.2139/ssrn.334304 | Worked seven-country example; the reference numbers for a known-answer test. |
| Idzorek, T. M. (2005). A step-by-step guide to the Black-Litterman model: Incorporating user-specified confidence levels. Working paper (later in *Forecasting Expected Returns in the Financial Markets*, Elsevier, 2007). | Maps an intuitive 0–100% confidence to Ω so that confidence c moves weights c of the way to the 100%-certain tilt. |
| Walters, J. (2014). The Black-Litterman model in detail. SSRN working paper. https://doi.org/10.2139/ssrn.1314585 | Careful derivation, including the posterior covariance and the role of τ. |
| Meucci, A. (2008). Fully flexible views: Theory and practice. *Risk* 21(10), 97–102. https://doi.org/10.2139/ssrn.1213325 | Entropy pooling: views as probability statements on any scenario set; classic BL is a special case. |
| Grinold, R. C. (1994). Alpha is volatility times IC times score. *Journal of Portfolio Management* 20(4), 9–16. https://doi.org/10.3905/jpm.1994.409482 | α_i = IC × σ_i × z_i converts a standardised signal into an expected excess return of the right scale. |
| Chan, L. K. C., Jegadeesh, N., & Lakonishok, J. (1996). Momentum strategies. *Journal of Finance* 51(5), 1681–1713. https://doi.org/10.1111/j.1540-6261.1996.tb05222.x | Earnings-estimate revisions predict returns; a candidate live signal. |
| Bradshaw, M. T., Brown, L. D., & Huang, K. (2013). Do sell-side analysts exhibit differential target price forecasting ability? *Review of Accounting Studies* 18(4), 930–955. https://doi.org/10.1007/s11142-012-9216-5 | Analyst price targets are systematically optimistic and poorly persistent; if used as views they need heavy shrinkage. |

## What STAR2.0 implements

**Prior (equilibrium, excess returns):** Π = δ Σ w_ref.

- δ is **derived**, not typed in: δ = SR_mkt / σ_mkt. The user-facing knob is a target market Sharpe (default 0.35); at σ_mkt ≈ 15%, δ ≈ 2.3 and the implied equity risk premium ≈ 5.2%.
- w_ref = free-float market-cap weights within the universe (labelled "universe reference portfolio", because 40 stocks are not the market); fallback to shares outstanding, then equal weight, each fallback recorded in `Provenance`. In backtests, current market caps are look-ahead, so equal weight is used as-of and stated.

**Views:** relative quintile views by default (top-quintile basket minus bottom-quintile basket of the composite score), because a cross-sectional signal says which stocks beat which, not what the market will do. Absolute views only where the user supplies one.

- View size from Grinold: q = IC × σ × z, with **IC measured** as the rolling rank information coefficient of the signal, clipped to [0.02, 0.10] — never asserted.
- Scores are cross-sectionally demeaned, so views can be negative.

**Uncertainty (Ω) — Idzorek:** for view k with confidence c_k, find ω_k such that the posterior weight tilt equals c_k × (the tilt at 100% certainty). τ = 0.05, and because Ω is solved for, stated confidence genuinely moves the posterior.

**Posterior:**

- M = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹
- μ_BL = M [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ q]  (excess returns)
- Σ_BL = Σ + M  (posterior covariance kept, so parameter uncertainty reaches the optimiser)

**Units:** μ_BL is a `ReturnEstimate` with `basis=EXCESS`; conversion to total happens once, via `to_total(rf)`, which raises if called twice. This makes legacy defect M2-01 unrepresentable.

**Research mode:** entropy pooling on the forward scenario set, for views like *"P(NIFTY falls more than 10% in 3 months) = 5%"*; street-consensus views (EPS revisions, analyst targets) as live-only, low-confidence views with measured IC.

## Pitfalls — and where the legacy went wrong

- δ too low makes the prior inert and lets views dominate (M2-05, M0-03).
- An absolute view on every asset is really a market call plus stock selection mixed together (M2B-01); relative views separate them.
- If Ω is proportional to τΣ, τ cancels out of the posterior exactly — for any number of views, not just one per asset (M2B-11). **Verified algebraically and by test in STAR2.0** (`tests/test_estimators_black_litterman.py::test_tau_cannot_change_the_answer_and_that_is_the_point`). Idzorek's calibration does **not** restore τ as a live control: because the fully-certain tilt is itself τ-free, solving Ω to hit a fraction of that tilt leaves the posterior τ-invariant too. What Idzorek fixes is different and more useful — it makes *stated confidence* the live control, with the property that confidence c moves the weights c of the way to the certain tilt. τ is therefore documented as an unidentifiable scale, not exposed as a knob. The legacy exposed `BL_TAU` in its input template while this identity made it inert.
- A mapping from scores to a fixed return band ignores how predictive the signal actually is (M2B-05); Grinold's IC scaling does not.
- Multiplying a signed, sector-relative score by a sector multiplier cannot tilt a sector (M2B-04); a sector view must be its own P-row.
- Analyst targets are biased upward (Bradshaw, Brown & Huang 2013).

## Tests

1. **He & Litterman (1999) known answer:** reproduce the equilibrium Π (to 1e-4) and posterior weights for their two views (to 1e-3).
2. **Idzorek tilt ratio:** 100% confidence reproduces the certainty tilt exactly; 50% gives half of it, in the view's direction.
3. **τ provably cannot change the answer** under either Ω rule (asserted to 1e-9), which is why confidence and not τ is the exposed control; and a view with confidence → 0 leaves the posterior at the prior.
4. **Convention guard:** calling `to_total` on the posterior twice raises `ConventionError`.
5. **Negative views are expressible:** a bottom-quintile basket receives a negative expected excess return relative to the top.
6. **Entropy pooling (research mode):** posterior scenario probabilities satisfy every view constraint exactly, and with a linear view on Gaussian scenarios the posterior mean matches classic BL.
