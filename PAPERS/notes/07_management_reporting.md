# 07 · Management reporting: decision-first, honest, printable

## Why it matters for STAR2.0

The reader is an investment committee that must decide three things: approve or reject the proposed rebalance, confirm the portfolio is within its limits, and decide whether to add or withdraw capital. The legacy report did not serve any of them: its Summary opened with the covariance method and simulation path count, labelled a one-year VaR as "1-day", contained a "HONEST RISK SUMMARY" of hard-coded claims, and — measured — had no page setup on any sheet (so all 12 sheets printed too wide in portrait, ~31 A4 pages), zero formulas and most numbers stored as text.

## Key references

| Reference | Takeaway for STAR2.0 |
|---|---|
| Tufte, E. R. (1983; 2nd ed. 2001). *The Visual Display of Quantitative Information.* Graphics Press. | Maximise the data-to-ink ratio; no chart junk; small multiples for comparisons. |
| Few, S. (2006). *Information Dashboard Design.* O'Reilly. | A dashboard is a single screen of the few measures that drive a decision, with context (targets, ranges) beside each number. |
| Minto, B. (1987). *The Pyramid Principle.* Pitman. | Lead with the conclusion, then the supporting arguments, then the evidence. |
| Basel Committee on Banking Supervision (2013). *Principles for effective risk data aggregation and risk reporting* (BCBS 239). https://www.bis.org/publ/bcbs239.htm | Risk reports must be accurate, complete, timely, clear and adaptable, with data lineage. |
| Britton, E., Fisher, P., & Whitley, J. (1998). The Inflation Report projections: understanding the fan chart. *Bank of England Quarterly Bulletin*, February 1998, 30–37. | Show uncertainty as probability bands that widen with the horizon. |
| CFA Institute. *Global Investment Performance Standards (GIPS).* https://www.gipsstandards.org | Present performance net of costs, with the benchmark and the period stated; never cherry-pick. |

## What STAR2.0 implements

**Structure (pyramid principle):** headline recommendation → what changed since the last run → limit compliance → expected outcome band over the selected horizon → key risks and stress results → evidence quality. Trades sit on their own page.

**Every recommendation is a record, not prose:** claim, the evidence field it rests on, the action, a confidence level *derived from the statistical test behind it* (weak evidence renders as "not established"), and a falsifier — what would change the conclusion. No numeric literal may appear in a narrative template (enforced by a test), which rules out the legacy's hard-coded claims.

**Honesty rules for numbers:**

- Every probability states its assumption ("assuming equilibrium returns, …").
- Every point estimate carries a range ("Sharpe 0.62, 95% interval −0.10 to 1.34").
- Risk figures always name their horizon, and every horizon comes from the same simulation.
- Where the data cannot support a claim, the report says so instead of printing a number.

**Seven charts, each titled with its takeaway rather than its type:** outcome fan chart for the selected horizon; backtest equity curve after costs versus NIFTY total return with a drawdown underlay; risk contributions versus weights; stress impact for the selected scenario, with data coverage shown; regime timeline; allocation change (proposed versus current); evidence panel (Sharpe interval, Deflated Sharpe, overfitting probability, VaR-backtest light). No 3D, no dual axes, direct labels, colour-blind-safe palette, identical colours in HTML and Excel.

**Interactivity:**

- HTML brief (first deliverable): self-contained and offline; horizon slider (any number of days) and scenario selector read pre-computed data embedded in the page, so switching is instant and always matches the engine.
- Excel (second deliverable): a dropdown and a "custom days" cell drive `INDEX`/`MATCH` formulas over a hidden data sheet, and native Excel charts redraw on change — no macros. A static "as printed at 1 year" sheet is always included in case a spreadsheet program does not recalculate.

**Print discipline:** a page budget of about 44 content rows × 11 columns per A4 landscape page; explicit print areas, landscape orientation, fit-to-width, repeated title rows, page numbers; oversized content splits into Part 1 / Part 2 rather than shrinking or dropping anything. Print-targeted pages: Decision & allocation, Risk & limits & stress, Trades. Appendix data sheets are labelled as not for print.

## Pitfalls — and where the legacy went wrong

- Leading with engine internals instead of the decision.
- Mislabelled horizons (a one-year figure called "1-day").
- Precise-looking point estimates with no uncertainty.
- Narrative claims that were never computed.
- Numbers stored as formatted text, so nothing can be recomputed or reformatted.
- No page setup, so the report cannot be printed legibly.
- Dark-background charts that waste ink and reproduce badly on paper.

## Tests

1. **Every evidence reference resolves** to a real field in the report model.
2. **No numeric literals** in narrative templates (AST scan).
3. **HTML integrity:** the data embedded in the brief equals the engine's result exactly.
4. **Excel toggles:** the horizon dropdown exists, formulas resolve, and native charts reference the live block (with `plotVisOnly` off so hidden-sheet data renders).
5. **Print fit:** every print area fits the A4 landscape budget, and rows across Part 1/Part 2 sum to the source rows — nothing dropped.
6. **All figures are stored as numbers** with number formats, never as pre-formatted text.
