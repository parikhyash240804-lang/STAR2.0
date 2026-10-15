# Vision, requested changes, and progress

Updated: **15 October 2026**. This is a living requirements ledger, not a claim that every mentioned feature is approved or complete.

Status vocabulary: **Done**, **In progress**, **Planned**, **Partly implemented**, **Needs specification**, **Proposed**, **Excluded by recorded decision**. Mark Done only after checking the behavior and relevant integration. Record dates, evidence, and changed decisions as work proceeds; preserve useful history.

## Confirmed directly in this chat

| ID | Request / decision | Status | Evidence / scope |
|---|---|---|---|
| OWNER-001 | Connect this local STAR2.0 folder to the supplied GitHub repository and upload all files. | Done | `main` on parikhyash240804-lang/STAR2.0; commits `64909a0` and `e34147f`. The second explicitly includes caches and compiled files. |
| OWNER-002 | Continue building this incomplete application from this folder/repository. | In progress | Owner's stated development intent. No completion percentage or release scope has been agreed. |
| OWNER-003 | Name the chat `1_Configurator`. | Done | Chat renamed. The label does not specify configurator features, platform, or design. |
| OWNER-004 | Review the whole project, explain progress/architecture/gaps, and run existing checks without changing application behavior. | Done | Baseline review on 15 October; 455 tests passed, quick/full offline generation succeeded; findings in `docs/REVIEW_2026-10-15.md`. |
| OWNER-005 | Maintain reusable project context, vision/change tracking, accurate README, and future-agent guidance; commit and push this documentation. | Done | `PROJECT_CONTEXT.md`, this file, `README.md`, `AGENTS.md`, and review notes created/updated in this documentation change. |

## Owner's new vision and feature requests

**Needs specification.** No additional app vision, new feature specification, implementation order, delivery platform, or configurator acceptance criteria has been stated directly in this chat. Add future owner requests here in their own terms. Do not silently promote the suggestions below into requirements.

For each future request, capture: ID, date, user intent, acceptance criteria, priority if stated, status, dependencies, verification, and completing commit. When a request supersedes an earlier decision, link the old item and explain the new direction.

## Inherited product direction from repository documents

These are **recorded prior decisions/designs**, with their original provenance preserved. They have not been newly reconfirmed in this chat. More recent explicit owner instructions override them.

| ID | Recorded direction | Current implementation / status | Source |
|---|---|---|---|
| DESIGN-001 | Separate an offline management HTML brief from a complete analyst Excel archive; both consume the same run results. | Partly implemented: HTML exists; new output workbook/archive and common run model are missing. | `docs/PERSONAS.md`, `PAPERS/notes/07_management_reporting.md` |
| DESIGN-002 | Answer rebalance approval, risk-limit compliance, and capital-allocation decisions; lead with one recommendation. | Partly implemented: guidance rules and sections exist; some inputs/checks are missing or overstated. | `docs/PERSONAS.md` |
| DESIGN-003 | Simple English, defined meaningful finance terms, Indian rupee formatting, one message per recommendation, detailed caveats collapsed. | Implemented in much of the brief and tested; some confidence/health claims still need correction. | Owner quotation and settled rules in `docs/PERSONAS.md` |
| DESIGN-004 | Multiple horizons and scenarios, instant offline response, mobile support and sensible printing. | Preset horizon/scenario controls exist; arbitrary-day UI, shared default selection and full-page scenario updates are missing. Browser/mobile/print behavior was not verified in this review. | `docs/PERSONAS.md`, research notes 06/07 |
| DESIGN-005 | Market/risk monitors: India VIX, correlation, diversification ratio and gold sleeve. | Core functions exist and have tests; only a subset is shown by the demo. | Owner selection dated 19 September 2026 in `docs/INDUSTRY_METRICS_ASSESSMENT.md` |
| DESIGN-006 | Execution and attribution: tranches, Brinson, hit rate/profit factor, conviction-based caps. | Core functions exist and are tested; demo/output integration is incomplete. | Same recorded owner selection |
| DESIGN-007 | Technical entry-timing overlay, including RSI/MACD/Bollinger, on the workbook trades page only; measure predictive value and keep it out of optimisation. | Planned in inherited documents; no current implementation found. | `docs/PERSONAS.md`, recorded industry-metrics selection |
| DESIGN-008 | Street-consensus signals and fundamental red flags are not being built. | Excluded by recorded decision; do not re-add merely because older research notes discuss them. | Final Status section of `docs/INDUSTRY_METRICS_ASSESSMENT.md` |
| DESIGN-009 | Retain an Excel-in / Excel-out workflow. | Partly implemented: Excel input helpers; only the frozen legacy produces the old output workbook. | `docs/AUDIT_legacy.md`, personas |

## Review findings and proposed engineering work

These are **reviewer findings/suggestions**, not newly approved feature requests. Stable issue IDs and evidence are in `docs/REVIEW_2026-10-15.md`.

| ID | Proposed work | Status |
|---|---|---|
| R01 | Connect validated configuration to a shared run pipeline and strategy; resolve incompatible config/runtime settings. | Proposed; open |
| R02 | Arm provider guards during every historical strategy call and pass/check all as-of inputs. | Proposed; open; missing guard reproduced |
| R03 | Enforce and report all applicable mandate checks; replace placeholder sectors and example holdings with real inputs. | Proposed; open |
| R04 | Use one explicit total-return benchmark and consistent simple/log portfolio returns throughout. | Proposed; open |
| R05 | Rebuild demo/scorecard risk validation with genuine prior forecasts and held-out calibration. | Proposed; open |
| R06 | Purge unfinished outcome windows from volatility bake-off training; rerun and version the evidence. | Proposed; open |
| R07 | Derive confidence from actual sample/tail/calibration information and retain all relevant warnings. | Proposed; open |
| R08 | Make simulation memory accounting cover all persistent/temporary arrays and generator state. | Proposed; open |
| R09 | Connect simulation selection and drift settings; qualify absent regime/FHS and sensitivity behavior. | Proposed; open |
| R10 | Implement the new analyst workbook/archive before claiming it accompanies a brief. | Proposed integration work against inherited design; open |
| R11 | Reconcile research/design descriptions and test claims with actual modules and behavior. | Proposed; partly addressed by current README/context/review; original notes preserved |
| R12 | Add installation/package metadata and reproducible setup; close validation and error-handling gaps. | Proposed; open |

Signals are absent, but selecting and building a signal strategy is **not automatically approved** by that finding. The owner must define that product direction before an assumed strategy is introduced.

## Change log

### 15 October 2026 — repository connection and baseline review

- Uploaded the project and then the remaining caches/compiled files as requested.
- Reviewed the source, tests, existing design/audit/research notes, data structures and generated brief; inspected available Git history.
- Verified 455 tests and quick/full brief generation on the current local environment. Recorded limitations and open issues separately from passed checks.
- Added context, requirements tracking and agent guidance; expanded the original two-line README. Preserved the historical docs and frozen legacy. No application logic changed.
