# STAR2.0

Portfolio construction, risk analysis, and decision reporting for Indian equities. STAR2.0 is an **incomplete application with a working Python analytics engine and an offline HTML demonstration**. It rebuilds the older Stardust engine in separate, testable modules.

The intended decisions are whether to approve a rebalance, whether the portfolio respects its limits, and whether the evidence supports adding or withdrawing capital. The current demo uses risk estimates, an equilibrium return prior, constraints, and trading controls; **it has no implemented stock-selection signal engine and no demonstrated return advantage**.

## Start here

- [Project context](PROJECT_CONTEXT.md): architecture, current progress, and handoff for a new chat.
- [Changes and vision](CHANGES.md): confirmed requests, inherited design decisions, and separate proposed work.
- [Latest review](docs/REVIEW_2026-10-15.md): verification results, open defects, and documentation discrepancies.
- [Contributor guidance](AGENTS.md): how to continue work without losing context.
- [Output personas](docs/PERSONAS.md): existing product direction for the management brief and analyst workbook.

## What exists

- Validated settings and Excel input-template read/write functions.
- Offline and Yahoo price providers, quality checks, caching, dated risk-free rates, and a calibrated total-return benchmark helper.
- HAR volatility, Yang-Zhang fallback, shrunk correlation, GJR-GARCH-t, equilibrium and Black-Litterman estimates, and a two-state regime model.
- Constrained portfolio optimisation, a holdings-limit routine, efficient frontier, risk-parity comparisons, and stability diagnostics.
- Horizon-specific simulation, risk measures, stress scenarios, backtesting with trading costs, attribution, and statistical checks.
- A self-contained HTML brief with charts, horizon/scenario buttons, and guidance.

**Still missing:** a user-facing configurator, a configuration-driven shared application pipeline, the new analyst output workbook/archive, installation metadata, and packaging. Several modules are available as library functions but are not connected to the demo. Some calculations and disclosures need correction before the demo can be treated as a validated decision tool; see the review.

## Run locally

Run commands from the repository root. Python **3.12.10** was used for the verified baseline; the source uses Python 3.11+ features. No dependency manifest or lockfile currently exists, and a fresh installation has not been verified.

For a separate development environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy scikit-learn arch openpyxl pyarrow pytest
```

Install `yfinance` separately if using `star.data.yahoo.YahooProvider`. The offline demo does not fetch live prices. The legacy engine additionally uses packages such as matplotlib, statsmodels, and Pillow; it is a historical baseline, not the recommended application entry point.

### Generate the HTML demonstration

```powershell
python -B tools/make_brief.py --quick --out runs/brief-quick.html
python -B tools/make_brief.py --out runs/brief-full.html
```

Open the resulting HTML file in a browser. `--quick` omits the walk-forward backtest and its track-record evidence. Both commands were verified on 15 October 2026, with outputs written outside the repository during the review.

The script currently fixes its universe, portfolio value (Rs. 1 crore), example existing holdings, placeholder sector labels, forecast length, and other settings in code. It **does not load an input workbook or `StarConfig`**. The bundled prices end on **11 September 2026**; the date shown in a generated brief is the data date, not today's date.

### Run the tests

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Verified baseline: **455 passed, 1 warning**, in 346.10 seconds. The warning was SciPy's singular-Jacobian/SVD fallback in the no-trade-band test. Passing tests do not establish predictive skill or remove the integration issues recorded in the review.

### Work with the Excel input template

These are library APIs; there is no application load/run button yet:

```python
from star.config.excel_input import read_input, write_template
from star.config.schema import StarConfig
from star.data.fixture import FixtureProvider

# Example universe only; replace with the intended universe when configuring a run.
config = StarConfig(tickers=tuple(FixtureProvider.default().universe[:20]))
write_template("star-input.xlsx", config)
loaded = read_input("star-input.xlsx")
```

The input workbook is distinct from the planned analyst output workbook. Template round-trip and validation are covered by tests.

### Research utilities

```powershell
python -B tools/validate_models.py
python -B tools/improve_volatility.py
```

These print research scorecards. They were inspected but not rerun during this review. Their evaluation methodology has open issues, and running them does not automatically refresh the constants or existing reports. Historical results in [VALIDATION.md](docs/VALIDATION.md) must be read with the qualifications in the latest review.

## Repository map

| Location | Role |
|---|---|
| `star/` | Current engine: core, config, data, estimators, optimise, forecast, risk, stress, backtest, report |
| `tools/` | Demo brief generator and research evaluation scripts |
| `tests/` | Offline automated tests and market-data fixtures |
| `data/reference/iima/` | Bundled December 2025 Indian four-factor and risk-free data release |
| `runs/brief.html` | Previously generated full demo, dated 11 September 2026 |
| `docs/` | Prior audits/design notes and the current review |
| `PAPERS/notes/` | Seven research/design notes; some descriptions are plans rather than shipped features |
| `legacy/` | Frozen modular Stardust 1.x baseline and parity/splitting tools |

The repository includes compiled Python and test-cache files because the owner explicitly requested uploading every local file. They are not the source of truth. New development belongs in `star/`; importing the legacy package changes process-level settings and should be isolated.
