# STAR2.0 research library

The literature behind every method choice in STAR2.0. Each note explains why the topic matters, the key papers (full citation, link and takeaway), exactly what STAR2.0 implements with formulas and defaults, where the legacy Stardust engine went wrong, and the tests that validate the implementation.

**Your own papers:** drop PDFs into `PAPERS/uploads/`. When a paper changes a method, record it in the relevant note under "Key papers" so the reasoning stays traceable.

**Citation policy:** citations were written from established references; where a volume, page range or link could not be stated with confidence it is marked "to confirm" rather than guessed.

## Topics

| # | Note | What it decides | STAR2.0 modules it informs |
|---|---|---|---|
| 01 | [Market regimes](notes/01_regime_markov.md) | 2-state Student-t hidden Markov model with filtered probabilities only; jump model, turbulence, absorption ratio and India VIX as cross-checks | `estimators/regime`, `scenarios/regime_bootstrap`, `stress/propagate` |
| 02 | [Black-Litterman and views](notes/02_black_litterman.md) | Equilibrium prior with derived δ; relative views sized by measured IC; Idzorek confidence; posterior covariance; excess/total units enforced by type | `estimators/equilibrium`, `views/*`, `core/types` |
| 03 | [Volatility and covariance](notes/03_covariance_volatility.md) | Yang–Zhang range volatility × Ledoit–Wolf-shrunk correlation; GJR-GARCH-t for scenarios; Dimson beta; factor model for attribution only | `estimators/ewma`, `estimators/covariance`, `estimators/garch` |
| 04 | [Backtesting](notes/04_backtesting.md) | One strategy object live and historical; look-ahead canary; drifted weights; Indian cost model; Sharpe intervals, Deflated Sharpe, PBO, MinTRL | `backtest/*`, `data/provider`, `data/view` |
| 05 | [Risk and stress](notes/05_risk_stress.md) | Horizon-aware ES/VaR without √t scaling; Kupiec/Christoffersen/Basel validation; Euler contributions; historical, hypothetical and reverse stress | `risk/*`, `stress/*` |
| 06 | [Construction and scenarios](notes/06_construction_scenarios.md) | Turnover-penalised mean-variance with a mandate; true frontier; enforced cardinality; stability in weight space; regime-conditional filtered historical simulation for every horizon | `optimize/*`, `scenarios/*` |
| 07 | [Management reporting](notes/07_management_reporting.md) | Decision-first brief; recommendations with evidence, confidence and falsifier; seven takeaway-titled charts; toggles without macros; A4 landscape print discipline | `report/*` |

## Author index

| Reference | Note |
|---|---|
| Acerbi & Székely (2014) | 05 |
| Acerbi & Tasche (2002) | 05 |
| Almgren, Thum, Hauptmann & Li (2005) | 04 |
| Ang & Bekaert (2002) | 01 |
| Arnott, Harvey & Markowitz (2019) | 04 |
| Artzner, Delbaen, Eber & Heath (1999) | 05 |
| Aydınhan, Kolm, Mulvey & Shu (2024) | 01 |
| Bailey, Borwein, López de Prado & Zhu (2017) | 04 |
| Bailey & López de Prado (2012, 2014) | 04 |
| Barone-Adesi, Giannopoulos & Vosper (1999) | 05, 06 |
| Basel Committee (1996 backtesting; 2013 BCBS 239; 2018 stress testing) | 05, 07 |
| Black & Litterman (1992) | 02 |
| Bradshaw, Brown & Huang (2013) | 02 |
| Breuer, Jandačka, Rheinberger & Summer (2009) | 05 |
| Brinson, Hood & Beebower (1986) | 04 |
| Britton, Fisher & Whitley (1998) | 06, 07 |
| CFA Institute, GIPS | 07 |
| Chan, Jegadeesh & Lakonishok (1996) | 02 |
| Chekhlov, Uryasev & Zabarankin (2005) | 05 |
| Choueifaty & Coignard (2008) | 06 |
| Christoffersen (1998) | 05 |
| DeMiguel, Garlappi & Uppal (2009) | 03 |
| Dimson (1979) | 03 |
| Engle (2002) | 03 |
| Engle & Bollerslev (1986) | 03 |
| Engle & Manganelli (2004) | 05 |
| Few (2006) | 07 |
| Gârleanu & Pedersen (2013) | 06 |
| Glasserman, Kang & Kang (2015) | 05 |
| Glosten, Jagannathan & Runkle (1993) | 03 |
| Grinold (1994) | 02 |
| Guidolin & Timmermann (2007) | 01 |
| Hamilton (1989) | 01 |
| Hansen (2005) | 04 |
| Harvey & Liu (2015) | 04 |
| Harvey, Liu & Zhu (2016) | 04 |
| He & Litterman (1999) | 02 |
| Higham (2002) | 03 |
| Idzorek (2005) | 02 |
| J.P. Morgan/Reuters RiskMetrics (1996) | 03 |
| Kritzman & Li (2010) | 01 |
| Kritzman, Li, Page & Rigobon (2011) | 01 |
| Kritzman, Page & Turkington (2012) | 01 |
| Kupiec (1995) | 05 |
| Laloux, Cizeau, Bouchaud & Potters (1999) | 03 |
| Ledoit & Wolf (2004, 2008, 2020) | 03, 04 |
| Lo (2002) | 04 |
| López de Prado (2016) | 06 |
| MacLean, Thorp & Ziemba (2010) | 06 |
| Maillard, Roncalli & Teïletche (2010) | 06 |
| Markowitz (1952) | 06 |
| McNeil & Frey (2000) | 05 |
| Meucci (2008) | 02 |
| Michaud (1989, 1998) | 03, 06 |
| Minto (1987) | 07 |
| Nystrup, Lindström & Madsen (2020) | 01 |
| Nystrup, Madsen & Lindström (2015) | 01 |
| Politis & Romano (1994) | 04, 06 |
| Rockafellar & Uryasev (2000) | 05 |
| Scherer (2002) | 06 |
| Tasche (2007) | 05 |
| Tufte (1983) | 07 |
| Walters (2014) | 02 |
| White (2000) | 04 |
| Yang & Zhang (2000) | 03 |

## Data sources referenced

- **IIMA Indian factor library** (Agarwalla, Jacob & Varma), release 2025-12, survivorship-bias adjusted: daily Carhart four-factor returns (market, size, value, momentum — not Fama-French 5) for Indian equities, used for risk attribution; bundled at `data/reference/iima/`.
- **Yahoo Finance via yfinance:** NSE prices (OHLCV), NIFTY 50, NIFTYBEES (total-return proxy), GOLDBEES, India VIX; company statements and analyst snapshots (current values only, never point-in-time).
- **RBI 91-day Treasury bill yields**, as carried in the IIMA file's RF column (per the IIMA FAQ, from annualised yields of new 91-day T-bill issues published by RBI): the dated risk-free curve. The data ends at the release month; later dates carry the last rate forward and are flagged.
