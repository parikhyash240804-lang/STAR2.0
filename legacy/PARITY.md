# Task 1 parity — the modular legacy engine reproduces the monolith exactly

`legacy/stardust_v1/` is the Stardust 1.x engine split from one 6,161-line
function into 11 modules. The split is only useful if it changed nothing, so it
is verified rather than asserted.

## Result (2026-09-13)

| Check | Result |
|---|---|
| Printed output (every intermediate table the engine prints) | **908 / 908 lines identical** |
| Excel workbook | **12 / 12 sheets, 0 differing cells** (1,856 cells compared) |
| Charts | **4 / 4 pixel-identical** |
| Verdict | **PARITY OK** |

## How it is verified

`tools/parity_check.py` replaces yfinance with a record/replay proxy so both
engines see byte-identical market data, then diffs everything they produce.

```bash
# 1. run the ORIGINAL monolith against live data, caching every yfinance response
python tools/parity_check.py record  <dir>/A_record

# 2. replay both engines on that cache (run these on the same calendar day)
python tools/parity_check.py orig    <dir>/A2_orig
python tools/parity_check.py mod     <dir>/B_mod

# 3. diff printed output, every Excel cell, and every chart pixel
python tools/parity_check.py compare <dir>/A2_orig <dir>/B_mod
```

Exit code 0 and `PARITY OK` means the port is faithful.

## What the harness controls for

- **Market data** — every `yf.download`, `.info`, `.quarterly_earnings` and
  `.fast_info` response is cached and replayed; a cache miss is reported with its
  full key rather than silently degrading the run.
- **The clock** — `pd.Timestamp.now` is frozen to the recording instant. The
  stdlib clock is deliberately *not* patched: replacing `datetime.datetime`
  breaks dateutil (it resolves `datetime.date` at call time and then rejects a
  real `pd.Timestamp`, so `pd.DateOffset` arithmetic fails). Instead the harness
  **refuses to replay a cache recorded on a different day**, because M6 derives
  its benchmark window from the stdlib clock and that leaks into a cache key.
  Crossing midnight silently loses every beta and changes the stress numbers —
  which looks exactly like a port defect but is pure harness drift.
- **Side effects** — the legacy report copies itself to `~/Downloads` and
  auto-opens Excel. Both are neutralised (`Path.home` redirected, `os.startfile`
  stubbed) so a test run cannot touch the real Downloads folder.
- **Output paths** — the two engines necessarily write to different folders, so
  absolute paths are normalised before comparison.

## Known, deliberate non-differences

The comparator ignores clock-derived text (timestamps, elapsed times, file
sizes) since those differ between any two runs of the *same* engine.

## Regenerating the modules

```bash
python tools/split_monolith.py ../../stardust_core.py stardust_v1
```

The splitter derives each stage's data contract with `ast` + `symtable` scope
analysis, so comprehension, loop and `except ... as e` variables are correctly
scoped, and a name that is read before being reassigned is recognised as a real
input. It reports two hazard classes that a closure→module split can introduce
(forward references and late binding); both are currently empty.
