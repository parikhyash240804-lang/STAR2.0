"""
STAR2.0 — quantitative portfolio analytics for Indian (NSE) equities.

A rebuild of the Stardust 1.x engine (one 6,161-line function) as layered
modules, each owning one model. Layer order is strict and a test enforces it: a
module may import only from lower or equal layers, and `report`/`app` are
imported by nothing.

    core -> config -> data -> estimators -> signals -> views -> optimize
         -> risk | scenarios -> stress -> backtest -> report -> pipeline -> app

The defects this engine exists to fix are catalogued in docs/AUDIT_legacy.md, and
each one has a regression test named after it. The frozen legacy engine lives in
`legacy/stardust_v1` as the "before" baseline (see legacy/PARITY.md).
"""

__version__ = "2.0.0.dev0"
