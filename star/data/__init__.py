"""
Data access: providers, caching, point-in-time views, quality gates.

Two rules this layer exists to enforce:

1. **Nothing reads the future.** `AsOfGuard` / `DataView` make a request for data
   past the cutoff raise, rather than trusting every future caller to remember.
2. **A degraded fetch is never silent.** The legacy turned a failed download into
   "all sectors Unknown" and a dead signal into a constant zero, both invisible in
   the output. Fallbacks here are recorded in `Provenance.degradations` and
   surfaced in the report.
"""
from .provider import (
    GUARD,
    AsOfGuard,
    FundamentalsProvider,
    PricePanel,
    PriceProvider,
    RiskFreeProvider,
    UniverseProvider,
)
from .view import DataView

__all__ = [
    "GUARD", "AsOfGuard", "PricePanel", "PriceProvider", "FundamentalsProvider",
    "UniverseProvider", "RiskFreeProvider", "DataView",
]
