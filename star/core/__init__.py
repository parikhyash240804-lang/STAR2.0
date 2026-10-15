"""
STAR2.0 core: the vocabulary every other layer speaks.

Nothing here imports from any other `star` package, and nothing here touches the
network, the filesystem or a clock. These are the structural guarantees the
legacy engine lacked:

* `types`    - units (excess/total, log/simple, daily/annual) carried as types,
               so the double-subtracted risk-free rate cannot be written
* `align`    - one canonical asset ordering, enforced by decorator
* `horizons` - forecast horizons as configuration rather than a hard-coded 252
* `errors`   - errors a user can act on (the legacy used bare `assert`)
"""
from .align import CanonicalIndex, aligned
from .errors import (
    AlignmentError,
    ConfigError,
    ConventionError,
    DataError,
    EstimationError,
    LookAheadError,
    OptimizationError,
    StarError,
)
from .horizons import HorizonGrid, HorizonSpec
from .types import (
    Basis,
    Compounding,
    CovarianceEstimate,
    Periodicity,
    Provenance,
    ReturnEstimate,
)

__all__ = [
    "CanonicalIndex", "aligned",
    "StarError", "ConfigError", "ConventionError", "AlignmentError", "DataError",
    "LookAheadError", "EstimationError", "OptimizationError",
    "HorizonSpec", "HorizonGrid",
    "Basis", "Compounding", "Periodicity", "Provenance", "ReturnEstimate",
    "CovarianceEstimate",
]
