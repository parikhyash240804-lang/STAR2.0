"""
STAR2.0 exception types.

Every error a user can cause carries a message a user can act on. The legacy
engine used bare `assert` for validation (removed under -O, and surfacing in the
GUI as a stack trace) and `except Exception: pass` for parsing, which turned bad
input into silently wrong numbers.
"""
from __future__ import annotations


class StarError(Exception):
    """Base class for every STAR2.0 error."""


class ConfigError(StarError):
    """Invalid configuration. Carries the collected issues, not just the first."""

    def __init__(self, issues):
        self.issues = list(issues)
        joined = "\n".join(f"  - {i}" for i in self.issues)
        super().__init__(f"{len(self.issues)} configuration problem(s):\n{joined}")


class ConventionError(StarError):
    """
    A return series was about to be used in the wrong units.

    This is the guard against the legacy's worst arithmetic bug class: an excess
    return treated as a total return (the risk-free rate subtracted twice), or a
    geometric mean mixed with arithmetic ones.
    """


class AlignmentError(StarError):
    """Two objects that must share an asset ordering do not."""


class DataError(StarError):
    """Data could not be fetched, or failed a quality gate."""


class LookAheadError(StarError):
    """
    Data after the as-of cutoff was requested.

    Raised by the backtest's data guard. If this fires, a backtest was about to
    use information that did not exist at the time.
    """


class EstimationError(StarError):
    """An estimator could not produce a usable result (and must not fake one)."""


class OptimizationError(StarError):
    """The optimiser failed, or succeeded on an ill-posed problem."""
