"""
One canonical asset ordering, enforced mechanically.

The legacy engine held its asset ordering together with six hand-written checks
of the form::

    assert list(arith_returns.index) == list(cov_annual.index)

That is the right instinct with the wrong mechanism: the checks only exist where
someone remembered to write one, they compare *equality* rather than repairing
order, and a silent reordering anywhere else produced a portfolio whose weights
belonged to different stocks than its risk numbers.

Here a run fixes a `CanonicalIndex` once, and the `@aligned` decorator reindexes
the named arguments of a function to it before the body runs. Misalignment then
becomes impossible rather than merely detected, and the guarantee is covered by a
permutation-invariance test: permuting the input order must permute the output
identically, never change its values.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .errors import AlignmentError


@dataclass(frozen=True, slots=True)
class CanonicalIndex:
    """The one asset ordering every object in a run must use."""

    index: pd.Index

    @classmethod
    def of(cls, tickers: Iterable[str]) -> "CanonicalIndex":
        idx = pd.Index(list(tickers), name="ticker")
        if idx.has_duplicates:
            dupes = sorted(set(idx[idx.duplicated()]))
            raise AlignmentError(f"duplicate tickers in the universe: {dupes}")
        if len(idx) == 0:
            raise AlignmentError("the universe is empty")
        return cls(index=idx)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def tickers(self) -> list[str]:
        return list(self.index)

    # ---- alignment ----
    def align(self, obj: Any, *, name: str = "value") -> Any:
        """
        Reindex `obj` onto this ordering.

        Series are aligned by index; DataFrames by columns, and also by index when
        they are square and carry the same labels (a covariance matrix). Anything
        that cannot supply every ticker is an error, not a NaN: a silently missing
        asset is how a portfolio ends up with weights for the wrong stocks.
        """
        if isinstance(obj, pd.Series):
            missing = self.index.difference(obj.index)
            if len(missing):
                raise AlignmentError(f"{name} is missing {len(missing)} ticker(s): {list(missing)[:8]}")
            return obj.reindex(self.index)

        if isinstance(obj, pd.DataFrame):
            # a covariance may arrive with rows and columns in DIFFERENT orders;
            # compare label sets, or rows and columns end up describing different stocks
            square = obj.shape[0] == obj.shape[1] and set(obj.index) == set(obj.columns)
            missing = self.index.difference(obj.columns)
            if len(missing):
                raise AlignmentError(f"{name} is missing {len(missing)} column(s): {list(missing)[:8]}")
            if square:
                return obj.reindex(index=self.index, columns=self.index)
            return obj.reindex(columns=self.index)

        # the typed estimates carry their data inside; align that, keep the units
        from dataclasses import replace
        from .types import CovarianceEstimate, ReturnEstimate
        if isinstance(obj, ReturnEstimate):
            return replace(obj, values=self.align(obj.values, name=name))
        if isinstance(obj, CovarianceEstimate):
            return replace(obj, matrix=self.align(obj.matrix, name=name))

        return obj      # scalars and non-pandas values pass through untouched

    def validate(self, obj: Any, *, name: str = "value") -> None:
        """Assert an object is already aligned (no repair). Used at seams."""
        if isinstance(obj, pd.Series):
            if not obj.index.equals(self.index):
                raise AlignmentError(f"{name} is not in canonical order")
        elif isinstance(obj, pd.DataFrame):
            if not obj.columns.equals(self.index):
                raise AlignmentError(f"{name} columns are not in canonical order")
            if obj.shape[0] == obj.shape[1] and set(obj.index) == set(obj.columns) \
                    and not obj.index.equals(self.index):
                raise AlignmentError(f"{name} index is not in canonical order")


def aligned(*params: str):
    """
    Reindex the named parameters onto the call's `CanonicalIndex` before running.

    The index is found among the bound arguments (by type), so it is explicit at
    the call site; the parameter names are listed here, so nothing is aligned by
    accident::

        @aligned("mu", "cov")
        def solve(index: CanonicalIndex, mu: pd.Series, cov: pd.DataFrame): ...
    """
    def decorate(fn):
        sig = inspect.signature(fn)
        unknown = set(params) - set(sig.parameters)
        if unknown:
            raise AlignmentError(f"@aligned names parameters {sorted(unknown)} that "
                                 f"{fn.__qualname__} does not take")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            canon = next((v for v in bound.arguments.values() if isinstance(v, CanonicalIndex)), None)
            if canon is None:
                raise AlignmentError(
                    f"{fn.__qualname__} is decorated with @aligned but received no "
                    f"CanonicalIndex argument")
            for p in params:
                if p in bound.arguments and bound.arguments[p] is not None:
                    bound.arguments[p] = canon.align(bound.arguments[p], name=p)
            return fn(*bound.args, **bound.kwargs)

        wrapper.__aligned_params__ = tuple(params)      # discoverable by tests
        return wrapper

    return decorate
