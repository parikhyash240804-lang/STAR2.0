"""
Forecast horizons as data.

The legacy engine had no horizon concept: `TRADING_DAYS = 252` served as both the
annualisation constant and the simulation length, hard-coded in ~40 places, and
only the terminal day of the simulated path was ever reduced to numbers. Asking
it "what about 7 days?" required editing code.

Here a horizon is a value, the set of horizons is configuration, and the
annualisation constant is kept strictly separate from the forecast length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .errors import ConfigError

#: trading-day lengths of the calendar units, given a trading year.
#: A week is 5 trading days by definition, not `periods_per_year / 52`.
_UNIT_DAYS = {
    "d": lambda ppy: 1.0,
    "w": lambda ppy: 5.0,
    "m": lambda ppy: ppy / 12.0,
    "q": lambda ppy: ppy / 4.0,
    "y": lambda ppy: float(ppy),
}
_TOKEN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([dwmqy]?)\s*$", re.IGNORECASE)

_DEFAULT_LABELS = {1: "1D", 2: "2D", 5: "1W", 10: "2W", 21: "1M", 63: "3M", 126: "6M", 252: "1Y"}


@dataclass(frozen=True, slots=True, order=True)
class HorizonSpec:
    """One forecast horizon: a number of trading days plus how to label it."""

    days: int
    label: str

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ConfigError([f"horizon must be at least 1 trading day, got {self.days}"])

    @classmethod
    def parse(cls, token: str | int, *, periods_per_year: float = 252.0) -> "HorizonSpec":
        """
        Parse "1d", "2d", "1w", "1m", "3m", "6m", "1y", "45" (bare = trading days).

        A bare integer is trading days, so "45" is a valid custom horizon - the
        owner can ask for 45 days without a code change.
        """
        if isinstance(token, (int, float)) and not isinstance(token, bool):
            days = int(round(float(token)))
            return cls(days=days, label=_DEFAULT_LABELS.get(days, f"{days}D"))
        m = _TOKEN.match(str(token))
        if not m:
            raise ConfigError([
                f"cannot read horizon {token!r}; use e.g. 1d, 2d, 1w, 1m, 3m, 6m, 1y, "
                f"or a plain number of trading days such as 45"
            ])
        qty, unit = float(m.group(1)), (m.group(2) or "d").lower()
        days = int(round(qty * _UNIT_DAYS[unit](periods_per_year)))
        if days < 1:
            raise ConfigError([f"horizon {token!r} is shorter than one trading day"])
        canonical = f"{int(qty)}{unit.upper()}" if qty == int(qty) else f"{qty}{unit.upper()}"
        return cls(days=days, label=_DEFAULT_LABELS.get(days, canonical))


@dataclass(frozen=True, slots=True)
class HorizonGrid:
    """
    The set of horizons a run reports, ordered and de-duplicated.

    `max_days` drives how long the forecast paths must be. Everything else is a
    lookup into the per-day term structure, so adding a horizon costs nothing at
    simulation time.
    """

    specs: tuple[HorizonSpec, ...]
    #: hard ceiling: beyond ~2 years a 5-year sample has an effective sample
    #: size of about 2, which is extrapolation rather than estimation
    cap_days: int = 756

    def __post_init__(self) -> None:
        if not self.specs:
            raise ConfigError(["at least one forecast horizon is required"])
        over = [s for s in self.specs if s.days > self.cap_days]
        if over:
            raise ConfigError([
                f"horizon {s.label} ({s.days} trading days) exceeds the {self.cap_days}-day cap; "
                f"raise cap_days explicitly if you accept that this is extrapolation"
                for s in over
            ])

    @classmethod
    def parse(cls, spec: str | Iterable[str | int], *, periods_per_year: float = 252.0,
              cap_days: int = 756) -> "HorizonGrid":
        tokens = spec.split(",") if isinstance(spec, str) else list(spec)
        parsed = [HorizonSpec.parse(t, periods_per_year=periods_per_year)
                  for t in tokens if str(t).strip() != ""]
        unique: dict[int, HorizonSpec] = {}
        for s in parsed:
            unique.setdefault(s.days, s)          # first label for a length wins
        ordered = tuple(sorted(unique.values(), key=lambda s: s.days))
        return cls(specs=ordered, cap_days=cap_days)

    @classmethod
    def default(cls, *, periods_per_year: float = 252.0) -> "HorizonGrid":
        return cls.parse("1d,2d,1w,1m,3m,6m,1y", periods_per_year=periods_per_year)

    @property
    def max_days(self) -> int:
        return max(s.days for s in self.specs)

    @property
    def days(self) -> tuple[int, ...]:
        return tuple(s.days for s in self.specs)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.specs)

    def label_for(self, days: int) -> str:
        for s in self.specs:
            if s.days == days:
                return s.label
        return _DEFAULT_LABELS.get(days, f"{days}D")

    def with_custom(self, days: int) -> "HorizonGrid":
        """
        Add an ad-hoc horizon (what the Excel/HTML 'custom days' box does).

        Existing specs keep their labels: re-parsing them from day counts would turn
        '3M' into '62D' whenever the calendar is not exactly 252 sessions a year.
        """
        if int(days) in self.days:
            return self
        new = HorizonSpec(days=int(days), label=_DEFAULT_LABELS.get(int(days), f"{int(days)}D"))
        return HorizonGrid(specs=tuple(sorted(self.specs + (new,), key=lambda s: s.days)),
                           cap_days=self.cap_days)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self):
        return iter(self.specs)
