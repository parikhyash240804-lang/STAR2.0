"""
STAR2.0 configuration: every setting a manager can change, with its default.

The legacy engine read ~50 settings through `params.get(KEY, default)` with no
validation at all (audit M0-16): a swapped VIEW_FLOOR/VIEW_CEILING silently
inverted every view (M2B-18), T_DIST_DF = 2 crashed after all upstream work
(M5-08), and the GUI's reader swallowed bad input with `except: pass`, so a typo
quietly became a default. Validation here collects *every* problem with a
message a user can act on, before any data is downloaded.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import pandas as pd

from ..core.errors import ConfigError
from ..core.horizons import HorizonGrid, HorizonSpec

ENGINES = ("regime_fhs", "block_bootstrap", "mvt")
DRIFTS = ("prior", "zero", "risk_free", "views")
VIEW_STYLES = ("relative_quintile", "none")
SIGNAL_MODES = ("price_only", "with_fundamentals")
REBALANCE_FREQS = ("ME", "3ME", "6ME", "YE")


@dataclass(frozen=True, slots=True)
class Mandate:
    """Limits the portfolio must respect; also the thresholds guidance judges against."""

    max_stock_weight: float = 0.10
    max_sector_weight: float = 0.30
    min_holdings: int = 15
    max_holdings: int = 25                  # genuinely enforced (legacy MAX_ASSETS was not, M3-06)
    max_annual_turnover: float = 1.50       # one-way, 150% a year
    max_drawdown_tolerance: float = 0.20
    max_adv_participation: float = 0.20     # trade at most 20% of average daily value
    max_days_to_exit: float = 5.0           # position must be exitable within 5 sessions
    max_expected_shortfall_1y: float | None = None
    max_tracking_error: float | None = None


@dataclass(frozen=True, slots=True)
class CostModel:
    """Indian delivery-equity trading costs, per leg (editable; FY2025-26 defaults)."""

    brokerage_rate: float = 0.0003
    brokerage_cap_inr: float = 20.0
    stt_delivery: float = 0.001             # both legs
    exchange_txn: float = 0.0000297         # NSE transaction charge
    sebi_fee: float = 0.000001
    stamp_duty_buy: float = 0.00015         # buy leg only
    gst_rate: float = 0.18                  # on brokerage + exchange + SEBI fees
    dp_charge_inr_per_scrip_sell: float = 15.93
    impact_coefficient: float = 1.0         # square-root impact k


@dataclass(frozen=True, slots=True)
class DataSettings:
    benchmark: str = "^NSEI"
    tri_proxy: str = "NIFTYBEES.NS"
    lookback_years: int = 5
    max_missing: float = 0.05
    min_history_sessions: int = 250
    min_adv_inr: float = 1_000_000.0
    risk_free_csv: str | None = None        # None = RBI 91-day T-bill from the IIMA file
    iima_factor_file: str | None = None     # None = the bundled IIMA release
    risk_free_fallback: float = 0.065       # used only if no dated source loads, and flagged


@dataclass(frozen=True, slots=True)
class ForecastSettings:
    horizons: str = "1d,2d,1w,1m,3m,6m,1y"
    default_horizon: str = "3m"
    engine: str = "regime_fhs"
    n_scenarios: int = 20_000
    drift: str = "prior"
    seed: int = 42
    max_bytes: int = 512 * 2**20


@dataclass(frozen=True, slots=True)
class ReturnSettings:
    target_market_sharpe: float = 0.35
    tau: float = 0.05
    view_style: str = "relative_quintile"
    view_confidence: float = 0.30


@dataclass(frozen=True, slots=True)
class BacktestSettings:
    rebalance: str = "3ME"
    window_years: int = 3
    signals_mode: str = "price_only"
    turnover_penalty: float = 0.001
    no_trade_band: float = 0.005


_SUFFIXED = re.compile(r"^[A-Z0-9&^.\-]+\.(NS|BO)$")


def normalise_tickers(raw) -> tuple[tuple[str, ...], list[str]]:
    """Upper-case, add `.NS` when no exchange suffix is given, de-duplicate in order."""
    out, notes, seen = [], [], set()
    for item in raw or ():
        t = str(item).strip().upper()
        if not t:
            continue
        if not t.endswith((".NS", ".BO")) and not t.startswith("^"):
            t = f"{t}.NS"
        if not (_SUFFIXED.match(t) or t.startswith("^")):
            notes.append(f"ticker {item!r} does not look like an NSE/BSE symbol")
            continue
        if t in seen:
            notes.append(f"duplicate ticker {t} removed")
            continue
        seen.add(t)
        out.append(t)
    return tuple(out), notes


def _range(issues, name, value, lo=None, hi=None, *, lo_open=False, hi_open=False):
    if value is None:
        issues.append(f"{name} is required")
        return
    if isinstance(value, float) and not math.isfinite(value):
        # NaN fails every comparison, so without this it would pass every range check
        issues.append(f"{name} = {value} is not a finite number")
        return
    if lo is not None and (value <= lo if lo_open else value < lo):
        issues.append(f"{name} = {value} must be {'>' if lo_open else '>='} {lo}")
    if hi is not None and (value >= hi if hi_open else value > hi):
        issues.append(f"{name} = {value} must be {'<' if hi_open else '<='} {hi}")


def _choice(issues, name, value, allowed):
    if value not in allowed:
        issues.append(f"{name} = {value!r} must be one of: {', '.join(allowed)}")


@dataclass(frozen=True, slots=True)
class StarConfig:
    tickers: tuple[str, ...]
    initial_value: float = 1_000_000.0
    as_of: str | None = None                # None = latest available session
    data: DataSettings = field(default_factory=DataSettings)
    forecast: ForecastSettings = field(default_factory=ForecastSettings)
    returns: ReturnSettings = field(default_factory=ReturnSettings)
    mandate: Mandate = field(default_factory=Mandate)
    costs: CostModel = field(default_factory=CostModel)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)

    # ---- derived ----
    def horizon_grid(self, *, periods_per_year: float = 252.0) -> HorizonGrid:
        grid = HorizonGrid.parse(self.forecast.horizons, periods_per_year=periods_per_year)
        default = HorizonSpec.parse(self.forecast.default_horizon, periods_per_year=periods_per_year)
        return grid if default.days in grid.days else grid.with_custom(default.days)

    # ---- validation ----
    def issues(self) -> list[str]:
        """Every problem with this configuration, in words a user can act on."""
        out: list[str] = []
        n = len(self.tickers)
        if n < 3:
            out.append(f"at least 3 tickers are needed, got {n}")
        if n > 150:
            out.append(f"at most 150 tickers are supported, got {n}")
        _range(out, "initial_value", self.initial_value, 0, lo_open=True)
        if self.as_of is not None:
            # strict ISO: "today" would make runs irreproducible, and 01/02/2024 is ambiguous
            # pandas accepts "today"/"now" even with an explicit format, so check the shape first
            iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(self.as_of))
            if not iso or pd.isna(pd.to_datetime(str(self.as_of), format="%Y-%m-%d", errors="coerce")):
                out.append(f"as_of = {self.as_of!r} must be a date written YYYY-MM-DD")

        d = self.data
        _range(out, "data.lookback_years", d.lookback_years, 1, 20)
        _range(out, "data.max_missing", d.max_missing, 0, 0.5)
        _range(out, "data.min_history_sessions", d.min_history_sessions, 60)
        _range(out, "data.min_adv_inr", d.min_adv_inr, 0)
        _range(out, "data.risk_free_fallback", d.risk_free_fallback, -0.05, 0.30)
        if d.risk_free_csv and not Path(d.risk_free_csv).exists():
            out.append(f"data.risk_free_csv file not found: {d.risk_free_csv}")
        if d.iima_factor_file and not Path(d.iima_factor_file).exists():
            out.append(f"data.iima_factor_file file not found: {d.iima_factor_file}")

        f = self.forecast
        try:
            self.horizon_grid()
        except ConfigError as e:
            out.extend(f"forecast: {i}" for i in e.issues)
        except (TypeError, AttributeError, ValueError):
            out.append(f"forecast.horizons / default_horizon could not be read "
                       f"({f.horizons!r}, {f.default_horizon!r})")
        _choice(out, "forecast.engine", f.engine, ENGINES)
        _choice(out, "forecast.drift", f.drift, DRIFTS)
        _range(out, "forecast.n_scenarios", f.n_scenarios, 1_000, 200_000)
        _range(out, "forecast.max_bytes", f.max_bytes, 16 * 2**20)

        r = self.returns
        _range(out, "returns.target_market_sharpe", r.target_market_sharpe, 0.05, 1.0)
        _range(out, "returns.tau", r.tau, 0, 1, lo_open=True)
        _choice(out, "returns.view_style", r.view_style, VIEW_STYLES)
        _range(out, "returns.view_confidence", r.view_confidence, 0, 1, lo_open=True)

        m = self.mandate
        _range(out, "mandate.max_stock_weight", m.max_stock_weight, 0, 1, lo_open=True)
        _range(out, "mandate.max_sector_weight", m.max_sector_weight, 0, 1, lo_open=True)
        if m.max_sector_weight < m.max_stock_weight:
            out.append("mandate.max_sector_weight cannot be below max_stock_weight")
        _range(out, "mandate.min_holdings", m.min_holdings, 1)
        if m.max_holdings < m.min_holdings:
            out.append(f"mandate.max_holdings ({m.max_holdings}) is below min_holdings ({m.min_holdings})")
        reachable = min(m.max_holdings, n) if n else m.max_holdings
        if isinstance(m.max_stock_weight, (int, float)) and math.isfinite(m.max_stock_weight)                 and m.max_stock_weight > 0 and m.max_stock_weight * reachable < 1.0:
            # the legacy never checked this (M3-07): the weights could not sum to 100%
            need = math.ceil(1.0 / m.max_stock_weight - 1e-9)
            out.append(f"infeasible mandate: at most {reachable} holdings capped at "
                       f"{m.max_stock_weight:.0%} each cannot reach a fully invested portfolio "
                       f"(need at least {need} holdings)")
        if n and m.min_holdings > n:
            out.append(f"mandate.min_holdings ({m.min_holdings}) exceeds the number of tickers ({n})")
        _range(out, "mandate.max_annual_turnover", m.max_annual_turnover, 0, lo_open=True)
        _range(out, "mandate.max_drawdown_tolerance", m.max_drawdown_tolerance, 0, 1, lo_open=True, hi_open=True)
        _range(out, "mandate.max_adv_participation", m.max_adv_participation, 0, 1, lo_open=True)
        _range(out, "mandate.max_days_to_exit", m.max_days_to_exit, 0, lo_open=True)

        c = self.costs
        for name in ("brokerage_rate", "stt_delivery", "exchange_txn", "sebi_fee", "stamp_duty_buy"):
            _range(out, f"costs.{name}", getattr(c, name), 0, 0.05)
        _range(out, "costs.gst_rate", c.gst_rate, 0, 0.5)
        _range(out, "costs.brokerage_cap_inr", c.brokerage_cap_inr, 0)
        _range(out, "costs.dp_charge_inr_per_scrip_sell", c.dp_charge_inr_per_scrip_sell, 0)
        _range(out, "costs.impact_coefficient", c.impact_coefficient, 0, 10)

        b = self.backtest
        _choice(out, "backtest.rebalance", b.rebalance, REBALANCE_FREQS)
        _choice(out, "backtest.signals_mode", b.signals_mode, SIGNAL_MODES)
        _range(out, "backtest.window_years", b.window_years, 1)
        if b.window_years >= d.lookback_years:
            out.append(f"backtest.window_years ({b.window_years}) must be shorter than "
                       f"data.lookback_years ({d.lookback_years})")
        _range(out, "backtest.turnover_penalty", b.turnover_penalty, 0, 0.1)
        _range(out, "backtest.no_trade_band", b.no_trade_band, 0, 0.2)
        return out

    def validated(self) -> "StarConfig":
        problems = self.issues()
        if problems:
            raise ConfigError(problems)
        return self

    # ---- flat key access, used by the Excel template ----
    _SECTIONS = ("data", "forecast", "returns", "mandate", "costs", "backtest")

    def flat(self) -> dict[str, object]:
        out = {"initial_value": self.initial_value, "as_of": self.as_of}
        for sec in self._SECTIONS:
            for k, v in asdict(getattr(self, sec)).items():
                out[f"{sec}.{k}"] = v
        return out

    @classmethod
    def field_types(cls) -> dict[str, type]:
        types: dict[str, type] = {"initial_value": float, "as_of": str}
        defaults = cls(tickers=())
        for sec in cls._SECTIONS:
            for fl in fields(getattr(defaults, sec)):
                default = getattr(getattr(defaults, sec), fl.name)
                types[f"{sec}.{fl.name}"] = type(default) if default is not None else float
        types["data.risk_free_csv"] = str
        types["data.iima_factor_file"] = str
        return types

    @classmethod
    def from_flat(cls, tickers, values: dict[str, object]) -> "StarConfig":
        base = cls(tickers=tuple(tickers))
        top = {k: v for k, v in values.items() if "." not in k}
        cfg = replace(base, **top)
        for sec in cls._SECTIONS:
            sub = {k.split(".", 1)[1]: v for k, v in values.items() if k.startswith(sec + ".")}
            if sub:
                cfg = replace(cfg, **{sec: replace(getattr(cfg, sec), **sub)})
        return cfg
