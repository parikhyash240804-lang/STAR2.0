"""
Build the offline test fixture from a recorded yfinance cache.

The parity harness already records every yfinance response for a full legacy run
(39 NSE tickers, 5 years, plus the benchmark and the midcap index). Reusing that
recording gives the test suite realistic Indian market data with no network
access, and keeps STAR2.0's tests comparable with the legacy baseline.

Usage:
    python build_fixture.py <path/to/yf_cache.pkl> [out_dir]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

FIELDS = ("Open", "High", "Low", "Close", "Volume")


def build(cache_path: Path, out_dir: Path) -> None:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    frames: dict[str, pd.DataFrame] = {}
    singles: dict[str, pd.DataFrame] = {}
    for key, (kind, value) in cache.items():
        if kind != "ok" or not isinstance(key, tuple) or key[0] != "download":
            continue
        if not isinstance(value, pd.DataFrame) or value.empty:
            continue
        symbols = key[1][0] if key[1] else "?"
        n_cols = (len(value.columns.get_level_values(1).unique())
                  if isinstance(value.columns, pd.MultiIndex) else 1)
        (frames if n_cols > 3 else singles)[str(symbols)] = value

    if not frames:
        raise SystemExit("no multi-ticker download found in the cache")

    universe = max(frames.values(), key=lambda d: d.shape[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for fld in FIELDS:
        if isinstance(universe.columns, pd.MultiIndex):
            if fld not in universe.columns.get_level_values(0):
                continue
            sub = universe[fld].copy()
        else:
            continue
        sub.index = pd.DatetimeIndex(sub.index).tz_localize(None)
        sub = sub.sort_index()
        path = out_dir / f"nse_{fld.lower()}.parquet"
        sub.astype("float64").to_parquet(path, compression="zstd")
        written.append((path.name, sub.shape))

    # benchmark / factor indices arrived as single-symbol downloads
    for sym, df in singles.items():
        clean = sym.strip("'\"")
        col = df["Close"] if "Close" in getattr(df, "columns", []) else None
        if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
            col = df["Close"]
        if col is None:
            continue
        s = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        name = clean.replace("^", "").replace(".", "_")
        path = out_dir / f"index_{name}.parquet"
        s.dropna().astype("float64").to_frame("close").to_parquet(path, compression="zstd")
        written.append((path.name, s.shape))

    for name, shape in written:
        print(f"  wrote {name:<28} {shape}")
    print(f"fixture written to {out_dir}")


#: reference series not present in the legacy recording: (symbol, start, why)
REFERENCE = (
    # starts in 2018 on purpose: Yahoo's adjusted series has a broken unit split in
    # Dec 2019 (-89.9% then +897% three sessions later), a real artifact the data
    # quality layer must repair, so the fixture keeps it as a regression case
    ("NIFTYBEES.NS", "2018-01-01", "NIFTY 50 total-return proxy (dividend-adjusted ETF)"),
    ("GOLDBEES.NS", None, "gold ETF, for the gold allocation analysis"),
    ("^INDIAVIX", None, "India VIX, for model-free volatility regime bands"),
)


def build_reference(out_dir: Path) -> None:
    """
    Fetch reference series live, aligned to the existing fixture's end date.

    Stores both the adjusted close (`close`) and the unadjusted close
    (`close_raw`) where they differ, so tests can see what the adjustment did.
    """
    import yfinance as yf

    stock_close = pd.read_parquet(out_dir / "nse_close.parquet")
    fixture_start = pd.DatetimeIndex(stock_close.index).min()
    fixture_end = pd.DatetimeIndex(stock_close.index).max()
    end_exclusive = fixture_end + pd.Timedelta(days=1)   # yfinance `end` is exclusive

    for symbol, start, why in REFERENCE:
        lo = pd.Timestamp(start) if start else fixture_start
        cols = {}
        for adjusted, col in ((True, "close"), (False, "close_raw")):
            df = yf.download(symbol, start=lo, end=end_exclusive, auto_adjust=adjusted,
                             progress=False, multi_level_index=False)
            if df is None or df.empty:
                raise SystemExit(f"no data returned for {symbol}")
            s = df["Close"].dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            cols[col] = s.astype("float64")
        frame = pd.DataFrame(cols).sort_index()
        if (frame["close"] - frame["close_raw"]).abs().max() < 1e-9:
            frame = frame[["close"]]            # index levels: nothing to adjust
        name = symbol.replace("^", "").replace(".", "_")
        path = out_dir / f"index_{name}.parquet"
        frame.to_parquet(path, compression="zstd")
        print(f"  wrote {path.name:<30} {frame.shape}  {frame.index.min().date()}..{frame.index.max().date()}  ({why})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reference":
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "nse"
        build_reference(out)
    else:
        cache = Path(sys.argv[1])
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "nse"
        build(cache, out)
