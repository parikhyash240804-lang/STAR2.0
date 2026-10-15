"""
Parity harness for Task 1: prove the modular legacy engine (legacy/stardust_v1)
reproduces the original monolith (stardust_core.py) exactly.

yfinance is replaced by a record/replay proxy so both engines see byte-identical
market data, and wall-clock "now" is frozen so date windows match.

  python parity_check.py record  <out_dir>   # original monolith, live data -> cache
  python parity_check.py orig    <out_dir>   # original monolith, replayed data
  python parity_check.py mod     <out_dir>   # modular engine,   replayed data
  python parity_check.py compare <dir_a> <dir_b>

Every run writes <out_dir>/stdout.txt and the engine outputs. The cache lives
next to this file's --cache path (default: <out_dir>/../yf_cache.pkl).

Side effects of the legacy report (copy to ~/Downloads, auto-open Excel) are
neutralised inside the harness process.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import importlib
import importlib.util
import io
import math
import os
import pickle
import re
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAR2 = HERE.parents[1]                 # .../STAR2.0
APP_ROOT = STAR2.parent                 # .../StardustApp (legacy monolith lives here)

# Mirrors stardust_app.DEFAULT_TICKERS / DEFAULT_SECTOR_MULTIPLIERS / DEFAULT_STRESS_SCENARIOS
DEFAULT_TICKERS = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'RELIANCE.NS', 'MARUTI.NS', 'INFY.NS',
    'LT.NS', 'POWERGRID.NS', 'HCLTECH.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
    'LUPIN.NS', 'ITC.NS', 'LALPATHLAB.NS', 'HAVELLS.NS', 'JINDALSTEL.NS',
    'TATASTEEL.NS', 'UNITDSPR.NS', 'TCI.NS', 'CIPLA.NS', 'GLAND.NS',
    'RAMCOCEM.NS', 'BLUEDART.NS', 'KPIL.NS', 'HYUNDAI.NS', 'REDINGTON.NS',
    'CHALET.NS', 'PVRINOX.NS', 'CEATLTD.NS', 'TIMKEN.NS', 'FINCABLES.NS',
    'GATEWAY.NS', 'GESHIP.NS', 'ANUP.NS', 'FINPIPE.NS', 'PRESTIGE.NS',
    'PNCINFRA.NS', 'CYIENT.NS', 'KEC.NS', 'NILKAMAL.NS',
]
DEFAULT_PARAMS = {
    'SECTOR_MULTIPLIERS': {
        'Financial Services': 1.15, 'Healthcare': 1.10, 'Industrials': 1.12, 'Energy': 1.05,
        'Consumer Defensive': 1.00, 'Utilities': 1.00, 'Real Estate': 1.00,
        'Communication Services': 1.00, 'Technology': 0.92, 'Basic Materials': 0.88,
        'Consumer Cyclical': 0.95,
    },
    'STRESS_SCENARIOS': {
        'COVID Crash (Mar 2020)': -0.38, '2008 Crisis (Lehman)': -0.60,
        'Rate Shock (2022)': -0.17, 'Russia-Ukraine (Feb 2022)': -0.15,
    },
}


# ----------------------------------------------------------------------------
# yfinance record / replay proxy
# ----------------------------------------------------------------------------
def _norm(v):
    import pandas as pd
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        return str(pd.Timestamp(v).date())
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return repr(v)


def _key(name, args, kwargs):
    return (name, tuple(_norm(a) for a in args),
            tuple(sorted((k, _norm(v)) for k, v in kwargs.items())))


class YFProxy:
    def __init__(self, mode: str, cache: dict):
        self.mode = mode
        self.cache = cache
        self.real = importlib.import_module('yfinance') if mode == 'record' else None
        self.misses = []

    def _fetch(self, key, getter):
        if self.mode == 'replay':
            if key not in self.cache:
                self.misses.append(key)
                raise RuntimeError(f'parity cache miss: {key}')
            kind, val = self.cache[key]
            if kind == 'exc':
                raise RuntimeError(val)
            return copy.deepcopy(val)
        try:
            val = getter()
        except Exception as e:           # record the failure, replay it later
            self.cache[key] = ('exc', f'{type(e).__name__}: {e}')
            raise
        try:
            self.cache[key] = ('ok', copy.deepcopy(val))
        except Exception:
            self.cache[key] = ('ok', val)
        return val

    def download(self, *a, **k):
        return self._fetch(_key('download', a, k), lambda: self.real.download(*a, **k))

    def Ticker(self, t):
        return _FakeTicker(self, t)

    def as_module(self):
        m = types.ModuleType('yfinance')
        m.download = self.download
        m.Ticker = self.Ticker
        m.__version__ = 'parity-proxy'
        return m


class _FakeTicker:
    def __init__(self, proxy: YFProxy, t: str):
        self._p, self._t = proxy, t

    def _real(self):
        return self._p.real.Ticker(self._t)

    @property
    def info(self):
        return self._p._fetch(('Ticker', self._t, 'info'), lambda: dict(self._real().info))

    @property
    def quarterly_earnings(self):
        return self._p._fetch(('Ticker', self._t, 'quarterly_earnings'),
                              lambda: self._real().quarterly_earnings)

    @property
    def fast_info(self):
        mc = self._p._fetch(('Ticker', self._t, 'fast_info.market_cap'),
                            lambda: getattr(self._real().fast_info, 'market_cap', None))
        return types.SimpleNamespace(market_cap=mc)


# ----------------------------------------------------------------------------
# environment neutralisation
# ----------------------------------------------------------------------------
def _freeze_now(frozen_iso: str):
    """
    Freeze every clock the engine reads: pandas' and the stdlib's.

    Missing the stdlib one is subtle and costly: M6 builds its benchmark window
    with `dt.datetime.now()`, so a replay on a different calendar day asks for a
    different date range, misses the cache, loses every beta, and silently
    changes the stress numbers and one chart.
    """
    import pandas as pd
    frozen = pd.Timestamp(frozen_iso)
    local_tz = _dt.datetime.now().astimezone().tzinfo

    def _now(cls, tz=None):
        if tz is None:
            return frozen
        return frozen.tz_localize(local_tz).tz_convert(tz)   # honour tz like the real one

    try:
        pd.Timestamp.now = classmethod(_now)
        pd.Timestamp.today = classmethod(_now)
    except TypeError:
        print('[parity] WARNING: could not freeze pd.Timestamp.now', file=sys.__stderr__)

    # The stdlib clock is deliberately NOT patched. Replacing datetime.datetime /
    # datetime.date with subclasses breaks dateutil, which resolves
    # `datetime.date` at call time and then rejects a real pd.Timestamp - so
    # pd.DateOffset arithmetic fails. Instead the harness requires that replays
    # run on the same calendar day as the recording (see _check_same_day), since
    # M6's `dt.datetime.now()` leaks into a yfinance cache key.


def _check_same_day(frozen_iso: str) -> None:
    """
    Refuse to replay a cache recorded on another day.

    M6 derives its benchmark window from the stdlib clock, which we cannot freeze
    safely, so the request lands in the yfinance cache key. Replaying across a
    date boundary silently loses every beta and changes the stress numbers - a
    difference that looks like a port defect but is pure harness drift.
    """
    rec = _dt.datetime.fromisoformat(frozen_iso).date()
    today = _dt.date.today()
    if rec != today:
        raise SystemExit(
            f"[parity] cache was recorded on {rec} but today is {today}.\n"
            f"          M6's benchmark window comes from the stdlib clock, so the\n"
            f"          keys will not match. Re-record first:\n"
            f"            python parity_check.py record <out_dir>")


def _neutralise_side_effects(out_dir: Path):
    fake_home = out_dir / '_home'
    (fake_home / 'Downloads').mkdir(parents=True, exist_ok=True)
    Path.home = classmethod(lambda cls: fake_home)          # report copies to ~/Downloads
    if hasattr(os, 'startfile'):
        os.startfile = lambda *a, **k: None                  # report auto-opens Excel


def run_engine(which: str, out_dir: Path, cache_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = 'record' if which == 'record' else 'replay'
    cache = {}
    if mode == 'replay':
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        frozen = cache['__meta__'][1]['now']
    else:
        frozen = _dt.datetime.now().isoformat(timespec='seconds')
        cache['__meta__'] = ('ok', {'now': frozen})

    proxy = YFProxy(mode, cache)
    real_yf = sys.modules.get('yfinance')
    sys.modules['yfinance'] = proxy.as_module()
    if mode == 'replay':          # the record run defines "now"; replays reuse it
        _check_same_day(frozen)
        _freeze_now(frozen)
    _neutralise_side_effects(out_dir)

    if which in ('record', 'orig'):
        spec = importlib.util.spec_from_file_location('stardust_core', APP_ROOT / 'stardust_core.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        engine = mod.run_stardust
    else:
        sys.path.insert(0, str(STAR2))
        from legacy.stardust_v1 import run_stardust as engine

    buf = io.StringIO()
    status = []
    t0 = time.time()
    err = None
    with contextlib.redirect_stdout(buf):
        try:
            engine(params=copy.deepcopy(DEFAULT_PARAMS), tickers=list(DEFAULT_TICKERS),
                   output_dir=out_dir, status_callback=status.append)
        except Exception as e:                       # keep going: record how far it got
            import traceback
            err = traceback.format_exc()
    (out_dir / 'stdout.txt').write_text(buf.getvalue(), encoding='utf-8')
    (out_dir / 'status.txt').write_text('\n'.join(status) + (f'\n\nERROR:\n{err}' if err else ''),
                                        encoding='utf-8')
    if mode == 'record':
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
    if real_yf is not None:
        sys.modules['yfinance'] = real_yf
    print(f'[parity] {which}: {time.time() - t0:.0f}s, status={status[-1] if status else None}, '
          f'error={"yes" if err else "no"}, cache_entries={len(cache)}, misses={len(proxy.misses)}',
          file=sys.__stdout__)
    for miss in proxy.misses:          # a miss silently degrades the run - name it
        print(f'[parity] CACHE MISS: {miss}', file=sys.__stdout__)
    if err:
        print(err, file=sys.__stdout__)


# ----------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------
_VOLATILE = re.compile(r'(\d{1,2}:\d{2}:\d{2})|(\d+(\.\d+)?\s*s\b)|elapsed|took|seconds|Copied to|Saved:|\bKB\b', re.I)


def _close(a, b, rtol=1e-9, atol=1e-12):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def compare(dir_a: Path, dir_b: Path) -> int:
    problems = 0

    def canon(v):
        """Each engine writes into its own folder, so absolute output paths
        differ by construction. Normalise them; everything else must match."""
        if isinstance(v, str):
            for d in (str(dir_a.resolve()), str(dir_b.resolve()), dir_a.name, dir_b.name):
                v = v.replace(d, '<OUT>')
        return v
    # 1) stdout (the engine prints every intermediate table, so this is a dense check)
    la = [canon(x) for x in (dir_a / 'stdout.txt').read_text(encoding='utf-8').splitlines()]
    lb = [canon(x) for x in (dir_b / 'stdout.txt').read_text(encoding='utf-8').splitlines()]
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb))
             if x != y and not (_VOLATILE.search(x) and _VOLATILE.search(y))]
    print(f'stdout: {len(la)} vs {len(lb)} lines; {len(diffs)} differing (volatile lines ignored)')
    for i, x, y in diffs[:15]:
        print(f'  L{i}: A| {x}\n        B| {y}')
    problems += len(diffs) + abs(len(la) - len(lb))

    # 2) Excel report, cell by cell
    import openpyxl
    xa, xb = dir_a / 'portfolio_report.xlsx', dir_b / 'portfolio_report.xlsx'
    if xa.exists() and xb.exists():
        wa = openpyxl.load_workbook(xa, data_only=True)
        wb = openpyxl.load_workbook(xb, data_only=True)
        print(f'excel sheets A={wa.sheetnames}\n             B={wb.sheetnames}')
        if wa.sheetnames != wb.sheetnames:
            problems += 1
        cell_diffs = 0
        for name in wa.sheetnames:
            if name not in wb.sheetnames:
                continue
            sa, sb = wa[name], wb[name]
            for row_a, row_b in zip(sa.iter_rows(values_only=True), sb.iter_rows(values_only=True)):
                for va, vb in zip(row_a, row_b):
                    va, vb = canon(va), canon(vb)
                    if isinstance(va, str) and isinstance(vb, str) and _VOLATILE.search(va) and _VOLATILE.search(vb):
                        continue
                    if isinstance(va, (_dt.datetime, _dt.date)):
                        continue
                    if not _close(va, vb):
                        cell_diffs += 1
                        if cell_diffs <= 15:
                            print(f'  [{name}] A={va!r}  B={vb!r}')
        print(f'excel: {cell_diffs} differing cells')
        problems += cell_diffs
    else:
        print(f'excel: missing (A={xa.exists()}, B={xb.exists()})')
        problems += int(xa.exists() != xb.exists())

    # 3) charts, pixel by pixel
    import matplotlib.image as mpimg
    import numpy as np
    for png in sorted(p.name for p in dir_a.glob('*.png')):
        pa, pb = dir_a / png, dir_b / png
        if not pb.exists():
            print(f'  chart {png}: missing in B'); problems += 1; continue
        ia, ib = mpimg.imread(pa), mpimg.imread(pb)
        same = ia.shape == ib.shape and np.array_equal(ia, ib)
        print(f'  chart {png}: {"identical" if same else "DIFFERENT"}')
        problems += (not same)
    print('PARITY OK' if problems == 0 else f'PARITY FAILED ({problems} problems)')
    return 0 if problems == 0 else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['record', 'orig', 'mod', 'compare'])
    ap.add_argument('a')
    ap.add_argument('b', nargs='?')
    ap.add_argument('--cache', default=None)
    ns = ap.parse_args()
    if ns.cmd == 'compare':
        sys.exit(compare(Path(ns.a), Path(ns.b)))
    out = Path(ns.a).resolve()
    cache_path = Path(ns.cache) if ns.cache else out.parent / 'yf_cache.pkl'
    run_engine(ns.cmd, out, cache_path)
