"""
On-disk data cache: faster runs, reproducible runs, and a data fingerprint.

Legacy defect M0-15: `END_DATE = now()` and live downloads on every run made
results irreproducible while the banner advertised reproducibility. STAR2.0 runs
against an explicit as-of date, and every dataset it reads is stored with a
SHA-256 fingerprint, so a report can name exactly which data produced it and the
same run can be repeated offline.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core.errors import DataError
from .provider import GUARD, PricePanel

_PANEL_FIELDS = ("close", "close_raw", "high", "low", "open_", "volume")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class DataCache:
    """Parquet files plus a JSON manifest per entry, verified on every read."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts) -> str:
        text = "|".join(str(p) for p in parts)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.parquet", self.root / f"{key}.json"

    def put(self, key: str, frame: pd.DataFrame, *, meta: dict) -> str:
        """Store a frame; returns its SHA-256 fingerprint."""
        buf = io.BytesIO()
        frame.to_parquet(buf, compression="zstd")
        data = buf.getvalue()
        digest = _digest(data)
        pq, js = self._paths(key)
        pq.write_bytes(data)
        manifest = {**meta, "sha256": digest, "rows": int(len(frame)),
                    "columns": [str(c) for c in frame.columns],
                    "stored_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        js.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        return digest

    def get(self, key: str) -> tuple[pd.DataFrame, dict] | None:
        """Return (frame, manifest), or None if absent or corrupted (and discard it)."""
        pq, js = self._paths(key)
        if not (pq.exists() and js.exists()):
            return None
        data = pq.read_bytes()
        manifest = json.loads(js.read_text(encoding="utf-8"))
        if _digest(data) != manifest.get("sha256"):
            pq.unlink(missing_ok=True)
            js.unlink(missing_ok=True)
            return None
        frame = pd.read_parquet(io.BytesIO(data))
        if isinstance(frame.index, pd.DatetimeIndex) or frame.index.dtype.kind == "M":
            frame.index = pd.DatetimeIndex(frame.index)
        return frame, manifest


@dataclass
class CachedPriceProvider:
    """Wraps any `PriceProvider`; the first request fetches, later ones replay."""

    inner: object
    cache: DataCache
    name: str = "cached"
    fingerprints: dict | None = None

    def __post_init__(self) -> None:
        self.name = f"cached:{getattr(self.inner, 'name', 'provider')}"
        self.fingerprints = {}

    def prices(self, tickers: Sequence[str], start, end, *, adjusted: bool = True) -> PricePanel:
        GUARD.check(end, what=f"{self.name}.prices")
        tickers = list(tickers)
        base = self.cache.key(getattr(self.inner, "name", "provider"), sorted(tickers),
                              pd.Timestamp(start).date(), pd.Timestamp(end).date(), adjusted)
        hits = {f: self.cache.get(f"{base}-{f}") for f in _PANEL_FIELDS}
        if hits["close"] is not None:
            frames = {f: (h[0] if h is not None else None) for f, h in hits.items()}
            self.fingerprints.update({f: h[1]["sha256"] for f, h in hits.items() if h is not None})
            as_of = pd.Timestamp(hits["close"][1]["as_of"])
            source = hits["close"][1]["source"]
        else:
            panel = self.inner.prices(tickers, start, end, adjusted=adjusted)
            frames = {f: getattr(panel, f) for f in _PANEL_FIELDS}
            as_of, source = panel.as_of, panel.source
            for f, frame in frames.items():
                if frame is not None:
                    self.fingerprints[f] = self.cache.put(
                        f"{base}-{f}", frame,
                        meta={"source": source, "as_of": as_of, "field": f,
                              "tickers": tickers, "start": start, "end": end})
        if frames["close"] is None:
            raise DataError("cached panel has no close prices")
        def order(frame):
            if frame is None:
                return None
            cols = [t for t in tickers if t in frame.columns]
            return GUARD.clamp(frame[cols], what=f"{self.name}.frame")
        frames = {f: order(v) for f, v in frames.items()}
        cutoff = GUARD.cutoff
        if cutoff is not None and as_of > cutoff:
            as_of = cutoff
        return PricePanel(close=frames["close"], close_raw=frames["close_raw"], high=frames["high"],
                          low=frames["low"], open_=frames["open_"], volume=frames["volume"],
                          source=source, as_of=as_of)

    def data_fingerprint(self) -> str:
        """One hash covering every dataset read so far - goes into the run's provenance."""
        joined = "|".join(f"{k}:{v}" for k, v in sorted(self.fingerprints.items()))
        return _digest(joined.encode("utf-8"))[:16]
