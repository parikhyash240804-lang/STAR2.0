"""
Process-level setup the legacy monolith performed at import time
(stardust_core.py lines 1-17), gathered in one idempotent function.
"""
import os
import sys

_DONE = False


def ensure_bootstrap() -> None:
    global _DONE
    if _DONE:
        return
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # windowed PyInstaller builds have no console streams
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")

    # Legacy behaviour, kept for parity: TLS certificate verification disabled
    # for every HTTPS request in the process. STAR2.0 does NOT do this.
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    _DONE = True
