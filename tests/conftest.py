"""
Shared test setup: make `star` and `legacy` importable without installation, and
stop the numeric libraries fighting each other over cores.

The thread pinning is not cosmetic. Measured on this machine, a run that had
already exercised the data layer took **146 seconds** for a GARCH-plus-optimiser
test that takes 2.5 seconds on its own - a 60x slowdown, and it moved from test
to test depending on ordering, which is what makes it hard to spot. The cause is
over-subscription: pandas/pyarrow, OpenBLAS and SciPy each start their own pool,
and on small matrices the coordination costs far more than the arithmetic saves.
The same effect applies to the shipped application, which also reads parquet and
then does small-matrix linear algebra, so the entry points set these too.

These must be set **before numpy is imported**, which is why they are at the top
of conftest rather than in a fixture.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ARROW_IO_THREADS"):
    os.environ.setdefault(_var, "1")

import sys                                                          # noqa: E402
from pathlib import Path                                            # noqa: E402

STAR2 = Path(__file__).resolve().parents[1]
if str(STAR2) not in sys.path:
    sys.path.insert(0, str(STAR2))

try:                                                                # pragma: no cover
    import pyarrow                                                  # noqa: E402

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
except Exception:                                                   # pyarrow is optional
    pass
