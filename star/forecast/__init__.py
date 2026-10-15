"""
The forward distribution: what this portfolio might be worth, day by day.

One simulation run produces **one** object, stored per day out to the maximum
horizon, and every horizon view in every output reads from it. A 2-day view and a
37-day view are lookups, not re-runs, which is what makes the horizon control in
the brief instant - and what makes it impossible for the HTML page, the Excel
pack and the engine to disagree with each other.

Legacy defects addressed (docs/AUDIT_legacy.md): independent t marginals mixed
through a Cholesky factor, i.i.d. days with constant covariance (M5-01/02), and -
the largest number-mover of all - a drift taken from the optimiser's own expected
returns, so the simulation confirmed the assumption it was given (M5-03).
"""
