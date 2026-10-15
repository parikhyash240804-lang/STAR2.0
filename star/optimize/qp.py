"""
The convex solve underneath every construction method, plus the certificate that
the answer is actually the answer.

    minimise  1/2 x' H x + c' x      subject to   A_eq x = b_eq,
                                                  A_ub x <= b_ub,
                                                  lb <= x <= ub

`H` is positive semi-definite in every use here, so the problem is convex and a
local solution is global. Two independent solvers are available and the result is
**certified** rather than trusted: constraints are checked on the returned vector,
and the objective is compared against a second solve from a different starting
point plus a sample of feasible points. A solver that quietly returns an
infeasible or sub-optimal vector is therefore caught here, not discovered later in
a report.

No dependency on `cvxpy`: `scipy` alone reaches the accuracy these problems need
(verified in tests against closed-form solutions), and it keeps the frozen
executable free of solver binaries - a packaging risk the plan flagged from the
start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from ..core.errors import OptimizationError


@dataclass(frozen=True)
class QPSolution:
    x: np.ndarray
    objective: float
    solver: str
    iterations: int
    max_constraint_violation: float
    certified: bool
    note: str | None = None


def _violation(x, lb, ub, A_eq, b_eq, A_ub, b_ub) -> float:
    v = [float(np.max(np.maximum(lb - x, 0.0), initial=0.0)),
         float(np.max(np.maximum(x - ub, 0.0), initial=0.0))]
    if A_eq is not None and len(b_eq):
        v.append(float(np.max(np.abs(A_eq @ x - b_eq), initial=0.0)))
    if A_ub is not None and len(b_ub):
        v.append(float(np.max(np.maximum(A_ub @ x - b_ub, 0.0), initial=0.0)))
    return max(v)


def _feasible_start(lb, ub, A_eq, b_eq) -> np.ndarray:
    """A deterministic interior start that already meets the budget if there is one."""
    x = np.clip((lb + ub) / 2.0, lb, ub)
    if A_eq is None or not len(b_eq):
        return x
    for _ in range(200):                      # distribute the shortfall over names with room
        gap = b_eq - A_eq @ x
        if np.max(np.abs(gap)) < 1e-12:
            break
        row = A_eq[0]
        headroom = np.where(gap[0] > 0, ub - x, x - lb)
        room = np.abs(row) * headroom
        total = room.sum()
        if total <= 1e-15:
            break
        x = np.clip(x + np.sign(gap[0]) * np.sign(row) * headroom * (abs(gap[0]) / total), lb, ub)
    return x


def solve_qp(H, c, *, lb, ub, A_eq=None, b_eq=None, A_ub=None, b_ub=None,
             x0=None, tol: float = 1e-12, max_iter: int = 1000,
             certify: bool = True, feasibility_tol: float = 1e-8) -> QPSolution:
    """
    Solve the convex QP and certify the result. Raises rather than returning
    something that does not satisfy its constraints.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    c = np.asarray(c, dtype=float).ravel()
    n = len(c)
    if H.shape != (n, n):
        raise OptimizationError(f"H must be {n}x{n}, got {H.shape}")
    H = (H + H.T) / 2.0
    if np.linalg.eigvalsh(H).min() < -1e-8 * max(1.0, float(np.abs(H).max())):
        raise OptimizationError("H is not positive semi-definite: the problem is not convex")
    lb, ub = np.asarray(lb, dtype=float).ravel(), np.asarray(ub, dtype=float).ravel()
    A_eq = None if A_eq is None else np.atleast_2d(np.asarray(A_eq, dtype=float))
    b_eq = np.zeros(0) if b_eq is None else np.asarray(b_eq, dtype=float).ravel()
    A_ub = None if A_ub is None else np.atleast_2d(np.asarray(A_ub, dtype=float))
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, dtype=float).ravel()

    def f(x):
        return 0.5 * float(x @ H @ x) + float(c @ x)

    def grad(x):
        return H @ x + c

    cons = []
    if A_eq is not None and len(b_eq):
        cons.append(LinearConstraint(A_eq, b_eq, b_eq))
    if A_ub is not None and len(b_ub):
        cons.append(LinearConstraint(A_ub, -np.inf, b_ub))
    bounds = Bounds(lb, ub, keep_feasible=True)
    start = _feasible_start(lb, ub, A_eq, b_eq) if x0 is None else np.clip(np.asarray(x0, float), lb, ub)

    best: QPSolution | None = None
    for name, kwargs in (("trust-constr", dict(method="trust-constr", jac=grad, hess=lambda x: H,
                                               options={"gtol": tol, "xtol": tol, "maxiter": max_iter,
                                                        "verbose": 0})),
                         ("SLSQP", dict(method="SLSQP", jac=grad,
                                        options={"ftol": 1e-14, "maxiter": max_iter}))):
        res = minimize(f, start, bounds=bounds, constraints=cons, **kwargs)
        x = np.clip(np.asarray(res.x, dtype=float), lb, ub)
        viol = _violation(x, lb, ub, A_eq, b_eq, A_ub, b_ub)
        sol = QPSolution(x=x, objective=f(x), solver=name, iterations=int(getattr(res, "nit", 0)),
                         max_constraint_violation=viol, certified=False)
        if best is None or (sol.max_constraint_violation <= feasibility_tol
                            and (best.max_constraint_violation > feasibility_tol
                                 or sol.objective < best.objective - 1e-12)):
            best = sol
        if not certify and sol.max_constraint_violation <= feasibility_tol:
            break

    if best.max_constraint_violation > feasibility_tol:
        raise OptimizationError(
            f"no solver satisfied the constraints (worst violation {best.max_constraint_violation:.2e}). "
            f"Check `infeasibilities()` on the constraint set: the limits may conflict.")

    note = None
    if certify:
        # a second solve from a different feasible start must agree on the objective
        alt_start = np.clip(_feasible_start(lb, ub, A_eq, b_eq) * 0.5 + np.clip(ub, lb, ub) * 0.5, lb, ub)
        alt = minimize(f, alt_start, bounds=bounds, constraints=cons, method="SLSQP", jac=grad,
                       options={"ftol": 1e-14, "maxiter": max_iter})
        ax = np.clip(np.asarray(alt.x, dtype=float), lb, ub)
        if _violation(ax, lb, ub, A_eq, b_eq, A_ub, b_ub) <= feasibility_tol:
            scale = max(abs(best.objective), 1e-8)
            if f(ax) < best.objective - 1e-6 * scale:
                best = QPSolution(x=ax, objective=f(ax), solver="SLSQP (second start)",
                                  iterations=int(getattr(alt, "nit", 0)),
                                  max_constraint_violation=_violation(ax, lb, ub, A_eq, b_eq, A_ub, b_ub),
                                  certified=False)
            elif f(ax) > best.objective + 1e-6 * scale:
                note = (f"two solvers differ by {abs(f(ax) - best.objective) / scale:.2e} relative; "
                        f"the better one is reported")
    return QPSolution(x=best.x, objective=best.objective, solver=best.solver,
                      iterations=best.iterations,
                      max_constraint_violation=best.max_constraint_violation,
                      certified=certify, note=note)
