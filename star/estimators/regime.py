"""
Regimes: a two-state Student-t hidden Markov model, plus the India VIX as an
independent, model-free second opinion.

Markets alternate between quiet stretches and turbulent ones, and the same
portfolio carries very different risk in each. The legacy had no regime model at
all - it used a five-bucket step function on a two-point 126-day spread whose
standard error is wider than every one of its buckets (M0-07), which is a coin
toss with a confident label.

What this module does:

* **Two states**, calm and stress, labelled deterministically by fitted
  volatility - never by whichever state the optimiser happened to number first.
* **Student-t emissions.** Daily equity returns have tails a Gaussian cannot
  represent; forcing a Gaussian makes the model invent extra regime switches to
  explain single large days.
* **Filtered probabilities only.** The probability reported for day t uses data
  up to and including t, never after. Smoothed probabilities (which condition on
  the whole sample) are not implemented at all, because using them in a backtest
  is look-ahead - and a function that does not exist cannot be called by mistake.
* **Deterministic.** The starting partition comes from the data (a volatility
  split), not from a random seed, so two runs on the same window agree exactly.

An honest boundary worth stating: within one fit, the *parameters* are estimated
by Baum-Welch over the whole window, so a historical timeline shows filtered
probabilities under parameters that saw the full window. The backtest refits at
each rebalance on data available then, so the decision path has no look-ahead;
the displayed history is labelled for what it is.

Expected duration of a state is 1 / (1 - p_ii).

See PAPERS/notes/01_regime_markov.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from ..core.errors import EstimationError
from ..core.types import Provenance

CALM, STRESS = 0, 1


@dataclass(frozen=True, slots=True)
class RegimeSettings:
    max_iter: int = 300
    tol: float = 1e-7
    min_sessions: int = 250
    nu_bounds: tuple[float, float] = (2.5, 40.0)
    min_scale: float = 1e-8


@dataclass(frozen=True)
class RegimeFit:
    """Fitted two-state model and the filtered probability path it implies."""

    mu: np.ndarray                    # location per state, daily
    scale: np.ndarray                 # t scale per state, daily
    nu: np.ndarray                    # tail index per state
    transition: np.ndarray            # A[i, j] = P(state j tomorrow | state i today)
    filtered: pd.DataFrame            # per date, P(state | data up to that date)
    log_likelihood: float
    iterations: int
    converged: bool
    provenance: Provenance

    @property
    def volatility(self) -> np.ndarray:
        """Conditional standard deviation per state (a t scale is not a standard deviation)."""
        return self.scale * np.sqrt(self.nu / np.maximum(self.nu - 2.0, 1e-9))

    @property
    def annual_volatility(self) -> np.ndarray:
        return self.volatility * np.sqrt(252.0)

    @property
    def p_stress(self) -> float:
        """Filtered probability of the stress state on the last observed day."""
        return float(self.filtered.iloc[-1, STRESS])

    def expected_duration(self, state: int = STRESS) -> float:
        p = float(self.transition[state, state])
        return float("inf") if p >= 1.0 else 1.0 / (1.0 - p)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "annual_volatility": self.annual_volatility,
            "annual_drift": self.mu * 252.0,
            "tail_index_nu": self.nu,
            "persistence": np.diag(self.transition),
            "expected_duration_days": [self.expected_duration(s) for s in (CALM, STRESS)],
            "time_share": self.filtered.mean().to_numpy(),
        }, index=["calm", "stress"])


def _t_logpdf(y: np.ndarray, mu: float, scale: float, nu: float) -> np.ndarray:
    d2 = ((y - mu) / scale) ** 2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu)
            - np.log(scale) - (nu + 1) / 2 * np.log1p(d2 / nu))


def _stationary(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = np.clip(v, 1e-12, None)
    return v / v.sum()


def _forward(logB: np.ndarray, A: np.ndarray, pi0: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Scaled forward pass. Returns (filtered probabilities, log-likelihood, scaling factors)."""
    T, K = logB.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    b = np.exp(logB - logB.max(axis=1, keepdims=True))      # scaled emissions, offset cancels
    a = pi0 * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / max(c[t], 1e-300)
    ll = float(np.sum(np.log(np.maximum(c, 1e-300))) + logB.max(axis=1).sum())
    return alpha, ll, c


def _backward(logB: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    T, K = logB.shape
    b = np.exp(logB - logB.max(axis=1, keepdims=True))
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (b[t + 1] * beta[t + 1]) / max(c[t + 1], 1e-300)
    return beta


def filtered_probabilities(returns: pd.Series, mu, scale, nu, transition) -> pd.DataFrame:
    """
    P(state | data up to and including each date), for given parameters.

    Exposed separately so a backtest can apply a model fitted as-of an earlier
    date to later data without refitting - and so the no-look-ahead property can
    be tested directly.
    """
    y = pd.Series(returns).dropna()
    A = np.asarray(transition, dtype=float)
    logB = np.column_stack([_t_logpdf(y.to_numpy(), mu[k], scale[k], nu[k]) for k in range(len(mu))])
    alpha, _, _ = _forward(logB, A, _stationary(A))
    return pd.DataFrame(alpha, index=y.index, columns=["calm", "stress"])


def _fit_nu(y, w, mu, scale, bounds) -> float:
    """Tail index by weighted maximum likelihood; bounded, so it cannot run to infinity."""
    def neg_ll(nu):
        return -float(np.sum(w * _t_logpdf(y, mu, scale, nu)))
    best = minimize_scalar(neg_ll, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return float(best.x)


def fit_regimes(returns: pd.Series, *, settings: RegimeSettings = RegimeSettings(),
                as_of=None) -> RegimeFit:
    """
    Baum-Welch EM for a two-state Student-t HMM on daily log returns.

    States are ordered calm-then-stress by fitted volatility, so state numbering
    is a property of the data and not of the optimiser.
    """
    y_s = pd.Series(returns).dropna()
    y_s = y_s[np.isfinite(y_s.to_numpy())]
    if len(y_s) < settings.min_sessions:
        raise EstimationError(f"a regime model needs at least {settings.min_sessions} sessions, "
                              f"got {len(y_s)}")
    y = y_s.to_numpy()
    T, K = len(y), 2

    # deterministic start: split on rolling volatility, so no random seed is involved
    roll = pd.Series(y).rolling(21, min_periods=5).std().bfill().to_numpy()
    hot = roll > np.median(roll)
    mu = np.array([y[~hot].mean(), y[hot].mean()])
    scale = np.array([max(y[~hot].std(), settings.min_scale), max(y[hot].std(), settings.min_scale)])
    nu = np.array([8.0, 5.0])
    A = np.array([[0.97, 0.03], [0.06, 0.94]])

    ll_old, converged, it = -np.inf, False, 0
    for it in range(1, settings.max_iter + 1):
        logB = np.column_stack([_t_logpdf(y, mu[k], scale[k], nu[k]) for k in range(K)])
        alpha, ll, c = _forward(logB, A, _stationary(A))
        beta = _backward(logB, A, c)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        b = np.exp(logB - logB.max(axis=1, keepdims=True))
        xi = np.zeros((K, K))
        for t in range(T - 1):
            num = (alpha[t][:, None] * A) * (b[t + 1] * beta[t + 1])[None, :]
            xi += num / max(num.sum(), 1e-300)
        A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)

        for k in range(K):
            d2 = ((y - mu[k]) / scale[k]) ** 2
            u = (nu[k] + 1.0) / (nu[k] + d2)                # scale-mixture E-step weight
            gw = gamma[:, k] * u
            mu[k] = float(np.sum(gw * y) / max(np.sum(gw), 1e-300))
            scale[k] = float(np.sqrt(max(np.sum(gw * (y - mu[k]) ** 2) / max(gamma[:, k].sum(), 1e-300),
                                         settings.min_scale ** 2)))
            nu[k] = _fit_nu(y, gamma[:, k], mu[k], scale[k], settings.nu_bounds)

        if abs(ll - ll_old) < settings.tol * max(abs(ll_old), 1.0):
            converged = True
            ll_old = ll
            break
        ll_old = ll

    order = np.argsort(scale * np.sqrt(nu / np.maximum(nu - 2.0, 1e-9)))   # calm first, by volatility
    mu, scale, nu = mu[order], scale[order], nu[order]
    A = A[np.ix_(order, order)]

    filtered = filtered_probabilities(y_s, mu, scale, nu, A)
    prov = Provenance(
        method="two-state Student-t hidden Markov model (Baum-Welch EM), filtered probabilities",
        window=(f"{T} sessions; stress volatility {scale[STRESS] * np.sqrt(nu[STRESS] / max(nu[STRESS] - 2, 1e-9)) * np.sqrt(252):.1%} "
                f"vs calm {scale[CALM] * np.sqrt(nu[CALM] / max(nu[CALM] - 2, 1e-9)) * np.sqrt(252):.1%}; "
                f"{it} EM iterations{'' if converged else ' (did not converge)'}"),
        as_of=pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(y_s.index[-1]),
        source="daily log returns",
        degradations=() if converged else ("regime EM hit the iteration limit; "
                                           "treat state probabilities as indicative",),
    )
    return RegimeFit(mu=mu, scale=scale, nu=nu, transition=A, filtered=filtered,
                     log_likelihood=float(ll_old), iterations=it, converged=converged,
                     provenance=prov)


# ---------------------------------------------------------------- India VIX ---
@dataclass(frozen=True)
class VixReading:
    """A model-free second opinion, from the price of index options."""

    level: float
    band: str                          # "calm" | "normal" | "stress"
    calm_below: float
    stress_above: float
    percentile: float
    as_of: pd.Timestamp
    lookback_sessions: int


def vix_bands(vix: pd.Series, *, as_of=None, lookback: int = 1260,
              calm_pct: float = 33.0, stress_pct: float = 80.0) -> VixReading:
    """
    Band today's India VIX against its own recent history.

    Thresholds are percentiles of the trailing window **up to as-of only**, so a
    band never knows about volatility that had not happened yet. The stress
    threshold sits at the 80th percentile: turbulence is meant to be unusual.
    """
    s = pd.Series(vix).dropna()
    if as_of is not None:
        s = s.loc[:pd.Timestamp(as_of)]
    s = s.iloc[-lookback:]
    if len(s) < 60:
        raise EstimationError(f"India VIX bands need at least 60 sessions, got {len(s)}")
    if not 0 < calm_pct < stress_pct < 100:
        raise EstimationError(f"percentiles must satisfy 0 < calm < stress < 100, got {calm_pct}/{stress_pct}")
    level = float(s.iloc[-1])
    calm_below, stress_above = (float(np.percentile(s.to_numpy(), calm_pct)),
                                float(np.percentile(s.to_numpy(), stress_pct)))
    band = "stress" if level >= stress_above else ("calm" if level <= calm_below else "normal")
    return VixReading(level=level, band=band, calm_below=calm_below, stress_above=stress_above,
                      percentile=float((s.to_numpy() <= level).mean() * 100),
                      as_of=pd.Timestamp(s.index[-1]), lookback_sessions=len(s))


@dataclass(frozen=True)
class RegimeOpinion:
    """What the gauges say, including when they disagree."""

    stress_probability: float
    hmm_says: str
    vix_says: str | None
    agree: bool
    expected_duration_days: float
    headline: str


def combine(fit: RegimeFit, vix: VixReading | None = None, *, stress_threshold: float = 0.6) -> RegimeOpinion:
    """
    One sentence a reader can act on - and an explicit note when the two gauges
    disagree, because "the models disagree" is information, not an inconvenience.
    """
    p = fit.p_stress
    hmm = "stress" if p >= stress_threshold else ("calm" if p <= 1 - stress_threshold else "mixed")
    vix_says = vix.band if vix is not None else None
    agree = vix_says is None or vix_says == hmm or (hmm == "mixed" and vix_says == "normal")
    duration = fit.expected_duration(STRESS)
    if hmm == "stress":
        head = (f"Markets are stressed: {p:.0%} likely. Spells like this have typically "
                f"lasted about {duration:.0f} days, with prices swinging about "
                f"{fit.annual_volatility[STRESS]:.0%} a year.")
    elif hmm == "calm":
        head = (f"Markets are calm: {1 - p:.0%} likely, with prices swinging about "
                f"{fit.annual_volatility[CALM]:.0%} a year.")
    else:
        head = (f"The market is between calm and stressed: {p:.0%} chance of stress. "
                f"Neither reading is settled.")
    if not agree:
        head += f" India VIX disagrees, reading {vix_says} at {vix.level:.1f}; treat the signal as uncertain."
    return RegimeOpinion(stress_probability=p, hmm_says=hmm, vix_says=vix_says, agree=agree,
                         expected_duration_days=duration, headline=head)
