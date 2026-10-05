"""Exact classical statistics for the detection engine.

Everything the threat detector concludes, it concludes from one of the tests in
this module.  There is no model fitting, no training, no learned parameters and
no scoring function whose coefficients came from data -- each test is a
closed-form or exactly-summed expression whose false-alarm rate is fixed
analytically before the system is switched on.

The module deliberately depends on nothing but the standard library and numpy.
SciPy is not used, so every distribution function here is implemented and
tested from its definition:

* binomial tails by exact summation in log space (:func:`binom_cdf`,
  :func:`binom_sf`),
* the chi-square survival function from the regularised incomplete gamma
  function via a series expansion and a Lentz continued fraction
  (:func:`chi2_sf`),
* the normal CDF from ``math.erf`` and its quantile by a rational
  approximation refined with Newton steps to machine precision,
* Clopper-Pearson intervals as Beta quantiles, the Beta itself coming from a
  Lentz continued fraction (:func:`betainc`, :func:`clopper_pearson`).

Vocabulary used throughout
--------------------------
``p-value``
    Probability, computed under the *null* hypothesis "the link is behaving as
    calibrated and nobody is interfering", of seeing a deviation at least as
    extreme as the one observed.  Small p-value = surprising = alarm.
``alpha``
    The false-alarm rate we are willing to accept.  Because the engine runs
    many tests at once, per-test alpha is derived from a family-wise budget by
    Bonferroni correction (:func:`bonferroni_alpha`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "log_binom", "binom_pmf", "binom_cdf", "binom_sf",
    "binom_test_greater", "binom_test_less", "binom_test_two_sided",
    "betainc", "beta_ppf", "clopper_pearson", "wilson_interval",
    "hoeffding_tail", "hoeffding_threshold", "binary_kl", "chernoff_kl_tail",
    "chernoff_kl_tail_below", "chernoff_threshold",
    "normal_cdf", "normal_sf", "normal_quantile",
    "gammainc_lower_reg", "gammainc_upper_reg", "chi2_sf", "chi2_test",
    "sampling_deviation_bound", "sampling_deviation_margin",
    "bonferroni_alpha", "sidak_alpha", "fisher_combine",
    "SPRT", "SPRTState", "CUSUM", "CUSUMState",
    "required_trials", "proportion_z_test", "mean_and_sem",
]

_LOG2 = math.log(2.0)


# --------------------------------------------------------------------------
# binomial
# --------------------------------------------------------------------------

def log_binom(n: int, k: int) -> float:
    """``log C(n, k)`` via log-gamma; exact enough for n up to millions."""
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def binom_pmf(k: int, n: int, p: float) -> float:
    """``P(X = k)`` for ``X ~ Binomial(n, p)``, computed in log space."""
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    lp = log_binom(n, k) + k * math.log(p) + (n - k) * math.log1p(-p)
    return math.exp(lp)


def binom_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` by exact summation from the smaller tail."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if k <= n * p:
        return min(1.0, math.fsum(binom_pmf(i, n, p) for i in range(0, k + 1)))
    return max(0.0, 1.0 - math.fsum(binom_pmf(i, n, p) for i in range(k + 1, n + 1)))


def binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` -- the upper tail *including* ``k``."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if k > n * p:
        return min(1.0, math.fsum(binom_pmf(i, n, p) for i in range(k, n + 1)))
    return max(0.0, 1.0 - math.fsum(binom_pmf(i, n, p) for i in range(0, k)))


def binom_test_greater(k: int, n: int, p0: float) -> float:
    """Exact p-value for "the true rate exceeds ``p0``" given ``k`` of ``n``.

    This is the workhorse of the engine: ``k`` observed errors out of ``n``
    checks, ``p0`` the calibrated error rate.  Returns ``P(X >= k)`` under the
    null, so a tiny value means the link is noisier than calibration allows.
    """
    return binom_sf(k, n, p0)


def binom_test_less(k: int, n: int, p0: float) -> float:
    """Exact p-value for "the true rate is below ``p0``"."""
    return binom_cdf(k, n, p0)


def binom_test_two_sided(k: int, n: int, p0: float) -> float:
    """Two-sided exact test by the method of small p-values.

    Sums the probability of every outcome no more likely than the observed
    one.  Used where a deviation in *either* direction is suspicious -- for
    instance an error rate that is implausibly *low* can indicate fabricated
    data rather than a healthy link.
    """
    if n == 0:
        return 1.0
    obs = binom_pmf(k, n, p0)
    tol = obs * (1 + 1e-12)
    return min(1.0, math.fsum(
        pm for pm in (binom_pmf(i, n, p0) for i in range(n + 1)) if pm <= tol
    ))


def _betacf(a: float, b: float, x: float,
            itmax: int = 400, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta, by modified Lentz.

    This is the standard Numerical Recipes recurrence.  It converges in a few
    tens of iterations over the range the caller is restricted to, which is
    what makes :func:`betainc` independent of ``n`` -- the whole reason this
    function exists.  ``_TINY`` guards the Lentz denominators against an exact
    zero, which is a numerical accident rather than a real pole.
    """
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``.

    Related to the binomial tail by the exact identity

        ``P(X >= k) = I_p(k, n - k + 1)``     for ``X ~ Binomial(n, p)``

    which is why this belongs in a module with no beta functions in its
    original design: it lets :func:`clopper_pearson` invert the binomial tail
    without summing it.  The symmetry ``I_x(a,b) = 1 - I_{1-x}(b,a)`` is used
    to keep the continued fraction on the side where it converges quickly.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, front * _betacf(a, b, x) / a)
    return max(0.0, 1.0 - front * _betacf(b, a, 1.0 - x) / b)


def beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse of :func:`betainc` in ``x``, by bisection.

    ``I_x(a, b)`` is strictly increasing in ``x``, so bisection is safe and
    needs no starting guess.  It exits once the bracket stops shrinking in
    double precision, which happens after about sixty steps; the iteration cap
    is a guard, not the normal path.
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:        # bracket exhausted at this precision
            break
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (conservative) confidence interval for a binomial proportion.

    Obtained by inverting the exact binomial test: the lower limit is the ``p``
    for which ``P(X >= k) = alpha/2`` and the upper limit the ``p`` for which
    ``P(X <= k) = alpha/2``.  There is no small-sample approximation anywhere
    in it, which is the point -- the engine's interesting samples are the small
    ones, and a normal approximation reports ``[0, 0]`` for zero mismatches out
    of thirty, which would certify a channel nobody has really looked at.

    Both limits are quantiles of a Beta distribution, by the identity in
    :func:`betainc`::

        low  = BetaInv(alpha/2;   k,     n - k + 1)
        high = BetaInv(1-alpha/2; k + 1, n - k)

    That formulation is what makes this affordable.  Inverting the binomial
    tail by summing it costs ``O(n)`` log-gamma calls *per bisection step*; at
    ``n = 384`` one interval came to roughly sixty thousand of them, and a
    single analysis asks for about eighty intervals.  Through the continued
    fraction the cost no longer depends on ``n`` at all.  The value is the
    same to the last couple of bits -- :meth:`tests.test_detect.TestEstimators`
    pins it against a direct summation of the binomial tail.
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    half = alpha / 2.0
    low = 0.0 if k == 0 else beta_ppf(half, k, n - k + 1)
    high = 1.0 if k == n else beta_ppf(1.0 - half, k + 1, n - k)
    return low, high


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval -- the closed form used for dashboard error bars.

    Much tighter than Clopper-Pearson at moderate ``n`` and well behaved at
    ``k = 0`` or ``k = n``, unlike the textbook normal-approximation interval.
    Reported alongside the exact interval so a reader can see both.
    """
    if n <= 0:
        return 0.0, 1.0
    z = normal_quantile(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------
# concentration inequalities (the security proofs use these, not the CLT)
# --------------------------------------------------------------------------

def hoeffding_tail(n: int, t: float) -> float:
    """``P(mean - E[mean] >= t) <= exp(-2 n t**2)`` for bounded [0,1] variables.

    Distribution-free and one-sided.  This is the bound quoted in the security
    theorems because it holds for *any* adversary strategy that yields
    independent per-check outcomes, without assuming the outcomes are
    identically distributed.
    """
    if t <= 0:
        return 1.0
    return math.exp(-2.0 * n * t * t)


def hoeffding_threshold(n: int, epsilon: float) -> float:
    """Smallest ``t`` with ``exp(-2 n t**2) <= epsilon``."""
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(1.0 / epsilon) / (2.0 * n))


def binary_kl(q: float, p: float) -> float:
    """Binary relative entropy ``D(q || p)`` in nats."""
    eps = 1e-15
    q = min(max(q, 0.0), 1.0)
    p = min(max(p, eps), 1.0 - eps)
    out = 0.0
    if q > 0:
        out += q * math.log(q / p)
    if q < 1:
        out += (1 - q) * math.log((1 - q) / (1 - p))
    return out


def chernoff_kl_tail(n: int, p: float, q: float) -> float:
    """``P(X/n >= q) <= exp(-n D(q||p))`` for ``q > p`` (Chernoff-Hoeffding).

    Strictly tighter than :func:`hoeffding_tail` and the bound actually used
    for the headline forgery probability, because near ``p = 3/4`` the KL form
    is dramatically better than the ``2 n t**2`` form.
    """
    if q <= p:
        return 1.0
    return math.exp(-n * binary_kl(q, p))


def chernoff_kl_tail_below(n: int, p: float, q: float) -> float:
    """``P(X/n <= q) <= exp(-n D(q||p))`` for ``q < p``.

    The mirror image of :func:`chernoff_kl_tail`, and the one the forgery
    bound needs: a forger's per-check error rate is *at least* 1/4, so what
    has to be bounded is the chance his observed rate falls *below* the
    acceptance threshold.  Same relative-entropy exponent, opposite tail --
    ``D(q||p)`` is not symmetric in its arguments, so the direction is not
    cosmetic.
    """
    if q >= p:
        return 1.0
    return math.exp(-n * binary_kl(q, p))


def chernoff_threshold(n: int, p: float, epsilon: float,
                       direction: str = "below") -> float:
    """Rate at which the Chernoff-KL tail beneath/above ``p`` drops to ``epsilon``.

    ``direction="below"`` returns the largest ``q < p`` with
    ``P(X/n <= q) <= epsilon`` -- i.e. how far below a forger's error rate a
    threshold can sit before the forger's chance of sneaking under it exceeds
    ``epsilon``.
    """
    if n <= 0:
        return p
    target = math.log(1.0 / epsilon) / n
    lo, hi = (0.0, p) if direction == "below" else (p, 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binary_kl(mid, p) > target:
            if direction == "below":
                lo = mid
            else:
                hi = mid
        else:
            if direction == "below":
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# normal distribution
# --------------------------------------------------------------------------

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def normal_quantile(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation + Newton polish).

    The Newton step against :func:`normal_cdf` drives the residual to ~1e-15,
    so this is exact for our purposes and needs no lookup table.
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must lie in [0, 1]")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    for _ in range(3):
        err = normal_cdf(x) - p
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        if pdf < 1e-300:
            break
        x -= err / pdf
    return x


# --------------------------------------------------------------------------
# chi-square, via the regularised incomplete gamma function
# --------------------------------------------------------------------------

def gammainc_lower_reg(s: float, x: float, iters: int = 1000,
                       tol: float = 1e-16) -> float:
    """Regularised lower incomplete gamma ``P(s, x)`` by its power series."""
    if x < 0 or s <= 0:
        raise ValueError("gammainc requires s > 0 and x >= 0")
    if x == 0:
        return 0.0
    ap = s
    total = 1.0 / s
    term = total
    for _ in range(iters):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * tol:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def gammainc_upper_reg(s: float, x: float, iters: int = 1000,
                       tol: float = 1e-16) -> float:
    """Regularised upper incomplete gamma ``Q(s, x)`` by a Lentz continued fraction."""
    if x < 0 or s <= 0:
        raise ValueError("gammainc requires s > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < s + 1.0:
        return 1.0 - gammainc_lower_reg(s, x, iters, tol)
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, iters + 1):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def chi2_sf(x: float, dof: int) -> float:
    """``P(chi2_dof >= x)``.  Exact to ~1e-15 for the dof values used here."""
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")
    if x <= 0:
        return 1.0
    return gammainc_upper_reg(dof / 2.0, x / 2.0)


def chi2_test(observed: Sequence[float],
              expected: Sequence[float],
              dof_reduction: int = 1) -> Tuple[float, int, float]:
    """Pearson goodness-of-fit test.  Returns ``(statistic, dof, p_value)``.

    The engine uses this on the histogram of teleportation Pauli-correction
    outcomes, which must be uniform over the four Bell results.  A biased
    histogram means somebody is tampering with the classical correction
    channel or with the Bell measurement itself -- and unlike an error-rate
    test, it catches tampering that happens to be *error-neutral*.
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    if obs.shape != exp.shape:
        raise ValueError("observed and expected must have the same shape")
    mask = exp > 0
    stat = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
    dof = max(1, int(mask.sum()) - dof_reduction)
    return stat, dof, chi2_sf(stat, dof)


# --------------------------------------------------------------------------
# finite-size random sampling (check rounds bound the unmeasured rounds)
# --------------------------------------------------------------------------

def sampling_deviation_bound(n_sample: int, n_rest: int, t: float) -> float:
    """Probability that a random sample misrepresents the population by ``t``.

    Check rounds are chosen at random *after* transmission, so the error rate
    measured on them constrains the error rate of the rounds that carry the
    signature.  For sampling without replacement, the Hoeffding-Serfling
    inequality gives

        P(|e_sample - e_rest| >= t) <= 2 exp(-2 t**2 k m / (k + m))

    with ``k`` sampled and ``m`` unsampled rounds.  This is what makes the
    security statement *finite-size*: nothing here assumes infinitely many
    rounds or an asymptotic rate.
    """
    if n_sample <= 0 or n_rest <= 0 or t <= 0:
        return 1.0
    k, m = n_sample, n_rest
    return min(1.0, 2.0 * math.exp(-2.0 * t * t * k * m / (k + m)))


def sampling_deviation_margin(n_sample: int, n_rest: int, epsilon: float) -> float:
    """The ``t`` at which :func:`sampling_deviation_bound` equals ``epsilon``."""
    if n_sample <= 0 or n_rest <= 0:
        return 1.0
    k, m = n_sample, n_rest
    return math.sqrt((k + m) * math.log(2.0 / epsilon) / (2.0 * k * m))


# --------------------------------------------------------------------------
# multiple testing
# --------------------------------------------------------------------------

def bonferroni_alpha(alpha_family: float, n_tests: int) -> float:
    """Per-test significance that holds the family-wise false-alarm rate.

    The engine runs eleven tests per session.  If each ran at alpha = 0.01 the
    system would false-alarm about 10% of the time on a perfectly healthy
    link, which would train operators to ignore it.  Bonferroni is chosen over
    tighter procedures because it needs no independence assumption -- the
    detectors share the same underlying data and are certainly not
    independent.
    """
    if n_tests <= 0:
        return alpha_family
    return alpha_family / n_tests


def sidak_alpha(alpha_family: float, n_tests: int) -> float:
    """Sidak correction: exact under independence, slightly less conservative."""
    if n_tests <= 0:
        return alpha_family
    return 1.0 - (1.0 - alpha_family) ** (1.0 / n_tests)


def fisher_combine(pvalues: Sequence[float]) -> Tuple[float, int, float]:
    """Fisher's method: combine independent p-values.  ``(stat, dof, p)``.

    Reported as a secondary summary only.  It is *not* used to raise the
    alarm, because the detectors are correlated and Fisher's method assumes
    they are not; the alarm decision uses Bonferroni, which does not.
    """
    ps = [min(max(p, 1e-300), 1.0) for p in pvalues]
    stat = -2.0 * math.fsum(math.log(p) for p in ps)
    dof = 2 * len(ps)
    return stat, dof, chi2_sf(stat, dof)


# --------------------------------------------------------------------------
# sequential / online monitors
# --------------------------------------------------------------------------

@dataclass
class SPRTState:
    log_lr: float = 0.0
    n: int = 0
    decision: str = "continue"     # "continue" | "accept_h1" | "accept_h0"
    trajectory: List[float] = field(default_factory=list)


@dataclass
class SPRT:
    """Wald's sequential probability ratio test on Bernoulli checks.

    Used for *online* forgery detection: instead of waiting for all ``L``
    checks, accumulate the log-likelihood ratio between "error rate = p1
    (forger)" and "error rate = p0 (honest)" and stop as soon as it crosses a
    boundary.  Among all tests with the same error rates, the SPRT minimises
    the expected number of observations (Wald-Wolfowitz), so this is the
    fastest possible honest alarm -- which matters when each observation costs
    an entangled pair.
    """

    p0: float
    p1: float
    alpha: float = 1e-3
    beta: float = 1e-3

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < self.p1 < 1.0:
            raise ValueError("SPRT requires 0 < p0 < p1 < 1")
        self.upper = math.log((1.0 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1.0 - self.alpha))

    def start(self) -> SPRTState:
        return SPRTState()

    def update(self, state: SPRTState, error: bool) -> SPRTState:
        inc = (math.log(self.p1 / self.p0) if error
               else math.log((1 - self.p1) / (1 - self.p0)))
        state.log_lr += inc
        state.n += 1
        state.trajectory.append(state.log_lr)
        if state.log_lr >= self.upper:
            state.decision = "accept_h1"
        elif state.log_lr <= self.lower:
            state.decision = "accept_h0"
        else:
            state.decision = "continue"
        return state

    def run(self, errors: Iterable[bool]) -> SPRTState:
        state = self.start()
        for e in errors:
            self.update(state, e)
            if state.decision != "continue":
                break
        return state

    def expected_samples_h1(self) -> float:
        """Wald's approximation for the expected stopping time under H1."""
        d = binary_kl(self.p1, self.p0)
        if d <= 0:
            return math.inf
        return self.upper / d


@dataclass
class CUSUMState:
    s: float = 0.0
    n: int = 0
    alarm_at: Optional[int] = None
    trajectory: List[float] = field(default_factory=list)


@dataclass
class CUSUM:
    """Page's cumulative-sum monitor for a sustained shift from ``p0`` to ``p1``.

    Catches the patient adversary who never pushes a single session over a
    threshold but leans on the error rate continuously -- and equally, catches
    hardware quietly degrading.  Where the SPRT asks "has the whole run been
    generated by ``p1`` rather than ``p0``", the CUSUM asks "did the rate
    *change* partway through", by resetting to zero whenever the accumulated
    evidence goes negative.

    The increment is the per-observation log-likelihood ratio

        ``llr(x) = x log(p1/p0) + (1-x) log((1-p1)/(1-p0))``

    and the statistic is ``S_i = max(0, S_{i-1} + llr(x_i))``, with an alarm
    the first time ``S_i > threshold_h``.  This is the classical Page
    formulation, and by Lorden's theorem it minimises the worst-case expected
    delay to detection among all monitors with the same false-alarm rate.

    **Feed it raw outcomes, not a running average.**  The observations must be
    the individual ``0/1`` results.  Passing a cumulative error rate instead
    makes the increments strongly autocorrelated -- once the running mean is
    above the reference it stays there for many steps -- so the statistic
    drifts up on honest data and the false-alarm rate stops bearing any
    relation to ``threshold_h``.

    Calibrating ``threshold_h``
    ---------------------------
    Under the null each excursion is a random walk with negative drift, and
    the exponential tilt that solves ``E_0[e**(theta*llr)] = 1`` is exactly
    ``theta = 1``, since ``sum_x P0(x) (P1(x)/P0(x)) = sum_x P1(x) = 1``.
    Wald's inequality then bounds the chance that any one excursion ever
    reaches ``h`` by ``e**-h``, and there are at most ``n`` restart points in
    ``n`` observations, so

        ``P(alarm within n) <= n e**-h``

    giving ``h = ln(n / alpha)`` -- see :meth:`for_window`.  The union bound
    ignores the overlap between excursions, so the true rate is comfortably
    below ``alpha``; the bound errs toward not alarming, which is the correct
    direction for a monitor whose alarms are read as evidence.
    """

    p0: float
    p1: float
    threshold_h: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < self.p1 < 1.0:
            raise ValueError(
                f"CUSUM needs 0 < p0 < p1 < 1, got p0={self.p0} p1={self.p1}")

    @classmethod
    def for_window(cls, p0: float, p1: float, n: int,
                   alpha: float = 1e-3) -> "CUSUM":
        """Threshold set so the false-alarm probability over ``n`` is ``<= alpha``.

        Derived, not tuned: ``h = ln(n/alpha)`` from the Wald-plus-union-bound
        argument above.  Every input is declared before the run.
        """
        if n <= 0:
            raise ValueError("CUSUM.for_window needs n >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        return cls(p0=p0, p1=p1, threshold_h=math.log(n / alpha))

    @property
    def k(self) -> float:
        """Midpoint of the two rates -- reported, not used by the update."""
        return 0.5 * (self.p0 + self.p1)

    def llr(self, x: float) -> float:
        """Log-likelihood ratio contributed by one observation."""
        x = 1.0 if x else 0.0
        return (x * math.log(self.p1 / self.p0)
                + (1.0 - x) * math.log((1.0 - self.p1) / (1.0 - self.p0)))

    def start(self) -> CUSUMState:
        return CUSUMState()

    def update(self, state: CUSUMState, outcome: float) -> CUSUMState:
        """Feed one raw outcome (``0``/``1`` or a bool)."""
        state.n += 1
        state.s = max(0.0, state.s + self.llr(outcome))
        state.trajectory.append(state.s)
        if state.alarm_at is None and state.s > self.threshold_h:
            state.alarm_at = state.n
        return state

    def run(self, outcomes: Iterable[float]) -> CUSUMState:
        state = self.start()
        for x in outcomes:
            self.update(state, x)
        return state


# --------------------------------------------------------------------------
# misc helpers
# --------------------------------------------------------------------------

def proportion_z_test(k: int, n: int, p0: float) -> Tuple[float, float]:
    """Score (z) test for a proportion.  Returns ``(z, one_sided_p)``.

    Kept only for reporting next to the exact test, so a reader can see how
    much the normal approximation would have understated a small-sample tail.
    Decisions always use the exact binomial p-value.
    """
    if n <= 0:
        return 0.0, 1.0
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return math.inf if k > 0 else 0.0, 0.0 if k > 0 else 1.0
    z = (k / n - p0) / se
    return z, normal_sf(z)


def required_trials(p0: float, p1: float, alpha: float = 1e-3,
                    beta: float = 1e-3) -> int:
    """Smallest ``n`` whose exact binomial test separates ``p0`` from ``p1``.

    Answers the operational question "how many checks do I need per signature
    to catch a forger with probability ``1 - beta`` while false-alarming at
    ``alpha``".  Found by scanning ``n`` and, for each, choosing the smallest
    critical value whose null tail is within ``alpha``.
    """
    if not 0.0 <= p0 < p1 <= 1.0:
        raise ValueError("required_trials expects p0 < p1")
    n = 1
    while n < 100000:
        crit = None
        for k in range(n + 1):
            if binom_sf(k, n, p0) <= alpha:
                crit = k
                break
        if crit is not None and binom_cdf(crit - 1, n, p1) <= beta:
            return n
        n += 1
    return n


def mean_and_sem(values: Sequence[float]) -> Tuple[float, float]:
    """Sample mean and standard error of the mean."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))
