"""Point and interval estimates built only from what a verifier measured.

Every function here takes verification outcomes and returns a number with an
honest uncertainty attached.  Nothing in this module may consult the
simulator's privileged view of the channel -- not the true fidelity of a
delivered qubit, not Eve's probe state, not the entanglement witnesses the
distribution layer records for the benefit of the test suite.  A deployed
verifier holds a list of measurement outcomes and a spec sheet, and that is
deliberately all this module is given.

Two habits are load-bearing:

**Interval, not just point.**  Every rate is reported with a Clopper-Pearson
interval rather than a bare ratio.  Clopper-Pearson is exact -- it inverts the
binomial CDF instead of assuming a normal shape -- which matters because the
interesting cases are precisely the small ones.  At ``n = 30`` with zero
mismatches a normal approximation reports the interval ``[0, 0]``, which would
let the engine certify a channel it has barely looked at.

**Say what the denominator is.**  Each estimate carries a ``conditioned_on``
string.  An error rate conditioned on the qubit having arrived is a different
quantity from an error rate over attempted rounds, and the gap between them is
exactly where a loss-based attack lives: an adversary who blocks the rounds he
disturbed leaves the conditional rate untouched.  The engine tests both, and
that is only meaningful if the two are never silently interchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import stats

__all__ = [
    "RateEstimate",
    "RateDifference",
    "estimate_rate",
    "pooled_rate",
    "rates_by_basis",
    "rates_by_position",
    "rates_by_level",
    "rates_by_verifier",
    "yield_estimate",
    "dark_estimate",
    "correction_histogram",
    "decoy_rate",
    "rate_difference",
    "fisher_exact_two_sided",
    "homogeneity_chi_square",
]


# --------------------------------------------------------------------------
# the estimate container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RateEstimate:
    """A Bernoulli rate with an exact confidence interval.

    ``k`` successes out of ``n`` trials, where "success" is whatever the
    caller is counting -- a mismatch, a detector click, a dark count.  The
    interval is Clopper-Pearson at level ``1 - alpha``.
    """

    label: str
    k: int
    n: int
    alpha: float
    ci_low: float
    ci_high: float
    conditioned_on: str = ""

    @property
    def value(self) -> float:
        """The point estimate, or 0.0 when nothing was observed.

        Returning zero for ``n == 0`` is a convenience for formatting, not a
        claim.  Callers that care about the difference must look at ``n``;
        every rule in :mod:`qds.detect.rules` does, and marks itself
        inapplicable rather than treating an empty sample as a clean one.
        """
        return self.k / self.n if self.n else 0.0

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "value": _clean(self.value),
            "k": int(self.k),
            "n": int(self.n),
            "ci_low": _clean(self.ci_low),
            "ci_high": _clean(self.ci_high),
            "alpha": _clean(self.alpha),
            "conditioned_on": self.conditioned_on,
        }


@dataclass(frozen=True)
class RateDifference:
    """Difference between two independent rates, with an exact p-value.

    The p-value is Fisher's exact test rather than a two-proportion z test.
    The z is reported alongside it so a reader can see how far the normal
    approximation would have been off, but it never drives a decision: the
    samples this engine compares are often a few dozen checks, and that is
    where the approximation fails in the direction that hides attacks.
    """

    label: str
    left: RateEstimate
    right: RateEstimate
    difference: float
    p_value: Optional[float]
    z: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "difference": _clean(self.difference),
            "p_value": _clean(self.p_value),
            "z": _clean(self.z),
        }


def _clean(x: object) -> object:
    """Coerce to a JSON-native value, mapping non-finite floats to ``None``.

    ``json.dumps`` will happily emit ``NaN`` and ``Infinity``, which are not
    JSON and which every strict parser -- including the browser's -- rejects.
    The dashboard reads these files directly, so the conversion happens here
    rather than being someone else's problem.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int,)):
        return int(x)
    try:
        f = float(x)                     # numpy scalars land here too
    except (TypeError, ValueError):
        return x
    return f if math.isfinite(f) else None


@lru_cache(maxsize=8192)
def _interval(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Memoised Clopper-Pearson.

    The exact interval is found by bisecting the binomial tail, and each tail
    evaluation is a sum over up to ``n`` terms, so one interval costs a few
    tens of thousands of log-gamma calls.  A single analysis asks for roughly
    eighty intervals -- pooled, per basis, per position, per level, per
    recipient, yield, darks, decoys -- which came to 445 ms before this cache
    existed, and parameter sweeps run thousands of analyses.

    The function is pure in ``(k, n, alpha)``, and the arguments repeat
    heavily in practice: ``n`` is fixed by the protocol configuration and
    ``k`` ranges over a handful of values near the noise floor, so across a
    sweep the cache hit rate is very high.  Caching is therefore a free
    speedup rather than a precision trade -- the numbers are identical, and
    :meth:`tests.test_detect.TestEstimators` checks them against the
    uncached path.
    """
    return stats.clopper_pearson(k, n, alpha)


def estimate_rate(label: str, k: int, n: int, alpha: float = 0.05,
                  conditioned_on: str = "") -> RateEstimate:
    """Wrap ``k``/``n`` in an exact interval."""
    k = int(k)
    n = int(n)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"{label}: need 0 <= k <= n, got k={k} n={n}")
    lo, hi = _interval(k, n, float(alpha))
    return RateEstimate(label=label, k=k, n=n, alpha=alpha,
                        ci_low=lo, ci_high=hi, conditioned_on=conditioned_on)


# --------------------------------------------------------------------------
# mismatch rates
# --------------------------------------------------------------------------

_ARRIVED = "slots that produced a detector outcome (excludes lost rounds)"
_ADDRESSED = "slots the verifier attempted to measure (includes lost rounds)"


def _iter_checks(reports: Iterable[object]):
    for r in reports:
        for c in getattr(r, "checks", ()):
            yield r, c


def pooled_rate(reports: Sequence[object], alpha: float = 0.05) -> RateEstimate:
    """Mismatch rate over every counted check in every report given.

    Pooling across reports is only legitimate when they share a signer and a
    key, because otherwise the underlying rate is not one parameter.  The
    engine only ever hands this function reports from a single session.
    """
    k = sum(1 for _, c in _iter_checks(reports) if c.mismatch)
    n = sum(1 for _, c in _iter_checks(reports) if c.counted)
    return estimate_rate("pooled_rate", k, n, alpha, _ARRIVED)


def rates_by_basis(reports: Sequence[object],
                   alpha: float = 0.05) -> Dict[str, RateEstimate]:
    """``{"Z": estimate, "X": estimate}``.

    Split out because the honest rates in the two bases are *not* equal --
    see :class:`qds.detect.rules.BasisConsistency` for why comparing them to
    each other rather than to their own declared values would be wrong.
    """
    totals = {"Z": [0, 0], "X": [0, 0]}
    for r in reports:
        for name, (k, n) in r.by_basis().items():
            totals[name][0] += k
            totals[name][1] += n
    return {name: estimate_rate(f"rate_{name}", k, n, alpha, _ARRIVED)
            for name, (k, n) in totals.items()}


def rates_by_position(reports: Sequence[object],
                      alpha: float = 0.05) -> Dict[int, RateEstimate]:
    """Mismatch rate per message-bit index.

    A forger who rewrote one bit of the message, or an adversary who spent a
    limited disturbance budget on the blocks that mattered, shows up as one
    position out of line rather than as a raised pooled rate.
    """
    totals: Dict[int, List[int]] = {}
    for r in reports:
        for j, (k, n) in r.by_position().items():
            cell = totals.setdefault(int(j), [0, 0])
            cell[0] += k
            cell[1] += n
    return {j: estimate_rate(f"rate_position_{j}", k, n, alpha, _ARRIVED)
            for j, (k, n) in sorted(totals.items())}


def rates_by_level(reports: Sequence[object],
                   alpha: float = 0.05) -> Dict[str, RateEstimate]:
    """Mismatch rate per verification level ("accept" / "transfer").

    The two levels apply different thresholds to the same physical channel,
    so the underlying rate should be identical.  A difference means the rate
    moved between the two measurements.
    """
    totals: Dict[str, List[int]] = {}
    for r in reports:
        cell = totals.setdefault(str(r.level), [0, 0])
        cell[0] += r.n_mismatch
        cell[1] += r.n_checked
    return {lv: estimate_rate(f"rate_level_{lv}", k, n, alpha, _ARRIVED)
            for lv, (k, n) in sorted(totals.items())}


def rates_by_verifier(reports: Sequence[object],
                      alpha: float = 0.05) -> Dict[str, RateEstimate]:
    """Mismatch rate per recipient.

    Recipients hold independently symmetrised halves of the same key and see
    the same signer, so their rates should agree.  Disagreement is the
    observable that distinguishes a link problem from a protocol problem.
    """
    totals: Dict[str, List[int]] = {}
    for r in reports:
        cell = totals.setdefault(str(r.verifier), [0, 0])
        cell[0] += r.n_mismatch
        cell[1] += r.n_checked
    return {name: estimate_rate(f"rate_verifier_{name}", k, n, alpha, _ARRIVED)
            for name, (k, n) in sorted(totals.items())}


# --------------------------------------------------------------------------
# detector statistics -- the denominator that includes lost rounds
# --------------------------------------------------------------------------

def yield_estimate(reports: Sequence[object] = (),
                   detector_logs: Sequence[Sequence[str]] = (),
                   alpha: float = 0.05) -> RateEstimate:
    """Fraction of addressed slots that produced any outcome at all.

    Conditioned on *attempted*, not on arrival, which is the whole point: an
    adversary who intercepts a qubit, learns something, and then blocks the
    round rather than resending a disturbed copy leaves every error rate in
    this module untouched and shows up only here.

    Accepts either verification reports (whose checks carry a ``status``) or
    raw ``PublicKeyStore.detector_log`` sequences, and pools whatever it is
    given.  The detector log covers every slot the store ever received,
    including those never addressed by a signature check, so it is the wider
    and preferable evidence when available.
    """
    clicks = 0
    total = 0
    for log in detector_logs:
        for status in log:
            total += 1
            clicks += int(status != "lost")
    for _, c in _iter_checks(reports):
        total += 1
        clicks += int(c.status != "missing")
    return estimate_rate("detector_yield", clicks, total, alpha, _ADDRESSED)


def dark_estimate(reports: Sequence[object] = (),
                  detector_logs: Sequence[Sequence[str]] = (),
                  alpha: float = 0.05) -> RateEstimate:
    """Fraction of addressed slots whose outcome came from a dark count.

    A dark count still produces a bit, and that bit is uniform, so darks
    raise the error rate as well as showing up here.  Counting them
    separately lets the engine tell "the detectors are noisy" from "the
    channel is noisy", which have different thresholds and different fixes.
    """
    dark = 0
    total = 0
    for log in detector_logs:
        for status in log:
            total += 1
            dark += int(status == "dark")
    for _, c in _iter_checks(reports):
        total += 1
        dark += int(c.status == "dark")
    return estimate_rate("dark_fraction", dark, total, alpha, _ADDRESSED)


def correction_histogram(history: Sequence[Tuple[int, int]]) -> List[int]:
    """Counts of the four Bell outcomes in the announced Pauli corrections.

    The teleportation transcript is the pair ``(u, v)`` the signer broadcasts
    so the recipient can undo ``Z^u X^v``.  Under *any* honest noise this
    pair is uniform on ``{0,1}**2`` -- that uniformity is exactly the
    statement that the announcement leaks nothing about the key, which is
    proved in :func:`qds.protocol.distribution.teleportation_leakage`.  So a
    biased histogram is not merely unusual, it contradicts a proof, and the
    only way to get one is for somebody to be interfering with the Bell
    measurement or the classical channel.

    Returned in index order ``(0,0), (0,1), (1,0), (1,1)``.
    """
    counts = [0, 0, 0, 0]
    for pair in history:
        u, v = int(pair[0]) & 1, int(pair[1]) & 1
        counts[(u << 1) | v] += 1
    return counts


def decoy_rate(calibrations: Sequence[object],
               alpha: float = 0.05) -> RateEstimate:
    """Mismatch rate on the decoy slots, pooled over recipients.

    Decoys are measured during calibration and consumed there, so this is the
    only record of them by the time a signature is verified.  They traverse
    the same channel as the signature slots and are chosen at random, which
    is what makes the comparison in
    :class:`qds.detect.rules.DecoyVersusSignature` meaningful.
    """
    k = sum(int(getattr(c, "decoy_mismatches", 0)) for c in calibrations)
    n = sum(int(getattr(c, "decoys_measured", 0)) for c in calibrations)
    return estimate_rate("decoy_rate", k, n, alpha, _ARRIVED)


# --------------------------------------------------------------------------
# comparing two rates
# --------------------------------------------------------------------------

def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Fisher's exact test on the 2x2 table, two-sided.

    Conditioning on both margins makes the null distribution of ``k1``
    hypergeometric, so the p-value is a finite sum with no approximation in
    it::

        P(K = k) = C(n1, k) C(n2, t - k) / C(n1 + n2, t),   t = k1 + k2

    The two-sided p-value is the total probability of every table no more
    likely than the one observed.  A relative tolerance guards the comparison
    because the probabilities are computed in log space and exact ties are
    common in small tables -- without it, symmetric tables that should
    contribute would be dropped by a last-bit rounding difference.
    """
    n1, n2, k1, k2 = int(n1), int(n2), int(k1), int(k2)
    if n1 <= 0 or n2 <= 0:
        return 1.0
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2):
        raise ValueError("fisher_exact_two_sided: counts outside their totals")
    t = k1 + k2
    n = n1 + n2
    lo = max(0, t - n2)
    hi = min(n1, t)
    if hi < lo:
        return 1.0

    log_denom = stats.log_binom(n, t)

    def log_p(k: int) -> float:
        return stats.log_binom(n1, k) + stats.log_binom(n2, t - k) - log_denom

    observed = log_p(k1)
    total = 0.0
    for k in range(lo, hi + 1):
        lp = log_p(k)
        # 1e-9 in log space is a relative tolerance of about 1 part in 1e9 on
        # the probability itself -- far tighter than any table we build, but
        # loose enough to keep genuine ties.
        if lp <= observed + 1e-9:
            total += math.exp(lp)
    return float(min(1.0, max(0.0, total)))


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> Optional[float]:
    """Pooled-variance score statistic, for reporting next to the exact test."""
    if n1 <= 0 or n2 <= 0:
        return None
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0
    return ((k1 / n1) - (k2 / n2)) / se


def rate_difference(label: str, left: RateEstimate,
                    right: RateEstimate) -> RateDifference:
    """Difference of two rates with an exact two-sided p-value.

    Returns ``p_value=None`` when either side is empty.  That is not a pass:
    the caller is expected to mark the test inapplicable, because "we did not
    look" and "we looked and found nothing" are different findings and only
    one of them is evidence.
    """
    if left.n <= 0 or right.n <= 0:
        return RateDifference(label=label, left=left, right=right,
                              difference=0.0, p_value=None, z=None)
    return RateDifference(
        label=label, left=left, right=right,
        difference=left.value - right.value,
        p_value=fisher_exact_two_sided(left.k, left.n, right.k, right.n),
        z=_two_proportion_z(left.k, left.n, right.k, right.n),
    )


def homogeneity_chi_square(
        groups: Sequence[Tuple[int, int]]) -> Tuple[Optional[float], int,
                                                    Optional[float]]:
    """Pearson test that several ``(k, n)`` groups share one rate.

    Returns ``(statistic, dof, p_value)`` with ``dof = G - 1`` for ``G``
    groups with a nonzero denominator.  The null rate is the pooled estimate,
    which costs one degree of freedom; the statistic sums over both cells of
    every group, so the 2G cells minus one pooled parameter minus G row
    totals leaves G-1.

    The statistic is assembled here and handed to :func:`stats.chi2_sf`
    directly rather than routed through :func:`stats.chi2_test`, whose
    ``dof_reduction`` argument would have to be told ``G + 1`` to arrive at
    the same place -- correct but opaque, and wrong in a way that is hard to
    notice if a group ends up empty.

    Returns ``(None, 0, None)`` when fewer than two groups have data, or when
    the pooled rate is 0 or 1.  A degenerate table has no test in it: if
    nothing anywhere failed, the groups agree trivially.
    """
    live = [(int(k), int(n)) for k, n in groups if int(n) > 0]
    if len(live) < 2:
        return None, 0, None
    total_k = sum(k for k, _ in live)
    total_n = sum(n for _, n in live)
    p = total_k / total_n
    if p <= 0.0 or p >= 1.0:
        return None, 0, None
    stat = 0.0
    for k, n in live:
        for observed, expected in ((k, n * p), (n - k, n * (1.0 - p))):
            if expected > 0:
                stat += (observed - expected) ** 2 / expected
    dof = len(live) - 1
    return float(stat), dof, float(stats.chi2_sf(stat, dof))
