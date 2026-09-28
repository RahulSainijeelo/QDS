"""Turning a declared noise floor into the numbers a verifier compares against.

Every threshold in this module is a function of quantities fixed **before**
any data is taken: the declared noise floor from the hardware spec sheet, the
number of checks the protocol will run, and the two forgery bounds that come
out of quantum mechanics rather than out of a measurement.  Nothing here reads
the session it is about to judge.

That restraint is the whole design.  An earlier version of the surrounding
session code estimated the threshold from the run's own decoy sample, which
fails twice over.  It fails formally, because extrapolating from a few dozen
decoys to the signature slots costs a Hoeffding-Serfling margin of
``sqrt(ln(2/eps) / 2n)`` -- at ``n = 96`` and ``eps = 1e-6`` that is 0.275,
wider than the entire interval ``[0, 1/4]`` the threshold has to live in, so
the estimate was not imprecise but vacuous.  And it fails operationally,
because a threshold derived from data the adversary can influence is a
threshold the adversary sets.  Decoys still have a job -- they test whether
the channel conforms to spec, and a failed conformance test is itself a
detection event -- but they do not get to move the goalposts.

The ordering that has to hold for any of this to mean anything is

    eta  <  s_a  <  s_v  <  1/4

with ``eta`` the honest floor, ``s_a`` the acceptance threshold, ``s_v`` the
stricter transfer threshold, and ``1/4`` the per-check error rate a forger
holding one copy of the public key cannot beat.  If the floor rises far enough
that no such ordering exists, the honest answer is that this link cannot carry
a signature, and :meth:`~qds.protocol.verification.VerificationPolicy.sound`
says so instead of returning a threshold anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import stats

__all__ = [
    "FORGER_ERROR_RATE",
    "BLIND_FORGER_ERROR_RATE",
    "DecisionThreshold",
    "critical_count",
    "honest_side_threshold",
    "forger_side_threshold",
    "security_exponents",
    "multiplicity_alpha",
    "threshold_ladder",
    "bits",
]

#: Per-check error rate of the best forger holding one copy of the public key.
#: ``1 - 3/4``; the 3/4 is the optimal single-copy discrimination probability
#: for the BB84 ensemble, achieved by the pretty-good measurement.
FORGER_ERROR_RATE = 0.25

#: Per-check error rate of a forger with no copy at all -- a blind guess.
BLIND_FORGER_ERROR_RATE = 0.50


def bits(p: Optional[float]) -> Optional[float]:
    """``-log2(p)``, the natural unit for a failure probability.

    Reported instead of the raw probability because the interesting values
    run to ``1e-60`` and beyond, where a decimal is unreadable and a reader
    cannot tell ``1e-9`` from ``1e-90`` at a glance.  Returns ``None`` for a
    probability of zero: the exponent is unbounded, and printing ``inf``
    would suggest a precision the finite-size analysis does not have.
    """
    if p is None:
        return None
    if p <= 0.0:
        return None
    if p >= 1.0:
        return 0.0
    return -math.log2(p)


@dataclass(frozen=True)
class DecisionThreshold:
    """A rate, the count it corresponds to at a given ``n``, and its warrant.

    ``count`` is the largest number of mismatches that still passes.  Storing
    it alongside the rate removes an entire class of off-by-one: a verifier
    compares integers, and ``k <= floor(rate * n)`` is the comparison the
    protocol actually makes.
    """

    label: str
    rate: float
    n: int
    count: int
    alpha: float
    null_rate: float
    method: str
    note: str = ""

    def passes(self, k: int) -> bool:
        return int(k) <= self.count

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "rate": _f(self.rate),
            "n": int(self.n),
            "count": int(self.count),
            "alpha": _f(self.alpha),
            "null_rate": _f(self.null_rate),
            "method": self.method,
            "note": self.note,
        }


def _f(x: object) -> object:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# exact critical values
# --------------------------------------------------------------------------

def critical_count(n: int, p0: float, alpha: float) -> int:
    """Smallest ``k`` whose upper tail under ``p0`` is at most ``alpha``.

    The exact answer, found by scanning rather than by inverting a normal
    approximation.  ``n`` here is a few hundred at most, so the scan costs
    nothing, and the approximation it replaces is wrong in the direction that
    matters: the normal tail understates the binomial tail on the right, so a
    verifier using it would flag more often than its stated ``alpha``.

    Returns ``n + 1`` when no count is significant at ``alpha``, which is the
    honest answer for "this sample cannot produce that much evidence".
    """
    n = int(n)
    if n <= 0:
        return 1
    for k in range(0, n + 1):
        if stats.binom_sf(k, n, p0) <= alpha:
            return k
    return n + 1


def honest_side_threshold(n: int, honest_rate: float, epsilon: float,
                          method: str = "kl") -> float:
    """Rate above ``honest_rate`` at which a false alarm costs ``epsilon``.

    This is the smallest threshold that an honest link will exceed with
    probability at most ``epsilon``.  Pushing the threshold any lower buys
    sensitivity at the price of crying wolf.
    """
    if n <= 0:
        return 1.0
    if method == "kl":
        return float(stats.chernoff_threshold(n, honest_rate, epsilon,
                                              direction="above"))
    if method == "hoeffding":
        return float(min(1.0, honest_rate
                         + stats.hoeffding_threshold(n, epsilon)))
    raise ValueError(f"method must be 'kl' or 'hoeffding', not {method!r}")


def forger_side_threshold(n: int, forger_rate: float, epsilon: float,
                          method: str = "kl") -> float:
    """Rate below ``forger_rate`` at which a missed forgery costs ``epsilon``.

    The mirror of :func:`honest_side_threshold`.  Raising the threshold any
    higher lets a forger slip under it more often than ``epsilon``.

    The relative-entropy form is the default and the Hoeffding form is a
    fallback because the two are not close here.  Hoeffding's ``exp(-2 n t^2)``
    only knows that the variable is bounded in ``[0, 1]``; the KL form knows
    the variance actually shrinks as the rate approaches 0 or 1.  At a forger
    rate of 1/4 and a threshold of 0.1, ``D(0.1 || 0.25) = 0.0817`` nats
    against a Hoeffding exponent of ``2 * 0.15^2 = 0.045`` -- close to a
    factor of two in the exponent, which over a few hundred checks is tens of
    orders of magnitude in the bound.  Hoeffding is kept only because it needs
    no assumption beyond boundedness, so it remains valid if the per-check
    outcomes are independent but not identically distributed.
    """
    if n <= 0:
        return 0.0
    if method == "kl":
        return float(stats.chernoff_threshold(n, forger_rate, epsilon,
                                              direction="below"))
    if method == "hoeffding":
        return float(max(0.0, forger_rate
                         - stats.hoeffding_threshold(n, epsilon)))
    raise ValueError(f"method must be 'kl' or 'hoeffding', not {method!r}")


# --------------------------------------------------------------------------
# what a threshold costs in both directions
# --------------------------------------------------------------------------

def security_exponents(n: int, threshold: float, honest_rate: float,
                       forger_rate: float = FORGER_ERROR_RATE,
                       blind_rate: float = BLIND_FORGER_ERROR_RATE
                       ) -> Dict[str, object]:
    """Exact and bounded failure probabilities for one threshold at one ``n``.

    Both directions, because a threshold is a trade and reporting only one
    side of it is advocacy rather than analysis:

    ``false_alarm``
        an honest run exceeds the threshold and a valid signature is rejected.
    ``missed_forgery``
        a forger holding one copy stays under the threshold and is accepted.
    ``missed_blind_forgery``
        the same for a forger with no copy at all.

    The exact column comes from the binomial itself.  The Chernoff column is
    the bound the written security argument quotes, and it is reported next to
    the exact value so a reader can see how much the bound gives away -- it is
    an upper bound on the failure probability, so it is always the more
    pessimistic of the two, and the gap is the price of a statement that holds
    for adversaries the simulation never ran.
    """
    n = int(n)
    if n <= 0:
        return {"n": 0, "threshold": _f(threshold), "applicable": False}

    accept_at_most = int(math.floor(threshold * n))

    exact_false_alarm = stats.binom_sf(accept_at_most + 1, n, honest_rate)
    exact_missed = stats.binom_cdf(accept_at_most, n, forger_rate)
    exact_missed_blind = stats.binom_cdf(accept_at_most, n, blind_rate)

    bound_false_alarm = stats.chernoff_kl_tail(n, honest_rate, threshold)
    bound_missed = stats.chernoff_kl_tail_below(n, forger_rate, threshold)
    bound_missed_blind = stats.chernoff_kl_tail_below(n, blind_rate, threshold)

    return {
        "n": n,
        "threshold": _f(threshold),
        "honest_rate": _f(honest_rate),
        "forger_rate": _f(forger_rate),
        "accept_at_most": accept_at_most,
        "applicable": True,
        "ordered": bool(honest_rate < threshold < forger_rate),
        "exact": {
            "false_alarm": _f(exact_false_alarm),
            "missed_forgery": _f(exact_missed),
            "missed_blind_forgery": _f(exact_missed_blind),
        },
        "chernoff": {
            "false_alarm": _f(bound_false_alarm),
            "missed_forgery": _f(bound_missed),
            "missed_blind_forgery": _f(bound_missed_blind),
        },
        "bits": {
            "false_alarm": _f(bits(exact_false_alarm)),
            "missed_forgery": _f(bits(exact_missed)),
            "missed_blind_forgery": _f(bits(exact_missed_blind)),
        },
    }


# --------------------------------------------------------------------------
# multiple testing
# --------------------------------------------------------------------------

def multiplicity_alpha(alpha_family: float, n_tests: int,
                       method: str = "bonferroni") -> float:
    """Per-test significance that holds the family-wise false-alarm rate.

    Bonferroni by default, and the choice is not conservatism for its own
    sake.  The detectors in :mod:`qds.detect.rules` are computed from the same
    slot measurements -- the pooled rate, the per-basis rates and the
    per-position rates are three views of one list of outcomes -- so they are
    strongly dependent.  Sidak's correction is exact only under independence
    and is not licensed here; Bonferroni holds under arbitrary dependence, by
    the union bound, which assumes nothing at all.  At eleven tests the two
    differ by under half a percent of the family alpha, so the guarantee is
    nearly free.

    ``method="sidak"`` is available for comparison and is used in the test
    suite to show how small the difference is.
    """
    if method == "bonferroni":
        return float(stats.bonferroni_alpha(alpha_family, n_tests))
    if method == "sidak":
        return float(stats.sidak_alpha(alpha_family, n_tests))
    raise ValueError(f"method must be 'bonferroni' or 'sidak', not {method!r}")


# --------------------------------------------------------------------------
# the one place every number comes from
# --------------------------------------------------------------------------

def threshold_ladder(spec_rate: float,
                     n_checks: int,
                     accept_threshold: Optional[float] = None,
                     transfer_threshold: Optional[float] = None,
                     alpha_family: float = 0.01,
                     n_tests: int = 11,
                     epsilon: float = 1e-9,
                     forger_rate: float = FORGER_ERROR_RATE,
                     multiplicity: str = "bonferroni") -> Dict[str, object]:
    """Assemble every threshold the engine will use, from declared inputs only.

    ``accept_threshold`` and ``transfer_threshold`` are normally supplied by
    the session's :class:`~qds.protocol.verification.VerificationPolicy`, so
    that the detection engine judges against exactly the numbers the protocol
    used rather than against a second set of its own.  When they are absent --
    a bare evidence bundle with no policy attached -- they are derived here
    from ``spec_rate`` and ``n_checks`` by the same construction the policy
    uses, and ``derived`` is set so the report says which happened.

    The returned dictionary is what every rule reads and what the JSON report
    echoes, so a number that appears in the dashboard can always be traced to
    one line of this function.
    """
    n = int(n_checks)
    spec = float(spec_rate)

    derived = accept_threshold is None or transfer_threshold is None
    if derived:
        # Put the acceptance threshold where a false alarm costs epsilon, and
        # the transfer threshold where a missed forgery costs epsilon, then
        # let soundness_problems complain if they have crossed over.
        lo = honest_side_threshold(n, spec, epsilon) if n > 0 else spec
        hi = forger_side_threshold(n, forger_rate, epsilon) if n > 0 else forger_rate
        accept_threshold = accept_threshold if accept_threshold is not None else lo
        transfer_threshold = (transfer_threshold if transfer_threshold is not None
                              else max(hi, accept_threshold))

    s_a = float(accept_threshold)
    s_v = float(transfer_threshold)
    alpha_test = multiplicity_alpha(alpha_family, n_tests, multiplicity)

    ladder = {
        "spec_rate": _f(spec),
        "accept": _f(s_a),
        "transfer": _f(s_v),
        "forger_rate": _f(forger_rate),
        "blind_forger_rate": _f(BLIND_FORGER_ERROR_RATE),
        "n_checks": n,
        "alpha_family": _f(alpha_family),
        "alpha_per_test": _f(alpha_test),
        "n_tests": int(n_tests),
        "multiplicity": multiplicity,
        "epsilon": _f(epsilon),
        "derived": bool(derived),
        "ordered": bool(spec < s_a <= s_v < forger_rate),
        "critical_count_vs_spec": critical_count(n, spec, alpha_test),
        "accept_count": int(math.floor(s_a * n)) if n > 0 else 0,
        "transfer_count": int(math.floor(s_v * n)) if n > 0 else 0,
    }
    ladder["security"] = {
        "accept": security_exponents(n, s_a, spec, forger_rate),
        "transfer": security_exponents(n, s_v, spec, forger_rate),
    }
    return ladder


def ladder_problems(ladder: Dict[str, object]) -> List[str]:
    """Reasons the ladder does not support a security claim, in plain words.

    Kept separate from :func:`threshold_ladder` so that an unusable ladder is
    still *returned* and still *reported*.  Silently substituting a workable
    threshold for an unworkable one is how a framework ends up certifying a
    link it should have refused.
    """
    out: List[str] = []
    spec = float(ladder["spec_rate"] or 0.0)
    s_a = float(ladder["accept"] or 0.0)
    s_v = float(ladder["transfer"] or 0.0)
    forger = float(ladder["forger_rate"] or FORGER_ERROR_RATE)
    n = int(ladder["n_checks"])

    if n <= 0:
        out.append("no checks were run, so no threshold means anything")
    if spec >= forger:
        out.append(
            f"declared noise floor {spec:.4f} is at or above the forger's "
            f"error rate {forger:.4f}; an honest link is indistinguishable "
            f"from an adversary and no threshold can separate them")
    if not spec < s_a:
        out.append(f"acceptance threshold {s_a:.4f} does not sit above the "
                   f"declared floor {spec:.4f}")
    if not s_a <= s_v:
        out.append(f"transfer threshold {s_v:.4f} is below the acceptance "
                   f"threshold {s_a:.4f}; a signature could be transferable "
                   f"without being acceptable")
    if not s_v < forger:
        out.append(f"transfer threshold {s_v:.4f} is at or above the forger "
                   f"rate {forger:.4f}; a forger would pass")
    return out
