"""Forgery probability as a function of key length.

The headline security figure of the scheme is a single bound.  A forger who
holds one copy of the quantum public key cannot push his per-check error rate
below ``1/4`` -- that number is the optimal single-copy discrimination error for
the BB84 ensemble (Lemma 2 of the whitepaper), not a tuning parameter -- so the
chance his observed rate over ``N`` checks falls under the transfer threshold
``s_v`` is bounded by a Chernoff-Hoeffding tail

    ``P(forger accepted) <= exp(-N * D(s_v || 1/4))``

where ``D`` is the binary relative entropy.  The exponent ``D(s_v || 1/4)`` is a
constant of the threshold placement; the probability therefore decays
geometrically in ``N``, which is the whole reason a one-in-a-billion target is
affordable with a key of only ~1000 checks.

This module is the forward map (bound as a function of ``N``) and its inverse
(the ``N`` a target forgery probability requires).  It computes nothing that is
not a closed form in declared quantities -- there is no fitting and no sampling
here.  Every tail is :func:`qds.detect.stats.chernoff_kl_tail_below`, reused
rather than reimplemented, so the number this module prints is the same number
the detection engine's ``security_exponents`` prints for the same threshold.

Reproducing the whitepaper
---------------------------
At the lab-grade floor the ``balanced`` placement puts ``s_v = 0.1687154``, for
which ``D(s_v || 1/4) = 0.0191914`` nats, and :func:`required_checks_for_forgery`
at ``epsilon = 1e-9`` returns **1080** -- the figure quoted in whitepaper §5.7.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from ..detect import stats
from ..detect.thresholds import (BLIND_FORGER_ERROR_RATE, FORGER_ERROR_RATE,
                                 bits)
from ._serial import finite as _f

__all__ = [
    "forgery_divergence",
    "forgery_survival_bound",
    "required_checks_for_forgery",
    "forgery_curve",
]


def forgery_divergence(transfer_threshold: float,
                       single_copy_bound: float = FORGER_ERROR_RATE) -> float:
    """The per-check forgery exponent ``D(s_v || 1/4)`` in nats.

    Strictly positive whenever ``0 <= s_v < 1/4``; it is this exponent, not the
    gap ``1/4 - s_v`` itself, that sets how fast the forgery bound decays, and
    the two are not proportional -- the relative entropy climbs steeply as the
    threshold moves away from the bound.
    """
    s_v = float(transfer_threshold)
    bound = float(single_copy_bound)
    if not 0.0 <= s_v < bound:
        raise ValueError(
            f"transfer threshold {s_v} must lie in [0, {bound}) for a forgery "
            f"exponent to exist")
    return float(stats.binary_kl(s_v, bound))


def forgery_survival_bound(n: int, transfer_threshold: float,
                           single_copy_bound: float = FORGER_ERROR_RATE) -> float:
    """``exp(-N * D(s_v || 1/4))`` -- the chance a one-copy forger is accepted.

    An upper bound, by Chernoff-Hoeffding, on the probability that a forger
    whose true per-check error rate is at least ``single_copy_bound`` is
    nonetheless observed with a rate at or below ``transfer_threshold`` over
    ``n`` checks.  Monotonically decreasing in ``n`` and in the gap from the
    threshold to the bound.  Returns ``1.0`` for a threshold at or above the
    bound, where no finite ``n`` secures anything.
    """
    return float(stats.chernoff_kl_tail_below(
        int(n), float(single_copy_bound), float(transfer_threshold)))


def required_checks_for_forgery(
        epsilon: float,
        transfer_threshold: float,
        single_copy_bound: float = FORGER_ERROR_RATE) -> Optional[int]:
    """Smallest ``N`` whose forgery bound falls to ``epsilon`` or below.

    The inverse of :func:`forgery_survival_bound`.  Because the bound is
    ``exp(-N * D)`` the inversion is exact and closed-form::

        N = ceil( ln(1/epsilon) / D(s_v || 1/4) )

    Returns ``None`` when no finite key length suffices -- a transfer threshold
    at or above the single-copy bound, where the exponent is zero or undefined
    and a forger is never priced out.  This is the operational answer "there is
    no such N", reported as ``None`` rather than a bogus large integer, matching
    how :func:`qds.detect.thresholds.bits` reports an unbounded exponent.
    Unlike :func:`forgery_divergence`, it does not raise on such a threshold:
    "how many checks?" has the honest answer "no finite number".
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    s_v = float(transfer_threshold)
    bound = float(single_copy_bound)
    if not 0.0 <= s_v < bound:
        return None
    divergence = stats.binary_kl(s_v, bound)
    if divergence <= 0.0:
        return None
    return int(math.ceil(math.log(1.0 / epsilon) / divergence))


def forgery_curve(transfer_threshold: float,
                  n_values: Sequence[int],
                  single_copy_bound: float = FORGER_ERROR_RATE
                  ) -> List[Dict[str, object]]:
    """The forgery bound sampled over ``n_values`` as plottable points.

    Each point is ``{"n", "bound", "bits"}`` where ``bits`` is ``-log2(bound)``,
    the readable unit for a probability that runs to ``1e-60``.  The list is in
    the order given; the caller chooses the grid.
    """
    out: List[Dict[str, object]] = []
    for n in n_values:
        b = forgery_survival_bound(int(n), transfer_threshold, single_copy_bound)
        out.append({"n": int(n), "bound": _f(b), "bits": _f(bits(b))})
    return out
