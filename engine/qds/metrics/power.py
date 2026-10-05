"""Detection power: how often the pooled-rate test fires on a real excess.

The forgery bound of :mod:`qds.metrics.forgery` answers "how well does a long
key resist an adversary at the ``1/4`` bound".  This module answers the
operational question underneath it: given a channel whose true per-check error
rate is some ``r`` *above* the declared floor -- because an attacker is
disturbing it, or because the hardware has drifted -- how probably does the
engine's pooled-rate detector actually raise the alarm, as a function of the
number of checks ``N``?

The detector in question is ``PooledRateVsSpec`` in :mod:`qds.detect.rules`: an
exact one-sided binomial test of the pooled mismatch count against the declared
floor, run at the Bonferroni-corrected per-test level ``alpha_per_test``.  It
rejects when the count reaches ``critical_count(N, floor, alpha_per_test)``.
Its *power* at a true rate ``r`` is therefore the exact binomial upper tail

    ``power(N, r) = P(X >= k_crit)``,   ``X ~ Binomial(N, r)``

reusing :func:`qds.detect.stats.binom_sf` and the engine's own
:func:`qds.detect.thresholds.critical_count` so the curve describes the detector
that actually ships, not an idealisation of it.

Three properties hold by construction and are pinned in the tests:

* at ``r = floor`` the power is the test's size, ``<= alpha_per_test`` (it is a
  tail at or beyond the critical value, which is where the level was spent);
* it is non-decreasing in ``N`` on any reasonable grid and ``-> 1`` as
  ``N -> infinity`` for every fixed ``r > floor``;
* it is non-decreasing in ``r`` at fixed ``N``.

Nothing is fitted and nothing is simulated; every value is an exact tail sum.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..detect import stats
from ..detect.thresholds import critical_count
from ._serial import finite as _f

__all__ = [
    "detection_power",
    "power_curve",
    "power_by_rate",
]


def detection_power(n: int, true_rate: float, floor_rate: float,
                    alpha_per_test: float) -> float:
    """Probability the pooled-rate test rejects when the true rate is ``r``.

    ``floor_rate`` is the declared null the critical value is computed against;
    ``true_rate`` is the rate actually in force.  With ``true_rate = floor_rate``
    this returns the test's realised size (at most ``alpha_per_test``); with
    ``true_rate > floor_rate`` it returns the detection probability.
    """
    crit = critical_count(int(n), float(floor_rate), float(alpha_per_test))
    return float(stats.binom_sf(crit, int(n), float(true_rate)))


def power_curve(true_rate: float, floor_rate: float, alpha_per_test: float,
                n_values: Sequence[int]) -> List[Dict[str, object]]:
    """Detection power versus key length, as plottable points.

    Each point is ``{"n", "power", "critical_count"}``.  The critical count is
    carried alongside so a reader can see the integer boundary the exact test
    uses, and why the curve has the small flat steps that a discrete test
    always has.
    """
    out: List[Dict[str, object]] = []
    for n in n_values:
        crit = critical_count(int(n), float(floor_rate), float(alpha_per_test))
        p = float(stats.binom_sf(crit, int(n), float(true_rate)))
        out.append({"n": int(n), "power": _f(p), "critical_count": int(crit)})
    return out


def power_by_rate(n: int, floor_rate: float, alpha_per_test: float,
                  rates: Sequence[float]) -> List[Dict[str, object]]:
    """Detection power versus the true error rate, at a fixed key length.

    The companion view to :func:`power_curve`: the critical value is fixed by
    ``n`` and ``floor_rate``, and each point ``{"rate", "power"}`` reports the
    chance of catching a channel sitting at that rate.  Non-decreasing in
    ``rate``.
    """
    crit = critical_count(int(n), float(floor_rate), float(alpha_per_test))
    out: List[Dict[str, object]] = []
    for r in rates:
        p = float(stats.binom_sf(crit, int(n), float(r)))
        out.append({"rate": _f(r), "power": _f(p)})
    return out
