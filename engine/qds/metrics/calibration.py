"""False-alarm calibration: the family-wise guarantee and a check on it.

The engine runs several detectors on one session, so the honest question is not
"what is one test's false-alarm rate" but "what is the chance *any* of them
fires on a healthy link".  The design answer is Bonferroni: run each of the
``n_tests`` tests at ``alpha_family / n_tests`` and the union bound caps the
family-wise rate at ``alpha_family`` under *arbitrary* dependence between the
tests -- which matters, because the detectors are computed from the same
measurements and are anything but independent.  That is the theoretical bound,
and it is a bound, not an estimate: :func:`family_wise_bound` just states it.

:func:`monte_carlo_false_alarm` is the empirical confirmation.  It does **no
quantum simulation** -- there is nothing to simulate, because under the null the
only thing that reaches a rate test is a count of mismatches, and that count is
exactly ``Binomial(N, floor)``.  So the Monte-Carlo draws binomial counts at the
declared floor with a seeded NumPy generator, applies the real critical value
from :func:`qds.detect.thresholds.critical_count`, and reports two rates:

* ``per_test_rate`` -- how often one pooled-rate test fires at the floor;
  should land at or under ``alpha_per_test``;
* ``family_wise_rate`` -- how often at least one of ``n_tests`` *independent*
  draws fires.  Independence is the near-worst case for the union bound (real
  dependence only lowers it), so a family-wise rate under ``alpha_family`` here
  is a conservative confirmation that the Bonferroni budget holds.

It is deliberately a count-level Monte-Carlo and nothing more, which is what
keeps it well under a second for tens of thousands of trials.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..detect.thresholds import critical_count, multiplicity_alpha
from ._serial import finite as _f

__all__ = [
    "family_wise_bound",
    "monte_carlo_false_alarm",
]

#: Default seed for the calibration Monte-Carlo.  Fixed so a snapshot is
#: reproducible bit-for-bit; the date it encodes carries no meaning.
DEFAULT_SEED = 20260105


def family_wise_bound(alpha_family: float = 0.01) -> float:
    """The theoretical family-wise false-alarm guarantee -- just ``alpha_family``.

    Stated as a function for symmetry with the empirical estimate it is compared
    against.  Bonferroni gives ``P(any test fires | healthy link) <=
    alpha_family`` with no independence assumption, so the bound is simply the
    budget itself.
    """
    return float(alpha_family)


def monte_carlo_false_alarm(n_checks: int, floor_rate: float,
                            alpha_family: float = 0.01, n_tests: int = 11,
                            trials: int = 20000,
                            seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """Empirical false-alarm rates from binomial draws at the declared floor.

    Returns a self-describing dict::

        {
          "n_checks", "floor_rate", "alpha_family", "n_tests",
          "alpha_per_test",            # alpha_family / n_tests (Bonferroni)
          "critical_count",            # the integer boundary the test uses
          "trials", "seed",
          "per_test_rate",             # empirical P(one test fires at floor)
          "family_wise_rate",          # empirical P(>=1 of n_tests fires)
          "per_test_se", "family_wise_se",   # binomial standard errors
          "family_wise_bound",         # = alpha_family, the thing being checked
          "within_budget"             # family_wise_rate <= alpha_family + 3 SE
        }

    The ``within_budget`` flag allows three standard errors of Monte-Carlo slack
    so the check is about the bound holding, not about one seed's luck.
    """
    n = int(n_checks)
    trials = int(trials)
    m = int(n_tests)
    floor = float(floor_rate)
    alpha_per_test = float(multiplicity_alpha(alpha_family, m))
    crit = critical_count(n, floor, alpha_per_test)

    rng = np.random.default_rng(int(seed))
    # One test: how often a single binomial count at the floor reaches crit.
    counts = rng.binomial(n, floor, size=trials)
    per_test_hits = int(np.count_nonzero(counts >= crit))
    per_test_rate = per_test_hits / trials if trials else 0.0

    # Family-wise: n_tests independent draws, alarm if ANY reaches crit.  Under
    # independence this is the near-worst case for the union bound.
    family = rng.binomial(n, floor, size=(trials, m))
    family_hits = int(np.count_nonzero(np.any(family >= crit, axis=1)))
    family_wise_rate = family_hits / trials if trials else 0.0

    def _se(rate: float) -> float:
        if trials <= 0:
            return 0.0
        return math.sqrt(max(rate * (1.0 - rate), 0.0) / trials)

    family_se = _se(family_wise_rate)
    within = family_wise_rate <= float(alpha_family) + 3.0 * family_se

    return {
        "n_checks": n,
        "floor_rate": _f(floor),
        "alpha_family": _f(alpha_family),
        "n_tests": m,
        "alpha_per_test": _f(alpha_per_test),
        "critical_count": int(crit),
        "trials": trials,
        "seed": int(seed),
        "per_test_rate": _f(per_test_rate),
        "family_wise_rate": _f(family_wise_rate),
        "per_test_se": _f(_se(per_test_rate)),
        "family_wise_se": _f(family_se),
        "family_wise_bound": _f(family_wise_bound(alpha_family)),
        "within_budget": bool(within),
    }
