"""Performance and analytics for the QDS framework -- statistics, not learning.

This package is the campaign-level counterpart to :mod:`qds.detect`.  Where the
detector judges one session, the metrics here describe the *scheme*: how its
forgery resistance, detection power, and false-alarm rate behave as a function
of key length and threshold placement.  It is the data layer behind the
analytics dashboard.

The binding constraint of the whole project applies here in full: **nothing is
trained, fitted, weighted, or learned.**  Every figure is either a closed-form
expression in quantities declared before any run -- the noise floor, the
thresholds, the ``1/4`` single-copy bound -- or an exact binomial tail, or a
single seeded Monte-Carlo over binomial counts (never a quantum state).  The
tail bounds and distributions are reused verbatim from :mod:`qds.detect.stats`
and :mod:`qds.detect.thresholds`; this package assembles them, it does not
reimplement the mathematics.

Three analyses, one summary
---------------------------
``forgery``
    ``exp(-N * D(s_v || 1/4))`` as a function of ``N`` and its inverse, the key
    length a target forgery probability requires.  Reproduces the whitepaper's
    ``N = 1080`` lab-grade, one-in-a-billion figure.
``power``
    the probability the pooled-rate exact binomial test fires when the true
    error rate exceeds the floor -- detection power versus ``N`` and versus the
    induced rate.
``calibration``
    the Bonferroni family-wise false-alarm guarantee, plus a fast seeded
    Monte-Carlo that confirms the empirical rate stays within budget.

The short path for a caller is :func:`evaluate` (alias :func:`summary`), which
returns a single JSON-serialisable dict bundling all three for a snapshot.
"""

from __future__ import annotations

from ..detect.thresholds import BLIND_FORGER_ERROR_RATE, FORGER_ERROR_RATE
from .calibration import family_wise_bound, monte_carlo_false_alarm
from .forgery import (forgery_curve, forgery_divergence,
                      forgery_survival_bound, required_checks_for_forgery)
from .performance import evaluate, summary
from .power import detection_power, power_by_rate, power_curve

__all__ = [
    # the short path
    "evaluate", "summary",
    # forgery
    "forgery_divergence", "forgery_survival_bound",
    "required_checks_for_forgery", "forgery_curve",
    # detection power
    "detection_power", "power_curve", "power_by_rate",
    # false-alarm calibration
    "family_wise_bound", "monte_carlo_false_alarm",
    # the declared bounds, re-exported for convenience
    "FORGER_ERROR_RATE", "BLIND_FORGER_ERROR_RATE",
]
