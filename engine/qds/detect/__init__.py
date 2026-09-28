"""Threat detection for the QDS link: statistics only, no learned models.

The constraint that shapes this package is that nothing in it may be trained,
fitted, or weighted from data.  Every decision comes out of a hypothesis test
with a null the protocol declared in advance, and every threshold is a closed
form in quantities fixed before the run.  That is not a handicap imposed from
outside -- it is what makes the resulting statement worth anything.  A learned
detector can tell you that a run looks unlike the runs it was shown.  A test
against the ``1/4`` single-copy bound tells you that an adversary who learned
enough to forge *must* have left a trace this large, whatever hardware he owns
and however much computation he has.  The second statement survives an
adversary the first has never seen.

Reading order, if you are coming to this cold:

``stats``
    scipy-free exact distributions and tail bounds.  Nothing QDS-specific.
``estimators``
    turns verification outcomes into rates with Clopper-Pearson intervals,
    and is careful about which denominator each rate uses.
``thresholds``
    the declared-inputs-only threshold ladder, and what each threshold costs
    in false alarms and missed forgeries.
``rules``
    the eleven detectors, one null hypothesis each.
``engine``
    the privilege boundary (:class:`Evidence`), the runner, the attribution
    table, and the verdict.

The short path for a caller is :func:`analyse_session`.
"""

from __future__ import annotations

from .engine import (Attribution, DetectionEngine, DetectionReport, Evidence,
                     PREMISE_RULES, analyse_session, attribute)
from .estimators import (RateDifference, RateEstimate, correction_histogram,
                         dark_estimate, decoy_rate, estimate_rate,
                         fisher_exact_two_sided, homogeneity_chi_square,
                         pooled_rate, rate_difference, rates_by_basis,
                         rates_by_level, rates_by_position, rates_by_verifier,
                         yield_estimate)
from .rules import (DEFAULT_RULES, BasisConsistency, CorrectionUniformity,
                    DarkExcess, DecoyVersusSignature, LevelHomogeneity,
                    PooledRateVsSpec, PooledRateVsTransfer,
                    PositionHomogeneity, RecipientAgreement, Rule,
                    SequentialOnset, TestOutcome, YieldVsDeclared,
                    default_rules)
from .thresholds import (BLIND_FORGER_ERROR_RATE, FORGER_ERROR_RATE,
                         DecisionThreshold, bits, critical_count,
                         forger_side_threshold, honest_side_threshold,
                         ladder_problems, multiplicity_alpha,
                         security_exponents, threshold_ladder)

__all__ = [
    # engine
    "Evidence", "DetectionEngine", "DetectionReport", "Attribution",
    "analyse_session", "attribute", "PREMISE_RULES",
    # rules
    "Rule", "TestOutcome", "default_rules", "DEFAULT_RULES",
    "PooledRateVsSpec", "PooledRateVsTransfer", "BasisConsistency",
    "PositionHomogeneity", "LevelHomogeneity", "YieldVsDeclared",
    "DarkExcess", "CorrectionUniformity", "DecoyVersusSignature",
    "RecipientAgreement", "SequentialOnset",
    # thresholds
    "FORGER_ERROR_RATE", "BLIND_FORGER_ERROR_RATE", "DecisionThreshold",
    "threshold_ladder", "ladder_problems", "security_exponents",
    "critical_count", "honest_side_threshold", "forger_side_threshold",
    "multiplicity_alpha", "bits",
    # estimators
    "RateEstimate", "RateDifference", "estimate_rate", "pooled_rate",
    "rates_by_basis", "rates_by_position", "rates_by_level",
    "rates_by_verifier", "yield_estimate", "dark_estimate",
    "correction_histogram", "decoy_rate", "rate_difference",
    "fisher_exact_two_sided", "homogeneity_chi_square",
]
