"""The campaign-level summary: one JSON-serialisable figure of merit bundle.

:func:`evaluate` ties the three analyses in this package -- forgery resistance,
detection power, and false-alarm calibration -- into a single dictionary sized
for a dashboard.  It is the function a snapshot generator calls, and its return
value is designed to be dropped straight into ``metrics.json`` and read back by a
chart with no post-processing: every leaf is a JSON-native scalar or ``None``,
every curve is a list of ``{x, value}`` points, and every constant the figures
depend on is echoed back under ``constants`` so a reader never has to guess what
floor or threshold produced a number.

The defaults reproduce the whitepaper.  With no arguments, :func:`evaluate`
reads the lab-grade noise floor from :data:`qds.channel.LAB_GRADE` and places the
thresholds with the ``balanced`` rule of
:class:`qds.protocol.verification.VerificationPolicy` -- the same construction
the protocol uses -- so the forgery figure comes out at ``N = 1080`` for a
one-in-a-billion target, matching whitepaper §5.7.  Pass ``spec_rate`` to analyse
a different floor (thresholds are then re-placed by the balanced rule), or pass
the thresholds explicitly to analyse a specific policy.

Nothing here is trained, fitted, or weighted.  The curves are closed-form tail
bounds and exact binomial tails; the single Monte-Carlo (false-alarm
calibration) draws binomial counts, never a quantum state, and is seeded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..detect.thresholds import (BLIND_FORGER_ERROR_RATE, FORGER_ERROR_RATE,
                                 multiplicity_alpha)
from . import calibration, forgery, power
from ._serial import finite as _f

__all__ = ["evaluate", "summary"]

#: Default key lengths for the curves.  Spans the reference geometries of the
#: whitepaper -- N = 384 (m=16, L=24), N = 768 (the default L=48), N = 1080 (the
#: lab-grade one-in-a-billion requirement) -- and brackets them on both sides.
_DEFAULT_N_GRID: Tuple[int, ...] = (
    96, 192, 384, 576, 768, 960, 1080, 1440, 1920)

#: Default forgery-target probabilities for the required-N table.
_DEFAULT_EPSILONS: Tuple[float, ...] = (1e-3, 1e-6, 1e-9, 1e-12)


def _lab_grade_balanced() -> Tuple[float, float, float]:
    """``(floor, s_a, s_v)`` for lab-grade hardware under the balanced rule.

    Imported lazily so the metrics package does not pull in the protocol and
    channel modules unless the whitepaper defaults are actually requested.
    """
    from ..channel import LAB_GRADE
    from ..protocol.verification import VerificationPolicy
    eta = float(LAB_GRADE.predicted_qber())
    policy = VerificationPolicy.balanced(eta)
    return eta, float(policy.s_a), float(policy.s_v)


def _balanced_for(floor: float) -> Tuple[float, float]:
    """``(s_a, s_v)`` placed on ``floor`` by the protocol's balanced rule."""
    from ..protocol.verification import VerificationPolicy
    policy = VerificationPolicy.balanced(float(floor))
    return float(policy.s_a), float(policy.s_v)


def evaluate(spec_rate: Optional[float] = None,
             accept_threshold: Optional[float] = None,
             transfer_threshold: Optional[float] = None,
             single_copy_bound: float = FORGER_ERROR_RATE,
             blind_bound: float = BLIND_FORGER_ERROR_RATE,
             alpha_family: float = 0.01,
             n_tests: int = 11,
             epsilon: float = 1e-9,
             epsilons: Sequence[float] = _DEFAULT_EPSILONS,
             n_grid: Optional[Sequence[int]] = None,
             power_rate: Optional[float] = None,
             rate_grid: Optional[Sequence[float]] = None,
             mc_checks: Optional[int] = None,
             mc_trials: int = 20000,
             mc_seed: int = calibration.DEFAULT_SEED) -> Dict[str, object]:
    """Bundle the forgery, power and calibration figures into one dict.

    Parameters are all optional; with none supplied the lab-grade whitepaper
    defaults are used (see the module docstring).  ``spec_rate`` alone re-places
    the thresholds with the balanced rule; supplying the thresholds pins them.

    Returns a JSON-serialisable dictionary with this shape::

        {
          "constants": {
            "spec_rate", "accept_threshold", "transfer_threshold",
            "single_copy_bound", "blind_bound",
            "alpha_family", "n_tests", "alpha_per_test", "epsilon"
          },
          "forgery": {
            "transfer_threshold", "single_copy_bound", "blind_bound",
            "divergence_nats",            # D(s_v || 1/4)
            "blind_divergence_nats",      # D(s_v || 1/2)
            "required_checks": 1080,      # N for the headline epsilon, one copy
            "required_checks_blind",      # N for the headline epsilon, blind
            "required_checks_by_epsilon": [ {"epsilon", "n", "n_blind"}, ... ],
            "curve":       [ {"n", "bound", "bits"}, ... ],   # one-copy
            "curve_blind": [ {"n", "bound", "bits"}, ... ]
          },
          "power": {
            "floor_rate", "alpha_per_test", "true_rate",
            "curve":   [ {"n", "power", "critical_count"}, ... ],  # vs N at true_rate
            "by_rate": [ {"rate", "power"}, ... ]                  # vs rate at N*
          },
          "calibration": { ... see monte_carlo_false_alarm ... },
          "grid": { "n_values": [...], "n_power_fixed": N*, "epsilons": [...] }
        }

    A tiny sample of the leaves (lab-grade defaults)::

        out["forgery"]["required_checks"]            == 1080
        out["forgery"]["divergence_nats"]            ~= 0.0191914
        out["power"]["curve"][0]                     == {"n": 96, "power": ..., "critical_count": ...}
        out["calibration"]["within_budget"]          is True
    """
    if spec_rate is None:
        floor, sa_default, sv_default = _lab_grade_balanced()
    else:
        floor = float(spec_rate)
        sa_default, sv_default = _balanced_for(floor)

    s_a = float(accept_threshold) if accept_threshold is not None else sa_default
    s_v = (float(transfer_threshold) if transfer_threshold is not None
           else sv_default)

    grid: List[int] = [int(n) for n in (n_grid if n_grid is not None
                                        else _DEFAULT_N_GRID)]
    alpha_per_test = float(multiplicity_alpha(alpha_family, n_tests))

    # ---- forgery -----------------------------------------------------------
    divergence = forgery.forgery_divergence(s_v, single_copy_bound)
    blind_divergence = forgery.forgery_divergence(s_v, blind_bound)
    required_by_eps: List[Dict[str, object]] = []
    for eps in epsilons:
        required_by_eps.append({
            "epsilon": _f(eps),
            "n": forgery.required_checks_for_forgery(eps, s_v, single_copy_bound),
            "n_blind": forgery.required_checks_for_forgery(eps, s_v, blind_bound),
        })

    forgery_block = {
        "transfer_threshold": _f(s_v),
        "single_copy_bound": _f(single_copy_bound),
        "blind_bound": _f(blind_bound),
        "divergence_nats": _f(divergence),
        "blind_divergence_nats": _f(blind_divergence),
        "required_checks": forgery.required_checks_for_forgery(
            epsilon, s_v, single_copy_bound),
        "required_checks_blind": forgery.required_checks_for_forgery(
            epsilon, s_v, blind_bound),
        "required_checks_by_epsilon": required_by_eps,
        "curve": forgery.forgery_curve(s_v, grid, single_copy_bound),
        "curve_blind": forgery.forgery_curve(s_v, grid, blind_bound),
    }

    # ---- detection power ---------------------------------------------------
    # Default the illustrated true rate to the acceptance threshold: a declared
    # quantity comfortably above the floor, so the power curve climbs across the
    # grid rather than sitting pinned at 0 or 1.
    true_rate = float(power_rate) if power_rate is not None else s_a
    n_power_fixed = forgery.required_checks_for_forgery(
        epsilon, s_v, single_copy_bound) or grid[-1]
    if rate_grid is not None:
        rates = [float(r) for r in rate_grid]
    else:
        # Nine points from the floor up to the transfer threshold.
        span = s_v - floor
        rates = [floor + span * i / 8.0 for i in range(9)]

    power_block = {
        "floor_rate": _f(floor),
        "alpha_per_test": _f(alpha_per_test),
        "true_rate": _f(true_rate),
        "curve": power.power_curve(true_rate, floor, alpha_per_test, grid),
        "by_rate": power.power_by_rate(n_power_fixed, floor, alpha_per_test,
                                       rates),
    }

    # ---- false-alarm calibration ------------------------------------------
    mc_n = int(mc_checks) if mc_checks is not None else int(n_power_fixed)
    calibration_block = calibration.monte_carlo_false_alarm(
        n_checks=mc_n, floor_rate=floor, alpha_family=alpha_family,
        n_tests=n_tests, trials=mc_trials, seed=mc_seed)

    return {
        "constants": {
            "spec_rate": _f(floor),
            "accept_threshold": _f(s_a),
            "transfer_threshold": _f(s_v),
            "single_copy_bound": _f(single_copy_bound),
            "blind_bound": _f(blind_bound),
            "alpha_family": _f(alpha_family),
            "n_tests": int(n_tests),
            "alpha_per_test": _f(alpha_per_test),
            "epsilon": _f(epsilon),
        },
        "forgery": forgery_block,
        "power": power_block,
        "calibration": calibration_block,
        "grid": {
            "n_values": [int(n) for n in grid],
            "n_power_fixed": int(n_power_fixed),
            "epsilons": [_f(e) for e in epsilons],
        },
    }


def summary(**kwargs: object) -> Dict[str, object]:
    """Alias for :func:`evaluate`.  Accepts the same keyword arguments."""
    return evaluate(**kwargs)  # type: ignore[arg-type]
