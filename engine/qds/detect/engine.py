"""The boundary between what a verifier knows and what the simulator knows.

:class:`Evidence` is the only object in this package that touches a session,
and it exists to make the privileged/unprivileged split a matter of code
rather than of discipline.  A running :class:`~qds.protocol.session.QDSSession`
carries two kinds of information.  One kind is what a deployed recipient would
actually hold: the outcomes he measured, the announcements he received, his own
detector log, and the spec sheet for the hardware he bought.  The other kind is
ground truth that only exists because this is a simulation -- the true fidelity
of each delivered qubit, the entanglement witnesses, the adversary's actual
probe state.  That second kind is invaluable for *checking* the engine and
would be fatal to *use* inside it: a detector that consults it is not detecting
anything, it is reading the answer.

So ``Evidence`` copies out the first kind and nothing else, and
``tests/test_detect.py`` reads this package off disk and asserts the names of
the privileged quantities appear nowhere in it.  That grep is part of the
security argument, not a style check.  Everything downstream -- the eleven
detectors, the attribution table, the numbers on the dashboard -- is a function
of ``Evidence`` alone, so if the boundary holds there, it holds everywhere.

The engine's job ends at "this link departs from its declared specification,
by this much, with this p-value."  It deliberately stops short of "you are
being attacked."  A raised error rate is consistent with an eavesdropper and
equally consistent with a drifting polarisation controller, and no amount of
statistics distinguishes them from inside the protocol.  What the framework
can say -- and what makes it useful -- is that the departure is there, that it
is larger than sampling noise, and that an adversary who learned anything about
the key *must* have produced one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import stats
from . import thresholds as T
from .estimators import (RateEstimate, correction_histogram, dark_estimate,
                         decoy_rate, pooled_rate, rates_by_basis,
                         rates_by_level, rates_by_position, rates_by_verifier,
                         yield_estimate)
from .rules import Rule, TestOutcome, default_rules

__all__ = [
    "Evidence",
    "Attribution",
    "DetectionReport",
    "DetectionEngine",
    "analyse_session",
]


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

@dataclass
class Evidence:
    """Everything a verifier legitimately knows, and nothing else."""

    reports: List[Any] = field(default_factory=list)
    calibrations: List[Any] = field(default_factory=list)
    detector_logs: Dict[str, List[str]] = field(default_factory=dict)
    corrections: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    #: declared, fixed before the run -- never estimated from this session
    spec_rate: float = 0.0
    spec_rate_z: Optional[float] = None
    spec_rate_x: Optional[float] = None
    accept_threshold: Optional[float] = None
    transfer_threshold: Optional[float] = None
    declared_yield: Optional[float] = None
    declared_dark_fraction: Optional[float] = None
    forger_rate: float = T.FORGER_ERROR_RATE

    alpha_family: float = 0.01
    estimate_alpha: float = 0.05
    n_tests: int = len(default_rules())

    signer: str = ""
    key_id: str = ""
    spec_violations: Tuple[str, ...] = ()

    #: Where the declared hardware figures came from.  ``"spec_sheet"`` means
    #: a declaration independent of the channel actually simulated;
    #: ``"simulated_model"`` means they were read off the same model the
    #: channel used, which makes :class:`~qds.detect.rules.YieldVsDeclared`
    #: and :class:`~qds.detect.rules.DarkExcess` comparisons of a quantity
    #: with itself.  Those tests then cannot fail, and saying so is the
    #: difference between a passed test and an absent one.
    declaration_source: str = "simulated_model"

    # -- derived, filled in by __post_init__ ------------------------------
    estimates: Dict[str, RateEstimate] = field(default_factory=dict, init=False)
    estimates_by_basis: Dict[str, RateEstimate] = field(default_factory=dict, init=False)
    estimates_by_position: Dict[int, RateEstimate] = field(default_factory=dict, init=False)
    estimates_by_level: Dict[str, RateEstimate] = field(default_factory=dict, init=False)
    estimates_by_verifier: Dict[str, RateEstimate] = field(default_factory=dict, init=False)
    correction_counts: List[int] = field(default_factory=list, init=False)
    ordered_mismatches: List[bool] = field(default_factory=list, init=False)
    ladder: Dict[str, Any] = field(default_factory=dict, init=False)
    rejections: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        a = self.estimate_alpha
        self.estimates = {
            "pooled_rate": pooled_rate(self.reports, a),
            "detector_yield": yield_estimate(
                self.reports if not self.detector_logs else (),
                list(self.detector_logs.values()), a),
            "dark_fraction": dark_estimate(
                self.reports if not self.detector_logs else (),
                list(self.detector_logs.values()), a),
            "decoy_rate": decoy_rate(self.calibrations, a),
        }
        self.estimates_by_basis = rates_by_basis(self.reports, a)
        self.estimates_by_position = rates_by_position(self.reports, a)
        self.estimates_by_level = rates_by_level(self.reports, a)
        self.estimates_by_verifier = rates_by_verifier(self.reports, a)

        counts = [0, 0, 0, 0]
        for history in self.corrections.values():
            for i, c in enumerate(correction_histogram(history)):
                counts[i] += c
        self.correction_counts = counts

        # Report order is transmission order within each report, which is what
        # a sequential monitor would have seen live.
        self.ordered_mismatches = [
            bool(c.mismatch) for r in self.reports for c in r.checks if c.counted
        ]

        self.ladder = T.threshold_ladder(
            spec_rate=self.spec_rate,
            n_checks=self.estimates["pooled_rate"].n,
            accept_threshold=self.accept_threshold,
            transfer_threshold=self.transfer_threshold,
            alpha_family=self.alpha_family,
            n_tests=self.n_tests,
            forger_rate=self.forger_rate,
        )

        # The protocol's own accept/reject decision, copied verbatim rather
        # than recomputed.  It is not a hypothesis test and carries no
        # p-value, so it is deliberately not one of the eleven rules -- giving
        # it a Bonferroni-corrected alpha would be a category error.  But it
        # is the decision that actually matters, and two whole attack families
        # are caught here and nowhere else: an impersonation attempt fails the
        # MAC check, and a replay fails the freshness check, neither of which
        # raises the error rate at all.  An engine that only ran the
        # statistical tests would call both of those runs clean.
        self.rejections = [
            {
                "verifier": str(r.verifier),
                "level": str(r.level),
                "rate": _f(r.mismatch_rate),
                "threshold": _f(r.threshold),
                "n_checked": int(r.n_checked),
                "rate_exceeded": bool(r.measured
                                      and r.mismatch_rate > r.threshold),
                "identity_ok": bool(r.identity_ok),
                "mac_ok": bool(r.mac_ok),
                "fresh": (None if r.freshness is None else bool(r.freshness.ok)),
                "reasons": list(r.reasons),
            }
            for r in self.reports if not r.accepted
        ]

    @property
    def protocol_rejects(self) -> bool:
        """Did any recipient refuse the signature on the protocol's own rules?"""
        return bool(self.rejections)

    @property
    def authentication_failed(self) -> bool:
        """Was a rejection caused by identity, MAC, or freshness rather than noise?

        Separated from the rate breach because the two mean different things.
        A rate breach says the quantum channel is disturbed and might be an
        eavesdropper or might be a bad day for the fibre.  An authentication
        failure is not ambiguous: the tag did not verify, or the counter went
        backwards, and honest noise cannot produce either.
        """
        return any(not r["identity_ok"] or not r["mac_ok"]
                   or r["fresh"] is False for r in self.rejections)

    @property
    def alpha_per_test(self) -> float:
        return float(self.ladder["alpha_per_test"])

    # -- construction from a live session ---------------------------------
    @classmethod
    def from_session(cls, session: Any, *, alpha_family: float = 0.01,
                     estimate_alpha: float = 0.05,
                     levels: Optional[Sequence[str]] = None,
                     declared_noise: Any = None) -> "Evidence":
        """Pull the unprivileged half of a session into an evidence bundle.

        Note what is *not* read.  ``DistributionResult.summary()`` embeds the
        simulator's ground-truth view under an ``"oracle"`` key; this method
        never calls it.  The stores are read only for the recipient's own
        detector log, the publicly announced Pauli corrections, and nothing
        else.

        ``declared_noise`` is the spec sheet: the :class:`~qds.channel.NoiseModel`
        the operator believes he bought, as against the one the channel is
        actually running.  Supplying it is what makes the loss and dark-count
        tests capable of failing.  Without it they are read off the simulated
        channel itself, which is a comparison of a quantity with itself, and
        the bundle records ``declaration_source="simulated_model"`` so the
        report does not present a vacuous test as a passed one.

        This is the shape of the real problem, not a simulation artefact.  A
        deployed verifier never observes the true transmittance of his fibre;
        he has a number from a datasheet and a commissioning measurement, and
        detection consists precisely of noticing that today's channel and that
        number have parted company.  The attack suite uses this to run a
        degraded channel against an honest declaration.
        """
        cfg = session.config
        reports = [r for key, r in sorted(session.reports.items())
                   if levels is None or r.level in levels]
        calibrations = list(session.calibrations.values())

        detector_logs = {name: list(store.detector_log)
                         for name, store in session.stores.items()}
        corrections = {name: list(store.correction_history)
                       for name, store in session.stores.items()}

        spec = float(cfg.spec_rate())
        simulated = cfg.distribution.noise
        declared = declared_noise if declared_noise is not None else simulated
        source = "spec_sheet" if declared_noise is not None else "simulated_model"
        spec_z, spec_x = _per_basis_spec(spec, declared)

        policy = session.policy
        accept = float(policy.s_a) if policy is not None else None
        transfer = float(policy.s_v) if policy is not None else None
        forger = float(getattr(policy, "forger_error_rate", T.FORGER_ERROR_RATE)
                       if policy is not None else T.FORGER_ERROR_RATE)

        loss = declared.loss
        arrive = loss.transmittance * loss.detector_efficiency
        return cls(
            reports=reports,
            calibrations=calibrations,
            detector_logs=detector_logs,
            corrections=corrections,
            spec_rate=spec,
            spec_rate_z=spec_z,
            spec_rate_x=spec_x,
            accept_threshold=accept,
            transfer_threshold=transfer,
            declared_yield=float(loss.yield_),
            # Per *addressed* slot, to match the denominator the estimator
            # uses.  NoiseModel.dark_fraction is per *registered* detection,
            # a different quantity, and substituting one for the other would
            # bias this test by a factor of the yield.
            declared_dark_fraction=float((1.0 - arrive) * loss.dark_count),
            forger_rate=forger,
            alpha_family=alpha_family,
            estimate_alpha=estimate_alpha,
            signer=str(cfg.signer),
            key_id=str(getattr(session.private_key, "key_id", "")),
            spec_violations=tuple(session.spec_violations),
            declaration_source=source,
        )


def _per_basis_spec(spec: float, noise: Any) -> Tuple[Optional[float],
                                                      Optional[float]]:
    """Split a pooled declared floor into per-basis floors.

    The declared noise floor is a single number on a spec sheet, but the
    honest error rate is not the same in both bases: memory dephasing spares
    Z eigenstates, and an X-basis check passes through more depolarising gate
    insertions than a Z-basis one.  The *ratio* between the two is a fixed
    property of the declared hardware design -- how many gates, whether the
    memory dephases -- and is therefore part of the specification, not
    something estimated from the run being judged.

    So the pooled floor is split in the proportion the declared model
    predicts.  If no model is available, or it predicts a zero pooled rate,
    both bases fall back to the pooled figure and
    :class:`~qds.detect.rules.BasisConsistency` says so in its output.
    """
    try:
        pooled = float(noise.predicted_qber(basis="both"))
        z = float(noise.predicted_qber(basis="Z"))
        x = float(noise.predicted_qber(basis="X"))
    except Exception:
        return None, None
    if pooled <= 0.0:
        return (spec, spec) if spec > 0 else (None, None)
    return spec * z / pooled, spec * x / pooled


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

#: Detectors whose firing means a *premise* of the threshold calculus is
#: false, rather than merely that the error rate is high.  The security
#: analysis assumes the loss budget is as declared, the classical transcript
#: is uniform, and the recipients hold equivalent keys.  If any of those is
#: violated, the thresholds are being applied to a situation they were not
#: derived for, and "the rate is still under the threshold" stops being
#: reassuring.
PREMISE_RULES = frozenset({
    "yield_vs_declared",
    "dark_excess",
    "correction_uniformity",
    "recipient_agreement",
    "decoy_vs_signature",
})


@dataclass(frozen=True)
class Attribution:
    """One candidate explanation, and an honest account of its limits."""

    label: str
    matched_rule: str
    rationale: str
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "matched_rule": self.matched_rule,
                "rationale": self.rationale, "priority": int(self.priority)}


#: The attribution table, in priority order.  Each entry is
#: ``(label, condition, rationale)`` where ``condition`` takes the set of
#: flagged rule names plus the evidence and returns a bool.
#:
#: This is a lookup table, not a model.  Nothing here is fitted, weighted, or
#: learned; the conditions are fixed boolean expressions over which detectors
#: fired, and the same inputs always produce the same output.  That is a
#: deliberate constraint of the project and also the only honest option: there
#: is no labelled corpus of real quantum attacks to learn from, and a
#: classifier trained on simulated ones would be certifying the simulator.
_TABLE: List[Tuple[str, Any, str]] = [
    (
        "authentication_failure",
        lambda f, e: e.authentication_failed,
        "A recipient rejected the signature because the MAC tag, the signer's "
        "registry entry, or the anti-replay ledger did not check out -- not "
        "because of anything in the quantum channel. Honest noise cannot cause "
        "this: the tag is over classical bytes and either matches or does not, "
        "and a counter does not run backwards by accident. This is what an "
        "impersonation or replay attempt looks like, and it is the one finding "
        "in this table with no innocent explanation short of operator error.",
    ),
    (
        "signature_rejected_on_rate",
        lambda f, e: any(r["rate_exceeded"] for r in e.rejections),
        "A recipient's measured error rate reached the acceptance threshold, "
        "so the protocol refused the signature on its own terms. This is the "
        "decision that matters operationally, and it is reported separately "
        "from the statistical tests because it is not one: the threshold was "
        "fixed in advance from the declared noise floor, and crossing it is a "
        "comparison, not an inference. Note that the rate can cross the "
        "threshold while every Bonferroni-corrected test still passes -- the "
        "tests are tuned to control false alarms across eleven simultaneous "
        "nulls, which necessarily costs sensitivity.",
    ),
    (
        "repudiation_or_targeted_link_attack",
        lambda f, e: "recipient_agreement" in f,
        "Two recipients of the same declaration measured materially different "
        "error rates. They hold independently symmetrised halves of one key "
        "and check one signature, so honest noise gives them the same rate up "
        "to sampling. A real split means either one link is being attacked "
        "specifically or the signer issued unequal keys -- which is the setup "
        "for later denying the signature. This test cannot tell those apart, "
        "and a single unlucky link outage would also produce it.",
    ),
    (
        "transcript_manipulation",
        lambda f, e: "correction_uniformity" in f,
        "The announced Bell-measurement outcomes are not uniform. Uniformity "
        "is the property that makes the classical transcript leak nothing "
        "about the key, and it holds under every honest noise model, so a "
        "bias is not explicable as hardware degradation in the way an error "
        "rate is. The realistic causes are interference with the Bell "
        "measurement, with the classical channel, or a defect in the signer's "
        "random number source.",
    ),
    (
        "channel_blocking_or_blinding",
        lambda f, e: "yield_vs_declared" in f and not (
            {"pooled_rate_vs_spec", "basis_consistency"} & f),
        "The detector yield departs from the declared loss budget while every "
        "error rate stays within spec. That combination is the signature of an "
        "adversary who removes evidence rather than corrupting it -- measure "
        "the qubit, then drop the round instead of resending a disturbed copy "
        "-- or, if the yield is high rather than low, of injected clicks. It "
        "is equally the signature of a fibre splice, a misaligned coupler, or "
        "a detector running at the wrong bias.",
    ),
    (
        "basis_biased_probe",
        lambda f, e: "basis_consistency" in f and "pooled_rate_vs_spec" not in f,
        "One measurement basis is out of spec while the pooled rate is not. An "
        "adversary who probes a single basis gains information at a "
        "disturbance cost concentrated in that basis, and pooling the two "
        "rates halves the apparent damage -- which is exactly how such an "
        "attack slips under a single-number threshold. A polarisation or "
        "waveplate misalignment that affects one basis produces the same "
        "asymmetry.",
    ),
    (
        "selective_slot_attack",
        lambda f, e: bool({"decoy_vs_signature", "position_homogeneity",
                           "level_homogeneity"} & f),
        "The error rate is not uniform across slots that crossed the channel "
        "under identical conditions. Honest noise has no way to prefer the "
        "signature slots over the decoys, or one message-bit position over "
        "another, because which is which is not revealed until afterwards. "
        "Non-uniformity therefore points at an adversary spending a limited "
        "disturbance budget where it buys the most -- though with few checks "
        "per group, an unlucky draw can imitate it.",
    ),
    (
        "forgery_or_measure_and_resend",
        lambda f, e: "pooled_rate_vs_spec" in f and _rate(e) >= 0.75 * float(
            e.ladder["forger_rate"] or T.FORGER_ERROR_RATE),
        "The error rate has climbed into the region a forger or a "
        "measure-and-resend eavesdropper produces. An adversary holding one "
        "copy of the public key cannot do better than 3/4 per check, so a rate "
        "approaching 1/4 is what both a forgery attempt and full "
        "intercept-resend look like from the verifier's side. No honest "
        "hardware in the supported presets reaches this rate, so degradation "
        "is an unlikely but not impossible alternative.",
    ),
    (
        "intercept_and_block",
        lambda f, e: "pooled_rate_vs_spec" in f and "yield_vs_declared" in f,
        "Both the error rate and the yield are out of spec. An adversary who "
        "measures every round and forwards only some produces both at once: "
        "the rounds he resends are disturbed, and the rounds he drops are "
        "missing. Hardware that is simultaneously lossy and noisy -- an aging "
        "fibre, a detector past its cooling budget -- produces the same pair.",
    ),
    (
        "elevated_channel_error",
        lambda f, e: "pooled_rate_vs_spec" in f,
        "The error rate exceeds the declared noise floor by more than sampling "
        "explains, with no other detector giving the departure a shape. This "
        "is the generic finding: something is disturbing the quantum channel. "
        "An eavesdropper must produce it, but so does any degradation of the "
        "link, and at this level of evidence the framework cannot and should "
        "not choose between them.",
    ),
    (
        "rate_exceeds_transfer_threshold",
        lambda f, e: "pooled_rate_vs_transfer" in f,
        "The error rate has passed the threshold at which a signature stops "
        "being safely forwardable. Whatever the cause, a recipient who accepts "
        "on this evidence cannot rely on the next party reaching the same "
        "conclusion.",
    ),
    (
        "sequential_onset_detected",
        lambda f, e: "sequential_onset" in f,
        "A sequential monitor would have stopped this run before it finished, "
        "meaning the evidence became decisive partway through rather than only "
        "in aggregate. Read together with the stopping index, this separates a "
        "problem that was present from the first check from one that began "
        "mid-session.",
    ),
]


def _rate(evidence: Evidence) -> float:
    return evidence.estimates["pooled_rate"].value


def attribute(flagged: Sequence[str], evidence: Evidence) -> List[Attribution]:
    """Match the flagged detectors against the table, most specific first.

    Returns every matching entry in priority order rather than only the first,
    because a real attack usually trips several detectors and the secondary
    matches are what let an operator tell a story about the primary one.  The
    first element is the primary attribution.
    """
    fired = set(flagged)
    out: List[Attribution] = []
    for i, (label, condition, rationale) in enumerate(_TABLE):
        try:
            hit = bool(condition(fired, evidence))
        except Exception:
            hit = False
        if hit:
            out.append(Attribution(label=label, matched_rule=_which(fired, label),
                                   rationale=rationale, priority=i))
    return out


def _which(fired: set, label: str) -> str:
    """Name the detector most responsible for a label, for traceability."""
    preference = {
        "authentication_failure": [],
        "signature_rejected_on_rate": [],
        "repudiation_or_targeted_link_attack": ["recipient_agreement"],
        "transcript_manipulation": ["correction_uniformity"],
        "channel_blocking_or_blinding": ["yield_vs_declared"],
        "basis_biased_probe": ["basis_consistency"],
        "selective_slot_attack": ["decoy_vs_signature", "position_homogeneity",
                                  "level_homogeneity"],
        "forgery_or_measure_and_resend": ["pooled_rate_vs_spec"],
        "intercept_and_block": ["yield_vs_declared", "pooled_rate_vs_spec"],
        "elevated_channel_error": ["pooled_rate_vs_spec"],
        "rate_exceeds_transfer_threshold": ["pooled_rate_vs_transfer"],
        "sequential_onset_detected": ["sequential_onset"],
    }
    for name in preference.get(label, []):
        if name in fired:
            return name
    return ""


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

@dataclass
class DetectionReport:
    """The engine's finding, in a form both a person and a browser can read."""

    verdict: str
    alarm: bool
    outcomes: List[TestOutcome]
    attributions: List[Attribution]
    evidence: Evidence
    combined_p_value: Optional[float]
    elapsed_seconds: float = 0.0

    @property
    def flagged(self) -> List[str]:
        return [o.name for o in self.outcomes if o.flagged]

    @property
    def n_applicable(self) -> int:
        return sum(1 for o in self.outcomes if o.applicable)

    def to_dict(self) -> Dict[str, Any]:
        ladder = self.evidence.ladder
        sec = ladder.get("security", {}).get("accept", {})
        bits_ = sec.get("bits", {}) if isinstance(sec, dict) else {}
        return {
            "verdict": self.verdict,
            "alarm": bool(self.alarm),
            "combined_p_value": _f(self.combined_p_value),
            "alpha": _f(self.evidence.alpha_family),
            "alpha_per_test": _f(self.evidence.alpha_per_test),
            "n_tests": len(self.outcomes),
            "n_applicable": self.n_applicable,
            "n_flagged": len(self.flagged),
            "flagged": list(self.flagged),
            "attribution": [a.to_dict() for a in self.attributions],
            "tests": [o.to_dict() for o in self.outcomes],
            "estimates": {k: v.to_dict()
                          for k, v in self.evidence.estimates.items()},
            "estimates_by_basis": {k: v.to_dict() for k, v
                                   in self.evidence.estimates_by_basis.items()},
            "thresholds": {
                "spec_rate": ladder.get("spec_rate"),
                "accept": ladder.get("accept"),
                "transfer": ladder.get("transfer"),
                "forger_rate": ladder.get("forger_rate"),
                "ordered": ladder.get("ordered"),
                "n_checks": ladder.get("n_checks"),
                "problems": T.ladder_problems(ladder),
            },
            "security_bits": {
                "false_alarm": bits_.get("false_alarm"),
                "missed_forgery": bits_.get("missed_forgery"),
                "missed_blind_forgery": bits_.get("missed_blind_forgery"),
            },
            "spec_violations": list(self.evidence.spec_violations),
            "protocol_rejects": bool(self.evidence.protocol_rejects),
            "authentication_failed": bool(self.evidence.authentication_failed),
            "rejections": [dict(r) for r in self.evidence.rejections],
            "declaration_source": self.evidence.declaration_source,
            "declared": {
                "yield": _f(self.evidence.declared_yield),
                "dark_fraction": _f(self.evidence.declared_dark_fraction),
                "spec_rate_z": _f(self.evidence.spec_rate_z),
                "spec_rate_x": _f(self.evidence.spec_rate_x),
                "note": (
                    "Declared figures came from an independent specification."
                    if self.evidence.declaration_source == "spec_sheet" else
                    "No independent specification was supplied, so the declared "
                    "loss and dark-count figures were read off the simulated "
                    "channel itself. The yield and dark-count tests are "
                    "therefore comparisons of a quantity with itself and cannot "
                    "fail; treat them as absent, not as passed."
                ),
            },
            "signer": self.evidence.signer,
            "key_id": self.evidence.key_id,
            "elapsed_seconds": _f(self.elapsed_seconds),
        }

    def summary(self) -> str:
        """One line, for a log or a terminal."""
        primary = self.attributions[0].label if self.attributions else "-"
        est = self.evidence.estimates["pooled_rate"]
        return (f"{self.verdict.upper():<12} rate={est.value:.4f} "
                f"({est.k}/{est.n})  flagged={len(self.flagged)}/"
                f"{self.n_applicable}  {primary}")


def _f(x: Any) -> Any:
    if x is None or isinstance(x, bool):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

class DetectionEngine:
    """Runs every detector against one evidence bundle and decides.

    The alarm decision is Bonferroni over the individual exact tests.  Fisher's
    combined p-value is computed and reported, but it does **not** drive the
    decision: Fisher's method assumes the p-values are independent, and these
    are computed from overlapping views of the same slot measurements.  Quoting
    it as the headline number would overstate the evidence by an unknown
    factor. It is here because it is a useful single summary when the reader
    already knows the caveat.
    """

    def __init__(self, rules: Optional[Sequence[Rule]] = None):
        self.rules: List[Rule] = list(rules) if rules is not None else default_rules()

    def run(self, evidence: Evidence) -> DetectionReport:
        t0 = time.perf_counter()
        outcomes = [rule.evaluate(evidence) for rule in self.rules]
        flagged = [o.name for o in outcomes if o.flagged]

        usable = [o.p_value for o in outcomes
                  if o.applicable and o.p_value is not None]
        combined = stats.fisher_combine(usable)[2] if usable else None

        verdict = self._verdict(outcomes, flagged, evidence)
        report = DetectionReport(
            verdict=verdict,
            alarm=bool(flagged) or evidence.protocol_rejects,
            outcomes=outcomes,
            attributions=attribute(flagged, evidence),
            evidence=evidence,
            combined_p_value=combined,
            elapsed_seconds=time.perf_counter() - t0,
        )
        return report

    @staticmethod
    def _verdict(outcomes: Sequence[TestOutcome], flagged: Sequence[str],
                 evidence: Evidence) -> str:
        """Three states, in the order the checks have to be applied.

        ``compromised``
            The protocol itself refused the signature -- a MAC or freshness
            failure, or a measured rate past the acceptance threshold -- or a
            detector fired whose firing means a *premise* of the threshold
            calculus is false, in which case a rate under the threshold is not
            evidence of anything.
        ``suspicious``
            A detector fired, but the protocol would still accept and the
            premises still hold.  Worth a look; not worth refusing a signature
            over.
        ``clean``
            Nothing fired and every recipient accepted.

        The ordering matters and is the whole of the fix for a bug this
        engine had: the protocol's rejection is checked *before* asking
        whether any test fired, because the two can disagree.  The eleven
        tests run at a Bonferroni-corrected alpha of about 9e-4 apiece, which
        deliberately costs sensitivity, so a run can sit just over the
        acceptance threshold with every test still passing.  Reporting that
        run as ``clean`` next to a rejected signature would be incoherent, and
        worse, it would train an operator to ignore the verdict.

        Note also what is *not* here: statistical significance alone does not
        make a run ``compromised``.  A detector can distinguish the rate from
        the floor long before the rate is large enough to matter, and
        conflating "measurably different" with "operationally unsafe" is how
        a monitoring system becomes noise.
        """
        if evidence.protocol_rejects:
            return "compromised"
        if not flagged:
            return "clean"
        rate = evidence.estimates["pooled_rate"]
        accept = evidence.ladder.get("accept")
        if rate.n > 0 and accept is not None and rate.value > float(accept):
            return "compromised"
        if set(flagged) & PREMISE_RULES:
            return "compromised"
        return "suspicious"


def analyse_session(session: Any, **kwargs: Any) -> DetectionReport:
    """Convenience: build evidence from a session and run the engine on it."""
    return DetectionEngine().run(Evidence.from_session(session, **kwargs))
