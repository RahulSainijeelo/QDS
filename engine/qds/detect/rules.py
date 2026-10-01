"""The individual detectors, and the evidence each one is entitled to see.

Each rule is a hypothesis test with a null drawn from the declared spec sheet,
not from the data it is judging.  A rule returns a :class:`TestOutcome`
carrying the statistic, an exact p-value where one exists, whether it fired,
and one sentence of plain English about what firing would mean.

Three conventions run through the file.

**Inapplicable is not a pass.**  A rule with no evidence -- no decoys, a
single recipient, zero counted checks -- returns ``applicable=False`` and a
``detail`` saying why.  The engine counts those separately and never lets them
inflate a clean verdict.  "We did not look" and "we looked and found nothing"
are different findings and only one of them is evidence.

**The null is per-quantity, not global.**  The honest error rate is not the
same in both bases, the honest yield is not the honest error rate, and the
honest correction histogram is uniform by proof rather than by measurement.
Each rule states which declared number it is testing against.

**A flag is a departure, not an accusation.**  Every ``interpretation``
string is written to survive being read out in an incident review.  A raised
error rate is consistent with an eavesdropper and equally consistent with a
misaligned waveplate; the engine measures the first fact and has no access to
the second.  Rules that pretend otherwise would be lying in a machine-readable
format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from . import stats
from .estimators import (correction_histogram, dark_estimate, decoy_rate,
                         fisher_exact_two_sided, homogeneity_chi_square,
                         pooled_rate, rate_difference, rates_by_basis,
                         rates_by_level, rates_by_position, rates_by_verifier,
                         yield_estimate)

if TYPE_CHECKING:                      # pragma: no cover
    from .engine import Evidence

__all__ = [
    "TestOutcome",
    "Rule",
    "PooledRateVsSpec",
    "PooledRateVsTransfer",
    "BasisConsistency",
    "PositionHomogeneity",
    "LevelHomogeneity",
    "YieldVsDeclared",
    "DarkExcess",
    "CorrectionUniformity",
    "DecoyVersusSignature",
    "RecipientAgreement",
    "SequentialOnset",
    "DEFAULT_RULES",
    "default_rules",
]


@dataclass
class TestOutcome:
    """The result of one detector.

    ``statistic`` is whatever the test's natural summary is -- a rate, a
    chi-square, a log-likelihood ratio -- and is meant for display.  The
    decision is made on ``p_value`` against the multiplicity-corrected alpha,
    or, for tests with no p-value, on an explicit rule recorded in ``detail``.
    """

    name: str
    title: str = ""
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    flagged: bool = False
    n: int = 0
    applicable: bool = True
    detail: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "statistic": _f(self.statistic),
            "p_value": _f(self.p_value),
            "threshold": _f(self.threshold),
            "flagged": bool(self.flagged),
            "n": int(self.n),
            "applicable": bool(self.applicable),
            "detail": _clean_tree(self.detail),
            "interpretation": self.interpretation,
        }


def _f(x: Any) -> Any:
    if x is None or isinstance(x, bool):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None


def _clean_tree(obj: Any) -> Any:
    """Recursively coerce a detail dict into JSON-native values."""
    if isinstance(obj, dict):
        return {str(k): _clean_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_tree(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    return _f(obj)


class Rule:
    """Base class.  A rule is stateless; all context arrives in ``evidence``."""

    name: str = ""
    title: str = ""

    def evaluate(self, evidence: "Evidence") -> TestOutcome:  # pragma: no cover
        raise NotImplementedError

    # -- helpers shared by subclasses ------------------------------------
    def _inapplicable(self, why: str, **detail: Any) -> TestOutcome:
        detail = dict(detail)
        detail["why_not"] = why
        return TestOutcome(name=self.name, title=self.title, applicable=False,
                           flagged=False, p_value=None, detail=detail,
                           interpretation=f"Not evaluated: {why}")

    def _decide(self, p: Optional[float], alpha: float) -> bool:
        return p is not None and p <= alpha


# --------------------------------------------------------------------------
# 1-2: the error rate against the two declared thresholds
# --------------------------------------------------------------------------

class PooledRateVsSpec(Rule):
    """Is the observed error rate consistent with the declared noise floor?

    The primary detector, and the one every attack on the quantum channel
    eventually trips.  An intercept-resend adversary measuring in a random
    basis pushes the per-check error toward 1/4; a partial or coherent attack
    pushes it part of the way.  Because the null is the *declared* floor
    rather than anything measured in this run, the p-value is a statement
    about the channel and not a restatement of the channel.
    """

    name = "pooled_rate_vs_spec"
    title = "Pooled error rate vs declared noise floor"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        est = evidence.estimates["pooled_rate"]
        if est.n <= 0:
            return self._inapplicable("no counted checks in any report")
        spec = evidence.ladder["spec_rate"] or 0.0
        p = stats.binom_test_greater(est.k, est.n, spec)
        alpha = evidence.alpha_per_test
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=est.value, p_value=p,
            threshold=evidence.ladder["accept"],
            flagged=self._decide(p, alpha), n=est.n,
            detail={
                "mismatches": est.k, "checks": est.n,
                "spec_rate": spec,
                "ci_low": est.ci_low, "ci_high": est.ci_high,
                "excess": est.value - spec,
                "alpha_per_test": alpha,
            },
            interpretation=(
                "The error rate on arrived qubits is higher than the declared "
                "hardware floor by more than sampling noise explains. That is "
                "what an eavesdropper on the quantum channel looks like, and "
                "also what a degraded link looks like; this test distinguishes "
                "neither from the other, only both from a healthy link."),
        )


class PooledRateVsTransfer(Rule):
    """Is the rate low enough for the signature to be *transferable*?

    Acceptance and transfer are different questions.  A recipient may be
    willing to act on a signature he verified himself at ``s_a`` while the
    same signature would not survive being forwarded, because a dishonest
    signer could have arranged for one recipient to see a cleaner record than
    the other.  The transfer threshold ``s_v`` sits strictly higher, and this
    test asks whether the observed rate has climbed past even that.
    """

    name = "pooled_rate_vs_transfer"
    title = "Pooled error rate vs transfer threshold"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        est = evidence.estimates["pooled_rate"]
        if est.n <= 0:
            return self._inapplicable("no counted checks in any report")
        s_v = evidence.ladder["transfer"] or 0.0
        p = stats.binom_test_greater(est.k, est.n, s_v)
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=est.value, p_value=p, threshold=s_v,
            flagged=self._decide(p, evidence.alpha_per_test), n=est.n,
            detail={"mismatches": est.k, "checks": est.n,
                    "transfer_threshold": s_v,
                    "forger_rate": evidence.ladder["forger_rate"]},
            interpretation=(
                "The error rate has passed the threshold that makes a "
                "signature safe to forward. Even if a recipient chooses to "
                "accept it, he cannot expect the next party to agree, so the "
                "non-repudiation guarantee no longer holds for this key."),
        )


# --------------------------------------------------------------------------
# 3: per-basis
# --------------------------------------------------------------------------

class BasisConsistency(Rule):
    """Does each basis match *its own* declared rate?

    The obvious test -- compare the Z error rate to the X error rate -- is
    wrong, and wrong in a way that would make the engine unusable.  The two
    honest rates are genuinely different.  Memory dephasing damages X
    eigenstates and leaves Z eigenstates untouched, and an X-basis check
    passes through six depolarising gate insertions against four for a Z-basis
    check.  On the lab-grade preset that asymmetry is large enough that a
    Z-equals-X test would fire on a perfectly healthy link.

    So each basis is tested against the rate the spec sheet predicts *for that
    basis*, with a Bonferroni-of-two correction inside the rule, and the
    reported statistic is the *excess* asymmetry -- how much further apart the
    two rates are than the declared design says they should be.

    This is the detector that catches a probe which is cheap in one basis and
    expensive in the other. ``BasisBiasedProbe(pz=0, px=0.12)`` produces a
    pooled rate of 6 percent, comfortably under a 10 percent threshold, while
    the X rate sits at 12 percent and is over it. Pooling hides the attack;
    splitting reveals it.
    """

    name = "basis_consistency"
    title = "Per-basis error rates vs their declared values"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        by_basis = evidence.estimates_by_basis
        z, x = by_basis["Z"], by_basis["X"]
        if z.n <= 0 or x.n <= 0:
            return self._inapplicable(
                "one basis has no counted checks, so there is nothing to "
                "compare against its declared rate",
                n_z=z.n, n_x=x.n)

        spec_z = evidence.spec_rate_z
        spec_x = evidence.spec_rate_x
        pooled_spec = evidence.ladder["spec_rate"] or 0.0
        fell_back = spec_z is None or spec_x is None
        if fell_back:
            spec_z = spec_x = pooled_spec

        p_z = stats.binom_test_greater(z.k, z.n, spec_z)
        p_x = stats.binom_test_greater(x.k, x.n, spec_x)
        # Bonferroni across the two bases, so the rule as a whole spends the
        # alpha it was allocated rather than twice it.
        alpha = evidence.alpha_per_test / 2.0
        p_min = min(p_z, p_x)

        declared_gap = (spec_x or 0.0) - (spec_z or 0.0)
        observed_gap = x.value - z.value
        excess = observed_gap - declared_gap

        note = ("per-basis spec rates supplied by the declared noise model"
                if not fell_back else
                "no per-basis spec available; both bases tested against the "
                "pooled floor, so this rule is only sensitive to gross "
                "asymmetry")

        return TestOutcome(
            name=self.name, title=self.title,
            statistic=excess, p_value=p_min, threshold=alpha,
            flagged=self._decide(p_min, alpha), n=z.n + x.n,
            detail={
                "z": {"k": z.k, "n": z.n, "rate": z.value,
                      "spec": spec_z, "p_value": _f(p_z)},
                "x": {"k": x.k, "n": x.n, "rate": x.value,
                      "spec": spec_x, "p_value": _f(p_x)},
                "declared_gap": declared_gap,
                "observed_gap": observed_gap,
                "excess_asymmetry": excess,
                "alpha_per_basis": alpha,
                "fell_back_to_pooled_spec": fell_back,
                "note": note,
            },
            interpretation=(
                "At least one measurement basis is noisier than the hardware "
                "specification predicts for that basis. A probe that disturbs "
                "one basis more than the other produces exactly this pattern "
                "and can hide entirely inside a pooled error rate."),
        )


# --------------------------------------------------------------------------
# 4-5: is the rate the same everywhere it should be?
# --------------------------------------------------------------------------

class _Homogeneity(Rule):
    """Shared machinery: are these groups drawing from one rate?"""

    group_label = "group"

    def _groups(self, evidence: "Evidence") -> Dict[Any, Any]:  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        groups = self._groups(evidence)
        live = {g: e for g, e in groups.items() if e.n > 0}
        if len(live) < 2:
            return self._inapplicable(
                f"fewer than two {self.group_label}s carry data, so there is "
                f"no variation to test",
                n_groups=len(live))
        stat, dof, p = homogeneity_chi_square(
            [(e.k, e.n) for e in live.values()])
        if p is None:
            return self._inapplicable(
                "the pooled rate is 0 or 1, so the table is degenerate and "
                "the groups agree trivially",
                n_groups=len(live))
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=stat, p_value=p, threshold=evidence.alpha_per_test,
            flagged=self._decide(p, evidence.alpha_per_test),
            n=sum(e.n for e in live.values()),
            detail={
                "dof": dof,
                "groups": {str(g): {"k": e.k, "n": e.n, "rate": e.value}
                           for g, e in sorted(live.items(), key=lambda kv: str(kv[0]))},
            },
            interpretation=self.interpretation_text,
        )

    interpretation_text = ""


class PositionHomogeneity(_Homogeneity):
    """Is the error rate the same across message-bit positions?

    Every position's slots traverse the same channel at the same time, so
    under any honest noise they share one rate.  An adversary with a limited
    disturbance budget does not spend it uniformly -- he spends it on the bits
    he needs to change -- and a forger who rewrote one bit leaves mismatches
    concentrated in that block.  Both show up here as inhomogeneity while the
    pooled rate may still look ordinary.
    """

    name = "position_homogeneity"
    title = "Error rate homogeneity across message-bit positions"
    group_label = "position"
    interpretation_text = (
        "The error rate is not the same across message-bit positions. "
        "Honest noise has no reason to prefer one bit of the message over "
        "another, so concentration points at tampering aimed at specific "
        "bits rather than at the channel as a whole.")

    def _groups(self, evidence: "Evidence") -> Dict[Any, Any]:
        return dict(evidence.estimates_by_position)


class LevelHomogeneity(_Homogeneity):
    """Is the error rate the same at the accept and transfer levels?

    The two levels apply different thresholds to the same physical link, so
    the underlying rate should be one number.  A difference means the rate
    moved between the two measurements -- which is what a dishonest signer
    arranging for one party to see a cleaner record than another would
    produce.
    """

    name = "level_homogeneity"
    title = "Error rate homogeneity across verification levels"
    group_label = "level"
    interpretation_text = (
        "The error rate differs between verification levels. The same key "
        "measured under two policies should give one rate; a split suggests "
        "the record one party sees is not the record another sees.")

    def _groups(self, evidence: "Evidence") -> Dict[Any, Any]:
        return dict(evidence.estimates_by_level)


# --------------------------------------------------------------------------
# 6-7: the detectors, whose denominator includes the rounds that never arrived
# --------------------------------------------------------------------------

class YieldVsDeclared(Rule):
    """Do as many slots register as the declared loss budget says they should?

    The only detector whose denominator counts rounds that never arrived, and
    therefore the only one that can see an adversary who *removes* evidence
    rather than corrupting it.  An intercept-and-block attacker measures a
    qubit, learns something, and then simply drops the round instead of
    resending a disturbed copy.  Every error rate in this engine is
    conditioned on arrival, so all of them stay clean; the yield falls.

    The test is two-sided on purpose.  A yield below the declared value is
    blocking.  A yield *above* it is just as suspicious: detector blinding and
    faked-state attacks work by injecting clicks the channel did not deliver,
    and a link that reports more detections than its own optics allow is
    reporting something other than photons.
    """

    name = "yield_vs_declared"
    title = "Detector yield vs declared loss budget"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        est = evidence.estimates["detector_yield"]
        declared = evidence.declared_yield
        if est.n <= 0:
            return self._inapplicable("no addressed slots were recorded")
        if declared is None:
            return self._inapplicable(
                "no declared loss budget was supplied, so there is nothing "
                "to compare the observed yield against")
        p = stats.binom_test_two_sided(est.k, est.n, declared)
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=est.value, p_value=p, threshold=declared,
            flagged=self._decide(p, evidence.alpha_per_test), n=est.n,
            detail={"clicks": est.k, "addressed": est.n,
                    "declared_yield": declared,
                    "ci_low": est.ci_low, "ci_high": est.ci_high,
                    "direction": ("below" if est.value < declared else "above")},
            interpretation=(
                "The fraction of slots that registered a detection does not "
                "match the declared loss budget. Fewer detections than "
                "expected is consistent with an adversary who measures and "
                "then blocks, which leaves every error rate untouched; more "
                "detections than expected is consistent with injected or "
                "faked clicks."),
        )


class DarkExcess(Rule):
    """Are more outcomes coming from dark counts than the detectors declare?

    A dark count is a click with no photon behind it, so it still produces a
    bit and that bit is uniform.  Excess darks therefore raise the error rate
    as well as showing up here, and separating the two lets the engine say
    "the detectors are noisy" rather than "the channel is compromised" when
    that is the truthful reading.
    """

    name = "dark_excess"
    title = "Dark-count fraction vs declared detector noise"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        est = evidence.estimates["dark_fraction"]
        declared = evidence.declared_dark_fraction
        if est.n <= 0:
            return self._inapplicable("no addressed slots were recorded")
        if declared is None:
            return self._inapplicable("no declared dark-count rate supplied")
        p = stats.binom_test_greater(est.k, est.n, declared)
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=est.value, p_value=p, threshold=declared,
            flagged=self._decide(p, evidence.alpha_per_test), n=est.n,
            detail={"dark": est.k, "addressed": est.n,
                    "declared_dark_fraction": declared,
                    "ci_low": est.ci_low, "ci_high": est.ci_high},
            interpretation=(
                "More outcomes are attributable to dark counts than the "
                "declared detector noise allows. Since a dark count "
                "contributes a uniformly random bit, this also inflates the "
                "measured error rate, and the two should be read together."),
        )


# --------------------------------------------------------------------------
# 8: the classical transcript
# --------------------------------------------------------------------------

class CorrectionUniformity(Rule):
    """Is the announced Pauli-correction transcript uniform on four outcomes?

    When the signer teleports a key qubit he performs a Bell measurement and
    broadcasts the pair ``(u, v)`` so the recipient can undo ``Z^u X^v``.
    That pair is uniform on ``{0,1}**2`` under any honest noise whatsoever,
    and the uniformity is not an empirical observation -- it is the statement
    that the announcement carries zero information about the key, proved in
    :func:`qds.protocol.distribution.teleportation_leakage`.

    So a biased histogram does not merely look unusual; it contradicts a
    proof.  The only ways to produce one are to interfere with the Bell
    measurement or to tamper with the classical channel carrying the result.
    This detector is also the only one that catches tampering which happens to
    be error-neutral, because it never looks at an error rate at all.
    """

    name = "correction_uniformity"
    title = "Uniformity of announced Pauli corrections"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        counts = evidence.correction_counts
        total = sum(counts)
        if total < 8:
            return self._inapplicable(
                "fewer than eight announcements recorded; a four-cell "
                "chi-square on that few counts has no power",
                total=total)
        expected = [total / 4.0] * 4
        stat, dof, p = stats.chi2_test(counts, expected, dof_reduction=1)
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=stat, p_value=p, threshold=evidence.alpha_per_test,
            flagged=self._decide(p, evidence.alpha_per_test), n=total,
            detail={"counts": list(counts), "expected": expected, "dof": dof,
                    "labels": ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]},
            interpretation=(
                "The Bell-measurement announcements are not uniform over the "
                "four outcomes. Uniformity is what makes the classical "
                "transcript information-free, so a bias means the Bell "
                "measurement or the announcement channel is being interfered "
                "with -- and unlike the error-rate tests, this one fires even "
                "when the interference costs no fidelity."),
        )


# --------------------------------------------------------------------------
# 9-10: comparing populations that should agree
# --------------------------------------------------------------------------

class DecoyVersusSignature(Rule):
    """Do the decoy slots and the signature slots see the same channel?

    Decoys are indistinguishable from signature slots in transit and are
    revealed only afterwards, so an adversary cannot tell which is which at
    the time he has to decide whether to attack.  If he attacks everything,
    the pooled-rate test catches him.  If he attacks only the slots that
    matter, the pooled rate over signature slots rises while the decoys stay
    clean, and this comparison catches him instead.

    The honest limitation, stated because it affects how the result should be
    read: :class:`~qds.protocol.session.Calibration` records only the decoy
    mismatch and total counts, with no basis breakdown.  Since the honest
    error rate differs by basis, the comparison implicitly assumes the two
    populations drew their bases from the same distribution -- true by
    construction in this implementation, but not something this rule verifies.
    """

    name = "decoy_vs_signature"
    title = "Decoy slots vs signature slots"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        decoy = evidence.estimates["decoy_rate"]
        sig = evidence.estimates["pooled_rate"]
        if decoy.n <= 0:
            return self._inapplicable(
                "no decoy slots were measured, so there is no independent "
                "sample of the same channel to compare against")
        if sig.n <= 0:
            return self._inapplicable("no counted signature checks")
        diff = rate_difference("decoy_vs_signature", sig, decoy)
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=diff.difference, p_value=diff.p_value,
            threshold=evidence.alpha_per_test,
            flagged=self._decide(diff.p_value, evidence.alpha_per_test),
            n=decoy.n + sig.n,
            detail={"signature": {"k": sig.k, "n": sig.n, "rate": sig.value},
                    "decoy": {"k": decoy.k, "n": decoy.n, "rate": decoy.value},
                    "difference": diff.difference, "z": _f(diff.z),
                    "test": "fisher_exact_two_sided"},
            interpretation=(
                "Signature slots and decoy slots disagree about how noisy the "
                "channel is, even though both crossed it under identical "
                "conditions. That pattern is what selective attack on the "
                "slots that carry meaning looks like."),
        )


class RecipientAgreement(Rule):
    """Do the recipients agree about the channel?

    Bob and Charlie hold independently symmetrised halves of the same public
    key and verify the same declaration.  Their measured rates should be two
    samples of one number.  Disagreement is the observable signature of
    repudiation -- a signer who arranges for one recipient to be able to
    verify while the other cannot -- and of an attack aimed at one link rather
    than at the signer.

    All pairs are tested, with a Bonferroni correction inside the rule so that
    adding a third recipient does not quietly triple the rule's false-alarm
    rate.
    """

    name = "recipient_agreement"
    title = "Agreement between recipients"

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        by_verifier = {k: v for k, v in evidence.estimates_by_verifier.items()
                       if v.n > 0}
        names = sorted(by_verifier)
        if len(names) < 2:
            return self._inapplicable(
                "fewer than two recipients produced counted checks, so there "
                "is no second opinion to compare against",
                recipients=names)
        pairs: List[Dict[str, Any]] = []
        worst: Optional[float] = None
        worst_gap = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = by_verifier[names[i]], by_verifier[names[j]]
                p = fisher_exact_two_sided(a.k, a.n, b.k, b.n)
                gap = a.value - b.value
                pairs.append({"left": names[i], "right": names[j],
                              "left_rate": a.value, "right_rate": b.value,
                              "difference": gap, "p_value": _f(p)})
                if worst is None or p < worst:
                    worst = p
                    worst_gap = gap
        alpha = evidence.alpha_per_test / max(1, len(pairs))
        return TestOutcome(
            name=self.name, title=self.title,
            statistic=worst_gap, p_value=worst, threshold=alpha,
            flagged=self._decide(worst, alpha),
            n=sum(v.n for v in by_verifier.values()),
            detail={"pairs": pairs, "alpha_per_pair": alpha,
                    "n_pairs": len(pairs)},
            interpretation=(
                "Two recipients of the same signature measured materially "
                "different error rates. Because they hold halves of one key "
                "and check one declaration, a genuine split means either one "
                "link is under attack or the signer distributed unequal "
                "keys -- the latter being the setup for repudiation."),
        )


# --------------------------------------------------------------------------
# 11: watching the run unfold rather than waiting for it to end
# --------------------------------------------------------------------------

class SequentialOnset(Rule):
    """Would a sequential monitor have stopped the run early?

    The batch tests above look at a session once it is over.  A sequential
    monitor looks at it as it happens, which matters because every check costs
    an entangled pair: stopping at check 40 instead of check 600 saves 560
    pairs and gets the operator to the problem sooner.

    Two monitors run side by side because they answer different questions.
    Wald's SPRT accumulates the log-likelihood ratio between "honest" and
    "forging" and stops the moment the evidence is decisive; among all tests
    with the same error rates it minimises the expected number of
    observations, so it is the fastest honest alarm available.  The CUSUM
    tracks slow drift instead, and catches an adversary who never pushes any
    one session over a threshold but leans on the rate a little, continuously.

    The SPRT needs ``0 < p0 < p1 < 1``, and a declared floor of exactly zero
    -- the ideal preset -- violates that. In that case ``p0`` is clamped to a
    small positive value and the substitution is recorded in ``detail``,
    because a monitor that silently redefines its own null hypothesis is worse
    than one that refuses to run.
    """

    name = "sequential_onset"
    title = "Sequential monitors (SPRT and CUSUM)"

    #: Smallest usable null rate for the SPRT. At a floor of exactly zero the
    #: likelihood ratio is infinite on the first error, which is true but
    #: useless; 1e-3 keeps the monitor finite and is far below any threshold
    #: the protocol would accept.
    MIN_NULL_RATE = 1e-3

    def evaluate(self, evidence: "Evidence") -> TestOutcome:
        sequence = evidence.ordered_mismatches
        if len(sequence) < 16:
            return self._inapplicable(
                "fewer than sixteen ordered checks; a sequential monitor "
                "needs a run to watch",
                n=len(sequence))

        spec = float(evidence.ladder["spec_rate"] or 0.0)
        forger = float(evidence.ladder["forger_rate"] or 0.25)
        clamped = spec < self.MIN_NULL_RATE
        p0 = max(spec, self.MIN_NULL_RATE)
        p1 = float(evidence.ladder["transfer"] or forger)
        if not p0 < p1:
            p1 = min(forger, max(p0 * 2.0, p0 + 1e-3))
        if not p0 < p1 < 1.0:
            return self._inapplicable(
                "the declared floor and the forger rate do not bracket a "
                "detectable shift", p0=p0, p1=p1)

        sprt = stats.SPRT(p0=p0, p1=p1, alpha=1e-3, beta=1e-3)
        state = sprt.run(bool(e) for e in sequence)

        # Raw per-check outcomes, not a running average: the CUSUM increment
        # is a per-observation log-likelihood ratio and its calibration
        # assumes the increments are independent.  Feeding it a cumulative
        # rate makes them strongly autocorrelated, and the monitor then drifts
        # up on honest data -- that mistake cost this rule a 6% false-alarm
        # rate against a designed 0.1%.
        cusum = stats.CUSUM.for_window(p0=p0, p1=p1, n=len(sequence),
                                       alpha=1e-3)
        cstate = cusum.run(bool(e) for e in sequence)

        fired = state.decision == "accept_h1" or cstate.alarm_at is not None
        stop_at = state.n if state.decision == "accept_h1" else cstate.alarm_at

        return TestOutcome(
            name=self.name, title=self.title,
            statistic=state.log_lr, p_value=None,
            threshold=sprt.upper, flagged=fired, n=len(sequence),
            detail={
                "sprt_decision": state.decision,
                "sprt_log_lr": _f(state.log_lr),
                "sprt_upper": _f(sprt.upper),
                "sprt_lower": _f(sprt.lower),
                "sprt_stopped_at": state.n,
                "cusum_alarm_at": cstate.alarm_at,
                "cusum_final": _f(cstate.s),
                "cusum_threshold": _f(cusum.threshold_h),
                "stopped_at": stop_at,
                "p0": p0, "p1": p1,
                "null_rate_clamped": clamped,
                "expected_samples_under_attack": _f(sprt.expected_samples_h1()),
                "decision_rule": ("flagged when the SPRT crosses its upper "
                                  "boundary or the CUSUM alarms; there is no "
                                  "p-value because the stopping rule, not a "
                                  "fixed sample, controls the error rates. "
                                  "both boundaries are derived from a "
                                  "declared alpha of 1e-3 -- Wald's for the "
                                  "SPRT, ln(n/alpha) for the CUSUM -- so the "
                                  "combined false-alarm rate over a run is "
                                  "bounded by 2e-3"),
            },
            interpretation=(
                "A sequential monitor would have stopped this run before it "
                "finished. The stopping index says how early the evidence "
                "became decisive, which is the operationally useful number: "
                "every check after it spent an entangled pair on a link "
                "already known to be bad."),
        )


# --------------------------------------------------------------------------

DEFAULT_RULES: Tuple[type, ...] = (
    PooledRateVsSpec,
    PooledRateVsTransfer,
    BasisConsistency,
    PositionHomogeneity,
    LevelHomogeneity,
    YieldVsDeclared,
    DarkExcess,
    CorrectionUniformity,
    DecoyVersusSignature,
    RecipientAgreement,
    SequentialOnset,
)


def default_rules() -> List[Rule]:
    """A fresh instance of every detector, in report order."""
    return [cls() for cls in DEFAULT_RULES]
