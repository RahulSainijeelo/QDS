"""One end-to-end run: enrol, distribute, symmetrise, calibrate, sign, verify.

This module is the assembly, not new physics.  Its job is to run the phases in
the right order and to make the two steps that are easy to skip -- and fatal
to skip -- impossible to forget.

Phase order
-----------
1. **Enrol.**  Each recipient learns which signer and which ``key_id`` to
   expect, over a channel authenticated by the shared key reservoir.  This is
   what turns "a signature from Alice" into a claim that can be false.
2. **Distribute.**  Bell pairs and teleported BB84 states, once per recipient,
   with independent randomness.  See :mod:`qds.protocol.distribution`.
3. **Symmetrise.**  Recipients privately permute their stored qubits among
   themselves.  See :func:`symmetrise_keys` -- without this step the scheme
   has no non-repudiation claim at all.
4. **Calibrate.**  Alice reveals the decoy entries; the recipients measure
   them and test the channel against its declared noise floor.  The thresholds
   come from that declared floor, fixed before the run; the decoys say whether
   this channel is living up to it.
5. **Sign and verify.**  One message, one key, ever.

Nothing here consults :class:`~qds.protocol.distribution.DistributionOracle`.
The oracle knows Eve's actual probe states and the exact fidelity of every
delivered qubit; a recipient knows neither.  Keeping the session blind to it
is what makes the reported acceptance and detection rates measurements rather
than restatements of the input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..channel import NoiseModel
from ..detect import stats
from .auth import FreshnessLedger, KeyReservoir, mac_tag, mac_verify
from .distribution import (DistributionConfig, DistributionResult,
                           distribute_public_key)
from .keys import (HeldQubit, KeyEntry, PrivateKey, PublicKeyStore,
                   SignerRegistry, Slot, SlotRole, generate_private_key)
from .signing import SignatureDeclaration, message_to_bits, sign_message
from .verification import (VerificationPolicy, VerificationReport,
                           measure_slot, verify_declaration)

__all__ = [
    "SessionConfig", "Calibration", "QDSSession", "SessionResult",
    "symmetrise_keys", "calibrate_from_decoys",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionConfig:
    """Everything that defines one protocol run."""

    signer: str = "alice"
    verifiers: Tuple[str, ...] = ("bob", "charlie")
    #: message length in bits; the key is ``2 * message_bits * L`` slots
    message_bits: int = 16
    #: repetitions per message bit -- this is the statistical sample size
    L: int = 48
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    symmetrise: bool = True
    #: Declared hardware noise floor, in per-check error rate.  The acceptance
    #: and transfer thresholds are derived from **this** number, fixed before
    #: any data is taken.  ``None`` means "ask the noise model for its own
    #: specification", which is what a deployment does when it reads the
    #: figure off the calibration certificate that shipped with the link.
    #:
    #: It is deliberately *not* estimated from this session's decoys.  A
    #: threshold chosen after seeing the data is a threshold the adversary has
    #: had a chance to influence, and at realistic decoy counts the estimate is
    #: far too wide to place one anyway -- see :func:`calibrate_from_decoys`.
    noise_floor_spec: Optional[float] = None
    #: confidence level for the decoy conformance test
    calibration_alpha: float = 0.01
    #: failure probability allowed for the decoy-to-signature extrapolation
    #: (reported as a pre-signing assurance figure; does not set thresholds)
    sampling_epsilon: float = 1e-6
    #: "balanced" maximises the worst security exponent; "thirds" splits the
    #: usable gap evenly and is easier to explain in a figure
    threshold_rule: str = "balanced"
    reservoir_size: int = 4096
    window_seconds: float = 300.0
    min_checks: int = 16

    @property
    def signature_slots(self) -> int:
        return 2 * self.message_bits * self.L

    def spec_rate(self) -> float:
        """The error rate the thresholds are built on."""
        if self.noise_floor_spec is not None:
            return float(self.noise_floor_spec)
        return float(self.distribution.noise.predicted_qber())

    def to_dict(self) -> Dict[str, object]:
        return {
            "signer": self.signer,
            "verifiers": list(self.verifiers),
            "message_bits": self.message_bits,
            "L": self.L,
            "signature_slots": self.signature_slots,
            "checks_per_signature": self.message_bits * self.L,
            "symmetrise": self.symmetrise,
            "noise_floor_spec": self.spec_rate(),
            "calibration_alpha": self.calibration_alpha,
            "sampling_epsilon": self.sampling_epsilon,
            "threshold_rule": self.threshold_rule,
            "distribution": self.distribution.to_dict(),
        }


# --------------------------------------------------------------------------
# symmetrisation
# --------------------------------------------------------------------------

def symmetrise_keys(stores: Sequence[PublicKeyStore],
                    rng: np.random.Generator) -> Dict[str, object]:
    """Privately permute each signature slot's qubits among the recipients.

    **Why this exists.**  The repudiation bound in
    :mod:`qds.protocol.verification` compares one recipient's mismatch count
    with another's and argues that both are samples from the same rate.  A
    dishonest *signer* breaks that assumption for free: send Bob clean states
    and Charlie dirty ones, and the message Bob accepts is one Charlie will
    reject -- repudiation with probability near 1, no statistics required.

    The countermeasure is physical.  Before any message exists, the
    recipients exchange a random selection of their stored qubits over a
    channel Alice cannot observe.  Every slot's copies are then held by a
    uniformly random recipient, so Alice's choice of who to cheat is
    decorrelated from who will check.  Any asymmetry she builds in is
    redistributed evenly, and the equal-rate premise becomes true rather than
    assumed.

    Three details that matter:

    * Only **signature** slots move.  Check pairs and decoys are sacrificed
      publicly, so permuting them would tell Alice nothing and gain nothing.
    * A qubit carries its own ``arrived``/``dark``/``correction`` flags, so
      those travel with it.  The per-store detector log does **not** move: it
      records what happened at that recipient's own detector, and that is
      still true afterwards.
    * The permutation must be secret from Alice but need not be secret from
      the recipients.  It is drawn here from the session RNG, and the attack
      suite can disable the whole step to demonstrate what it buys.
    """
    live = [s for s in stores if s is not None]
    if len(live) < 2:
        return {"performed": False, "reason": "fewer than two recipients",
                "slots_permuted": 0, "moved": 0}

    common = set(live[0].qubits)
    for s in live[1:]:
        common &= set(s.qubits)
    slots = sorted(s for s in common if s.j >= 0)

    moved = 0
    n = len(live)
    for slot in slots:
        perm = rng.permutation(n)
        held = [s.qubits[slot] for s in live]
        for dest, src in enumerate(perm):
            live[dest].qubits[slot] = held[int(src)]
            if int(src) != dest:
                moved += 1
    return {"performed": True, "recipients": [s.verifier for s in live],
            "slots_permuted": len(slots), "moved": moved,
            "fraction_moved": moved / (len(slots) * n) if slots else 0.0}


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """What the decoys say about the channel, and whether it is within spec.

    This is a **conformance test**, not a threshold-setting step.  The
    distinction is the correction of a real design error, so it is worth
    stating plainly.

    An earlier version estimated the error rate from the decoys and placed the
    acceptance threshold above that estimate.  Two things are wrong with that.
    First, a threshold fixed after seeing the data is one the adversary has had
    an opportunity to move; thresholds belong to the protocol, not to the run.
    Second, it does not work numerically: the decoy sample is small, and the
    Hoeffding-Serfling margin for extrapolating from ``n`` decoys is at least
    ``sqrt(ln(2/eps) / 2n)``, which at ``n = 96`` and ``eps = 1e-6`` is 0.275 --
    wider than the entire interval ``[0, 1/4]`` the threshold has to live in.
    The estimate was not merely imprecise, it was vacuous, and adding it to the
    threshold made every security exponent collapse to zero.

    So the decoys now answer the question they can actually answer: *is this
    channel performing at or below its certified noise floor?*  A failure is a
    detection event, which is what the threat-detection engine consumes.
    """

    verifier: str
    decoys_measured: int
    decoy_mismatches: int
    point_estimate: float
    #: two-sided Clopper-Pearson interval at ``alpha``; exact, no normal
    #: approximation, so it is valid at single-digit mismatch counts
    lower_confidence: float
    upper_confidence: float
    #: the declared noise floor the thresholds were built from
    spec_rate: float
    #: one-sided p-value for "the true rate is at or below spec".  Small means
    #: the channel is provably degraded, whether by hardware or by Eve.
    p_value_vs_spec: float
    #: True when the data are inconsistent with spec at level ``alpha``
    spec_violation: bool
    #: Hoeffding-Serfling half-width for extrapolating the decoy rate to the
    #: unmeasured signature slots.  Reported as a *pre-signing assurance*
    #: figure -- how well a recipient knows his key before a message exists --
    #: and it does not gate anything.  The accept/reject decision is taken on
    #: the signature checks themselves, which are measured directly, so there
    #: is no unsampled population to extrapolate to at that point.
    sampling_margin: float
    #: ``upper_confidence + sampling_margin``: the rate a recipient can certify
    #: for his whole key before signing.  Below 1/4 means the decoy budget is
    #: large enough to be informative on its own.
    certified_rate: float
    alpha: float
    sampling_epsilon: float
    #: the analytic prediction, for comparison in the dashboard only
    predicted_rate: Optional[float] = None

    @property
    def informative(self) -> bool:
        """Whether the decoy budget certifies anything useful pre-signing."""
        return self.certified_rate < 0.25

    def to_dict(self) -> Dict[str, object]:
        return {
            "verifier": self.verifier,
            "decoys_measured": self.decoys_measured,
            "decoy_mismatches": self.decoy_mismatches,
            "point_estimate": self.point_estimate,
            "lower_confidence": self.lower_confidence,
            "upper_confidence": self.upper_confidence,
            "spec_rate": self.spec_rate,
            "p_value_vs_spec": self.p_value_vs_spec,
            "spec_violation": self.spec_violation,
            "sampling_margin": self.sampling_margin,
            "certified_rate": self.certified_rate,
            "informative": self.informative,
            "alpha": self.alpha,
            "sampling_epsilon": self.sampling_epsilon,
            "predicted_rate": self.predicted_rate,
        }


def calibrate_from_decoys(store: PublicKeyStore,
                          decoy_key: Mapping[Slot, KeyEntry],
                          rng: np.random.Generator,
                          spec_rate: float,
                          noise: Optional[NoiseModel] = None,
                          alpha: float = 0.01,
                          sampling_epsilon: float = 1e-6,
                          n_signature_slots: Optional[int] = None,
                          predicted_rate: Optional[float] = None) -> Calibration:
    """Measure the decoys and test them against the declared noise floor.

    Decoy slots are ordinary transmissions whose key entries Alice reveals
    immediately, so a recipient can measure them without spending signature
    capacity.  Because Eve cannot tell a decoy from a signature slot while it
    is in flight -- the roles are only announced afterwards -- anything she
    does to the key shows up in the decoys at the same rate.

    Three numbers come out, each answering a different question:

    1. **Point estimate** ``k/n``: what the sample saw.
    2. **Clopper-Pearson interval** at ``alpha``: where the true rate lies,
       exactly, by inverting the binomial test.  Exact inversion matters here
       because a good channel produces single-digit mismatch counts, where a
       normal approximation is simply wrong.
    3. **One-sided p-value against ``spec_rate``**: the probability that a
       conforming channel would produce this many mismatches or more.  This is
       the conformance test, and a small value is a detection event.

    The Hoeffding-Serfling margin is computed as well, but as a reported
    diagnostic rather than a gate -- see :class:`Calibration`.
    """
    checked = mismatches = 0
    for slot, entry in decoy_key.items():
        held = store.qubits.get(slot)
        if held is None or held.consumed:
            continue
        bit, status = measure_slot(held, entry.basis, rng, noise)
        if bit is None:
            continue
        checked += 1
        mismatches += int(bit != entry.bit)

    point = mismatches / checked if checked else 0.0
    lcl, ucl = stats.clopper_pearson(mismatches, checked, alpha) if checked \
        else (0.0, 1.0)
    p_spec = stats.binom_test_greater(mismatches, checked, spec_rate) if checked \
        else 1.0
    rest = int(n_signature_slots if n_signature_slots is not None
               else max(len(store.qubits) - checked, 1))
    margin = stats.sampling_deviation_margin(checked, rest, sampling_epsilon) \
        if checked else 0.5
    return Calibration(verifier=store.verifier, decoys_measured=checked,
                       decoy_mismatches=mismatches, point_estimate=point,
                       lower_confidence=lcl, upper_confidence=ucl,
                       spec_rate=float(spec_rate), p_value_vs_spec=p_spec,
                       spec_violation=bool(checked and p_spec < alpha),
                       sampling_margin=margin,
                       certified_rate=min(max(ucl + margin, 0.0), 1.0),
                       alpha=alpha, sampling_epsilon=sampling_epsilon,
                       predicted_rate=predicted_rate)


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

@dataclass
class SessionResult:
    """Everything one run produced, in JSON-ready form."""

    config: SessionConfig
    private_key_summary: Dict[str, object]
    distribution: Dict[str, Dict[str, object]]
    symmetrisation: Dict[str, object]
    calibration: Dict[str, Dict[str, object]]
    policy: Dict[str, object]
    declaration: Optional[Dict[str, object]]
    verifications: Dict[str, Dict[str, object]]
    transfers: Dict[str, Dict[str, object]]
    timings: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "private_key": self.private_key_summary,
            "distribution": self.distribution,
            "symmetrisation": self.symmetrisation,
            "calibration": self.calibration,
            "policy": self.policy,
            "declaration": self.declaration,
            "verifications": self.verifications,
            "transfers": self.transfers,
            "timings": self.timings,
        }


class QDSSession:
    """A signer, a set of recipients, and one key's worth of protocol.

    The object is stateful on purpose: a public key is consumable, so the
    phases must happen once, in order.  Calling them out of order raises
    rather than quietly producing a meaningless number.
    """

    def __init__(self, config: Optional[SessionConfig] = None,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None):
        self.config = config or SessionConfig()
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.timings: Dict[str, float] = {}

        self.private_key: Optional[PrivateKey] = None
        self.results: Dict[str, DistributionResult] = {}
        self.stores: Dict[str, PublicKeyStore] = {}
        self.registries: Dict[str, SignerRegistry] = {}
        self.ledgers: Dict[str, FreshnessLedger] = {}
        self.reservoirs: Dict[str, KeyReservoir] = {}
        self.calibrations: Dict[str, Calibration] = {}
        self.policy: Optional[VerificationPolicy] = None
        self.symmetrisation: Dict[str, object] = {"performed": False}
        self.declaration: Optional[SignatureDeclaration] = None
        #: every report this session produced, keyed ``"name:level"``.  The
        #: detection engine and the dashboard both want the objects, not the
        #: dictionaries, and re-verifying to get them back is impossible --
        #: verification consumes the qubits it measures.
        self.reports: Dict[str, VerificationReport] = {}
        self._counter = 0

    # -- phase 1 ----------------------------------------------------------
    def enrol(self) -> None:
        """Establish key reservoirs and register the signer with each party.

        The enrolment message is itself MAC'd, and the index of the key that
        authenticated it is stored in the registry.  That gives the attack
        suite something real to attack: an impersonator who cannot produce a
        valid enrolment tag simply never gets a registry entry, and every
        later signature he sends is rejected on identity grounds before a
        single qubit is measured.
        """
        t0 = time.perf_counter()
        cfg = self.config
        self.private_key = generate_private_key(
            signer=cfg.signer, message_bits=cfg.message_bits, L=cfg.L,
            rng=self.rng)
        for name in cfg.verifiers:
            res = KeyReservoir(party_a=cfg.signer, party_b=name,
                               size=cfg.reservoir_size,
                               seed=int(self.rng.integers(0, 2 ** 31 - 1)))
            self.reservoirs[name] = res
            registry = SignerRegistry(verifier=name)
            key = res.take()
            payload = (b"TQDS-ENROL-v1|" + cfg.signer.encode()
                       + b"|" + self.private_key.key_id.encode()
                       + b"|" + name.encode())
            tag = mac_tag(key, payload)
            if not mac_verify(key, payload, tag):        # pragma: no cover
                raise RuntimeError("enrolment MAC failed to self-verify")
            registry.enrol(cfg.signer, self.private_key.key_id,
                           at=time.time(), mac_index=key.index)
            self.registries[name] = registry
            self.ledgers[name] = FreshnessLedger(
                verifier=name, window_seconds=cfg.window_seconds)
        self.timings["enrol"] = time.perf_counter() - t0

    # -- phase 2 ----------------------------------------------------------
    def distribute(self) -> None:
        """Deliver one quantum public key per recipient."""
        if self.private_key is None:
            raise RuntimeError("call enrol() before distribute()")
        t0 = time.perf_counter()
        for name in self.config.verifiers:
            res = distribute_public_key(self.private_key, name,
                                        self.config.distribution, self.rng)
            self.results[name] = res
            self.stores[name] = res.store
        self.timings["distribute"] = time.perf_counter() - t0

    # -- phase 3 ----------------------------------------------------------
    def symmetrise(self) -> None:
        if not self.stores:
            raise RuntimeError("call distribute() before symmetrise()")
        t0 = time.perf_counter()
        if self.config.symmetrise:
            self.symmetrisation = symmetrise_keys(
                [self.stores[n] for n in self.config.verifiers], self.rng)
        else:
            self.symmetrisation = {
                "performed": False,
                "reason": "disabled by configuration; the non-repudiation "
                          "bound does not hold for this run",
                "slots_permuted": 0, "moved": 0}
        self.timings["symmetrise"] = time.perf_counter() - t0

    # -- phase 4 ----------------------------------------------------------
    def calibrate(self) -> VerificationPolicy:
        """Build the thresholds from spec, then test the channel against them.

        Order matters and is the opposite of what it was.  The policy is
        derived from :meth:`SessionConfig.spec_rate` -- a declared figure that
        exists before the run -- and the decoys are then measured to check
        whether this particular channel is living up to it.  A threshold
        derived from the run's own data would be a threshold the adversary
        could push around, and at realistic decoy counts it would be too wide
        to place at all.

        Every recipient calibrates independently.  The policy is shared, which
        is what makes their mismatch counts directly comparable -- the
        repudiation argument compares two counts against the same pair of
        thresholds, so they cannot be per-recipient.
        """
        if not self.stores:
            raise RuntimeError("call distribute() before calibrate()")
        t0 = time.perf_counter()
        cfg = self.config
        noise = cfg.distribution.noise
        spec = cfg.spec_rate()
        predicted = noise.predicted_qber()
        for name in cfg.verifiers:
            res = self.results[name]
            self.calibrations[name] = calibrate_from_decoys(
                store=self.stores[name], decoy_key=res.decoy_key,
                rng=self.rng, spec_rate=spec, noise=noise,
                alpha=cfg.calibration_alpha,
                sampling_epsilon=cfg.sampling_epsilon,
                n_signature_slots=cfg.signature_slots,
                predicted_rate=predicted)
        kwargs = dict(min_checks=cfg.min_checks,
                      window_seconds=cfg.window_seconds)
        if cfg.threshold_rule == "balanced":
            self.policy = VerificationPolicy.balanced(spec, **kwargs)
        elif cfg.threshold_rule == "thirds":
            self.policy = VerificationPolicy.from_noise_floor(spec, **kwargs)
        else:
            raise ValueError(
                f"threshold_rule must be 'balanced' or 'thirds', not "
                f"{cfg.threshold_rule!r}")
        self.timings["calibrate"] = time.perf_counter() - t0
        return self.policy

    @property
    def spec_violations(self) -> Tuple[str, ...]:
        """Recipients whose decoys are inconsistent with the declared spec.

        Non-empty means the channel is measurably worse than certified.  That
        is a detection event in its own right: it does not say whether the
        cause is a degraded fibre or an eavesdropper, only that continuing to
        sign on this link is not covered by the security analysis.
        """
        return tuple(name for name, c in self.calibrations.items()
                     if c.spec_violation)

    # -- phase 5 ----------------------------------------------------------
    def sign(self, message: bytes, **kwargs) -> SignatureDeclaration:
        """Sign one message.  The key is spent afterwards."""
        if self.private_key is None:
            raise RuntimeError("call enrol() before sign()")
        if self.declaration is not None:
            raise RuntimeError(
                "this private key has already signed; signing twice with a "
                "one-time key reveals both halves of every block and destroys "
                "unforgeability")
        t0 = time.perf_counter()
        self._counter += 1
        self.declaration = sign_message(
            private_key=self.private_key, message=message,
            recipients=self.config.verifiers, reservoirs=self.reservoirs,
            counter=self._counter, rng=self.rng, **kwargs)
        self.timings["sign"] = time.perf_counter() - t0
        return self.declaration

    def verify(self, verifier: str,
               declaration: Optional[SignatureDeclaration] = None,
               level: str = "accept",
               now: Optional[float] = None,
               policy: Optional[VerificationPolicy] = None
               ) -> VerificationReport:
        """Run one recipient's verification of a declaration."""
        if self.policy is None and policy is None:
            raise RuntimeError("call calibrate() before verify()")
        decl = declaration if declaration is not None else self.declaration
        if decl is None:
            raise RuntimeError("nothing has been signed yet")
        t0 = time.perf_counter()
        report = verify_declaration(
            declaration=decl, store=self.stores[verifier],
            policy=policy or self.policy, rng=self.rng,
            registry=self.registries.get(verifier),
            reservoir=self.reservoirs.get(verifier),
            ledger=self.ledgers.get(verifier),
            noise=self.config.distribution.noise, level=level, now=now)
        self.timings[f"verify:{verifier}:{level}"] = time.perf_counter() - t0
        self.reports[f"{verifier}:{level}"] = report
        return report

    def transfer(self, holder: str, to: str,
                 declaration: Optional[SignatureDeclaration] = None,
                 **kwargs) -> VerificationReport:
        """Forward a declaration and have the next party verify it.

        ``holder`` is recorded but has no power: the object handed on is
        byte-identical to what Alice produced, and ``to`` checks Alice's own
        MAC tag, not the forwarder's.  That is the whole point of
        transferability -- Charlie's confidence does not rest on Bob.
        """
        decl = declaration if declaration is not None else self.declaration
        report = self.verify(to, decl, level="transfer", **kwargs)
        report.reasons.append(f"forwarded by '{holder}'")
        return report

    # -- convenience -------------------------------------------------------
    def run(self, message: bytes) -> SessionResult:
        """All five phases, then verify with everyone and transfer once."""
        self.enrol()
        self.distribute()
        self.symmetrise()
        self.calibrate()
        decl = self.sign(message)

        verifications: Dict[str, Dict[str, object]] = {}
        for name in self.config.verifiers[:1]:
            verifications[name] = self.verify(name, decl).to_dict()
        transfers: Dict[str, Dict[str, object]] = {}
        holder = self.config.verifiers[0]
        for name in self.config.verifiers[1:]:
            transfers[name] = self.transfer(holder, name, decl).to_dict()

        return SessionResult(
            config=self.config,
            private_key_summary=self.private_key.summary(),
            distribution={n: r.summary() for n, r in self.results.items()},
            symmetrisation=self.symmetrisation,
            calibration={n: c.to_dict() for n, c in self.calibrations.items()},
            policy=self.policy.to_dict(),
            declaration=decl.summary(),
            verifications=verifications,
            transfers=transfers,
            timings=dict(self.timings),
        )
