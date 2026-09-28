"""Verification: projective measurement, mismatch counting, and the thresholds.

The check itself
----------------
For every revealed entry ``(a_i, c_i)`` the verifier measures his stored qubit
``i`` in the Pauli basis named by ``a_i`` -- ``Z`` for ``a=0``, ``X`` for
``a=1`` -- and demands the outcome ``c_i``.  That is a projective measurement
onto ``|psi(a_i, c_i)><psi(a_i, c_i)|``: if Alice really sent that state and
the channel were perfect, the outcome would be ``c_i`` with probability
exactly 1.  Every mismatch is therefore a physical event -- noise or an
adversary -- and the only question is how many are too many.

Two thresholds, not one
-----------------------
A single threshold cannot give both unforgeability and non-repudiation, so
the scheme uses two, with ``s_a < s_v``:

``s_a`` -- **acceptance**
    used by the recipient Alice sent to.

``s_v`` -- **transfer** (looser)
    used by anyone the message is forwarded to.

The three gaps each buy one security property, and each is priced by a
relative-entropy exponent rather than by a normal approximation:

``e_honest -> s_a``
    room for honest noise.  Too tight and legitimate messages are rejected;
    the cost is ``exp(-N D(s_a || e_honest))``.

``s_a -> s_v``
    room against **repudiation**.  Alice's only route to disowning a message
    is to craft a signature that lands below ``s_a`` for the recipient and
    above ``s_v`` for the forwardee.  Whichever side of the midpoint ``m`` the
    true mismatch rate falls on, one of those two events is a large
    deviation, so the cost is ``exp(-N min(D(s_v||m), D(s_a||m)))``.

    **This bound is only valid after key symmetrisation.**  Read literally it
    assumes both parties' mismatch rates have the same mean, and a cheating
    Alice can violate that trivially by sending clean states to the recipient
    and dirty ones to the forwardee.  The fix is physical, not statistical:
    before any message is signed, the parties secretly exchange a random half
    of their stored qubits (:func:`~qds.protocol.session.symmetrise_keys`), so
    Alice cannot know which slots will be checked by whom.  The session layer
    performs that exchange by default and
    :meth:`VerificationPolicy.soundness_problems` has no way to see whether it
    happened -- so a deployment that skips it has no non-repudiation claim,
    only an unforgeability one.

``s_v -> 1/4``
    room against **forgery**.  1/4 is not a design choice, it is the optimal
    single-copy error rate from
    :func:`~qds.protocol.keys.single_copy_pass_probability`.  No measurement,
    no computation and no amount of time lets a forger beat it, so the cost is
    ``exp(-N D(s_v || 1/4))`` against an adversary of unbounded power.

If ``e_honest >= 1/4`` the hardware is simply too noisy for the scheme to be
secure at any threshold, and :meth:`VerificationPolicy.soundness_problems`
says so instead of silently producing a policy that cannot be met.

Order of operations is a security property
------------------------------------------
Identity, MAC and freshness are checked **before** any qubit is measured, and
:func:`verify_declaration` refuses to measure if they fail.  A stored public
key is consumable -- measurement destroys it -- so a verifier who measured
first could have his entire key burned by an adversary sending junk
declarations he was never going to accept.  ``policy.always_measure`` exists
only so the attack suite can deliberately weaken the verifier and isolate the
physics from the bookkeeping; it is not something an honest deployment sets.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import linalg as L
from ..channel import NoiseModel
from ..detect import stats
from .auth import FreshnessLedger, FreshnessVerdict, KeyReservoir
from .keys import (FORGER_PASS_PROB_BLIND, FORGER_PASS_PROB_ONE_COPY,
                   HeldQubit, PublicKeyStore, SignerRegistry, Slot)
from .signing import SignatureDeclaration

__all__ = [
    "FORGER_ERROR_RATE", "BLIND_FORGER_ERROR_RATE",
    "VerificationPolicy", "SlotCheck", "VerificationReport",
    "verify_declaration", "measure_slot",
]

#: Per-check error rate of the strongest forger -- a dishonest recipient who
#: holds one copy of the public key and uses the optimal measurement.
FORGER_ERROR_RATE = 1.0 - FORGER_PASS_PROB_ONE_COPY      # 0.25

#: Per-check error rate of a forger with no copy at all.
BLIND_FORGER_ERROR_RATE = 1.0 - FORGER_PASS_PROB_BLIND   # 0.50


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationPolicy:
    """Where the two thresholds sit, and what that costs in each direction."""

    honest_error_rate: float
    s_a: float
    s_v: float
    #: reject outright below this many usable checks; a threshold on three
    #: checks is arithmetic, not evidence
    min_checks: int = 16
    #: clock window for the freshness test, seconds
    window_seconds: float = 300.0
    #: deliberately weakened verifier: measure even when the classical layer
    #: has already rejected.  Attack-suite use only.
    always_measure: bool = False
    forger_error_rate: float = FORGER_ERROR_RATE

    # -- construction ------------------------------------------------------
    @classmethod
    def from_noise_floor(cls, honest_error_rate: float,
                         accept_fraction: float = 1.0 / 3.0,
                         transfer_fraction: float = 2.0 / 3.0,
                         **kwargs) -> "VerificationPolicy":
        """Place ``s_a`` and ``s_v`` inside the usable gap ``[e, 1/4)``.

        With the defaults the gap is split into equal thirds, so the room for
        honest noise, the room against repudiation and the room against
        forgery are all the same width.  That is the symmetric choice, not a
        tuned one; :meth:`failure_probabilities` prices any other split.
        """
        e = float(honest_error_rate)
        gap = FORGER_ERROR_RATE - e
        if gap <= 0.0:
            # hardware too noisy: still produce a policy so callers can
            # inspect soundness_problems() rather than catching an exception
            return cls(honest_error_rate=e, s_a=e, s_v=e, **kwargs)
        if not 0.0 < accept_fraction < transfer_fraction < 1.0:
            raise ValueError(
                "need 0 < accept_fraction < transfer_fraction < 1")
        return cls(honest_error_rate=e,
                   s_a=e + accept_fraction * gap,
                   s_v=e + transfer_fraction * gap,
                   **kwargs)

    @classmethod
    def balanced(cls, honest_error_rate: float, grid: int = 64,
                 rounds: int = 6, **kwargs) -> "VerificationPolicy":
        """Place the thresholds to maximise the *worst* security exponent.

        The equal-thirds split of :meth:`from_noise_floor` is symmetric in
        threshold space, but the four failure modes do not convert rate into
        exponent at the same rate: near ``e`` the relative entropy climbs
        steeply, near 1/4 it does not, and the repudiation exponent depends
        only on the width of the middle gap.  Maximising
        ``min(E_abort, E_forge, E_repudiation)`` therefore shortens the key --
        typically by a factor of two or more -- for identical security.

        Solved by a coarse grid followed by a local refinement.  The objective
        is smooth and unimodal in the interior of the feasible triangle, so
        this converges to the optimum rather than merely near it; the search
        costs microseconds and runs once per policy.
        """
        e = float(honest_error_rate)
        if e >= FORGER_ERROR_RATE:
            return cls(honest_error_rate=e, s_a=e, s_v=e, **kwargs)

        def worst(s_a: float, s_v: float) -> float:
            if not (e < s_a < s_v < FORGER_ERROR_RATE):
                return -1.0
            m = 0.5 * (s_a + s_v)
            return min(stats.binary_kl(s_a, e),
                       stats.binary_kl(s_v, FORGER_ERROR_RATE),
                       stats.binary_kl(s_v, m),
                       stats.binary_kl(s_a, m))

        lo, hi = e, FORGER_ERROR_RATE
        best = (e + (hi - e) / 3.0, e + 2.0 * (hi - e) / 3.0)
        span = hi - lo
        for _ in range(rounds):                  # coarse grid, then zoom in
            step = span / grid
            a0 = max(lo, best[0] - span / 2.0)
            v0 = max(lo, best[1] - span / 2.0)
            best_val = worst(*best)
            for i in range(grid):
                s_a = a0 + i * step
                for k in range(grid):
                    s_v = v0 + k * step
                    val = worst(s_a, s_v)
                    if val > best_val:
                        best_val, best = val, (s_a, s_v)
            span = max(span / (grid / 8.0), 1e-12)
        return cls(honest_error_rate=e, s_a=best[0], s_v=best[1], **kwargs)

    # -- soundness ---------------------------------------------------------
    def soundness_problems(self) -> List[str]:
        """Reasons this policy cannot deliver its security claims, if any."""
        out: List[str] = []
        if self.honest_error_rate >= FORGER_ERROR_RATE:
            out.append(
                f"honest error rate {self.honest_error_rate:.4f} is at or above "
                f"the optimal single-copy forger rate {FORGER_ERROR_RATE:.2f}; "
                "no threshold can separate honest noise from forgery on this "
                "hardware")
        if not self.honest_error_rate < self.s_a:
            out.append(f"s_a={self.s_a:.4f} does not exceed the honest rate "
                       f"{self.honest_error_rate:.4f}: honest messages will be "
                       "rejected about half the time")
        if not self.s_a < self.s_v:
            out.append(f"s_a={self.s_a:.4f} must be strictly below "
                       f"s_v={self.s_v:.4f} or the scheme is repudiable")
        if not self.s_v < FORGER_ERROR_RATE:
            out.append(f"s_v={self.s_v:.4f} is at or above {FORGER_ERROR_RATE:.2f}: "
                       "a forger holding one copy of the key passes by default")
        return out

    @property
    def sound(self) -> bool:
        return not self.soundness_problems()

    def threshold(self, level: str) -> float:
        if level == "accept":
            return self.s_a
        if level == "transfer":
            return self.s_v
        raise ValueError(f"level must be 'accept' or 'transfer', not {level!r}")

    # -- what the thresholds cost -----------------------------------------
    def failure_probabilities(self, n_checks: int) -> Dict[str, float]:
        """Exponential bounds on each way the scheme can fail, at ``n_checks``.

        Every entry is a Chernoff-Hoeffding relative-entropy bound.  None of
        them assumes a normal approximation, a large-``N`` limit, or anything
        about the adversary's computational power.
        """
        n = int(n_checks)
        m = 0.5 * (self.s_a + self.s_v)
        return {
            "honest_abort": stats.chernoff_kl_tail(
                n, self.honest_error_rate, self.s_a),
            "forgery_one_copy": stats.chernoff_kl_tail_below(
                n, self.forger_error_rate, self.s_v),
            "forgery_blind": stats.chernoff_kl_tail_below(
                n, BLIND_FORGER_ERROR_RATE, self.s_v),
            "repudiation": math.exp(-n * min(stats.binary_kl(self.s_v, m),
                                             stats.binary_kl(self.s_a, m))),
        }

    def security_bits(self, n_checks: int) -> Dict[str, float]:
        """:meth:`failure_probabilities` as ``-log2 p``, which reads better."""
        out = {}
        for k, p in self.failure_probabilities(n_checks).items():
            out[k] = float("inf") if p <= 0.0 else -math.log2(min(p, 1.0))
        return out

    def required_checks(self, epsilon: float = 1e-9) -> int:
        """Smallest ``N`` at which *every* failure bound is at most ``epsilon``.

        This is the number that sizes the key: ``N = k * L`` checks means
        ``L = N / k`` repetitions per message bit.  It grows only
        logarithmically in ``1/epsilon``, which is why a 1-in-a-billion
        security target is affordable.
        """
        if not self.sound:
            return -1
        lo, hi = 1, 2
        while max(self.failure_probabilities(hi).values()) > epsilon:
            hi *= 2
            if hi > 1 << 24:                      # pragma: no cover - guard
                return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if max(self.failure_probabilities(mid).values()) > epsilon:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def to_dict(self) -> Dict[str, object]:
        return {
            "honest_error_rate": self.honest_error_rate,
            "s_a": self.s_a,
            "s_v": self.s_v,
            "forger_error_rate": self.forger_error_rate,
            "blind_forger_error_rate": BLIND_FORGER_ERROR_RATE,
            "min_checks": self.min_checks,
            "window_seconds": self.window_seconds,
            "sound": self.sound,
            "problems": self.soundness_problems(),
        }


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

def measure_slot(held: HeldQubit, basis: int, rng: np.random.Generator,
                 noise: Optional[NoiseModel] = None) -> Tuple[Optional[int], str]:
    """Measure one stored qubit and return ``(bit, status)``.

    ``status`` is one of ``"ok"`` (a real click), ``"dark"`` (the detector
    fired without a photon, so the bit is a fair coin and known to be
    worthless), or ``"missing"`` (nothing registered).  The qubit is consumed
    either way: the slot is spent whether or not it produced an answer, which
    is exactly how a one-time quantum key behaves.

    The verifier's own readout error is applied here and nowhere else.  It is
    deliberately *not* folded into the delivered state, because it is a
    property of his detector rather than of the qubit, and only the rounds he
    actually reads out are affected by it.
    """
    label = "X" if basis else "Z"
    rho = held.take()
    if not held.arrived:
        return None, "missing"
    if held.n_qubits != 1:
        raise ValueError(
            f"slot {held.slot} holds {held.n_qubits} qubits; check pairs are "
            "measured by the entanglement monitor, not the signature check")
    bit, _post, _p = L.measure_qubit(rho, 0, label, rng)
    if noise is not None:
        bit = noise.flip_readout(int(bit), rng)
    return int(bit), ("dark" if held.dark else "ok")


@dataclass
class SlotCheck:
    """One projective measurement and its verdict.

    Everything here is observable by the verifier.  Nothing in this record
    depends on knowing the true state, which is what lets the detection
    engine consume it without becoming an oracle.
    """

    slot: Slot
    basis: int
    expected: int
    observed: Optional[int]
    status: str          #: "ok" / "dark" / "missing"

    @property
    def counted(self) -> bool:
        """Did this slot contribute to the mismatch statistic?

        Dark counts do: the detector reported something, and the verifier has
        no way to know it was spurious.  Excluding them would be an oracle
        move and would also hide exactly the damage that a blinding attack
        does.
        """
        return self.observed is not None

    @property
    def mismatch(self) -> bool:
        return self.counted and self.observed != self.expected

    def to_dict(self) -> Dict[str, object]:
        return {"slot": list(self.slot), "basis": "X" if self.basis else "Z",
                "expected": self.expected, "observed": self.observed,
                "status": self.status, "mismatch": self.mismatch}


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

@dataclass
class VerificationReport:
    """The full, observable outcome of one verification attempt."""

    verifier: str
    signer: str
    key_id: str
    message: bytes
    level: str
    policy: VerificationPolicy
    threshold: float

    checks: List[SlotCheck] = field(default_factory=list)
    identity_ok: bool = False
    identity_problems: List[str] = field(default_factory=list)
    mac_ok: bool = False
    mac_problems: List[str] = field(default_factory=list)
    freshness: Optional[FreshnessVerdict] = None
    measured: bool = False
    accepted: bool = False
    reasons: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    # -- counts ------------------------------------------------------------
    @property
    def n_checked(self) -> int:
        return sum(1 for c in self.checks if c.counted)

    @property
    def n_mismatch(self) -> int:
        return sum(1 for c in self.checks if c.mismatch)

    @property
    def n_missing(self) -> int:
        return sum(1 for c in self.checks if c.status == "missing")

    @property
    def n_dark(self) -> int:
        return sum(1 for c in self.checks if c.status == "dark")

    @property
    def mismatch_rate(self) -> float:
        n = self.n_checked
        return self.n_mismatch / n if n else 0.0

    @property
    def detection_yield(self) -> float:
        """Fraction of addressed slots that produced any outcome at all."""
        return self.n_checked / len(self.checks) if self.checks else 0.0

    def by_basis(self) -> Dict[str, Tuple[int, int]]:
        """``basis -> (mismatches, checks)``.

        Reported separately because a basis-asymmetric attack can sit under
        the pooled threshold while being obvious in one basis -- see
        ``qds.channel.BasisBiasedProbe``.
        """
        out = {"Z": [0, 0], "X": [0, 0]}
        for c in self.checks:
            if not c.counted:
                continue
            cell = out["X" if c.basis else "Z"]
            cell[1] += 1
            cell[0] += int(c.mismatch)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def by_position(self) -> Dict[int, Tuple[int, int]]:
        """``message-bit index -> (mismatches, checks)``.

        Localises tampering: a forger who rewrote one bit of the message
        produces mismatches concentrated in that block rather than spread
        across all of them.
        """
        out: Dict[int, List[int]] = {}
        for c in self.checks:
            if not c.counted:
                continue
            cell = out.setdefault(c.slot.j, [0, 0])
            cell[1] += 1
            cell[0] += int(c.mismatch)
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    # -- statistics --------------------------------------------------------
    def p_value_vs_honest(self) -> float:
        """Exact binomial p-value for "noisier than calibration allows"."""
        return stats.binom_test_greater(self.n_mismatch, self.n_checked,
                                        self.policy.honest_error_rate)

    def p_value_vs_forger(self) -> float:
        """Exact p-value for "as clean as a forger could plausibly manage".

        Small means the observed mismatch count is *too low* to have come
        from a one-copy forger, i.e. positive evidence of authenticity rather
        than mere absence of evidence against it.
        """
        return stats.binom_test_less(self.n_mismatch, self.n_checked,
                                     self.policy.forger_error_rate)

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        return stats.clopper_pearson(self.n_mismatch, self.n_checked, alpha)

    def margin(self) -> float:
        """Signed distance from the threshold; negative means accepted."""
        return self.mismatch_rate - self.threshold

    def to_dict(self) -> Dict[str, object]:
        n = self.n_checked
        lo, hi = self.confidence_interval()
        return {
            "verifier": self.verifier,
            "signer": self.signer,
            "key_id": self.key_id,
            "message_hex": self.message.hex(),
            "level": self.level,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "identity_ok": self.identity_ok,
            "identity_problems": list(self.identity_problems),
            "mac_ok": self.mac_ok,
            "mac_problems": list(self.mac_problems),
            "freshness": self.freshness.to_dict() if self.freshness else None,
            "measured": self.measured,
            "slots_addressed": len(self.checks),
            "checks": n,
            "mismatches": self.n_mismatch,
            "missing": self.n_missing,
            "dark": self.n_dark,
            "mismatch_rate": self.mismatch_rate,
            "detection_yield": self.detection_yield,
            "margin": self.margin(),
            "by_basis": {k: {"mismatch": v[0], "checks": v[1]}
                         for k, v in self.by_basis().items()},
            "by_position": {str(k): {"mismatch": v[0], "checks": v[1]}
                            for k, v in self.by_position().items()},
            "p_value_vs_honest": self.p_value_vs_honest(),
            "p_value_vs_forger": self.p_value_vs_forger(),
            "clopper_pearson_95": [lo, hi],
            "failure_bounds": self.policy.failure_probabilities(n),
            "security_bits": self.policy.security_bits(n),
            "elapsed_seconds": self.elapsed_seconds,
        }


# --------------------------------------------------------------------------
# the verifier
# --------------------------------------------------------------------------

def verify_declaration(declaration: SignatureDeclaration,
                       store: PublicKeyStore,
                       policy: VerificationPolicy,
                       rng: np.random.Generator,
                       registry: Optional[SignerRegistry] = None,
                       reservoir: Optional[KeyReservoir] = None,
                       ledger: Optional[FreshnessLedger] = None,
                       noise: Optional[NoiseModel] = None,
                       level: str = "accept",
                       now: Optional[float] = None) -> VerificationReport:
    """Run one complete verification and return everything it observed.

    The four gates, in order, are identity, authentication, freshness and the
    quantum check.  The first three are deterministic -- they reject with
    probability 1 and carry no threshold -- which is why they come first: a
    problem that can be settled by bookkeeping should never be handed to a
    statistical test.
    """
    started = time.perf_counter()
    if now is None:
        now = time.time()
    report = VerificationReport(
        verifier=store.verifier, signer=declaration.signer,
        key_id=declaration.key_id, message=declaration.message,
        level=level, policy=policy, threshold=policy.threshold(level),
    )

    # -- gate 1: is this signer enrolled, with this key? -------------------
    if registry is None:
        report.identity_ok = True
        report.identity_problems = []
    else:
        record, problems = registry.lookup(declaration.signer,
                                           declaration.key_id)
        report.identity_ok = record is not None and not problems
        report.identity_problems = problems
    if not report.identity_ok:
        report.reasons.extend(report.identity_problems)
    if store.signer != declaration.signer or store.key_id != declaration.key_id:
        report.identity_ok = False
        msg = (f"held public key is {store.signer}/{store.key_id} but the "
               f"declaration cites {declaration.signer}/{declaration.key_id}")
        report.identity_problems.append(msg)
        report.reasons.append(msg)

    # -- gate 2: was the classical payload authenticated? ------------------
    report.mac_ok, report.mac_problems = declaration.check_mac(store.verifier,
                                                               reservoir)
    if not report.mac_ok:
        report.reasons.extend(report.mac_problems)

    # -- gate 3: is it fresh? ---------------------------------------------
    slot_tuples = declaration.slot_tuples()
    if ledger is not None:
        report.freshness = ledger.check(
            signer=declaration.signer, counter=declaration.counter,
            nonce=declaration.nonce, timestamp=declaration.timestamp,
            now=now, slots=slot_tuples)
        if not report.freshness.ok:
            report.reasons.extend(report.freshness.reasons)
    else:
        report.freshness = FreshnessVerdict(ok=True)

    classical_ok = (report.identity_ok and report.mac_ok
                    and report.freshness.ok)

    # -- gate 4: the physics ----------------------------------------------
    if classical_ok or policy.always_measure:
        report.measured = True
        for slot in declaration.slots():
            entry = declaration.entry(slot)
            try:
                held = store.get(slot)
            except KeyError:
                report.checks.append(SlotCheck(slot=slot, basis=entry.basis,
                                               expected=entry.bit,
                                               observed=None, status="missing"))
                continue
            if held.consumed:
                report.checks.append(SlotCheck(slot=slot, basis=entry.basis,
                                               expected=entry.bit,
                                               observed=None, status="missing"))
                continue
            bit, status = measure_slot(held, entry.basis, rng, noise)
            report.checks.append(SlotCheck(slot=slot, basis=entry.basis,
                                           expected=entry.bit, observed=bit,
                                           status=status))
        if ledger is not None:
            # burn the slots whether or not we end up accepting: they are
            # physically gone, and the ledger is what makes that observable
            ledger.commit(declaration.signer, declaration.counter,
                          declaration.nonce, slot_tuples)

    # -- decision ----------------------------------------------------------
    n = report.n_checked
    if not classical_ok:
        report.accepted = False
    elif n < policy.min_checks:
        report.accepted = False
        report.reasons.append(
            f"only {n} usable checks, policy requires at least "
            f"{policy.min_checks}; too few to distinguish noise from forgery")
    elif report.mismatch_rate >= report.threshold:
        report.accepted = False
        report.reasons.append(
            f"mismatch rate {report.mismatch_rate:.4f} ({report.n_mismatch}/{n}) "
            f"reaches the {level} threshold {report.threshold:.4f}")
    else:
        report.accepted = True

    if ledger is not None:
        if report.accepted:
            ledger.accepted += 1
        else:
            ledger.rejected += 1

    report.elapsed_seconds = time.perf_counter() - started
    return report
