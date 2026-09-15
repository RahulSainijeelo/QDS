"""Private keys, quantum public keys, and identity enrolment.

Key structure
-------------
A message of ``message_bits`` bits is signed one bit at a time, Lamport style.
For every bit position ``j`` and every possible value ``b`` of that bit, Alice
draws an independent private key block of ``L`` entries::

    K[j, b] = ((a_1, c_1), (a_2, c_2), ..., (a_L, c_L)),   a_i, c_i in {0, 1}

Entry ``(a, c)`` names one of the four BB84 states ``|0>, |1>, |+>, |->``:
``a`` selects the basis (Z or X) and ``c`` the eigenvalue.  The corresponding
**quantum public key** is the product state of those ``L`` qubits, one copy per
recipient, delivered by teleportation before any message exists.

Why the four-state set
---------------------
The two bases are mutually unbiased, so the four states are pairwise
non-orthogonal.  A recipient holding one copy therefore cannot determine
``(a, c)``: the optimal measurement (the pretty-good measurement, which is
optimal here because the set is geometrically uniform) identifies the state
with probability exactly 1/2, and no amount of computation improves on that.
That gap between "holds the state" and "knows the key" is the entire source of
unforgeability -- it is a statement about physics, not about complexity.

To sign bit value ``b`` at position ``j``, Alice publishes ``K[j, b]``.  The
recipient measures his stored qubit ``i`` in basis ``a_i`` and checks that the
outcome equals ``c_i``.  Honest and noiseless, every check passes; a forger who
guessed the block fails a constant fraction of them.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "KeyEntry", "PrivateKey", "generate_private_key",
    "Slot", "SlotRole", "HeldQubit", "PublicKeyStore",
    "SignerRecord", "SignerRegistry",
    "FORGER_PASS_PROB_ONE_COPY", "FORGER_PASS_PROB_BLIND",
    "blind_guess_pass_probability", "single_copy_pass_probability",
]


class KeyEntry(NamedTuple):
    """One private-key entry: ``basis`` in {0=Z, 1=X}, ``bit`` in {0, 1}."""

    basis: int
    bit: int

    @property
    def basis_label(self) -> str:
        return "X" if self.basis else "Z"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return {(0, 0): "|0>", (0, 1): "|1>", (1, 0): "|+>", (1, 1): "|->"}[
            (self.basis, self.bit)
        ]


class Slot(NamedTuple):
    """Address of one public-key qubit.

    ``j`` = message bit position, ``b`` = the message-bit value this block
    would sign, ``i`` = repetition index within the block.  Check and decoy
    rounds use the reserved values in :class:`SlotRole`.
    """

    j: int
    b: int
    i: int


class SlotRole:
    """Reserved ``j`` values for slots that never carry a signature."""

    CHECK = -1   #: sacrificed to estimate the entanglement quality
    DECOY = -2   #: revealed immediately to estimate the end-to-end error rate


#: Per-check pass probability of a forger who has no copy of the public key
#: and simply guesses ``(a, c)``.  Derivation in docs/WHITEPAPER.md, Lemma 1.
FORGER_PASS_PROB_BLIND = 0.5

#: Per-check pass probability of a forger holding exactly one copy of the
#: public-key qubit and using the optimal (pretty-good) measurement.
#: Derivation in docs/WHITEPAPER.md, Lemma 2.
FORGER_PASS_PROB_ONE_COPY = 0.75


def blind_guess_pass_probability() -> float:
    """Pass probability per check for a key-blind forger.

    The verifier holds ``|psi(a, c)>`` and is told ``(a', c')``.  He measures in
    basis ``a'`` and demands outcome ``c'``:

    * ``a' = a``: the outcome is deterministic, so the check passes iff
      ``c' = c`` -- probability 1/2 over the unknown ``c``.
    * ``a' != a``: the bases are mutually unbiased, so the outcome is uniform
      and the check passes with probability 1/2.

    Both branches give 1/2, hence 1/2 overall, independently of the forger's
    guessing strategy.  A blind forger cannot beat a coin flip per check.
    """
    return 0.5


def single_copy_pass_probability() -> float:
    """Pass probability per check for a forger holding one public-key copy.

    The pretty-good measurement on the four-state set has POVM elements
    ``M_i = |psi_i><psi_i| / 2``.  Given the true state ``|0>`` it returns
    ``|0>`` with probability 1/2, ``|1>`` with probability 0, and each of
    ``|+>, |->`` with probability 1/4.  So the forger names the correct state
    half the time (check passes), names the *conjugate* basis half the time
    (check passes with probability 1/2 because the outcome is then uniform),
    and never names the right basis with the wrong bit.  Total:

        1/2 * 1 + 1/2 * 1/2 = 3/4

    This 3/4 -- not 1/2 -- is the number the acceptance threshold must be set
    against, because a dishonest *recipient* is exactly a forger holding one
    copy.  It is why the honest error rate has to sit below 1/4.
    """
    return 0.75


# --------------------------------------------------------------------------
# private key
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PrivateKey:
    """Alice's private key for one signing session (one-time use)."""

    signer: str
    key_id: str
    message_bits: int
    L: int
    blocks: Dict[Tuple[int, int], Tuple[KeyEntry, ...]]

    def block(self, j: int, b: int) -> Tuple[KeyEntry, ...]:
        return self.blocks[(j, b)]

    def slots(self) -> List[Slot]:
        """Every signature-carrying slot, in canonical order."""
        return [
            Slot(j, b, i)
            for j in range(self.message_bits)
            for b in (0, 1)
            for i in range(self.L)
        ]

    def entry(self, slot: Slot) -> KeyEntry:
        return self.blocks[(slot.j, slot.b)][slot.i]

    @property
    def total_slots(self) -> int:
        return 2 * self.message_bits * self.L

    def reveal(self, message_bits: Sequence[int]) -> Dict[int, Tuple[KeyEntry, ...]]:
        """The part of the private key that signing a given message discloses.

        Exactly one block per bit position: the block for the *other* value is
        never revealed, which is what stops an adversary who has seen one
        signature from signing a different message.
        """
        if len(message_bits) != self.message_bits:
            raise ValueError(
                f"message has {len(message_bits)} bits, key is for {self.message_bits}"
            )
        return {j: self.block(j, int(bit)) for j, bit in enumerate(message_bits)}

    def summary(self) -> Dict[str, object]:
        return {
            "signer": self.signer,
            "key_id": self.key_id,
            "message_bits": self.message_bits,
            "L": self.L,
            "signature_slots": self.total_slots,
            "private_key_bits": 2 * self.total_slots,
        }


def generate_private_key(signer: str, message_bits: int, L: int,
                         rng: np.random.Generator,
                         key_id: Optional[str] = None) -> PrivateKey:
    """Draw a fresh private key uniformly at random.

    Uniformity is required by Lemma 1: if the basis bits were biased, a forger
    could do better than a coin flip per check.
    """
    if message_bits < 1 or L < 1:
        raise ValueError("message_bits and L must be positive")
    blocks: Dict[Tuple[int, int], Tuple[KeyEntry, ...]] = {}
    for j in range(message_bits):
        for b in (0, 1):
            raw = rng.integers(0, 2, size=(L, 2))
            blocks[(j, b)] = tuple(
                KeyEntry(int(a), int(c)) for a, c in raw
            )
    if key_id is None:
        key_id = "pk-" + "".join(
            f"{int(x):x}" for x in rng.integers(0, 16, size=8)
        )
    return PrivateKey(signer=signer, key_id=key_id, message_bits=message_bits,
                      L=L, blocks=blocks)


# --------------------------------------------------------------------------
# the verifier's quantum memory
# --------------------------------------------------------------------------

@dataclass
class HeldQubit:
    """One qubit of a quantum public key, as actually held by a recipient.

    ``rho`` is the *reduced* single-qubit density matrix.  That is not a
    simplification: any eavesdropper probe still entangled with this qubit
    cannot influence the recipient's measurement statistics, so the marginal
    is a complete description of everything the verifier can observe.  Joint
    states including Eve's probe are kept by the attack layer where they are
    needed to quantify her information gain.
    """

    slot: Slot
    rho: np.ndarray
    correction: Tuple[int, int] = (0, 0)   #: Pauli-correction bits (u, v)
    arrived: bool = True                   #: False if the transmission was lost
    dark: bool = False                     #: outcome fabricated by a dark count
    consumed: bool = False
    storage_intervals: int = 0

    def take(self) -> np.ndarray:
        """Consume the qubit (measurement destroys it -- one-time by physics)."""
        if self.consumed:
            raise RuntimeError(f"public-key slot {self.slot} was already measured")
        self.consumed = True
        return self.rho


@dataclass
class PublicKeyStore:
    """Everything one recipient holds for one signer's public key."""

    verifier: str
    signer: str
    key_id: str
    qubits: Dict[Slot, HeldQubit] = field(default_factory=dict)
    check_slots: List[Slot] = field(default_factory=list)
    decoy_slots: List[Slot] = field(default_factory=list)
    #: (u, v) Pauli-correction bits announced during distribution, in order
    correction_history: List[Tuple[int, int]] = field(default_factory=list)
    #: per-transmission detector outcome: "click" / "dark" / "lost"
    detector_log: List[str] = field(default_factory=list)

    def add(self, held: HeldQubit) -> None:
        self.qubits[held.slot] = held
        self.correction_history.append(held.correction)

    def get(self, slot: Slot) -> HeldQubit:
        try:
            return self.qubits[slot]
        except KeyError as exc:
            raise KeyError(
                f"{self.verifier} holds no public-key qubit for slot {slot}"
            ) from exc

    @property
    def n_available(self) -> int:
        return sum(1 for q in self.qubits.values() if not q.consumed and q.arrived)

    @property
    def yield_observed(self) -> float:
        if not self.detector_log:
            return 1.0
        clicks = sum(1 for s in self.detector_log if s != "lost")
        return clicks / len(self.detector_log)


# --------------------------------------------------------------------------
# identity binding
# --------------------------------------------------------------------------

@dataclass
class SignerRecord:
    """What a verifier knows about an enrolled signer."""

    signer: str
    key_id: str
    enrolled_at: float
    #: MAC key index that authenticated the enrolment message
    enrolment_mac_index: int = -1
    stores: Dict[str, PublicKeyStore] = field(default_factory=dict)


@dataclass
class SignerRegistry:
    """Per-verifier list of signers whose public keys have been enrolled.

    Impersonation resistance starts here and it is not statistical.  A
    signature naming a signer with no enrolment record, or naming an enrolled
    signer but a different ``key_id``, is rejected outright: there is no
    quantum public key to check it against, so there is nothing to be
    statistically uncertain about.
    """

    verifier: str
    records: Dict[str, SignerRecord] = field(default_factory=dict)

    def enrol(self, signer: str, key_id: str, at: float,
              mac_index: int = -1) -> SignerRecord:
        rec = SignerRecord(signer=signer, key_id=key_id, enrolled_at=at,
                           enrolment_mac_index=mac_index)
        self.records[signer] = rec
        return rec

    def lookup(self, signer: str, key_id: str) -> Tuple[Optional[SignerRecord], List[str]]:
        """Resolve a claimed identity.  Returns ``(record, problems)``."""
        problems: List[str] = []
        rec = self.records.get(signer)
        if rec is None:
            problems.append(f"no enrolled public key for signer '{signer}'")
            return None, problems
        if not hmac.compare_digest(rec.key_id, key_id):
            problems.append(
                f"signature cites key_id '{key_id}' but '{signer}' is enrolled "
                f"with '{rec.key_id}'"
            )
            return rec, problems
        return rec, problems

    def known_signers(self) -> List[str]:
        return sorted(self.records)
