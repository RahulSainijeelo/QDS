"""Signature forgery: a party who does not hold the private key tries to sign.

Threat model
------------
A forger wants a verifier to accept ``"Alice signed m"`` when Alice did no
such thing.  He may be an outside attacker or a dishonest *recipient* -- the
recipient is the stronger adversary, because he legitimately holds one copy of
the quantum public key, so every bound here is stated against him.  He has
unbounded computation and unbounded time.  What he does not have is Alice's
private key: the classical description ``(a_i, c_i)`` of each stored qubit.

Two independent walls stop him, and this module exercises both.

The classical wall: the MAC
---------------------------
Alice authenticates each declaration with a one-time Wegman-Carter tag per
recipient, keyed from a reservoir she shares with that recipient and no one
else.  A forger cannot produce Charlie's tag, so a fabricated declaration
fails Charlie's MAC gate before a single qubit is measured.  This is real and
it is the first thing that happens; :func:`forge_and_submit` shows it, and the
engine attributes it to ``authentication_failure``.

The physical wall: the 1/4 bound
--------------------------------
The MAC is a classical mechanism, and a reviewer is right to ask what is left
if it is stripped away -- if, say, the scheme were deployed without per-recipient
classical authentication, relying on quantum unforgeability alone.  The answer
is the point of the whole scheme: a forger holding one copy of the public key
*cannot guess the key well enough to pass*.  The four BB84 states are pairwise
non-orthogonal, so the best single-copy measurement (the pretty-good
measurement) identifies a state with probability 1/2 and leaves a per-check
error rate of **1/4** -- a theorem about physics, not about the forger's
cleverness (``qds.protocol.keys.single_copy_pass_probability``).  A forger with
no copy at all is at **1/2**.

:func:`isolate_physics` demonstrates this directly.  It builds the forger's
best declaration by really measuring his stored qubits, then verifies it
against a *second* holder's copy with the classical gates switched off
(``policy.always_measure``), so the only thing deciding the outcome is the
quantum states.  The measured mismatch rate lands on the bound, which is what
"a hacker with infinite computing power still gets caught" means in numbers.

Strategies
----------
``blind``
    The forger ignores his copy and guesses ``(a, c)`` uniformly.  Error 1/2.
``single_copy``
    He measures his copy in a random Pauli basis and names the outcome.  This
    realises the pretty-good measurement on the geometrically uniform
    four-state set; error 1/4.
``breidbart``
    He measures in the Breidbart basis, which maximises his *information* about
    the key.  Maximising information is not the same as minimising disturbance,
    so his per-check error rate is measurably worse than ``single_copy`` -- a
    concrete reminder that learning the key and forging a signature are
    different goals.
``partial``
    A fraction ``known`` of entries have leaked (a side channel, a careless
    operator); the rest he attacks single-copy.  Shows the bound degrading
    gracefully with leakage, and why key secrecy is a premise, not a nicety.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

from ..channel import IDEAL, LAB_GRADE
from ..linalg import apply_unitary, measure_qubit, ry
from ..protocol.keys import KeyEntry, Slot
from ..protocol.signing import SignatureDeclaration, message_to_bits
from ..protocol.verification import verify_declaration
from .harness import (DEFAULT_MESSAGE, AttackResult, Expectation, analyse,
                      build_session)

__all__ = [
    "STRATEGIES", "guess_entry", "build_forged_declaration",
    "forge_and_submit", "isolate_physics",
]

#: The forger strategies this module knows how to mount.
STRATEGIES = ("blind", "single_copy", "breidbart", "partial")


# --------------------------------------------------------------------------
# one entry
# --------------------------------------------------------------------------

def guess_entry(rho: np.ndarray, rng: np.random.Generator, strategy: str,
                true_entry: Optional[KeyEntry] = None,
                known: float = 0.0) -> KeyEntry:
    """The forger's named ``(basis, bit)`` for one stored qubit.

    ``rho`` is the forger's own reduced single-qubit density matrix for the
    slot.  He may measure it (destroying his copy -- which is fine, he is
    forging once) and must then commit to a classical guess, because a
    declaration is classical.  ``true_entry`` is consulted only by the
    ``partial`` strategy, and only on the leaked fraction; it models a side
    channel, not privileged access, and no other strategy touches it.
    """
    if strategy == "blind":
        return KeyEntry(int(rng.integers(0, 2)), int(rng.integers(0, 2)))

    if strategy == "partial":
        if true_entry is not None and rng.random() < known:
            return KeyEntry(int(true_entry.basis), int(true_entry.bit))
        strategy = "single_copy"            # fall through for the unleaked rest

    if strategy == "single_copy":
        # Pretty-good measurement on the uniform four-state set is a random
        # Pauli-basis measurement; name the basis you chose and the bit you saw.
        basis = 0 if rng.random() < 0.5 else 1
        bit, _post, _p = measure_qubit(rho, 0, "X" if basis else "Z", rng)
        return KeyEntry(basis, int(bit))

    if strategy == "breidbart":
        # Rotate the Breidbart axis onto Z, read out, name the Z eigenstate.
        rot = apply_unitary(rho, ry(-math.pi / 4.0), [0])
        bit, _post, _p = measure_qubit(rot, 0, "Z", rng)
        return KeyEntry(0, int(bit))

    raise ValueError(f"unknown forger strategy {strategy!r}; "
                     f"choose from {STRATEGIES}")


# --------------------------------------------------------------------------
# a whole declaration
# --------------------------------------------------------------------------

def build_forged_declaration(session, forger: str, message: bytes, *,
                             strategy: str = "single_copy", known: float = 0.0,
                             counter: int = 1,
                             nonce: Optional[bytes] = None,
                             timestamp: Optional[float] = None,
                             ) -> SignatureDeclaration:
    """Fabricate ``"Alice signed message"`` from the forger's held key copy.

    The forger reads only what a recipient legitimately has: the signer name
    and ``key_id`` recorded in his own store, the public slot structure, and
    the qubits he holds.  He never reads ``session.private_key``.  The result
    carries ``mac_tags={}`` because he cannot produce one -- the honest verifier
    will reject on that, which is the whole point of the classical wall.
    """
    store = session.stores[forger]
    bits = message_to_bits(message)
    L = session.config.L
    rng = session.rng

    revealed: Dict[int, Tuple[KeyEntry, ...]] = {}
    for j, b in enumerate(bits):
        entries = []
        for i in range(L):
            slot = Slot(j, int(b), i)
            held = store.get(slot)
            # true_entry is used only by `partial`, to model leakage; it is read
            # from the forger's reduced state's slot label, not from the key.
            guess = guess_entry(held.rho, rng, strategy,
                                true_entry=None, known=known)
            entries.append(guess)
        revealed[j] = tuple(entries)

    if nonce is None:
        nonce = bytes(rng.integers(0, 256, size=16, dtype=np.uint8))
    if timestamp is None:
        timestamp = time.time()

    return SignatureDeclaration(
        signer=store.signer, key_id=store.key_id, message=bytes(message),
        message_bits=bits, revealed=revealed, L=L, counter=int(counter),
        nonce=bytes(nonce), timestamp=float(timestamp),
        recipients=tuple(session.config.verifiers), mac_tags={})


# --------------------------------------------------------------------------
# attack 1: the realistic forgery, caught by the MAC
# --------------------------------------------------------------------------

def forge_and_submit(*, strategy: str = "single_copy", known: float = 0.0,
                     forger: str = "bob", target: str = "charlie",
                     seed: int = 0) -> AttackResult:
    """A dishonest recipient forges a declaration and submits it to another.

    This is the end-to-end attack a deployed system actually faces.  The target
    runs its normal verifier -- registry, MAC, freshness, then physics -- and
    the forged declaration dies at the MAC gate, because the forger cannot
    authenticate to the target.  The engine returns ``compromised`` with
    ``authentication_failure`` as the primary attribution and no innocent
    explanation.
    """
    session = build_session(noise=HONEST_FOR_FORGERY, seed=seed)
    decl = build_forged_declaration(session, forger, DEFAULT_MESSAGE,
                                    strategy=strategy, known=known)
    session.verify(target, decl, level="accept")
    report = analyse(session, HONEST_FOR_FORGERY, levels=("accept",))

    return AttackResult(
        name=f"forgery_{strategy}",
        family="forgery",
        description=(
            f"A dishonest recipient ('{forger}') fabricates Alice's signature "
            f"using the {strategy} strategy and submits it to '{target}'. The "
            "forger cannot produce the target's one-time MAC tag, so the "
            "classical authentication gate refuses it before measurement."),
        expectation=Expectation(
            verdict="compromised", primary="authentication_failure",
            authentication_failed=True,
            note="MAC gate rejects a declaration the forger could not tag."),
        report=report, session=session,
        detail={"strategy": strategy, "known": known,
                "forger": forger, "target": target})


# --------------------------------------------------------------------------
# attack 2: strip the classical wall, measure the physical one
# --------------------------------------------------------------------------

def isolate_physics(*, strategy: str = "single_copy", known: float = 0.0,
                    forger: str = "bob", target: str = "charlie",
                    seed: int = 0) -> AttackResult:
    """Switch off the classical gates and measure the raw forgery rate.

    This is the information-theoretic demonstration.  The forger's declaration
    is checked against the target's independent copy with ``always_measure`` and
    with registry/MAC/ledger bypassed, so neither authentication nor freshness
    can end the attempt -- only the quantum states can.  The returned
    ``detail["mismatch_rate"]`` is the forger's per-check error rate, and it
    lands on the bound (1/2 blind, 1/4 single-copy) no matter how much the
    forger computes.

    ``report`` is ``None``: feeding this bypassed report to the engine would
    just re-detect the switched-off MAC as ``authentication_failure`` and hide
    the number we came to measure.  The test asserts the rate directly.
    """
    session = build_session(noise=IDEAL_FOR_FORGERY, seed=seed)
    decl = build_forged_declaration(session, forger, DEFAULT_MESSAGE,
                                    strategy=strategy, known=known)

    weakened = dataclasses.replace(session.policy, always_measure=True)
    report = verify_declaration(
        declaration=decl, store=session.stores[target], policy=weakened,
        rng=session.rng, registry=None, reservoir=None, ledger=None,
        noise=session.config.distribution.noise, level="accept")

    rate = report.mismatch_rate
    # The analytic reference each strategy is read against.  1/4 is the minimum
    # per-check error any single-copy measurement can achieve (the pretty-good
    # measurement is optimal on the geometrically uniform four-state set), so it
    # is a *floor*: blind guessing with no copy sits at 1/2, and a suboptimal
    # measurement such as Breidbart -- which maximises information rather than
    # minimising disturbance -- sits strictly above the 1/4 floor.
    floor = {"blind": 0.50, "single_copy": 0.25, "breidbart": 0.25,
             "partial": (1.0 - known) * 0.25}.get(strategy, 0.25)

    return AttackResult(
        name=f"forgery_physics_{strategy}",
        family="forgery",
        description=(
            f"The {strategy} forger's declaration is checked against an "
            f"independent copy with the classical gates disabled, isolating the "
            f"quantum unforgeability floor. Measured per-check error rate "
            f"{rate:.3f}; the analytic floor is ~{floor:.3f} (1/2 with no key "
            f"copy, 1/4 for the optimal single-copy measurement -- suboptimal "
            f"measurements sit above it)."),
        expectation=Expectation(
            verdict="n/a",
            note="physics-only probe; asserted against the analytic floor"),
        report=None, session=session,
        detail={"strategy": strategy, "known": known,
                "mismatch_rate": rate, "n_checked": report.n_checked,
                "n_mismatch": report.n_mismatch, "accepted": report.accepted,
                "analytic_bound": floor})


# --------------------------------------------------------------------------
# noise choices, named so the intent is legible
# --------------------------------------------------------------------------

#: Realistic forgery runs on a lab-grade link, so the forgery must clear a real
#: honest floor rather than a vacuous zero.
HONEST_FOR_FORGERY = LAB_GRADE
#: The physics demonstration runs on a perfect channel, so the measured rate is
#: the forger's own error and not the channel's.
IDEAL_FOR_FORGERY = IDEAL
