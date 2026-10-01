"""Repudiation and targeted-link attacks, and the symmetrisation that stops them.

Repudiation is the attack unique to *signatures*: not an outsider forging a
name, but the legitimate signer later disowning something she really signed.
In a teleportation QDS scheme the lever is key asymmetry.  A dishonest Alice --
or an eavesdropper who attacks one recipient's link and not another's -- can
arrange for Bob to hold a clean key copy and Charlie a dirty one.  Then the
very declaration Bob accepts is one Charlie's higher error rate makes him
reject, and Alice can repudiate: "Charlie's copy never verified."  No
statistics are needed; the split is engineered.

The detector that sees this is ``recipient_agreement``: two recipients of the
same declaration hold independently symmetrised halves of one key and should
measure the same error rate up to sampling, so a real split is evidence of a
targeted link or an unequal key.  It is a *premise* detector -- if the two
copies are not equivalent, the repudiation bound was derived for a situation
that does not hold, and "the rate is under threshold" stops being reassuring.

The second half of this module is the countermeasure, and it is the more
important demonstration.  Before any message exists, the recipients privately
permute their stored signature qubits among themselves over a channel Alice
cannot see (:func:`~qds.protocol.session.symmetrise_keys`).  Whatever asymmetry
was built in is redistributed uniformly, so the engineered split disappears and
the equal-rate premise becomes *true* rather than assumed.  The two functions
here mount the identical attack with symmetrisation off and on, so the test
suite can show the same dirt producing a repudiation signature in one case and
no ``recipient_agreement`` flag in the other -- the countermeasure earning its
place.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..channel import LAB_GRADE, NoiseModel
from ..detect import analyse_session
from ..detect.engine import DetectionReport
from ..linalg import apply_depolarizing
from ..protocol.distribution import DistributionConfig
from ..protocol.session import QDSSession, SessionConfig
from .harness import (CHECK_PAIRS, DECOY_SLOTS, DEFAULT_MESSAGE, L_REPS,
                      MESSAGE_BITS, AttackResult, Expectation)

__all__ = [
    "VARIANTS", "DIRT", "targeted_link_attack", "symmetrisation_defuses",
    "mount",
]

VARIANTS = ("targeted_link_attack", "symmetrisation_defuses")

#: Depolarising strength injected into one recipient's signature qubits.  Tuned
#: against the measured thresholds: it must lift the victim's error rate far
#: enough above the other recipient's that the agreement test sees a
#: statistically significant split (Fisher exact under a Bonferroni budget),
#: yet keep the victim's rate under the *transfer* threshold (s_v ~= 0.169) so
#: the signature is not simply rejected on rate -- a rate rejection would be
#: attributed to the rate and mask the repudiation shape.  At p=0.24 the victim
#: lands near 0.13, comfortably below s_v and clearly above the clean link.
DIRT: float = 0.24


def _inject_dirt(session: QDSSession, victim: str, p: float) -> int:
    """Depolarise ``victim``'s delivered signature qubits by strength ``p``.

    Targets only the one-qubit signature slots (``j >= 0``); check pairs and
    decoys are public and are never moved or dirtied.  Returns how many slots
    were touched, for the detail record.  Mutating ``held.rho`` in place models
    an adversary acting on that recipient's link alone.
    """
    store = session.stores[victim]
    touched = 0
    for slot, held in store.qubits.items():
        if slot.j >= 0 and held.n_qubits == 1:
            held.rho = apply_depolarizing(held.rho, 0, p)
            touched += 1
    return touched


def _build(symmetrise: bool, victim: str, p: float, seed: int) -> tuple:
    """Run the phases by hand so dirt lands between distribute and symmetrise."""
    dist = DistributionConfig(noise=LAB_GRADE, check_pairs=CHECK_PAIRS,
                              decoy_slots=DECOY_SLOTS)
    cfg = SessionConfig(signer="alice", verifiers=("bob", "charlie"),
                        message_bits=MESSAGE_BITS, L=L_REPS, distribution=dist,
                        symmetrise=symmetrise)
    session = QDSSession(config=cfg, seed=seed)
    session.enrol()
    session.distribute()
    touched = _inject_dirt(session, victim, p)     # Eve attacks one link
    session.symmetrise()                           # countermeasure (or not)
    session.calibrate()
    return session, touched


def _finish(session, name, description, expectation, detail) -> AttackResult:
    session.sign(DEFAULT_MESSAGE)
    verifiers = session.config.verifiers
    session.verify(verifiers[0], level="accept")
    for other in verifiers[1:]:
        session.transfer(verifiers[0], other)
    report = analyse_session(session, declared_noise=LAB_GRADE,
                             levels=("accept", "transfer"))
    # record each recipient's measured error rate so the split (and its
    # collapse under symmetrisation) is visible in the result, not just the flag
    by_verifier = {}
    for name_v, est in report.evidence.estimates_by_verifier.items():
        by_verifier[name_v] = {"errors": est.k, "checks": est.n,
                               "rate": (est.k / est.n) if est.n else None}
    detail = dict(detail, rates_by_verifier=by_verifier)
    return AttackResult(name=name, family="repudiation", description=description,
                        expectation=expectation, report=report, session=session,
                        detail=detail)


def targeted_link_attack(*, victim: str = "charlie", p: float = DIRT,
                         seed: int = 0) -> AttackResult:
    """One recipient's link is dirtied and symmetrisation is disabled."""
    session, touched = _build(symmetrise=False, victim=victim, p=p, seed=seed)
    return _finish(
        session, "repudiation_targeted_link",
        f"Symmetrisation is disabled and Eve depolarises '{victim}''s signature "
        f"qubits (p={p:.3f}). That recipient's error rate pulls away from the "
        "other's, so the same declaration one accepts the other would reject -- "
        "the split a repudiating signer relies on. The agreement detector sees "
        "it.",
        Expectation(verdict="compromised",
                    primary="repudiation_or_targeted_link_attack",
                    flags_include=("recipient_agreement",),
                    note="asymmetric key copies; premise of the bound is broken"),
        {"victim": victim, "p": p, "slots_dirtied": touched,
         "symmetrised": False})


def symmetrisation_defuses(*, victim: str = "charlie", p: float = DIRT,
                           seed: int = 0) -> AttackResult:
    """The identical attack, with symmetrisation left on to redistribute it."""
    session, touched = _build(symmetrise=True, victim=victim, p=p, seed=seed)
    return _finish(
        session, "repudiation_symmetrised",
        f"The same injection into '{victim}''s qubits (p={p:.3f}), but the "
        "recipients first permute their signature qubits privately. The dirt is "
        "redistributed across both parties: their error rates converge, so the "
        "agreement detector no longer sees a split and the repudiation lever is "
        "gone. The disturbance itself does not vanish -- both recipients now "
        "carry an equal share, so the signature is still caught on its pooled "
        "rate. Symmetrisation defeats the repudiation, not the eavesdropper.",
        Expectation(verdict="compromised",
                    flags_exclude=("recipient_agreement",),
                    note="redistribution equalises the recipients; the split "
                         "(repudiation lever) is gone though the dirt remains"),
        {"victim": victim, "p": p, "slots_dirtied": touched,
         "symmetrised": True})


def mount(variant: str, **kwargs) -> AttackResult:
    """Dispatch by name, for the CLI and the sweep runner."""
    fns = {"targeted_link_attack": targeted_link_attack,
           "symmetrisation_defuses": symmetrisation_defuses}
    if variant not in fns:
        raise ValueError(f"unknown repudiation variant {variant!r}; "
                         f"choose from {VARIANTS}")
    return fns[variant](**kwargs)
