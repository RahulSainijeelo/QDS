"""User impersonation: a declaration that is not from the signer it claims.

Impersonation resistance in this scheme is deterministic, not statistical, and
it is settled before any qubit is measured.  A declaration carries three
classical bindings -- the signer's enrolled identity, a one-time MAC tag per
recipient, and the recipient list -- and each is checked at a gate that rejects
with probability one.  There is nothing to be statistically uncertain about: a
tag either verifies under the shared one-time key or it does not, and a signer
is either in the verifier's registry or is not.

That is why every attack in this module ends the same way -- ``compromised``
with ``authentication_failure`` as the primary attribution and
``authentication_failed=True`` -- and why that verdict has, in the engine's own
words, "no innocent explanation short of operator error".  Honest noise cannot
flip a MAC tag or invent a registry entry.  The four variants differ only in
*which* gate does the rejecting, which is worth exercising separately because
the engine records the distinction and an operator triaging an alert needs it:

``unauthenticated``
    The declaration carries no tag at all (``authenticate=False``).  This is
    the naive impersonator who simply omits what he cannot compute.  The MAC
    gate rejects on a missing tag.
``tampered_payload``
    A genuine declaration whose revealed key material was altered in transit by
    a dishonest forwarder.  The tag was valid for the original bytes and cannot
    be recomputed, so it fails to authenticate the altered payload -- the case
    the MAC exists to stop.
``unenrolled_signer``
    A declaration naming a signer the verifier never enrolled.  The identity
    gate rejects: there is no public key on file to check it against.
``wrong_key_id``
    A declaration naming an enrolled signer but a different ``key_id`` -- an
    attacker substituting his own key under a trusted name.  The identity gate
    rejects on the mismatch.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ..protocol.signing import SignatureDeclaration
from .harness import (DEFAULT_MESSAGE, HONEST_LINK, AttackResult, Expectation,
                      analyse, build_session, sign_honest)

__all__ = [
    "VARIANTS", "unauthenticated", "tampered_payload", "unenrolled_signer",
    "wrong_key_id", "mount",
]

VARIANTS = ("unauthenticated", "tampered_payload", "unenrolled_signer",
            "wrong_key_id")

_EXPECT = Expectation(
    verdict="compromised", primary="authentication_failure",
    authentication_failed=True,
    note="a classical binding gate rejects before any measurement")


def _finish(session, target: str, decl: SignatureDeclaration, name: str,
            description: str) -> AttackResult:
    session.verify(target, decl, level="accept")
    report = analyse(session, HONEST_LINK, levels=("accept",))
    return AttackResult(name=name, family="impersonation",
                        description=description, expectation=_EXPECT,
                        report=report, session=session,
                        detail={"target": target})


def unauthenticated(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Submit a declaration with no MAC tag at all."""
    session = build_session(noise=HONEST_LINK, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE, authenticate=False)
    return _finish(session, target, decl, "impersonation_unauthenticated",
                   "An impersonator submits a declaration carrying no MAC tag; "
                   "the authentication gate rejects a missing tag outright.")


def tampered_payload(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Alter the revealed key material after signing, keeping the stale tag."""
    session = build_session(noise=HONEST_LINK, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE)
    # a dishonest forwarder flips one revealed entry's bit; the tag was computed
    # over the original bytes and cannot be recomputed without Alice's key
    revealed = dict(decl.revealed)
    j0 = sorted(revealed)[0]
    block = list(revealed[j0])
    first = block[0]
    block[0] = type(first)(first.basis, first.bit ^ 1)
    revealed[j0] = tuple(block)
    tampered = dataclasses.replace(decl, revealed=revealed)
    return _finish(session, target, tampered, "impersonation_tampered_payload",
                   "A dishonest forwarder rewrites a revealed key entry; the "
                   "one-time tag no longer authenticates the altered payload.")


def unenrolled_signer(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Name a signer the verifier has no enrolment record for."""
    session = build_session(noise=HONEST_LINK, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE)
    forged = dataclasses.replace(decl, signer="mallory")
    return _finish(session, target, forged, "impersonation_unenrolled_signer",
                   "A declaration names an unenrolled signer 'mallory'; the "
                   "identity gate has no public key on file and rejects.")


def wrong_key_id(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Name an enrolled signer but a key_id the verifier did not enrol."""
    session = build_session(noise=HONEST_LINK, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE)
    forged = dataclasses.replace(decl, key_id="pk-00000000")
    return _finish(session, target, forged, "impersonation_wrong_key_id",
                   "A declaration cites Alice's name but an attacker's key_id; "
                   "the identity gate rejects the key_id mismatch.")


def mount(variant: str, **kwargs) -> AttackResult:
    """Dispatch by name, for the CLI and the sweep runner."""
    fns = {"unauthenticated": unauthenticated,
           "tampered_payload": tampered_payload,
           "unenrolled_signer": unenrolled_signer,
           "wrong_key_id": wrong_key_id}
    if variant not in fns:
        raise ValueError(f"unknown impersonation variant {variant!r}; "
                         f"choose from {VARIANTS}")
    return fns[variant](**kwargs)
