"""Replay: re-presenting a signature, or tampering with its freshness fields.

A quantum signature proves authorship but says nothing about *when*, so a
captured declaration could otherwise be injected again later -- or a spent
one-time key could be dressed up as a new signing.  Three classical bindings
close that gap, all authenticated by the same one-time MAC and all checked by
:class:`~qds.protocol.auth.FreshnessLedger` before measurement: a strictly
increasing ``counter``, a never-before-seen ``nonce``, and a ``timestamp`` that
must fall inside an acceptance window.  The ledger additionally records which
public-key slots have been measured, because a one-time key is physically spent
by verification and must never be checked twice.

The attacks here show each binding doing its job, and -- importantly -- show
that the freshness fields cannot be *edited* to dodge the ledger, because the
MAC covers them.  An attacker who refreshes the nonce to defeat the
seen-before check finds he has invalidated the tag instead.  That interlock is
the point: freshness and authentication are not two separate hurdles an
attacker clears one at a time, they are welded together.

``exact_replay``
    A declaration accepted once is presented again verbatim.  The counter no
    longer increases, the nonce has been seen, and the slots are spent; the
    ledger rejects on all three.
``fresh_nonce``
    The replay carries a new nonce to beat the seen-before check.  The MAC was
    computed over the old nonce, so it no longer authenticates -- the tamper is
    caught by the authentication gate, not the freshness one.
``stale_timestamp`` / ``future_timestamp``
    A declaration whose timestamp sits outside the acceptance window, on either
    side.  Rejected on freshness while the tag itself is perfectly valid --
    which is exactly how a captured-and-held declaration presents.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ..channel import IDEAL
from .harness import (DEFAULT_MESSAGE, AttackResult, Expectation, analyse,
                      build_session, sign_honest, verify_round_trip)

__all__ = [
    "VARIANTS", "CLOCK", "exact_replay", "fresh_nonce", "stale_timestamp",
    "future_timestamp", "mount",
]

VARIANTS = ("exact_replay", "fresh_nonce", "stale_timestamp",
            "future_timestamp")

#: A fixed clock epoch so the freshness window behaves deterministically in
#: tests rather than depending on the wall clock.
CLOCK: float = 1_000_000.0

_EXPECT = Expectation(
    verdict="compromised", primary="authentication_failure",
    authentication_failed=True,
    note="freshness or the MAC that binds it rejects before measurement")


def exact_replay(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Accept a declaration once, then present the identical bytes again."""
    session = build_session(noise=IDEAL, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE, timestamp=CLOCK)
    # first presentation: accepted, and the ledger burns counter/nonce/slots
    first = session.verify(target, decl, level="accept", now=CLOCK)
    # second presentation: identical, so the ledger rejects it
    second = session.verify(target, decl, level="accept", now=CLOCK)
    report = analyse(session, IDEAL, levels=("accept",))
    return AttackResult(
        name="replay_exact", family="replay",
        description=(
            "A declaration Bob already accepted is replayed verbatim. The "
            "freshness ledger rejects it: the counter no longer increases, the "
            "nonce has been seen, and the public-key slots are already spent."),
        expectation=_EXPECT, report=report, session=session,
        detail={"target": target,
                "first_accepted": bool(first.accepted),
                "second_accepted": bool(second.accepted),
                "freshness_reasons":
                    list(second.freshness.reasons) if second.freshness else []})


def fresh_nonce(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Replay with a freshened nonce; the MAC was over the old one."""
    session = build_session(noise=IDEAL, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE, timestamp=CLOCK)
    session.verify(target, decl, level="accept", now=CLOCK)
    replayed = dataclasses.replace(decl, nonce=b"\xff" * 16)
    second = session.verify(target, replayed, level="accept", now=CLOCK)
    report = analyse(session, IDEAL, levels=("accept",))
    return AttackResult(
        name="replay_fresh_nonce", family="replay",
        description=(
            "An attacker swaps in a new nonce to beat the seen-before check. "
            "The one-time MAC was computed over the original nonce, so the "
            "altered declaration fails authentication instead."),
        expectation=_EXPECT, report=report, session=session,
        detail={"target": target, "mac_ok": bool(second.mac_ok)})


def stale_timestamp(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """Present a validly-signed declaration long after its timestamp."""
    session = build_session(noise=IDEAL, seed=seed)
    decl = sign_honest(session, DEFAULT_MESSAGE, timestamp=CLOCK)
    window = session.policy.window_seconds
    now = CLOCK + 10.0 * window              # far past the acceptance window
    session.verify(target, decl, level="accept", now=now)
    report = analyse(session, IDEAL, levels=("accept",))
    return AttackResult(
        name="replay_stale_timestamp", family="replay",
        description=(
            "A captured declaration is injected long after it was signed. Its "
            "MAC is still valid, but the timestamp falls outside the freshness "
            "window, so it is rejected as not live."),
        expectation=_EXPECT, report=report, session=session,
        detail={"target": target, "now_minus_timestamp": now - CLOCK})


def future_timestamp(*, target: str = "bob", seed: int = 0) -> AttackResult:
    """A declaration timestamped beyond the window in the future."""
    session = build_session(noise=IDEAL, seed=seed)
    window = session.policy.window_seconds
    decl = sign_honest(session, DEFAULT_MESSAGE, timestamp=CLOCK + 10.0 * window)
    session.verify(target, decl, level="accept", now=CLOCK)
    report = analyse(session, IDEAL, levels=("accept",))
    return AttackResult(
        name="replay_future_timestamp", family="replay",
        description=(
            "A declaration carries a timestamp well in the future, as a "
            "pre-dated injection would. It is outside the acceptance window "
            "and rejected on freshness."),
        expectation=_EXPECT, report=report, session=session,
        detail={"target": target, "timestamp_minus_now": 10.0 * window})


def mount(variant: str, **kwargs) -> AttackResult:
    """Dispatch by name, for the CLI and the sweep runner."""
    fns = {"exact_replay": exact_replay, "fresh_nonce": fresh_nonce,
           "stale_timestamp": stale_timestamp,
           "future_timestamp": future_timestamp}
    if variant not in fns:
        raise ValueError(f"unknown replay variant {variant!r}; "
                         f"choose from {VARIANTS}")
    return fns[variant](**kwargs)
