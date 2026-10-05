"""The QDS attack simulation suite.

Every attack the framework claims to stop, mounted as a *real* run of the
protocol against a hostile input and judged by the detection engine on the
unprivileged transcript -- never on the simulator's ground truth.  The families:

``forgery``
    A party without the private key tries to sign.  Caught classically by the
    one-time MAC, and -- with the MAC stripped away -- bounded physically at a
    1/4 per-check error by quantum unforgeability.
``impersonation``
    A declaration that is not from the signer it names.  Rejected
    deterministically at the identity / MAC gates before measurement.
``replay``
    A spent signature re-presented, or its freshness fields tampered with.
    Caught by the anti-replay ledger and the MAC that binds it.
``channel``
    Eve acting on the qubits in flight: intercept-resend, entanglement
    breaking, hidden depolarising, coherent probing, basis-biased probing,
    loss-blinding, dark-count injection.  Caught as excess error, lost yield,
    or basis asymmetry above the calibrated floor.
``repudiation``
    A dishonest signer (or a targeted link) builds unequal key copies so one
    recipient accepts what another rejects -- and the private key-symmetrisation
    step that defuses it.

Each entry point returns an :class:`~qds.attacks.harness.AttackResult` carrying
both the engine's finding and the attack's own :class:`Expectation`, produced by
different code paths so a test can check one against the other.  :func:`run_all`
mounts the canonical instance of every attack; :data:`CATALOGUE` maps
``family -> (variant -> callable)`` for the CLI and sweep runner.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from . import forgery, impersonation, manipulation, repudiation, replay
from .harness import (AttackResult, Expectation, analyse, build_session,
                      sign_honest, verify_round_trip)

__all__ = [
    "forgery", "impersonation", "replay", "manipulation", "repudiation",
    "AttackResult", "Expectation", "build_session", "sign_honest",
    "verify_round_trip", "analyse",
    "CATALOGUE", "FAMILIES", "run_all", "mount",
]

#: ``family -> {variant -> zero-arg-ish callable}``.  Every callable takes only
#: keyword arguments and returns an :class:`AttackResult`, so the CLI can invoke
#: any attack by two names without knowing its signature.
CATALOGUE: Dict[str, Dict[str, Callable[..., AttackResult]]] = {
    "forgery": {
        "blind": lambda **k: forgery.forge_and_submit(strategy="blind", **k),
        "single_copy":
            lambda **k: forgery.forge_and_submit(strategy="single_copy", **k),
        "breidbart":
            lambda **k: forgery.forge_and_submit(strategy="breidbart", **k),
        "partial":
            lambda **k: forgery.forge_and_submit(strategy="partial",
                                                 known=0.5, **k),
        "physics_blind":
            lambda **k: forgery.isolate_physics(strategy="blind", **k),
        "physics_single_copy":
            lambda **k: forgery.isolate_physics(strategy="single_copy", **k),
        "physics_breidbart":
            lambda **k: forgery.isolate_physics(strategy="breidbart", **k),
    },
    "impersonation": {v: (lambda v=v, **k: impersonation.mount(v, **k))
                      for v in impersonation.VARIANTS},
    "replay": {v: (lambda v=v, **k: replay.mount(v, **k))
               for v in replay.VARIANTS},
    "channel": {v: (lambda v=v, **k: manipulation.mount(v, **k))
                for v in manipulation.VARIANTS},
    "repudiation": {v: (lambda v=v, **k: repudiation.mount(v, **k))
                    for v in repudiation.VARIANTS},
}

#: The attack families, in a stable reporting order.
FAMILIES: List[str] = ["forgery", "impersonation", "replay", "channel",
                       "repudiation"]


def mount(family: str, variant: str, **kwargs) -> AttackResult:
    """Mount one attack by ``family`` and ``variant`` name."""
    try:
        fam = CATALOGUE[family]
    except KeyError as exc:
        raise ValueError(f"unknown attack family {family!r}; "
                         f"choose from {FAMILIES}") from exc
    try:
        fn = fam[variant]
    except KeyError as exc:
        raise ValueError(f"unknown {family} variant {variant!r}; "
                         f"choose from {sorted(fam)}") from exc
    return fn(**kwargs)


def run_all(*, seed: int = 0) -> List[AttackResult]:
    """Mount the canonical instance of every attack, in reporting order."""
    out: List[AttackResult] = []
    for family in FAMILIES:
        for variant in CATALOGUE[family]:
            out.append(CATALOGUE[family][variant](seed=seed))
    return out
