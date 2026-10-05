"""Channel manipulation: Eve acts on the quantum link itself.

These are the attacks the scheme exists to catch -- an adversary touching the
qubits in flight.  The security claim is physical, not computational: Eve cannot
extract information from a travelling EPR half without disturbing it, and the
disturbance shows up as excess error, lost yield, or basis asymmetry on top of
the honest floor fixed at calibration.  Each attack here inserts a real
:class:`~qds.channel.Intervention` into the distribution phase (or degrades the
honest :class:`~qds.channel.NoiseModel` for the loss-based ones), runs the whole
protocol, and lets the detection engine judge the unprivileged transcript.

The honest reference is always the independent datasheet ``LAB_GRADE``, passed
to the engine as ``declared_noise``.  That is the real verifier's situation --
he owns a commissioning number, not a live view of today's fibre -- and it is
what lets the loss and dark-count tests be *capable* of failing instead of
reading their own reference off the channel under attack.

The variants span the disturbance menu:

``intercept_resend``
    The textbook attack: measure every flying qubit and resend the outcome.
    Full classical information about one basis, total loss of entanglement, and
    a ~1/4 error rate -- over the acceptance threshold, so the signature is
    refused outright.
``entanglement_breaking``
    Discard the EPR half and inject an uncorrelated qubit.  Integrity is gone
    even though a qubit still arrives.
``collective_depolarizing``
    The subtle adversary who hides beneath the noise floor with a weak
    symmetric probe.  The point is not to name the attack but that the induced
    disturbance is still bounded and visible when pushed hard enough.
``coherent_probe``
    The information-versus-disturbance trade-off made concrete: entangle a
    stored ancilla and read it only after the key is public.
``basis_biased``
    Damages one basis far more than the other, keeping the *pooled* rate under
    threshold -- which is exactly why the engine tests each basis separately.
``loss_blinding``
    Suppress transmission (channel blocking) so the detector yield collapses
    below the declared loss budget, while the errors that do get through stay in
    spec.  The yield detector catches what the rate detector cannot.
``dark_injection``
    Flood the detector with dark counts, the mechanism a blinding attack uses to
    fake data.  The dark-count premise detector sees the registered-click budget
    blow past the datasheet.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ..channel import LAB_GRADE, LossModel, NoiseModel, make_intervention
from .harness import (CHECK_PAIRS, DECOY_SLOTS, AttackResult, Expectation,
                      analyse, build_session, sign_honest, verify_round_trip)

__all__ = [
    "VARIANTS", "intercept_resend", "entanglement_breaking",
    "collective_depolarizing", "coherent_probe", "basis_biased",
    "loss_blinding", "dark_injection", "mount",
]

VARIANTS = ("intercept_resend", "entanglement_breaking",
            "collective_depolarizing", "coherent_probe", "basis_biased",
            "loss_blinding", "dark_injection")

#: The honest datasheet every channel attack is declared against.
DECLARED: NoiseModel = LAB_GRADE


def _run(session, name: str, description: str, expectation: Expectation,
         detail: Optional[dict] = None) -> AttackResult:
    """Sign honestly, verify round-trip, and analyse against the datasheet."""
    sign_honest(session)
    verify_round_trip(session)
    report = analyse(session, DECLARED, levels=("accept", "transfer"))
    return AttackResult(name=name, family="channel", description=description,
                        expectation=expectation, report=report, session=session,
                        detail=detail or {})


def intercept_resend(*, strength: float = 1.0, seed: int = 0) -> AttackResult:
    """Measure-and-resend on every flying qubit."""
    session = build_session(
        noise=LAB_GRADE,
        intervention=make_intervention("intercept_resend", basis="random",
                                       strength=strength), seed=seed)
    return _run(session, "channel_intercept_resend",
                "Eve intercepts, measures in a random basis, and resends every "
                "flying qubit. Entanglement is destroyed and the per-check error "
                "climbs to about one quarter -- past the acceptance threshold, so "
                "the signature is refused.",
                Expectation(verdict="compromised",
                            primary="signature_rejected_on_rate",
                            flags_include=("pooled_rate_vs_spec",),
                            note="full intercept-resend drives QBER to ~1/4"),
                {"strength": strength})


def entanglement_breaking(*, strength: float = 1.0,
                          seed: int = 0) -> AttackResult:
    """Replace the EPR half with an uncorrelated mixed qubit."""
    session = build_session(
        noise=LAB_GRADE,
        intervention=make_intervention("entanglement_breaking",
                                       strength=strength, substitute="mixed"),
        seed=seed)
    return _run(session, "channel_entanglement_breaking",
                "Eve discards the transmitted EPR half and injects an "
                "uncorrelated mixed qubit. A qubit still arrives, so a naive "
                "protocol proceeds, but the correlation that carries the key is "
                "gone and the error rate explodes.",
                Expectation(verdict="compromised",
                            flags_include=("pooled_rate_vs_spec",),
                            note="substituted qubit is uncorrelated with the key"),
                {"strength": strength})


def collective_depolarizing(*, p: float = 0.08, seed: int = 0) -> AttackResult:
    """A weak symmetric probe that tries to hide under the noise floor."""
    session = build_session(
        noise=LAB_GRADE,
        intervention=make_intervention("collective_depolarizing", p=p),
        seed=seed)
    return _run(session, "channel_collective_depolarizing",
                f"Eve applies a symmetric depolarising probe (p={p:.3f}) chosen "
                "to imitate honest channel noise. Pushed to this strength the "
                "excess disturbance clears the calibrated floor and is caught, "
                "even though the attack is designed to look like degradation.",
                Expectation(verdict="compromised",
                            flags_include=("pooled_rate_vs_spec",),
                            note="disturbance exceeds the calibrated floor"),
                {"p": p})


def coherent_probe(*, theta: float = 0.5, seed: int = 0) -> AttackResult:
    """Entangle an ancilla with the flying qubit; read it after key reveal."""
    session = build_session(
        noise=LAB_GRADE,
        intervention=make_intervention("coherent_probe", theta=theta,
                                       delayed_measurement=True), seed=seed)
    return _run(session, "channel_coherent_probe",
                f"Eve entangles a stored ancilla with each flying qubit "
                f"(theta={theta:.3f}) and measures it only after the private key "
                "is revealed. The information she gains is paid for in "
                "disturbance on the complementary basis, which the engine sees.",
                Expectation(verdict="compromised",
                            flags_include=("pooled_rate_vs_spec",),
                            note="information gain is paid for in disturbance"),
                {"theta": theta})


def basis_biased(*, px: float = 0.03, pz: float = 0.0, L: int = 48,
                 seed: int = 0) -> AttackResult:
    """Damage one basis just enough to slip the pooled rate under threshold.

    The strength is deliberately mild and the key length doubled: at this level
    the Z-basis error clears its own per-basis spec but the *pooled* rate stays
    statistically consistent with the declared floor, so the signature is
    accepted and only the per-basis test objects.  That is the regime the
    separate-basis detector exists for, and it is where the pooled-only view is
    blind.
    """
    session = build_session(
        noise=LAB_GRADE,
        intervention=make_intervention("basis_biased", px=px, pz=pz),
        L=L, seed=seed)
    return _run(session, "channel_basis_biased",
                f"Eve runs a mild asymmetric probe (p_X={px:.3f}, p_Z={pz:.3f}) "
                "that loads its disturbance onto one basis. The pooled error "
                "rate stays under the acceptance threshold, so a single-number "
                "check passes the signature -- but the engine's separate "
                "per-basis test sees the loaded basis and raises a flag.",
                Expectation(verdict="suspicious", primary="basis_biased_probe",
                            flags_include=("basis_consistency",),
                            flags_exclude=("pooled_rate_vs_spec",),
                            note="asymmetry hides in the pool, shows per-basis"),
                {"px": px, "pz": pz, "L": L})


def loss_blinding(*, transmittance: float = 0.5, seed: int = 0) -> AttackResult:
    """Suppress transmission so the yield falls below the declared budget."""
    lossy = dataclasses.replace(
        LAB_GRADE,
        loss=dataclasses.replace(LAB_GRADE.loss, transmittance=transmittance))
    # pin the acceptance floor to the honest datasheet's QBER so the protocol
    # judges errors against the commissioning spec, not the blocked channel;
    # widen the sampling so enough rounds survive the loss to clear min_checks.
    session = build_session(noise=lossy, spec=LAB_GRADE.predicted_qber(),
                            check_pairs=4 * CHECK_PAIRS,
                            decoy_slots=4 * DECOY_SLOTS, seed=seed)
    return _run(session, "channel_loss_blinding",
                f"Eve blocks the channel (transmittance {transmittance:.2f} "
                "against a declared 0.92), measuring and dropping rounds rather "
                "than resending disturbed ones. The errors that get through stay "
                "in spec, but the detector yield collapses below the declared "
                "loss budget and the yield detector catches it.",
                Expectation(verdict="compromised",
                            primary="channel_blocking_or_blinding",
                            flags_include=("yield_vs_declared",),
                            flags_exclude=("pooled_rate_vs_spec",),
                            note="yield departs the loss budget; rate stays in spec"),
                {"transmittance": transmittance,
                 "declared_transmittance": LAB_GRADE.loss.transmittance})


def dark_injection(*, dark_count: float = 0.5, seed: int = 0) -> AttackResult:
    """Flood the detector with dark counts to fake data."""
    darky = dataclasses.replace(
        LAB_GRADE,
        loss=dataclasses.replace(LAB_GRADE.loss, dark_count=dark_count))
    session = build_session(noise=darky, spec=LAB_GRADE.predicted_qber(),
                            check_pairs=4 * CHECK_PAIRS,
                            decoy_slots=4 * DECOY_SLOTS, seed=seed)
    return _run(session, "channel_dark_injection",
                f"Eve floods the detector with dark counts (rate {dark_count:.2f} "
                "against a declared 1e-5), the mechanism a blinding attack uses "
                "to manufacture clicks with no qubit behind them. The registered "
                "detections blow past the datasheet's dark budget and the "
                "fraction of them that are random coins inflates the error.",
                Expectation(verdict="compromised",
                            flags_include=("dark_excess",),
                            note="dark-count budget exceeded; a premise is broken"),
                {"dark_count": dark_count,
                 "declared_dark_count": LAB_GRADE.loss.dark_count})


def mount(variant: str, **kwargs) -> AttackResult:
    """Dispatch by name, for the CLI and the sweep runner."""
    fns = {"intercept_resend": intercept_resend,
           "entanglement_breaking": entanglement_breaking,
           "collective_depolarizing": collective_depolarizing,
           "coherent_probe": coherent_probe,
           "basis_biased": basis_biased,
           "loss_blinding": loss_blinding,
           "dark_injection": dark_injection}
    if variant not in fns:
        raise ValueError(f"unknown channel variant {variant!r}; "
                         f"choose from {VARIANTS}")
    return fns[variant](**kwargs)
