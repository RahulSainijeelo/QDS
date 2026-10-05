"""Shared scaffolding for the attack suite.

Every attack in this package is a *real* run of the protocol against a
deliberately hostile input, not a hand-written error rate.  The forger really
measures a stored qubit and names a guess; the impersonator really submits a
declaration with no valid tag; the replay really re-presents a spent counter;
the channel attacks really insert a :class:`~qds.channel.Intervention` into the
distribution phase.  The detection engine then sees exactly what a deployed
verifier would see -- the unprivileged half of the session, via
:class:`~qds.detect.Evidence` -- and returns a verdict.  Nothing here reaches
into the simulator's ground truth, and nothing hand-sets a statistic the engine
is supposed to compute.

Two properties make the resulting tests trustworthy rather than circular.

*The expectation and the evidence are produced by different code paths.*  An
:class:`Expectation` records what the attack is *designed* to trigger, in
plain English plus a verdict label.  The :class:`AttackResult` carries the
engine's *actual* finding.  The test file asserts the engine's finding against
hard-coded literals and, separately, against the attack's own expectation --
so a bug that moved both in lockstep would still have to move a literal in the
test, which no attack code can reach.

*The declared spec sheet is always supplied.*  Every analysis passes
``declared_noise=`` to :func:`~qds.detect.analyse_session`.  Without it the
loss and dark-count detectors read their reference off the very channel they
are judging and cannot fail (see ``project_qds_detect_done`` and
``Evidence.from_session``).  The channel-blocking and detector-blinding attacks
below depend on that reference being an *independent* honest datasheet, which
is exactly the situation a real verifier is in: he has a commissioning number,
not a view of today's fibre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..channel import IDEAL, LAB_GRADE, Intervention, NoIntervention, NoiseModel
from ..detect import analyse_session
from ..detect.engine import DetectionReport
from ..protocol.distribution import DistributionConfig
from ..protocol.session import QDSSession, SessionConfig
from ..protocol.signing import SignatureDeclaration, message_to_bits

__all__ = [
    "DEFAULT_MESSAGE", "MESSAGE_BITS", "L_REPS", "CHECK_PAIRS", "DECOY_SLOTS",
    "HONEST_LINK", "Expectation", "AttackResult", "build_session",
    "sign_honest", "verify_round_trip", "analyse",
]

# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------
# These are small on purpose.  A signature over 8 bits with L=24 gives
# ``8 * 24 = 192`` signature checks per verifier -- enough for a Clopper-Pearson
# interval tight enough to separate 1/4 from a few-percent honest floor, and
# enough to clear the ``min_checks`` gate several times over -- while keeping a
# whole session to roughly a second.  The security *claims* are sized by
# ``VerificationPolicy.required_checks`` and documented in the whitepaper; these
# numbers are sized for a test suite that has to run often.

DEFAULT_MESSAGE: bytes = b"\xb0"      #: one byte -> exactly ``MESSAGE_BITS`` bits
MESSAGE_BITS: int = 8
L_REPS: int = 24
CHECK_PAIRS: int = 32
DECOY_SLOTS: int = 32

#: The honest hardware every attack is judged against.  A lab-grade link has a
#: small but non-zero error floor, so an attack has to clear real noise rather
#: than a vacuous zero.  Forgery uses :data:`IDEAL` instead, to land the
#: information-theoretic bound without the floor in the way.
HONEST_LINK: NoiseModel = LAB_GRADE


# --------------------------------------------------------------------------
# what an attack is supposed to do
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Expectation:
    """What an attack is *designed* to make the engine conclude.

    This is the attack author's claim, carried alongside the run so the CLI
    and the dashboard can narrate "this is what should happen" next to what
    did.  The test suite compares it to the engine's finding, but never lets
    it stand in for a hard-coded assertion -- see the module docstring.
    """

    verdict: str                                  #: clean / suspicious / compromised
    primary: Optional[str] = None                 #: expected ``attributions[0].label``
    flags_include: Tuple[str, ...] = ()           #: detectors that must have fired
    flags_exclude: Tuple[str, ...] = ()           #: detectors that must *not* have fired
    authentication_failed: Optional[bool] = None  #: expected classical-gate failure
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "primary": self.primary,
            "flags_include": list(self.flags_include),
            "flags_exclude": list(self.flags_exclude),
            "authentication_failed": self.authentication_failed,
            "note": self.note,
        }


@dataclass
class AttackResult:
    """One executed attack and the engine's verdict on it.

    ``report`` is ``None`` only for the physics-isolation forgery probes, which
    bypass the classical layer to expose the raw mismatch rate and are asserted
    directly against the information-theoretic bound rather than through the
    engine (whose classical gates would otherwise fire first and mask it).
    """

    name: str
    family: str
    description: str
    expectation: Expectation
    report: Optional[DetectionReport] = None
    session: Any = None
    detail: Dict[str, Any] = field(default_factory=dict)

    # -- convenience views over the engine's finding ----------------------
    @property
    def verdict(self) -> Optional[str]:
        return self.report.verdict if self.report is not None else None

    @property
    def primary(self) -> Optional[str]:
        if self.report is None or not self.report.attributions:
            return None
        return self.report.attributions[0].label

    @property
    def labels(self) -> List[str]:
        if self.report is None:
            return []
        return [a.label for a in self.report.attributions]

    @property
    def flagged(self) -> List[str]:
        return list(self.report.flagged) if self.report is not None else []

    @property
    def authentication_failed(self) -> Optional[bool]:
        if self.report is None:
            return None
        return bool(self.report.evidence.authentication_failed)

    # -- self-check: does the engine's finding match the design? ----------
    def discrepancies(self) -> List[str]:
        """Empty iff the engine agreed with the attack's :class:`Expectation`.

        Used by the CLI's self-check and by the test suite as a second,
        coarser guard behind the hard-coded literal assertions.
        """
        e = self.expectation
        out: List[str] = []
        if e.verdict == "n/a":
            # a physics-isolation probe: deliberately not judged by the engine,
            # asserted directly against the analytic bound in the test suite.
            if self.report is not None:
                out.append("expected no engine report for an 'n/a' probe")
            return out
        if self.report is None:
            return ["no engine report (physics-isolation probe)"]
        if self.verdict != e.verdict:
            out.append(f"verdict {self.verdict!r} != expected {e.verdict!r}")
        if e.primary is not None and self.primary != e.primary:
            out.append(f"primary {self.primary!r} != expected {e.primary!r}")
        fired = set(self.flagged)
        for name in e.flags_include:
            if name not in fired:
                out.append(f"expected detector {name!r} to fire; it did not")
        for name in e.flags_exclude:
            if name in fired:
                out.append(f"detector {name!r} fired but should not have")
        if e.authentication_failed is not None \
                and self.authentication_failed != e.authentication_failed:
            out.append(
                f"authentication_failed={self.authentication_failed} "
                f"!= expected {e.authentication_failed}")
        return out

    @property
    def matched(self) -> bool:
        return not self.discrepancies()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "expectation": self.expectation.to_dict(),
            "detail": dict(self.detail),
        }
        if self.report is not None:
            est = self.report.evidence.estimates.get("pooled_rate")
            d["finding"] = {
                "verdict": self.report.verdict,
                "alarm": bool(self.report.alarm),
                "flagged": list(self.report.flagged),
                "attribution": [a.label for a in self.report.attributions],
                "authentication_failed":
                    bool(self.report.evidence.authentication_failed),
                "protocol_rejects":
                    bool(self.report.evidence.protocol_rejects),
                "pooled_rate": (None if est is None else est.value),
                "summary": self.report.summary(),
            }
            d["matched"] = self.matched
            d["discrepancies"] = self.discrepancies()
        return d


# --------------------------------------------------------------------------
# building and running a session
# --------------------------------------------------------------------------

def build_session(*, noise: NoiseModel = HONEST_LINK,
                   intervention: Optional[Intervention] = None,
                   message_bits: int = MESSAGE_BITS, L: int = L_REPS,
                   check_pairs: int = CHECK_PAIRS, decoy_slots: int = DECOY_SLOTS,
                   symmetrise: bool = True,
                   verifiers: Sequence[str] = ("bob", "charlie"),
                   spec: Optional[float] = None,
                   seed: int = 0) -> QDSSession:
    """Run a session through calibration, ready to be signed and verified.

    ``noise`` is the honest hardware.  ``intervention`` is the attack inserted
    into the distribution phase -- it is kept *separate* from ``noise`` so the
    acceptance threshold is built on the honest floor and the attack must clear
    it.  ``spec`` pins the declared error floor explicitly; left ``None`` the
    session reads it off ``noise`` (the honest model), which is what we want
    when the attack lives in the intervention or in a loss term rather than in
    the Pauli noise.
    """
    dist = DistributionConfig(
        noise=noise,
        intervention=intervention if intervention is not None else NoIntervention(),
        check_pairs=check_pairs, decoy_slots=decoy_slots)
    cfg = SessionConfig(
        signer="alice", verifiers=tuple(verifiers), message_bits=message_bits,
        L=L, distribution=dist, symmetrise=symmetrise, noise_floor_spec=spec)
    session = QDSSession(config=cfg, seed=seed)
    session.enrol()
    session.distribute()
    session.symmetrise()
    session.calibrate()
    return session


def sign_honest(session: QDSSession, message: bytes = DEFAULT_MESSAGE,
                **kwargs: Any) -> SignatureDeclaration:
    """Alice signs ``message`` honestly.  ``kwargs`` go to :meth:`sign`.

    ``authenticate=False`` is the hook the impersonation family uses; replay
    uses ``timestamp=`` to backdate; everything else leaves the defaults.
    """
    return session.sign(message, **kwargs)


def verify_round_trip(session: QDSSession,
                      declaration: Optional[SignatureDeclaration] = None,
                      now: Optional[float] = None) -> None:
    """Have Bob accept and forward to Charlie, exactly as :meth:`run` does.

    Populates ``session.reports`` with ``bob:accept`` and ``charlie:transfer``
    so the recipient-agreement detector has two independent views to compare.
    """
    verifiers = session.config.verifiers
    holder = verifiers[0]
    session.verify(holder, declaration, level="accept", now=now)
    for name in verifiers[1:]:
        session.transfer(holder, name, declaration, now=now)


def analyse(session: QDSSession, declared: NoiseModel,
            **kwargs: Any) -> DetectionReport:
    """Run the detection engine on a session against an honest spec sheet.

    ``declared`` is mandatory here on purpose: it is the independent datasheet
    the loss and dark-count tests need to be capable of failing.
    """
    return analyse_session(session, declared_noise=declared, **kwargs)
