"""Bell-state entanglement distribution and teleported public-key delivery.

This is the quantum half of the protocol: how a recipient comes to *hold* a
quantum public key without ever learning what it is.

Why teleportation and not direct transmission
---------------------------------------------
Alice could mail the BB84 states down the fibre.  She teleports them instead,
for three reasons that matter to a security argument rather than to
convenience:

1. **The classical side channel is provably empty.**  Teleportation announces
   two bits ``(u, v)``.  Their distribution is uniform on ``{0,1}^2`` for
   *every* input state -- see :func:`bell_outcome_distribution`, which computes
   it exactly, and ``tests/test_protocol.py``, which asserts it.  So an
   adversary who records the entire public classical transcript of the
   distribution phase learns exactly zero bits of the private key.  Direct
   transmission has no such transcript to reason about; teleportation turns
   "the channel leaks nothing" into an arithmetic identity.

2. **Entanglement is testable, a transmitted qubit is not.**  Alice and each
   recipient sacrifice a fraction of their pairs and measure them in
   :data:`~qds.linalg.CHSH_SETTINGS`.  A CHSH value above the local bound of 2
   certifies that the link really did carry entanglement; nothing an
   eavesdropper does to the flying qubit can fake it, because a separable
   state cannot exceed 2.  A directly transmitted unknown qubit offers no
   comparable handle.

3. **The key is created where it is needed.**  Alice's BB84 state never enters
   the channel at all, so channel loss cannot be used to *select* which key
   slots arrive: the qubit that is lost is an EPR half, which carries no key
   information.

Division of monitoring labour
-----------------------------
Two different populations of sacrificed rounds measure two different things,
and conflating them is a real (and common) modelling error:

``CHECK`` pairs
    An EPR pair that is *never* teleported through.  Both halves are measured
    promptly, so these rounds see channel noise but **not** quantum-memory
    noise.  They certify entanglement (CHSH, Bell fidelity, concurrence).

``DECOY`` slots
    Full teleported BB84 states that Alice reveals immediately after
    distribution.  They travel the entire path -- EPR creation, channel, Bell
    measurement, Pauli correction, and the same time in quantum memory as a
    real signature slot -- so they, and only they, measure the end-to-end
    error rate the acceptance threshold is set against.

Register layout
---------------
Teleportation round (:func:`teleport_state`)::

    q0 = S   the BB84 signature qubit Alice prepares
    q1 = A   Alice's half of the EPR pair
    q2 = B   the half that travels to the recipient and is stored
    q3 = P   Eve's probe, present only if the intervention asks for one

Entanglement-check round (:func:`distribute_pair`)::

    q0 = A   Alice's half        q1 = B   the recipient's half
    q2 = P   Eve's probe, if any

The circuit follows the convention pinned by ``tests/test_backends.py``: EPR
pair by ``H`` then ``CNOT``, Bell measurement by ``CNOT`` then ``H`` then two
``Z``-basis readouts, correction ``Z^u X^v`` on the recipient's half.

What the verifier may see
-------------------------
:func:`distribute_public_key` returns a :class:`DistributionResult` split
deliberately in two.  ``result.store`` is everything the recipient physically
holds and everything Alice announced -- the detection engine is given *only*
this.  ``result.oracle`` is ground truth: Eve's joint states, the true Bell
outcomes before readout error, the exact delivered fidelities.  It exists so
the analytics layer can plot what the detector could not have known, and so
the tests can check the detector against reality.  Nothing in ``qds.detect``
imports it, which makes "the alarm uses only observable data" a structural
property of the code rather than a claim in a README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import linalg as L
from ..channel import IDEAL, Intervention, NoIntervention, NoiseModel
from .keys import HeldQubit, KeyEntry, PrivateKey, PublicKeyStore, Slot, SlotRole

__all__ = [
    "DistributionConfig", "PairRound", "TeleportRound",
    "DistributionOracle", "DistributionResult",
    "distribute_pair", "teleport_state", "distribute_public_key",
    "bell_outcome_distribution", "teleportation_leakage",
]


# --------------------------------------------------------------------------
# small exact-propagation helpers
# --------------------------------------------------------------------------

def _project(rho: np.ndarray, qubit: int, bit: int) -> Tuple[float, np.ndarray]:
    """``(probability, unnormalised P rho P)`` for a Z-basis projector.

    ``rho`` may itself be unnormalised, in which case the returned probability
    is the *joint* probability of this outcome and whatever produced ``rho``.
    That is what makes chaining two calls give ``p(u, v)`` directly.
    """
    return L.project_z(rho, qubit, bit)


def _bell_branches(rho: np.ndarray, q0: int, q1: int,
                   tol: float = 1e-15) -> List[Tuple[int, int, float, np.ndarray]]:
    """Every ``(u, v, p(u,v), post_state)`` branch of the Bell readout."""
    branches: List[Tuple[int, int, float, np.ndarray]] = []
    for u in (0, 1):
        pu, ru = _project(rho, q0, u)
        if pu <= tol:
            continue
        for v in (0, 1):
            puv, ruv = _project(ru, q1, v)
            if puv <= tol:
                continue
            branches.append((u, v, puv, ruv / puv))
    return branches


def _correction(u: int, v: int) -> Optional[np.ndarray]:
    """The single Pauli ``Z^u X^v``, or ``None`` when no correction is needed."""
    if u == 0 and v == 0:
        return None
    if u == 1 and v == 0:
        return L.Z
    if u == 0 and v == 1:
        return L.X
    return L.Z @ L.X


def _readout_weight(true_bit: int, announced: int, pm: float) -> float:
    return pm if announced != true_bit else 1.0 - pm


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributionConfig:
    """Everything that fixes one distribution phase.

    ``check_pairs`` and ``decoy_slots`` are the monitoring budget.  They cost
    nothing in signature capacity -- they are extra rounds, not stolen ones --
    but they cost channel time, so the dashboard exposes the trade-off between
    monitoring budget and statistical power.

    ``storage_intervals`` is how long a public-key qubit sits in quantum
    memory between distribution and signing, in units of the memory model's
    dephasing interval.  Decoy slots are aged identically, which is the only
    reason they are a fair proxy for the signature slots.

    ``exact`` replaces outcome sampling with an exact ensemble average over
    the four Bell outcomes and the readout-error distribution.  The delivered
    state is then the true mixture rather than one sample from it, which makes
    fidelity and error-rate figures free of Monte-Carlo noise.  Stochastic
    interventions (intercept-resend, for instance) still consume the RNG:
    ``exact`` is a statement about the protocol's own randomness, not Eve's.
    """

    noise: NoiseModel = IDEAL
    intervention: Intervention = field(default_factory=NoIntervention)
    check_pairs: int = 64
    decoy_slots: int = 64
    storage_intervals: int = 1
    exact: bool = False
    #: gate noise is applied to the correction step even when the correction
    #: is the identity, because real hardware idles for the same duration
    idle_is_noisy: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "check_pairs": self.check_pairs,
            "decoy_slots": self.decoy_slots,
            "storage_intervals": self.storage_intervals,
            "exact": self.exact,
            "intervention": self.intervention.to_dict(),
            "predicted_qber": self.noise.predicted_qber(),
        }


# --------------------------------------------------------------------------
# per-round records
# --------------------------------------------------------------------------

@dataclass
class PairRound:
    """One sacrificed EPR pair, as it actually arrived."""

    index: int
    rho_ab: np.ndarray                      #: joint two-qubit state, A then B
    status: str                             #: "click" / "dark" / "lost"
    rho_be: Optional[np.ndarray] = None     #: joint (B, Eve) state, oracle only

    @property
    def chsh(self) -> float:
        return L.chsh_value(self.rho_ab, (0, 1))

    @property
    def bell_fidelity(self) -> float:
        return float(np.real(
            L.BELL_STATES["Phi+"].conj() @ self.rho_ab @ L.BELL_STATES["Phi+"]
        ))

    @property
    def concurrence(self) -> float:
        return L.concurrence(self.rho_ab)


@dataclass
class TeleportRound:
    """One teleported public-key qubit."""

    slot: Slot
    entry: KeyEntry
    rho_b: np.ndarray                       #: what the recipient holds
    true_bits: Tuple[int, int]              #: Alice's Bell outcome before readout error
    announced: Tuple[int, int]              #: what Alice broadcast
    status: str
    rho_bp: Optional[np.ndarray] = None     #: joint (B, Eve) state, oracle only
    #: Eve's probe *conditioned on the broadcast bits*, as
    #: ``(u, v) -> (probability, rho_P)``.  Keeping the conditioning explicit
    #: is not pedantry: Eve hears ``(u, v)``, so her state of knowledge is the
    #: conditional one, and averaging over it understates her by exactly the
    #: amount that makes a coherent probe look harmless.  Oracle only.
    eve_conditional: Dict[Tuple[int, int], Tuple[float, np.ndarray]] = \
        field(default_factory=dict)

    @property
    def target(self) -> np.ndarray:
        return L.bb84_state(self.entry.basis, self.entry.bit)

    @property
    def fidelity(self) -> float:
        """Probability the honest verifier's check on this slot *passes*.

        For a BB84 target this is exactly ``<psi|rho|psi>``: the verifier
        measures in basis ``a`` and demands outcome ``c``, and ``|psi>`` is
        the corresponding eigenvector.  Readout error at his end is applied
        later, by the verification layer.
        """
        return L.fidelity_pure(self.rho_b, self.target)

    @property
    def misannounced(self) -> bool:
        return self.true_bits != self.announced


# --------------------------------------------------------------------------
# the two primitive rounds
# --------------------------------------------------------------------------

def distribute_pair(cfg: DistributionConfig, rng: np.random.Generator,
                    index: int = 0, storage_intervals: int = 0) -> PairRound:
    """Share one Bell pair ``|Phi+> = (|00> + |11>)/sqrt(2)``.

    Alice keeps ``q0`` and sends ``q1``.  Only the travelling half sees the
    channel, the intervention and the loss model, which is the asymmetry that
    makes a CHSH drop attributable to the link rather than to Alice.
    """
    noise, ev = cfg.noise, cfg.intervention
    n_probe = ev.probe_qubits()
    n = 2 + n_probe
    probe = 2 if n_probe else None

    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0

    rho = L.apply_unitary(rho, L.H, [0])
    rho = noise.apply_gate_noise(rho, [0])
    rho = L.apply_unitary(rho, L.CNOT, [0, 1])
    rho = noise.apply_gate_noise(rho, [0, 1])

    rho = noise.apply_channel(rho, 1)
    rho = ev.on_transit(rho, 1, rng, probe=probe)

    if storage_intervals:
        rho = noise.apply_memory(rho, 1, storage_intervals)

    status = noise.loss.sample(rng)
    rho_ab = L.partial_trace(rho, [0, 1])
    if status != "click":
        # Either nothing arrived, or a dark count fabricated an outcome.  In
        # both cases the recipient's half carries no correlation with Alice's;
        # a dark count still *reports* something, so the state it reports is a
        # fair coin, i.e. maximally mixed and uncorrelated.
        rho_ab = np.kron(L.partial_trace(rho, [0]), np.eye(2) / 2.0)

    rho_be = L.partial_trace(rho, [1, 2]) if probe is not None else None
    return PairRound(index=index, rho_ab=rho_ab, status=status, rho_be=rho_be)


def teleport_state(entry: KeyEntry, slot: Slot, cfg: DistributionConfig,
                   rng: np.random.Generator) -> TeleportRound:
    """Deliver one BB84 public-key qubit by teleportation.

    The full chain: build the EPR pair, send half of it down the channel (this
    is where Eve gets her only access), Bell-measure Alice's half against the
    signature qubit, broadcast the two correction bits, apply ``Z^u X^v`` at
    the far end, then age the result in quantum memory.
    """
    noise, ev = cfg.noise, cfg.intervention
    n_probe = ev.probe_qubits()
    n = 3 + n_probe
    probe = 3 if n_probe else None
    pm = noise.measurement_error

    psi = L.bb84_state(entry.basis, entry.bit)
    rho = np.kron(L.density(psi), _zero(n - 1))

    # -- 1. entangle q1 with q2 -------------------------------------------
    rho = L.apply_unitary(rho, L.H, [1])
    rho = noise.apply_gate_noise(rho, [1])
    rho = L.apply_unitary(rho, L.CNOT, [1, 2])
    rho = noise.apply_gate_noise(rho, [1, 2])

    # -- 2. q2 crosses the channel; Eve acts here and only here ------------
    rho = noise.apply_channel(rho, 2)
    rho = ev.on_transit(rho, 2, rng, probe=probe)

    # -- 3. Alice's Bell measurement on (q0, q1) ---------------------------
    rho = L.apply_unitary(rho, L.CNOT, [0, 1])
    rho = noise.apply_gate_noise(rho, [0, 1])
    rho = L.apply_unitary(rho, L.H, [0])
    rho = noise.apply_gate_noise(rho, [0])

    keep = [2, 3] if probe is not None else [2]
    branches = _bell_branches(rho, 0, 1)
    #: announced (u, v) -> [total probability, unnormalised sum of rho_P]
    eve_acc: Dict[Tuple[int, int], List] = {}

    def _record_eve(au: int, av: int, weight: float, sub: np.ndarray) -> None:
        """Accumulate Eve's probe state conditioned on the broadcast bits.

        The correction is a unitary acting on the recipient's qubit alone, so
        it cannot change Eve's marginal -- ``sub`` may be taken before or
        after it with identical result.
        """
        if probe is None or weight <= 0.0:
            return
        rho_p = L.partial_trace(sub, [1])
        slot_acc = eve_acc.setdefault((au, av), [0.0, np.zeros_like(rho_p)])
        slot_acc[0] += weight
        slot_acc[1] = slot_acc[1] + weight * rho_p

    if cfg.exact:
        acc = None
        for u, v, p, post in branches:
            sub = L.partial_trace(post, keep)
            for au in (0, 1):
                wu = _readout_weight(u, au, pm)
                if wu <= 0.0:
                    continue
                for av in (0, 1):
                    wv = _readout_weight(v, av, pm)
                    if wv <= 0.0:
                        continue
                    w = p * wu * wv
                    corrected = _apply_correction(sub, au, av, noise, cfg)
                    term = w * corrected
                    acc = term if acc is None else acc + term
                    _record_eve(au, av, w, sub)
        out = acc
        # the reported outcome is still a sample, so downstream code that
        # inspects the transcript sees a well-formed one
        true_bits = _sample_branch(branches, rng)
        announced = (noise.flip_readout(true_bits[0], rng),
                     noise.flip_readout(true_bits[1], rng))
    else:
        u, v, post = _draw_branch(branches, rng)
        true_bits = (u, v)
        announced = (noise.flip_readout(u, rng), noise.flip_readout(v, rng))
        sub = L.partial_trace(post, keep)
        out = _apply_correction(sub, announced[0], announced[1], noise, cfg)
        _record_eve(announced[0], announced[1], 1.0, sub)

    eve_conditional = {k: (w, m / w) for k, (w, m) in eve_acc.items() if w > 0.0}

    # -- 4. quantum memory -------------------------------------------------
    if cfg.storage_intervals:
        out = noise.apply_memory(out, 0, cfg.storage_intervals)

    status = noise.loss.sample(rng)
    rho_bp = out if probe is not None else None
    rho_b = L.partial_trace(out, [0]) if probe is not None else out
    if status != "click":
        # nothing usable arrived; a dark count still yields a random outcome
        rho_b = np.eye(2, dtype=complex) / 2.0

    return TeleportRound(slot=slot, entry=entry, rho_b=rho_b,
                         true_bits=true_bits, announced=announced,
                         status=status, rho_bp=rho_bp,
                         eve_conditional=eve_conditional)


def _zero(n: int) -> np.ndarray:
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _apply_correction(rho: np.ndarray, u: int, v: int, noise: NoiseModel,
                      cfg: DistributionConfig) -> np.ndarray:
    """``Z^u X^v`` on qubit 0 of ``rho``, plus one gate-noise insertion.

    The two Paulis are composed into a single gate, so the correction costs
    one noisy slot rather than two -- that is how it would be compiled on real
    hardware, and it is the count :data:`NoiseModel.GATE_SLOTS_PER_CHECK`
    assumes.
    """
    C = _correction(u, v)
    if C is not None:
        rho = L.apply_unitary(rho, C, [0])
    if cfg.idle_is_noisy or C is not None:
        rho = noise.apply_gate_noise(rho, [0])
    return rho


def _draw_branch(branches, rng) -> Tuple[int, int, np.ndarray]:
    probs = np.array([b[2] for b in branches], dtype=float)
    probs = probs / probs.sum()
    k = int(rng.choice(len(branches), p=probs))
    u, v, _, post = branches[k]
    return u, v, post


def _sample_branch(branches, rng) -> Tuple[int, int]:
    probs = np.array([b[2] for b in branches], dtype=float)
    probs = probs / probs.sum()
    k = int(rng.choice(len(branches), p=probs))
    return branches[k][0], branches[k][1]


# --------------------------------------------------------------------------
# the classical transcript leaks nothing: an exact statement
# --------------------------------------------------------------------------

def bell_outcome_distribution(entry: KeyEntry,
                              cfg: Optional[DistributionConfig] = None
                              ) -> Dict[Tuple[int, int], float]:
    """Exact distribution of the announced correction bits ``(u, v)``.

    Computed by propagating the density matrix, not by sampling.  For every
    one of the four BB84 inputs the answer is ``1/4`` on each outcome, so the
    public classical transcript of the distribution phase is a sequence of
    fair coin flips statistically independent of the private key.  This is the
    formal content of "teleportation is a safe public-key delivery channel",
    and :func:`teleportation_leakage` turns it into a single number.
    """
    cfg = cfg or DistributionConfig()
    noise, ev = cfg.noise, cfg.intervention
    n_probe = ev.probe_qubits()
    n = 3 + n_probe
    probe = 3 if n_probe else None
    rng = np.random.default_rng(0)

    psi = L.bb84_state(entry.basis, entry.bit)
    rho = np.kron(L.density(psi), _zero(n - 1))
    rho = L.apply_unitary(rho, L.H, [1])
    rho = noise.apply_gate_noise(rho, [1])
    rho = L.apply_unitary(rho, L.CNOT, [1, 2])
    rho = noise.apply_gate_noise(rho, [1, 2])
    rho = noise.apply_channel(rho, 2)
    rho = ev.on_transit(rho, 2, rng, probe=probe)
    rho = L.apply_unitary(rho, L.CNOT, [0, 1])
    rho = noise.apply_gate_noise(rho, [0, 1])
    rho = L.apply_unitary(rho, L.H, [0])
    rho = noise.apply_gate_noise(rho, [0])

    pm = noise.measurement_error
    out: Dict[Tuple[int, int], float] = {(u, v): 0.0
                                         for u in (0, 1) for v in (0, 1)}
    for u, v, p, _ in _bell_branches(rho, 0, 1):
        for au in (0, 1):
            for av in (0, 1):
                out[(au, av)] += p * _readout_weight(u, au, pm) * \
                    _readout_weight(v, av, pm)
    return out


def teleportation_leakage(cfg: Optional[DistributionConfig] = None) -> float:
    """Total-variation distance between transcripts for different key entries.

    The adversary's best strategy for reading the private key off the public
    classical channel is to distinguish the ``(u, v)`` distribution produced
    by one key entry from that produced by another; her advantage is bounded
    by the largest total-variation distance between any two of the four.  The
    function returns that maximum.  It is ``0`` to machine precision, honest
    channel or not, because entanglement-assisted transmission puts *all* of
    the state into the shared correlation and none of it into the broadcast.
    """
    cfg = cfg or DistributionConfig()
    dists = [bell_outcome_distribution(KeyEntry(a, c), cfg)
             for a in (0, 1) for c in (0, 1)]
    worst = 0.0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            tv = 0.5 * sum(abs(dists[i][k] - dists[j][k]) for k in dists[i])
            worst = max(worst, tv)
    return worst


# --------------------------------------------------------------------------
# full distribution phase
# --------------------------------------------------------------------------

@dataclass
class DistributionOracle:
    """Ground truth about one distribution phase.

    **Not available to the detection engine.**  Nothing under ``qds.detect``
    imports this class; it exists for the analytics layer, the whitepaper
    figures, and the tests, all of which are allowed to know things the
    verifier cannot.
    """

    pairs: List[PairRound] = field(default_factory=list)
    rounds: List[TeleportRound] = field(default_factory=list)

    #: A round counts towards an observable statistic exactly when the
    #: recipient's detector produced an outcome.  A dark count did produce
    #: one -- a worthless one -- so it is *included*, which is precisely how a
    #: dark count degrades a real QBER estimate.  Only ``"lost"`` rounds,
    #: where nothing was registered at all, are dropped.
    REGISTERED = ("click", "dark")

    def _live_pairs(self) -> List[PairRound]:
        return [p for p in self.pairs if p.status in self.REGISTERED]

    # -- channel quality ---------------------------------------------------
    def mean_chsh(self) -> float:
        good = [p.chsh for p in self._live_pairs()]
        return float(np.mean(good)) if good else 0.0

    def mean_bell_fidelity(self) -> float:
        good = [p.bell_fidelity for p in self._live_pairs()]
        return float(np.mean(good)) if good else 0.0

    def mean_concurrence(self) -> float:
        good = [p.concurrence for p in self._live_pairs()]
        return float(np.mean(good)) if good else 0.0

    # -- delivered key quality --------------------------------------------
    def exact_qber(self, include: str = "all") -> float:
        """Exact per-check failure probability of the delivered qubits.

        This is the quantity :meth:`NoiseModel.predicted_qber` approximates
        analytically.  Here it is computed from the states that were actually
        delivered, so it is correct by construction even for attacks the
        analytic formula knows nothing about.
        """
        sel = self._select(include)
        if not sel:
            return 0.0
        return float(np.mean([1.0 - r.fidelity for r in sel]))

    def qber_by_basis(self, include: str = "all") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for label, basis in (("Z", 0), ("X", 1)):
            sel = [r for r in self._select(include) if r.entry.basis == basis]
            out[label] = float(np.mean([1.0 - r.fidelity for r in sel])) if sel else 0.0
        return out

    def announcement_error_rate(self) -> float:
        sel = self._select("all")
        if not sel:
            return 0.0
        return float(np.mean([1.0 if r.misannounced else 0.0 for r in sel]))

    # -- what Eve actually got --------------------------------------------
    def eve_distinguishability(self) -> float:
        """Trace distance between Eve's probe states for the two Z-basis inputs.

        ``0`` means her ancilla is uncorrelated with the key and she has
        learned nothing; ``1`` means she can read the bit off her probe with
        certainty.  By Helstrom, her optimal probability of guessing the bit
        is ``(1 + d) / 2``.  Paired with :meth:`exact_qber` this is the
        empirical information--disturbance curve the whitepaper plots.

        The conditioning matters and is easy to get wrong.  Eve touches the
        travelling half *before* the Pauli correction ``Z^u X^v`` is applied,
        so her probe is correlated with ``c XOR v``, not with ``c``.  She
        hears ``v`` broadcast, so she can undo it for free -- and an analysis
        that averages over ``v`` without conditioning finds a trace distance
        of exactly zero and concludes, wrongly, that a coherent probe learns
        nothing.  Grouping by ``c XOR v`` is what makes this an upper bound on
        a real adversary rather than a flattering one.

        Every round counts, including ones the recipient's detector missed:
        Eve probed the qubit in transit, so her information is not
        conditioned on his detector firing.
        """
        groups: Dict[int, List[Tuple[float, np.ndarray]]] = {0: [], 1: []}
        for r in self.rounds:
            if r.entry.basis != 0 or not r.eve_conditional:
                continue
            for (_u, v), (w, rho_p) in r.eve_conditional.items():
                groups[r.entry.bit ^ v].append((w, rho_p))
        if not groups[0] or not groups[1]:
            return 0.0
        mixes = []
        for g in (0, 1):
            tot = sum(w for w, _ in groups[g])
            mixes.append(sum(w * m for w, m in groups[g]) / tot)
        return L.trace_distance(mixes[0], mixes[1])

    def eve_guess_probability(self) -> float:
        """Helstrom-optimal probability that Eve guesses a Z-basis key bit.

        ``1/2`` is a coin flip -- she knows nothing.  This is the number the
        whitepaper's information--disturbance figure puts on the y-axis.
        """
        return 0.5 * (1.0 + self.eve_distinguishability())

    def information_disturbance(self) -> Dict[str, float]:
        """Both sides of the trade-off, as one record.

        For the canonical coherent probe these satisfy
        ``d = 2 sqrt(e (1 - e))`` exactly, which says an adversary cannot buy
        information without paying in disturbance.  That identity, not any
        trained model, is what the alarm ultimately rests on.
        """
        d = self.eve_distinguishability()
        e = self.qber_by_basis()["X"]
        return {
            "eve_trace_distance": d,
            "eve_guess_probability": 0.5 * (1.0 + d),
            "disturbance_x_basis": e,
            "predicted_trace_distance": 2.0 * math.sqrt(max(e * (1.0 - e), 0.0)),
        }

    def _select(self, include: str) -> List[TeleportRound]:
        live = [r for r in self.rounds if r.status in self.REGISTERED]
        if include == "decoy":
            return [r for r in live if r.slot.j == SlotRole.DECOY]
        if include == "signature":
            return [r for r in live if r.slot.j >= 0]
        if include == "all":
            return live
        raise ValueError(f"include must be 'all', 'decoy' or 'signature', "
                         f"not {include!r}")

    def summary(self) -> Dict[str, object]:
        return {
            "check_pairs": len(self.pairs),
            "teleported_slots": len(self.rounds),
            "mean_chsh": self.mean_chsh(),
            "mean_bell_fidelity": self.mean_bell_fidelity(),
            "mean_concurrence": self.mean_concurrence(),
            "exact_qber": self.exact_qber(),
            "exact_qber_by_basis": self.qber_by_basis(),
            "exact_qber_decoy": self.exact_qber("decoy"),
            "announcement_error_rate": self.announcement_error_rate(),
            "information_disturbance": self.information_disturbance(),
        }


@dataclass
class DistributionResult:
    """Outcome of distributing one public key to one recipient."""

    store: PublicKeyStore
    #: decoy entries, revealed publicly by Alice right after distribution
    decoy_key: Dict[Slot, KeyEntry]
    oracle: DistributionOracle
    config: DistributionConfig

    def summary(self) -> Dict[str, object]:
        return {
            "verifier": self.store.verifier,
            "signer": self.store.signer,
            "key_id": self.store.key_id,
            "slots_held": len(self.store.qubits),
            "available": self.store.n_available,
            "check_slots": len(self.store.check_slots),
            "decoy_slots": len(self.store.decoy_slots),
            "yield_observed": self.store.yield_observed,
            "config": self.config.to_dict(),
            "oracle": self.oracle.summary(),
        }


def distribute_public_key(private_key: PrivateKey, verifier: str,
                          cfg: DistributionConfig,
                          rng: np.random.Generator) -> DistributionResult:
    """Run one complete distribution phase for one recipient.

    Order of rounds matters and is deliberate: check pairs first, then decoys,
    then the signature slots, all interleaved with the *same* channel and the
    *same* intervention.  An adversary who wanted to behave well during the
    monitored rounds and badly during the rest would have to distinguish them,
    and she cannot: the slot labels are never broadcast until after the qubits
    have arrived.  The simulator honours that by handing the intervention no
    information about which round it is touching.
    """
    store = PublicKeyStore(verifier=verifier, signer=private_key.signer,
                           key_id=private_key.key_id)
    oracle = DistributionOracle()
    decoy_key: Dict[Slot, KeyEntry] = {}

    # -- 1. sacrificed EPR pairs: is the link still entangled? -------------
    for i in range(cfg.check_pairs):
        pr = distribute_pair(cfg, rng, index=i, storage_intervals=0)
        oracle.pairs.append(pr)
        slot = Slot(SlotRole.CHECK, 0, i)
        store.check_slots.append(slot)
        store.qubits[slot] = HeldQubit(slot=slot, rho=pr.rho_ab,
                                       arrived=pr.status != "lost",
                                       dark=pr.status == "dark")
        store.detector_log.append(pr.status)

    # -- 2. decoy slots: what is the true end-to-end error rate? -----------
    for i in range(cfg.decoy_slots):
        entry = KeyEntry(int(rng.integers(0, 2)), int(rng.integers(0, 2)))
        slot = Slot(SlotRole.DECOY, 0, i)
        tr = teleport_state(entry, slot, cfg, rng)
        oracle.rounds.append(tr)
        decoy_key[slot] = entry
        store.decoy_slots.append(slot)
        _install(store, tr, cfg)

    # -- 3. the public key proper ------------------------------------------
    for slot in private_key.slots():
        tr = teleport_state(private_key.entry(slot), slot, cfg, rng)
        oracle.rounds.append(tr)
        _install(store, tr, cfg)

    return DistributionResult(store=store, decoy_key=decoy_key, oracle=oracle,
                              config=cfg)


def _install(store: PublicKeyStore, tr: TeleportRound,
             cfg: DistributionConfig) -> None:
    store.add(HeldQubit(
        slot=tr.slot,
        rho=tr.rho_b,
        correction=tr.announced,
        arrived=tr.status != "lost",
        dark=tr.status == "dark",
        storage_intervals=cfg.storage_intervals,
    ))
    store.detector_log.append(tr.status)
