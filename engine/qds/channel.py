"""Quantum channel models: honest hardware noise and adversarial interventions.

Two very different things live in this module and the distinction matters for
the security argument:

*Honest noise* (:class:`NoiseModel`) is what an uncompromised link does on its
own -- depolarisation in fibre, dephasing in quantum memory, imperfect gates,
detector dark counts, photon loss.  It is characterised once during
calibration and it sets the *noise floor* against which every statistical test
is run.

*Adversarial interventions* (the ``Intervention`` subclasses) are what Eve
does.  The whole point of the framework is that Eve cannot act on the quantum
channel without adding disturbance on top of the honest noise floor, so any
intervention strong enough to be useful is also strong enough to be seen.

Every map here is expressed as an explicit Kraus set, so the simulation is
exact rather than sampled.  Photon loss is the one exception: a threshold
detector either clicks or does not, so loss is modelled as a classical
erasure event at the protocol layer (see :class:`LossModel`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .linalg import (
    H,
    I2,
    X,
    Y,
    Z,
    apply_kraus,
    apply_unitary,
    ry,
    rz,
)

__all__ = [
    "depolarizing_kraus", "pauli_kraus", "dephasing_kraus", "bitflip_kraus",
    "amplitude_damping_kraus", "coherent_rotation",
    "NoiseModel", "LossModel", "IDEAL", "LAB_GRADE", "FIELD_GRADE",
    "Intervention", "NoIntervention", "InterceptResend", "EntanglementBreaking",
    "CollectiveDepolarizing", "CoherentProbe", "BasisBiasedProbe",
    "make_intervention",
]


# --------------------------------------------------------------------------
# Kraus sets
# --------------------------------------------------------------------------

def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """Single-qubit depolarising channel, ``rho -> (1-p) rho + p I/2``.

    Expressed in the Pauli-twirl form with weights ``(1-3p/4, p/4, p/4, p/4)``
    so that ``p`` is the standard depolarising *parameter*: an eigenstate of
    any Pauli emerges with error probability ``p/2``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("depolarising parameter must lie in [0, 1]")
    return [
        math.sqrt(1.0 - 3.0 * p / 4.0) * I2,
        math.sqrt(p / 4.0) * X,
        math.sqrt(p / 4.0) * Y,
        math.sqrt(p / 4.0) * Z,
    ]


def pauli_kraus(px: float, py: float, pz: float) -> List[np.ndarray]:
    """Asymmetric Pauli channel.  Useful for modelling biased attacks."""
    p0 = 1.0 - px - py - pz
    if min(p0, px, py, pz) < -1e-12:
        raise ValueError("Pauli probabilities must be non-negative and sum <= 1")
    return [
        math.sqrt(max(p0, 0.0)) * I2,
        math.sqrt(max(px, 0.0)) * X,
        math.sqrt(max(py, 0.0)) * Y,
        math.sqrt(max(pz, 0.0)) * Z,
    ]


def dephasing_kraus(p: float) -> List[np.ndarray]:
    """Phase-flip channel: leaves Z eigenstates alone, damages X and Y."""
    return [math.sqrt(1.0 - p) * I2, math.sqrt(p) * Z]


def bitflip_kraus(p: float) -> List[np.ndarray]:
    return [math.sqrt(1.0 - p) * I2, math.sqrt(p) * X]


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return [k0, k1]


def coherent_rotation(theta: float, axis: str = "Y") -> np.ndarray:
    """Slow systematic misalignment of a polarisation/phase reference."""
    if axis.upper() == "Y":
        return ry(theta)
    if axis.upper() == "Z":
        return rz(theta)
    raise ValueError("axis must be 'Y' or 'Z'")


# --------------------------------------------------------------------------
# honest hardware model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LossModel:
    """Photon loss and detector imperfections.

    ``transmittance`` is the probability that a transmitted qubit reaches the
    far end at all; ``detector_efficiency`` folds in the click probability.
    ``dark_count`` is the probability that a non-arrival still produces a
    (uniformly random) outcome, which is how a real threshold detector fakes
    data and how detector-blinding attacks hide.
    """

    transmittance: float = 1.0
    detector_efficiency: float = 1.0
    dark_count: float = 0.0

    @property
    def yield_(self) -> float:
        """Probability of registering an outcome for one transmitted qubit."""
        arrive = self.transmittance * self.detector_efficiency
        return arrive + (1.0 - arrive) * self.dark_count

    def sample(self, rng: np.random.Generator) -> str:
        """Return ``"click"``, ``"dark"`` or ``"lost"`` for one transmission."""
        if rng.random() < self.transmittance * self.detector_efficiency:
            return "click"
        if rng.random() < self.dark_count:
            return "dark"
        return "lost"


@dataclass(frozen=True)
class NoiseModel:
    """Calibrated honest-hardware noise for one protocol run.

    Attributes
    ----------
    channel_depolarizing:
        Depolarising parameter applied to each qubit that travels down the
        quantum channel (one half of every EPR pair).
    memory_dephasing:
        Dephasing applied to a stored EPR half per storage interval; quantum
        memory is the dominant error source in a real QDS deployment because
        public-key halves must be held until a message is signed.
    gate_error:
        Depolarising parameter applied after each one- or two-qubit gate.
    measurement_error:
        Probability that a projective measurement outcome is reported flipped
        (classical readout error).
    misalignment:
        Systematic rotation angle (radians) of the remote reference frame.
    loss:
        Loss / detector model.
    """

    channel_depolarizing: float = 0.0
    memory_dephasing: float = 0.0
    gate_error: float = 0.0
    measurement_error: float = 0.0
    misalignment: float = 0.0
    loss: LossModel = field(default_factory=LossModel)

    # -- convenience -------------------------------------------------------
    @property
    def is_ideal(self) -> bool:
        return (
            self.channel_depolarizing == 0.0
            and self.memory_dephasing == 0.0
            and self.gate_error == 0.0
            and self.measurement_error == 0.0
            and self.misalignment == 0.0
            and self.loss.yield_ == 1.0
        )

    def apply_channel(self, rho: np.ndarray, qubit: int) -> np.ndarray:
        """Honest transmission of ``qubit`` down the quantum channel."""
        if self.misalignment:
            rho = apply_unitary(rho, coherent_rotation(self.misalignment), [qubit])
        if self.channel_depolarizing:
            rho = apply_kraus(rho, depolarizing_kraus(self.channel_depolarizing), [qubit])
        return rho

    def apply_memory(self, rho: np.ndarray, qubit: int, intervals: int = 1) -> np.ndarray:
        if self.memory_dephasing and intervals > 0:
            # composing k dephasing channels of strength p gives strength
            # (1 - (1-2p)^k)/2; do it exactly rather than k times numerically
            p = 0.5 * (1.0 - (1.0 - 2.0 * self.memory_dephasing) ** intervals)
            rho = apply_kraus(rho, dephasing_kraus(p), [qubit])
        return rho

    def apply_gate_noise(self, rho: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
        if self.gate_error:
            for q in qubits:
                rho = apply_kraus(rho, depolarizing_kraus(self.gate_error), [q])
        return rho

    def flip_readout(self, bit: int, rng: np.random.Generator) -> int:
        if self.measurement_error and rng.random() < self.measurement_error:
            return 1 - bit
        return bit

    # -- analytic prediction ----------------------------------------------
    #: How many noisy gate slots sit between preparation of the signature
    #: qubit and the verifier's projective check.  Counted from the circuit in
    #: ``qds/protocol/distribution.py``: H and CNOT to make the EPR pair
    #: (2 slots, one of which acts on the half that later carries the state),
    #: CNOT and H for the Bell measurement (2 slots), and the Pauli correction
    #: on the verifier's half (1 slot).  ``tests/test_channel.py`` checks this
    #: constant against the exactly propagated density matrix.
    GATE_SLOTS_PER_CHECK: int = 5

    def predicted_qber(self, gate_slots: Optional[int] = None) -> float:
        """Analytic honest error rate per signature check.

        A check fails when the verifier's qubit is flipped *in the basis he
        measures*.  Each independent error source contributes a flip
        probability, and flips compose by the odd-parity rule
        ``e = (1 - prod_i (1 - 2 e_i)) / 2``:

        ``channel_depolarizing`` :math:`p_c`
            The travelling EPR half is depolarised, which flips any fixed
            basis with probability :math:`p_c/2`.
        ``memory_dephasing`` :math:`p_s`
            Dephasing is basis-selective: it never flips a Z-basis outcome and
            always-with-probability-:math:`p_s` flips an X-basis one.  The
            signature basis is drawn uniformly from {Z, X}, so the averaged
            flip probability is :math:`p_s/2`.  (It would be :math:`2p_s/3`
            for a six-state protocol; this scheme uses four states.)
        ``gate_error`` :math:`p_g`
            ``gate_slots`` depolarising insertions of :math:`p_g`, each
            flipping with probability :math:`p_g/2`.
        ``misalignment`` :math:`\\theta`
            A coherent ``Ry`` error, flipping with probability
            :math:`\\sin^2(\\theta/2)`.
        ``measurement_error`` :math:`p_m`
            Classical readout flip, probability :math:`p_m`.

        This is the noise floor :math:`\\eta` that the threshold calculus
        needs *before* a run starts; ``qds.protocol.distribution`` can compute
        the same quantity exactly by density-matrix propagation, and the two
        agree to better than one part in a thousand at realistic noise levels.
        """
        slots = self.GATE_SLOTS_PER_CHECK if gate_slots is None else gate_slots
        flips = []
        if self.channel_depolarizing:
            flips.append(self.channel_depolarizing / 2.0)
        if self.memory_dephasing:
            flips.append(self.memory_dephasing / 2.0)
        if self.gate_error:
            flips.append(slots * self.gate_error / 2.0)
        if self.misalignment:
            flips.append(math.sin(self.misalignment / 2.0) ** 2)
        if self.measurement_error:
            flips.append(self.measurement_error)
        prod = 1.0
        for e in flips:
            prod *= (1.0 - 2.0 * min(max(e, 0.0), 0.5))
        return 0.5 * (1.0 - prod)


#: A perfect link.  Used to demonstrate *deterministic* acceptance.
IDEAL = NoiseModel()

#: Optical-table numbers: a very good short link.
LAB_GRADE = NoiseModel(
    channel_depolarizing=0.008,
    memory_dephasing=0.004,
    gate_error=0.001,
    measurement_error=0.002,
    misalignment=0.01,
    loss=LossModel(transmittance=0.92, detector_efficiency=0.85, dark_count=1e-5),
)

#: Deployed-fibre numbers: metropolitan span, warm hardware.
FIELD_GRADE = NoiseModel(
    channel_depolarizing=0.03,
    memory_dephasing=0.015,
    gate_error=0.004,
    measurement_error=0.008,
    misalignment=0.04,
    loss=LossModel(transmittance=0.55, detector_efficiency=0.7, dark_count=5e-5),
)

PRESET_NOISE: Dict[str, NoiseModel] = {
    "ideal": IDEAL,
    "lab": LAB_GRADE,
    "field": FIELD_GRADE,
}


# --------------------------------------------------------------------------
# adversarial interventions on the quantum channel
# --------------------------------------------------------------------------

class Intervention:
    """Base class for what Eve does to a travelling qubit.

    An intervention is handed the *joint* state of the protocol register and
    the index of the qubit currently in flight.  It may attach ancillae, but
    in this simulator Eve's probe is pre-allocated by the caller so that the
    register size stays fixed and exact.
    """

    name = "none"
    #: human-readable description used by the reporting layer
    description = ""

    def on_transit(self, rho: np.ndarray, qubit: int,
                   rng: np.random.Generator,
                   probe: Optional[int] = None) -> np.ndarray:
        return rho

    def probe_qubits(self) -> int:
        """How many ancilla qubits Eve needs in the register."""
        return 0

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"name": self.name, "description": self.description}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        return d


class NoIntervention(Intervention):
    name = "none"
    description = "Honest channel; only calibrated hardware noise is present."


class InterceptResend(Intervention):
    """Measure the flying qubit, then send a fresh qubit in the outcome state.

    The textbook attack.  Eve gains full classical information about one basis
    and destroys the entanglement completely.  When she guesses the basis at
    random the induced error rate on signature checks is 25%.
    """

    name = "intercept_resend"

    def __init__(self, basis: str = "random", strength: float = 1.0):
        self.basis = basis
        self.strength = float(strength)   # fraction of pairs she touches
        self.description = (
            f"Intercept-and-resend on {self.strength:.0%} of transmissions, "
            f"measuring in the {basis} basis."
        )

    def on_transit(self, rho, qubit, rng, probe=None):
        from .linalg import measure_qubit, density, ket
        if rng.random() >= self.strength:
            return rho
        basis = self.basis
        if basis == "random":
            basis = "Z" if rng.random() < 0.5 else "X"
        elif basis == "breidbart":
            # Breidbart basis: halfway between Z and X, the optimal single-copy
            # guess for the BB84 four-state set
            rho = apply_unitary(rho, ry(-math.pi / 4), [qubit])
            bit, rho, _ = measure_qubit(rho, qubit, "Z", rng)
            rho = _replace_qubit(rho, qubit, ket([bit]))
            return apply_unitary(rho, ry(math.pi / 4), [qubit])
        bit, rho, _ = measure_qubit(rho, qubit, basis, rng)
        fresh = ket([bit])
        if basis == "X":
            fresh = H @ fresh
        return _replace_qubit(rho, qubit, fresh)


class EntanglementBreaking(Intervention):
    """Discard the flying qubit and inject an uncorrelated substitute.

    The crudest denial-of-integrity attack: the verifier still receives *a*
    qubit, so naive protocols proceed happily, but the EPR correlation is
    gone.  Detected by the CHSH and fidelity monitors immediately.
    """

    name = "entanglement_breaking"

    def __init__(self, strength: float = 1.0, substitute: str = "mixed"):
        self.strength = float(strength)
        self.substitute = substitute
        self.description = (
            f"Replace {self.strength:.0%} of transmitted EPR halves with an "
            f"uncorrelated {substitute} qubit."
        )

    def on_transit(self, rho, qubit, rng, probe=None):
        from .linalg import ket
        if rng.random() >= self.strength:
            return rho
        if self.substitute == "mixed":
            return _replace_qubit_mixed(rho, qubit)
        return _replace_qubit(rho, qubit, ket([0]))


class CollectiveDepolarizing(Intervention):
    """A weak symmetric attack, indistinguishable from extra channel noise.

    This is the interesting adversary: she deliberately hides beneath the
    noise floor.  The framework's answer is not to detect the *attack* but to
    bound the information she can have gained given the observed disturbance,
    which is what the threshold calculus in ``detect/thresholds.py`` does.
    """

    name = "collective_depolarizing"

    def __init__(self, p: float = 0.05):
        self.p = float(p)
        self.description = (
            f"Symmetric depolarising probe of strength p={self.p:.3f}, chosen "
            "to imitate honest channel noise."
        )

    def on_transit(self, rho, qubit, rng, probe=None):
        return apply_kraus(rho, depolarizing_kraus(self.p), [qubit])


class CoherentProbe(Intervention):
    """Entangle a probe with the flying qubit and store it.

    Eve applies a controlled rotation of strength ``theta`` between the
    travelling qubit and her own ancilla, then keeps the ancilla.  At
    ``theta = 0`` she learns nothing and disturbs nothing; at ``theta = pi/2``
    she learns the Z-basis value perfectly and induces 50% error in X.  This
    is the canonical information-versus-disturbance trade-off, and the
    simulator measures both sides of it.
    """

    name = "coherent_probe"

    def __init__(self, theta: float = 0.3, delayed_measurement: bool = True):
        self.theta = float(theta)
        self.delayed_measurement = bool(delayed_measurement)
        self.description = (
            f"Coherent entangling probe, theta={self.theta:.3f} rad"
            + (", measured only after the private key is revealed."
               if delayed_measurement else ", measured immediately.")
        )

    def probe_qubits(self) -> int:
        return 1

    def on_transit(self, rho, qubit, rng, probe=None):
        if probe is None:
            return rho
        # controlled-RY(2*theta): |0>_t leaves probe alone, |1>_t rotates it
        c_ry = np.eye(4, dtype=complex)
        c_ry[2:, 2:] = ry(2.0 * self.theta)
        return apply_unitary(rho, c_ry, [qubit, probe])


class BasisBiasedProbe(Intervention):
    """Attack that damages one basis far more than the other.

    Included specifically to show why the engine tests the Z-basis and
    X-basis error rates *separately* as well as jointly: a basis-asymmetric
    attack can keep the pooled error rate under threshold while pushing one
    basis well over it.
    """

    name = "basis_biased"

    def __init__(self, pz: float = 0.0, px: float = 0.12):
        self.pz = float(pz)
        self.px = float(px)
        self.description = (
            f"Asymmetric Pauli probe (p_X={self.px:.3f}, p_Z={self.pz:.3f}) that "
            "hides in the pooled error rate but not in the per-basis rates."
        )

    def on_transit(self, rho, qubit, rng, probe=None):
        return apply_kraus(rho, pauli_kraus(self.px, 0.0, self.pz), [qubit])


# --------------------------------------------------------------------------
# helpers for replacing a single qubit inside a joint state
# --------------------------------------------------------------------------

def _replace_qubit(rho: np.ndarray, qubit: int, psi: np.ndarray) -> np.ndarray:
    """Discard ``qubit`` and re-insert a fresh pure qubit in state ``psi``.

    Implemented as a genuine CP map (reset followed by preparation) rather
    than by index surgery, so the rest of the register keeps whatever
    correlations survive.
    """
    from .linalg import density
    reset = [np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
             np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)]
    rho = apply_kraus(rho, reset, [qubit])          # -> |0><0| on that qubit
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    prep = np.zeros((2, 2), dtype=complex)
    prep[:, 0] = psi / np.linalg.norm(psi)
    prep[:, 1] = 0.0
    return apply_kraus(rho, [prep], [qubit])


def _replace_qubit_mixed(rho: np.ndarray, qubit: int) -> np.ndarray:
    """Replace ``qubit`` with the maximally mixed state."""
    reset = [np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
             np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)]
    rho = apply_kraus(rho, reset, [qubit])
    return apply_kraus(rho, [I2 / math.sqrt(2.0), X / math.sqrt(2.0)], [qubit])


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_INTERVENTIONS = {
    "none": NoIntervention,
    "intercept_resend": InterceptResend,
    "entanglement_breaking": EntanglementBreaking,
    "collective_depolarizing": CollectiveDepolarizing,
    "coherent_probe": CoherentProbe,
    "basis_biased": BasisBiasedProbe,
}


def make_intervention(name: str, **kwargs) -> Intervention:
    try:
        cls = _INTERVENTIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown intervention {name!r}; choose from {sorted(_INTERVENTIONS)}"
        ) from exc
    return cls(**kwargs)
