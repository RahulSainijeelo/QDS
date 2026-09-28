"""Exact multi-qubit linear algebra on density matrices.

Everything in this project is built on top of this module.  There is no
approximation anywhere: states are represented as full complex density
matrices of shape ``(2**n, 2**n)``, unitaries are applied by tensor
contraction, noise is applied through Kraus operator sums, and measurements
are genuine projective measurements (Lueders rule) sampled from the exact
Born probabilities.

Register sizes in the TQDS protocol never exceed four qubits (signature
qubit + two halves of an EPR pair + optional eavesdropper probe), so exact
density-matrix propagation costs at most a 16x16 matrix multiply.  We pay
that cost happily in exchange for having zero modelling error.

Conventions
-----------
* Qubit ``0`` is the *most significant* tensor factor, i.e. the basis state
  index of an n-qubit register is ``sum(bit_q << (n - 1 - q))``.  This matches
  the usual textbook ordering ``|q0 q1 ... q_{n-1}>`` and is the ordering used
  by ``numpy.kron``.  (Qiskit uses the opposite convention internally; the
  Qiskit backend adapter is responsible for translating, see
  ``qds/backends/qiskit_backend.py``.)
* All functions are pure: they return new arrays and never mutate inputs.
"""

from __future__ import annotations

import cmath
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "I2", "X", "Y", "Z", "H", "S", "SDG", "T", "CNOT", "SWAP",
    "PAULI", "PAULI_LABELS",
    "rx", "ry", "rz", "phase", "pauli_string_op",
    "ket", "density", "bell_state", "BELL_STATES", "bb84_state",
    "apply_unitary", "apply_kraus", "apply_left", "apply_right",
    "apply_depolarizing", "apply_dephasing", "apply_bitflip",
    "apply_pauli_channel", "apply_1q", "project_z",
    "partial_trace", "reorder_qubits",
    "measure_qubit", "measure_pauli", "expectation",
    "axis_op", "measure_axis", "expectation_axis",
    "CHSH_SETTINGS", "chsh_value",
    "probabilities", "fidelity_pure", "fidelity_states", "purity",
    "trace_distance", "is_density_matrix", "random_unitary", "random_ket",
    "concurrence", "bell_fidelity_from_correlators",
]

# --------------------------------------------------------------------------
# single-qubit and two-qubit constants
# --------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)

CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=complex,
)
SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]], dtype=complex,
)

PAULI: Dict[str, np.ndarray] = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_LABELS = ("I", "X", "Y", "Z")


def rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    e = cmath.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, e.conjugate()]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, cmath.exp(1j * theta)]], dtype=complex)


def pauli_string_op(label: str) -> np.ndarray:
    """Tensor product of Pauli matrices, e.g. ``"XZ" -> X (x) Z``."""
    out = np.array([[1.0 + 0j]])
    for ch in label:
        out = np.kron(out, PAULI[ch.upper()])
    return out


# --------------------------------------------------------------------------
# state construction
# --------------------------------------------------------------------------

def ket(bits: Sequence[int] | str) -> np.ndarray:
    """Computational basis column vector, e.g. ``ket("01")``."""
    if isinstance(bits, str):
        bits = [int(c) for c in bits]
    n = len(bits)
    v = np.zeros(2 ** n, dtype=complex)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    v[idx] = 1.0
    return v


def density(psi: np.ndarray) -> np.ndarray:
    """Density matrix of a (not necessarily normalised) state vector."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError("cannot build a density matrix from the zero vector")
    psi = psi / nrm
    return np.outer(psi, psi.conj())


#: The four maximally entangled two-qubit Bell states.
BELL_STATES: Dict[str, np.ndarray] = {
    "Phi+": (ket("00") + ket("11")) / math.sqrt(2),
    "Phi-": (ket("00") - ket("11")) / math.sqrt(2),
    "Psi+": (ket("01") + ket("10")) / math.sqrt(2),
    "Psi-": (ket("01") - ket("10")) / math.sqrt(2),
}


def bell_state(kind: str = "Phi+") -> np.ndarray:
    """Density matrix of a Bell state."""
    return density(BELL_STATES[kind])


def bb84_state(basis: int, bit: int) -> np.ndarray:
    """One of the four BB84 states as a *state vector*.

    ``basis=0`` is the computational (Z) basis, ``basis=1`` the Hadamard (X)
    basis, so the four returned states are ``|0>, |1>, |+>, |->``.  They form
    two mutually unbiased bases; no measurement can distinguish all four with
    certainty, which is the physical resource that makes the signature
    unforgeable.
    """
    v = ket([bit])
    if basis:
        v = H @ v
    return v


# --------------------------------------------------------------------------
# operator application by tensor contraction
# --------------------------------------------------------------------------

def _as_tensor(rho: np.ndarray, n: int) -> np.ndarray:
    return rho.reshape((2,) * (2 * n))


def _from_tensor(t: np.ndarray, n: int) -> np.ndarray:
    return t.reshape((2 ** n, 2 ** n))


def _n_qubits(rho: np.ndarray) -> int:
    dim = rho.shape[0]
    n = int(round(math.log2(dim)))
    if 2 ** n != dim or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"not a square 2^n density matrix: shape {rho.shape}")
    return n


def apply_left(rho: np.ndarray, A: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Return ``A_full @ rho`` where ``A`` acts on ``qubits``."""
    n = _n_qubits(rho)
    k = len(qubits)
    t = _as_tensor(rho, n)
    At = np.asarray(A, dtype=complex).reshape((2,) * (2 * k))
    # contract A's column legs (k..2k-1) with rho's row legs (`qubits`)
    t = np.tensordot(At, t, axes=(list(range(k, 2 * k)), list(qubits)))
    t = np.moveaxis(t, list(range(k)), list(qubits))
    return _from_tensor(t, n)


def apply_right(rho: np.ndarray, B: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Return ``rho @ B_full`` where ``B`` acts on ``qubits``."""
    n = _n_qubits(rho)
    k = len(qubits)
    t = _as_tensor(rho, n)
    # (rho B)_{i j} = sum_k rho_{i k} B_{k j}: contract B's *row* legs with
    # rho's column legs, leaving B's column legs as the new column legs.
    Bt = np.asarray(B, dtype=complex).reshape((2,) * (2 * k))
    col = [q + n for q in qubits]
    t = np.tensordot(Bt, t, axes=(list(range(k)), col))
    t = np.moveaxis(t, list(range(k)), col)
    return _from_tensor(t, n)


def apply_unitary(rho: np.ndarray, U: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Return ``U rho U†`` with ``U`` acting on ``qubits`` (in that order)."""
    U = np.asarray(U, dtype=complex)
    if U.shape == (2, 2) and len(qubits) == 1:
        return apply_1q(rho, U, int(qubits[0]))
    return apply_right(apply_left(rho, U, qubits), U.conj().T, qubits)


# --------------------------------------------------------------------------
# fast exact paths for single-qubit Pauli channels
# --------------------------------------------------------------------------
# The generic Kraus route costs eight tensor contractions per depolarising
# insertion, and the teleportation circuit applies seven of them per round.
# The identities below are exact rewrites, not approximations -- the
# simulator's hot loop runs an order of magnitude faster for exactly the same
# numbers, and ``tests/test_linalg.py`` asserts agreement with
# :func:`apply_kraus` to machine precision rather than taking that on trust.

def _qubit_view(rho: np.ndarray, qubit: int) -> Tuple[np.ndarray, int]:
    """Reshape ``rho`` to ``(left, 2, right, left, 2, right)`` around ``qubit``."""
    n = _n_qubits(rho)
    if not 0 <= qubit < n:
        raise ValueError(f"qubit {qubit} outside a {n}-qubit register")
    left = 1 << qubit
    right = 1 << (n - 1 - qubit)
    return rho.reshape(left, 2, right, left, 2, right), n


def apply_depolarizing(rho: np.ndarray, qubit: int, p: float) -> np.ndarray:
    """``rho -> (1 - p) rho + p (I/2) ⊗ Tr_qubit(rho)``.

    The Pauli-twirl form ``(1-3p/4) rho + (p/4)(X rho X + Y rho Y + Z rho Z)``
    and this replacement form are the same map; the replacement form needs one
    partial trace instead of three conjugations, so it is what runs.
    """
    if p <= 0.0:
        return rho
    t, n = _qubit_view(rho, qubit)
    traced = t[:, 0, :, :, 0, :] + t[:, 1, :, :, 1, :]
    out = (1.0 - p) * t
    half = (0.5 * p) * traced
    out[:, 0, :, :, 0, :] += half
    out[:, 1, :, :, 1, :] += half
    return out.reshape(1 << n, 1 << n)


def apply_dephasing(rho: np.ndarray, qubit: int, p: float) -> np.ndarray:
    """``rho -> (1 - p) rho + p Z rho Z``: scale the coherences by ``1 - 2p``."""
    if p <= 0.0:
        return rho
    t, n = _qubit_view(rho, qubit)
    out = t.copy()
    factor = 1.0 - 2.0 * p
    out[:, 0, :, :, 1, :] *= factor
    out[:, 1, :, :, 0, :] *= factor
    return out.reshape(1 << n, 1 << n)


def apply_bitflip(rho: np.ndarray, qubit: int, p: float) -> np.ndarray:
    """``rho -> (1 - p) rho + p X rho X``."""
    if p <= 0.0:
        return rho
    t, n = _qubit_view(rho, qubit)
    out = (1.0 - p) * t + p * t[:, ::-1, :, :, ::-1, :]
    return out.reshape(1 << n, 1 << n)


def apply_pauli_channel(rho: np.ndarray, qubit: int,
                        px: float, py: float, pz: float) -> np.ndarray:
    """``rho -> p0 rho + px X rho X + py Y rho Y + pz Z rho Z``.

    Uses the index identities ``(X rho X)_{ij} = rho_{1-i, 1-j}``,
    ``(Z rho Z)_{ij} = (-1)^{i+j} rho_{ij}`` and, since ``Y = iXZ``,
    ``(Y rho Y)_{ij} = (-1)^{i+j} rho_{1-i, 1-j}`` on the target qubit's two
    indices, so the whole channel is four scaled array views added together.
    """
    p0 = 1.0 - px - py - pz
    if min(px, py, pz) < 0.0 or p0 < -1e-12:
        raise ValueError("Pauli probabilities must be non-negative and sum <= 1")
    if px == py == pz == 0.0:
        return rho
    t, n = _qubit_view(rho, qubit)
    flip = t[:, ::-1, :, :, ::-1, :]
    out = max(p0, 0.0) * t + px * flip
    if pz or py:
        # sign pattern (-1)^{i+j}: +1 on the diagonal blocks, -1 off them
        sgn = np.empty((1, 2, 1, 1, 2, 1), dtype=float)
        sgn[0, 0, 0, 0, 0, 0] = sgn[0, 1, 0, 0, 1, 0] = 1.0
        sgn[0, 0, 0, 0, 1, 0] = sgn[0, 1, 0, 0, 0, 0] = -1.0
        if pz:
            out += pz * (sgn * t)
        if py:
            out += py * (sgn * flip)
    return out.reshape(1 << n, 1 << n)


def apply_1q(rho: np.ndarray, U: np.ndarray, qubit: int) -> np.ndarray:
    """``U rho U†`` for a single-qubit ``U``, without any axis shuffling.

    ``apply_unitary`` is general and pays for it: two ``tensordot`` calls plus
    two ``moveaxis`` calls, and the axis bookkeeping costs more than the eight
    multiply-adds that actually do the work at these register sizes.  Here the
    target index is exposed by reshaping alone -- ``(left, 2, rest)`` for the
    row index, ``(rest, 2, right)`` for the column index -- so each side is one
    broadcast ``matmul``.
    """
    U = np.asarray(U, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError("apply_1q expects a 2x2 matrix")
    n = _n_qubits(rho)
    if not 0 <= qubit < n:
        raise ValueError(f"qubit {qubit} outside a {n}-qubit register")
    d = 1 << n
    left = 1 << qubit
    right = 1 << (n - 1 - qubit)
    # row side: (I ⊗ U ⊗ I) rho
    out = np.matmul(U, rho.reshape(left, 2, right * d)).reshape(d, d)
    # column side: rho (I ⊗ U† ⊗ I); U†[m, j] = conj(U[j, m]), so the matrix
    # that left-multiplies the exposed column index is U†.T = U.conj()
    out = np.matmul(U.conj(), out.reshape(d * left, 2, right)).reshape(d, d)
    return out


def project_z(rho: np.ndarray, qubit: int, bit: int) -> Tuple[float, np.ndarray]:
    """``(probability, P rho P)`` for the Z-basis projector ``P = |bit><bit|``.

    Sandwiching by a computational-basis projector keeps exactly one block of
    the reshaped array and zeroes the rest, so this is a masked copy rather
    than two contractions.  ``rho`` may be unnormalised, in which case the
    probability returned is the joint probability of this outcome together
    with whatever produced ``rho`` -- which is what makes chained calls give
    ``p(u, v)`` directly.
    """
    n = _n_qubits(rho)
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    left = 1 << qubit
    right = 1 << (n - 1 - qubit)
    t = rho.reshape(left, 2, right, left, 2, right)
    out = np.zeros_like(t)
    block = t[:, bit, :, :, bit, :]
    out[:, bit, :, :, bit, :] = block
    # trace of the kept block: sum over (l, r) of its diagonal entries
    prob = float(np.einsum("lrlr->", block).real)
    return prob, out.reshape(1 << n, 1 << n)


def apply_kraus(rho: np.ndarray, kraus: Iterable[np.ndarray],
                qubits: Sequence[int]) -> np.ndarray:
    """Return ``sum_i K_i rho K_i†`` -- a completely positive map.

    Trace preservation is *not* enforced here; :func:`qds.channel` builds the
    Kraus sets and is responsible for ``sum K†K = I`` (or for documenting a
    deliberately trace-decreasing map, as used for photon loss).
    """
    out = np.zeros_like(rho)
    for K in kraus:
        K = np.asarray(K, dtype=complex)
        out = out + apply_right(apply_left(rho, K, qubits), K.conj().T, qubits)
    return out


def reorder_qubits(rho: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """Permute qubits so that output qubit ``i`` is input qubit ``order[i]``."""
    n = _n_qubits(rho)
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of range(n)")
    t = _as_tensor(rho, n)
    perm = list(order) + [q + n for q in order]
    t = np.transpose(t, perm)
    return _from_tensor(t, n)


def partial_trace(rho: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Trace out every qubit not in ``keep`` (``keep`` order is preserved)."""
    n = _n_qubits(rho)
    keep = list(keep)
    drop = [q for q in range(n) if q not in keep]
    t = _as_tensor(rho, n)
    # move kept legs to the front on both row and column sides
    perm = keep + drop + [q + n for q in keep] + [q + n for q in drop]
    t = np.transpose(t, perm)
    dk, dd = 2 ** len(keep), 2 ** len(drop)
    t = t.reshape(dk, dd, dk, dd)
    return np.einsum("iaja->ij", t)


# --------------------------------------------------------------------------
# measurement and expectation values
# --------------------------------------------------------------------------

#: Unitary that rotates the eigenbasis of a Pauli operator onto the Z basis.
_TO_Z = {
    "Z": I2,
    "X": H,
    "Y": H @ S.conj().T,   # maps |+i> -> |0>, |-i> -> |1>
}


def _basis_rotation(basis: str) -> np.ndarray:
    try:
        return _TO_Z[basis.upper()]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ValueError(f"unknown measurement basis {basis!r}") from exc


def probabilities(rho: np.ndarray) -> np.ndarray:
    """Computational-basis outcome distribution (real, non-negative)."""
    p = np.real(np.diag(rho))
    return np.clip(p, 0.0, None)


def measure_qubit(rho: np.ndarray, qubit: int, basis: str,
                  rng: np.random.Generator) -> Tuple[int, np.ndarray, float]:
    """Projective measurement of one qubit in the X, Y or Z eigenbasis.

    Returns ``(outcome_bit, post_measurement_state, probability_of_outcome)``.
    The post-measurement state follows the projection postulate,
    ``rho -> P rho P / tr(P rho P)``, so a subsequent measurement of the same
    observable is deterministic.  ``rho`` may be subnormalised (photon loss);
    outcome probabilities are then computed conditional on detection.
    """
    R = _basis_rotation(basis)
    rot = apply_unitary(rho, R, [qubit])
    n = _n_qubits(rho)
    diag = probabilities(rot).reshape((2,) * n)
    axes = tuple(a for a in range(n) if a != qubit)
    p = diag.sum(axis=axes)          # length-2 marginal, may sum to < 1
    tot = float(p.sum())
    if tot <= 0:
        raise ValueError("measuring a state with zero trace")
    p0 = float(p[0]) / tot
    bit = 0 if rng.random() < p0 else 1
    proj = np.zeros((2, 2), dtype=complex)
    proj[bit, bit] = 1.0
    post = apply_unitary(rot, proj, [qubit])
    post_tr = float(np.real(np.trace(post)))
    if post_tr <= 0:  # pragma: no cover - guarded by the sampling above
        raise ValueError("projected onto a zero-probability subspace")
    post = post / post_tr
    post = apply_unitary(post, R.conj().T, [qubit])   # rotate back
    return bit, post, (p0 if bit == 0 else 1.0 - p0)


def measure_pauli(rho: np.ndarray, bases: Dict[int, str],
                  rng: np.random.Generator) -> Tuple[Dict[int, int], np.ndarray]:
    """Measure several qubits, each in its own Pauli eigenbasis."""
    outcomes: Dict[int, int] = {}
    state = rho
    for q in sorted(bases):
        bit, state, _ = measure_qubit(state, q, bases[q], rng)
        outcomes[q] = bit
    return outcomes, state


def expectation(rho: np.ndarray, bases: Dict[int, str]) -> float:
    """Exact expectation value of a tensor product of Paulis.

    ``bases={0:'X', 1:'X'}`` on a 2-qubit state returns ``<XX>``.  Qubits not
    listed are implicitly assigned the identity.
    """
    n = _n_qubits(rho)
    label = "".join(bases.get(q, "I") for q in range(n))
    op = pauli_string_op(label)
    return float(np.real(np.trace(op @ rho)))


def axis_op(theta: float) -> np.ndarray:
    """The observable ``cos(theta) Z + sin(theta) X``.

    A measurement along an axis in the x-z plane.  Needed because the CHSH
    test requires settings at 45 degrees to the Pauli axes -- ``Z`` and ``X``
    alone cannot violate the inequality maximally.
    """
    return math.cos(theta) * Z + math.sin(theta) * X


def measure_axis(rho: np.ndarray, qubit: int, theta: float,
                 rng: np.random.Generator) -> Tuple[int, np.ndarray, float]:
    """Projective measurement of ``cos(theta) Z + sin(theta) X`` on one qubit.

    Implemented by rotating the chosen axis onto ``Z`` with ``Ry(-theta)``,
    measuring, and rotating back, so it is the same projection postulate as
    :func:`measure_qubit` -- outcome ``0`` means eigenvalue ``+1``.
    """
    R = ry(-theta)
    rot = apply_unitary(rho, R, [qubit])
    bit, post, prob = measure_qubit(rot, qubit, "Z", rng)
    post = apply_unitary(post, R.conj().T, [qubit])
    return bit, post, prob


def expectation_axis(rho: np.ndarray, theta_a: float, theta_b: float,
                     qubits: Sequence[int] = (0, 1)) -> float:
    """Exact ``<(n_a . sigma) (x) (n_b . sigma)>`` for axes in the x-z plane."""
    n = _n_qubits(rho)
    ops = [I2] * n
    ops[qubits[0]] = axis_op(theta_a)
    ops[qubits[1]] = axis_op(theta_b)
    full = np.array([[1.0 + 0j]])
    for o in ops:
        full = np.kron(full, o)
    return float(np.real(np.trace(full @ rho)))


# --------------------------------------------------------------------------
# figures of merit
# --------------------------------------------------------------------------

#: CHSH measurement angles (radians, in the x-z plane) that maximally violate
#: the inequality for ``|Phi+>``: Alice at 0 and pi/2, Bob at pi/4 and -pi/4.
#: With these settings ``E(a, b) = cos(theta_a - theta_b)`` and the combination
#: ``S = E00 + E01 + E10 - E11`` reaches ``2 sqrt(2)``, the Tsirelson bound.
CHSH_SETTINGS = {
    "A0": 0.0,
    "A1": math.pi / 2,
    "B0": math.pi / 4,
    "B1": -math.pi / 4,
}


def chsh_value(rho: np.ndarray, qubits: Sequence[int] = (0, 1)) -> float:
    """Exact CHSH statistic ``S`` for a two-qubit state at the optimal settings.

    Any local hidden-variable model -- equivalently, any state Eve could have
    substituted after measuring, and any classical correlation she could have
    manufactured -- obeys ``|S| <= 2``.  Quantum mechanics allows up to
    ``2 sqrt(2) = 2.828``.  The gap between the two is the part of the
    detection engine that no adversary can talk her way around, however much
    computing power she has.
    """
    a0, a1 = CHSH_SETTINGS["A0"], CHSH_SETTINGS["A1"]
    b0, b1 = CHSH_SETTINGS["B0"], CHSH_SETTINGS["B1"]
    e00 = expectation_axis(rho, a0, b0, qubits)
    e01 = expectation_axis(rho, a0, b1, qubits)
    e10 = expectation_axis(rho, a1, b0, qubits)
    e11 = expectation_axis(rho, a1, b1, qubits)
    return e00 + e01 + e10 - e11


def fidelity_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    """``<psi| rho |psi>`` -- fidelity with a pure target state."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return float(np.real(psi.conj() @ rho @ psi))


def _pure_vector(rho: np.ndarray, tol: float = 1e-10) -> np.ndarray | None:
    """Return the state vector if ``rho`` is (numerically) pure, else ``None``."""
    if abs(purity(rho) - 1.0) > tol:
        return None
    w, v = np.linalg.eigh(rho)
    return v[:, int(np.argmax(w))]


def fidelity_states(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity ``(tr sqrt(sqrt(rho) sigma sqrt(rho)))**2``.

    Two formulations are used, for a numerical reason worth stating.  The
    textbook expression needs the square roots of the eigenvalues of
    ``sqrt(rho) sigma sqrt(rho)``, and ``d(sqrt x)/dx`` blows up at ``x = 0``:
    for a *rank-deficient* argument -- above all a pure state, which is the
    case this project compares against most often -- a rounding error of
    ``1e-16`` in an eigenvalue becomes ``1e-8`` in the fidelity.  So:

    * if either argument is pure, use the exact closed form
      ``F = <psi| sigma |psi>``, which involves no square root at all;
    * otherwise use ``F = (sum_i sqrt(lambda_i))**2`` over the eigenvalues of
      the product ``rho sigma`` (real and non-negative even though the product
      is not Hermitian).  This agrees with the ``sqrt``-based expression to
      ``1e-13`` on full-rank states and avoids forming ``sqrt(rho)``.
    """
    psi = _pure_vector(rho)
    if psi is not None:
        return float(np.real(psi.conj() @ sigma @ psi))
    phi = _pure_vector(sigma)
    if phi is not None:
        return float(np.real(phi.conj() @ rho @ phi))
    ev = np.clip(np.real(np.linalg.eigvals(rho @ sigma)), 0.0, None)
    return float(np.sum(np.sqrt(ev)) ** 2)


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(rho - sigma)
    return float(0.5 * np.sum(np.abs(ev)))


def is_density_matrix(rho: np.ndarray, tol: float = 1e-9) -> bool:
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if abs(float(np.real(np.trace(rho))) - 1.0) > 1e-6:
        return False
    return bool(np.min(np.real(np.linalg.eigvalsh(rho))) > -tol)


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence of a two-qubit state (0 = separable, 1 = Bell)."""
    if rho.shape != (4, 4):
        raise ValueError("concurrence is defined for two qubits")
    yy = np.kron(Y, Y)
    rho_tilde = yy @ rho.conj() @ yy
    ev = np.real(np.linalg.eigvals(rho @ rho_tilde))
    ev = np.sqrt(np.clip(np.sort(ev)[::-1], 0.0, None))
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def bell_fidelity_from_correlators(zz: float, xx: float, yy: float) -> float:
    """Fidelity with |Phi+> reconstructed from three correlators.

    ``F = (1 + <ZZ> + <XX> - <YY>) / 4`` for the ``|Phi+>`` state.  This is the
    estimator the detection engine uses, because ``<ZZ>``, ``<XX>`` and
    ``<YY>`` are exactly the quantities a verifier can measure on sacrificed
    EPR pairs with local projective measurements and a public discussion.
    """
    return (1.0 + zz + xx - yy) / 4.0


# --------------------------------------------------------------------------
# random objects (tests and randomised attacks)
# --------------------------------------------------------------------------

def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary of dimension ``2**n`` via QR of a Ginibre matrix."""
    dim = 2 ** n
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_ket(n: int, rng: np.random.Generator) -> np.ndarray:
    dim = 2 ** n
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return v / np.linalg.norm(v)
