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
    return apply_right(apply_left(rho, U, qubits), U.conj().T, qubits)


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
