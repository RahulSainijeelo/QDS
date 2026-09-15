"""The backend contract.

A backend is a small set of primitives that are enough to express the whole
TQDS protocol: prepare, evolve unitarily, apply a Kraus map, measure
projectively, take Pauli expectation values, and trace out.  ``State`` is
deliberately opaque -- for the numpy backend it is a density matrix, for the
Qiskit backend it is a ``qiskit.quantum_info.DensityMatrix``.

Qubit ordering
--------------
Qubit ``0`` is the leading tensor factor, so an n-qubit basis label
``"b0 b1 ... b_{n-1}"`` has index ``sum(b_q << (n - 1 - q))``.  Backends are
required to expose :meth:`Backend.to_numpy` in *this* convention, which is
what makes cross-validation between backends a straight array comparison.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np


class BackendError(RuntimeError):
    """Raised for unknown or unusable backends."""


class Backend(abc.ABC):
    """Minimal exact-simulation interface used by the protocol layer."""

    #: short identifier, appears in every emitted report
    name: str = "abstract"

    # -- preparation -------------------------------------------------------
    @abc.abstractmethod
    def zero(self, n: int) -> Any:
        """``|0...0>`` on ``n`` qubits."""

    @abc.abstractmethod
    def from_ket(self, psi: np.ndarray) -> Any:
        """State from a (normalised) state vector in this module's ordering."""

    @abc.abstractmethod
    def tensor(self, a: Any, b: Any) -> Any:
        """Join two registers; ``a``'s qubits come first."""

    # -- evolution ---------------------------------------------------------
    @abc.abstractmethod
    def apply(self, state: Any, U: np.ndarray, qubits: Sequence[int]) -> Any:
        """Apply unitary ``U`` to ``qubits`` (``qubits[0]`` = leading factor)."""

    @abc.abstractmethod
    def apply_kraus(self, state: Any, kraus: Iterable[np.ndarray],
                    qubits: Sequence[int]) -> Any:
        """Apply the CP map ``sum_i K_i . K_i†`` to ``qubits``."""

    # -- extraction --------------------------------------------------------
    @abc.abstractmethod
    def measure(self, state: Any, qubit: int, basis: str,
                rng: np.random.Generator) -> Tuple[int, Any, float]:
        """Projective measurement in the ``X``/``Y``/``Z`` eigenbasis.

        Returns ``(bit, post_state, probability_of_that_outcome)`` with the
        post-measurement state renormalised (projection postulate).
        """

    @abc.abstractmethod
    def expectation(self, state: Any, bases: Dict[int, str]) -> float:
        """Expectation value of a tensor product of Paulis."""

    @abc.abstractmethod
    def partial_trace(self, state: Any, keep: Sequence[int]) -> Any:
        """Trace out all qubits except ``keep`` (order preserved)."""

    @abc.abstractmethod
    def to_numpy(self, state: Any) -> np.ndarray:
        """Density matrix as a numpy array in this module's qubit ordering."""

    @abc.abstractmethod
    def num_qubits(self, state: Any) -> int:
        """Number of qubits in ``state``."""

    # -- shared helpers ----------------------------------------------------
    def fidelity(self, state: Any, target_ket: np.ndarray) -> float:
        """``<target|rho|target>``."""
        rho = self.to_numpy(state)
        psi = np.asarray(target_ket, dtype=complex).reshape(-1)
        psi = psi / np.linalg.norm(psi)
        return float(np.real(psi.conj() @ rho @ psi))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
