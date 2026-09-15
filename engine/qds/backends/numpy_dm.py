"""Exact density-matrix backend built on :mod:`qds.linalg` and numpy only.

This is the reference implementation and the one every shipped number comes
from.  Registers in the TQDS protocol are at most four qubits, so a step is a
16x16 matrix multiplication -- there is no reason to approximate anything.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from .. import linalg as L
from .base import Backend


class NumpyDensityMatrixBackend(Backend):
    name = "numpy"

    # -- preparation -------------------------------------------------------
    def zero(self, n: int) -> np.ndarray:
        rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def from_ket(self, psi: np.ndarray) -> np.ndarray:
        return L.density(psi)

    def tensor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.kron(a, b)

    # -- evolution ---------------------------------------------------------
    def apply(self, state: np.ndarray, U: np.ndarray,
              qubits: Sequence[int]) -> np.ndarray:
        return L.apply_unitary(state, U, qubits)

    def apply_kraus(self, state: np.ndarray, kraus: Iterable[np.ndarray],
                    qubits: Sequence[int]) -> np.ndarray:
        return L.apply_kraus(state, kraus, qubits)

    # -- extraction --------------------------------------------------------
    def measure(self, state: np.ndarray, qubit: int, basis: str,
                rng: np.random.Generator) -> Tuple[int, np.ndarray, float]:
        return L.measure_qubit(state, qubit, basis, rng)

    def expectation(self, state: np.ndarray, bases: Dict[int, str]) -> float:
        return L.expectation(state, bases)

    def partial_trace(self, state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
        return L.partial_trace(state, keep)

    def to_numpy(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=complex)

    def num_qubits(self, state: np.ndarray) -> int:
        return int(round(np.log2(state.shape[0])))
