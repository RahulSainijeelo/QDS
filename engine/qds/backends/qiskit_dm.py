"""Qiskit backend: the same protocol expressed with ``qiskit.quantum_info``.

Why this exists
---------------
The numpy backend is the fast path, but a reviewer should not have to trust
it.  This module implements the identical :class:`~qds.backends.base.Backend`
contract using Qiskit's own ``DensityMatrix`` / ``Operator`` primitives, so
the full protocol -- Bell pair creation, teleportation, Pauli correction,
projective verification, every attack -- can be re-run on Qiskit and compared
against the numpy path element by element::

    pip install qiskit
    python3 -m unittest discover -s tests -p 'test_*.py'

Only ``qiskit.quantum_info`` is used.  There is deliberately no dependency on
``qiskit-aer``, on the primitives (``Sampler``/``Estimator``) or on the
transpiler, because those are the parts of the Qiskit API that have churned
across major versions; ``DensityMatrix``, ``Operator`` and ``Pauli`` have been
stable since Qiskit Terra 0.x and work unchanged on Qiskit 1.x and 2.x.

Qubit ordering
--------------
Qiskit is little-endian (its qubit 0 is the *least* significant tensor
factor); this project is big-endian (qubit 0 is the *most* significant).  The
adapter therefore maps project qubit ``q`` to Qiskit qubit ``n - 1 - q``.
With that relabelling the two conventions produce *identical* density-matrix
index orderings, so :meth:`to_numpy` is a straight array hand-off and
cross-validation is an exact array comparison rather than a permutation
puzzle.  For a multi-qubit operator the ``qargs`` list is additionally
reversed, because Qiskit reads ``qargs[0]`` as the least significant factor of
the operator's own matrix while this project writes ``qubits[0]`` as the most
significant.

.. warning::
   Unlike every other module in this package, the code here was not executed
   during development (the build environment had no network access and so no
   Qiskit install).  It is written against the documented, long-stable
   ``quantum_info`` API and is checked by the Qiskit cross-validation tests in
   ``tests/test_backends.py``, which assert agreement with the numpy backend to
   1e-12.  Run the test suite once after installing Qiskit.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from .. import linalg as L
from .base import Backend


class QiskitDensityMatrixBackend(Backend):
    name = "qiskit"

    def __init__(self) -> None:
        from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

        self._DensityMatrix = DensityMatrix
        self._Operator = Operator
        self._partial_trace = partial_trace

    # -- index translation -------------------------------------------------
    @staticmethod
    def _qargs(n: int, qubits: Sequence[int]) -> list:
        """Project qubit list -> Qiskit ``qargs`` (relabelled and reversed)."""
        return [n - 1 - q for q in reversed(list(qubits))]

    # -- preparation -------------------------------------------------------
    def zero(self, n: int) -> Any:
        return self._DensityMatrix.from_int(0, dims=(2,) * n)

    def from_ket(self, psi: np.ndarray) -> Any:
        from qiskit.quantum_info import Statevector

        psi = np.asarray(psi, dtype=complex).reshape(-1)
        psi = psi / np.linalg.norm(psi)
        return self._DensityMatrix(Statevector(psi))

    def tensor(self, a: Any, b: Any) -> Any:
        # Qiskit's ``A.tensor(B)`` has data ``kron(A.data, B.data)``, i.e. A's
        # qubits occupy the leading tensor factors -- exactly this project's
        # ``tensor`` semantics.
        return a.tensor(b)

    # -- evolution ---------------------------------------------------------
    def apply(self, state: Any, U: np.ndarray, qubits: Sequence[int]) -> Any:
        n = self.num_qubits(state)
        return state.evolve(self._Operator(np.asarray(U, dtype=complex)),
                            qargs=self._qargs(n, qubits))

    def apply_kraus(self, state: Any, kraus: Iterable[np.ndarray],
                    qubits: Sequence[int]) -> Any:
        """``sum_i K_i rho K_i†`` as an explicit sum of Qiskit evolutions.

        Each term is computed by Qiskit (``DensityMatrix.evolve`` with a
        non-unitary ``Operator`` performs ``K rho K†``); only the outer sum is
        done here.  This avoids constructing ``Kraus``/``SuperOp`` objects,
        some versions of which validate trace preservation -- and one of the
        maps this project needs (reset-then-prepare, used when Eve substitutes
        a qubit) is deliberately trace preserving only on its support.
        """
        n = self.num_qubits(state)
        qargs = self._qargs(n, qubits)
        total = None
        for K in kraus:
            term = state.evolve(self._Operator(np.asarray(K, dtype=complex)),
                                qargs=qargs)
            total = term.data if total is None else total + term.data
        if total is None:
            return state
        return self._DensityMatrix(total, dims=(2,) * n)

    # -- extraction --------------------------------------------------------
    def measure(self, state: Any, qubit: int, basis: str,
                rng: np.random.Generator) -> Tuple[int, Any, float]:
        """Projective measurement via a basis rotation plus Qiskit's own
        single-qubit ``measure``.

        Only ever called on one qubit at a time, so the outcome label Qiskit
        returns is a single character and there is no label-ordering
        ambiguity.  The RNG is this project's generator, seeded from the run
        configuration, so runs stay reproducible across backends.
        """
        n = self.num_qubits(state)
        R = L._basis_rotation(basis)             # rotates the eigenbasis onto Z
        qargs = self._qargs(n, [qubit])
        rot = state.evolve(self._Operator(R), qargs=qargs)

        probs = np.asarray(rot.probabilities(qargs=qargs), dtype=float)
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("measuring a state with zero trace")
        p0 = float(probs[0]) / total
        bit = 0 if rng.random() < p0 else 1

        proj = np.zeros((2, 2), dtype=complex)
        proj[bit, bit] = 1.0
        post = rot.evolve(self._Operator(proj), qargs=qargs)
        tr = float(np.real(np.trace(post.data)))
        if tr <= 0:  # pragma: no cover
            raise ValueError("projected onto a zero-probability subspace")
        post = self._DensityMatrix(post.data / tr, dims=(2,) * n)
        post = post.evolve(self._Operator(R.conj().T), qargs=qargs)
        return bit, post, (p0 if bit == 0 else 1.0 - p0)

    def expectation(self, state: Any, bases: Dict[int, str]) -> float:
        n = self.num_qubits(state)
        qubits = sorted(bases)
        label = "".join(bases[q] for q in qubits)
        op = self._Operator(L.pauli_string_op(label))
        val = state.expectation_value(op, qargs=self._qargs(n, qubits))
        return float(np.real(val))

    def partial_trace(self, state: Any, keep: Sequence[int]) -> Any:
        n = self.num_qubits(state)
        keep = list(keep)
        drop = [q for q in range(n) if q not in keep]
        reduced = self._partial_trace(state, self._qargs(n, drop))
        if keep != sorted(keep):
            # Qiskit returns the survivors in ascending original index order;
            # restore the caller's requested order.
            order = [sorted(keep).index(q) for q in keep]
            data = L.reorder_qubits(np.asarray(reduced.data, dtype=complex), order)
            return self._DensityMatrix(data, dims=(2,) * len(keep))
        return reduced

    def to_numpy(self, state: Any) -> np.ndarray:
        return np.asarray(state.data, dtype=complex)

    def num_qubits(self, state: Any) -> int:
        return int(state.num_qubits)
