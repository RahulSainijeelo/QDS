"""Backend contract tests.

Every test here runs against *each* backend that can be instantiated on this
machine, so installing Qiskit turns this file into a genuine cross-validation
of the two independent implementations without a single line changing.  With
only numpy installed it still pins the contract the protocol layer relies on.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from qds import backends
from qds import linalg as L

BACKENDS = backends.available_backends()


class TestBackendContract(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)

    def _each(self):
        """``(name, backend)`` for every backend installed on this machine."""
        return [(name, backends.get_backend(name)) for name in BACKENDS]

    def test_registry(self):
        self.assertIn("numpy", BACKENDS)
        with self.assertRaises(backends.BackendError):
            backends.get_backend("quantum-hopes-and-dreams")

    def test_zero_state(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                for n in (1, 2, 3):
                    rho = be.to_numpy(be.zero(n))
                    self.assertEqual(rho.shape, (2 ** n, 2 ** n))
                    self.assertAlmostEqual(float(np.real(rho[0, 0])), 1.0, places=12)
                    self.assertAlmostEqual(float(np.real(np.trace(rho))), 1.0, places=12)

    def test_from_ket_and_num_qubits(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.from_ket(L.ket("101"))
                self.assertEqual(be.num_qubits(st), 3)
                self.assertTrue(np.allclose(be.to_numpy(st), L.density(L.ket("101"))))

    def test_tensor_is_big_endian(self):
        """tensor(a, b) must put ``a`` in the leading factors."""
        for name, be in self._each():
            with self.subTest(backend=name):
                a = be.from_ket(L.ket("0"))
                b = be.from_ket(L.ket("1"))
                got = be.to_numpy(be.tensor(a, b))
                self.assertTrue(np.allclose(got, L.density(L.ket("01"))))

    def test_bell_pair_construction(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.zero(2)
                st = be.apply(st, L.H, [0])
                st = be.apply(st, L.CNOT, [0, 1])
                self.assertTrue(np.allclose(be.to_numpy(st), L.bell_state("Phi+"),
                                            atol=1e-12))

    def test_apply_on_a_specific_qubit(self):
        """X on qubit 1 of a 3-qubit register must flip only qubit 1."""
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.from_ket(L.ket("000"))
                st = be.apply(st, L.X, [1])
                self.assertTrue(np.allclose(be.to_numpy(st), L.density(L.ket("010")),
                                            atol=1e-12))

    def test_apply_multi_qubit_operator_ordering(self):
        """CNOT on [2, 0] controls with qubit 2 and targets qubit 0."""
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.from_ket(L.ket("001"))
                st = be.apply(st, L.CNOT, [2, 0])
                self.assertTrue(np.allclose(be.to_numpy(st), L.density(L.ket("101")),
                                            atol=1e-12))

    def test_apply_kraus(self):
        from qds.channel import depolarizing_kraus
        for name, be in self._each():
            with self.subTest(backend=name):
                for p in (0.0, 0.25, 1.0):
                    st = be.from_ket(L.ket("01"))
                    st = be.apply_kraus(st, depolarizing_kraus(p), [1])
                    want = L.apply_kraus(L.density(L.ket("01")),
                                         depolarizing_kraus(p), [1])
                    self.assertTrue(np.allclose(be.to_numpy(st), want, atol=1e-12), p)

    def test_measure_is_a_projective_measurement(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                for basis, psi, want in (("Z", L.ket("1"), 1),
                                         ("X", L.bb84_state(1, 1), 1),
                                         ("Y", L.S @ L.H @ L.ket("0"), 0)):
                    st = be.from_ket(psi)
                    bit, post, prob = be.measure(st, 0, basis, self.rng)
                    self.assertEqual(bit, want, basis)
                    self.assertAlmostEqual(prob, 1.0, places=12)
                    again, _, p2 = be.measure(post, 0, basis, self.rng)
                    self.assertEqual(again, bit)
                    self.assertAlmostEqual(p2, 1.0, places=12)

    def test_measure_collapses_the_partner(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.zero(2)
                st = be.apply(st, L.H, [0])
                st = be.apply(st, L.CNOT, [0, 1])
                b0, post, _ = be.measure(st, 0, "Z", self.rng)
                b1, _, p = be.measure(post, 1, "Z", self.rng)
                self.assertEqual(b0, b1)
                self.assertAlmostEqual(p, 1.0, places=12)

    def test_expectation(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.zero(2)
                st = be.apply(st, L.H, [0])
                st = be.apply(st, L.CNOT, [0, 1])
                self.assertAlmostEqual(be.expectation(st, {0: "Z", 1: "Z"}), 1.0, places=12)
                self.assertAlmostEqual(be.expectation(st, {0: "X", 1: "X"}), 1.0, places=12)
                self.assertAlmostEqual(be.expectation(st, {0: "Y", 1: "Y"}), -1.0, places=12)
                self.assertAlmostEqual(be.expectation(st, {1: "Z"}), 0.0, places=12)

    def test_expectation_on_a_subset_of_a_larger_register(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.tensor(be.from_ket(L.BELL_STATES["Phi+"]), be.from_ket(L.ket("1")))
                self.assertAlmostEqual(be.expectation(st, {0: "Z", 1: "Z"}), 1.0, places=12)
                self.assertAlmostEqual(be.expectation(st, {2: "Z"}), -1.0, places=12)

    def test_partial_trace(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.tensor(be.from_ket(L.BELL_STATES["Phi+"]), be.from_ket(L.ket("1")))
                keep_bell = be.to_numpy(be.partial_trace(st, [0, 1]))
                self.assertTrue(np.allclose(keep_bell, L.bell_state("Phi+"), atol=1e-12))
                keep_last = be.to_numpy(be.partial_trace(st, [2]))
                self.assertTrue(np.allclose(keep_last, L.density(L.ket("1")), atol=1e-12))
                half = be.to_numpy(be.partial_trace(st, [0]))
                self.assertTrue(np.allclose(half, np.eye(2) / 2, atol=1e-12))

    def test_partial_trace_respects_requested_order(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                st = be.tensor(be.from_ket(L.ket("0")), be.from_ket(L.ket("1")))
                ab = be.to_numpy(be.partial_trace(st, [0, 1]))
                ba = be.to_numpy(be.partial_trace(st, [1, 0]))
                self.assertTrue(np.allclose(ab, L.density(L.ket("01")), atol=1e-12))
                self.assertTrue(np.allclose(ba, L.density(L.ket("10")), atol=1e-12))

    def test_fidelity_helper(self):
        for name, be in self._each():
            with self.subTest(backend=name):
                a = be.from_ket(L.ket("01"))
                b = be.from_ket(L.ket("01"))
                c = be.from_ket(L.ket("00"))
                self.assertAlmostEqual(be.fidelity(a, b), 1.0, places=10)
                self.assertAlmostEqual(be.fidelity(a, c), 0.0, places=10)

    def test_teleportation_runs_identically_on_every_backend(self):
        """The full teleportation primitive, backend by backend.

        This is the operation the protocol is built on, so it gets a contract
        test of its own: q0 = payload, (q1, q2) = EPR pair, Bell measurement on
        (q0, q1), Pauli correction ``Z^u X^v`` on q2.
        """
        for name, be in self._each():
            with self.subTest(backend=name):
                for psi in (L.ket("0"), L.ket("1"), L.bb84_state(1, 0),
                            L.bb84_state(1, 1)):
                    st = be.tensor(be.from_ket(psi), be.zero(2))
                    st = be.apply(st, L.H, [1])
                    st = be.apply(st, L.CNOT, [1, 2])
                    st = be.apply(st, L.CNOT, [0, 1])
                    st = be.apply(st, L.H, [0])
                    u, st, _ = be.measure(st, 0, "Z", self.rng)
                    v, st, _ = be.measure(st, 1, "Z", self.rng)
                    if v:
                        st = be.apply(st, L.X, [2])
                    if u:
                        st = be.apply(st, L.Z, [2])
                    out = be.to_numpy(be.partial_trace(st, [2]))
                    self.assertAlmostEqual(L.fidelity_pure(out, psi), 1.0, places=10)


@unittest.skipUnless(len(BACKENDS) > 1, "only one backend available")
class TestCrossValidation(unittest.TestCase):
    """Element-by-element agreement between numpy and Qiskit.

    Skipped when Qiskit is not installed.  ``python -m qds.cli selfcheck
    --cross-validate`` runs the same comparison over the whole protocol.
    """

    def test_random_circuits_agree(self):
        rng_seed = 5150
        results = []
        for name in BACKENDS:
            be = backends.get_backend(name)
            rng = np.random.default_rng(rng_seed)
            st = be.zero(3)
            for _ in range(12):
                k = int(rng.integers(1, 3))
                qubits = list(rng.choice(3, size=k, replace=False))
                U = L.random_unitary(k, rng)
                st = be.apply(st, U, [int(q) for q in qubits])
            results.append(be.to_numpy(st))
        for other in results[1:]:
            self.assertTrue(np.allclose(results[0], other, atol=1e-12))

    def test_measurement_statistics_agree(self):
        counts = []
        for name in BACKENDS:
            be = backends.get_backend(name)
            rng = np.random.default_rng(777)
            ones = 0
            for _ in range(400):
                st = be.from_ket(L.ry(0.8) @ L.ket("0"))
                bit, _, _ = be.measure(st, 0, "Z", rng)
                ones += bit
            counts.append(ones)
        # identical RNG stream and identical probabilities => identical samples
        self.assertEqual(len(set(counts)), 1, counts)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
