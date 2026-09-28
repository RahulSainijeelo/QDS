"""Tests for the exact density-matrix core.

Every assertion here is checked against a *defining property* (a commutation
relation, a projection postulate, an analytically known constant) rather than
against a number copied from somewhere else.  If a test needs a numeric
constant, the constant is derived in the test itself.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from qds import linalg as L


class TestOperatorAlgebra(unittest.TestCase):
    def test_pauli_relations(self):
        self.assertTrue(np.allclose(L.X @ L.X, L.I2))
        self.assertTrue(np.allclose(L.Y @ L.Y, L.I2))
        self.assertTrue(np.allclose(L.Z @ L.Z, L.I2))
        self.assertTrue(np.allclose(L.X @ L.Y, 1j * L.Z))
        self.assertTrue(np.allclose(L.Y @ L.Z, 1j * L.X))
        self.assertTrue(np.allclose(L.Z @ L.X, 1j * L.Y))
        # anticommutation
        for A, B in ((L.X, L.Y), (L.Y, L.Z), (L.Z, L.X)):
            self.assertTrue(np.allclose(A @ B + B @ A, np.zeros((2, 2))))

    def test_unitarity(self):
        for U in (L.X, L.Y, L.Z, L.H, L.S, L.SDG, L.T, L.CNOT, L.SWAP,
                  L.rx(0.7), L.ry(-1.3), L.rz(2.1), L.phase(0.4)):
            self.assertTrue(np.allclose(U @ U.conj().T, np.eye(U.shape[0])))

    def test_hadamard_conjugates_x_and_z(self):
        self.assertTrue(np.allclose(L.H @ L.Z @ L.H, L.X))
        self.assertTrue(np.allclose(L.H @ L.X @ L.H, L.Z))

    def test_rotation_generators(self):
        """Rx/Ry/Rz must be exp(-i theta P / 2)."""
        for gen, rot in ((L.X, L.rx), (L.Y, L.ry), (L.Z, L.rz)):
            for theta in (0.0, 0.3, -1.1, math.pi):
                want = (math.cos(theta / 2) * L.I2
                        - 1j * math.sin(theta / 2) * gen)
                self.assertTrue(np.allclose(rot(theta), want), (gen, theta))

    def test_pauli_string_op(self):
        self.assertTrue(np.allclose(L.pauli_string_op("XZ"), np.kron(L.X, L.Z)))
        self.assertTrue(np.allclose(L.pauli_string_op("III"), np.eye(8)))


class TestStatePreparation(unittest.TestCase):
    def test_ket_is_big_endian(self):
        """Qubit 0 is the most significant factor: |01> is index 1, |10> is 2."""
        self.assertEqual(int(np.argmax(np.abs(L.ket("01")))), 1)
        self.assertEqual(int(np.argmax(np.abs(L.ket("10")))), 2)
        self.assertTrue(np.allclose(L.ket("01"), np.kron(L.ket("0"), L.ket("1"))))

    def test_bb84_states_form_two_mutually_unbiased_bases(self):
        states = {(a, c): L.bb84_state(a, c) for a in (0, 1) for c in (0, 1)}
        # within a basis: orthogonal
        for a in (0, 1):
            ip = abs(states[(a, 0)].conj() @ states[(a, 1)])
            self.assertAlmostEqual(ip, 0.0, places=12)
        # across bases: overlap exactly 1/2 in probability -- unbiased
        for c0 in (0, 1):
            for c1 in (0, 1):
                p = abs(states[(0, c0)].conj() @ states[(1, c1)]) ** 2
                self.assertAlmostEqual(p, 0.5, places=12)

    def test_bell_states_are_maximally_entangled(self):
        for name in L.BELL_STATES:
            rho = L.bell_state(name)
            self.assertTrue(L.is_density_matrix(rho))
            self.assertAlmostEqual(L.purity(rho), 1.0, places=12)
            self.assertAlmostEqual(L.concurrence(rho), 1.0, places=12)
            half = L.partial_trace(rho, [0])
            self.assertTrue(np.allclose(half, np.eye(2) / 2))

    def test_bell_states_are_orthonormal(self):
        names = list(L.BELL_STATES)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                ip = abs(L.BELL_STATES[a].conj() @ L.BELL_STATES[b]) ** 2
                self.assertAlmostEqual(ip, 1.0 if i == j else 0.0, places=12)


class TestEvolution(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20240917)

    def test_apply_left_and_right_against_explicit_kron(self):
        rng = self.rng
        for n, qubits in ((2, [0]), (2, [1]), (3, [1]), (3, [0, 2]), (3, [2, 0])):
            rho = L.density(L.random_ket(n, rng))
            k = len(qubits)
            U = L.random_unitary(k, rng)
            got = L.apply_unitary(rho, U, qubits)
            # build the full operator by hand
            full = np.eye(2 ** n, dtype=complex)
            if k == 1:
                ops = [L.I2] * n
                ops[qubits[0]] = U
                full = ops[0]
                for o in ops[1:]:
                    full = np.kron(full, o)
            else:
                # place U on `qubits` by permuting to the front
                order = list(qubits) + [q for q in range(n) if q not in qubits]
                perm = np.zeros((2 ** n, 2 ** n), dtype=complex)
                for idx in range(2 ** n):
                    bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
                    new = [bits[q] for q in order]
                    j = 0
                    for b in new:
                        j = (j << 1) | b
                    perm[j, idx] = 1.0
                blk = np.kron(U, np.eye(2 ** (n - k), dtype=complex))
                full = perm.conj().T @ blk @ perm
            self.assertTrue(np.allclose(got, full @ rho @ full.conj().T),
                            (n, qubits))

    def test_cnot_makes_a_bell_pair(self):
        rho = L.density(L.ket("00"))
        rho = L.apply_unitary(rho, L.H, [0])
        rho = L.apply_unitary(rho, L.CNOT, [0, 1])
        self.assertAlmostEqual(L.fidelity_pure(rho, L.BELL_STATES["Phi+"]),
                               1.0, places=12)

    def test_cnot_direction_matters(self):
        """CNOT on [0,1] controls with qubit 0; on [1,0] with qubit 1."""
        rho = L.density(L.ket("01"))
        a = L.apply_unitary(rho, L.CNOT, [0, 1])
        b = L.apply_unitary(rho, L.CNOT, [1, 0])
        self.assertAlmostEqual(L.fidelity_pure(a, L.ket("01")), 1.0, places=12)
        self.assertAlmostEqual(L.fidelity_pure(b, L.ket("11")), 1.0, places=12)

    def test_swap(self):
        rho = L.density(L.ket("01"))
        self.assertAlmostEqual(
            L.fidelity_pure(L.apply_unitary(rho, L.SWAP, [0, 1]), L.ket("10")),
            1.0, places=12)

    def test_partial_trace_matches_kron_structure(self):
        rng = self.rng
        a = L.density(L.random_ket(1, rng))
        b = L.density(L.random_ket(2, rng))
        joint = np.kron(a, b)
        self.assertTrue(np.allclose(L.partial_trace(joint, [0]), a))
        self.assertTrue(np.allclose(L.partial_trace(joint, [1, 2]), b))

    def test_partial_trace_preserves_requested_order(self):
        rng = self.rng
        rho = L.density(L.random_ket(3, rng))
        ab = L.partial_trace(rho, [0, 2])
        ba = L.partial_trace(rho, [2, 0])
        self.assertTrue(np.allclose(ba, L.reorder_qubits(ab, [1, 0])))

    def test_reorder_qubits_is_an_involution_on_swaps(self):
        rng = self.rng
        rho = L.density(L.random_ket(3, rng))
        once = L.reorder_qubits(rho, [2, 1, 0])
        twice = L.reorder_qubits(once, [2, 1, 0])
        self.assertTrue(np.allclose(twice, rho))

    def test_kraus_sum_preserves_trace_when_complete(self):
        rng = self.rng
        rho = L.density(L.random_ket(2, rng))
        kraus = [math.sqrt(0.7) * L.I2, math.sqrt(0.3) * L.X]
        out = L.apply_kraus(rho, kraus, [1])
        self.assertAlmostEqual(float(np.real(np.trace(out))), 1.0, places=12)
        self.assertTrue(L.is_density_matrix(out))


class TestMeasurement(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(31337)

    def test_eigenstates_measure_deterministically(self):
        cases = [
            (L.ket("0"), "Z", 0), (L.ket("1"), "Z", 1),
            (L.H @ L.ket("0"), "X", 0), (L.H @ L.ket("1"), "X", 1),
            (L.S @ L.H @ L.ket("0"), "Y", 0), (L.S @ L.H @ L.ket("1"), "Y", 1),
        ]
        for psi, basis, want in cases:
            for _ in range(20):
                bit, _, p = L.measure_qubit(L.density(psi), 0, basis, self.rng)
                self.assertEqual(bit, want, (basis, want))
                self.assertAlmostEqual(p, 1.0, places=12)

    def test_conjugate_basis_is_uniform(self):
        """A Z eigenstate measured in X gives 1/2, exactly."""
        for basis, psi in (("X", L.ket("0")), ("Z", L.H @ L.ket("0")),
                           ("Y", L.ket("0")), ("Y", L.H @ L.ket("0"))):
            _, _, p = L.measure_qubit(L.density(psi), 0, basis, self.rng)
            self.assertAlmostEqual(p, 0.5, places=12)

    def test_repeated_measurement_is_idempotent(self):
        """The projection postulate: re-measuring gives the same answer."""
        rng = self.rng
        for basis in ("Z", "X", "Y"):
            rho = L.density(L.random_ket(2, rng))
            bit, post, _ = L.measure_qubit(rho, 1, basis, rng)
            for _ in range(5):
                again, post, p = L.measure_qubit(post, 1, basis, rng)
                self.assertEqual(again, bit)
                self.assertAlmostEqual(p, 1.0, places=12)

    def test_measurement_collapses_a_bell_pair_into_agreement(self):
        rng = self.rng
        for _ in range(50):
            rho = L.bell_state("Phi+")
            b0, post, _ = L.measure_qubit(rho, 0, "Z", rng)
            b1, _, p = L.measure_qubit(post, 1, "Z", rng)
            self.assertEqual(b0, b1)
            self.assertAlmostEqual(p, 1.0, places=12)

    def test_sampled_frequencies_match_born_rule(self):
        rng = self.rng
        theta = 0.9
        rho = L.density(L.ry(theta) @ L.ket("0"))
        p1 = math.sin(theta / 2) ** 2
        n = 4000
        ones = sum(L.measure_qubit(rho, 0, "Z", rng)[0] for _ in range(n))
        # 5 sigma of a binomial
        tol = 5 * math.sqrt(p1 * (1 - p1) / n)
        self.assertLess(abs(ones / n - p1), tol)

    def test_expectation_matches_sampling(self):
        rng = self.rng
        rho = L.bell_state("Phi+")
        self.assertAlmostEqual(L.expectation(rho, {0: "Z", 1: "Z"}), 1.0, places=12)
        self.assertAlmostEqual(L.expectation(rho, {0: "X", 1: "X"}), 1.0, places=12)
        self.assertAlmostEqual(L.expectation(rho, {0: "Y", 1: "Y"}), -1.0, places=12)
        self.assertAlmostEqual(L.expectation(rho, {0: "Z"}), 0.0, places=12)

    def test_measure_pauli_multi_qubit(self):
        rng = self.rng
        outcomes, post = L.measure_pauli(L.bell_state("Psi+"), {0: "Z", 1: "Z"}, rng)
        self.assertNotEqual(outcomes[0], outcomes[1])
        self.assertTrue(L.is_density_matrix(post))


class TestAxisMeasurementAndCHSH(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4242)

    def test_axis_op_is_a_unit_observable(self):
        for theta in (0.0, 0.3, math.pi / 4, -math.pi / 3, math.pi):
            A = L.axis_op(theta)
            self.assertTrue(np.allclose(A @ A, L.I2))          # eigenvalues +-1
            self.assertTrue(np.allclose(A, A.conj().T))        # Hermitian
        self.assertTrue(np.allclose(L.axis_op(0.0), L.Z))
        self.assertTrue(np.allclose(L.axis_op(math.pi / 2), L.X))

    def test_measure_axis_reduces_to_z_and_x(self):
        rng = self.rng
        for _ in range(20):
            bit, _, p = L.measure_axis(L.density(L.ket("1")), 0, 0.0, rng)
            self.assertEqual(bit, 1)
            self.assertAlmostEqual(p, 1.0, places=12)
            bit, _, p = L.measure_axis(L.density(L.bb84_state(1, 0)), 0,
                                       math.pi / 2, rng)
            self.assertEqual(bit, 0)
            self.assertAlmostEqual(p, 1.0, places=12)

    def test_correlator_is_cosine_of_the_angle(self):
        """For |Phi+> and axes in the x-z plane, E(a,b) = cos(a - b)."""
        rho = L.bell_state("Phi+")
        for a in (0.0, 0.4, math.pi / 2):
            for b in (-0.7, 0.0, math.pi / 4, 1.2):
                self.assertAlmostEqual(L.expectation_axis(rho, a, b),
                                       math.cos(a - b), places=12)

    def test_tsirelson_bound_is_attained(self):
        self.assertAlmostEqual(L.chsh_value(L.bell_state("Phi+")),
                               2 * math.sqrt(2), places=12)

    def test_local_bound_holds_for_separable_states(self):
        rng = self.rng
        for _ in range(30):
            a = L.density(L.random_ket(1, rng))
            b = L.density(L.random_ket(1, rng))
            self.assertLessEqual(abs(L.chsh_value(np.kron(a, b))), 2.0 + 1e-12)
        # classically correlated mixture, still local
        mix = 0.5 * (L.density(L.ket("00")) + L.density(L.ket("11")))
        self.assertLessEqual(abs(L.chsh_value(mix)), 2.0 + 1e-12)
        self.assertAlmostEqual(L.chsh_value(np.eye(4) / 4), 0.0, places=12)

    def test_chsh_degrades_linearly_with_depolarising_noise(self):
        from qds.channel import depolarizing_kraus
        for p in (0.0, 0.1, 0.25, 0.5, 1.0):
            rho = L.apply_kraus(L.bell_state("Phi+"), depolarizing_kraus(p), [1])
            self.assertAlmostEqual(L.chsh_value(rho),
                                   2 * math.sqrt(2) * (1 - p), places=12)

    def test_sampled_chsh_converges(self):
        rng = self.rng
        rho = L.bell_state("Phi+")
        terms = {}
        for na in ("A0", "A1"):
            for nb in ("B0", "B1"):
                ta, tb = L.CHSH_SETTINGS[na], L.CHSH_SETTINGS[nb]
                tot = 0
                for _ in range(600):
                    b0, post, _ = L.measure_axis(rho, 0, ta, rng)
                    b1, _, _ = L.measure_axis(post, 1, tb, rng)
                    tot += 1 if b0 == b1 else -1
                terms[na + nb] = tot / 600
        s = terms["A0B0"] + terms["A0B1"] + terms["A1B0"] - terms["A1B1"]
        self.assertLess(abs(s - 2 * math.sqrt(2)), 0.3)
        self.assertGreater(s, 2.0)      # a real violation from sampled data


class TestFiguresOfMerit(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(99)

    def test_fidelity_pure_agrees_with_uhlmann_on_pure_targets(self):
        rng = self.rng
        for _ in range(20):
            rho = L.density(L.random_ket(2, rng))
            psi = L.random_ket(2, rng)
            self.assertAlmostEqual(L.fidelity_pure(rho, psi),
                                   L.fidelity_states(rho, L.density(psi)),
                                   places=10)

    def test_fidelity_bounds_and_symmetry(self):
        rng = self.rng
        for _ in range(20):
            a = L.density(L.random_ket(2, rng))
            b = L.density(L.random_ket(2, rng))
            f = L.fidelity_states(a, b)
            self.assertGreaterEqual(f, -1e-12)
            self.assertLessEqual(f, 1.0 + 1e-9)
            self.assertAlmostEqual(f, L.fidelity_states(b, a), places=10)
            self.assertAlmostEqual(L.fidelity_states(a, a), 1.0, places=10)

    def test_trace_distance_properties(self):
        rng = self.rng
        a = L.density(L.ket("0"))
        b = L.density(L.ket("1"))
        self.assertAlmostEqual(L.trace_distance(a, b), 1.0, places=12)
        self.assertAlmostEqual(L.trace_distance(a, a), 0.0, places=12)
        # Fuchs-van de Graaf: 1 - sqrt(F) <= D <= sqrt(1 - F)
        for _ in range(20):
            x = L.density(L.random_ket(1, rng))
            y = L.density(L.random_ket(1, rng))
            f = L.fidelity_states(x, y)
            d = L.trace_distance(x, y)
            self.assertLessEqual(1 - math.sqrt(max(f, 0.0)) - 1e-9, d)
            self.assertLessEqual(d, math.sqrt(max(1 - f, 0.0)) + 1e-9)

    def test_bell_fidelity_from_correlators(self):
        """F = (1 + <ZZ> + <XX> - <YY>)/4 must equal the direct overlap."""
        from qds.channel import depolarizing_kraus, dephasing_kraus
        for kraus in (depolarizing_kraus(0.0), depolarizing_kraus(0.2),
                      dephasing_kraus(0.3), depolarizing_kraus(1.0)):
            rho = L.apply_kraus(L.bell_state("Phi+"), kraus, [1])
            est = L.bell_fidelity_from_correlators(
                L.expectation(rho, {0: "Z", 1: "Z"}),
                L.expectation(rho, {0: "X", 1: "X"}),
                L.expectation(rho, {0: "Y", 1: "Y"}))
            self.assertAlmostEqual(est, L.fidelity_pure(rho, L.BELL_STATES["Phi+"]),
                                   places=12)

    def test_concurrence_threshold_matches_fidelity_half(self):
        """A Werner-like state is entangled exactly when F > 1/2."""
        from qds.channel import depolarizing_kraus
        for p in (0.0, 0.2, 1 / 3, 0.4, 2 / 3, 0.7, 1.0):
            rho = L.apply_kraus(L.bell_state("Phi+"), depolarizing_kraus(p), [1])
            f = L.fidelity_pure(rho, L.BELL_STATES["Phi+"])
            c = L.concurrence(rho)
            self.assertEqual(f > 0.5 + 1e-9, c > 1e-9, (p, f, c))

    def test_random_unitary_is_unitary(self):
        rng = self.rng
        for n in (1, 2, 3):
            U = L.random_unitary(n, rng)
            self.assertTrue(np.allclose(U @ U.conj().T, np.eye(2 ** n)))


class TestFastPaths(unittest.TestCase):
    """The specialised channel routines must be rewrites, not approximations.

    ``apply_depolarizing`` and friends exist only because the generic Kraus
    route was the simulator's bottleneck -- roughly an order of magnitude of
    the runtime went into axis bookkeeping for eight-flop contractions.  A
    faster path that is even slightly different is worse than no path at all,
    because every security threshold in the framework is calibrated against
    numbers these functions produce.  So each one is pinned to the generic
    implementation it replaces at machine precision, over every qubit position
    of registers up to four qubits and over the full parameter range including
    the degenerate endpoints 0 and 1.
    """

    def setUp(self):
        self.rng = np.random.default_rng(20250920)

    def _rho(self, n):
        A = (self.rng.normal(size=(2 ** n, 2 ** n))
             + 1j * self.rng.normal(size=(2 ** n, 2 ** n)))
        r = A @ A.conj().T
        return r / np.trace(r).real

    def _kraus_depolarizing(self, p):
        return [math.sqrt(1.0 - 3.0 * p / 4.0) * L.I2, math.sqrt(p / 4.0) * L.X,
                math.sqrt(p / 4.0) * L.Y, math.sqrt(p / 4.0) * L.Z]

    def test_pauli_channels_match_the_generic_kraus_route(self):
        for n in (1, 2, 3, 4):
            for q in range(n):
                rho = self._rho(n)
                for p in (0.0, 1e-3, 0.017, 0.25, 0.5, 0.913, 1.0):
                    with self.subTest(n=n, q=q, p=p):
                        self.assertTrue(np.allclose(
                            L.apply_kraus(rho, self._kraus_depolarizing(p), [q]),
                            L.apply_depolarizing(rho, q, p), atol=1e-14))
                        self.assertTrue(np.allclose(
                            L.apply_kraus(rho, [math.sqrt(1 - p) * L.I2,
                                                math.sqrt(p) * L.Z], [q]),
                            L.apply_dephasing(rho, q, p), atol=1e-14))
                        self.assertTrue(np.allclose(
                            L.apply_kraus(rho, [math.sqrt(1 - p) * L.I2,
                                                math.sqrt(p) * L.X], [q]),
                            L.apply_bitflip(rho, q, p), atol=1e-14))

    def test_asymmetric_pauli_channel_matches_kraus(self):
        for n in (1, 2, 3):
            for q in range(n):
                rho = self._rho(n)
                for _ in range(8):
                    px, py, pz = self.rng.dirichlet([1, 1, 1, 1])[:3]
                    kraus = [math.sqrt(max(1 - px - py - pz, 0.0)) * L.I2,
                             math.sqrt(px) * L.X, math.sqrt(py) * L.Y,
                             math.sqrt(pz) * L.Z]
                    self.assertTrue(np.allclose(
                        L.apply_kraus(rho, kraus, [q]),
                        L.apply_pauli_channel(rho, q, px, py, pz), atol=1e-14))

    def test_depolarizing_is_the_uniform_pauli_channel(self):
        """The replacement form and the twirl form are the same map."""
        rho = self._rho(3)
        for p in (0.03, 0.4, 1.0):
            self.assertTrue(np.allclose(L.apply_depolarizing(rho, 1, p),
                                        L.apply_pauli_channel(rho, 1, p / 4, p / 4, p / 4),
                                        atol=1e-14), p)

    def test_apply_1q_matches_apply_unitary(self):
        gates = [L.H, L.X, L.Y, L.Z, L.S, L.SDG, L.T, L.ry(0.7), L.rz(-1.3), L.Z @ L.X]
        for n in (1, 2, 3, 4):
            for q in range(n):
                rho = self._rho(n)
                for U in gates + [L.random_unitary(1, self.rng) for _ in range(3)]:
                    with self.subTest(n=n, q=q):
                        self.assertTrue(np.allclose(
                            L.apply_right(L.apply_left(rho, U, [q]), U.conj().T, [q]),
                            L.apply_1q(rho, U, q), atol=1e-14))

    def test_project_z_matches_sandwiching_and_reports_joint_probability(self):
        for n in (1, 2, 3, 4):
            for q in range(n):
                rho = self._rho(n)
                for b in (0, 1):
                    P = np.zeros((2, 2), dtype=complex)
                    P[b, b] = 1.0
                    ref = L.apply_right(L.apply_left(rho, P, [q]), P, [q])
                    prob, got = L.project_z(rho, q, b)
                    with self.subTest(n=n, q=q, b=b):
                        self.assertTrue(np.allclose(ref, got, atol=1e-14))
                        self.assertAlmostEqual(prob, float(np.real(np.trace(ref))),
                                               places=13)

    def test_chained_projections_give_the_joint_probability(self):
        """Two chained calls must yield p(u, v), not p(v | u).

        The Bell readout in the distribution layer relies on this: it projects
        qubit 0, then projects the unnormalised remainder, and reads the
        second probability as the joint one.
        """
        rho = self._rho(3)
        total = 0.0
        for u in (0, 1):
            _, ru = L.project_z(rho, 0, u)
            for v in (0, 1):
                puv, _ = L.project_z(ru, 1, v)
                total += puv
        self.assertAlmostEqual(total, 1.0, places=13)

    def test_fast_paths_do_not_mutate_their_input(self):
        rho = self._rho(3)
        snapshot = rho.copy()
        L.apply_depolarizing(rho, 1, 0.3)
        L.apply_dephasing(rho, 2, 0.3)
        L.apply_bitflip(rho, 0, 0.3)
        L.apply_pauli_channel(rho, 1, 0.1, 0.2, 0.05)
        L.apply_1q(rho, L.H, 2)
        L.project_z(rho, 0, 1)
        self.assertTrue(np.array_equal(rho, snapshot))

    def test_fast_paths_preserve_trace_and_hermiticity(self):
        rho = self._rho(4)
        for out in (L.apply_depolarizing(rho, 2, 0.31),
                    L.apply_dephasing(rho, 3, 0.22),
                    L.apply_bitflip(rho, 1, 0.44),
                    L.apply_pauli_channel(rho, 0, 0.1, 0.2, 0.05),
                    L.apply_1q(rho, L.H, 1)):
            self.assertAlmostEqual(float(np.real(np.trace(out))), 1.0, places=13)
            self.assertTrue(np.allclose(out, out.conj().T, atol=1e-14))
            self.assertTrue(L.is_density_matrix(out))

    def test_zero_strength_is_the_identity_map(self):
        rho = self._rho(2)
        for out in (L.apply_depolarizing(rho, 0, 0.0),
                    L.apply_dephasing(rho, 1, 0.0),
                    L.apply_bitflip(rho, 0, 0.0),
                    L.apply_pauli_channel(rho, 1, 0.0, 0.0, 0.0)):
            self.assertTrue(np.allclose(out, rho, atol=1e-15))

    def test_full_depolarizing_erases_the_qubit(self):
        rho = L.bell_state("Phi+")
        out = L.apply_depolarizing(rho, 0, 1.0)
        self.assertTrue(np.allclose(out, np.eye(4) / 4.0, atol=1e-14))

    def test_invalid_arguments_are_rejected(self):
        rho = self._rho(2)
        with self.assertRaises(ValueError):
            L.apply_depolarizing(rho, 5, 0.1)
        with self.assertRaises(ValueError):
            L.apply_pauli_channel(rho, 0, 0.6, 0.6, 0.0)
        with self.assertRaises(ValueError):
            L.apply_1q(rho, np.eye(4), 0)
        with self.assertRaises(ValueError):
            L.project_z(rho, 0, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
