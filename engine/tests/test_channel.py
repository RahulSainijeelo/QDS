"""Tests for the channel models.

The noise floor is load-bearing: every statistical threshold in the detection
engine is set relative to ``NoiseModel.predicted_qber()``.  So the analytic
formula is checked against the *exactly propagated density matrix* rather than
against a plausible-looking number, and the interventions are checked against
closed-form disturbance expressions derived in the docstrings.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from qds import channel as C
from qds import linalg as L


def exact_qber(nm: C.NoiseModel) -> float:
    """Honest per-check error rate, averaged over the four BB84 states.

    Computed by propagating the density matrix through exactly the same error
    sources ``predicted_qber`` claims to model -- no sampling, no fitting.
    """
    total = 0.0
    for basis in (0, 1):
        for bit in (0, 1):
            target = L.bb84_state(basis, bit)
            rho = nm.apply_channel(L.density(target), 0)
            rho = nm.apply_memory(rho, 0, 1)
            if nm.gate_error:
                for _ in range(nm.GATE_SLOTS_PER_CHECK):
                    rho = L.apply_kraus(rho, C.depolarizing_kraus(nm.gate_error), [0])
            e = 1.0 - L.fidelity_pure(rho, target)
            pm = nm.measurement_error
            total += e * (1 - pm) + (1 - e) * pm
    return total / 4.0


class TestKrausSets(unittest.TestCase):
    def _assert_trace_preserving(self, kraus, label):
        s = sum(np.asarray(K).conj().T @ np.asarray(K) for K in kraus)
        self.assertTrue(np.allclose(s, np.eye(s.shape[0]), atol=1e-12), label)

    def test_every_channel_is_trace_preserving(self):
        for p in (0.0, 0.05, 0.3, 1.0):
            self._assert_trace_preserving(C.depolarizing_kraus(p), f"depol {p}")
            self._assert_trace_preserving(C.dephasing_kraus(p), f"dephase {p}")
            self._assert_trace_preserving(C.bitflip_kraus(p), f"bitflip {p}")
            self._assert_trace_preserving(C.amplitude_damping_kraus(p), f"damp {p}")
        self._assert_trace_preserving(C.pauli_kraus(0.1, 0.2, 0.05), "pauli")

    def test_depolarizing_parameter_convention(self):
        """rho -> (1-p) rho + p I/2, exactly, for every input state."""
        rng = np.random.default_rng(1)
        for p in (0.0, 0.13, 0.5, 1.0):
            for psi in (L.ket("0"), L.bb84_state(1, 1), L.random_ket(1, rng)):
                rho = L.density(psi)
                got = L.apply_kraus(rho, C.depolarizing_kraus(p), [0])
                want = (1 - p) * rho + p * np.eye(2) / 2
                self.assertTrue(np.allclose(got, want, atol=1e-13), (p,))

    def test_depolarizing_flip_probability_is_half_p(self):
        for p in (0.0, 0.02, 0.1, 0.4, 1.0):
            for basis in (0, 1):
                target = L.bb84_state(basis, 0)
                out = L.apply_kraus(L.density(target), C.depolarizing_kraus(p), [0])
                self.assertAlmostEqual(1 - L.fidelity_pure(out, target),
                                       p / 2, places=13)

    def test_dephasing_is_basis_selective(self):
        """Z outcomes immune, X outcomes flip with probability p exactly."""
        for p in (0.0, 0.07, 0.25, 0.5):
            z = L.apply_kraus(L.density(L.bb84_state(0, 0)), C.dephasing_kraus(p), [0])
            x = L.apply_kraus(L.density(L.bb84_state(1, 0)), C.dephasing_kraus(p), [0])
            self.assertAlmostEqual(1 - L.fidelity_pure(z, L.bb84_state(0, 0)),
                                   0.0, places=14)
            self.assertAlmostEqual(1 - L.fidelity_pure(x, L.bb84_state(1, 0)),
                                   p, places=14)

    def test_amplitude_damping_drives_to_ground(self):
        out = L.apply_kraus(L.density(L.ket("1")), C.amplitude_damping_kraus(1.0), [0])
        self.assertAlmostEqual(L.fidelity_pure(out, L.ket("0")), 1.0, places=13)

    def test_pauli_kraus_rejects_impossible_weights(self):
        with self.assertRaises(ValueError):
            C.pauli_kraus(0.6, 0.6, 0.0)
        with self.assertRaises(ValueError):
            C.depolarizing_kraus(1.5)


class TestNoiseModel(unittest.TestCase):
    def test_ideal_is_exactly_noiseless(self):
        self.assertTrue(C.IDEAL.is_ideal)
        self.assertEqual(C.IDEAL.predicted_qber(), 0.0)
        self.assertAlmostEqual(exact_qber(C.IDEAL), 0.0, places=14)

    def test_memory_dephasing_composes_in_closed_form(self):
        nm = C.NoiseModel(memory_dephasing=0.03)
        for k in (1, 2, 5, 17):
            closed = nm.apply_memory(L.density(L.bb84_state(1, 0)), 0, intervals=k)
            stepwise = L.density(L.bb84_state(1, 0))
            for _ in range(k):
                stepwise = L.apply_kraus(stepwise, C.dephasing_kraus(0.03), [0])
            self.assertTrue(np.allclose(closed, stepwise, atol=1e-13), k)

    def test_misalignment_is_a_coherent_rotation(self):
        for theta in (0.0, 0.01, 0.04, 0.3):
            nm = C.NoiseModel(misalignment=theta)
            rho = nm.apply_channel(L.density(L.ket("0")), 0)
            self.assertAlmostEqual(float(np.real(rho[1, 1])),
                                   math.sin(theta / 2) ** 2, places=14)

    def test_predicted_qber_matches_exact_propagation(self):
        cases = {
            "lab": C.LAB_GRADE,
            "field": C.FIELD_GRADE,
            "channel": C.NoiseModel(channel_depolarizing=0.10),
            "memory": C.NoiseModel(memory_dephasing=0.20),
            "readout": C.NoiseModel(measurement_error=0.05),
            "misalign": C.NoiseModel(misalignment=0.25),
            "gates": C.NoiseModel(gate_error=0.01),
            "all": C.NoiseModel(0.02, 0.01, 0.003, 0.005, 0.03),
        }
        for label, nm in cases.items():
            analytic, exact = nm.predicted_qber(), exact_qber(nm)
            self.assertLess(abs(analytic - exact), 2e-3, label)
            self.assertLess(abs(analytic - exact) / max(exact, 1e-12), 0.06, label)

    def test_presets_leave_headroom_below_the_forgery_bound(self):
        """The security condition eta < T < 1/4 must be satisfiable."""
        for nm in (C.LAB_GRADE, C.FIELD_GRADE):
            self.assertLess(nm.predicted_qber(), 0.25)
            self.assertLess(nm.predicted_qber(), 0.10)   # room for a real threshold

    def test_readout_flip_rate(self):
        rng = np.random.default_rng(2)
        nm = C.NoiseModel(measurement_error=0.2)
        n = 20000
        flips = sum(1 for _ in range(n) if nm.flip_readout(0, rng) != 0)
        self.assertLess(abs(flips / n - 0.2), 0.02)
        clean = C.NoiseModel()
        self.assertEqual(clean.flip_readout(1, rng), 1)


class TestLossModel(unittest.TestCase):
    def test_sampled_yield_matches_analytic_yield(self):
        rng = np.random.default_rng(3)
        for lm in (C.LossModel(),
                   C.LossModel(0.55, 0.7, 5e-5),
                   C.LossModel(0.9, 0.9, 0.01)):
            n = 40000
            hits = sum(1 for _ in range(n) if lm.sample(rng) != "lost")
            self.assertLess(abs(hits / n - lm.yield_), 0.01)

    def test_outcome_labels(self):
        rng = np.random.default_rng(4)
        always = C.LossModel(1.0, 1.0, 0.0)
        never = C.LossModel(0.0, 1.0, 0.0)
        dark = C.LossModel(0.0, 1.0, 1.0)
        self.assertEqual(always.sample(rng), "click")
        self.assertEqual(never.sample(rng), "lost")
        self.assertEqual(dark.sample(rng), "dark")


class TestInterventions(unittest.TestCase):
    """What each attack does to the observables the engine actually monitors."""

    def setUp(self):
        self.rng = np.random.default_rng(20260914)

    def _stats(self, rho):
        zz = L.expectation(rho, {0: "Z", 1: "Z"})
        xx = L.expectation(rho, {0: "X", 1: "X"})
        yy = L.expectation(rho, {0: "Y", 1: "Y"})
        return (L.chsh_value(rho),
                L.bell_fidelity_from_correlators(zz, xx, yy),
                L.concurrence(rho))

    def _average(self, make, trials):
        acc = np.zeros(3)
        for _ in range(trials):
            rho = make().on_transit(L.bell_state("Phi+"), 1, self.rng, probe=None)
            self.assertAlmostEqual(float(np.real(np.trace(rho))), 1.0, places=12)
            self.assertTrue(L.is_density_matrix(rho))
            acc += np.array(self._stats(rho))
        return acc / trials

    def test_no_intervention_preserves_everything(self):
        s, f, c = self._average(C.NoIntervention, 1)
        self.assertAlmostEqual(s, 2 * math.sqrt(2), places=12)
        self.assertAlmostEqual(f, 1.0, places=12)
        self.assertAlmostEqual(c, 1.0, places=12)

    def test_intercept_resend_destroys_the_bell_violation(self):
        for basis in ("random", "Z", "X", "breidbart"):
            s, f, c = self._average(lambda b=basis: C.InterceptResend(b), 300)
            self.assertLessEqual(abs(s), 2.0 + 0.05, basis)
            self.assertLess(abs(f - 0.5), 0.05, basis)
            self.assertLess(c, 0.05, basis)

    def test_partial_intercept_resend_scales(self):
        s, f, c = self._average(lambda: C.InterceptResend("random", 0.5), 400)
        self.assertGreater(f, 0.60)
        self.assertLess(f, 0.85)

    def test_entanglement_breaking_is_total(self):
        s, f, c = self._average(C.EntanglementBreaking, 1)
        self.assertAlmostEqual(s, 0.0, places=12)
        self.assertAlmostEqual(f, 0.25, places=12)   # F with a random qubit
        self.assertAlmostEqual(c, 0.0, places=12)

    def test_collective_depolarizing_matches_its_parameter(self):
        for p in (0.0, 0.05, 0.3):
            s, f, c = self._average(lambda p=p: C.CollectiveDepolarizing(p), 1)
            self.assertAlmostEqual(s, 2 * math.sqrt(2) * (1 - p), places=12)
            self.assertAlmostEqual(f, 1 - 3 * p / 4, places=12)

    def test_basis_biased_probe_is_asymmetric(self):
        """px only: X-basis checks fail, Z-basis checks do not."""
        probe = C.BasisBiasedProbe(pz=0.0, px=0.12)
        errs = {}
        for basis in (0, 1):
            target = L.bb84_state(basis, 0)
            out = probe.on_transit(L.density(target), 0, self.rng)
            errs[basis] = 1 - L.fidelity_pure(out, target)
        self.assertAlmostEqual(errs[0], 0.12, places=12)   # X flips Z outcomes
        self.assertAlmostEqual(errs[1], 0.0, places=12)
        pooled = (errs[0] + errs[1]) / 2
        self.assertAlmostEqual(pooled, 0.06, places=12)
        # the point of the test: pooled 6% hides under a 10% threshold while
        # one basis sits at 12% and is over it
        self.assertLess(pooled, 0.10)
        self.assertGreater(max(errs.values()), 0.10)

    def test_coherent_probe_information_disturbance_tradeoff(self):
        """e_X = sin^2(theta/2) and Eve's distinguishability = |sin theta|.

        The two are tied by ``d = 2 sqrt(e (1 - e))``, so any nonzero
        information gain forces a strictly positive error rate.  That
        inequality is the physical content of the alarm.
        """
        for theta in (0.0, 0.1, 0.2, 0.4, math.pi / 4, math.pi / 2):
            probe = C.CoherentProbe(theta)
            self.assertEqual(probe.probe_qubits(), 1)
            errs = []
            for basis in (0, 1):
                target = L.bb84_state(basis, 0)
                joint = np.kron(L.density(target), L.density(L.ket("0")))
                out = probe.on_transit(joint, 0, self.rng, probe=1)
                errs.append(max(0.0, 1 - L.fidelity_pure(
                    L.partial_trace(out, [0]), target)))
            self.assertAlmostEqual(errs[0], 0.0, places=12)
            self.assertAlmostEqual(errs[1], math.sin(theta / 2) ** 2, places=12)

            probe_states = []
            for bit in (0, 1):
                joint = np.kron(L.density(L.ket([bit])), L.density(L.ket("0")))
                out = probe.on_transit(joint, 0, self.rng, probe=1)
                probe_states.append(L.partial_trace(out, [1]))
            d = L.trace_distance(*probe_states)
            e = errs[1]
            self.assertAlmostEqual(d, abs(math.sin(theta)), places=12)
            self.assertAlmostEqual(d, 2 * math.sqrt(e * (1 - e)), places=12)

    def test_coherent_probe_needs_no_disturbance_only_when_useless(self):
        zero = C.CoherentProbe(0.0)
        rho = np.kron(L.bell_state("Phi+"), L.density(L.ket("0")))
        out = zero.on_transit(rho, 1, self.rng, probe=2)
        self.assertTrue(np.allclose(out, rho, atol=1e-13))

    def test_replacement_helpers_stay_physical(self):
        for make in (lambda r: C._replace_qubit(r, 1, L.ket("1")),
                     lambda r: C._replace_qubit_mixed(r, 1)):
            out = make(L.bell_state("Phi+"))
            self.assertAlmostEqual(float(np.real(np.trace(out))), 1.0, places=13)
            self.assertTrue(L.is_density_matrix(out))
            self.assertAlmostEqual(L.concurrence(out), 0.0, places=13)

    def test_registry_round_trip(self):
        for name in ("none", "intercept_resend", "entanglement_breaking",
                     "collective_depolarizing", "coherent_probe", "basis_biased"):
            iv = C.make_intervention(name)
            self.assertEqual(iv.name, name)
            self.assertIsInstance(iv.to_dict(), dict)
            self.assertIn("description", iv.to_dict())
        with self.assertRaises(ValueError):
            C.make_intervention("wishful_thinking")

    def test_registry_passes_keyword_arguments(self):
        iv = C.make_intervention("coherent_probe", theta=0.42)
        self.assertAlmostEqual(iv.theta, 0.42)
        iv = C.make_intervention("collective_depolarizing", p=0.07)
        self.assertAlmostEqual(iv.p, 0.07)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
