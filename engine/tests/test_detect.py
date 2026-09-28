"""Tests for the detection engine.

Two of these matter more than the rest.

:class:`TestPrivilegeBoundary` reads this package's own source off disk and
asserts that the names of the simulator's privileged quantities appear nowhere
in it.  A detector that consults the true channel fidelity, or the entanglement
witnesses, or Eve's actual probe state, is not detecting anything -- it is
reading the answer and would pass every other test in this file while being
worthless in deployment.  The grep is the only mechanism that catches that, so
it is part of the security argument rather than a style check.

:class:`TestFalseAlarmCalibration` generates runs in which the null hypothesis
is true *by construction* -- Bernoulli draws at exactly the declared floor,
with no quantum simulation involved -- and checks that the engine's family-wise
alarm rate comes out at or under the alpha it advertises.  Without this, "the
engine flagged the attack" is not evidence of anything: a detector that flags
everything catches every attack.

The rule-level tests build :class:`~qds.protocol.verification.VerificationReport`
objects directly rather than running sessions.  That is a deliberate trade.  A
real session costs about 1.6 seconds, which rules out the thousands of trials
calibration needs, and synthetic reports let the null be exactly true instead
of approximately true -- so a false alarm is unambiguously the engine's fault
and not a disagreement between ``SessionConfig.spec_rate()`` and the sampler.
Real sessions are used in :class:`TestLiveSession`, where the thing under test
is the wiring rather than the statistics.
"""

from __future__ import annotations

import json
import math
import pathlib
import types
import unittest

import numpy as np

from qds.channel import LossModel, NoiseModel, PRESET_NOISE, make_intervention
from qds.detect import (Evidence, DetectionEngine, PREMISE_RULES,
                        analyse_session, attribute, default_rules)
from qds.detect import estimators as est
from qds.detect import stats
from qds.detect import thresholds as th
from qds.protocol.distribution import DistributionConfig
from qds.protocol.keys import Slot
from qds.protocol.session import QDSSession, SessionConfig
from qds.protocol.verification import (SlotCheck, VerificationPolicy,
                                       VerificationReport)

DETECT_DIR = pathlib.Path(est.__file__).parent


# --------------------------------------------------------------------------
# synthetic evidence
# --------------------------------------------------------------------------

def make_policy(honest: float = 0.01, s_a: float = 0.05,
                s_v: float = 0.12) -> VerificationPolicy:
    return VerificationPolicy(honest_error_rate=honest, s_a=s_a, s_v=s_v,
                              min_checks=1)


def make_report(specs, *, verifier: str = "bob", level: str = "accept",
                threshold: float = 0.05, accepted: bool = True,
                identity_ok: bool = True, mac_ok: bool = True,
                fresh=True, measured: bool = True,
                policy: VerificationPolicy = None) -> VerificationReport:
    """Build a report from ``(j, basis, mismatch, status)`` tuples.

    ``observed`` is set to ``None`` for a missing slot, because
    :attr:`SlotCheck.counted` is defined as ``observed is not None`` -- a
    synthetic check that sets ``status="missing"`` but leaves an observed bit
    in place would be counted, and the loss tests would then be measuring
    something that cannot happen.
    """
    checks = []
    for i, (j, basis, mismatch, status) in enumerate(specs):
        observed = None if status == "missing" else (1 if mismatch else 0)
        checks.append(SlotCheck(slot=Slot(int(j), 0, i), basis=int(basis),
                                expected=0, observed=observed, status=status))
    verdict = None if fresh is None else types.SimpleNamespace(
        ok=bool(fresh), reasons=[])
    return VerificationReport(
        verifier=verifier, signer="alice", key_id="pk-test", message=b"\xb0",
        level=level, policy=policy or make_policy(), threshold=threshold,
        checks=checks, identity_ok=identity_ok, identity_problems=[],
        mac_ok=mac_ok, mac_problems=[], freshness=verdict, measured=measured,
        accepted=accepted, reasons=[], elapsed_seconds=0.0)


def honest_specs(rng, n: int, rate_z: float, rate_x: float, positions: int = 8):
    """``n`` checks, half in each basis, mismatches drawn at the declared rate."""
    out = []
    for i in range(n):
        basis = i % 2
        rate = rate_x if basis else rate_z
        out.append((i % positions, basis, bool(rng.random() < rate), "ok"))
    return out


def make_calibration(k: int, n: int):
    return types.SimpleNamespace(decoy_mismatches=int(k), decoys_measured=int(n))


def synthetic_evidence(rng, *, n_per_verifier: int = 96, verifiers=("bob",),
                       spec_z: float = 0.010, spec_x: float = 0.014,
                       declared_yield: float = 0.78,
                       declared_dark: float = 0.004,
                       n_slots: int = 384, alpha_family: float = 0.01,
                       s_a: float = 0.05, s_v: float = 0.12,
                       decoys: int = 64) -> Evidence:
    """An evidence bundle in which every null hypothesis is exactly true."""
    spec = 0.5 * (spec_z + spec_x)
    policy = make_policy(spec, s_a, s_v)
    reports = []
    for name in verifiers:
        reports.append(make_report(
            honest_specs(rng, n_per_verifier, spec_z, spec_x),
            verifier=name, level="accept", threshold=s_a, policy=policy))

    logs = {}
    for name in verifiers:
        log = []
        for _ in range(n_slots):
            u = rng.random()
            if u < declared_dark:
                log.append("dark")
            elif u < declared_yield:
                log.append("click")
            else:
                log.append("lost")
        logs[name] = log

    corrections = {name: [(int(rng.integers(2)), int(rng.integers(2)))
                          for _ in range(256)] for name in verifiers}
    k_decoy = int(rng.binomial(decoys, spec))
    return Evidence(
        reports=reports, calibrations=[make_calibration(k_decoy, decoys)],
        detector_logs=logs, corrections=corrections,
        spec_rate=spec, spec_rate_z=spec_z, spec_rate_x=spec_x,
        accept_threshold=s_a, transfer_threshold=s_v,
        declared_yield=declared_yield, declared_dark_fraction=declared_dark,
        alpha_family=alpha_family, declaration_source="spec_sheet")


# --------------------------------------------------------------------------
# the privilege boundary
# --------------------------------------------------------------------------

class TestPrivilegeBoundary(unittest.TestCase):
    """The detection package must not be able to see the simulator's truth."""

    #: Names of the privileged quantities the distribution layer records for
    #: the benefit of the test suite.  Searched as attribute accesses and
    #: identifiers, not as bare words, so that a docstring can still explain
    #: why an oracle would be wrong without tripping the test.
    FORBIDDEN = (
        "DistributionOracle",
        ".oracle",
        "exact_qber",
        "mean_chsh",
        "mean_bell_fidelity",
        "mean_concurrence",
        "information_disturbance",
        "announcement_error_rate",
    )

    def test_no_privileged_symbols_in_source(self):
        offenders = []
        for path in sorted(DETECT_DIR.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in self.FORBIDDEN:
                if token in text:
                    offenders.append(f"{path.name}: {token}")
        self.assertEqual(offenders, [], "\n".join(
            ["detection code referenced the simulator's privileged view:"]
            + offenders))

    def test_summary_oracle_is_never_read(self):
        """``DistributionResult.summary()`` carries ground truth; stay out."""
        engine_src = (DETECT_DIR / "engine.py").read_text(encoding="utf-8")
        self.assertNotIn("summary()", engine_src)

    def test_evidence_from_session_carries_no_truth(self):
        cfg = SessionConfig(message_bits=8, L=8, distribution=DistributionConfig(
            noise=PRESET_NOISE["lab"], check_pairs=16, decoy_slots=16))
        session = QDSSession(cfg, seed=3)
        session.run(message=b"\xb0")
        ev = Evidence.from_session(session)
        blob = json.dumps(DetectionEngine().run(ev).to_dict())
        for token in ("chsh", "concurrence", "fidelity", "exact_qber",
                      "oracle", "disturbance"):
            self.assertNotIn(token, blob.lower(),
                             f"{token!r} leaked into the detection report")


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

class TestEstimators(unittest.TestCase):

    def test_zero_mismatches_does_not_certify_the_channel(self):
        """The motivating case for using an exact interval at all."""
        e = est.estimate_rate("t", 0, 30)
        self.assertEqual(e.value, 0.0)
        self.assertEqual(e.ci_low, 0.0)
        # A normal approximation gives [0, 0] here, which would let the engine
        # declare a barely-inspected channel error-free.
        self.assertGreater(e.ci_high, 0.09)
        self.assertLess(e.ci_high, 0.12)

    def test_interval_brackets_the_point_estimate(self):
        for k, n in ((0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (3, 200)):
            e = est.estimate_rate("t", k, n)
            self.assertLessEqual(e.ci_low, e.value + 1e-12, (k, n))
            self.assertGreaterEqual(e.ci_high, e.value - 1e-12, (k, n))

    def test_empty_sample_is_not_a_zero_rate(self):
        e = est.estimate_rate("t", 0, 0)
        self.assertEqual(e.n, 0)
        self.assertEqual(e.value, 0.0)     # a formatting convenience...
        self.assertEqual(e.ci_high, 1.0)   # ...but the interval says nothing

    def test_bad_counts_rejected(self):
        with self.assertRaises(ValueError):
            est.estimate_rate("t", 5, 3)
        with self.assertRaises(ValueError):
            est.estimate_rate("t", -1, 3)

    def test_denominators_differ_and_are_labelled(self):
        specs = [(0, 0, False, "ok"), (0, 1, True, "ok"),
                 (1, 0, False, "missing"), (1, 1, False, "dark")]
        r = make_report(specs)
        pooled = est.pooled_rate([r])
        y = est.yield_estimate([r])
        self.assertEqual((pooled.k, pooled.n), (1, 3))   # missing not counted
        self.assertEqual((y.k, y.n), (3, 4))             # missing counted below
        self.assertNotEqual(pooled.conditioned_on, y.conditioned_on)
        self.assertIn("excludes lost", pooled.conditioned_on)
        self.assertIn("includes lost", y.conditioned_on)

    def test_dark_counts_are_counted_as_checks(self):
        """A dark count produces a bit, so it has to enter the error rate."""
        r = make_report([(0, 0, True, "dark")])
        self.assertEqual(est.pooled_rate([r]).n, 1)
        self.assertEqual(est.dark_estimate([r]).k, 1)

    def test_correction_histogram_index_order(self):
        counts = est.correction_histogram([(0, 0), (0, 1), (1, 0), (1, 1),
                                           (1, 1)])
        self.assertEqual(counts, [1, 1, 1, 2])

    def test_by_basis_split(self):
        specs = [(0, 0, False, "ok"), (0, 0, False, "ok"),
                 (0, 1, True, "ok"), (0, 1, True, "ok")]
        by = est.rates_by_basis([make_report(specs)])
        self.assertEqual((by["Z"].k, by["Z"].n), (0, 2))
        self.assertEqual((by["X"].k, by["X"].n), (2, 2))

    def test_fisher_exact_against_hand_computed_table(self):
        # 2x2 table [[3, 1], [1, 3]]: the classic tea-tasting arrangement,
        # whose two-sided exact p is 2 * C(4,3)C(4,1)/C(8,4) = 32/70 + ...
        p = est.fisher_exact_two_sided(3, 4, 1, 4)
        self.assertAlmostEqual(p, 0.4857142857142857, places=12)

    def test_fisher_exact_is_symmetric_and_bounded(self):
        for k1, n1, k2, n2 in ((0, 20, 10, 20), (5, 50, 5, 50), (1, 7, 6, 9)):
            a = est.fisher_exact_two_sided(k1, n1, k2, n2)
            b = est.fisher_exact_two_sided(k2, n2, k1, n1)
            self.assertAlmostEqual(a, b, places=12)
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)

    def test_fisher_exact_identical_tables_give_p_one(self):
        self.assertAlmostEqual(est.fisher_exact_two_sided(5, 50, 5, 50), 1.0,
                               places=12)

    def test_fisher_exact_empty_side(self):
        self.assertEqual(est.fisher_exact_two_sided(0, 0, 3, 10), 1.0)

    def test_fisher_beats_normal_approximation_on_small_tables(self):
        """Where the z test would have claimed significance and is wrong."""
        k1, n1, k2, n2 = 4, 10, 0, 10
        p_exact = est.fisher_exact_two_sided(k1, n1, k2, n2)
        z = est._two_proportion_z(k1, n1, k2, n2)
        p_normal = 2.0 * stats.normal_sf(abs(z))
        self.assertGreater(p_exact, p_normal)
        self.assertGreater(p_exact, 0.05)     # exact: not significant
        self.assertLess(p_normal, 0.05)       # normal: significant. wrong.

    def test_homogeneity_dof_is_groups_minus_one(self):
        stat, dof, p = est.homogeneity_chi_square([(5, 50), (6, 50), (4, 50)])
        self.assertEqual(dof, 2)
        self.assertIsNotNone(stat)
        self.assertGreater(p, 0.5)

    def test_homogeneity_ignores_empty_groups(self):
        stat, dof, p = est.homogeneity_chi_square([(5, 50), (6, 50), (0, 0)])
        self.assertEqual(dof, 1)

    def test_homogeneity_degenerate_tables_have_no_test(self):
        self.assertEqual(est.homogeneity_chi_square([(0, 50), (0, 50)]),
                         (None, 0, None))
        self.assertEqual(est.homogeneity_chi_square([(5, 50)]), (None, 0, None))

    def test_homogeneity_detects_a_split(self):
        stat, dof, p = est.homogeneity_chi_square([(0, 100), (40, 100)])
        self.assertLess(p, 1e-9)

    def test_rate_difference_empty_side_has_no_p_value(self):
        d = est.rate_difference("t", est.estimate_rate("a", 0, 0),
                                est.estimate_rate("b", 3, 30))
        self.assertIsNone(d.p_value)

    def test_to_dict_is_json_native(self):
        e = est.estimate_rate("t", 0, 0)
        blob = json.dumps(e.to_dict(), allow_nan=False)
        self.assertIn('"label": "t"', blob)


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------

class TestThresholds(unittest.TestCase):

    def test_critical_count_matches_brute_force(self):
        for n in (10, 33, 96, 198):
            for p0 in (0.005, 0.02, 0.1):
                for alpha in (0.05, 1e-3, 1e-6):
                    k = th.critical_count(n, p0, alpha)
                    if k <= n:
                        self.assertLessEqual(stats.binom_sf(k, n, p0), alpha)
                        if k > 0:
                            self.assertGreater(stats.binom_sf(k - 1, n, p0),
                                               alpha)

    def test_critical_count_admits_impossibility(self):
        """At n=3 no count is significant at 1e-6; say so rather than lie."""
        self.assertEqual(th.critical_count(3, 0.01, 1e-6), 4)

    def test_kl_threshold_is_tighter_than_hoeffding(self):
        n, forger, eps = 200, 0.25, 1e-9
        kl = th.forger_side_threshold(n, forger, eps, "kl")
        hoef = th.forger_side_threshold(n, forger, eps, "hoeffding")
        # Tighter on the forger side means *higher*: more room to accept
        # honest runs for the same guarantee against a forger.
        self.assertGreater(kl, hoef)

    def test_honest_side_threshold_controls_false_alarms(self):
        n, honest, eps = 200, 0.02, 1e-6
        t = th.honest_side_threshold(n, honest, eps)
        self.assertGreater(t, honest)
        self.assertLessEqual(stats.chernoff_kl_tail(n, honest, t), eps * 1.001)

    def test_thresholds_reject_bad_method(self):
        with self.assertRaises(ValueError):
            th.honest_side_threshold(10, 0.01, 1e-6, method="bogus")
        with self.assertRaises(ValueError):
            th.multiplicity_alpha(0.01, 11, method="bogus")

    def test_security_exponents_off_by_one(self):
        """Accept iff ``k <= floor(T*n)``; both tails must agree with that."""
        n, T, honest = 100, 0.08, 0.02
        s = th.security_exponents(n, T, honest)
        self.assertEqual(s["accept_at_most"], 8)
        # false alarm = P(honest run is rejected) = P(k >= 9)
        self.assertAlmostEqual(s["exact"]["false_alarm"],
                               stats.binom_sf(9, n, honest), places=15)
        # missed forgery = P(forger passes) = P(k <= 8)
        self.assertAlmostEqual(s["exact"]["missed_forgery"],
                               stats.binom_cdf(8, n, 0.25), places=15)

    def test_chernoff_bound_dominates_the_exact_probability(self):
        for n in (50, 200, 500):
            s = th.security_exponents(n, 0.1, 0.02)
            for key in ("false_alarm", "missed_forgery",
                        "missed_blind_forgery"):
                self.assertGreaterEqual(s["chernoff"][key] + 1e-18,
                                        s["exact"][key],
                                        f"n={n} {key}: bound below the truth")

    def test_security_grows_with_more_checks(self):
        prev = -1.0
        for n in (50, 100, 200, 400):
            b = th.security_exponents(n, 0.1, 0.02)["bits"]["missed_forgery"]
            self.assertGreater(b, prev)
            prev = b

    def test_bits_of_zero_is_none_not_infinity(self):
        self.assertIsNone(th.bits(0.0))
        self.assertEqual(th.bits(1.0), 0.0)
        self.assertAlmostEqual(th.bits(0.25), 2.0)

    def test_bonferroni_is_more_conservative_than_sidak(self):
        b = th.multiplicity_alpha(0.01, 11, "bonferroni")
        s = th.multiplicity_alpha(0.01, 11, "sidak")
        self.assertLess(b, s)
        # ...but only just, so the dependence-free guarantee is nearly free
        self.assertLess((s - b) / s, 0.01)

    def test_ladder_ordering_holds_for_the_presets(self):
        for name in ("ideal", "lab", "field"):
            noise = PRESET_NOISE[name]
            spec = noise.predicted_qber()
            ladder = th.threshold_ladder(spec, 198)
            self.assertEqual(th.ladder_problems(ladder), [], name)
            self.assertTrue(ladder["ordered"], name)
            self.assertLess(spec, ladder["accept"], name)
            self.assertLessEqual(ladder["accept"], ladder["transfer"], name)
            self.assertLess(ladder["transfer"], 0.25, name)

    def test_ladder_refuses_a_hopeless_link(self):
        ladder = th.threshold_ladder(0.30, 198)
        problems = th.ladder_problems(ladder)
        self.assertTrue(problems)
        self.assertFalse(ladder["ordered"])
        self.assertTrue(any("forger" in p for p in problems))

    def test_ladder_records_whether_it_derived_the_thresholds(self):
        self.assertTrue(th.threshold_ladder(0.01, 100)["derived"])
        self.assertFalse(th.threshold_ladder(0.01, 100, 0.05, 0.12)["derived"])

    def test_ladder_uses_supplied_thresholds_verbatim(self):
        """The engine must judge against the numbers the protocol used."""
        ladder = th.threshold_ladder(0.01, 100, 0.0411, 0.1234)
        self.assertAlmostEqual(ladder["accept"], 0.0411)
        self.assertAlmostEqual(ladder["transfer"], 0.1234)

    def test_no_checks_is_a_problem_not_a_pass(self):
        problems = th.ladder_problems(th.threshold_ladder(0.01, 0))
        self.assertTrue(any("no checks" in p for p in problems))

    def test_ladder_json_serialisable(self):
        json.dumps(th.threshold_ladder(0.0, 198), allow_nan=False)


# --------------------------------------------------------------------------
# rules: inapplicable is not a pass
# --------------------------------------------------------------------------

class TestRulesInapplicable(unittest.TestCase):

    def test_empty_evidence_flags_nothing_and_passes_nothing(self):
        ev = Evidence(spec_rate=0.01, accept_threshold=0.05,
                      transfer_threshold=0.12)
        report = DetectionEngine().run(ev)
        self.assertEqual(report.flagged, [])
        self.assertEqual(report.n_applicable, 0,
                         "a rule claimed to be applicable with no data")
        for outcome in report.outcomes:
            self.assertFalse(outcome.applicable, outcome.name)
            self.assertIn("why_not", outcome.detail, outcome.name)

    def test_verdict_on_no_data_is_not_clean_by_luck(self):
        """Nothing fired, but nothing could have. Report it as such."""
        report = DetectionEngine().run(Evidence(spec_rate=0.01))
        self.assertEqual(report.verdict, "clean")
        self.assertEqual(report.n_applicable, 0)
        self.assertTrue(th.ladder_problems(report.evidence.ladder))

    def test_missing_declaration_marks_yield_test_inapplicable(self):
        rng = np.random.default_rng(0)
        ev = synthetic_evidence(rng)
        ev.declared_yield = None
        ev.declared_dark_fraction = None
        report = DetectionEngine().run(ev)
        names = {o.name: o for o in report.outcomes}
        self.assertFalse(names["yield_vs_declared"].applicable)
        self.assertFalse(names["dark_excess"].applicable)


# --------------------------------------------------------------------------
# rules: each one fires on its own target
# --------------------------------------------------------------------------

class TestRulesFire(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def _run(self, ev):
        return DetectionEngine().run(ev)

    def test_pooled_rate_vs_spec(self):
        ev = synthetic_evidence(self.rng)
        ev.reports = [make_report(honest_specs(self.rng, 96, 0.20, 0.20),
                                  threshold=0.05)]
        ev.__post_init__()
        self.assertIn("pooled_rate_vs_spec", self._run(ev).flagged)

    def test_pooled_rate_vs_transfer(self):
        ev = synthetic_evidence(self.rng, s_v=0.12)
        ev.reports = [make_report(honest_specs(self.rng, 96, 0.30, 0.30),
                                  level="transfer", threshold=0.12)]
        ev.__post_init__()
        self.assertIn("pooled_rate_vs_transfer", self._run(ev).flagged)

    def test_basis_consistency_catches_a_basis_biased_probe(self):
        """The case a single pooled threshold cannot see.

        Z at spec, X at 12%: pooled lands near 6%, comfortably under a 10%
        acceptance threshold, while the X basis alone is far out of spec.
        """
        ev = synthetic_evidence(self.rng, n_per_verifier=400, s_a=0.10,
                                spec_z=0.010, spec_x=0.014)
        ev.reports = [make_report(honest_specs(self.rng, 400, 0.010, 0.12),
                                  threshold=0.10)]
        ev.__post_init__()
        report = self._run(ev)
        pooled = report.evidence.estimates["pooled_rate"].value
        self.assertLess(pooled, 0.10, "premise of this test broke")
        self.assertIn("basis_consistency", report.flagged)

    def test_basis_consistency_does_not_fire_on_a_healthy_asymmetry(self):
        """Honest Z and X rates genuinely differ; that must not alarm.

        Memory dephasing spares Z eigenstates and an X check passes through
        more gate insertions, so a detector that tested Z against X rather
        than each against its own declared value would false-alarm on every
        healthy link.
        """
        ev = synthetic_evidence(self.rng, n_per_verifier=800,
                                spec_z=0.006, spec_x=0.020)
        ev.reports = [make_report(honest_specs(self.rng, 800, 0.006, 0.020),
                                  threshold=0.05)]
        ev.__post_init__()
        by = est.rates_by_basis(ev.reports)
        self.assertGreater(by["X"].value, 2.0 * by["Z"].value,
                           "premise of this test broke")
        self.assertNotIn("basis_consistency", self._run(ev).flagged)

    def test_position_homogeneity_catches_one_rewritten_bit(self):
        specs = []
        for i in range(800):
            j = i % 8
            rate = 0.35 if j == 3 else 0.010
            specs.append((j, i % 2, bool(self.rng.random() < rate), "ok"))
        ev = synthetic_evidence(self.rng)
        ev.reports = [make_report(specs, threshold=0.20)]
        ev.__post_init__()
        self.assertIn("position_homogeneity", self._run(ev).flagged)

    def test_level_homogeneity_catches_a_rate_that_moved(self):
        ev = synthetic_evidence(self.rng)
        ev.reports = [
            make_report(honest_specs(self.rng, 400, 0.01, 0.01),
                        level="accept", threshold=0.20),
            make_report(honest_specs(self.rng, 400, 0.18, 0.18),
                        level="transfer", threshold=0.20),
        ]
        ev.__post_init__()
        self.assertIn("level_homogeneity", self._run(ev).flagged)

    def test_yield_low_fires(self):
        """Intercept-and-block: remove the evidence instead of corrupting it."""
        ev = synthetic_evidence(self.rng, declared_yield=0.78)
        ev.detector_logs = {"bob": ["click"] * 200 + ["lost"] * 200}
        ev.__post_init__()
        report = self._run(ev)
        self.assertIn("yield_vs_declared", report.flagged)
        self.assertNotIn("pooled_rate_vs_spec", report.flagged)

    def test_yield_high_also_fires(self):
        """Injected clicks are as anomalous as missing ones; test both tails."""
        ev = synthetic_evidence(self.rng, declared_yield=0.78)
        ev.detector_logs = {"bob": ["click"] * 400}
        ev.__post_init__()
        self.assertIn("yield_vs_declared", self._run(ev).flagged)

    def test_dark_excess_fires_one_sided(self):
        ev = synthetic_evidence(self.rng, declared_dark=0.004)
        ev.detector_logs = {"bob": ["dark"] * 40 + ["click"] * 360}
        ev.__post_init__()
        self.assertIn("dark_excess", self._run(ev).flagged)

    def test_dark_deficit_does_not_fire(self):
        """Fewer darks than declared is good news, not an alarm."""
        ev = synthetic_evidence(self.rng, declared_dark=0.05)
        ev.detector_logs = {"bob": ["click"] * 400}
        ev.__post_init__()
        self.assertNotIn("dark_excess", self._run(ev).flagged)

    def test_correction_uniformity_catches_a_biased_transcript(self):
        ev = synthetic_evidence(self.rng)
        ev.corrections = {"bob": [(0, 0)] * 200 + [(1, 1)] * 20}
        ev.__post_init__()
        report = self._run(ev)
        self.assertIn("correction_uniformity", report.flagged)
        self.assertEqual(report.verdict, "compromised",
                         "a broken leakage premise must not be 'suspicious'")

    def test_correction_uniformity_needs_a_sample(self):
        ev = synthetic_evidence(self.rng)
        ev.corrections = {"bob": [(0, 0)] * 4}
        ev.__post_init__()
        names = {o.name: o for o in self._run(ev).outcomes}
        self.assertFalse(names["correction_uniformity"].applicable)

    def test_decoy_versus_signature_catches_selective_targeting(self):
        """Decoys clean, signature slots dirty: impossible under honest noise.

        Which slots are decoys is not revealed until after they cross the
        channel, so no physical process can prefer one set over the other.
        """
        ev = synthetic_evidence(self.rng, n_per_verifier=300)
        ev.reports = [make_report(honest_specs(self.rng, 300, 0.12, 0.12),
                                  threshold=0.20)]
        ev.calibrations = [make_calibration(1, 300)]
        ev.__post_init__()
        report = self._run(ev)
        self.assertIn("decoy_vs_signature", report.flagged)
        self.assertEqual(report.verdict, "compromised")

    def test_recipient_agreement_catches_a_split(self):
        ev = synthetic_evidence(self.rng, verifiers=("bob", "charlie"))
        ev.reports = [
            make_report(honest_specs(self.rng, 300, 0.010, 0.014),
                        verifier="bob", threshold=0.20),
            make_report(honest_specs(self.rng, 300, 0.150, 0.150),
                        verifier="charlie", threshold=0.20),
        ]
        ev.__post_init__()
        report = self._run(ev)
        self.assertIn("recipient_agreement", report.flagged)
        self.assertEqual(report.attributions[0].label,
                         "repudiation_or_targeted_link_attack")

    def test_recipient_agreement_needs_two_recipients(self):
        ev = synthetic_evidence(self.rng, verifiers=("bob",))
        names = {o.name: o for o in self._run(ev).outcomes}
        self.assertFalse(names["recipient_agreement"].applicable)

    def test_sequential_onset_stops_early_on_a_bad_run(self):
        ev = synthetic_evidence(self.rng, n_per_verifier=300)
        ev.reports = [make_report(honest_specs(self.rng, 300, 0.25, 0.25),
                                  threshold=0.30)]
        ev.__post_init__()
        names = {o.name: o for o in self._run(ev).outcomes}
        onset = names["sequential_onset"]
        self.assertTrue(onset.flagged)
        self.assertIsNone(onset.p_value,
                          "a sequential test has no fixed-sample p-value")
        self.assertIn("decision_rule", onset.detail)

    def test_sequential_onset_clamps_a_zero_null(self):
        """``stats.SPRT`` requires ``0 < p0 < p1``; the ideal preset has p0=0."""
        ev = synthetic_evidence(self.rng, spec_z=0.0, spec_x=0.0)
        ev.reports = [make_report(honest_specs(self.rng, 200, 0.0, 0.0),
                                  threshold=0.05)]
        ev.__post_init__()
        names = {o.name: o for o in self._run(ev).outcomes}
        onset = names["sequential_onset"]
        self.assertTrue(onset.applicable)
        self.assertTrue(onset.detail.get("null_rate_clamped"))
        self.assertFalse(onset.flagged)


# --------------------------------------------------------------------------
# false-alarm calibration
# --------------------------------------------------------------------------

class TestFalseAlarmCalibration(unittest.TestCase):
    """The test that makes every "the engine caught it" claim meaningful.

    Every quantity is drawn from exactly the distribution the engine was told
    to expect, so any alarm is a false one.  The engine advertises a
    family-wise alpha of 0.01 across eleven Bonferroni-corrected tests; the
    observed rate must not exceed it by more than sampling noise.

    The sequential monitor is excluded from the headline figure and checked
    separately.  SPRT and CUSUM run at their own fixed error rates rather than
    at the Bonferroni-corrected alpha -- they are not fixed-sample tests and
    have no p-value to correct -- so folding them into the family would be
    comparing a number against an alpha it was never derived from.
    """

    TRIALS = 400
    ALPHA = 0.01

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260928)
        cls.reports = []
        for _ in range(cls.TRIALS):
            ev = synthetic_evidence(rng, alpha_family=cls.ALPHA)
            cls.reports.append(DetectionEngine().run(ev))

    def test_fixed_sample_family_wise_alarm_rate_is_controlled(self):
        excluded = {"sequential_onset"}
        alarms = sum(1 for r in self.reports
                     if set(r.flagged) - excluded)
        rate = alarms / self.TRIALS
        # Three-sigma allowance on TRIALS Bernoulli draws at ALPHA.
        slack = 3.0 * math.sqrt(self.ALPHA * (1 - self.ALPHA) / self.TRIALS)
        self.assertLessEqual(
            rate, self.ALPHA + slack,
            f"{alarms}/{self.TRIALS} honest runs flagged; the engine is "
            f"crying wolf and its attack detections mean nothing")

    def test_no_single_detector_dominates_the_false_alarms(self):
        counts = {}
        for r in self.reports:
            for name in r.flagged:
                if name != "sequential_onset":
                    counts[name] = counts.get(name, 0) + 1
        for name, c in counts.items():
            # Each test individually runs at ALPHA/11; allow generous slack
            # for a 400-trial estimate but catch a systematically broken null.
            self.assertLessEqual(
                c, 6, f"{name} fired {c}/{self.TRIALS} times on honest data; "
                      f"its null hypothesis is probably wrong, not unlucky")

    def test_sequential_monitor_false_alarm_rate(self):
        onset = sum(1 for r in self.reports if "sequential_onset" in r.flagged)
        self.assertLessEqual(
            onset / self.TRIALS, 0.05,
            f"the sequential monitor alarmed on {onset}/{self.TRIALS} honest "
            f"runs; SPRT at alpha=1e-3 plus CUSUM at h=5 should be far rarer")

    def test_honest_runs_are_mostly_clean(self):
        clean = sum(1 for r in self.reports if r.verdict == "clean")
        self.assertGreaterEqual(clean / self.TRIALS, 0.90)

    def test_every_detector_was_exercised(self):
        """A test that is always inapplicable is not protecting anything."""
        applicable = set()
        for r in self.reports:
            applicable.update(o.name for o in r.outcomes if o.applicable)
        expected = {rule.name for rule in default_rules()}
        self.assertEqual(applicable, expected,
                         f"never applicable: {sorted(expected - applicable)}")


# --------------------------------------------------------------------------
# verdict and attribution
# --------------------------------------------------------------------------

class TestVerdict(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_clean_run(self):
        report = DetectionEngine().run(synthetic_evidence(self.rng))
        self.assertEqual(report.verdict, "clean")
        self.assertFalse(report.alarm)
        self.assertEqual(report.attributions, [])

    def test_protocol_rejection_dominates_a_silent_test_suite(self):
        """The bug this ordering exists to prevent.

        The eleven tests run at alpha/11 apiece, which costs sensitivity by
        design, so a run can sit just past the acceptance threshold with every
        test still passing.  Calling that ``clean`` next to a signature the
        protocol refused would teach an operator to ignore the verdict.
        """
        ev = synthetic_evidence(self.rng, s_a=0.05)
        ev.reports = [make_report(
            honest_specs(self.rng, 96, 0.010, 0.014), threshold=0.05,
            accepted=False,
            )]
        # Force the rejection without making the rate large enough to trip a
        # test, which is exactly the regime that produced the bug.
        ev.reports[0].reasons = ["mismatch rate reached the accept threshold"]
        ev.__post_init__()
        ev.rejections[0]["rate_exceeded"] = True
        report = DetectionEngine().run(ev)
        self.assertEqual(report.flagged, [])
        self.assertEqual(report.verdict, "compromised")
        self.assertTrue(report.alarm, "a rejected signature must raise alarm")
        self.assertEqual(report.attributions[0].label,
                         "signature_rejected_on_rate")

    def test_mac_failure_is_compromised_with_no_innocent_explanation(self):
        """Impersonation and replay raise no error rate at all."""
        ev = synthetic_evidence(self.rng)
        ev.reports = [make_report(honest_specs(self.rng, 96, 0.010, 0.014),
                                  accepted=False, mac_ok=False)]
        ev.__post_init__()
        report = DetectionEngine().run(ev)
        self.assertEqual(report.flagged, [],
                         "premise: no statistical test should fire here")
        self.assertTrue(ev.authentication_failed)
        self.assertEqual(report.verdict, "compromised")
        self.assertEqual(report.attributions[0].label, "authentication_failure")

    def test_stale_replay_is_compromised(self):
        ev = synthetic_evidence(self.rng)
        ev.reports = [make_report(honest_specs(self.rng, 96, 0.010, 0.014),
                                  accepted=False, fresh=False)]
        ev.__post_init__()
        report = DetectionEngine().run(ev)
        self.assertTrue(ev.authentication_failed)
        self.assertEqual(report.attributions[0].label, "authentication_failure")

    def test_premise_failure_is_compromised_even_under_threshold(self):
        ev = synthetic_evidence(self.rng)
        ev.corrections = {"bob": [(0, 0)] * 300}
        ev.__post_init__()
        report = DetectionEngine().run(ev)
        self.assertLess(report.evidence.estimates["pooled_rate"].value,
                        report.evidence.ladder["accept"])
        self.assertTrue(set(report.flagged) & PREMISE_RULES)
        self.assertEqual(report.verdict, "compromised")

    def test_significant_but_operationally_small_is_only_suspicious(self):
        """Statistical significance is not the same as operational risk."""
        ev = synthetic_evidence(self.rng, n_per_verifier=4000, s_a=0.20)
        ev.reports = [make_report(honest_specs(self.rng, 4000, 0.030, 0.030),
                                  threshold=0.20)]
        ev.__post_init__()
        report = DetectionEngine().run(ev)
        self.assertIn("pooled_rate_vs_spec", report.flagged)
        self.assertLess(report.evidence.estimates["pooled_rate"].value, 0.20)
        self.assertEqual(report.verdict, "suspicious")

    def test_attribution_is_deterministic_and_ordered(self):
        ev = synthetic_evidence(self.rng, verifiers=("bob", "charlie"))
        ev.reports = [
            make_report(honest_specs(self.rng, 300, 0.010, 0.014),
                        verifier="bob", threshold=0.30),
            make_report(honest_specs(self.rng, 300, 0.200, 0.200),
                        verifier="charlie", threshold=0.30),
        ]
        ev.__post_init__()
        first = attribute(DetectionEngine().run(ev).flagged, ev)
        second = attribute(DetectionEngine().run(ev).flagged, ev)
        self.assertEqual([a.label for a in first], [a.label for a in second])
        self.assertEqual([a.priority for a in first],
                         sorted(a.priority for a in first))

    def test_every_rationale_admits_the_ambiguity(self):
        """No attribution may claim to have identified an attacker.

        Raised error rates and degraded hardware are observationally
        identical from inside the protocol.  The framework detects a departure
        from a declared specification; it does not read intent, and the
        rationale strings are the only place that promise is kept.
        """
        from qds.detect.engine import _TABLE
        hedges = ("equally", "also", "unlikely", "cannot tell", "would also",
                  "alternative", "operator error", "not ambiguous",
                  "innocent", "unlucky", "separately", "comparison")
        for label, _cond, rationale in _TABLE:
            self.assertTrue(
                any(h in rationale for h in hedges),
                f"the rationale for {label!r} states a cause without "
                f"acknowledging the alternative explanation")

    def test_combined_p_value_is_reported_but_not_decisive(self):
        ev = synthetic_evidence(self.rng)
        report = DetectionEngine().run(ev)
        self.assertIsNotNone(report.combined_p_value)
        # Fisher's method assumes independence these tests do not have, so a
        # small combined p must not by itself raise the alarm.
        self.assertEqual(report.alarm, bool(report.flagged))


# --------------------------------------------------------------------------
# JSON contract
# --------------------------------------------------------------------------

class TestJsonContract(unittest.TestCase):
    """The dashboard parses this; a NaN anywhere breaks the page."""

    REQUIRED = ("verdict", "alarm", "combined_p_value", "alpha",
                "alpha_per_test", "n_tests", "n_applicable", "n_flagged",
                "flagged", "attribution", "tests", "estimates", "thresholds",
                "security_bits", "protocol_rejects", "authentication_failed",
                "rejections", "declaration_source", "declared",
                "elapsed_seconds")

    def _blobs(self):
        rng = np.random.default_rng(99)
        yield DetectionEngine().run(Evidence(spec_rate=0.0)).to_dict()
        yield DetectionEngine().run(synthetic_evidence(rng)).to_dict()
        ev = synthetic_evidence(rng)
        ev.reports = [make_report(honest_specs(rng, 200, 0.3, 0.3),
                                  accepted=False, mac_ok=False)]
        ev.__post_init__()
        yield DetectionEngine().run(ev).to_dict()

    def test_strict_json_no_nan_or_infinity(self):
        for blob in self._blobs():
            json.dumps(blob, allow_nan=False)

    def test_required_keys_present(self):
        for blob in self._blobs():
            for key in self.REQUIRED:
                self.assertIn(key, blob)

    def test_test_records_have_a_stable_shape(self):
        for blob in self._blobs():
            self.assertEqual(len(blob["tests"]), 11)
            for t in blob["tests"]:
                for key in ("name", "title", "statistic", "p_value",
                            "threshold", "flagged", "n", "applicable",
                            "detail", "interpretation"):
                    self.assertIn(key, t)
                self.assertTrue(t["interpretation"],
                                f"{t['name']} has no plain-words reading")

    def test_vacuous_declaration_is_disclosed(self):
        """A test that cannot fail must not look like a test that passed."""
        cfg = SessionConfig(message_bits=8, L=8, distribution=DistributionConfig(
            noise=PRESET_NOISE["lab"], check_pairs=16, decoy_slots=16))
        session = QDSSession(cfg, seed=5)
        session.run(message=b"\xb0")
        blob = analyse_session(session).to_dict()
        self.assertEqual(blob["declaration_source"], "simulated_model")
        self.assertIn("cannot", blob["declared"]["note"])

    def test_independent_declaration_is_disclosed(self):
        cfg = SessionConfig(message_bits=8, L=8, distribution=DistributionConfig(
            noise=PRESET_NOISE["lab"], check_pairs=16, decoy_slots=16))
        session = QDSSession(cfg, seed=5)
        session.run(message=b"\xb0")
        blob = analyse_session(
            session, declared_noise=PRESET_NOISE["lab"]).to_dict()
        self.assertEqual(blob["declaration_source"], "spec_sheet")


# --------------------------------------------------------------------------
# live sessions: the wiring
# --------------------------------------------------------------------------

def run_session(*, noise=None, intervention=None, seed=11, message_bits=8,
                L=16, **dk):
    cfg = SessionConfig(
        message_bits=message_bits, L=L,
        distribution=DistributionConfig(
            noise=noise if noise is not None else PRESET_NOISE["lab"],
            intervention=(intervention if intervention is not None
                          else make_intervention("none")),
            **dk))
    session = QDSSession(cfg, seed=seed)
    session.run(message=b"\xb0")
    return session


class TestLiveSession(unittest.TestCase):
    """End to end against the real protocol, with the real channel."""

    LAB = PRESET_NOISE["lab"]

    @classmethod
    def setUpClass(cls):
        cls.honest = run_session()
        cls.attacked = run_session(
            intervention=make_intervention("intercept_resend", strength=1.0))

    def test_honest_lab_session_is_clean(self):
        report = analyse_session(self.honest, declared_noise=self.LAB)
        self.assertEqual(report.verdict, "clean", report.summary())
        self.assertEqual(report.flagged, [])

    def test_every_detector_is_applicable_on_a_real_session(self):
        report = analyse_session(self.honest, declared_noise=self.LAB)
        self.assertEqual(report.n_applicable, 11,
                         [o.name for o in report.outcomes
                          if not o.applicable])

    def test_intercept_resend_is_caught(self):
        report = analyse_session(self.attacked, declared_noise=self.LAB)
        self.assertEqual(report.verdict, "compromised", report.summary())
        self.assertIn("pooled_rate_vs_spec", report.flagged)
        rate = report.evidence.estimates["pooled_rate"].value
        # Full intercept-resend on BB84 states costs the adversary about a
        # quarter of the checks.
        self.assertGreater(rate, 0.15)
        labels = [a.label for a in report.attributions]
        self.assertIn("forgery_or_measure_and_resend", labels)

    def test_ideal_channel_accepts_deterministically(self):
        """Zero noise, zero loss: acceptance is not probabilistic."""
        session = run_session(noise=PRESET_NOISE["ideal"], L=8,
                              check_pairs=16, decoy_slots=16)
        report = analyse_session(session, declared_noise=PRESET_NOISE["ideal"])
        self.assertEqual(report.evidence.estimates["pooled_rate"].k, 0)
        self.assertEqual(report.verdict, "clean")
        self.assertFalse(report.evidence.protocol_rejects)

    def test_undeclared_loss_is_invisible_without_a_declaration(self):
        """Documents the limitation rather than hiding it.

        With no spec sheet the declared yield is read off the same channel the
        simulation ran, so the loss test compares a quantity with itself.  The
        report says so; this test pins that behaviour down so nobody later
        mistakes it for a passing check.
        """
        lossy = NoiseModel(0.008, 0.004, 0.001, 0.002, 0.01,
                           LossModel(0.55, 0.85, 1e-5))
        session = run_session(noise=lossy, L=8, check_pairs=16, decoy_slots=16)
        blind = analyse_session(session)
        self.assertNotIn("yield_vs_declared", blind.flagged)
        self.assertEqual(blind.to_dict()["declaration_source"],
                         "simulated_model")
        self.assertIn("cannot fail", blind.to_dict()["declared"]["note"])
        informed = analyse_session(session, declared_noise=self.LAB)
        self.assertIn("yield_vs_declared", informed.flagged)
        self.assertEqual(informed.attributions[0].label,
                         "channel_blocking_or_blinding")

    def test_report_summary_is_one_line(self):
        summary = analyse_session(self.honest).summary()
        self.assertNotIn("\n", summary)
        self.assertIn("rate=", summary)

    def test_analysis_is_cheap_relative_to_simulation(self):
        report = analyse_session(self.honest)
        self.assertLess(report.elapsed_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
