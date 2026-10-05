"""Tests for :mod:`qds.metrics` -- the performance / analytics layer.

Stdlib ``unittest`` only, numpy allowed (the engine already depends on it); no
pytest, no scipy.  The assertions pin the mathematical identities the module
claims rather than golden numbers: relative-entropy non-negativity, the forgery
bound's monotonicity in key length and in the threshold gap, the reproduction of
the whitepaper's ``N = 1080`` lab-grade figure, detection-power monotonicity and
its limit, and the Monte-Carlo false-alarm rate staying within the Bonferroni
budget.  Everything runs in a couple of seconds.
"""

from __future__ import annotations

import json
import math
import unittest

from qds import metrics
from qds.detect import stats
from qds.detect.thresholds import critical_count, multiplicity_alpha

# The whitepaper's lab-grade, balanced-rule transfer threshold (§5.5/§5.7).
LAB_FLOOR = 0.012397223954321968
LAB_S_A = 0.039622126617503375
LAB_S_V = 0.16871541421373465
QUARTER = 0.25


def _all_finite_json(obj) -> bool:
    """True if every numeric leaf is a finite JSON scalar (None allowed)."""
    if isinstance(obj, dict):
        return all(_all_finite_json(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite_json(v) for v in obj)
    if isinstance(obj, bool) or obj is None:
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    return isinstance(obj, str)


class TestForgeryDivergence(unittest.TestCase):

    def test_relative_entropy_nonnegative_and_zero_at_equality(self):
        self.assertEqual(stats.binary_kl(0.1, 0.1), 0.0)
        for q in (0.0, 0.05, 0.1, 0.1687, 0.24):
            self.assertGreater(metrics.forgery_divergence(q, QUARTER), 0.0)

    def test_divergence_grows_as_threshold_leaves_the_bound(self):
        # D(s_v || 1/4) must increase as s_v moves further below 1/4.
        d_near = metrics.forgery_divergence(0.20, QUARTER)
        d_far = metrics.forgery_divergence(0.05, QUARTER)
        self.assertGreater(d_far, d_near)

    def test_threshold_at_or_above_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.forgery_divergence(0.25, QUARTER)
        with self.assertRaises(ValueError):
            metrics.forgery_divergence(0.30, QUARTER)


class TestForgeryBound(unittest.TestCase):

    def test_bound_equals_closed_form(self):
        d = metrics.forgery_divergence(LAB_S_V, QUARTER)
        for n in (100, 384, 768, 1080):
            self.assertAlmostEqual(
                metrics.forgery_survival_bound(n, LAB_S_V),
                math.exp(-n * d), places=12)

    def test_bound_strictly_decreasing_in_n(self):
        prev = 1.0
        for n in (96, 192, 384, 768, 1080, 1920):
            b = metrics.forgery_survival_bound(n, LAB_S_V)
            self.assertLess(b, prev)
            prev = b

    def test_bound_decreasing_as_gap_to_quarter_widens(self):
        # A threshold further below 1/4 gives a smaller survival bound at fixed n.
        n = 768
        self.assertLess(metrics.forgery_survival_bound(n, 0.05),
                        metrics.forgery_survival_bound(n, 0.20))

    def test_bound_is_unity_when_threshold_reaches_bound(self):
        self.assertEqual(metrics.forgery_survival_bound(1000, 0.25), 1.0)
        self.assertEqual(metrics.forgery_survival_bound(1000, 0.30), 1.0)


class TestRequiredChecks(unittest.TestCase):

    def test_reproduces_whitepaper_lab_grade_figure(self):
        # §5.7: lab-grade hardware needs N = 1080 for a 1e-9 forgery target.
        n = metrics.required_checks_for_forgery(1e-9, LAB_S_V)
        self.assertIsNotNone(n)
        self.assertLessEqual(abs(n - 1080), 2)

    def test_required_checks_derived_from_live_policy_also_gives_1080(self):
        # Derive s_v from the engine itself rather than the literal above.
        from qds.channel import LAB_GRADE
        from qds.protocol.verification import VerificationPolicy
        eta = float(LAB_GRADE.predicted_qber())
        s_v = VerificationPolicy.balanced(eta).s_v
        self.assertEqual(metrics.required_checks_for_forgery(1e-9, s_v), 1080)

    def test_reproduces_ideal_and_field_grade_figures(self):
        # §5.7 table: ideal -> 669, field-grade -> 1885.
        from qds.channel import FIELD_GRADE, IDEAL
        from qds.protocol.verification import VerificationPolicy
        for preset, expected in ((IDEAL, 669), (FIELD_GRADE, 1885)):
            s_v = VerificationPolicy.balanced(
                float(preset.predicted_qber())).s_v
            self.assertEqual(
                metrics.required_checks_for_forgery(1e-9, s_v), expected)

    def test_required_checks_increasing_as_target_tightens(self):
        prev = 0
        for eps in (1e-3, 1e-6, 1e-9, 1e-12):
            n = metrics.required_checks_for_forgery(eps, LAB_S_V)
            self.assertGreater(n, prev)
            prev = n

    def test_inverse_of_the_bound(self):
        # The N it returns must actually push the bound under epsilon.
        eps = 1e-9
        n = metrics.required_checks_for_forgery(eps, LAB_S_V)
        self.assertLessEqual(metrics.forgery_survival_bound(n, LAB_S_V), eps)
        self.assertGreater(
            metrics.forgery_survival_bound(n - 1, LAB_S_V), eps)

    def test_no_finite_n_at_the_bound(self):
        self.assertIsNone(metrics.required_checks_for_forgery(1e-9, 0.25))

    def test_rejects_degenerate_epsilon(self):
        for bad in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(ValueError):
                metrics.required_checks_for_forgery(bad, LAB_S_V)


class TestDetectionPower(unittest.TestCase):

    def setUp(self):
        self.alpha = multiplicity_alpha(0.01, 11)

    def test_size_at_floor_within_alpha(self):
        # True rate == floor: the realised size is at most alpha_per_test.
        for n in (384, 768, 1080, 1920):
            p = metrics.detection_power(n, LAB_FLOOR, LAB_FLOOR, self.alpha)
            self.assertLessEqual(p, self.alpha + 1e-12)

    def test_power_nondecreasing_in_n(self):
        for r in (0.05, 0.08, LAB_S_V):
            seq = [metrics.detection_power(n, r, LAB_FLOOR, self.alpha)
                   for n in range(100, 1600, 100)]
            for a, b in zip(seq, seq[1:]):
                self.assertGreaterEqual(b, a - 1e-12)

    def test_power_tends_to_one(self):
        for r in (0.05, 0.08, LAB_S_V):
            self.assertGreater(
                metrics.detection_power(4000, r, LAB_FLOOR, self.alpha), 0.999)

    def test_power_nondecreasing_in_rate(self):
        rates = [LAB_FLOOR + 0.02 * i for i in range(8)]
        pts = metrics.power_by_rate(768, LAB_FLOOR, self.alpha, rates)
        powers = [p["power"] for p in pts]
        for a, b in zip(powers, powers[1:]):
            self.assertGreaterEqual(b, a - 1e-12)

    def test_power_matches_exact_tail(self):
        n, r = 768, 0.08
        crit = critical_count(n, LAB_FLOOR, self.alpha)
        self.assertAlmostEqual(
            metrics.detection_power(n, r, LAB_FLOOR, self.alpha),
            stats.binom_sf(crit, n, r), places=12)

    def test_tiny_sample_cannot_fire(self):
        # When no count is significant, critical_count is n+1 and power is 0.
        self.assertEqual(critical_count(1, LAB_FLOOR, self.alpha), 2)
        self.assertEqual(
            metrics.detection_power(1, 0.2, LAB_FLOOR, self.alpha), 0.0)

    def test_curve_carries_critical_count(self):
        pts = metrics.power_curve(0.08, LAB_FLOOR, self.alpha, [96, 768])
        self.assertEqual([p["n"] for p in pts], [96, 768])
        for p in pts:
            self.assertIn("critical_count", p)
            self.assertIsInstance(p["critical_count"], int)


class TestCalibration(unittest.TestCase):

    def test_family_wise_bound_is_the_budget(self):
        self.assertEqual(metrics.family_wise_bound(0.01), 0.01)

    def test_monte_carlo_within_budget(self):
        out = metrics.monte_carlo_false_alarm(
            n_checks=1080, floor_rate=LAB_FLOOR, trials=8000, seed=7)
        # Per-test size within alpha_per_test, family-wise within alpha_family,
        # each with three standard errors of Monte-Carlo slack.
        self.assertLessEqual(
            out["per_test_rate"],
            out["alpha_per_test"] + 3 * out["per_test_se"])
        self.assertLessEqual(
            out["family_wise_rate"],
            out["alpha_family"] + 3 * out["family_wise_se"])
        self.assertTrue(out["within_budget"])

    def test_monte_carlo_is_deterministic_under_seed(self):
        a = metrics.monte_carlo_false_alarm(
            n_checks=768, floor_rate=LAB_FLOOR, trials=4000, seed=123)
        b = metrics.monte_carlo_false_alarm(
            n_checks=768, floor_rate=LAB_FLOOR, trials=4000, seed=123)
        self.assertEqual(a["per_test_rate"], b["per_test_rate"])
        self.assertEqual(a["family_wise_rate"], b["family_wise_rate"])

    def test_monte_carlo_output_is_json_native(self):
        out = metrics.monte_carlo_false_alarm(
            n_checks=512, floor_rate=LAB_FLOOR, trials=2000, seed=1)
        self.assertTrue(_all_finite_json(out))
        json.dumps(out)  # must not raise


class TestEvaluate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # One evaluation, small MC, reused across assertions to stay fast.
        cls.out = metrics.evaluate(mc_trials=4000)

    def test_json_serialisable_and_all_finite(self):
        json.dumps(self.out)
        self.assertTrue(_all_finite_json(self.out))

    def test_top_level_shape(self):
        self.assertEqual(
            set(self.out.keys()),
            {"constants", "forgery", "power", "calibration", "grid"})

    def test_defaults_reproduce_1080(self):
        self.assertEqual(self.out["forgery"]["required_checks"], 1080)

    def test_constants_echo_lab_grade_policy(self):
        c = self.out["constants"]
        self.assertAlmostEqual(c["spec_rate"], LAB_FLOOR, places=9)
        self.assertAlmostEqual(c["transfer_threshold"], LAB_S_V, places=9)
        self.assertAlmostEqual(c["single_copy_bound"], 0.25, places=12)
        self.assertEqual(c["n_tests"], 11)

    def test_forgery_curve_is_monotone_decreasing(self):
        bounds = [p["bound"] for p in self.out["forgery"]["curve"]]
        for a, b in zip(bounds, bounds[1:]):
            self.assertLess(b, a)

    def test_power_curve_climbs(self):
        powers = [p["power"] for p in self.out["power"]["curve"]]
        self.assertLess(powers[0], powers[-1])
        for a, b in zip(powers, powers[1:]):
            self.assertGreaterEqual(b, a - 1e-12)

    def test_calibration_within_budget(self):
        self.assertTrue(self.out["calibration"]["within_budget"])

    def test_summary_is_evaluate(self):
        self.assertIs(metrics.summary, metrics.summary)
        other = metrics.summary(mc_trials=1000, mc_seed=99)
        self.assertEqual(other["forgery"]["required_checks"], 1080)

    def test_custom_spec_rate_replaces_thresholds(self):
        # A different floor must re-place the thresholds (balanced rule).
        out = metrics.evaluate(spec_rate=0.0, mc_trials=1000)
        self.assertEqual(out["forgery"]["required_checks"], 669)  # ideal row


if __name__ == "__main__":
    unittest.main()
