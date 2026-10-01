"""Tests for the attack simulation suite.

These tests are the other half of the security argument: the detection engine's
own tests (``test_detect``) prove it does not cry wolf on honest noise and does
not read the simulator's ground truth; these prove that when a *real* attack is
mounted, the engine actually catches it and attributes it correctly.

The design guards against circularity in two independent ways, both of which
this file exercises:

*Hard-coded literals.*  Every assertion below compares the engine's finding to
a verdict, attribution, or numeric band typed directly into this file.  The
attack code cannot reach these literals, so a bug that shifted an attack and
its own :class:`~qds.attacks.harness.Expectation` together would still have to
move a number a human wrote here -- which it cannot.

*The expectation cross-check.*  :class:`TestExpectationsAgree` additionally
asserts that every attack's engine finding matches the attack author's
:class:`Expectation`, which is produced by a different code path.  That is the
coarser, broader guard sitting behind the specific literals.

The physics-isolation forgery probes are asserted against the
information-theoretic floor directly (1/2 no-copy, 1/4 optimal single-copy),
not through the engine -- the engine's classical gates would fire first and
mask the number.  The Breidbart probe is checked to sit *above* the single-copy
floor, because it maximises information rather than minimising disturbance and
must therefore forge worse than the pretty-good measurement.

A full session costs about 1.6 s, so the whole catalogue is mounted exactly
*once* in :func:`setUpModule` and every test reads from the shared result set.
Only determinism needs a second run, and it re-mounts a small representative
subset rather than the whole catalogue.
"""

from __future__ import annotations

import unittest

import qds.attacks as attacks

#: The canonical run of every attack, mounted once and shared by all tests.
RESULTS = {}


def setUpModule():
    """Mount the whole catalogue a single time, keyed by attack name."""
    for r in attacks.run_all(seed=0):
        RESULTS[r.name] = r


def _r(name):
    return RESULTS[name]


# ----------------------------------------------------------------------------
# forgery: the classical MAC wall
# ----------------------------------------------------------------------------

class TestForgeryClassical(unittest.TestCase):
    """A forger without the private key dies at the per-recipient MAC gate."""

    def test_every_strategy_fails_authentication(self):
        for strategy in ("blind", "single_copy", "breidbart", "partial"):
            with self.subTest(strategy=strategy):
                r = _r(f"forgery_{strategy}")
                self.assertEqual(r.verdict, "compromised")
                self.assertEqual(r.primary, "authentication_failure")
                self.assertTrue(r.authentication_failed)

    def test_mac_rejects_before_any_physics_flag(self):
        # the declaration dies classically, so no statistical detector should
        # even be consulted -- the flag list is empty.
        self.assertEqual(_r("forgery_single_copy").flagged, [])


# ----------------------------------------------------------------------------
# forgery: the physical 1/4 wall
# ----------------------------------------------------------------------------

class TestForgeryPhysics(unittest.TestCase):
    """With the MAC stripped away, quantum unforgeability bounds the forger."""

    def test_single_copy_lands_on_the_quarter_floor(self):
        r = _r("forgery_physics_single_copy")
        # the pretty-good measurement is optimal: the per-check error sits at
        # 1/4, the minimum a one-copy forger can achieve.
        self.assertAlmostEqual(r.detail["mismatch_rate"], 0.25, delta=0.05)
        self.assertEqual(r.detail["analytic_bound"], 0.25)

    def test_blind_guess_sits_at_one_half(self):
        r = _r("forgery_physics_blind")
        rate = r.detail["mismatch_rate"]
        # no copy at all: a coin flip per entry, error 1/2.
        self.assertGreaterEqual(rate, 0.45)
        self.assertLessEqual(rate, 0.60)
        self.assertEqual(r.detail["analytic_bound"], 0.50)

    def test_breidbart_forges_worse_than_the_optimal_measurement(self):
        single = _r("forgery_physics_single_copy")
        breid = _r("forgery_physics_breidbart")
        # Breidbart maximises *information*, not forging success.  Its per-check
        # error must therefore sit strictly above the single-copy 1/4 floor --
        # the concrete statement that learning a key and forging differ.
        self.assertGreater(breid.detail["mismatch_rate"],
                           single.detail["mismatch_rate"] + 0.02)
        self.assertGreaterEqual(breid.detail["mismatch_rate"], 0.25)

    def test_physics_probes_carry_no_engine_verdict(self):
        # the probe bypasses the classical layer on purpose; it is asserted
        # against the floor, not routed through the engine.
        r = _r("forgery_physics_single_copy")
        self.assertIsNone(r.verdict)
        self.assertIsNone(r.report)
        self.assertEqual(r.discrepancies(), [])   # 'n/a' probe is vacuously ok


# ----------------------------------------------------------------------------
# impersonation: deterministic classical gates
# ----------------------------------------------------------------------------

class TestImpersonation(unittest.TestCase):
    """A declaration not from the signer it names is rejected before measurement."""

    def test_every_variant_fails_authentication(self):
        for name in ("impersonation_unauthenticated",
                     "impersonation_tampered_payload",
                     "impersonation_unenrolled_signer",
                     "impersonation_wrong_key_id"):
            with self.subTest(attack=name):
                r = _r(name)
                self.assertEqual(r.verdict, "compromised")
                self.assertEqual(r.primary, "authentication_failure")
                self.assertTrue(r.authentication_failed)


# ----------------------------------------------------------------------------
# replay: freshness welded to the MAC
# ----------------------------------------------------------------------------

class TestReplay(unittest.TestCase):
    """Re-presenting a signature, or editing its freshness fields, is caught."""

    def test_every_variant_fails_authentication(self):
        for name in ("replay_exact", "replay_fresh_nonce",
                     "replay_stale_timestamp", "replay_future_timestamp"):
            with self.subTest(attack=name):
                r = _r(name)
                self.assertEqual(r.verdict, "compromised")
                self.assertEqual(r.primary, "authentication_failure")
                self.assertTrue(r.authentication_failed)

    def test_exact_replay_accepted_once_then_rejected(self):
        r = _r("replay_exact")
        # the ledger burns the counter/nonce/slots on first use, so the first
        # presentation is accepted and the identical second one is not.
        self.assertTrue(r.detail["first_accepted"])
        self.assertFalse(r.detail["second_accepted"])

    def test_refreshing_the_nonce_breaks_the_mac(self):
        # editing the nonce to beat the seen-before check invalidates the tag
        # that was computed over the original nonce.
        self.assertFalse(_r("replay_fresh_nonce").detail["mac_ok"])


# ----------------------------------------------------------------------------
# channel: disturbance above the calibrated floor
# ----------------------------------------------------------------------------

class TestChannel(unittest.TestCase):
    """Eve acting on the flying qubits shows up as error, lost yield, or bias."""

    def test_intercept_resend_rejected_on_rate(self):
        r = _r("channel_intercept_resend")
        self.assertEqual(r.verdict, "compromised")
        self.assertEqual(r.primary, "signature_rejected_on_rate")
        self.assertIn("pooled_rate_vs_spec", r.flagged)

    def test_entanglement_breaking_flags_the_rate(self):
        r = _r("channel_entanglement_breaking")
        self.assertEqual(r.verdict, "compromised")
        self.assertIn("pooled_rate_vs_spec", r.flagged)

    def test_collective_depolarizing_clears_the_floor_when_pushed(self):
        r = _r("channel_collective_depolarizing")
        self.assertEqual(r.verdict, "compromised")
        self.assertIn("pooled_rate_vs_spec", r.flagged)

    def test_coherent_probe_pays_for_information_in_disturbance(self):
        r = _r("channel_coherent_probe")
        self.assertEqual(r.verdict, "compromised")
        self.assertIn("pooled_rate_vs_spec", r.flagged)

    def test_basis_biased_hides_in_the_pool_shows_per_basis(self):
        r = _r("channel_basis_biased")
        # a mild one-basis probe keeps the *pooled* rate acceptable, so the
        # signature is not rejected outright -- but the separate per-basis test
        # sees the loaded basis.  This is the detector the pooled view is blind
        # to, and the verdict is the softer 'suspicious'.
        self.assertEqual(r.verdict, "suspicious")
        self.assertEqual(r.primary, "basis_biased_probe")
        self.assertIn("basis_consistency", r.flagged)
        self.assertNotIn("pooled_rate_vs_spec", r.flagged)

    def test_loss_blinding_caught_on_yield_not_rate(self):
        r = _r("channel_loss_blinding")
        # channel blocking: the errors that get through stay in spec, but the
        # detector yield collapses below the declared loss budget.
        self.assertEqual(r.verdict, "compromised")
        self.assertEqual(r.primary, "channel_blocking_or_blinding")
        self.assertIn("yield_vs_declared", r.flagged)
        self.assertNotIn("pooled_rate_vs_spec", r.flagged)

    def test_dark_injection_breaks_the_dark_count_premise(self):
        r = _r("channel_dark_injection")
        self.assertEqual(r.verdict, "compromised")
        self.assertIn("dark_excess", r.flagged)


# ----------------------------------------------------------------------------
# repudiation: the key-symmetry premise, and the countermeasure
# ----------------------------------------------------------------------------

class TestRepudiation(unittest.TestCase):
    """A split in recipients' error rates, and the symmetrisation that closes it."""

    def test_targeted_link_produces_a_detectable_split(self):
        r = _r("repudiation_targeted_link")
        rates = r.detail["rates_by_verifier"]
        bob, charlie = rates["bob"]["rate"], rates["charlie"]["rate"]
        # the attacked recipient's rate pulls clearly above the clean one's ...
        self.assertGreater(charlie, bob + 0.05)
        # ... yet stays under the transfer threshold, so the signature is NOT
        # simply rejected on rate -- which is what lets the agreement test, not
        # the rate test, be the thing that speaks.
        self.assertLess(charlie, 0.17)
        self.assertEqual(r.verdict, "compromised")
        self.assertEqual(r.primary, "repudiation_or_targeted_link_attack")
        self.assertIn("recipient_agreement", r.flagged)

    def test_symmetrisation_collapses_the_split(self):
        r = _r("repudiation_symmetrised")
        rates = r.detail["rates_by_verifier"]
        bob, charlie = rates["bob"]["rate"], rates["charlie"]["rate"]
        # private permutation redistributes the dirt: the two rates converge,
        # so the agreement detector no longer sees a split and repudiation is
        # impossible.
        self.assertLess(abs(bob - charlie), 0.03)
        self.assertNotIn("recipient_agreement", r.flagged)
        # the dirt itself does not vanish -- both carry an equal share now, so
        # the signature is still caught.  Symmetrisation defeats the
        # repudiation, not the eavesdropper.
        self.assertEqual(r.verdict, "compromised")

    def test_symmetrisation_is_the_only_difference_between_the_two(self):
        # the two runs are the identical attack; only the countermeasure flag
        # differs, and only the countermeasure's effect (the split) differs.
        off, on = _r("repudiation_targeted_link"), _r("repudiation_symmetrised")
        self.assertFalse(off.detail["symmetrised"])
        self.assertTrue(on.detail["symmetrised"])
        self.assertIn("recipient_agreement", off.flagged)
        self.assertNotIn("recipient_agreement", on.flagged)


# ----------------------------------------------------------------------------
# the expectation cross-check, determinism, and catalogue wiring
# ----------------------------------------------------------------------------

class TestExpectationsAgree(unittest.TestCase):
    """Every attack's engine finding matches the attack author's Expectation.

    This is the second, broader guard behind the hard-coded literals above: the
    :class:`Expectation` is written in each attack's own module, the finding
    comes from the engine, and the two are produced by different code paths.
    """

    def test_all_attacks_match_their_expectation(self):
        for name, r in RESULTS.items():
            with self.subTest(attack=name):
                self.assertEqual(
                    r.discrepancies(), [],
                    f"{name} diverged from its expectation: "
                    f"{r.discrepancies()}")


class TestDeterminism(unittest.TestCase):
    """A seeded suite is reproducible -- the whole point of seed-pinning it.

    Re-mounts one representative attack per family (a cheap subset) and checks
    it reproduces the shared run bit for bit; a full second ``run_all`` would
    double an already slow suite for no extra coverage of the property.
    """

    #: ``(family, variant, stored-result-name)`` -- one per family.
    SUBSET = (
        ("forgery", "single_copy", "forgery_single_copy"),
        ("impersonation", "unauthenticated", "impersonation_unauthenticated"),
        ("replay", "exact_replay", "replay_exact"),
        ("channel", "intercept_resend", "channel_intercept_resend"),
        ("repudiation", "targeted_link_attack", "repudiation_targeted_link"),
    )

    def test_representative_attacks_reproduce(self):
        for family, variant, name in self.SUBSET:
            with self.subTest(attack=name):
                again = attacks.mount(family, variant, seed=0)
                old = RESULTS[name]
                self.assertEqual(again.name, name)
                self.assertEqual((again.verdict, again.primary, again.flagged),
                                 (old.verdict, old.primary, old.flagged))


class TestCatalogue(unittest.TestCase):
    """The CLI-facing registry is complete and dispatches correctly."""

    def test_every_family_is_present(self):
        self.assertEqual(set(attacks.FAMILIES), set(attacks.CATALOGUE))
        for family in attacks.FAMILIES:
            self.assertTrue(attacks.CATALOGUE[family],
                            f"{family} has no variants")

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            attacks.mount("no_such_family", "x")
        with self.assertRaises(ValueError):
            attacks.mount("channel", "no_such_variant")

    def test_run_all_covers_the_whole_catalogue(self):
        n_expected = sum(len(v) for v in attacks.CATALOGUE.values())
        self.assertEqual(len(RESULTS), n_expected)


if __name__ == "__main__":
    unittest.main()
