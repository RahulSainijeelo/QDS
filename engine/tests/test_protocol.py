"""Tests for the quantum protocol package (``qds.protocol``).

This package had no dedicated unit tests; the detection engine
(``test_detect``) and the attack suite (``test_attacks``) exercised it
indirectly, but nothing pinned the protocol primitives themselves.  These
tests close that gap: they check the key material, the Wegman-Carter
authentication, the teleported distribution, signing, verification, and the
end-to-end session assembly against values a human wrote down.

Non-circularity is the whole discipline here, enforced two ways.

*Hard-coded literals.*  Every assertion compares a protocol output to a number
typed directly into this file -- ``0.25`` for the Bell-outcome probabilities
and the single-copy forger floor, ``0.5`` for the blind-forger floor, ``0.0``
for an ideal-channel mismatch and for the teleportation leakage, and exact
structural counts (``2 * message_bits * L`` slots, ``L`` entries per block).
The code under test cannot reach these literals, so a bug that moved an output
would still have to move a constant a human owns -- which it cannot.

*Independent closed forms.*  Where the quantity is not a bare constant, it is
checked against a formula computed by a *different* code path.  The sharpest of
these is the physics-faithfulness test: the empirical mismatch rate measured by
a full density-matrix session (``qds.protocol.session``) is compared against the
analytic noise floor ``NoiseModel.predicted_qber()`` (``qds.channel``), two
modules that share no arithmetic.  Likewise ``symmetrise_keys`` is checked by a
pooled-fidelity sum this file computes from the true BB84 targets, never from a
value the symmetriser returned.

A full session is the only expensive object, so the two representative sessions
(one ideal, one noisy) are mounted exactly once in :func:`setUpModule` the way
``test_attacks`` mounts its catalogue; every other fixture is microseconds of
small-register linear algebra built fresh in the test that needs it.  Geometry
is kept small (``message_bits`` 8, ``L`` 12-24) and every RNG is seeded, so the
suite is deterministic and finishes far under the harness cap.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from qds.channel import IDEAL, NoiseModel
from qds.linalg import bb84_state, fidelity_pure
from qds.protocol.auth import (FreshnessLedger, KeyReservoir, MacKey, mac_tag,
                               mac_verify)
from qds.protocol.distribution import (DistributionConfig,
                                       bell_outcome_distribution,
                                       distribute_public_key, teleport_state,
                                       teleportation_leakage)
from qds.protocol.keys import (FORGER_PASS_PROB_BLIND,
                               FORGER_PASS_PROB_ONE_COPY, KeyEntry, PrivateKey,
                               PublicKeyStore, SignerRegistry, Slot, SlotRole,
                               generate_private_key)
from qds.protocol.session import (QDSSession, SessionConfig,
                                   calibrate_from_decoys, symmetrise_keys)
from qds.protocol.signing import (SigningError, bits_to_message,
                                   message_to_bits, sign_message)
from qds.protocol.verification import (BLIND_FORGER_ERROR_RATE,
                                       FORGER_ERROR_RATE, VerificationPolicy,
                                       verify_declaration)

#: One byte -> exactly eight message bits, so ``message_bits=8`` keys sign it.
MESSAGE = b"\xb0"

#: A symmetric honest channel used for the faithfulness cross-check: pure
#: hardware noise, no loss (so every slot is counted) and no adversary.
NOISY = NoiseModel(channel_depolarizing=0.04, gate_error=0.01,
                   measurement_error=0.01)

#: Shared sessions, mounted once in setUpModule (the only costly fixtures).
IDEAL_SESSION: QDSSession = None       # type: ignore[assignment]
NOISY_SESSION: QDSSession = None       # type: ignore[assignment]


def setUpModule():
    """Run the two representative sessions a single time each."""
    global IDEAL_SESSION, NOISY_SESSION
    IDEAL_SESSION = QDSSession(
        config=SessionConfig(
            message_bits=8, L=16, verifiers=("bob", "charlie"),
            distribution=DistributionConfig(noise=IDEAL, check_pairs=8,
                                            decoy_slots=8)),
        seed=0)
    IDEAL_SESSION.run(MESSAGE)

    NOISY_SESSION = QDSSession(
        config=SessionConfig(
            message_bits=8, L=24, verifiers=("bob", "charlie"),
            distribution=DistributionConfig(noise=NOISY, check_pairs=8,
                                            decoy_slots=8)),
        seed=7)
    NOISY_SESSION.run(b"\x6a")


def _fresh_key(seed: int, message_bits: int = 8, L: int = 16) -> tuple:
    """A seeded private key plus its ideal-channel public-key store for Bob."""
    rng = np.random.default_rng(seed)
    pk = generate_private_key("alice", message_bits, L, rng)
    cfg = DistributionConfig(noise=IDEAL, check_pairs=8, decoy_slots=8)
    res = distribute_public_key(pk, "bob", cfg, rng)
    return pk, res, rng


# ----------------------------------------------------------------------------
# keys.py -- Lamport-style one-time key material
# ----------------------------------------------------------------------------

class TestPrivateKeyStructure(unittest.TestCase):
    """A private key is 2*k*L BB84-named entries, two blocks per message bit."""

    def test_block_count_and_shape(self):
        rng = np.random.default_rng(1)
        pk = generate_private_key("alice", message_bits=6, L=11, rng=rng)
        # one block for each (position j, bit value b): 2 * message_bits blocks
        self.assertEqual(len(pk.blocks), 2 * 6)
        for j in range(6):
            for b in (0, 1):
                block = pk.block(j, b)
                self.assertEqual(len(block), 11)       # L entries per block
                for e in block:
                    self.assertIsInstance(e, KeyEntry)
                    self.assertIn(e.basis, (0, 1))
                    self.assertIn(e.bit, (0, 1))

    def test_total_slots_is_two_k_L(self):
        pk = generate_private_key("alice", 6, 11, np.random.default_rng(2))
        self.assertEqual(pk.total_slots, 2 * 6 * 11)
        self.assertEqual(len(pk.slots()), 2 * 6 * 11)

    def test_slots_are_canonical_and_unique(self):
        pk = generate_private_key("alice", 4, 5, np.random.default_rng(3))
        slots = pk.slots()
        self.assertEqual(len(set(slots)), len(slots))     # no duplicates
        for s in slots:
            self.assertEqual(pk.entry(s), pk.block(s.j, s.b)[s.i])

    def test_reveal_discloses_exactly_one_block_per_position(self):
        pk = generate_private_key("alice", 8, 7, np.random.default_rng(4))
        bits = message_to_bits(MESSAGE)
        revealed = pk.reveal(bits)
        self.assertEqual(sorted(revealed), list(range(8)))
        for j, bit in enumerate(bits):
            self.assertEqual(revealed[j], pk.block(j, bit))

    def test_reveal_rejects_a_wrong_length_message(self):
        pk = generate_private_key("alice", 8, 4, np.random.default_rng(5))
        with self.assertRaises(ValueError):
            pk.reveal([0, 1, 0])              # 3 bits, key is for 8

    def test_slot_roles_are_the_reserved_negatives(self):
        # check and decoy slots are flagged by reserved j values a real
        # signature position (j >= 0) can never collide with.
        self.assertEqual(SlotRole.CHECK, -1)
        self.assertEqual(SlotRole.DECOY, -2)

    def test_forger_bound_constants_are_exact(self):
        # the two information-theoretic pass probabilities the whole threshold
        # calculus rests on, asserted as the bare fractions a human derived.
        self.assertEqual(FORGER_PASS_PROB_BLIND, 0.5)
        self.assertEqual(FORGER_PASS_PROB_ONE_COPY, 0.75)


class TestIdentityBinding(unittest.TestCase):
    """Enrolment and lookup are deterministic identity gates, not statistics."""

    def test_public_key_store_add_and_get(self):
        from qds.protocol.keys import HeldQubit
        store = PublicKeyStore(verifier="bob", signer="alice", key_id="pk-1")
        slot = Slot(0, 0, 0)
        store.add(HeldQubit(slot=slot, rho=np.eye(2, dtype=complex) / 2.0))
        self.assertIs(store.get(slot).slot, slot)
        with self.assertRaises(KeyError):
            store.get(Slot(9, 9, 9))

    def test_registry_enrol_then_lookup_succeeds(self):
        reg = SignerRegistry(verifier="bob")
        reg.enrol("alice", "pk-1", at=0.0)
        rec, problems = reg.lookup("alice", "pk-1")
        self.assertIsNotNone(rec)
        self.assertEqual(problems, [])
        self.assertEqual(reg.known_signers(), ["alice"])

    def test_lookup_rejects_unenrolled_signer(self):
        reg = SignerRegistry(verifier="bob")
        rec, problems = reg.lookup("mallory", "pk-1")
        self.assertIsNone(rec)
        self.assertEqual(len(problems), 1)

    def test_lookup_rejects_wrong_key_id(self):
        reg = SignerRegistry(verifier="bob")
        reg.enrol("alice", "pk-REAL", at=0.0)
        rec, problems = reg.lookup("alice", "pk-FAKE")
        self.assertIsNotNone(rec)                 # the signer exists ...
        self.assertEqual(len(problems), 1)        # ... but the key_id is wrong


# ----------------------------------------------------------------------------
# auth.py -- information-theoretic authentication of the classical channel
# ----------------------------------------------------------------------------

class TestOneTimeMac(unittest.TestCase):
    """A Wegman-Carter tag authenticates the exact message under the exact key."""

    KEY = MacKey(index=0, a=0x0123456789ABCDEF, b=0xFEDCBA9876543210)
    MSG = b"the quantum fox jumps the classical dog"

    def test_valid_tag_verifies(self):
        tag = mac_tag(self.KEY, self.MSG)
        self.assertTrue(mac_verify(self.KEY, self.MSG, tag))

    def test_tampered_message_is_rejected(self):
        tag = mac_tag(self.KEY, self.MSG)
        self.assertFalse(mac_verify(self.KEY, self.MSG + b"!", tag))
        self.assertFalse(mac_verify(self.KEY, b"a different message", tag))

    def test_tampered_tag_is_rejected(self):
        tag = mac_tag(self.KEY, self.MSG)
        self.assertFalse(mac_verify(self.KEY, self.MSG, tag ^ 1))

    def test_wrong_key_is_rejected(self):
        tag = mac_tag(self.KEY, self.MSG)
        other = MacKey(index=1, a=0x1111111111111111, b=0x2222222222222222)
        self.assertFalse(mac_verify(other, self.MSG, tag))

    def test_degenerate_hash_key_is_refused(self):
        # a = 0 makes the polynomial hash constant; the type refuses to exist.
        with self.assertRaises(ValueError):
            MacKey(index=2, a=0, b=7)


class TestKeyReservoir(unittest.TestCase):
    """QKD key is handed out strictly in sequence and never reissued."""

    def test_keys_are_issued_in_order_and_consumed(self):
        res = KeyReservoir("alice", "bob", size=4, seed=11)
        k0, k1 = res.take(), res.take()
        self.assertEqual((k0.index, k1.index), (0, 1))
        self.assertEqual(res.consumed, 2)
        self.assertEqual(res.remaining, 2)
        self.assertNotEqual(k0.a, 0)              # reservoir never emits a = 0

    def test_peek_does_not_consume(self):
        res = KeyReservoir("alice", "bob", size=4, seed=12)
        taken = res.take()
        peeked = res.peek(0)                      # verifier side reads the same
        self.assertEqual((peeked.a, peeked.b), (taken.a, taken.b))
        self.assertEqual(res.consumed, 1)         # peek did not advance it

    def test_exhaustion_refuses_to_reuse(self):
        res = KeyReservoir("alice", "bob", size=2, seed=13)
        res.take(); res.take()
        with self.assertRaises(RuntimeError):
            res.take()                            # reuse would void the bound

    def test_peek_out_of_range_raises(self):
        res = KeyReservoir("alice", "bob", size=2, seed=14)
        with self.assertRaises(IndexError):
            res.peek(5)


class TestFreshnessLedger(unittest.TestCase):
    """Replay is a deterministic bookkeeping failure: probability 1, no threshold."""

    def test_a_fresh_message_passes(self):
        led = FreshnessLedger("bob", window_seconds=300.0)
        v = led.check("alice", counter=1, nonce=b"n1", timestamp=1000.0, now=1000.0)
        self.assertTrue(v.ok)

    def test_exact_replay_is_rejected_after_commit(self):
        led = FreshnessLedger("bob", window_seconds=300.0)
        led.commit("alice", counter=1, nonce=b"n1")
        v = led.check("alice", counter=1, nonce=b"n1", timestamp=1000.0, now=1000.0)
        self.assertFalse(v.ok)
        # both the counter and the nonce bindings fire on a verbatim replay
        self.assertGreaterEqual(len(v.reasons), 2)

    def test_a_genuinely_new_message_still_passes(self):
        led = FreshnessLedger("bob", window_seconds=300.0)
        led.commit("alice", counter=1, nonce=b"n1")
        v = led.check("alice", counter=2, nonce=b"n2", timestamp=1000.0, now=1000.0)
        self.assertTrue(v.ok)

    def test_stale_timestamp_is_rejected(self):
        led = FreshnessLedger("bob", window_seconds=300.0)
        v = led.check("alice", counter=1, nonce=b"n1",
                      timestamp=1000.0, now=1000.0 + 100000.0)
        self.assertFalse(v.ok)

    def test_consumed_slots_are_burned(self):
        led = FreshnessLedger("bob", window_seconds=300.0)
        led.commit("alice", counter=1, nonce=b"n1", slots=[(0, 0, 0)])
        # even with a fresh counter and nonce, re-measuring a spent slot fails
        v = led.check("alice", counter=2, nonce=b"n2", timestamp=1000.0,
                      now=1000.0, slots=[(0, 0, 0)])
        self.assertFalse(v.ok)


# ----------------------------------------------------------------------------
# distribution.py -- teleported public-key delivery
# ----------------------------------------------------------------------------

class TestBellOutcomeLeakage(unittest.TestCase):
    """The classical teleportation transcript is a sequence of fair coins."""

    def test_outcome_distribution_is_uniform_for_every_bb84_input(self):
        for a in (0, 1):
            for c in (0, 1):
                with self.subTest(entry=(a, c)):
                    dist = bell_outcome_distribution(KeyEntry(a, c))
                    self.assertEqual(set(dist), {(0, 0), (0, 1), (1, 0), (1, 1)})
                    for p in dist.values():
                        self.assertAlmostEqual(p, 0.25, places=12)
                    self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)

    def test_leakage_is_zero_on_an_ideal_channel(self):
        # the max total-variation distance between any two key entries' (u, v)
        # transcripts: identically zero because all the state is in the shared
        # correlation, none in the broadcast.
        self.assertAlmostEqual(teleportation_leakage(), 0.0, places=12)

    def test_leakage_stays_zero_even_through_noise(self):
        cfg = DistributionConfig(noise=NoiseModel(channel_depolarizing=0.1))
        self.assertAlmostEqual(teleportation_leakage(cfg), 0.0, places=12)


class TestTeleportation(unittest.TestCase):
    """After the Pauli correction the recipient holds the input state exactly."""

    def test_ideal_teleport_recovers_each_bb84_state(self):
        cfg = DistributionConfig(noise=IDEAL, check_pairs=0, decoy_slots=0)
        for a in (0, 1):
            for c in (0, 1):
                with self.subTest(entry=(a, c)):
                    tr = teleport_state(KeyEntry(a, c), Slot(0, c, 0), cfg,
                                        np.random.default_rng(20))
                    # fidelity with the intended BB84 target is 1 on an ideal
                    # channel -- the honest verifier's check passes w.p. 1.
                    self.assertAlmostEqual(tr.fidelity, 1.0, places=12)
                    self.assertEqual(tr.status, "click")

    def test_delivered_marginal_is_a_valid_single_qubit_state(self):
        cfg = DistributionConfig(noise=IDEAL, check_pairs=0, decoy_slots=0)
        tr = teleport_state(KeyEntry(1, 0), Slot(0, 0, 0), cfg,
                            np.random.default_rng(21))
        self.assertEqual(tr.rho_b.shape, (2, 2))
        self.assertAlmostEqual(float(np.real(np.trace(tr.rho_b))), 1.0, places=12)


class TestDistribution(unittest.TestCase):
    """One distribution phase lays down check, decoy and signature slots."""

    def test_slot_structure_counts(self):
        rng = np.random.default_rng(30)
        pk = generate_private_key("alice", 8, 5, rng)
        cfg = DistributionConfig(noise=IDEAL, check_pairs=7, decoy_slots=9)
        res = distribute_public_key(pk, "bob", cfg, rng)
        store = res.store
        self.assertEqual(len(store.check_slots), 7)
        self.assertEqual(len(store.decoy_slots), 9)
        self.assertEqual(len(res.decoy_key), 9)
        # every slot is held exactly once: checks + decoys + signature slots
        self.assertEqual(len(store.qubits), 7 + 9 + pk.total_slots)
        sig = [s for s in store.qubits if s.j >= 0]
        chk = [s for s in store.qubits if s.j == SlotRole.CHECK]
        dec = [s for s in store.qubits if s.j == SlotRole.DECOY]
        self.assertEqual((len(sig), len(chk), len(dec)),
                         (pk.total_slots, 7, 9))

    def test_store_identity_matches_the_signer(self):
        rng = np.random.default_rng(31)
        pk = generate_private_key("alice", 4, 3, rng)
        res = distribute_public_key(
            pk, "bob", DistributionConfig(noise=IDEAL, check_pairs=2,
                                          decoy_slots=2), rng)
        self.assertEqual(res.store.verifier, "bob")
        self.assertEqual(res.store.signer, "alice")
        self.assertEqual(res.store.key_id, pk.key_id)


# ----------------------------------------------------------------------------
# signing.py -- private key + message -> transferable declaration
# ----------------------------------------------------------------------------

class TestSigning(unittest.TestCase):
    """Signing publishes exactly the key blocks for the claimed bits."""

    def test_message_to_bits_is_big_endian(self):
        self.assertEqual(message_to_bits(b"\x80"), (1, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(message_to_bits(b"\x01"), (0, 0, 0, 0, 0, 0, 0, 1))

    def test_bits_round_trip(self):
        for data in (b"\x00", b"\xff", b"\xb0\x6a", b"quantum"):
            with self.subTest(data=data):
                self.assertEqual(bits_to_message(message_to_bits(data)), data)

    def test_message_length_determines_bit_count(self):
        self.assertEqual(len(message_to_bits(b"\xab")), 8)
        self.assertEqual(len(message_to_bits(b"\xab\xcd\xef")), 24)

    def test_bits_to_message_needs_whole_bytes(self):
        with self.assertRaises(ValueError):
            bits_to_message([1, 0, 1])            # 3 bits is not a byte

    def test_revealed_entries_are_the_private_key_blocks(self):
        rng = np.random.default_rng(40)
        pk = generate_private_key("alice", 8, 5, rng)
        decl = sign_message(pk, MESSAGE, ["bob"], {}, counter=1, rng=rng,
                            authenticate=False)
        bits = message_to_bits(MESSAGE)
        self.assertEqual(decl.message_bits, bits)
        self.assertEqual(decl.L, 5)
        self.assertEqual(decl.n_checks, 8 * 5)
        self.assertEqual(len(decl.slots()), 8 * 5)
        for j, bit in enumerate(bits):
            # the published block is the private-key block for the claimed bit
            self.assertEqual(decl.revealed[j], pk.block(j, bit))

    def test_wrong_length_message_raises(self):
        pk = generate_private_key("alice", 8, 4, np.random.default_rng(41))
        with self.assertRaises(SigningError):
            sign_message(pk, b"\x00\x00", ["bob"], {}, counter=1,
                         rng=np.random.default_rng(0), authenticate=False)

    def test_empty_recipient_set_raises(self):
        pk = generate_private_key("alice", 8, 4, np.random.default_rng(42))
        with self.assertRaises(SigningError):
            sign_message(pk, MESSAGE, [], {}, counter=1,
                         rng=np.random.default_rng(0), authenticate=False)


# ----------------------------------------------------------------------------
# verification.py -- projective check, mismatch counting, the two thresholds
# ----------------------------------------------------------------------------

class TestThresholds(unittest.TestCase):
    """The forger floors are 1/4 and 1/2, and the thresholds order e < s_a < s_v < 1/4."""

    def test_forger_error_rates_are_exact(self):
        self.assertEqual(FORGER_ERROR_RATE, 0.25)        # 1 - 3/4, one copy
        self.assertEqual(BLIND_FORGER_ERROR_RATE, 0.5)   # 1 - 1/2, no copy

    def test_equal_thirds_placement(self):
        # from_noise_floor splits the gap [e, 1/4) into equal thirds; at e = 0
        # that is s_a = 1/12 and s_v = 1/6, numbers typed here by hand.
        p = VerificationPolicy.from_noise_floor(0.0)
        self.assertAlmostEqual(p.s_a, 0.0833333, places=6)
        self.assertAlmostEqual(p.s_v, 0.1666667, places=6)

    def test_threshold_ordering_holds_for_both_rules(self):
        for p in (VerificationPolicy.from_noise_floor(0.01),
                  VerificationPolicy.balanced(0.01)):
            with self.subTest(policy=type(p).__name__):
                self.assertLess(p.honest_error_rate, p.s_a)
                self.assertLess(p.s_a, p.s_v)
                self.assertLess(p.s_v, FORGER_ERROR_RATE)
                self.assertTrue(p.sound)
                # the transfer threshold is the looser of the two
                self.assertEqual(p.threshold("accept"), p.s_a)
                self.assertEqual(p.threshold("transfer"), p.s_v)
                self.assertGreater(p.threshold("transfer"),
                                   p.threshold("accept"))

    def test_too_noisy_hardware_is_not_sound(self):
        # e >= 1/4 leaves no gap for any threshold to live in.
        p = VerificationPolicy(honest_error_rate=0.30, s_a=0.30, s_v=0.30)
        self.assertFalse(p.sound)
        self.assertTrue(p.soundness_problems())

    def test_unknown_level_raises(self):
        p = VerificationPolicy.from_noise_floor(0.01)
        with self.assertRaises(ValueError):
            p.threshold("whenever")


class TestVerification(unittest.TestCase):
    """An honest check passes at rate 0; a key that disagrees is driven to reject."""

    def test_honest_declaration_on_ideal_channel_is_accepted(self):
        pk, res, rng = _fresh_key(seed=50)
        reservoir = KeyReservoir("alice", "bob", size=16, seed=51)
        decl = sign_message(pk, MESSAGE, ["bob"], {"bob": reservoir},
                            counter=1, rng=rng)
        policy = VerificationPolicy.from_noise_floor(0.0)
        report = verify_declaration(decl, res.store, policy, rng,
                                    reservoir=reservoir, noise=IDEAL)
        self.assertTrue(report.identity_ok)
        self.assertTrue(report.mac_ok)
        self.assertTrue(report.measured)
        # ideal delivery + deterministic readout => every check matches
        self.assertEqual(report.mismatch_rate, 0.0)
        self.assertEqual(report.n_mismatch, 0)
        self.assertTrue(report.accepted)

    def test_corrupted_revealed_entries_fail_on_rate(self):
        pk, res, rng = _fresh_key(seed=60)
        # flip the bit of every key entry: the delivered qubits are the TRUE
        # states, so each flipped claim mismatches -- the classical gates still
        # pass (the tag is computed over the corrupted payload), so the failure
        # is unambiguously the physics, not the bookkeeping.
        flipped = {kb: tuple(KeyEntry(e.basis, 1 - e.bit) for e in blk)
                   for kb, blk in pk.blocks.items()}
        pk_bad = dataclasses.replace(pk, blocks=flipped)
        reservoir = KeyReservoir("alice", "bob", size=16, seed=61)
        decl = sign_message(pk_bad, MESSAGE, ["bob"], {"bob": reservoir},
                            counter=1, rng=rng)
        policy = VerificationPolicy.from_noise_floor(0.0)
        report = verify_declaration(decl, res.store, policy, rng,
                                    reservoir=reservoir, noise=IDEAL)
        self.assertTrue(report.mac_ok)            # classical layer is satisfied
        self.assertTrue(report.measured)
        self.assertEqual(report.mismatch_rate, 1.0)   # every flipped bit misses
        self.assertFalse(report.accepted)


# ----------------------------------------------------------------------------
# session.py -- the five-phase assembly
# ----------------------------------------------------------------------------

class TestSession(unittest.TestCase):
    """End to end: zero-noise anchor, symmetry, calibration, physics faithfulness."""

    def test_ideal_run_is_the_zero_noise_soundness_anchor(self):
        # a perfect link accepts with mismatch rate exactly 0, at both the
        # accept and the (looser) transfer threshold.
        acc = IDEAL_SESSION.reports["bob:accept"]
        tra = IDEAL_SESSION.reports["charlie:transfer"]
        self.assertEqual(acc.mismatch_rate, 0.0)
        self.assertTrue(acc.accepted)
        self.assertEqual(tra.mismatch_rate, 0.0)
        self.assertTrue(tra.accepted)
        self.assertTrue(IDEAL_SESSION.policy.sound)

    def test_symmetrisation_conserves_the_pooled_error(self):
        # distribute the same key to two recipients over the SAME symmetric
        # channel, then permute the signature slots between them.
        rng = np.random.default_rng(70)
        pk = generate_private_key("alice", 8, 12, rng)
        cfg = DistributionConfig(noise=NoiseModel(channel_depolarizing=0.08),
                                 check_pairs=0, decoy_slots=0)
        rb = distribute_public_key(pk, "bob", cfg, rng)
        rc = distribute_public_key(pk, "charlie", cfg, rng)
        stores = [rb.store, rc.store]

        def pooled_error(stores):
            # sum over (recipient, slot) of 1 - <psi|rho|psi> against the true
            # BB84 target, computed here from the key -- never from the
            # symmetriser's own output.
            total = 0.0
            for s in stores:
                for slot in pk.slots():
                    target = bb84_state(pk.entry(slot).basis, pk.entry(slot).bit)
                    total += 1.0 - fidelity_pure(s.qubits[slot].rho, target)
            return total

        before = pooled_error(stores)
        info = symmetrise_keys(stores, rng)
        after = pooled_error(stores)

        self.assertTrue(info["performed"])
        self.assertEqual(info["slots_permuted"], pk.total_slots)
        self.assertGreater(info["moved"], 0)      # it really did redistribute
        # the permutation only relabels who holds which qubit, so the pooled
        # error over both recipients is invariant.
        self.assertAlmostEqual(before, after, places=9)

    def test_symmetrisation_is_skipped_for_a_lone_recipient(self):
        rng = np.random.default_rng(71)
        pk = generate_private_key("alice", 4, 4, rng)
        res = distribute_public_key(
            pk, "bob", DistributionConfig(noise=IDEAL, check_pairs=0,
                                          decoy_slots=0), rng)
        info = symmetrise_keys([res.store], rng)
        self.assertFalse(info["performed"])

    def test_calibration_floor_matches_an_ideal_channel(self):
        rng = np.random.default_rng(80)
        pk = generate_private_key("alice", 8, 4, rng)
        cfg = DistributionConfig(noise=IDEAL, check_pairs=0, decoy_slots=24)
        res = distribute_public_key(pk, "bob", cfg, rng)
        cal = calibrate_from_decoys(res.store, res.decoy_key, rng,
                                    spec_rate=0.0, noise=IDEAL,
                                    predicted_rate=IDEAL.predicted_qber())
        self.assertEqual(cal.decoys_measured, 24)    # all arrive on an ideal link
        self.assertEqual(cal.decoy_mismatches, 0)
        self.assertEqual(cal.point_estimate, 0.0)
        self.assertEqual(cal.lower_confidence, 0.0)
        self.assertFalse(cal.spec_violation)
        self.assertEqual(cal.predicted_rate, 0.0)

    def test_measured_rate_matches_the_analytic_noise_floor(self):
        # the faithfulness cross-check: the empirical mismatch rate from a full
        # density-matrix session must agree with the closed-form predicted_qber
        # computed independently in qds.channel.
        predicted = NOISY.predicted_qber()
        # internal consistency of the closed form: the pooled rate is the mean
        # of the two per-basis rates, since the signature basis is uniform.
        self.assertAlmostEqual(
            predicted,
            0.5 * (NOISY.predicted_qber(basis="Z")
                   + NOISY.predicted_qber(basis="X")),
            places=12)
        self.assertTrue(0.0 < predicted < FORGER_ERROR_RATE)

        acc = NOISY_SESSION.reports["bob:accept"]
        tra = NOISY_SESSION.reports["charlie:transfer"]
        n_mismatch = acc.n_mismatch + tra.n_mismatch
        n_checked = acc.n_checked + tra.n_checked
        empirical = n_mismatch / n_checked
        # ~380 pooled checks: the sampling spread is ~0.012, so a 0.03 band is
        # a few standard deviations and comfortably non-circular.
        self.assertAlmostEqual(empirical, predicted, delta=0.03)


class TestDeterminism(unittest.TestCase):
    """A seeded run is reproducible bit for bit -- the point of seed-pinning."""

    def _signature(self, seed):
        s = QDSSession(
            config=SessionConfig(
                message_bits=8, L=12, verifiers=("bob", "charlie"),
                distribution=DistributionConfig(noise=IDEAL, check_pairs=4,
                                                decoy_slots=4)),
            seed=seed)
        s.run(MESSAGE)
        r = s.reports["bob:accept"]
        return (s.declaration.nonce, r.n_mismatch, r.n_checked, r.accepted,
                s.policy.s_a, s.policy.s_v)

    def test_same_seed_reproduces_the_whole_flow(self):
        self.assertEqual(self._signature(1234), self._signature(1234))

    def test_different_seeds_differ(self):
        # the nonce alone must change; a seed that produced identical nonces
        # would be a broken RNG wiring, not reproducibility.
        self.assertNotEqual(self._signature(1234)[0], self._signature(5678)[0])


if __name__ == "__main__":
    unittest.main()
