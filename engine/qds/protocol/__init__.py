"""The teleportation-based quantum digital signature protocol.

Read the modules in this order; each one assumes the previous:

:mod:`~qds.protocol.keys`
    what a private key is, what a recipient holds, and the two forgery
    bounds (1/2 blind, 3/4 with one copy) that everything else is measured
    against.
:mod:`~qds.protocol.auth`
    Wegman-Carter one-time MACs over GF(2^64) and the deterministic
    anti-replay ledger -- the classical half, kept free of hash functions so
    the security claim stays information-theoretic.
:mod:`~qds.protocol.distribution`
    Bell pairs, teleportation, and the proof that the classical transcript
    leaks nothing.
:mod:`~qds.protocol.signing`
    a message plus a private key becomes a transferable declaration.
:mod:`~qds.protocol.verification`
    projective measurement, mismatch counting, and where the two thresholds
    sit.
:mod:`~qds.protocol.session`
    the phases in order, including the symmetrisation step the
    non-repudiation bound depends on.
"""

from .auth import (FreshnessLedger, FreshnessVerdict, KeyReservoir, MacKey,
                   mac_tag, mac_verify)
from .distribution import (DistributionConfig, DistributionOracle,
                           DistributionResult, PairRound, TeleportRound,
                           bell_outcome_distribution, distribute_pair,
                           distribute_public_key, teleport_state,
                           teleportation_leakage)
from .keys import (FORGER_PASS_PROB_BLIND, FORGER_PASS_PROB_ONE_COPY,
                   HeldQubit, KeyEntry, PrivateKey, PublicKeyStore,
                   SignerRecord, SignerRegistry, Slot, SlotRole,
                   generate_private_key)
from .session import (Calibration, QDSSession, SessionConfig, SessionResult,
                      calibrate_from_decoys, symmetrise_keys)
from .signing import (SignatureDeclaration, bits_to_message, message_to_bits,
                      sign_message)
from .verification import (BLIND_FORGER_ERROR_RATE, FORGER_ERROR_RATE,
                           SlotCheck, VerificationPolicy, VerificationReport,
                           measure_slot, verify_declaration)

__all__ = [
    # keys
    "KeyEntry", "PrivateKey", "generate_private_key", "Slot", "SlotRole",
    "HeldQubit", "PublicKeyStore", "SignerRecord", "SignerRegistry",
    "FORGER_PASS_PROB_BLIND", "FORGER_PASS_PROB_ONE_COPY",
    # auth
    "MacKey", "KeyReservoir", "mac_tag", "mac_verify",
    "FreshnessLedger", "FreshnessVerdict",
    # distribution
    "DistributionConfig", "DistributionResult", "DistributionOracle",
    "PairRound", "TeleportRound", "distribute_pair", "teleport_state",
    "distribute_public_key", "bell_outcome_distribution",
    "teleportation_leakage",
    # signing
    "SignatureDeclaration", "sign_message", "message_to_bits",
    "bits_to_message",
    # verification
    "VerificationPolicy", "VerificationReport", "SlotCheck",
    "verify_declaration", "measure_slot",
    "FORGER_ERROR_RATE", "BLIND_FORGER_ERROR_RATE",
    # session
    "SessionConfig", "QDSSession", "SessionResult", "Calibration",
    "symmetrise_keys", "calibrate_from_decoys",
]
