"""Signing: private key + message -> a transferable signed declaration.

What a signature *is* here
--------------------------
There is no trapdoor and no hard problem.  To sign the bit string
``m = m_0 m_1 ... m_{k-1}`` Alice simply **publishes** the private-key blocks
that correspond to the bits she is claiming::

    signature(m) = ( K[0, m_0], K[1, m_1], ..., K[k-1, m_{k-1}] )

Each ``K[j, b]`` is ``L`` entries ``(a_i, c_i)`` naming BB84 states.  The
recipient already holds those states as qubits and can check the claim by
measuring.  The blocks for the *complementary* bit values are never revealed,
which is what stops a recipient who has seen one signature from signing a
different message: to flip bit ``j`` he would have to produce ``K[j, 1 - m_j]``,
and nothing he has ever seen depends on it.

Why the key is linear in the message length
-------------------------------------------
The obvious economy -- hash the message and sign the digest -- is not
available.  A collision-resistant hash is a computational assumption, and the
entire point of this scheme is that its security survives an adversary with
unbounded computation.  An information-theoretic alternative (a universal hash
under a shared one-time key) would need a key shared *identically* with every
recipient; with pairwise keys Alice could hand different recipients different
hash keys and thereby repudiate.  So the key really is ``2 * k * L`` entries
long and that cost is structural, not an implementation shortcut.

What travels with the signature
-------------------------------
The quantum part proves *authorship*.  It says nothing about *when*, and
nothing about *who is talking to whom*, so three classical bindings ride along
and each is authenticated by the one-time MAC from :mod:`qds.protocol.auth`:

``counter`` and ``nonce``
    freshness -- see :class:`~qds.protocol.auth.FreshnessLedger`.
``recipients``
    the exact set of parties Alice intends; without it a dishonest recipient
    could tell one verifier the message was broadcast and another that it was
    private.
``timestamp``
    a coarse liveness bound, checked against a window rather than trusted.

One MAC tag per recipient, computed by Alice
--------------------------------------------
:attr:`SignatureDeclaration.mac_tags` carries a separate tag for *every*
recipient, all produced by Alice at signing time from her pairwise reservoirs.
The alternative -- Bob re-authenticating with his own Bob-Charlie key before
forwarding -- would let Bob alter the declaration and re-tag it, so Charlie
would be verifying Bob's word rather than Alice's.  With Alice's tags, a
forwarded declaration is authenticated end to end and Bob's only power is to
refuse to forward, which is a denial of service and not a repudiation.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .auth import KeyReservoir, MacKey, mac_tag, mac_verify
from .keys import KeyEntry, PrivateKey, Slot

__all__ = [
    "NONCE_BYTES", "SigningError",
    "message_to_bits", "bits_to_message", "message_capacity_bits",
    "SignatureDeclaration", "sign_message",
]

#: Nonce width.  128 bits makes an accidental repeat negligible without any
#: assumption about the randomness source beyond it being unbiased.
NONCE_BYTES = 16


class SigningError(RuntimeError):
    """Raised when a signing request cannot be honoured safely."""


# --------------------------------------------------------------------------
# message <-> bits
# --------------------------------------------------------------------------

def message_to_bits(data: bytes) -> Tuple[int, ...]:
    """Big-endian bit expansion: ``b'\\x80'`` -> ``(1, 0, 0, 0, 0, 0, 0, 0)``."""
    return tuple((byte >> (7 - k)) & 1 for byte in data for k in range(8))


def bits_to_message(bits: Sequence[int]) -> bytes:
    """Inverse of :func:`message_to_bits`."""
    if len(bits) % 8:
        raise ValueError(f"need a whole number of bytes, got {len(bits)} bits")
    out = bytearray()
    for i in range(0, len(bits), 8):
        value = 0
        for b in bits[i:i + 8]:
            value = (value << 1) | (int(b) & 1)
        out.append(value)
    return bytes(out)


def message_capacity_bits(private_key: PrivateKey) -> int:
    """How long a message this key can sign.  One key, one message, ever."""
    return private_key.message_bits


# --------------------------------------------------------------------------
# canonical encoding
# --------------------------------------------------------------------------
# Every field is length-prefixed so the concatenation is injective: two
# different declarations cannot produce the same byte string.  That matters
# because the Wegman-Carter bound is a statement about *distinct* messages,
# and an ambiguous encoding would silently turn two messages into one.

def _enc_bytes(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def _enc_str(text: str) -> bytes:
    return _enc_bytes(text.encode("utf-8"))


def _enc_u64(value: int) -> bytes:
    return struct.pack(">Q", int(value) & ((1 << 64) - 1))


def _enc_block(entries: Sequence[KeyEntry]) -> bytes:
    """``L`` entries as one byte each: ``(basis << 1) | bit``."""
    return _enc_bytes(bytes((int(e.basis) << 1) | int(e.bit) for e in entries))


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SignatureDeclaration:
    """Everything Alice broadcasts when she signs one message.

    This object is the *entire* wire format.  A verifier needs nothing else
    except the qubits he already holds and the MAC key he already shares --
    which is the property that makes the scheme transferable: Bob can hand
    this to Charlie unchanged and Charlie's check is as strong as Bob's.
    """

    signer: str
    key_id: str
    message: bytes
    message_bits: Tuple[int, ...]
    #: ``j -> K[j, message_bits[j]]``, the revealed half of the private key
    revealed: Dict[int, Tuple[KeyEntry, ...]]
    L: int
    counter: int
    nonce: bytes
    timestamp: float
    recipients: Tuple[str, ...]
    #: recipient -> (reservoir key index, tag).  Empty only for an
    #: unauthenticated declaration, which the verifier will reject.
    mac_tags: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # -- structure ---------------------------------------------------------
    def slots(self) -> List[Slot]:
        """The public-key positions this declaration asks to be measured."""
        return [
            Slot(j, int(self.message_bits[j]), i)
            for j in sorted(self.revealed)
            for i in range(len(self.revealed[j]))
        ]

    def slot_tuples(self) -> List[Tuple[int, int, int]]:
        """:meth:`slots` as plain tuples, for the freshness ledger."""
        return [(s.j, s.b, s.i) for s in self.slots()]

    def entry(self, slot: Slot) -> KeyEntry:
        """The claimed state for one slot."""
        return self.revealed[slot.j][slot.i]

    @property
    def n_checks(self) -> int:
        return sum(len(block) for block in self.revealed.values())

    # -- authentication ----------------------------------------------------
    def signed_bytes(self) -> bytes:
        """The exact byte string the MAC covers.

        Note what is *inside*: the revealed key material and the recipient
        list, not just the message.  A dishonest forwarder who drops a few
        key entries to nudge the mismatch count, or who rewrites the
        recipient set, changes this string and cannot retag it.
        """
        parts = [
            _enc_str("TQDS-DECL-v1"),
            _enc_str(self.signer),
            _enc_str(self.key_id),
            _enc_u64(self.counter),
            _enc_bytes(self.nonce),
            _enc_bytes(struct.pack(">d", float(self.timestamp))),
            _enc_bytes(self.message),
            _enc_u64(len(self.message_bits)),
            _enc_bytes(bytes(int(b) & 1 for b in self.message_bits)),
            _enc_u64(self.L),
            _enc_u64(len(self.recipients)),
        ]
        parts.extend(_enc_str(r) for r in self.recipients)
        parts.append(_enc_u64(len(self.revealed)))
        for j in sorted(self.revealed):
            parts.append(_enc_u64(j))
            parts.append(_enc_block(self.revealed[j]))
        return b"".join(parts)

    def tag_for(self, verifier: str) -> Optional[Tuple[int, int]]:
        return self.mac_tags.get(verifier)

    def check_mac(self, verifier: str,
                  reservoir: Optional[KeyReservoir]) -> Tuple[bool, List[str]]:
        """Verify this declaration's tag for one recipient.

        Returns ``(ok, problems)``.  A missing reservoir, a missing tag, an
        out-of-range key index and a bad tag are reported as distinct
        problems because the detection engine attributes them differently: a
        bad tag is an active forgery attempt, a missing tag is usually a
        misconfigured or excluded recipient.
        """
        problems: List[str] = []
        if reservoir is None:
            problems.append(
                f"no shared authentication key with signer '{self.signer}'")
            return False, problems
        entry = self.mac_tags.get(verifier)
        if entry is None:
            problems.append(
                f"declaration carries no MAC tag for '{verifier}' "
                f"(tags present for: {sorted(self.mac_tags) or 'nobody'})")
            return False, problems
        index, tag = entry
        try:
            key = reservoir.peek(index)
        except (IndexError, ValueError) as exc:
            problems.append(f"MAC key index {index} is not usable: {exc}")
            return False, problems
        if not mac_verify(key, self.signed_bytes(), tag):
            problems.append(
                f"MAC tag does not authenticate this declaration under key "
                f"index {index}; the classical payload was altered or forged")
            return False, problems
        return True, problems

    # -- reporting ---------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "signer": self.signer,
            "key_id": self.key_id,
            "message_hex": self.message.hex(),
            "message_bits": len(self.message_bits),
            "L": self.L,
            "checks": self.n_checks,
            "counter": self.counter,
            "nonce": self.nonce.hex(),
            "timestamp": self.timestamp,
            "recipients": list(self.recipients),
            "mac_tags": {k: {"index": v[0], "tag": f"{v[1]:016x}"}
                         for k, v in self.mac_tags.items()},
            "declaration_bytes": len(self.signed_bytes()),
        }


# --------------------------------------------------------------------------
# signing
# --------------------------------------------------------------------------

def sign_message(private_key: PrivateKey, message: bytes,
                 recipients: Sequence[str],
                 reservoirs: Mapping[str, KeyReservoir],
                 counter: int,
                 rng: Optional[np.random.Generator] = None,
                 timestamp: Optional[float] = None,
                 nonce: Optional[bytes] = None,
                 authenticate: bool = True) -> SignatureDeclaration:
    """Sign ``message`` under ``private_key`` for a fixed recipient set.

    ``reservoirs`` maps recipient name to the Alice-recipient key reservoir.
    One key is consumed per recipient and never reused -- that is enforced by
    :meth:`~qds.protocol.auth.KeyReservoir.take`, not by convention here.

    ``authenticate=False`` produces a declaration with no tags.  It exists so
    the attack suite can mount an unauthenticated-impersonation attempt
    against a real verifier and watch it be refused; honest callers leave it
    alone.
    """
    bits = message_to_bits(message)
    if len(bits) != private_key.message_bits:
        raise SigningError(
            f"key '{private_key.key_id}' signs exactly {private_key.message_bits} "
            f"bits ({private_key.message_bits // 8} bytes); this message is "
            f"{len(bits)} bits. The key length is linear in the message length "
            f"by design -- see the module docstring."
        )
    if not recipients:
        raise SigningError("a declaration must name at least one recipient")

    revealed = {j: private_key.block(j, int(b)) for j, b in enumerate(bits)}
    if nonce is None:
        nonce = (bytes(rng.integers(0, 256, size=NONCE_BYTES, dtype=np.uint8))
                 if rng is not None else os.urandom(NONCE_BYTES))
    if timestamp is None:
        timestamp = time.time()

    decl = SignatureDeclaration(
        signer=private_key.signer,
        key_id=private_key.key_id,
        message=bytes(message),
        message_bits=bits,
        revealed=revealed,
        L=private_key.L,
        counter=int(counter),
        nonce=bytes(nonce),
        timestamp=float(timestamp),
        recipients=tuple(recipients),
        mac_tags={},
    )
    if not authenticate:
        return decl

    payload = decl.signed_bytes()
    tags: Dict[str, Tuple[int, int]] = {}
    for name in recipients:
        res = reservoirs.get(name)
        if res is None:
            raise SigningError(
                f"no authentication key reservoir shared with '{name}'; "
                "the classical channel cannot be authenticated without one")
        key = res.take()
        tags[name] = (key.index, mac_tag(key, payload))
    # frozen dataclass: rebuild rather than mutate, so the tagged object and
    # the bytes that were tagged can never drift apart
    return SignatureDeclaration(
        signer=decl.signer, key_id=decl.key_id, message=decl.message,
        message_bits=decl.message_bits, revealed=decl.revealed, L=decl.L,
        counter=decl.counter, nonce=decl.nonce, timestamp=decl.timestamp,
        recipients=decl.recipients, mac_tags=tags,
    )
