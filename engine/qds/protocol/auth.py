"""Information-theoretically secure authentication of the classical channel.

A quantum signature scheme is only as strong as the classical messages that
accompany it: the Pauli-correction bits, the revealed private key, the nonce
and the counter all travel over an authenticated-but-public channel.  If that
authentication rested on a computational assumption (HMAC-SHA256, RSA, an
elliptic curve) the whole information-theoretic security claim would collapse
to that assumption -- which is precisely the thing a quantum adversary is
expected to break.

So there is no hash function anywhere in this module.  Classical messages are
authenticated with a **Wegman-Carter one-time MAC** built from a strongly
universal hash family over the finite field GF(2^64), keyed from the
information-theoretically secure key reservoir that Alice and each recipient
established by QKD.  Against an adversary with unbounded computing power the
probability of producing a valid tag for a message she has not seen is at most
``(blocks + 1) / 2**64``, and that bound is a counting argument over the key
space, not a hardness assumption.

Key discipline
--------------
The security proof requires a *fresh* key pair per authenticated message.
:class:`KeyReservoir` enforces this: keys are handed out strictly in sequence,
never reissued, and the reservoir refuses to authenticate once exhausted
rather than silently reusing material.  Reuse of a Wegman-Carter key leaks the
hash key and destroys the bound, so this is not a detail we let the caller get
wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "GF64_MODULUS_BITS", "gf64_mul", "gf64_pow",
    "MacKey", "KeyReservoir", "mac_tag", "mac_verify", "MAC_FORGERY_BOUND_LOG2",
    "FreshnessLedger", "FreshnessVerdict",
]

# --------------------------------------------------------------------------
# GF(2^64) arithmetic
# --------------------------------------------------------------------------
# Field is GF(2)[x] / (x^64 + x^4 + x^3 + x + 1).  The reduction constant below
# is the low part of that polynomial, i.e. x^64 == x^4 + x^3 + x + 1 == 0x1b.
# tests/test_auth.py runs Rabin's irreducibility test on it rather than taking
# the choice on trust.
GF64_MODULUS_BITS = (1 << 64) | 0x1B
_REDUCE = 0x1B
_MASK64 = (1 << 64) - 1

#: Number of bits of security of one tag; forgery probability <= blocks/2**64.
MAC_FORGERY_BOUND_LOG2 = 64


def gf64_mul(a: int, b: int) -> int:
    """Carry-less multiply modulo the field polynomial (constant-time-ish)."""
    a &= _MASK64
    b &= _MASK64
    result = 0
    for _ in range(64):
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & (1 << 64):
            a = (a & _MASK64) ^ _REDUCE
    return result & _MASK64


def gf64_pow(a: int, e: int) -> int:
    """Exponentiation by squaring in GF(2^64)."""
    out, base = 1, a & _MASK64
    while e:
        if e & 1:
            out = gf64_mul(out, base)
        base = gf64_mul(base, base)
        e >>= 1
    return out


def _blocks(data: bytes) -> List[int]:
    """Split a byte string into 64-bit field elements, length-prefixed.

    The length prefix is what makes the family strongly universal on
    *variable-length* inputs: without it, appending zero bytes would not
    change the hash and an adversary could extend a message for free.
    """
    padded = len(data).to_bytes(8, "big") + data
    if len(padded) % 8:
        padded += b"\x00" * (8 - len(padded) % 8)
    return [int.from_bytes(padded[i:i + 8], "big") for i in range(0, len(padded), 8)]


# --------------------------------------------------------------------------
# one-time MAC
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MacKey:
    """One-time Wegman-Carter key: a hash key ``a`` and a one-time pad ``b``."""

    index: int
    a: int
    b: int
    owner: str = ""

    def __post_init__(self) -> None:
        if self.a == 0:
            # a = 0 would make the hash constant; the reservoir never emits it
            raise ValueError("degenerate MAC key (a = 0)")


def mac_tag(key: MacKey, data: bytes) -> int:
    """Tag = (polynomial hash of ``data`` under ``a``) XOR ``b``.

    ``h_a(M) = sum_i M_i a^(i+1)`` evaluated in GF(2^64).  Two distinct
    messages collide for at most ``deg`` of the ``2**64`` possible ``a``
    values, and the XOR with a fresh ``b`` hides the hash value itself, giving
    the ``(blocks+1)/2**64`` substitution bound.
    """
    acc = 0
    power = key.a
    for blk in _blocks(data):
        acc ^= gf64_mul(blk, power)
        power = gf64_mul(power, key.a)
    return acc ^ (key.b & _MASK64)


def mac_verify(key: MacKey, data: bytes, tag: int) -> bool:
    """Constant-time-ish tag comparison."""
    expected = mac_tag(key, data)
    return (expected ^ (tag & _MASK64)) == 0


@dataclass
class KeyReservoir:
    """A finite supply of information-theoretically secure shared key.

    Models the output of a QKD link between two named parties.  Keys are
    consumed in strict sequence and never reissued; ``consumed`` is public
    accounting that both parties can compare, so a desynchronisation (the
    signature of a replay or an injected message) is itself detectable.
    """

    party_a: str
    party_b: str
    size: int = 4096
    seed: Optional[int] = None
    consumed: int = 0
    _keys: List[Tuple[int, int]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.seed is None:
            rand = os.urandom
        else:
            import numpy as np

            gen = np.random.default_rng(self.seed)

            def rand(n: int) -> bytes:
                return bytes(int(x) for x in gen.integers(0, 256, size=n))

        self._keys = []
        for _ in range(self.size):
            a = int.from_bytes(rand(8), "big") or 1
            b = int.from_bytes(rand(8), "big")
            self._keys.append((a, b))

    @property
    def label(self) -> str:
        return f"{self.party_a}<->{self.party_b}"

    @property
    def remaining(self) -> int:
        return self.size - self.consumed

    def take(self) -> MacKey:
        """Consume the next one-time key."""
        if self.consumed >= self.size:
            raise RuntimeError(
                f"authentication key reservoir {self.label} exhausted; refusing "
                "to reuse a one-time MAC key (that would void the security bound)"
            )
        a, b = self._keys[self.consumed]
        key = MacKey(index=self.consumed, a=a, b=b, owner=self.label)
        self.consumed += 1
        return key

    def peek(self, index: int) -> MacKey:
        """Read the key at ``index`` without consuming (verifier side)."""
        if not 0 <= index < self.size:
            raise IndexError(f"MAC key index {index} outside reservoir")
        a, b = self._keys[index]
        return MacKey(index=index, a=a, b=b, owner=self.label)


# --------------------------------------------------------------------------
# replay / freshness accounting
# --------------------------------------------------------------------------

@dataclass
class FreshnessVerdict:
    """Outcome of the deterministic anti-replay checks."""

    ok: bool
    reasons: List[str] = field(default_factory=list)

    def fail(self, reason: str) -> "FreshnessVerdict":
        self.ok = False
        self.reasons.append(reason)
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


@dataclass
class FreshnessLedger:
    """Per-verifier record of what has already been accepted.

    Three independent bindings have to hold for a signed message to be fresh,
    and each one alone stops a different replay variant:

    ``counter``
        Strictly increasing per signer.  Stops verbatim replay and
        out-of-order injection.
    ``nonce``
        Never-before-seen random value inside the MAC'd payload.  Stops replay
        with a modified counter, because the tag covers the nonce.
    ``slots``
        Public-key positions are one-time: measuring a stored qubit destroys
        it.  A replayed signature necessarily points at slots this verifier
        has already consumed, so the ledger catches it even if the classical
        metadata is perfect.

    Detection here is deterministic -- probability 1, no threshold, no
    statistics.  That is deliberate: replay is a classical bookkeeping failure
    and does not need a physical test.
    """

    verifier: str
    window_seconds: float = 300.0
    last_counter: Dict[str, int] = field(default_factory=dict)
    seen_nonces: Dict[str, set] = field(default_factory=dict)
    consumed_slots: set = field(default_factory=set)
    accepted: int = 0
    rejected: int = 0

    def check(self, signer: str, counter: int, nonce: bytes, timestamp: float,
              now: float, slots: Optional[List[Tuple[int, int, int]]] = None
              ) -> FreshnessVerdict:
        v = FreshnessVerdict(ok=True)

        last = self.last_counter.get(signer)
        if last is not None and counter <= last:
            v.fail(
                f"counter {counter} does not exceed the last accepted counter {last}"
            )

        nonces = self.seen_nonces.setdefault(signer, set())
        if nonce in nonces:
            v.fail(f"nonce {nonce.hex()[:16]} has been presented before")

        if abs(now - timestamp) > self.window_seconds:
            v.fail(
                f"timestamp is {abs(now - timestamp):.1f}s from local clock, "
                f"outside the {self.window_seconds:.0f}s acceptance window"
            )

        if slots:
            replayed = [s for s in slots if s in self.consumed_slots]
            if replayed:
                v.fail(
                    f"{len(replayed)} of {len(slots)} public-key slots were already "
                    "consumed by an earlier verification"
                )
        return v

    def commit(self, signer: str, counter: int, nonce: bytes,
               slots: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """Record a verification attempt as having happened.

        Called whether or not the signature was accepted: a rejected attempt
        still burns the quantum slots it measured, and still means that nonce
        has been seen.
        """
        self.last_counter[signer] = max(counter, self.last_counter.get(signer, -1))
        self.seen_nonces.setdefault(signer, set()).add(nonce)
        if slots:
            self.consumed_slots.update(slots)

    def to_dict(self) -> Dict[str, object]:
        return {
            "verifier": self.verifier,
            "window_seconds": self.window_seconds,
            "signers_tracked": len(self.last_counter),
            "nonces_seen": sum(len(s) for s in self.seen_nonces.values()),
            "slots_consumed": len(self.consumed_slots),
            "accepted": self.accepted,
            "rejected": self.rejected,
        }
