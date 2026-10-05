# Threat Model

This document states what the scheme defends against, what it assumes, and where
it stops. It covers two distinct adversaries. The first is the one the protocol
is *about* — an attacker on the quantum and classical channels trying to forge or
repudiate a signature. The second is quieter and particular to this being a
simulation: the detection engine itself, which must not be allowed to cheat by
reading the simulator's ground truth. Both are in scope here. The security
*proofs* live in `docs/WHITEPAPER.md` §5–7; this document is the operational
statement of assumptions and boundaries.

---

## 1. What is being protected

A digital signature scheme makes three promises, and QDS makes the same three
without any computational assumption:

**Unforgeability.** No one but the signer can produce a message–signature pair
that an honest recipient accepts. Here "no one" includes an adversary with
unbounded computing power and a quantum computer, because the guarantee rests on
the no-cloning theorem and the geometry of the BB84 states, not on a problem
being hard to invert.

**Non-repudiation.** A signer who signed cannot later disavow it, and cannot sign
in a way that makes one recipient accept while another rejects. This is the
promise that distinguishes a signature from a shared secret, and it is the one
symmetrisation (Lemma 3 in the whitepaper) exists to secure.

**Transferability.** A recipient who accepts can forward the message to a second
recipient with justified confidence the second will also accept. This is why
there are two thresholds rather than one: the acceptance threshold `s_a` governs
whether I accept, and the stricter transfer threshold `s_v` governs whether I can
expect you to.

The ordering `eta < s_a < s_v < 1/4` is the whole security story in one line: the
honest noise floor sits below acceptance, acceptance below transfer, and transfer
below the `1/4` error rate that any forger is bound to incur.

---

## 2. The adversary

The scheme assumes a single worst-case adversary who may be an outsider, a
dishonest recipient, or a dishonest signer, and who controls both channels
between the parties.

**On the quantum channel the adversary may do anything quantum mechanics
permits.** Intercept every travelling qubit and resend a fabricated one;
entangle an ancilla probe with each qubit and measure it later, after the bases
are public; apply a weak collective channel tuned to stay under threshold; bias
the disturbance toward one basis to hide inside a pooled rate; block, delay, or
inject detector clicks. The attack suite (`attacks/`, documented as a specified
interface in `docs/ARCHITECTURE.md` §8) is built to realise exactly these, and
the eight snapshot scenarios in `docs/DETECTION.md` §7 are their measured
outcomes. What the adversary cannot do is copy an unknown quantum state or
distinguish non-orthogonal states better than their geometry allows — those are
not engineering gaps but theorems, and every bound in the whitepaper is a
consequence of them.

**On the classical channel the adversary may read, modify, replay, reorder, and
inject at will.** This sounds fatal and is not, because every classical message
that matters carries a one-time Wegman–Carter MAC (§4). The adversary may corrupt
a declaration, but cannot make a corrupted one authenticate.

**The adversary may be a participant.** A dishonest recipient who wants to forge
is modelled as holding one legitimate copy of the signer's quantum key — the
strongest honest-participant position — and Lemma 2 bounds his success at `3/4`
per check, i.e. an error rate of `1/4`, achieved by the pretty-good measurement
and optimal because the four BB84 states are geometrically uniform. A dishonest
signer who wants to repudiate is defeated by symmetrisation forcing the recipients'
views to agree.

---

## 3. Trust assumptions

The guarantees above are conditional. These are the conditions, and a deployment
that breaks one of them does not get the security the proofs describe — stated
plainly here because an unexamined assumption is the most common way a sound
protocol fails in the field.

*Authenticated classical channels.* Each pair of parties shares pre-distributed
secret key for the one-time MAC. The scheme authenticates messages; it does not
bootstrap the first authentication. Key distribution and renewal are out of scope
and assumed solved (e.g. by QKD or a trusted courier).

*Honest symmetrisation.* Non-repudiation depends on the signature qubits being
physically permuted among recipients so that no party knows which half another
holds. The protocol models this (`session.symmetrise_keys`), but whether it
happened honestly cannot be verified from the policy or the evidence — it is a
trust assumption, and `docs/WHITEPAPER.md` §10 lists it as such.

*Trusted devices.* This is not a device-independent scheme. The state preparation,
the measurement apparatus, and the random number generators are trusted to do
what they claim. A compromised RNG on the signer's side, for instance, shows up in
the `correction_uniformity` test as a candidate for `transcript_manipulation` —
but the test flags it as a *bug*, and the scheme does not defend against a signer
who is sabotaging his own equipment.

*Declared-before-run thresholds.* Every threshold is derived from a noise floor
declared before the run, never estimated from the run being judged. This is both a
security property and a trust assumption: the declared sheet must honestly describe
the hardware. A sheet that overstates the floor relaxes the thresholds; the
`degraded-link` scenario is precisely what it looks like when the channel is worse
than its sheet, and the engine catches it (`docs/DETECTION.md` §7).

---

## 4. The classical layer

Two mechanisms carry the classical-channel defence, and both are
information-theoretic to match the quantum layer — there would be little point
pairing an unconditionally secure signature with a computationally secure MAC.

**Authentication.** A one-time Wegman–Carter MAC over the field
`GF(2^64) = GF(2)[x]/(x^64 + x^4 + x^3 + x + 1)` (reduction constant `0x1B`). No
hash function, no block cipher: the security is the `epsilon`-almost-universal
property of polynomial evaluation, so a forged tag succeeds with probability
bounded by the field arithmetic alone, against any adversary. One-time means each
key authenticates one message; key reuse breaks it, and is the implementer's
responsibility to prevent.

**Anti-replay.** A declaration is fresh only if it passes four deterministic
predicates: a strictly increasing counter, an unseen nonce, a timestamp inside a
300-second window, and a slot set that has not already been consumed. These are
checked in the freshness gate before any qubit is measured, so a replayed or
stale declaration is rejected without spending quantum resources. (A source-level
note: the whitepaper §10 records a "three versus four predicates" wording
inconsistency between two files; the implementation checks four, and the
freshness gate in `verification.py` is authoritative.)

The verification order is deliberate and is part of the threat model: identity,
then MAC, then freshness, then the quantum check. A declaration that fails any
classical gate is refused *before* measurement unless the policy explicitly sets
`always_measure`, because an adversary should not be able to make a verifier burn
entangled pairs by sending garbage.

---

## 5. The detector's own threat model: the privilege boundary

Everything above is the protocol's threat model. This section is about a threat to
the *evaluation*. Because QDS here is simulated, the program that generates a run
knows everything: Eve's exact probe states, the true fidelity, the real QBER, the
CHSH value. A detection engine that peeked at any of these would score perfectly
and prove nothing — it would certify the simulator, not the scheme. So the
detector is treated as an adversary too, one whose "attack" is reading privileged
state, and it is contained structurally.

The containment is a single rule: **the detection package consumes exactly one
object, `Evidence`, assembled only from what a real verifier could observe.** The
simulator's ground truth is never passed in. This is not left to discipline; three
tests in `tests/test_detect.py::TestPrivilegeBoundary` enforce it, and they run in
the same suite as everything else.

*Source grep.* `test_no_privileged_symbols_in_source` reads every `.py` file in
the detection package and fails if any of eight forbidden identifiers appears
anywhere in the text:

```
DistributionOracle   .oracle                exact_qber
mean_chsh            mean_bell_fidelity      mean_concurrence
information_disturbance                      announcement_error_rate
```

These are the names under which the distribution layer records its privileged
view. If detection code so much as mentions one, the build goes red.

*AST guard on `.summary()`.* `DistributionResult.summary()` carries ground truth,
but the detector legitimately defines a `DetectionReport.summary()` of its own, so
a grep would be both too strict and too loose. `test_summary_oracle_is_never_read`
instead parses `engine.py` and fails on any call to `.summary()` whose receiver is
not `self` — the precise thing that would let the oracle in, and nothing else.

*Output scrub.* `test_evidence_from_session_carries_no_truth` runs the engine for
real and then walks the serialised report, failing if any privileged word
(`chsh`, `concurrence`, `fidelity`, `oracle`, `disturbance`, `eve_`, `true_`, …)
appears as a key, or if any of the eight forbidden identifiers appears anywhere in
the JSON. An interpretation string is allowed to say the words "costs no fidelity"
— explaining what a detector means is the point of that field — but a fidelity
*number* can never arrive, because a number has to come in under a key.

One deliberate consequence: the protocol layer is permitted an upward import of
`stats` — `verification.py` and `session.py` both do `from ..detect import stats`,
so the protocol can reuse the exact statistics. That is a dependency on the *math*,
not on the oracle, and it does not breach the boundary because `stats` holds no
privileged state. `docs/ARCHITECTURE.md` §5–6 documents this as the one allowed
exception.

---

## 6. Out of scope

Stated so that absence is not mistaken for an oversight:

Key distribution and the first authentication are assumed solved (§3). Side
channels — timing, power, electromagnetic emanation from real devices — are
outside a simulation that models the protocol rather than the hardware.
Denial of service is only partially addressed: the classical gates refuse to
spend quantum resources on unauthenticated declarations (§4), but an adversary who
simply cuts the fibre is detected, not prevented, and the scheme has nothing to
say about availability beyond reporting the loss. Device independence is not
claimed (§3). And the simulation itself has limits — the Qiskit cross-validation
backend exists but was never executed in this environment, snapshot key lengths
sit below the `epsilon = 1e-9` requirement for display reasons, and two
source-level inconsistencies are recorded openly; all of these are enumerated in
`docs/WHITEPAPER.md` §10 rather than hidden here.

The through-line of every item above is the same commitment the detector makes
when it reports a candidate cause instead of announcing an attack: state what is
assumed, draw the boundary where the evidence actually ends, and never let the
convenience of a simulation stand in for a guarantee the scheme has not earned.
