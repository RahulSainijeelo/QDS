# Teleportation-Based Quantum Digital Signatures with Statistical Threat Detection

**Status:** working preprint accompanying the reference implementation in this
repository. Every numerical claim below was produced by the code in `engine/`
and can be regenerated; see §9.

---

## Abstract

We describe and implement a quantum digital signature (QDS) scheme in which the
public key is a set of BB84 states delivered to each recipient by quantum
teleportation over pre-shared Bell pairs, and the message is signed one bit at a
time in the Lamport style. Security rests on a single physical fact rather than
on any computational assumption: a recipient holding one copy of the quantum
public key cannot determine which of four pairwise non-orthogonal states he
holds, and so cannot produce a signature that passes more than a bounded
fraction of the verifier's checks. We prove the two bounds that fix the scheme's
geometry — a forger with no copy of the public key passes an individual check
with probability `1/2` (Lemma 1), and a forger holding exactly one copy passes
with probability `3/4` and no better (Lemma 2) — and show that these force the
threshold ordering `eta < s_a < s_v < 1/4` on which unforgeability,
non-repudiation and transferability all depend simultaneously.

We then address a question the standard security proofs leave open. An
asymptotic bound tells an operator that a sufficiently strong attack must leave
a trace; it does not tell him whether *this* run was attacked. We build a
detection layer of eleven hypothesis tests, each with a null hypothesis declared
before the run and no fitted or learned parameters, and we enforce by
construction and by test that the detector may read only what a real verifier
could observe. On eight simulated scenarios the layer separates a healthy link
from intercept-resend, entanglement-breaking and basis-biased attacks, and
separates both from an innocently degraded link. It also *fails* to flag two
deliberately weak attacks that stay below the acceptance threshold, and we
report that failure with the same prominence as the successes, because it is the
honest characterisation of what a finite key buys.

---

## 1. Introduction

A digital signature must provide three things. It must be **unforgeable**: only
the holder of the private key can produce a signature that verifies. It must be
**non-repudiable**: having signed, the signer cannot later disown the message.
And it must be **transferable**: a recipient who accepts a signature can forward
it to a third party who will also accept it, without the third party having to
trust the forwarder.

Classical signatures obtain all three from computational assumptions — that
factoring is hard, that a discrete logarithm is hard, that a hash function is
collision-resistant. Those assumptions are reasonable today and are not
permanent. A signature is a long-lived object: a contract signed now may need to
remain attributable for decades, and an adversary who records the signature now
and breaks the assumption later breaks the signature retroactively. Quantum
digital signatures replace the computational assumption with a physical one, and
so are secure against an adversary with unbounded computational power.

The scheme implemented here follows the Gottesman–Chuang lineage: the private
key is classical, the public key is quantum, and verification is a projective
measurement. Two design choices distinguish this implementation.

The first is **delivery by teleportation**. Rather than transmitting the
public-key states down the channel directly, the parties consume a pre-shared
Bell pair per slot and teleport the state into the recipient's memory. The
classical transcript of a teleportation — the two Pauli-correction bits — is
uniformly random and independent of the teleported state, which we verify
numerically in §5.3. The practical consequence is that the entanglement
distribution phase is separable from and prior to the key-delivery phase, so the
link can be characterised, and the entanglement quality monitored by CHSH, using
resources that carry no key material.

The second is that the **detection layer is a first-class component with a
privilege boundary**. It is easy, and wrong, to evaluate a QDS simulator by
comparing its output against the simulator's own ground truth. If the code that
decides "this run was attacked" can read the exact channel fidelity or the
eavesdropper's probe state, the resulting detection rate is a restatement of the
input rather than a measurement. We therefore define a single data structure,
`Evidence`, that is the only object the detection engine may consume, populate
it exclusively from quantities a real verifier observes, and assert the boundary
with a test that greps the detection package for the forbidden symbols. That
test is part of the security argument, not a style check.

### 1.1 What this document claims and does not claim

This is a simulation and an analysis, not a deployment report. The quantum
channel, the memory, the detectors and the adversary are all models, and the
density matrices are exact rather than sampled. Consequently the numbers in §8
characterise *the scheme under these models*, and the honest reading of them is
comparative — how detection degrades as an attack weakens, which tests catch
which attack — rather than absolute. Nothing here has been run on hardware. §10
states the limitations in full, including one component that has never been
executed.

---

## 2. Notation and preliminaries

### 2.1 The four-state ensemble

Write the BB84 states as

```
|psi(0,0)> = |0>        |psi(0,1)> = |1>
|psi(1,0)> = |+>        |psi(1,1)> = |->
```

so that the first index `a` selects a basis — `Z` for `a = 0`, `X` for `a = 1` —
and the second index `c` selects an eigenvalue within it. The two bases are
mutually unbiased: `|<psi(0,c) | psi(1,c')>|^2 = 1/2` for every `c, c'`. The
ensemble `{|psi(a,c)>}` with uniform priors is **geometrically uniform**, meaning
the four states are related to one another by a group of unitaries under which
the ensemble is invariant. This is the structural property that makes Lemma 2
tight, and it is the reason the scheme uses four states rather than two.

Throughout, `eta` denotes the declared honest per-check error rate (the noise
floor from the hardware specification), `s_a` the acceptance threshold, `s_v`
the transfer threshold, and `N` the number of checks a verifier performs on one
signature.

### 2.2 Key structure

To sign a message of `m` bits, the signer draws, for every bit position
`j in {0, ..., m-1}` and for **both** possible values `b in {0, 1}` of that bit,
an independent block of `L` entries drawn uniformly at random:

```
K[j, b] = ( (a_1, c_1), (a_2, c_2), ..., (a_L, c_L) ),    a_i, c_i in {0,1}
```

The private key is the whole collection, `2mL` entries or `4mL` bits. The
quantum public key is the corresponding product state

```
|PK> = tensor over (j, b, i) of |psi(a_i^{j,b}, c_i^{j,b})>
```

one independent copy per recipient, delivered before any message exists. A slot
is addressed by the triple `(j, b, i)`, and two reserved values of `j` mark
slots that never carry signature material: `j = -1` for **check pairs**, whose
entanglement is measured and sacrificed to monitor the link, and `j = -2` for
**decoys**, whose key entries are revealed immediately so the recipient can test
the end-to-end error rate without spending signature capacity.

To sign bit value `b` at position `j`, the signer publishes `K[j, b]` and
retains `K[j, 1-b]` forever. The recipient measures his stored qubit for each
slot `(j, b, i)` in the basis `a_i` he has just been told, and demands the
outcome `c_i`. Honest and noiseless, every check passes with probability exactly
1; each mismatch is therefore a physical event, and the only question is how
many are too many.

That the *unrevealed* block is never disclosed is what prevents an adversary who
has seen one signature from signing a different message. It is also why the key
is strictly one-time: signing twice with the same key reveals both blocks at
some position, and unforgeability is gone.

**Reference geometry.** The snapshots in §8 use `m = 16` and `L = 24`, giving
`2mL = 768` signature slots, of which `mL = 384` are revealed by any one
message, plus 64 check pairs and 64 decoys. The implementation's own default is
`m = 16`, `L = 48`.

---

## 3. The protocol

The five phases run in a fixed order, and two of them are easy to omit and fatal
to omit.

**Phase 1 — Enrolment.** Each recipient learns which signer and which `key_id`
to expect, over a channel authenticated by a pre-shared one-time key reservoir.
The enrolment payload is domain-separated (`TQDS-ENROL-v1`) and MAC'd, and the
index of the key that authenticated it is recorded. This phase is what turns
"a signature from Alice" into a claim capable of being false: an impersonator
who cannot produce a valid enrolment tag never obtains a registry entry, and
every later signature he sends is refused on identity grounds before a single
qubit is measured.

**Phase 2 — Distribution.** For each recipient and each slot, the parties
establish a Bell pair `|Phi+>` and the signer teleports the slot's BB84 state
into the recipient's memory, broadcasting the two correction bits `(u, v)`. The
recipient applies `Z^u X^v` and stores the result. Check pairs are left
undisturbed as joint two-qubit states so that a CHSH correlator can be evaluated
across the two halves.

**Phase 3 — Symmetrisation.** The recipients privately permute their stored
signature qubits among themselves, so that each slot's copies end up held by a
uniformly random recipient. Only signature slots move; check pairs and decoys
are sacrificed publicly, so permuting them would gain nothing. §5.6 explains why
the non-repudiation bound is void without this step.

**Phase 4 — Calibration.** The signer reveals the decoy entries; each recipient
measures them and tests the observed rate against the *declared* noise floor.
This is a conformance test, not a threshold-setting step — see §6.

**Phase 5 — Signing and verification.** One message, one key, ever. The signer
emits a `SignatureDeclaration` carrying the message, the revealed key blocks,
the counter, a 16-byte nonce, a timestamp, and one Wegman–Carter MAC tag **per
recipient**, all produced by the signer. The verifier then applies four gates in
a fixed order: identity, authentication, freshness, and only then the quantum
measurement.

### 3.1 Why the gate order is itself a security property

The first three gates are deterministic: they reject with probability 1 and
carry no threshold. The fourth consumes the public key irreversibly. A verifier
who measured first could have his entire stored key burned by an adversary
sending junk declarations that were never going to be accepted — a denial-of-
service that costs the adversary nothing and the verifier everything. Hence
`verify_declaration` refuses to measure when the classical layer has already
rejected. The implementation exposes a flag (`always_measure`) that deliberately
weakens the verifier in exactly this way, so that the attack suite can isolate
the physics from the bookkeeping; no honest deployment sets it.

---

## 4. The classical layer

The classical channel must be authenticated, and the authentication must not
reintroduce a computational assumption — otherwise the scheme's headline claim
is lost at the last step. We therefore use a Wegman–Carter one-time MAC over
`GF(2^64) = GF(2)[x] / (x^64 + x^4 + x^3 + x + 1)`, with no hash function
anywhere:

```
tag = ( XOR over i of  M_i * a^(i+1) )  XOR  b
```

for message blocks `M_i` and a fresh key pair `(a, b)` drawn from the reservoir
and never reused. Substitution security is a counting argument over the field: a
forger's candidate difference is a polynomial of bounded degree in `a`, so at
most `deg` of the `2^64` possible values of `a` admit a collision, giving a
forgery probability of order `blocks / 2^64`. This is information-theoretic and
holds against unbounded computation.

Replay is handled deterministically rather than statistically, by four
independent predicates that must all hold: the counter must strictly exceed the
last accepted counter for that signer; the nonce must not have been presented
before; the timestamp must lie within a 300-second window; and none of the slots
the declaration addresses may already have been consumed by an earlier
verification. Each predicate rejects with probability 1 when violated, so none of
them consumes any of the statistical error budget. The last is the one that
cannot be evaded: slots are physically spent by measurement, and the ledger makes
that fact observable.

Note that the per-recipient MAC tags are all produced by the **signer**. A
forwarded declaration is byte-identical to what the signer emitted, and the
third party checks the signer's own tag rather than the forwarder's. This is
precisely what makes the signature transferable rather than merely
re-authenticated at each hop.

---

## 5. Security analysis

### 5.1 Lemma 1 — a forger with no copy of the public key

> **Lemma 1.** Let a forger who holds no copy of the public-key qubit for a
> given slot announce an arbitrary guess `(a', c')` for that slot's key entry.
> The verifier, holding `|psi(a,c)>` for the true entry `(a, c)` drawn uniformly
> at random, measures in basis `a'` and demands outcome `c'`. The check passes
> with probability exactly `1/2`, independently of the forger's strategy.

*Proof.* Condition on whether the announced basis matches the true one.

If `a' = a`, the verifier measures in the basis of which his state is an
eigenstate, so the outcome is deterministic and equals `c`. The check passes if
and only if `c' = c`. Since `c` is uniform on `{0, 1}` and independent of
everything the forger knows, this occurs with probability `1/2`.

If `a' != a`, the two bases are mutually unbiased, so the outcome is uniform on
`{0, 1}` regardless of `c`, and the check passes with probability `1/2`.

Both branches give `1/2`, so the total is `1/2` whatever distribution the forger
induces over `a'`. Note where uniformity of the key is used: if the basis bits
were biased, the forger could bias `a'` to match and do better than a coin
flip. ∎

The corresponding per-check *error* rate is `1 - 1/2 = 1/2`, exposed in the
implementation as `BLIND_FORGER_ERROR_RATE`.

### 5.2 Lemma 2 — a forger holding exactly one copy

> **Lemma 2.** Let a forger hold exactly one copy of the public-key qubit
> `|psi(a,c)>`, with `(a, c)` uniform over the four BB84 states, and let him
> announce a guess derived from any measurement he likes, followed by any
> computation he likes. The check passes with probability at most `3/4`, and the
> bound is achieved by the pretty-good measurement.

*Proof.* The optimal strategy is to perform a measurement, obtain an estimate of
which of the four states he holds, and announce it. For the geometrically
uniform ensemble `{|psi_i>}` with uniform priors, the pretty-good (square-root)
measurement is optimal, and its POVM elements are

```
M_i = |psi_i><psi_i| / 2 ,      i = 1..4,     sum_i M_i = I
```

Take the true state to be `|0>`, i.e. `(a,c) = (0,0)`; the geometric uniformity
of the ensemble makes the other three cases identical. The outcome probabilities
are

```
P(name |0>) = <0| M_1 |0> = 1/2
P(name |1>) = <0| M_2 |0> = 0
P(name |+>) = P(name |->) = 1/4
```

Now compose with the verifier's check. When the forger names the true state,
the check passes with probability 1. When he names a state in the *conjugate*
basis — total probability `1/4 + 1/4 = 1/2` — the verifier measures in the
wrong basis, the outcome is uniform, and the check passes with probability
`1/2`. When he names the same basis with the wrong bit, the check fails
outright; that happens with probability 0. Hence

```
P(pass) = 1/2 * 1  +  1/2 * 1/2  +  0  =  3/4
```

Optimality of the pretty-good measurement for this ensemble makes `3/4` an upper
bound over all measurement strategies, and no post-processing can improve on the
measurement, so `3/4` stands against an adversary of unbounded computational
power. ∎

We verified Lemma 2 numerically and independently of the implementation's own
tests: the POVM above sums to the identity to machine precision, and each of the
four true states yields `P(pass) = 0.750000`.

The corresponding per-check error rate is `1 - 3/4 = 1/4`, exposed as
`FORGER_ERROR_RATE`. **This — not `1/2` — is the number the thresholds must be
set against**, because a dishonest *recipient* is exactly a forger holding one
copy. It is the reason the honest error rate must sit below `1/4`.

A corollary worth stating, because the attack suite exercises it directly: `1/4`
is the *minimum* per-check error over all single-copy strategies, not a
representative one. A measurement tuned to maximise the forger's *information*
rather than to minimise his disturbance — the Breidbart measurement — is
deliberately suboptimal for passing the check, so its error rate sits strictly
*above* the `1/4` floor. The `breidbart` variant in `qds.attacks.forgery` shows
exactly that, measuring well clear of `single_copy`; the `3/4` of the lemma is
therefore the forger's best case, and setting the thresholds against it is
conservative against every other single-copy measurement.

### 5.3 The teleportation transcript leaks nothing

> **Proposition 1.** For every BB84 input state, the joint distribution of the
> two Bell-measurement outcome bits `(u, v)` broadcast during teleportation is
> uniform on `{0,1}^2`, and is therefore independent of the teleported state.

Numerically, `bell_outcome_distribution` returns `1/4` for each of the four
outcomes for each of the four inputs, to within `1e-16`; and
`teleportation_leakage`, which computes the maximum total-variation distance
between the outcome distributions over all six pairs of distinct input states,
returns `5.55e-17` — zero to machine precision.

The consequence is that the classical broadcast, although public, reveals nothing
about the private key, so an adversary gains nothing by recording it. The
correction the recipient applies is

| `(u, v)` | correction |
|---|---|
| `(0, 0)` | `I` (none) |
| `(1, 0)` | `Z` |
| `(0, 1)` | `X` |
| `(1, 1)` | `Z X` |

applied as a single composed gate so that no intermediate state is ever
materialised.

### 5.4 The threshold ordering

Everything above combines into one chain of inequalities that must hold for the
scheme to mean anything:

```
eta  <  s_a  <  s_v  <  1/4
```

Each gap purchases exactly one property, and each is priced by a relative-entropy
(Chernoff–Hoeffding) exponent rather than by a normal approximation:

The gap from `eta` to `s_a` is **room for honest noise**. Too tight and
legitimate messages are rejected; the false-alarm probability is bounded by
`exp(-N D(s_a || eta))`.

The gap from `s_a` to `s_v` is **room against repudiation** (§5.6), with failure
probability bounded by `exp(-N min(D(s_v || mu), D(s_a || mu)))` where
`mu = (s_a + s_v)/2`.

The gap from `s_v` to `1/4` is **room against forgery**. The endpoint is not a
design choice but the optimal single-copy error rate of Lemma 2, so the bound
`exp(-N D(s_v || 1/4))` holds against an adversary of unbounded power.

If `eta >= 1/4` the hardware is simply too noisy for the scheme to be secure at
*any* threshold. The implementation reports this rather than silently returning
a threshold anyway: `VerificationPolicy.soundness_problems()` enumerates the
violated inequalities in plain language, and the detection layer's
`ladder_problems()` does the same for its own threshold ladder. Returning an
unusable ladder *and reporting it* is deliberate; substituting a workable
threshold for an unworkable one is how a framework ends up certifying a link it
should have refused.

### 5.5 Threshold placement

Two placement rules are implemented. The `thirds` rule splits the usable gap
`[eta, 1/4)` evenly, which is symmetric in threshold space and easy to draw. The
`balanced` rule instead maximises the *worst* of the four security exponents,
which is the quantity that actually determines how long the key must be.

The two are not equivalent, because the failure modes do not convert rate into
exponent at the same rate: near `eta` the relative entropy climbs steeply, near
`1/4` it does not, and the repudiation exponent depends only on the width of the
middle gap. At the lab-grade noise floor `eta = 0.0123972`:

| rule | `s_a` | `s_v` |
|---|---|---|
| `thirds` | 0.0833333 | 0.1666667 |
| `balanced` | 0.0396221 | 0.1687154 |

The `balanced` solution pulls `s_a` down hard and leaves `s_v` almost unchanged,
and it is the default. The search is a coarse grid followed by local refinement;
the objective is smooth and unimodal inside the feasible triangle, so it
converges to the optimum rather than merely near it, and it costs microseconds.

### 5.6 Non-repudiation requires a physical step, not a statistical one

The repudiation bound of §5.4 compares one recipient's mismatch count with
another's and argues that both are samples from the same underlying rate. Read
literally, that premise is false, and a dishonest *signer* can violate it for
free: send Bob clean states and Charlie dirty ones, and the message Bob accepts
is one Charlie will reject. That is repudiation with probability near 1, no
statistics required.

> **Lemma 3 (symmetrisation).** If, before any message exists, the recipients
> privately permute their stored signature qubits so that each slot's copies are
> held by a uniformly random recipient, and the permutation is unknown to the
> signer, then the signer's choice of whom to cheat is independent of who will
> check, and any asymmetry she introduces is redistributed evenly across
> recipients.

The countermeasure is therefore physical rather than statistical. Three details
matter. Only signature slots move, since check pairs and decoys are sacrificed
publicly. A qubit carries its `arrived`, `dark` and `correction` flags with it,
but the per-recipient detector log does **not** move, because it records what
happened at that recipient's own detector and remains true afterwards. And the
permutation must be secret from the signer but need not be secret from the
recipients.

One honest consequence deserves emphasis: `soundness_problems()` has no way to
observe whether symmetrisation happened. A deployment that skips it still gets a
policy object that reports itself sound, and still has a valid unforgeability
claim — but it has **no non-repudiation claim at all**. The implementation
performs the step by default and records `performed: true` with the number of
slots permuted, so the omission is at least visible in the session record.

### 5.7 Finite-size cost

The key length follows from requiring every failure bound to fall below a target
`epsilon`. Because all four bounds are exponential in `N`, the required `N` grows
only logarithmically in `1/epsilon`, which is why a one-in-a-billion target is
affordable at all. At `epsilon = 1e-9`:

| noise floor | `eta` | `s_a` | `s_v` | checks required |
|---|---|---|---|---|
| ideal | 0.0 | 0.0011584 | 0.1480548 | 669 |
| lab-grade | 0.0123972 | 0.0396221 | 0.1687154 | 1080 |
| field-grade | 0.0469849 | 0.0814832 | 0.1877680 | 1885 |

These are checks, i.e. `N = mL`, so the lab-grade figure corresponds to
`m = 16, L = 68` or any other factorisation. It is worth being explicit that the
reference geometry of §2.2 does **not** reach it: `m = 16, L = 24` gives
`N = 384`, and the implementation's own default `m = 16, L = 48` gives `N = 768`.
At `N = 768` and the lab-grade floor the four exponents are

| failure mode | bits (`-log2 p`) |
|---|---|
| honest abort | 21.26 |
| forgery, one copy | 21.26 |
| forgery, blind | 265.15 |
| repudiation | 21.26 |

so roughly `2^-21`, or one in two million, rather than one in a billion. The
three tight bounds coincide to four significant figures, which is the signature
of the `balanced` rule working as intended: it equalises the worst exponents.
The blind-forgery bound is enormously slacker because `1/2` is far from `s_v`,
which is the quantitative content of Lemma 1 being the easy case.

---

## 6. Thresholds must come from declared inputs only

An earlier version of this design estimated the acceptance threshold from the
run's own decoy sample. That is wrong twice over, and the correction is worth
stating plainly because the failure is not obvious.

It fails **operationally**: a threshold derived from data the adversary can
influence is a threshold the adversary sets. Thresholds belong to the protocol,
not to the run.

It fails **formally**: extrapolating a rate measured on `n` decoys to the
unmeasured signature slots costs a Hoeffding–Serfling margin of at least
`sqrt(ln(2/epsilon) / 2n)`. At `n = 96` and `epsilon = 1e-6` that is `0.275` —
wider than the entire interval `[0, 1/4]` in which the threshold must live. The
estimate was not merely imprecise, it was vacuous, and adding it to the threshold
collapsed every security exponent to zero.

So the decoys now answer the question they can actually answer: *is this channel
performing at or below its certified noise floor?* That is a one-sided exact
binomial conformance test, and a failure is a detection event in its own right —
it does not say whether the cause is a degraded fibre or an eavesdropper, only
that continuing to sign on this link is not covered by the security analysis.
The Hoeffding–Serfling margin is still computed and reported, as a *pre-signing
assurance* figure describing how well a recipient knows his key before a message
exists, but it gates nothing. At the decision point there is no unsampled
population to extrapolate to, because the signature checks are measured directly.

The same discipline governs the detection layer: every threshold is a closed form
in the declared noise floor, the number of checks, and the two forgery bounds of
§5. Nothing in the layer reads the session it is about to judge in order to
decide what "normal" means.

---

## 7. Threat detection as hypothesis testing

### 7.1 Why not a learned detector

Nothing in the detection layer is trained, fitted, or weighted from data. This
is a deliberate restriction and it is what makes the output worth anything. A
learned detector can tell you that a run looks unlike the runs it was shown. A
test against the `1/4` single-copy bound of Lemma 2 tells you that an adversary
who learned enough to forge **must** have left a trace at least this large,
whatever hardware he owns and however much computation he has. The second
statement survives an adversary the first has never seen.

### 7.2 The privilege boundary

The simulator knows things no verifier could: the eavesdropper's actual probe
states, the exact fidelity of every delivered qubit, the true QBER, the CHSH
value, the concurrence. A single dataclass, `Evidence`, is the only object the
detection engine consumes, and it is populated exclusively from observable
quantities — verification reports, calibration results, detector logs,
correction histories — together with figures **declared in advance** and
annotated in the source as *"declared, fixed before the run — never estimated
from this session"*.

The boundary is enforced three ways: structurally, since the engine's entry
points accept only `Evidence`; by review, since the declared fields are marked;
and by an automated test that scans the entire `qds.detect` package for a
forbidden symbol list — `DistributionOracle`, `.oracle`, `exact_qber`,
`mean_chsh`, `mean_bell_fidelity`, `mean_concurrence`,
`information_disturbance`, `announcement_error_rate`. If any appears, the test
fails. That test is part of the security argument.

### 7.3 The eleven tests

Each test has one null hypothesis, stated in advance, and is reported with its
statistic, its exact p-value, its per-test significance level and a plain-language
interpretation. In order:

| # | test | null hypothesis |
|---|---|---|
| 1 | `pooled_rate_vs_spec` | the pooled error rate is at or below the declared floor |
| 2 | `pooled_rate_vs_transfer` | the pooled rate is below the transfer threshold |
| 3 | `basis_consistency` | the Z-basis and X-basis rates are equal |
| 4 | `position_homogeneity` | all message-bit positions share one rate |
| 5 | `level_homogeneity` | accept-level and transfer-level rates agree |
| 6 | `yield_vs_declared` | detector yield matches the declared loss budget |
| 7 | `dark_excess` | the dark-count fraction is at or below declared |
| 8 | `correction_uniformity` | the `(u,v)` corrections are uniform on four outcomes |
| 9 | `decoy_vs_signature` | decoy and signature slots share one rate |
| 10 | `recipient_agreement` | all recipients share one rate |
| 11 | `sequential_onset` | the rate is constant in time (no mid-run onset) |

Five of these — 6, 7, 8, 9, 10 — are **premise tests**. They do not look for an
attack directly; they check the assumptions the other six rely on. A failure
among them means the evidence bundle itself is suspect, which is why the engine
treats them differently when forming a verdict.

Test 3 exists because a basis-asymmetric attack can sit under a pooled threshold
while being obvious in one basis; pooling halves the apparent damage. Test 8
matters because Proposition 1 makes uniformity of the corrections a *prediction*
of the protocol, so a departure is evidence that something is interfering with
the Bell measurement itself. Test 11 uses a CUSUM/SPRT construction and catches
an adversary who switches on partway through a run, which every rate test
averages away.

### 7.4 Multiplicity

Eleven simultaneous tests at `alpha = 0.01` each would produce a family-wise
false-alarm rate far above 1%. We apply a Bonferroni correction, giving
`alpha_per_test = 0.01 / 11 = 9.0909e-4`.

Bonferroni rather than Šidák, and this is not conservatism for its own sake. The
tests are computed from the same slot measurements — the pooled rate, the
per-basis rates and the per-position rates are three views of one list of
outcomes — so they are strongly dependent. Šidák is exact only under
independence and is not licensed here; Bonferroni follows from the union bound
and assumes nothing at all. At eleven tests the two differ by under half a
percent of the family alpha, so the guarantee is nearly free. Fisher's combined
p-value is also reported, but it is never decisive, precisely because the
independence it assumes does not hold.

### 7.5 Verdicts and attribution

The engine returns one of three verdicts. Crucially, **protocol rejection is
checked first**: if a recipient's measured rate reached the acceptance threshold,
the protocol refused the signature on its own terms, and that is reported
separately from the statistical tests because it is not one. The threshold was
fixed in advance; crossing it is a comparison, not an inference.

This matters because the two can disagree. The rate can cross the threshold
while every Bonferroni-corrected test still passes — the tests are tuned to
control false alarms across eleven simultaneous nulls, which necessarily costs
sensitivity. Reporting both, separately, is the only honest presentation.

A twelve-entry attribution table then maps flagged-test patterns to candidate
causes, in priority order, with a rationale. Its output is explicitly a
*candidate* cause: the engine reports departures from specification, and never
announces that an attack has occurred.

### 7.6 Statistics without scipy

The statistical layer is implemented from scratch — log-gamma binomial tails,
Lentz continued fractions for the incomplete beta and gamma functions,
Clopper–Pearson exact intervals, Chernoff–KL tail bounds,
Hoeffding–Serfling sampling margins, Fisher's exact test, chi-square
homogeneity, SPRT and CUSUM. The motive is not asceticism: exact methods are
required because a good channel produces single-digit mismatch counts, where a
normal approximation is simply wrong, and wrong in the dangerous direction. The
normal tail understates the binomial tail on the right, so a verifier using it
would flag more often than its stated `alpha`. Critical values are therefore
found by exact scanning, which at `N` of a few hundred costs nothing.

The choice of KL over Hoeffding bounds is quantitative. Hoeffding's
`exp(-2 N t^2)` knows only that the variable is bounded in `[0,1]`; the KL form
knows the variance shrinks as the rate approaches 0 or 1. At a forger rate of
`1/4` and a threshold of `0.1`, `D(0.1 || 0.25) = 0.0817` nats against a
Hoeffding exponent of `2 * 0.15^2 = 0.045` — nearly a factor of two in the
exponent, which over a few hundred checks is tens of orders of magnitude in the
bound. Hoeffding is retained only as a fallback valid when the per-check
outcomes are independent but not identically distributed.

---

## 8. Numerical results

Eight scenarios, all judged against the **lab-grade specification sheet**
declared once before any run (`eta = 0.0123972`, `s_a = 0.0396221`,
`s_v = 0.1687154`), at `m = 16, L = 24`. Each is a full protocol run followed by
a full detection pass, with a fixed seed. The `expectation` column is what the
scenario was written to demonstrate; `verdict` is what the engine actually said.

| scenario | expectation | verdict | pooled rate | flagged |
|---|---|---|---|---|
| `honest-lab` | clean | clean | 0.0186 | 0/11 |
| `intercept-resend` | compromised | compromised | 0.2375 | 4/11 |
| `intercept-resend-partial` | compromised | compromised | 0.1488 | 3/11 |
| `basis-biased` | suspicious | compromised | 0.0873 | 3/11 |
| `collective-depolarizing` | suspicious | **clean** | 0.0290 | 0/11 |
| `coherent-probe` | suspicious | **clean** | 0.0230 | 0/11 |
| `entanglement-breaking` | compromised | compromised | 0.3203 | 4/11 |
| `degraded-link` | detection, no adversary | compromised | 0.0199 | 1/11 |

### 8.1 What went right

The honest link produced 11 mismatches in 590 checks and flagged nothing,
confirming that the Bonferroni-corrected family does not cry wolf on a clean
run. Full intercept-resend, at a pooled rate of 0.2375, sits essentially at the
`1/4` single-copy bound — which is what Lemma 2 predicts, since measuring and
resending *is* the single-copy strategy — and flagged tests 1, 2, 3 and 11.
Halving the attack to every other round dropped the rate to 0.1488 and cost the
transfer-threshold test, but rate-vs-spec, basis consistency and sequential
onset all still fired. The entanglement-breaking channel at 0.3203 exceeds even
the blind-forger rate in one basis and is trivially visible.

The basis-biased probe shows the per-basis test earning its place. An asymmetric
Pauli-X probe (`p_X = 0.12`, `p_Z = 0`) corrupts the Z-basis checks almost
exclusively, driving the Z rate to 0.1614 while the X rate stayed at 0.0069, for a
pooled rate of 0.0873. That pooled rate already crosses
the acceptance threshold `s_a = 0.0396`, so the signature is rejected on its rate
alone and the engine returns `compromised` where the probe's author expected only
`suspicious` — the single-number test is not blind here. What the per-basis test
adds is the *shape*: `basis_consistency` fires at p ≈ 1.6e-44 and localises the
excess to one basis, which is why the attribution carries `elevated_channel_error`
beside the rate rejection. The per-basis test becomes the *sole* catch only at a
milder bias, one that keeps the pooled rate under `s_a`.

The degraded link is the case the design cares most about. There is no
adversary; the fibre is dimmer and the detectors noisier than the sheet they were
sold against. The pooled error rate is 0.0199 — comfortably below `s_a` — and
exactly one test fired: `yield_vs_declared`, the premise test for the loss
budget. The attribution came back `channel_blocking_or_blinding`. This is a real
detection with an innocent cause, and the layer located it in the yield rather
than in the error rate, which is the distinction the design exists to preserve.

### 8.2 What went wrong, and why we are reporting it

Two scenarios **missed**. A collective depolarising probe at `p = 0.05` produced
a pooled rate of 0.0290, and a coherent probe with delayed measurement at
`theta = 0.3` produced 0.0230. Both were written expecting a `suspicious`
verdict. Both came back **clean, with zero of eleven tests flagged.**

This is not a bug, and it should not be smoothed over. Both rates sit below
`s_a = 0.0396`, so the protocol itself accepts the signature — correctly, by its
own declared terms. And both sit close enough to the declared floor of 0.0124
that, at `N` of roughly 590 checks and a per-test alpha of `9.09e-4`, the exact
binomial test cannot reject the null. The detector is behaving exactly as
specified; the specification simply does not have the sensitivity, at this key
length, to see an attack this weak.

Three things follow. First, the honest characterisation of the layer is that it
reliably catches attacks that push the error rate past the acceptance threshold,
and does not reliably catch attacks that stay below it. Second, that boundary is
a function of `N`: the same attacks at the `N = 1080` required by §5.7 would face
a substantially tighter test, and quantifying that curve is the obvious next
experiment. Third, and most important, a weak attack that stays below `s_a` is
one that also gains correspondingly little information — the whole force of
Lemma 2 is that information gain and disturbance are linked — so "undetected"
here does not mean "successful forgery". It means the adversary paid for
invisibility in the currency the protocol charges.

### 8.3 On the word "compromised"

The `degraded-link` scenario returns the verdict `compromised` with no
adversary present anywhere in the simulation. The vocabulary is doing something
specific and it is worth stating: `compromised` means *this link is not operating
within the conditions its security analysis assumed*, not *an eavesdropper was
detected*. That is the only claim the evidence supports, and the attribution
table is careful to offer candidate causes rather than conclusions. A reader who
reads `compromised` as "attacked" will misread the degraded-link row.

---

## 9. Reproduction

Every number in this document comes from the code in this repository. With
`numpy` available:

```bash
# the eight scenarios of §8, written to web/public/data/
python3 web/scripts/generate_snapshots.py

# the threshold tables of §5.5 and §5.7
cd engine && python3 -c "
from qds.channel import IDEAL, LAB_GRADE, FIELD_GRADE
from qds.protocol.verification import VerificationPolicy
for nm, name in ((IDEAL,'ideal'), (LAB_GRADE,'lab'), (FIELD_GRADE,'field')):
    q = nm.predicted_qber(); p = VerificationPolicy.balanced(q)
    print(name, q, p.s_a, p.s_v, p.required_checks(1e-9))"

# Proposition 1
cd engine && python3 -c "
from qds.protocol.distribution import teleportation_leakage
print(teleportation_leakage())"

# the test suite  (standard-library unittest, not pytest)
cd engine && python3 -m unittest discover -s tests -p 'test_*.py'
```

The two placement rules, the noise presets and the scenario definitions are the
only knobs; everything else is derived. `docs/ARCHITECTURE.md` describes the
module layout, `docs/DETECTION.md` documents the eleven tests in full, and
`docs/THREAT_MODEL.md` states the adversary model and the trust assumptions.

---

## 10. Limitations

**Simulation only.** Nothing here has been run on quantum hardware. The channel,
memory, detectors and adversary are models. The density-matrix evolution is exact
rather than sampled, which removes one class of error and introduces an obvious
ceiling on the size of system that can be simulated.

**The noise model is analytic where it matters.** The predicted QBER is composed
from independent flip sources by odd-parity composition, with gate-error
insertions counted separately per basis (four reaching a Z-basis check, six
reaching an X-basis check, derived by propagating each Pauli error through the
teleportation circuit). This is more careful than a single lumped rate — lumping
overstates the error by about 8% at `p_g = 0.04` — but it is still a model, and
its agreement with the simulated outcome is a consistency check, not a
validation against hardware.

**Key lengths in the snapshots are below the `epsilon = 1e-9` requirement.** As
§5.7 sets out, `m = 16, L = 24` yields `N = 384` and the default `L = 48` yields
`N = 768`, against the 1080 checks lab-grade hardware needs for a
one-in-a-billion target. The snapshots are sized for interactive regeneration,
not for a security claim. Reported exponents are always computed at the actual
`N`, so no figure in this document overstates the security of the run it
describes.

**Detection sensitivity is bounded, and two scenarios show it.** See §8.2.

**Symmetrisation cannot be verified from the policy object.** See §5.6. A
deployment that skips it retains unforgeability and loses non-repudiation, and
no soundness check will say so.

**The Qiskit backend has never been executed.** The engine supports pluggable
backends and ships a `numpy` reference implementation plus a Qiskit
`quantum_info` adapter intended for cross-validation at a tolerance of `1e-12`.
The development environment had no network access, so Qiskit could not be
installed, and the adapter has therefore never run. It should be treated as
untested code. The cross-validation claim is a claim about the intended design,
not a result. Related: the adapter must translate qubit indices, since this
codebase orders qubits big-endian (qubit 0 most significant, matching
`np.kron`) and Qiskit does the opposite.

**Minor documentation drift in the source.** Two inconsistencies were found while
preparing this document and are recorded here rather than silently corrected: the
MAC module states the substitution bound as `(blocks+1)/2^64` in two places and
`blocks/2^64` in a third, and the freshness ledger's docstring describes "three
independent bindings" while the code implements four predicates (the timestamp
window being the fourth). Neither affects a security claim; both should be
reconciled.

---

## 11. Related work

The scheme is in the Gottesman–Chuang tradition of quantum digital signatures
with quantum public keys, and inherits the Lamport one-bit-at-a-time structure
from classical one-time signatures. The four-state ensemble and the mutually
unbiased bases are those of BB84. The single-copy discrimination bound of Lemma 2
is the pretty-good-measurement result for geometrically uniform ensembles. The
classical authentication layer is Wegman–Carter. The statistical machinery is
standard — Clopper–Pearson exact intervals, Chernoff–Hoeffding relative-entropy
tail bounds, Hoeffding–Serfling sampling without replacement, Wald's SPRT,
Page's CUSUM, Bonferroni and Šidák multiplicity corrections, Fisher's exact test
— and is implemented here from primitives only to avoid a dependency, not
because anything about it is novel.

What we believe is less common is the combination: teleportation-based delivery
with an explicitly separated entanglement-monitoring phase; thresholds
constrained to declared inputs only, with the decoy sample demoted from
threshold-setting to conformance testing; and a detection layer with a
machine-enforced privilege boundary against the simulator's own ground truth.

---

*Companion documents:* `docs/ARCHITECTURE.md` (module layout and data flow),
`docs/DETECTION.md` (the eleven tests in detail), `docs/THREAT_MODEL.md`
(adversary model and trust assumptions).
