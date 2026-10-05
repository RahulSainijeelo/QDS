# Detection

The detection layer answers one question about a completed protocol run: *is
this link behaving the way its specification says it should, or not?* It answers
with eleven hypothesis tests, a verdict, and a prioritised list of candidate
causes. Nothing in it is trained, fitted, or weighted. This document is the
reference for all three outputs. The science behind the thresholds is in
`docs/WHITEPAPER.md` §5–7; the module layout is in `docs/ARCHITECTURE.md`.

---

## 1. The principle

A learned detector tells you that a run looks unlike the runs it was shown. A
test against the `1/4` single-copy bound tells you that an adversary who learned
enough to forge *must* have left a trace at least this large — whatever hardware
he owns, however much he computes. The second statement survives an adversary the
first has never seen, and it is the only kind of statement this layer makes.

That is why every threshold is a closed form in quantities fixed before the run:
the declared noise floor, the number of checks, and the two forgery bounds that
come out of quantum mechanics rather than out of a measurement. And it is why the
attribution step is a fixed lookup table, not a classifier: there is no labelled
corpus of real quantum attacks to learn from, and a classifier trained on
simulated ones would certify the simulator, not the scheme.

---

## 2. What the engine is allowed to see

The engine consumes exactly one object, `Evidence`, and it is built only from
things a real verifier could observe plus figures declared before the run. The
simulator's ground truth — Eve's probe states, the exact fidelity, the true
QBER, CHSH, concurrence — is structurally unavailable and is kept so by a test
that greps the whole `qds.detect` package for a forbidden symbol list. See
`docs/ARCHITECTURE.md` §5 for the enforcement mechanism. If you add a test, it
may read `Evidence` and nothing else.

A corollary governs the declared figures. When the caller passes a spec sheet
(`analyse_session(session, declared_noise=SPEC)`), the evidence records
`declaration_source="spec_sheet"` and the yield and dark-count tests are live.
When no sheet is passed, those figures would have to be read off the very channel
being judged — a comparison of a quantity with itself — so the two tests mark
themselves **absent** rather than running. An absent test is not a passing test,
and the report distinguishes them.

---

## 3. The eleven tests

Each test has one null hypothesis declared in advance, an exact statistic, a
p-value, a per-test significance level (`alpha_per_test`, the Bonferroni-corrected
family alpha), and a plain-language interpretation carried in the report. A test
may also be **inapplicable** — too few checks, or a required declared figure
absent — in which case it neither passes nor flags and is excluded from the
applicable count.

### Rate tests

**1. `pooled_rate_vs_spec` — pooled error rate vs declared noise floor.** Null:
the pooled error rate is at or below the declared floor. An exact one-sided
binomial test. *A rate above the floor by more than sampling noise explains is
what an eavesdropper on the quantum channel looks like — and also what a degraded
link looks like. This test distinguishes neither from the other, only both from a
healthy link.* This is the workhorse, and it is first for a reason.

**2. `pooled_rate_vs_transfer` — pooled error rate vs transfer threshold.** Null:
the pooled rate is below the transfer threshold `s_v`. *Crossing it means that
even if a recipient accepts, he cannot expect the next party to agree, so the
non-repudiation guarantee no longer holds for this key.*

**3. `basis_consistency` — per-basis error rates vs their declared values.** Null:
each of the Z-basis and X-basis rates is at or below the value the spec predicts
*for that basis* (the model predicts them separately — four gate insertions reach
a Z check, six reach an X check). *A probe that disturbs one basis more than the
other produces exactly this pattern and can hide entirely inside a pooled rate.*
This is the test that exists for exactly that regime. In the shipped basis-biased scenario the bias is strong enough that the pooled rate crosses the floor as well, so here the per-basis split corroborates and localises the excess rather than being the only test to see it.

**4. `position_homogeneity` — error rate across message-bit positions.** Null: all
message-bit positions share one rate. A chi-square homogeneity test. *Honest noise
has no reason to prefer one bit of the message over another, so concentration
points at tampering aimed at specific bits rather than at the channel as a whole.*

**5. `level_homogeneity` — error rate across verification levels.** Null:
accept-level and transfer-level rates agree. *The same key measured under two
policies should give one rate; a split suggests the record one party sees is not
the record another sees.*

### Premise tests

These five do not look for an attack directly. They check the assumptions the
other six rely on, so that a failure among them flags the evidence bundle itself
as suspect.

**6. `yield_vs_declared` — detector yield vs declared loss budget.** Null: the
fraction of slots that registered a detection matches the declared loss budget.
*Fewer detections than expected is consistent with an adversary who measures and
then blocks, which leaves every error rate untouched; more than expected is
consistent with injected or faked clicks.* This is the only test that fired on
the degraded-link scenario.

**7. `dark_excess` — dark-count fraction vs declared detector noise.** Null: the
dark-count fraction is at or below declared. *A dark count contributes a uniformly
random bit, so an excess also inflates the measured error rate; the two should be
read together.*

**8. `correction_uniformity` — uniformity of announced Pauli corrections.** Null:
the `(u,v)` corrections are uniform over the four outcomes. A chi-square
goodness-of-fit test. *Uniformity is what makes the classical transcript
information-free (Proposition 1 in the whitepaper), so a bias means the Bell
measurement or the announcement channel is being interfered with — and unlike the
error-rate tests, this one fires even when the interference costs no fidelity.*

**9. `decoy_vs_signature` — decoy slots vs signature slots.** Null: decoy and
signature slots share one rate. A Fisher exact two-sided test. *Both crossed the
channel under identical conditions and which is which is not revealed until
afterwards, so disagreement is what selective attack on the slots that carry
meaning looks like.*

**10. `recipient_agreement` — agreement between recipients.** Null: all recipients
share one rate. *They hold symmetrised halves of one key and check one
declaration, so a genuine split means either one link is under attack or the
signer distributed unequal keys — the latter being the setup for repudiation.*

### Sequential test

**11. `sequential_onset` — SPRT and CUSUM.** Null: the rate is constant in time,
with no mid-run onset. Wald's sequential probability ratio test and Page's CUSUM,
run over the checks in order. *A sequential monitor that would have stopped the
run reports the stopping index — how early the evidence became decisive — which
is the operationally useful number, because every check after it spent an
entangled pair on a link already known to be bad.* This catches an adversary who
switches on partway through, which every aggregate rate test averages away.

---

## 4. Multiplicity

Eleven simultaneous tests at `alpha = 0.01` each would blow the family-wise
false-alarm rate far past 1%. The family alpha is held at 0.01 and divided by
Bonferroni: `alpha_per_test = 0.01 / 11 = 9.0909e-4`.

Bonferroni, not Šidák, because the tests are strongly dependent — the pooled
rate, the per-basis rates and the per-position rates are three views of one list
of outcomes. Šidák is exact only under independence; Bonferroni follows from the
union bound and assumes nothing. At eleven tests they differ by under half a
percent of the family alpha, so the guarantee is nearly free. Fisher's combined
p-value is reported but **never decisive**, precisely because the independence it
assumes does not hold here.

The false-alarm control is not asserted, it is tested: `TestFalseAlarmCalibration`
generates runs where the null is true by construction — Bernoulli draws at exactly
the declared floor, no quantum simulation — and checks that the family-wise alarm
rate comes out at or under the advertised alpha.

---

## 5. The verdict

The engine returns one of three verdicts: `clean`, `suspicious`, or
`compromised`. The logic that produces them checks **protocol rejection first**,
and this ordering is deliberate.

If a recipient's measured rate reached the acceptance threshold, the protocol
refused the signature on its own terms. That is not a statistical inference — the
threshold was fixed in advance, and crossing it is a comparison — so it is
reported in its own fields (`protocol_rejects`, `rejections`) and drives the
verdict before any test is consulted. The same holds for a classical rejection:
`authentication_failed` is set when the MAC, the registry entry, or the ledger
refused the declaration, and that is checked ahead of everything quantum.

Only if the protocol itself accepted does the verdict fall to the tests: flagged
tests push toward `suspicious` or `compromised` according to which fired and how
many, with the premise tests weighted differently from the direct ones.

The crucial honesty here is that **the two can disagree**. The rate can cross the
acceptance threshold while every Bonferroni-corrected test still passes — the
tests are tuned to control false alarms across eleven simultaneous nulls, which
necessarily costs sensitivity. A verdict of `compromised` driven by
`protocol_rejects` with zero flagged tests is not a contradiction; it is the
protocol being stricter than the multiplicity-corrected test family, which is
exactly the regime the thresholds were designed for.

One vocabulary note that the whitepaper §8.3 also makes: `compromised` means
*this link is not operating within the conditions its security analysis assumed*,
not *an eavesdropper was detected*. The degraded-link scenario returns
`compromised` with no adversary anywhere in the simulation.

---

## 6. Attribution

When something fires, a twelve-entry lookup table maps the pattern of flagged
tests (plus a few evidence flags) to candidate causes, in priority order, most
specific first. It returns **every** match, not just the first, because a real
attack trips several detectors and the secondary matches are what let an operator
tell a story about the primary one. The first element is the primary attribution.

Each entry is a fixed boolean condition over which detectors fired — nothing
learned — and each carries a rationale that names the innocent explanation
alongside the adversarial one. In priority order:

| # | label | fires when | the honest caveat it carries |
|---|---|---|---|
| 1 | `authentication_failure` | the classical layer rejected | the one finding with no innocent explanation short of operator error |
| 2 | `signature_rejected_on_rate` | a recipient's rate hit `s_a` | a comparison, not an inference; can fire with every test still passing |
| 3 | `repudiation_or_targeted_link_attack` | `recipient_agreement` flagged | a single unlucky link outage also produces it |
| 4 | `transcript_manipulation` | `correction_uniformity` flagged | a defective RNG on the signer's own side produces it too — a bug, not an attack |
| 5 | `channel_blocking_or_blinding` | `yield_vs_declared` flagged, rate/basis clean | equally a fibre splice, a misaligned coupler, a detector at the wrong bias |
| 6 | `basis_biased_probe` | `basis_consistency` flagged, pooled clean | a waveplate a degree out of alignment produces the same asymmetry |
| 7 | `selective_slot_attack` | decoy/position/level split | with few checks per group, an unlucky draw imitates it |
| 8 | `forgery_or_measure_and_resend` | rate ≥ 0.75·forger_rate | no honest preset reaches this rate, but degradation is not impossible |
| 9 | `intercept_and_block` | rate *and* yield out of spec | an aging fibre that is both lossy and noisy produces the same pair |
| 10 | `elevated_channel_error` | `pooled_rate_vs_spec` flagged, no shape | the generic finding: "this link is not the link that was specified" |
| 11 | `rate_exceeds_transfer_threshold` | `pooled_rate_vs_transfer` flagged | a comparison against a fixed number, not a claim about cause |
| 12 | `sequential_onset_detected` | `sequential_onset` flagged | says *when*, not *what*; an early unlucky cluster can trip it |

The design commitment visible in every row is that the engine reports a
*candidate* cause and never announces that an attack has occurred. Attribution 8
is the only one tied to a numeric region — the rate has climbed to within
three-quarters of the forger rate, i.e. into the band a measure-and-resend
eavesdropper or an actual forger produces — and even it names degradation as the
unlikely-but-possible alternative.

---

## 7. What the layer catches, and what it misses

Measured on the eight snapshot scenarios at `m=16, L=24` (`N ≈ 384`–602 usable
checks), judged against the declared lab-grade sheet:

| scenario | verdict | pooled rate | flagged | primary attribution |
|---|---|---|---|---|
| `honest-lab` | clean | 0.0186 | 0/11 | — |
| `intercept-resend` | compromised | 0.2375 | 4/11 | signature_rejected_on_rate |
| `intercept-resend-partial` | compromised | 0.1488 | 3/11 | signature_rejected_on_rate |
| `basis-biased` | compromised | 0.0873 | 3/11 | signature_rejected_on_rate |
| `collective-depolarizing` | **clean** | 0.0290 | 0/11 | — |
| `coherent-probe` | **clean** | 0.0230 | 0/11 | — |
| `entanglement-breaking` | compromised | 0.3203 | 4/11 | signature_rejected_on_rate |
| `degraded-link` | compromised | 0.0199 | 1/11 | channel_blocking_or_blinding |

The honest run flags nothing. The strong attacks are caught decisively, the
basis-biased probe crosses the acceptance threshold on its pooled rate and is corroborated by the per-basis test that names the asymmetry, and the
innocently degraded link is caught in the *yield* rather than the error rate —
the distinction the whole design exists to preserve.

Two weak attacks — a collective depolarising probe at `p=0.05` and a coherent
probe at `theta=0.3` — come back **clean, zero tests flagged**. This is reported
here with the same prominence as the successes because it is the honest
characterisation. Both sit below the acceptance threshold `s_a=0.0396`, so the
protocol accepts them correctly on its own terms, and both are close enough to
the floor that the exact test cannot reject the null at this key length. The
detector is behaving exactly as specified; the specification does not have the
sensitivity, at `N ≈ 590`, to see an attack this weak. The boundary is a function
of `N` — the whitepaper §5.7 shows lab-grade hardware needs `N=1080` for a
one-in-a-billion target — and a weak attack that stays below `s_a` is also one
that gained correspondingly little information, which is the whole content of
Lemma 2. Undetected here does not mean successful forgery; it means the adversary
paid for invisibility in the currency the protocol charges.

---

## 8. Using the engine

```python
from qds.protocol.session import QDSSession, SessionConfig
from qds.protocol.distribution import DistributionConfig
from qds.channel import LAB_GRADE, make_intervention
from qds.detect import analyse_session

cfg = SessionConfig(
    distribution=DistributionConfig(
        noise=LAB_GRADE,
        intervention=make_intervention("intercept_resend", strength=1.0),
    ),
    noise_floor_spec=LAB_GRADE.predicted_qber(),   # declared, before the run
)
session = QDSSession(cfg, seed=12)
session.run(b"hi")

report = analyse_session(session, declared_noise=LAB_GRADE)
print(report.verdict)                 # 'compromised'
print(report.to_dict()["flagged"])    # the tests that fired
```

Pass `declared_noise` so the yield and dark-count premise tests are live;
without it they mark themselves absent (§2). The returned `DetectionReport`
serialises to the snapshot contract documented in `docs/ARCHITECTURE.md` §7.
