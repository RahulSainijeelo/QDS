# Architecture

This document describes how the implementation is organised: the modules, the
data that flows between them, the conventions that are easy to get wrong, and
the boundaries that are enforced rather than merely intended. It is the
engineering companion to `docs/WHITEPAPER.md`, which covers the science, and
`docs/DETECTION.md`, which covers the eleven tests in detail.

The repository has two halves. `engine/` is a dependency-light Python package
(`qds`) that implements the protocol, the quantum simulation and the detection
layer. `web/` is a Next.js dashboard that reads static JSON snapshots the engine
produces and renders them; it never runs the engine at request time. A generator
script, `web/scripts/generate_snapshots.py`, is the bridge: it runs the real
engine and serialises the result.

---

## 1. Layering

The engine is strictly layered. Each layer depends only on those below it, and
the dependency never runs the other way.

```
            +-------------------------------------------------+
            |  web/  (Next.js dashboard — reads JSON only)    |
            +-------------------------------------------------+
                               ^  static JSON
                               |
            +-------------------------------------------------+
            |  web/scripts/generate_snapshots.py              |
            |  (runs the engine, serialises the result)       |
            +-------------------------------------------------+
                               ^
                               |
   engine/qds/
   +---------------------+   +-----------------------------+
   |  detect/            |   |  attacks/   (built)         |
   |  statistics-only    |   |  cli.py     (built)         |
   |  threat detection   |   |  metrics/   (spec)          |
   +---------------------+   +-----------------------------+
            ^   consumes only Evidence
            |
   +-------------------------------------------------------+
   |  protocol/   keys, auth, distribution, signing,       |
   |              verification, session                    |
   +-------------------------------------------------------+
            ^
            |
   +---------------------+   +-----------------------------+
   |  channel.py         |   |  backends/                  |
   |  noise + adversary  |   |  numpy (ref) + qiskit (val) |
   +---------------------+   +-----------------------------+
            ^                            ^
            |                            |
   +-------------------------------------------------------+
   |  linalg.py   density matrices, gates, measurement     |
   +-------------------------------------------------------+
```

Two directions of dependency are deliberately absent. The protocol layer does
not import the detection layer — detection consumes the protocol's output, never
the reverse — with one surgical exception noted in §6. And the detection layer
does not import the simulator's ground truth, which is the privilege boundary of
§5 and is enforced by a test.

---

## 2. The module map

### 2.1 `linalg.py` — the numerical floor

Everything quantum bottoms out here: the Pauli matrices and Hadamard, single-
and two-qubit gate application on a density matrix, the standard channels
(depolarising, dephasing, Pauli, amplitude damping) as Kraus maps, projective
measurement in a named basis, and partial trace.

The one convention that pervades the whole engine is **qubit ordering**. This
codebase is **big-endian**: qubit 0 is the most significant tensor factor, which
is the ordering `np.kron` produces naturally. Qiskit uses the opposite
convention, and the Qiskit adapter is the only place that convention is
translated (§3). Every index passed to `linalg`, `channel` and `protocol` is in
the big-endian convention.

### 2.2 `channel.py` — honest noise and adversarial interventions

Two conceptually different things live here, and the separation is load-bearing
for the security argument. `NoiseModel` is what an uncompromised link does on its
own: channel depolarisation, memory dephasing, gate error, measurement (readout)
error, misalignment, and a `LossModel` for transmittance, detector efficiency
and dark counts. Three presets are provided — `IDEAL`, `LAB_GRADE`,
`FIELD_GRADE` — whose predicted QBERs are 0.0, 0.0124 and 0.0470 respectively.

Adversarial behaviour is a separate hierarchy of `Intervention` subclasses —
`NoIntervention`, `InterceptResend`, `EntanglementBreaking`,
`CollectiveDepolarizing`, `CoherentProbe`, `BasisBiasedProbe` — constructed
through `make_intervention(name, **kwargs)`. The whole point of the framework is
that an intervention adds disturbance *on top of* the honest floor, so anything
strong enough to be useful is strong enough to be seen.

`NoiseModel.predicted_qber()` is an analytic prediction, not a measurement. It
composes independent flip sources by odd-parity composition
(`e = (1 - prod_i (1 - 2 e_i)) / 2`) and — importantly — counts gate-error
insertions **separately per basis**: four reach a Z-basis check, six reach an
X-basis check, derived by propagating each Pauli error through the teleportation
circuit. Lumping them into one rate overstates the error by about 8% at
`p_g = 0.04`. The per-basis predictions are exposed via
`predicted_qber(basis="Z"|"X")`.

### 2.3 `backends/` — pluggable state representation

`backends/base.py` defines the `Backend` ABC: ten abstract methods (`zero`,
`from_ket`, `tensor`, `apply`, `apply_kraus`, `measure`, `expectation`,
`partial_trace`, `to_numpy`, `num_qubits`) plus a concrete `fidelity`. Two
implementations are registered: `numpy` (`numpy_dm.py`), the reference, and
`qiskit` (`qiskit_dm.py`), intended for cross-validation via `quantum_info` at a
tolerance of `1e-12`. `get_backend(name="numpy")` returns an instance;
`available_backends()` lists what is installed.

**The Qiskit backend has never been executed** — the development environment had
no network access to install Qiskit, so the two cross-validation tests skip with
"only one backend available". It should be treated as untested. See
`docs/THREAT_MODEL.md` §limitations and the whitepaper §10.

### 2.4 `protocol/` — the scheme itself

Read these in order; each assumes the previous.

`keys.py` defines what a private key is (`PrivateKey`, blocks indexed by
`(j, b)`, `2mL` entries), how a slot is addressed (`Slot(j, b, i)`, with
`SlotRole.CHECK = -1` and `SlotRole.DECOY = -2`), what a recipient holds
(`HeldQubit`, `PublicKeyStore`), and identity enrolment (`SignerRegistry`, whose
`key_id` comparison uses `hmac.compare_digest`). It also fixes the two forgery
bounds as module constants — `FORGER_PASS_PROB_BLIND = 0.5` and
`FORGER_PASS_PROB_ONE_COPY = 0.75` — and cites the whitepaper's Lemma 1 and
Lemma 2 for their derivations. Within the protocol layer those constants are the
single source of truth: `verification.py` derives its error rates from them as
`1 - p`, so `1/4` is not re-typed as a literal there. The detection layer is a
different matter and the difference is deliberate — `detect/` may not import
`protocol/` (that is the privilege boundary, §5), so `detect/thresholds.py`
cannot reach `keys.py` and instead declares its own error-rate constants,
`FORGER_ERROR_RATE = 0.25` and `BLIND_FORGER_ERROR_RATE = 0.50`. These are the
same two bounds expressed as error rates rather than pass probabilities, and they
must be kept consistent with `keys.py` by hand; the layering forbids deriving one
from the other, and no test currently cross-checks the two copies, so an editor
changing a bound in one layer must change it in the other.

`auth.py` is the classical layer: the Wegman–Carter MAC over
`GF(2^64) = GF(2)[x]/(x^64 + x^4 + x^3 + x + 1)` (reduction constant `0x1B`), the
one-time `KeyReservoir` that refuses to reissue a key, and the `FreshnessLedger`
with its four replay predicates (counter, nonce, timestamp window, consumed
slots).

`distribution.py` implements Bell-pair creation, teleportation, and the proof
obligations around them: `bell_outcome_distribution` (uniform `1/4` for every
input) and `teleportation_leakage` (zero to machine precision). The recipient's
correction map is `Z^u X^v`, applied as one composed gate.

`signing.py` turns a message plus a private key into a `SignatureDeclaration`:
domain tag `TQDS-DECL-v1`, 16-byte nonce, and one MAC tag per recipient, all
produced by the signer.

`verification.py` is the verifier. `VerificationPolicy` holds the two thresholds
and knows how to place them (`balanced` vs `from_noise_floor`/`thirds`), what
they cost (`failure_probabilities`, `security_bits`), how long a key must be
(`required_checks`), and whether the policy is sound at all
(`soundness_problems`). `verify_declaration` runs the four gates in order and
returns a `VerificationReport` carrying every observable the detection layer will
later consume. `measure_slot` is the single projective measurement.

`session.py` is the assembly: `SessionConfig`, `QDSSession` with its five phase
methods (`enrol`, `distribute`, `symmetrise`, `calibrate`, `sign`/`verify`), and
the convenience `run()`. It also owns `symmetrise_keys` (Lemma 3) and
`calibrate_from_decoys` (conformance testing, not threshold setting). The object
is stateful on purpose — a public key is consumable, so the phases must happen
once, in order, and calling them out of order raises rather than returning a
meaningless number.

### 2.5 `detect/` — statistics-only threat detection

`stats.py` is scipy-free exact statistics (binomial tails via log-gamma, Lentz
continued fractions for incomplete beta/gamma, Clopper–Pearson, Chernoff–KL,
Hoeffding–Serfling, Fisher, chi-square, SPRT, CUSUM). `estimators.py` turns
verification outcomes into rates with exact intervals, careful about which
denominator each rate uses. `thresholds.py` is the declared-inputs-only threshold
ladder. `rules.py` is the eleven detectors, one null hypothesis each. `engine.py`
is the `Evidence` boundary, the runner, the attribution table, and the verdict.
The short path for a caller is `analyse_session(session, declared_noise=...)`.

### 2.6 `metrics/` — the one specified-but-unbuilt interface

`engine/qds/metrics/__init__.py` is an empty stub: the package intended to
aggregate information-gain-versus-disturbance curves over a campaign is not yet
implemented. It is the sole module in the engine that is still a specification
rather than code; §8 states the contract its implementation must satisfy.

`attacks/` is no longer in that category — it is built and tested.
`attacks/__init__.py` exports the catalogue (`CATALOGUE`, `FAMILIES`, `mount`,
`run_all`), and the package (`forgery.py`, `impersonation.py`, `manipulation.py`,
`replay.py`, `repudiation.py`, with `harness.py`) orchestrates five attack
families as real protocol runs over the adversarial *interventions* that already
live in `channel.py`. The command-line front-end `cli.py` drives both the suite
and the detection engine; see §8.

---

## 3. The qubit-ordering convention

Because this is the single most common source of subtle bugs in a quantum
simulation, it is worth isolating. The engine is big-endian throughout: in an
`n`-qubit state, qubit 0 is the leftmost (most significant) tensor factor. This
matches `np.kron(a, b)` placing `a` in the high bits. The Qiskit adapter, and
only the Qiskit adapter, translates to Qiskit's little-endian convention using

```python
_qargs(n, qubits) = [n - 1 - q for q in reversed(list(qubits))]
```

which is a double transformation (reverse the list *and* complement each index)
because the two conventions differ in both respects. Any future backend must
present the big-endian convention at the `Backend` interface regardless of what
it uses internally.

---

## 4. Data flow: one session, end to end

```
SessionConfig
   |
   v
QDSSession.enrol()       -> KeyReservoir, SignerRegistry, FreshnessLedger per recipient
QDSSession.distribute()  -> PublicKeyStore per recipient (teleported qubits + logs)
QDSSession.symmetrise()  -> qubits permuted across recipients (signature slots only)
QDSSession.calibrate()   -> Calibration per recipient + one shared VerificationPolicy
QDSSession.sign(msg)     -> SignatureDeclaration (one MAC tag per recipient)
QDSSession.verify(...)   -> VerificationReport   (observable outcome)
QDSSession.transfer(...) -> VerificationReport   (at the transfer threshold)
   |
   |  session.reports : Dict["verifier:level" -> VerificationReport]
   v
analyse_session(session, declared_noise=SPEC)
   |
   |  builds Evidence from reports + calibrations + logs + DECLARED figures
   v
DetectionReport   -> .to_dict()  -> JSON snapshot  -> dashboard
```

The reason the session retains the `VerificationReport` **objects**, keyed
`"verifier:level"`, rather than only their dictionaries, is that verification
consumes the qubits it measures. Re-running verification to recover a report is
physically impossible, so the objects are kept.

---

## 5. The privilege boundary

This is the architectural decision the whole detection layer is organised
around, and it is enforced, not merely documented.

The simulator knows quantities no real verifier could: `DistributionOracle`
holds Eve's actual probe states and the exact fidelity of every delivered qubit;
the channel knows the true QBER, the CHSH value, the concurrence. If the code
that decides "this run was attacked" could read any of those, the resulting
detection rate would be a restatement of the input, not a measurement.

So a single dataclass, `detect.Evidence`, is the *only* object the detection
engine consumes. It is populated from two kinds of thing: observable outcomes
(`VerificationReport`s, `Calibration`s, detector logs, correction histories) and
figures **declared before the run** — the spec rate, the declared yield, the
declared dark fraction, the thresholds — each annotated in the source as
"declared, fixed before the run — never estimated from this session". The
`declaration_source` field records where the declared figures came from
(`"spec_sheet"` when the generator passes `declared_noise`), and if they are
instead read off the channel being judged, the yield and dark-count tests mark
themselves *absent* rather than silently comparing a quantity with itself.

The boundary is enforced three ways. Structurally, the engine's entry points
accept only `Evidence`. By review, the declared fields carry the annotation. And
by an automated test, `TestPrivilegeBoundary`, which reads the entire
`qds.detect` package source off disk and asserts that a forbidden symbol list —
`DistributionOracle`, `.oracle`, `exact_qber`, `mean_chsh`, `mean_bell_fidelity`,
`mean_concurrence`, `information_disturbance`, `announcement_error_rate` —
appears nowhere in it. That grep is part of the security argument, and it
currently passes.

---

## 6. The one allowed upward dependency

`protocol/verification.py` and `protocol/session.py` both import
`from ..detect import stats`. These two imports are the only places the protocol
layer reaches into the detection layer, and the dependency is deliberate:
`stats` is pure, QDS-agnostic numerical code (binomial tails, Clopper–Pearson)
with no knowledge of sessions, evidence or verdicts. The verifier needs exact
p-values and confidence intervals to put in its report, and duplicating the
continued-fraction machinery would be worse than the dependency. `stats` sits at
the bottom of the detection layer precisely so it can be shared without dragging
the rest of `detect` up into the protocol. No other `detect` module may be
imported from `protocol`.

---

## 7. The snapshot contract (engine ↔ dashboard)

The dashboard is a pure view: it reads JSON and renders it, and it never imports
Python. The contract between the two is therefore the shape of the files under
`web/public/data/`, all produced by `generate_snapshots.py` running the real
engine.

`index.json` is the catalogue: `generated_at`, `geometry` (message_bits, L,
signature_slots, checks_per_signature), `declared_spec` (preset, predicted_qber,
yield), the list of eleven `rules`, and a `scenarios` array of headline cards.
Each card carries `id`, `label`, `family`, `blurb`, `expectation`, `verdict`,
`alarm`, `pooled_rate`, `mismatches`, `checks`, the CI bounds, the three
thresholds, `n_flagged`/`n_applicable`/`n_tests`, the `flagged` list, the
`primary_attribution`, the `protocol_rejects` and `authentication_failed` flags,
and `combined_p_value`.

`scenarios/<id>.json` is the full document for one scenario: the headline fields
plus `declared_spec`, the actual `channel` parameters, `generated_at`,
`wall_seconds`, the complete `session` document (`SessionResult.to_dict()`) and
the complete `detection` document (`DetectionReport.to_dict()`). The detection
document's top-level keys are `verdict`, `alarm`, `combined_p_value`, `alpha`,
`alpha_per_test`, `n_tests`, `n_applicable`, `n_flagged`, `flagged`,
`attribution`, `tests`, `estimates`, `estimates_by_basis`, `thresholds`,
`security_bits`, `spec_violations`, `protocol_rejects`, `authentication_failed`,
`rejections`, `declaration_source`, `declared`, `signer`, `key_id`,
`elapsed_seconds`.

`session-lab.json` is a single standalone LAB_GRADE run kept at the top level for
convenience; its `config.L` is 24 and its `symmetrisation.slots_permuted` is 768.

Two serialisation rules are enforced by the generator and must not be relaxed:
numpy scalars are coerced to native Python types, and non-finite floats
(`NaN`, `Infinity`) are rewritten to `null`, because `JSON.parse` in the browser
rejects them. Both are handled by `_scrub`/`_plain` in the generator.

**Status of the dashboard.** The data contract is real and populated: `index.json`
and eight `scenarios/*.json` exist, and the TypeScript layer that reads them
(`web/lib`) is built. The Next.js application proper — `web/app` and the
`web/components` — is built as well: `package.json`, `pnpm-lock.yaml`,
`next.config.mjs`, and `tsconfig.json` are present, `next build` produces a static
export (`out/`), and an opt-in live mode (`QDS_LIVE=1`) exposes a single on-demand
engine endpoint. See `web/README.md`.

---

## 8. The specified-but-unbuilt interface

One component remains documented here as the interface its eventual
implementation must satisfy, rather than as working code: `qds.metrics`. The
attack suite and the command-line front-end, formerly in this section, are now
built and tested — see §2.6 and the note below.

**`qds.metrics`.** Aggregate figures of merit over a campaign: the disturbance an
attack induces against the information it gains, the detection probability as a
function of key length `N`, and the false-alarm rate under the declared null. The
whitepaper §8.2 notes that two weak attacks currently evade detection at
`N ≈ 590`; quantifying the detection-versus-`N` curve is exactly what this package
is for. Its `__init__.py` is an empty stub today.

**Now built — `qds.attacks` and `qds.cli`.** The attack suite is the campaign
layer over the `channel.Intervention` subclasses: five families (`forgery`,
`impersonation`, `replay`, `channel`, `repudiation`) spanning 24 variants, each
mounted as a real protocol run and judged by the engine on the unprivileged
transcript. `qds.cli` (`python3 -m qds`) is a thin front-end over that suite and
the detection engine, with subcommands `list`, `run`, `attack`, `sweep`,
`selfcheck`, and `bench`; `selfcheck` exits non-zero if any attack diverges from
its declared expectation. Both are covered by the test suite.

---

## 9. Testing

The suite is **`unittest`-based and requires no third-party test runner**. Run it
with

```bash
cd engine && python3 -m unittest discover -s tests -p 'test_*.py'
```

It currently reports **276 passing, 2 skipped**. The two skips are the Qiskit
cross-validation tests, which skip with "only one backend available" because
Qiskit is not installed; they are not failures. `numpy` is the only runtime
dependency.

Do **not** reach for `pytest`: it is not a declared dependency and is not
present in the reference environment. The tests are written as
`unittest.TestCase` classes precisely so the suite runs against the standard
library alone.

Two test classes carry more weight than the rest. `TestPrivilegeBoundary`
enforces §5. `TestFalseAlarmCalibration` generates runs where the null is true
by construction (Bernoulli draws at exactly the declared floor, no quantum
simulation) and checks that the family-wise alarm rate comes out at or under the
advertised alpha — without which "the engine flagged the attack" would be
evidence of nothing, since a detector that flags everything catches every
attack.

---

## 10. Conventions worth knowing before editing

The forgery bounds live in `keys.py` as pass probabilities; `verification.py`
derives its error rates from them, but `detect/thresholds.py` must keep its own
error-rate copy because the privilege boundary forbids it importing `protocol/` —
change a bound in one layer and you must change it in the other by hand.
Qubit ordering is big-endian; translate only at a foreign backend. Thresholds are
functions of declared inputs; never estimate one from the run being judged. The
detection layer may import `stats` from itself and nothing from `protocol`; the
protocol may import only `stats` from `detect`. Every rate carries its
denominator explicitly, because "slots that produced an outcome" and "slots the
verifier attempted" are different populations and conflating them hides exactly
the damage a blinding attack does. And an unusable policy or ladder is returned
and reported, never silently replaced with a workable one.
