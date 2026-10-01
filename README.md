# QDS — Quantum Digital Signatures

A simulation and detection framework for **teleportation-based quantum digital
signatures**: Lamport-style one-bit-at-a-time signing whose public keys are BB84
states delivered by quantum teleportation, with an unconditional security
argument and a statistics-only engine that decides whether a finished run stayed
inside the conditions that argument assumed.

The scheme makes the three promises any signature scheme must — **unforgeability,
non-repudiation, transferability** — but earns them from the no-cloning theorem
and the geometry of four quantum states rather than from any problem being hard to
compute. A forger who learned enough to sign is bound to disturb the key by at
least a `1/4` error rate (Lemma 2); honest noise sits far below that; and the
whole protocol lives in the gap, enforced by the threshold ordering
`eta < s_a < s_v < 1/4`.

## The one binding constraint

**No machine learning anywhere in detection.** Not a classifier, not a fitted
threshold, not a learned anomaly score. Every threshold is a closed form in
quantities fixed *before* the run — the declared noise floor, the number of
checks, and the two forgery bounds that come out of quantum mechanics — and
attribution is a fixed lookup table, not a model. The reason is in one sentence: a
detector trained on simulated attacks would certify the simulator, not the scheme,
and a bound proved against all adversaries survives one it has never seen. This
constraint shaped every design decision below, and `docs/DETECTION.md` §1 argues
it in full.

## Repository layout

```
engine/        the protocol, the physics simulation, and the detection engine (Python)
  qds/
    linalg.py          density matrices, Kronecker products, partial trace
    channel.py         noise and loss models, adversarial interventions
    backends/          two state-vector backends: numpy (reference) + qiskit (validation)
    protocol/          keys, distribution, signing, verification, auth, session
    detect/            the eleven tests, thresholds, estimators, attribution engine
    attacks/           adversary suite — SPECIFIED; implementation in progress
    metrics/           ground-truth metrics — SPECIFIED (stub)
  tests/               unittest suite (177 tests)
web/           the analytics dashboard
  lib/                 TypeScript data contract + loaders + formatting (built)
  public/data/         committed engine output: index + 8 scenarios (built)
  scripts/             generate_snapshots.py — regenerates public/data from the engine
  app/, components/    Next.js app shell — SPECIFIED; no package.json yet
docs/          WHITEPAPER · ARCHITECTURE · DETECTION · THREAT_MODEL
```

## Status

This is honest about what runs today and what is written down as a designed
interface. Nothing in the first group is aspirational; nothing in the second
pretends to work.

| Component | State |
|---|---|
| Linear algebra, channel, noise/loss models | **built**, tested |
| Protocol: keys, distribution, signing, verification, auth, session | **built**, tested |
| Detection: 11 tests, thresholds, estimators, attribution | **built**, tested |
| NumPy backend | **built**, tested |
| Qiskit backend + cross-validation | **built**; skips when Qiskit absent |
| Test suite | **177 tests, 2 skipped** (the two are the Qiskit cross-checks) |
| Dashboard data contract, loaders, snapshots | **built** — real engine output in `web/public/data` |
| Snapshot generator | **built** — one command reproduces the whole dataset |
| Next.js app shell (`web/app`, `package.json`, build config) | **specified**, not built — no runnable app yet |
| Dashboard components (`web/components`) | **specified**; implementation in progress |
| Attack suite (`qds/attacks`) | **specified**; implementation in progress |
| Ground-truth metrics (`qds/metrics`) | **specified** (stub) |

The specified interfaces are documented as designed in `docs/ARCHITECTURE.md` §8.
Some are being implemented separately and in parallel — where a row says "in
progress," treat the code as landing rather than finished: it is not yet wired
into the package (`qds/attacks/__init__.py` exports nothing) or into a runnable
app (there is no `package.json`), and nothing in it is covered by the test suite
or by the verification behind these docs. The dashboard's *data* contract,
however, is already real and populated, so the missing app shell is presentation,
not substance.

## Requirements

Python 3.10+ and **NumPy** — that is the entire hard dependency list. The
statistics are **scipy-free by design**: every distribution function is
implemented from scratch in `qds/detect/stats.py` (log-gamma binomial tails,
Lentz continued-fraction incomplete betas, Clopper–Pearson intervals, Chernoff–KL
and Hoeffding–Serfling bounds, Fisher's exact test, chi-square, SPRT, CUSUM), so
there is nothing to pin and nothing to drift. Qiskit is **optional**, used only by
the second backend to cross-validate the reference one to `1e-12`; without it the
two cross-validation tests skip and everything else is unaffected.

```bash
pip install numpy          # qiskit optional, for cross-validation only
```

## Quickstart

```bash
# 1. run the test suite  (NOTE: unittest, not pytest)
cd engine
python3 -m unittest discover -s tests -p 'test_*.py'

# 2. sign a message, then ask the detector what it saw
python3 - <<'PY'
from qds.protocol.session import QDSSession, SessionConfig
from qds.protocol.distribution import DistributionConfig
from qds.channel import LAB_GRADE, make_intervention
from qds.detect import analyse_session

cfg = SessionConfig(
    distribution=DistributionConfig(
        noise=LAB_GRADE,
        intervention=make_intervention("intercept_resend", strength=1.0),
    ),
    noise_floor_spec=LAB_GRADE.predicted_qber(),   # declared before the run
)
session = QDSSession(cfg, seed=12)
session.run(b"hi")
report = analyse_session(session, declared_noise=LAB_GRADE)
print(report.verdict, report.to_dict()["flagged"])
PY

# 3. regenerate the dashboard's data from the engine
python3 web/scripts/generate_snapshots.py
```

The suite runs under the standard-library `unittest` runner. **It is not a pytest
suite** — `pytest` is not a dependency and the discovery pattern above is the
supported way to run it.

## Documentation

Four documents in `docs/`, each with a different job:

[`WHITEPAPER.md`](docs/WHITEPAPER.md) is the academic preprint — the protocol,
the three lemmas, the teleportation-leaks-nothing proposition, the threshold
derivation, and the numerical results with the two honest misses reported in
full. [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the engineering companion —
module map, the qubit-ordering convention, the data-flow, the privilege boundary,
and the snapshot contract. [`DETECTION.md`](docs/DETECTION.md) is the reference
for the eleven tests, the Bonferroni multiplicity, the three-verdict logic, and
the twelve-entry attribution table. [`THREAT_MODEL.md`](docs/THREAT_MODEL.md)
states the adversary, the trust assumptions, the classical MAC and anti-replay
layer, and the detector's own privilege boundary.

New readers: start with the WHITEPAPER for *why it is secure*, then ARCHITECTURE
for *how the code is shaped*. Operators: DETECTION and THREAT_MODEL.

## A note on the data

Every number the dashboard shows is produced by running the real engine and
serialising the result — `web/scripts/generate_snapshots.py` writes
`web/public/data`, and nothing there is hand-written or adjusted for
presentation. If a scenario's verdict looks surprising, the engine said it: two of
the eight scenarios are weak attacks that come back *clean* because they stay below
the acceptance threshold, and that is reported with the same prominence as the
catches (`docs/DETECTION.md` §7). The framework's whole stance is to say what the
evidence supports and stop there.
