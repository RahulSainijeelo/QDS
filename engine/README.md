# QDS Engine

The Python package behind the framework: the protocol, the physics simulation it
runs on, the statistics-only detection engine, the attack suite, and the
command-line front-end. Everything here is built and under test except one
package, `qds/metrics` (an empty stub), which is marked as such in the root
`README.md`.

For the design rationale read `docs/ARCHITECTURE.md`; for the detector read
`docs/DETECTION.md`; for the adversary and trust model read `docs/THREAT_MODEL.md`.
This file is the operational guide to the code.

## Dependencies

**NumPy only.** The statistics are scipy-free by design — every distribution
function and tail bound is implemented from scratch in `qds/detect/stats.py`, so
there is nothing to pin and nothing to drift. **Qiskit is optional**: it backs the
second state-vector backend, which exists solely to cross-validate the NumPy
reference to `1e-12`. Without Qiskit installed, the two cross-validation tests skip
and nothing else changes.

```bash
pip install numpy          # qiskit optional
```

Python 3.10 or newer.

## Running the tests

```bash
cd engine
python3 -m unittest discover -s tests -p 'test_*.py'
```

The suite is **276 tests, 2 skipped** (the two skips are the Qiskit cross-checks,
which skip when only one backend is available). It runs under the standard-library
`unittest` runner.

**This is not a pytest suite.** `pytest` is not a dependency; the discovery command
above is the supported way to run it, and `pytest` will not collect these tests the
way the runner expects.

## The public API

`qds/__init__.py` is intentionally empty — there is no top-level re-export, and you
import from the subpackages, each of which declares an explicit `__all__`. The four
entry points a caller needs:

```python
# 1. the channel: noise/loss presets and the adversary factory
from qds.channel import IDEAL, LAB_GRADE, FIELD_GRADE, PRESET_NOISE, \
                        NoiseModel, LossModel, make_intervention

# 2. the protocol: configure a session and run it
from qds.protocol.session import QDSSession, SessionConfig
from qds.protocol.distribution import DistributionConfig

# 3. the detector: the short path is one function
from qds.detect import analyse_session              # -> DetectionReport
from qds.detect import Evidence, DetectionEngine     # the longer path

# 4. the backends: pick a simulator
from qds.backends import get_backend, available_backends
```

A minimal end-to-end run is in the root `README.md` quickstart. The detector's
short path, `analyse_session(session, declared_noise=SPEC)`, is almost always what
you want; pass `declared_noise` so the yield and dark-count premise tests run live
rather than marking themselves absent (`docs/DETECTION.md` §2).

## Module tour

The full map with the dependency rules is `docs/ARCHITECTURE.md` §2. In brief:

`linalg.py` is the density-matrix toolkit — Kronecker products, partial traces,
the big-endian qubit convention (qubit 0 is most significant, matching `np.kron`;
Qiskit is opposite and is translated only inside its adapter). `channel.py` holds
the `NoiseModel` and `LossModel`, the three presets (`IDEAL`, `LAB_GRADE`,
`FIELD_GRADE`), and `make_intervention`, which builds the adversarial channels.

`backends/` has two state-vector simulators behind one interface: `numpy_dm`
(the reference) and `qiskit_dm` (validation). Select one with
`get_backend("numpy")` or `get_backend("qiskit")`; `available_backends()` reports
which are importable.

`protocol/` is the scheme itself — `keys`, `distribution`, `signing`,
`verification`, `auth`, and `session`, which composes the five phases
(enrol → distribute → symmetrise → calibrate → sign+verify) in `QDSSession.run`.

`detect/` is the statistics-only engine, documented in full in `docs/DETECTION.md`:
`stats` (scipy-free distributions), `estimators` (rates with Clopper–Pearson
intervals), `thresholds` (the declared-inputs-only ladder), `rules` (the eleven
detectors), and `engine` (the `Evidence` boundary, the runner, the twelve-entry
attribution table, the verdict).

`attacks/` is the built, tested adversary suite: five families (`forgery`,
`impersonation`, `replay`, `channel`, `repudiation`) across 24 variants, each
mounted as a real protocol run and judged by the engine on the unprivileged
transcript. `attacks/__init__.py` exports `CATALOGUE`, `FAMILIES`, `mount`, and
`run_all`. `cli.py` is the thin command-line front-end over it and the detector —
`python3 -m qds {list,run,attack,sweep,selfcheck,bench}` — which adds no physics
of its own. `metrics/` is the one **specified-but-unbuilt** interface: an empty
`__init__.py`, documented as designed in `docs/ARCHITECTURE.md` §8.

## The privilege boundary

The detection package consumes exactly one object, `Evidence`, assembled only from
what a real verifier could observe. It is structurally forbidden from reading the
simulator's ground truth — Eve's probe states, the true QBER, the exact fidelity —
and three tests in `tests/test_detect.py::TestPrivilegeBoundary` enforce this: a
source grep against eight forbidden identifiers, an AST guard that forbids calling
`.summary()` on anything but `self`, and an output scrub over the serialised
report. If you extend the detector, it may read `Evidence` and nothing else.
`docs/THREAT_MODEL.md` §5 explains why.
