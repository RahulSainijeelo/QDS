#!/usr/bin/env python3
"""Produce the campaign-metrics JSON the analytics dashboard reads.

Every figure here comes from :func:`qds.metrics.evaluate` -- the closed-form
performance layer -- never hand-written or tuned for presentation.  The engine
is imported exactly the way :mod:`generate_snapshots` imports it (via
``sys.path`` to the sibling ``engine`` directory).

The scheme's behaviour depends on the hardware it runs over, so this script
evaluates the three reference noise models the engine ships -- ``ideal``,
``lab``, and ``field`` (:data:`qds.channel.PRESET_NOISE`) -- and bundles a full
figure-of-merit set for each.  The thresholds for every preset are placed by the
protocol's own ``balanced`` rule on that preset's noise floor, so a preset entry
is the same construction the live protocol would use on that hardware -- not a
second, parallel calculation.  This reproduces the whitepaper's one-in-a-billion
key-length table directly: ideal needs N = 669 checks, lab-grade 1080, and
field-grade 1885 (whitepaper §5.7).

The binding constraint of the project holds here in full: nothing in the output
is trained, fitted, or weighted.  Every curve is a closed-form tail bound or an
exact binomial tail, and the single Monte-Carlo per preset (false-alarm
calibration) draws seeded binomial counts, never a quantum state.

Usage
-----
    python3 web/scripts/generate_metrics.py [--out DIR] [--trials N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

# Import the engine from the sibling directory without needing it installed --
# the same mechanism generate_snapshots.py uses.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.normpath(os.path.join(_HERE, "..", "..", "engine"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import qds.metrics as metrics  # noqa: E402
from qds.channel import PRESET_NOISE  # noqa: E402

# The reference hardware, in order of increasing noise.  Labels and blurbs are
# the dashboard's framing of presets the engine defines; the floor and every
# derived figure come from the engine, not from here.
_PRESETS = [
    ("ideal", "Ideal", "A noiseless channel -- the information-theoretic best case."),
    ("lab", "Lab-grade", "Shielded laboratory hardware; the whitepaper's headline figures."),
    ("field", "Field-grade", "A deployed link with realistic loss and dephasing."),
]


def _preset_entry(key: str, label: str, blurb: str, trials: int) -> Dict[str, object]:
    """A full :func:`qds.metrics.evaluate` bundle for one hardware preset.

    The noise floor is read from the engine's preset and the thresholds are
    re-placed on it by the protocol's balanced rule (``evaluate`` does this when
    given ``spec_rate``), so the entry matches what the live protocol would do
    on that hardware.
    """
    floor = float(PRESET_NOISE[key].predicted_qber())
    bundle = metrics.evaluate(spec_rate=floor, mc_trials=trials)
    return {"key": key, "label": label, "blurb": blurb, "floor": floor, **bundle}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "public", "data"),
                    help="directory to write metrics.json into")
    ap.add_argument("--trials", type=int, default=20000,
                    help="Monte-Carlo trials for false-alarm calibration, per preset")
    args = ap.parse_args()

    presets: List[Dict[str, object]] = [
        _preset_entry(key, label, blurb, args.trials)
        for key, label, blurb in _PRESETS
    ]

    doc = {
        "generated_at": time.time(),
        "default_preset": "lab",
        "trials": args.trials,
        "presets": presets,
    }

    out_dir = os.path.normpath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "metrics.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    # A short, honest console summary -- the required-N table is the headline.
    print(f"wrote {out_file}")
    for p in presets:
        f = p["forgery"]
        print(f"  {p['key']:>5}: floor={p['floor']:.6f}  "
              f"required_checks(1e-9)={f['required_checks']}  "  # type: ignore[index]
              f"within_budget={p['calibration']['within_budget']}")  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
