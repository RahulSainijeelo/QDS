#!/usr/bin/env python3
"""Produce the JSON snapshots the dashboard reads.

Every number the dashboard displays is produced here by running the real
engine -- :class:`qds.protocol.session.QDSSession` followed by
:func:`qds.detect.analyse_session` -- and serialising what comes back.
Nothing is hand-written, interpolated, or adjusted for presentation.  If a
scenario's verdict looks wrong, the engine said it, and that is the point:
the dashboard is a view onto the engine, not an illustration of it.

Two details are worth stating because they change what the detectors can do.

``declared_noise``
    Passed to :func:`analyse_session` so the evidence bundle records
    ``declaration_source="spec_sheet"``.  Without it the declared loss and
    dark-count figures are read off the same channel being judged, the yield
    and dark-count tests become comparisons of a quantity with itself, and the
    report marks them absent rather than passed.  Every scenario here declares
    the *lab* spec sheet, so those two tests are live throughout.

``degraded-link``
    The one scenario with no adversary at all: the channel runs worse than the
    sheet it was sold against.  It exists so the dashboard can show a
    detection that is real and yet innocent, which is the distinction the
    engine is careful never to collapse.

Usage
-----
    python3 web/scripts/generate_snapshots.py [--out DIR] [--bits N] [--L N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

# Import the engine from the sibling directory without needing it installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.normpath(os.path.join(_HERE, "..", "..", "engine"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import numpy as np  # noqa: E402

from qds.channel import (FIELD_GRADE, LAB_GRADE, LossModel,  # noqa: E402
                         NoiseModel, make_intervention)
from qds.detect import analyse_session  # noqa: E402
from qds.protocol.distribution import DistributionConfig  # noqa: E402
from qds.protocol.session import QDSSession, SessionConfig  # noqa: E402


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

#: The spec sheet every scenario is judged against.  Declared once, here,
#: before any run -- which is the property that makes the thresholds derived
#: from it meaningful.
SPEC = LAB_GRADE

#: A link that is genuinely worse than the sheet: same error physics, a
#: dimmer channel and noisier detectors.  Used only by ``degraded-link``.
DEGRADED = NoiseModel(
    channel_depolarizing=LAB_GRADE.channel_depolarizing,
    memory_dephasing=LAB_GRADE.memory_dephasing,
    gate_error=LAB_GRADE.gate_error,
    measurement_error=LAB_GRADE.measurement_error,
    misalignment=LAB_GRADE.misalignment,
    loss=LossModel(
        transmittance=FIELD_GRADE.loss.transmittance,
        detector_efficiency=FIELD_GRADE.loss.detector_efficiency,
        dark_count=FIELD_GRADE.loss.dark_count,
    ),
)


class Scenario:
    """One run's worth of configuration, plus how to describe it."""

    def __init__(self, sid: str, label: str, family: str, blurb: str,
                 expectation: str, seed: int,
                 intervention: str = "none",
                 intervention_kwargs: Dict[str, Any] | None = None,
                 noise: NoiseModel = LAB_GRADE,
                 symmetrise: bool = True):
        self.id = sid
        self.label = label
        self.family = family
        self.blurb = blurb
        self.expectation = expectation
        self.seed = seed
        self.intervention = intervention
        self.intervention_kwargs = intervention_kwargs or {}
        self.noise = noise
        self.symmetrise = symmetrise


SCENARIOS: List[Scenario] = [
    Scenario(
        "honest-lab", "Honest link", "baseline",
        "Lab-grade hardware running inside its specification, with no "
        "adversary. Establishes what a clean run looks like and sets the "
        "false-alarm reference for every other scenario.",
        "clean", seed=11,
    ),
    Scenario(
        "intercept-resend", "Intercept and resend", "attack",
        "Eve measures every travelling qubit and forwards a fresh one in the "
        "state she found. The textbook attack: full information about one "
        "basis, paid for with a disturbance no amount of care can hide.",
        "compromised", seed=12,
        intervention="intercept_resend", intervention_kwargs={"strength": 1.0},
    ),
    Scenario(
        "intercept-resend-partial", "Intercept and resend, half the rounds",
        "attack",
        "The same attack applied to half the rounds. Eve trades information "
        "for invisibility, which is the trade every real eavesdropper makes; "
        "the question the dashboard answers is how far down that curve the "
        "detectors still reach.",
        "compromised", seed=13,
        intervention="intercept_resend", intervention_kwargs={"strength": 0.5},
    ),
    Scenario(
        "basis-biased", "Basis-biased probe", "attack",
        "Disturbance aimed almost entirely at the X basis. Pooling the two "
        "bases halves the apparent damage, so this is the attack a "
        "single-number threshold is worst at seeing and the per-basis test "
        "exists to catch.",
        "suspicious", seed=14,
        intervention="basis_biased", intervention_kwargs={"pz": 0.0, "px": 0.12},
    ),
    Scenario(
        "collective-depolarizing", "Collective depolarising probe", "attack",
        "A weak depolarising channel applied to every round. It mimics honest "
        "degradation closely enough that the rate test is the only thing that "
        "separates them, and only in aggregate.",
        "suspicious", seed=15,
        intervention="collective_depolarizing", intervention_kwargs={"p": 0.05},
    ),
    Scenario(
        "coherent-probe", "Coherent probe with delayed measurement", "attack",
        "Eve entangles a probe with each qubit and defers her measurement "
        "until after the bases are announced. The subtlest attack in the "
        "suite: the disturbance is small by construction and the information "
        "gain arrives later.",
        "suspicious", seed=16,
        intervention="coherent_probe",
        intervention_kwargs={"theta": 0.3, "delayed_measurement": True},
    ),
    Scenario(
        "entanglement-breaking", "Entanglement-breaking channel", "attack",
        "The channel destroys the correlation the protocol runs on. Severe "
        "and easy to see, included as the upper end of the scale rather than "
        "as a plausible covert attack.",
        "compromised", seed=17,
        intervention="entanglement_breaking",
        intervention_kwargs={"strength": 0.6, "substitute": "mixed"},
    ),
    Scenario(
        "degraded-link", "Degraded link, no adversary", "hardware",
        "No eavesdropper. The fibre is dimmer and the detectors noisier than "
        "the sheet the thresholds were built from. A real detection with an "
        "innocent cause, and the reason the engine reports departures from "
        "specification rather than announcing attacks.",
        "detection without an adversary", seed=18,
        noise=DEGRADED,
    ),
]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def _plain(obj: Any) -> Any:
    """Coerce numpy scalars and non-finite floats into JSON-native values.

    ``json.dump`` cannot serialise ``np.float64`` and will happily emit
    ``NaN``/``Infinity``, which are not valid JSON and which ``JSON.parse``
    rejects in the browser. Both are turned into ``null`` here so the
    dashboard never has to guess what a malformed number meant.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def _scrub(obj: Any) -> Any:
    """Recursively replace non-finite floats, which json.dump emits raw."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_scrub(v) for v in obj.tolist()]
    return obj


def write_json(path: str, payload: Any) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(_scrub(payload), indent=1, default=_plain,
                      allow_nan=False, sort_keys=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return len(text)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run_scenario(sc: Scenario, message_bits: int, L: int) -> Dict[str, Any]:
    """Run one scenario end to end and return its snapshot document."""
    dist = DistributionConfig(
        noise=sc.noise,
        intervention=make_intervention(sc.intervention, **sc.intervention_kwargs),
    )
    cfg = SessionConfig(
        message_bits=message_bits,
        L=L,
        distribution=dist,
        symmetrise=sc.symmetrise,
        # The thresholds come from the sheet, never from the channel actually
        # running -- that is the whole point of declaring it in advance.
        noise_floor_spec=SPEC.predicted_qber(),
    )

    t0 = time.perf_counter()
    session = QDSSession(cfg, seed=sc.seed)
    result = session.run(b"hi")
    report = analyse_session(session, declared_noise=SPEC)
    elapsed = time.perf_counter() - t0

    detection = report.to_dict()
    session_doc = result.to_dict()

    return {
        "id": sc.id,
        "label": sc.label,
        "family": sc.family,
        "blurb": sc.blurb,
        "expectation": sc.expectation,
        "seed": sc.seed,
        "intervention": {
            "name": sc.intervention,
            "kwargs": dict(sc.intervention_kwargs),
            "description": dist.intervention.description,
        },
        "declared_spec": {
            "preset": "lab",
            "predicted_qber": float(SPEC.predicted_qber()),
            "yield": float(SPEC.loss.yield_),
            "transmittance": float(SPEC.loss.transmittance),
            "detector_efficiency": float(SPEC.loss.detector_efficiency),
            "dark_count": float(SPEC.loss.dark_count),
        },
        "channel": {
            "channel_depolarizing": float(sc.noise.channel_depolarizing),
            "memory_dephasing": float(sc.noise.memory_dephasing),
            "gate_error": float(sc.noise.gate_error),
            "measurement_error": float(sc.noise.measurement_error),
            "misalignment": float(sc.noise.misalignment),
            "transmittance": float(sc.noise.loss.transmittance),
            "detector_efficiency": float(sc.noise.loss.detector_efficiency),
            "dark_count": float(sc.noise.loss.dark_count),
            "yield": float(sc.noise.loss.yield_),
            "predicted_qber": float(sc.noise.predicted_qber()),
            "matches_declared_spec": bool(
                abs(sc.noise.loss.yield_ - SPEC.loss.yield_) < 1e-12
                and abs(sc.noise.predicted_qber() - SPEC.predicted_qber()) < 1e-12
            ),
        },
        "generated_at": time.time(),
        "wall_seconds": elapsed,
        "session": session_doc,
        "detection": detection,
    }


def headline(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The few numbers the catalogue page needs, lifted from a full snapshot."""
    det = doc["detection"]
    pooled = det["estimates"].get("pooled_rate", {})
    return {
        "id": doc["id"],
        "label": doc["label"],
        "family": doc["family"],
        "blurb": doc["blurb"],
        "expectation": doc["expectation"],
        "intervention": doc["intervention"]["name"],
        "verdict": det["verdict"],
        "alarm": bool(det["alarm"]),
        "pooled_rate": pooled.get("value"),
        "mismatches": pooled.get("k"),
        "checks": pooled.get("n"),
        "ci_low": pooled.get("ci_low"),
        "ci_high": pooled.get("ci_high"),
        "spec_rate": det["thresholds"].get("spec_rate"),
        "accept_threshold": det["thresholds"].get("accept"),
        "transfer_threshold": det["thresholds"].get("transfer"),
        "n_flagged": det["n_flagged"],
        "n_applicable": det["n_applicable"],
        "n_tests": det["n_tests"],
        "flagged": list(det["flagged"]),
        "primary_attribution": (det["attribution"][0]["label"]
                                if det["attribution"] else None),
        "protocol_rejects": bool(det["protocol_rejects"]),
        "authentication_failed": bool(det["authentication_failed"]),
        "combined_p_value": det.get("combined_p_value"),
        "wall_seconds": doc["wall_seconds"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = os.path.normpath(os.path.join(_HERE, "..", "public", "data"))
    ap.add_argument("--out", default=default_out,
                    help="output directory (default: web/public/data)")
    ap.add_argument("--bits", type=int, default=16, help="message length in bits")
    ap.add_argument("--L", type=int, default=24, dest="L",
                    help="repetitions per message bit")
    ap.add_argument("--only", default=None,
                    help="comma-separated scenario ids to regenerate")
    args = ap.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    chosen = [s for s in SCENARIOS if wanted is None or s.id in wanted]
    if not chosen:
        print(f"no scenario matched {args.only!r}", file=sys.stderr)
        return 2

    print(f"engine   {_ENGINE}")
    print(f"output   {args.out}")
    print(f"geometry message_bits={args.bits} L={args.L} "
          f"({2 * args.bits * args.L} signature slots)\n")

    cards: List[Dict[str, Any]] = []
    for sc in chosen:
        print(f"  {sc.id:<26} ", end="", flush=True)
        doc = run_scenario(sc, args.bits, args.L)
        size = write_json(os.path.join(args.out, "scenarios", f"{sc.id}.json"), doc)
        card = headline(doc)
        cards.append(card)
        rate = card["pooled_rate"] or 0.0
        print(f"{card['verdict']:<12} rate={rate:7.4f} "
              f"flagged={card['n_flagged']}/{card['n_applicable']} "
              f"{doc['wall_seconds']:5.1f}s  {size // 1024}kB")

    index = {
        "generated_at": time.time(),
        "geometry": {
            "message_bits": args.bits,
            "L": args.L,
            "signature_slots": 2 * args.bits * args.L,
            "checks_per_signature": args.bits * args.L,
        },
        "declared_spec": {
            "preset": "lab",
            "predicted_qber": float(SPEC.predicted_qber()),
            "yield": float(SPEC.loss.yield_),
        },
        "rules": [
            "pooled_rate_vs_spec", "pooled_rate_vs_transfer",
            "basis_consistency", "position_homogeneity", "level_homogeneity",
            "yield_vs_declared", "dark_excess", "correction_uniformity",
            "decoy_vs_signature", "recipient_agreement", "sequential_onset",
        ],
        "scenarios": cards,
    }
    # Only rewrite the index when the whole suite ran; a partial run would
    # otherwise silently drop the scenarios it did not regenerate.
    if wanted is None:
        write_json(os.path.join(args.out, "index.json"), index)
        print(f"\nwrote index.json with {len(cards)} scenarios")
    else:
        print(f"\npartial run ({len(cards)} scenarios); index.json left alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
