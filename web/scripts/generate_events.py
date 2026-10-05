#!/usr/bin/env python3
"""Derive a chronological security-event log for each scenario.

"Logging of security events" is a stated deliverable of the framework.  This
script does **not** run the engine: it is a view over the already-generated
scenario snapshots in ``web/public/data/scenarios``.  For each scenario it
replays, in order, what the recorded run contains --

* the protocol phases from the session document (enrol, distribute,
  symmetrise, calibrate, sign, and each verify:<recipient>:<level>), timed;
* each of the eleven detection tests (info when it passed, alert when it
  flagged, carrying the test's p-value and name);
* every protocol-level rejection and any authentication failure, as alerts;
* the engine's terminal verdict, with a severity that follows it (clean =
  info, suspicious = warning, compromised = alert).

Each event is ``{seq, phase, severity, code, message, data}`` with
``severity`` in ``{info, warning, alert}``.  Messages are concise and factual;
every field is read straight off the serialised snapshot, nothing is invented.
An ``events/index.json`` lists the scenarios with their event and alert counts.

Usage
-----
    python3 web/scripts/generate_events.py [--data DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np  # noqa: E402  (kept for serialisation parity)


# --------------------------------------------------------------------------
# serialisation -- identical hygiene to generate_snapshots.py
# --------------------------------------------------------------------------

def _plain(obj: Any) -> Any:
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
# small formatters
# --------------------------------------------------------------------------

#: Terminal-verdict severity.  The three routes to a non-clean verdict all land
#: here; the mapping is the engine's own escalation.
VERDICT_SEVERITY = {"clean": "info", "suspicious": "warning",
                    "compromised": "alert"}

#: Chronological rank for the protocol phases; verify phases sort after the
#: setup phases, accept before transfer, in recipient order.
_BASE_RANK = {"enrol": 0, "distribute": 1, "symmetrise": 2, "calibrate": 3,
              "sign": 4}


def fmt_p(p: Any) -> str:
    """A compact p-value readout: scientific when tiny, 'n/a' when absent."""
    if p is None:
        return "n/a"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(p):
        return "n/a"
    if p <= 0.0:
        return "0"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def num(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"


def phase_rank(name: str, verifiers: List[str]):
    if name in _BASE_RANK:
        return (_BASE_RANK[name], 0, 0)
    if name.startswith("verify:"):
        parts = name.split(":")
        verifier = parts[1] if len(parts) > 1 else ""
        level = parts[2] if len(parts) > 2 else ""
        lvl = {"accept": 0, "transfer": 1}.get(level, 2)
        vidx = verifiers.index(verifier) if verifier in verifiers else 99
        return (5, lvl, vidx)
    return (9, 0, 0)


# --------------------------------------------------------------------------
# the event streams
# --------------------------------------------------------------------------

def describe_phase(name: str, seconds: Any, session: Dict[str, Any],
                   detection: Dict[str, Any], config: Dict[str, Any]):
    """One operator-readable line for a protocol phase."""
    data: Dict[str, Any] = {"seconds": seconds}
    if name == "enrol":
        signer = config.get("signer")
        verifiers = list(config.get("verifiers", []))
        data.update(signer=signer, verifiers=verifiers)
        return ("info", "phase.enrol",
                f"Enrolment: signer '{signer}' and recipient(s) "
                f"{', '.join(verifiers)} registered with public keys", data)
    if name == "distribute":
        slots = config.get("signature_slots")
        checks = config.get("checks_per_signature")
        data.update(signature_slots=slots, checks_per_signature=checks)
        return ("info", "phase.distribute",
                f"Distribution: entangled halves delivered and measured over "
                f"{slots} signature slots", data)
    if name == "symmetrise":
        sym = session.get("symmetrisation", {}) or {}
        performed = sym.get("performed")
        moved = sym.get("moved")
        data.update(performed=performed, moved=moved,
                    slots_permuted=sym.get("slots_permuted"))
        if performed:
            msg = (f"Symmetrisation: {moved} signature qubits privately "
                   "permuted among recipients to equalise key copies")
        else:
            msg = "Symmetrisation skipped (disabled for this run)"
        return ("info", "phase.symmetrise", msg, data)
    if name == "calibrate":
        thr = detection.get("thresholds", {}) or {}
        data.update(accept=thr.get("accept"), transfer=thr.get("transfer"),
                    spec_rate=thr.get("spec_rate"))
        return ("info", "phase.calibrate",
                f"Calibration: acceptance threshold {num(thr.get('accept'))}, "
                f"transfer threshold {num(thr.get('transfer'))} fixed from "
                f"declared floor {num(thr.get('spec_rate'))}", data)
    if name == "sign":
        decl = session.get("declaration", {}) or {}
        mh = decl.get("message_hex")
        data.update(message_hex=mh, checks=decl.get("checks"),
                    counter=decl.get("counter"))
        return ("info", "phase.sign",
                f"Signing: message 0x{mh} signed, {decl.get('checks')} one-time "
                f"checks committed (counter {decl.get('counter')})", data)
    if name.startswith("verify:"):
        parts = name.split(":")
        verifier = parts[1] if len(parts) > 1 else ""
        level = parts[2] if len(parts) > 2 else ""
        coll = session.get("transfers" if level == "transfer"
                           else "verifications", {}) or {}
        rec = coll.get(verifier, {}) or {}
        accepted = rec.get("accepted")
        data.update(verifier=verifier, level=level, accepted=accepted,
                    mismatch_rate=rec.get("mismatch_rate"),
                    threshold=rec.get("threshold"),
                    mismatches=rec.get("mismatches"), checks=rec.get("checks"))
        outcome = "accepted" if accepted else "refused"
        return ("info", "phase.verify",
                f"Verification ({verifier}, {level}): signature {outcome} -- "
                f"error rate {num(rec.get('mismatch_rate'))} vs threshold "
                f"{num(rec.get('threshold'))} "
                f"({rec.get('mismatches')}/{rec.get('checks')})", data)
    return ("info", "phase", f"Phase '{name}' completed", data)


def phase_events(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    session = doc.get("session", {}) or {}
    detection = doc.get("detection", {}) or {}
    config = session.get("config", {}) or {}
    verifiers = list(config.get("verifiers", []))
    timings = session.get("timings", {}) or {}
    events = []
    for name in sorted(timings, key=lambda n: phase_rank(n, verifiers)):
        sev, code, msg, data = describe_phase(
            name, timings[name], session, detection, config)
        events.append({"phase": name, "severity": sev, "code": code,
                       "message": msg, "data": data})
    return events


def test_events(detection: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    for t in detection.get("tests", []) or []:
        name = t.get("name")
        title = t.get("title") or name
        p = t.get("p_value")
        applicable = t.get("applicable", True)
        flagged = bool(t.get("flagged"))
        data = {"name": name, "p_value": p, "statistic": t.get("statistic"),
                "threshold": t.get("threshold"), "applicable": applicable,
                "flagged": flagged}
        if not applicable:
            events.append({"phase": "detect", "severity": "info",
                           "code": "test.skip",
                           "message": f"{name} not applicable: {title}",
                           "data": data})
        elif flagged:
            events.append({"phase": "detect", "severity": "alert",
                           "code": "test.flag",
                           "message": f"{name} flagged (p={fmt_p(p)}): {title}",
                           "data": data})
        else:
            events.append({"phase": "detect", "severity": "info",
                           "code": "test.pass",
                           "message": f"{name} passed (p={fmt_p(p)}): {title}",
                           "data": data})
    return events


def rejection_events(detection: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    rejections = detection.get("rejections", []) or []
    for r in rejections:
        verifier = r.get("verifier")
        level = r.get("level")
        reasons = list(r.get("reasons", []) or [])
        reason = reasons[0] if reasons else "protocol refused the signature"
        phase = (f"verify:{verifier}:{level}"
                 if verifier and level else "verify")
        events.append({
            "phase": phase, "severity": "alert", "code": "protocol.reject",
            "message": f"Signature refused for '{verifier}' at {level}: {reason}",
            "data": {"verifier": verifier, "level": level, "rate": r.get("rate"),
                     "threshold": r.get("threshold"),
                     "rate_exceeded": r.get("rate_exceeded"),
                     "mac_ok": r.get("mac_ok"), "identity_ok": r.get("identity_ok"),
                     "fresh": r.get("fresh"), "reasons": reasons}})
    if detection.get("protocol_rejects") and not rejections:
        events.append({"phase": "verify", "severity": "alert",
                       "code": "protocol.reject",
                       "message": "Protocol rejected the signature", "data": {}})
    return events


def auth_events(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    detection = doc.get("detection", {}) or {}
    session = doc.get("session", {}) or {}
    if not detection.get("authentication_failed"):
        return []
    problems: List[str] = []
    for coll in ("verifications", "transfers"):
        for v, rec in (session.get(coll) or {}).items():
            if rec.get("mac_ok") is False:
                probs = rec.get("mac_problems") or []
                problems += [f"{v}: {p}" for p in probs] or [f"{v}: MAC failed"]
            if rec.get("identity_ok") is False:
                probs = rec.get("identity_problems") or []
                problems += ([f"{v}: {p}" for p in probs]
                             or [f"{v}: identity rejected"])
    tail = f" ({'; '.join(problems)})" if problems else ""
    return [{"phase": "authenticate", "severity": "alert", "code": "auth.fail",
             "message": ("Authentication failed: a classical binding gate "
                         "(identity or one-time MAC) rejected the declaration "
                         "before measurement" + tail),
             "data": {"problems": problems}}]


def verdict_event(detection: Dict[str, Any]) -> Dict[str, Any]:
    verdict = detection.get("verdict")
    sev = VERDICT_SEVERITY.get(verdict, "warning")
    attribution = detection.get("attribution") or []
    primary = attribution[0].get("label") if attribution else None
    nf = detection.get("n_flagged")
    na = detection.get("n_applicable")
    msg = (f"Verdict: {str(verdict).upper()} -- "
           f"{nf}/{na} detection tests flagged")
    if primary:
        msg += f"; primary attribution '{primary}'"
    return {"phase": "verdict", "severity": sev, "code": f"verdict.{verdict}",
            "message": msg,
            "data": {"verdict": verdict,
                     "combined_p_value": detection.get("combined_p_value"),
                     "n_flagged": nf, "n_applicable": na,
                     "n_tests": detection.get("n_tests"),
                     "primary_attribution": primary,
                     "protocol_rejects": detection.get("protocol_rejects"),
                     "authentication_failed":
                         detection.get("authentication_failed")}}


def build_events(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Protocol phases, then detection tests, then rejections/auth, then verdict."""
    detection = doc.get("detection", {}) or {}
    stream: List[Dict[str, Any]] = []
    stream += phase_events(doc)
    stream += test_events(detection)
    stream += rejection_events(detection)
    stream += auth_events(doc)
    stream.append(verdict_event(detection))
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(stream, 1):
        out.append({"seq": i, "phase": e["phase"], "severity": e["severity"],
                    "code": e["code"], "message": e["message"],
                    "data": e.get("data", {})})
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def scenario_order(data_dir: str, present: List[str]) -> List[str]:
    """Prefer the order index.json records; fall back to sorted ids."""
    order: List[str] = []
    idx_path = os.path.join(data_dir, "index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as fh:
                order = [c["id"] for c in json.load(fh).get("scenarios", [])]
        except (ValueError, KeyError, OSError):
            order = []
    ordered = [i for i in order if i in present]
    ordered += [i for i in present if i not in order]
    return ordered


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(os.path.join(here, "..", "public", "data"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=default_data,
                    help="dashboard data directory (default: web/public/data)")
    args = ap.parse_args()

    scen_dir = os.path.join(args.data, "scenarios")
    if not os.path.isdir(scen_dir):
        print(f"no scenarios directory at {scen_dir}", file=sys.stderr)
        return 2
    present = sorted(f[:-5] for f in os.listdir(scen_dir)
                     if f.endswith(".json"))
    if not present:
        print(f"no scenario snapshots in {scen_dir}", file=sys.stderr)
        return 2
    ids = scenario_order(args.data, present)
    out_dir = os.path.join(args.data, "events")

    print(f"data     {args.data}")
    print(f"output   {out_dir}")
    print(f"scenarios {len(ids)}\n")

    index_rows: List[Dict[str, Any]] = []
    tot_events = tot_alerts = 0
    for sid in ids:
        with open(os.path.join(scen_dir, f"{sid}.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        events = build_events(doc)
        write_json(os.path.join(out_dir, f"{sid}.json"),
                   {"scenario": sid, "generated_at": time.time(),
                    "events": events})
        alerts = sum(1 for e in events if e["severity"] == "alert")
        warnings = sum(1 for e in events if e["severity"] == "warning")
        info = sum(1 for e in events if e["severity"] == "info")
        verdict = (doc.get("detection", {}) or {}).get("verdict")
        index_rows.append({"scenario": sid, "label": doc.get("label"),
                           "verdict": verdict, "events": len(events),
                           "alerts": alerts, "warnings": warnings, "info": info})
        tot_events += len(events)
        tot_alerts += alerts
        print(f"  {sid:<26} {len(events):>3} events  "
              f"{alerts:>2} alert  {warnings:>1} warn  ({verdict})")

    write_json(os.path.join(out_dir, "index.json"),
               {"generated_at": time.time(), "scenarios": index_rows,
                "totals": {"scenarios": len(index_rows), "events": tot_events,
                           "alerts": tot_alerts}})
    print(f"\nwrote {len(index_rows)} event logs + index.json "
          f"({tot_events} events, {tot_alerts} alerts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
