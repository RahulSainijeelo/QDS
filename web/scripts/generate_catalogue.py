#!/usr/bin/env python3
"""Produce the attack-catalogue JSON the dashboard reads.

Every entry here is derived from the engine's own attack suite -- the
``qds.attacks`` package -- never hand-written or tuned for presentation.  The
engine is imported exactly the way :mod:`generate_snapshots` imports it (via
``sys.path`` to the sibling ``engine`` directory), and the canonical instance
of each attack is mounted with ``seed=0``, which is precisely what
:func:`qds.attacks.run_all` does.  From each run we read:

* the attack's own ``name`` and ``description`` (the f-string the attack code
  builds, so it carries the real parameters);
* its :class:`~qds.attacks.harness.Expectation` -- the verdict/attribution the
  attack is *designed* to produce, which is the "expected effect" the code
  states for itself; and
* the keyword knobs that define the canonical instance, read off the
  implementing function's signature.

Family labels come from each module's docstring and family blurbs from the
package docstring, so the prose is the engine's prose.  The result is a view
onto the attack suite, not an illustration of it.

Usage
-----
    python3 web/scripts/generate_catalogue.py [--out DIR] [--seed N]
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List

# Import the engine from the sibling directory without needing it installed --
# the same mechanism generate_snapshots.py uses.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.normpath(os.path.join(_HERE, "..", "..", "engine"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import numpy as np  # noqa: E402

import qds.attacks as attacks  # noqa: E402
from qds.attacks import (forgery, impersonation, manipulation,  # noqa: E402
                         repudiation, replay)


# --------------------------------------------------------------------------
# the family -> module and forgery binding maps
# --------------------------------------------------------------------------

#: The family key ``channel`` is served by the ``manipulation`` module (stated
#: in the qds.attacks package docstring); every other family key is its own
#: module's name.  This is the one place the key/module distinction lives.
FAMILY_MODULE = {
    "forgery": forgery,
    "impersonation": impersonation,
    "replay": replay,
    "channel": manipulation,
    "repudiation": repudiation,
}

#: Forgery is the only family whose catalogue entries are lambdas that bake in a
#: strategy, so the implementing function and the baked keyword(s) are named
#: here -- read straight off ``qds.attacks.CATALOGUE``.  For every other family
#: the variant key *is* the name of the function that implements it, so its
#: parameters introspect directly from the module.
_FORGERY_BINDING = {
    "blind": (forgery.forge_and_submit, {"strategy": "blind"}),
    "single_copy": (forgery.forge_and_submit, {"strategy": "single_copy"}),
    "breidbart": (forgery.forge_and_submit, {"strategy": "breidbart"}),
    "partial": (forgery.forge_and_submit, {"strategy": "partial", "known": 0.5}),
    "physics_blind": (forgery.isolate_physics, {"strategy": "blind"}),
    "physics_single_copy":
        (forgery.isolate_physics, {"strategy": "single_copy"}),
    "physics_breidbart": (forgery.isolate_physics, {"strategy": "breidbart"}),
}


# --------------------------------------------------------------------------
# serialisation -- identical hygiene to generate_snapshots.py
# --------------------------------------------------------------------------

def _plain(obj: Any) -> Any:
    """Coerce numpy scalars and non-finite floats into JSON-native values."""
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
# deriving prose and parameters from the code
# --------------------------------------------------------------------------

_MARKER = re.compile(r"^``(\w+)``\s*$")


def family_blurbs() -> Dict[str, str]:
    """Parse the per-family definition list out of the package docstring.

    The ``qds.attacks`` docstring describes each family as a reST definition
    item -- a ``\`\`family\`\``` marker on its own line followed by an indented
    body.  Those bodies are the authoritative one-paragraph family summaries,
    so they are the blurbs, read verbatim (only whitespace is collapsed).
    """
    doc = attacks.__doc__ or ""
    fams = set(attacks.FAMILIES)
    out: Dict[str, str] = {}
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        m = _MARKER.match(lines[i].strip())
        if m and m.group(1) in fams:
            key = m.group(1)
            i += 1
            buf: List[str] = []
            while i < len(lines) and lines[i].startswith("    ") \
                    and lines[i].strip():
                buf.append(lines[i].strip())
                i += 1
            out[key] = re.sub(r"\s+", " ", " ".join(buf)).strip()
        else:
            i += 1
    return out


def family_label(module) -> str:
    """A short human label, taken from the first line of the module docstring.

    Each attack module opens with ``Title: one-line summary`` (or, for the
    repudiation module, a title clause ended by a comma), so the text before
    the first colon -- or comma -- is the family's title.
    """
    first = (module.__doc__ or "").strip().splitlines()[0].strip()
    if ":" in first:
        return first.split(":", 1)[0].strip()
    if "," in first:
        return first.split(",", 1)[0].strip()
    return first


def params_for(family: str, variant: str) -> Dict[str, Any]:
    """The keyword knobs that fix this variant's canonical instance.

    Introspected from the implementing function's signature (defaults only),
    with the forgery strategy that the catalogue lambda bakes in applied on
    top, and ``seed=0`` recorded because that is the seed ``run_all`` mounts
    every attack at.
    """
    if family == "forgery":
        func, baked = _FORGERY_BINDING[variant]
    else:
        func, baked = getattr(FAMILY_MODULE[family], variant), {}
    params: Dict[str, Any] = {}
    for name, p in inspect.signature(func).parameters.items():
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.default is not inspect.Parameter.empty:
            params[name] = p.default
    params.update(baked)
    params.setdefault("seed", 0)
    return params


# --------------------------------------------------------------------------
# building the catalogue
# --------------------------------------------------------------------------

def build_catalogue(seed: int) -> Dict[str, Any]:
    blurbs = family_blurbs()
    families: List[Dict[str, Any]] = []
    n_variants = 0
    for family in attacks.FAMILIES:
        module = FAMILY_MODULE[family]
        variants: List[Dict[str, Any]] = []
        for key, call in attacks.CATALOGUE[family].items():
            # Mount the canonical instance exactly as run_all does; read the
            # attack's own name, description and designed expectation.
            result = call(seed=seed)
            variants.append({
                "key": key,
                "name": result.name,
                "description": result.description,
                "params": params_for(family, key),
                "expectation": result.expectation.to_dict(),
            })
            n_variants += 1
        families.append({
            "key": family,
            "label": family_label(module),
            "module": module.__name__,
            "blurb": blurbs.get(family, ""),
            "count": len(variants),
            "variants": variants,
        })
    return {
        "generated_at": time.time(),
        "seed": seed,
        "families": families,
        "totals": {"families": len(families), "variants": n_variants},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = os.path.normpath(os.path.join(_HERE, "..", "public", "data"))
    ap.add_argument("--out", default=default_out,
                    help="output directory (default: web/public/data)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed every attack is mounted at (default: 0)")
    args = ap.parse_args()

    print(f"engine   {_ENGINE}")
    print(f"output   {args.out}")
    print(f"mounting {sum(len(attacks.CATALOGUE[f]) for f in attacks.FAMILIES)} "
          f"attacks across {len(attacks.FAMILIES)} families at seed={args.seed}\n")

    catalogue = build_catalogue(args.seed)
    for fam in catalogue["families"]:
        print(f"  {fam['key']:<14} {fam['count']:>2} variants  "
              f"({fam['label']})")
    path = os.path.join(args.out, "attacks.json")
    size = write_json(path, catalogue)
    t = catalogue["totals"]
    print(f"\nwrote attacks.json: {t['families']} families, "
          f"{t['variants']} variants, {size // 1024}kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
