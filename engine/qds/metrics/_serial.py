"""One job: make a number safe to drop into a JSON snapshot.

Every figure this package returns ends up in ``metrics.json`` and then on a
dashboard, so a stray ``inf`` or ``nan`` is not a cosmetic problem -- it breaks
the serialiser and the chart.  :func:`finite` is the single chokepoint every
public function pushes its floats through, with exactly the semantics the
detection layer already uses in ``qds.detect.thresholds._f``: non-finite becomes
``None`` (an unbounded exponent is honestly "no finite value", not ``1e308``),
and anything that will not cast to ``float`` is passed through untouched.
"""

from __future__ import annotations

import math

__all__ = ["finite"]


def finite(x: object) -> object:
    """Coerce ``x`` to a JSON-native finite ``float``, else ``None``.

    ``None`` in, ``None`` out.  A finite number casts to ``float``.  ``inf`` or
    ``nan`` becomes ``None`` rather than a value a JSON reader cannot represent.
    A non-numeric object (a string label, say) is returned unchanged.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None
