"""Pluggable simulation backends.

The protocol layer never touches numpy or Qiskit directly.  It talks to a
:class:`~qds.backends.base.Backend`, so the identical protocol code and the
identical detection engine run on either simulator.  That is what makes the
Qiskit path a *validation* of the fast path rather than a separate
implementation that might quietly disagree with it.

Available backends
------------------
``numpy`` (default)
    Exact density-matrix propagation implemented in :mod:`qds.linalg`.  No
    third-party dependency beyond numpy.  This is the backend used for every
    number in the shipped reports.
``qiskit``
    The same operations expressed with :mod:`qiskit.quantum_info`
    (``DensityMatrix``, ``Operator``, ``Kraus``).  Requires ``pip install
    qiskit``.  The Qiskit cross-validation tests in ``tests/test_backends.py``
    confirm the two backends agree to machine precision; they run when Qiskit
    is installed and skip otherwise.
"""

from __future__ import annotations

from typing import Dict, List

from .base import Backend, BackendError
from .numpy_dm import NumpyDensityMatrixBackend

__all__ = ["Backend", "BackendError", "get_backend", "available_backends",
           "NumpyDensityMatrixBackend"]

_REGISTRY: Dict[str, type] = {
    "numpy": NumpyDensityMatrixBackend,
}


def _load_qiskit() -> type:
    from .qiskit_dm import QiskitDensityMatrixBackend
    return QiskitDensityMatrixBackend


def get_backend(name: str = "numpy") -> Backend:
    """Instantiate a backend by name (``"numpy"`` or ``"qiskit"``)."""
    key = (name or "numpy").lower()
    if key in _REGISTRY:
        return _REGISTRY[key]()
    if key == "qiskit":
        try:
            cls = _load_qiskit()
        except ImportError as exc:
            raise BackendError(
                "the qiskit backend needs Qiskit installed: pip install qiskit"
            ) from exc
        _REGISTRY["qiskit"] = cls
        return cls()
    raise BackendError(f"unknown backend {name!r}; try one of {available_backends()}")


def available_backends() -> List[str]:
    """Backend names that can actually be instantiated right now."""
    out = ["numpy"]
    try:
        import qiskit  # noqa: F401
        out.append("qiskit")
    except Exception:
        pass
    return out
