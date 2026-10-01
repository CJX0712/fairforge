"""The single backend detection point for the whole project.

``available_backends()`` is the ONLY place that decides whether an optional
dependency (fairlearn / optuna) is importable. Everything else must ask here.
When an optional backend is missing, we log exactly one line and use the
native (pure numpy/sklearn) fallback implementation.

Author: 晨星
"""

import importlib
import logging

logger = logging.getLogger("fairforge")

_CACHE: dict[str, bool] | None = None


def _probe(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # ImportError or broken install -> treat as missing
        return False


def available_backends(refresh: bool = False) -> dict[str, bool]:
    """Return ``{"fairlearn": bool, "optuna": bool}`` (cached)."""
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = {"fairlearn": _probe("fairlearn"), "optuna": _probe("optuna")}
    return dict(_CACHE)


def get_backend(name: str, requested: str = "auto") -> tuple[str, bool]:
    """Resolve a backend request for an optional dependency ``name``.

    Returns ``(resolved_backend, used_fallback)`` where resolved_backend is
    either ``name`` (the optional library) or ``"native"``.
    """
    if requested not in ("auto", name, "native"):
        raise ValueError(f"backend must be 'auto', '{name}' or 'native', got {requested!r}")
    has = available_backends().get(name, False)
    if requested == "native" or (requested in ("auto", name) and not has):
        if requested != "native":
            logger.info(
                "optional backend '%s' unavailable -> falling back to native implementation", name
            )
        return "native", True
    return name, False


def backend_version(name: str) -> str | None:
    if not available_backends().get(name, False):
        return None
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", "unknown")


def reset_cache() -> None:
    """Invalidate the detection cache (used by tests / after installs)."""
    global _CACHE
    _CACHE = None
