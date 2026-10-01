"""Mitigator base class + name registry + backend resolution.

Backend convention: every mitigator accepts ``backend in {"auto", "fairlearn",
"native"}``. ``auto`` tries the optional fairlearn backend and silently (one
log line, via core.backend) falls back to the native numpy/sklearn
implementation. The resolved backend + fallback flag is recorded in every
result so benchmark reports can label ``backend|fallback``.

Author: 晨星
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fairforge.core.backend import get_backend
from fairforge.core.errors import MitigatorError
from fairforge.core.types import Dataset

REGISTRY: dict[str, type["Mitigator"]] = {}


def register(name: str):
    """Class decorator adding a Mitigator to the global registry."""

    def _wrap(cls: type["Mitigator"]) -> type["Mitigator"]:
        if name in REGISTRY:
            raise MitigatorError(f"mitigator {name!r} already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return _wrap


@dataclass
class MitigatorResult:
    """What a mitigator produces: exactly one of the fields is meaningful."""

    X: np.ndarray | None = None  # pre-processing: transformed features
    sample_weight: np.ndarray | None = None  # pre-processing: reweighting
    y_pred: np.ndarray | None = None  # in/post-processing: predictions
    y_score: np.ndarray | None = None  # post-processing: group-aware scores
    metadata: dict[str, Any] = field(default_factory=dict)


class Mitigator(ABC):
    """Base class for all mitigation algorithms."""

    name: str = "base"

    #: constraint-aware algorithms map this to their fairness moment/objective
    CONSTRAINT_MOMENTS = {}  # overridden in subclasses that support constraints

    def __init__(
        self,
        backend: str = "auto",
        random_state: int = 42,
        constraint: str = "demographic_parity_diff",
        **params: Any,
    ) -> None:
        unknown = set(params) - set(self.valid_params())
        if unknown:
            raise MitigatorError(f"{self.name}: unknown params {sorted(unknown)}")
        self.backend_request = backend
        self.random_state = int(random_state)
        self.constraint = constraint
        self.params = dict(params)
        self.resolved_backend: str = "native"
        self.used_fallback: bool = True
        self.n_trials: int = 0
        self.metadata: dict[str, Any] = {}

    def valid_params(self) -> set[str]:
        """Hyperparameter names accepted by this mitigator (HPO search space)."""
        return set()

    def resolve_backend(self, optional_name: str = "fairlearn") -> None:
        resolved, fallback = get_backend(optional_name, self.backend_request)
        self.resolved_backend = resolved
        self.used_fallback = fallback

    @abstractmethod
    def fit(self, dataset: Dataset) -> "Mitigator": ...

    @abstractmethod
    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray: ...

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise MitigatorError(f"{self.name} is not fitted; call fit() first")


def create(
    mitigator: str,
    backend: str = "auto",
    random_state: int = 42,
    constraint: str = "demographic_parity_diff",
    **params: Any,
) -> Mitigator:
    """Instantiate a registered mitigator by name."""
    if mitigator not in REGISTRY:
        raise MitigatorError(f"unknown mitigator {mitigator!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[mitigator](
        backend=backend, random_state=random_state, constraint=constraint, **params
    )


def list_mitigators() -> list[str]:
    return sorted(REGISTRY)
