"""Canonical data containers.

Author: 晨星
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fairforge.core.errors import DataError


@dataclass
class Dataset:
    """A dataset with a binary sensitive attribute.

    Conventions (project-wide, do not change):
      * privileged group is ``A == 1``, unprivileged group is ``A == 0``.
      * ``y == 1`` is the favorable outcome.
      * ``y_cf`` is the *counterfactual unbiased* label (before discrimination
        mechanisms were applied); it is ground truth for generator diagnostics.
    """

    X: np.ndarray
    A: np.ndarray
    y: np.ndarray
    y_cf: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    sensitive_name: str = "sensitive"
    scenario: str = "custom"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.A = np.asarray(self.A).astype(int).ravel()
        self.y = np.asarray(self.y).astype(int).ravel()
        if self.y_cf is not None:
            self.y_cf = np.asarray(self.y_cf).astype(int).ravel()
        if self.X.ndim != 2:
            raise DataError(f"X must be 2-D, got shape {self.X.shape}")
        n = self.X.shape[0]
        if not (len(self.A) == len(self.y) == n):
            raise DataError(
                f"length mismatch: X has {n} rows, A has {len(self.A)}, y has {len(self.y)}"
            )
        if not set(np.unique(self.A)).issubset({0, 1}):
            raise DataError("A must be binary with values in {0, 1} (1=privileged, 0=unprivileged)")
        if not set(np.unique(self.y)).issubset({0, 1}):
            raise DataError("y must be binary with values in {0, 1} (1=favorable)")
        if not self.feature_names:
            self.feature_names = [f"x{i}" for i in range(self.X.shape[1])]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def group_mask(self, group: int) -> np.ndarray:
        return self.A == group

    def subset(self, mask: np.ndarray) -> "Dataset":
        mask = np.asarray(mask, dtype=bool)
        return Dataset(
            X=self.X[mask],
            A=self.A[mask],
            y=self.y[mask],
            y_cf=self.y_cf[mask] if self.y_cf is not None else None,
            feature_names=list(self.feature_names),
            sensitive_name=self.sensitive_name,
            scenario=self.scenario,
        )


@dataclass
class RunResult:
    """Outcome of a single mitigator run on a dataset."""

    mitigator: str
    backend: str
    used_fallback: bool
    accuracy: float
    metrics: dict[str, float]
    y_pred: np.ndarray | None = None
    n_trials: int = 0
    elapsed_s: float = 0.0
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)
