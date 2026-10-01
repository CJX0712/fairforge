"""Optuna-backed hyperparameter tuner with a native random-search fallback.

The tuner itself is backend-agnostic: callers pass an objective function
``f(params: dict) -> float`` (maximize) and a search space. If optuna is
unavailable we degrade to seeded random search - same interface, same output
shape, deterministic under ``seed``.

Author: 晨星
"""

import logging
import time

import numpy as np

from fairforge.core.backend import available_backends
from fairforge.core.config import Config

logger = logging.getLogger("fairforge")

# space entry forms:
#   ("float", low, high) | ("int", low, high) | ("categorical", [choices...])


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    """Draw one random point from the space (native fallback path)."""
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float":
            params[name] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "int":
            params[name] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "categorical":
            choices = spec[1]
            params[name] = choices[int(rng.integers(0, len(choices)))]
        else:
            raise ValueError(f"unknown space kind {kind!r} for {name!r}")
    return params


class Tuner:
    """Small maximize-objective tuner. Records every trial for Pareto analysis."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.trials: list[dict] = []

    def tune(
        self, objective, space: dict, n_trials: int | None = None, timeout_s: float | None = None
    ) -> list[dict]:
        n_trials = int(n_trials or self.config.n_trials)
        timeout_s = float(timeout_s if timeout_s is not None else self.config.timeout_s)
        self.trials = []
        if available_backends().get("optuna", False):
            return self._tune_optuna(objective, space, n_trials, timeout_s)
        logger.info("optuna unavailable -> seeded random-search fallback")
        return self._tune_random(objective, space, n_trials, timeout_s)

    def _record(self, params: dict, value: float) -> None:
        self.trials.append({"params": dict(params), "value": float(value)})

    def _tune_optuna(self, objective, space, n_trials, timeout_s) -> list[dict]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _obj(trial: "optuna.Trial") -> float:
            params = {}
            for name, spec in space.items():
                kind = spec[0]
                if kind == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif kind == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                else:
                    params[name] = trial.suggest_categorical(name, spec[1])
            value = float(objective(params))
            self._record(params, value)
            return value

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(_obj, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
        return self.trials

    def _tune_random(self, objective, space, n_trials, timeout_s) -> list[dict]:
        rng = np.random.default_rng(self.config.seed)
        start = time.monotonic()
        for _ in range(n_trials):
            if time.monotonic() - start > timeout_s:
                break
            params = sample_params(space, rng)
            self._record(params, float(objective(params)))
        return self.trials
