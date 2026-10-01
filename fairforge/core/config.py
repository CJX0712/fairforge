"""Runtime configuration with FAIRFORGE_* environment overrides.

Author: 晨星
"""

import os
from dataclasses import dataclass

from fairforge.core.errors import ConfigError

_ENV_PREFIX = "FAIRFORGE_"


@dataclass(frozen=True)
class Config:
    """Global runtime configuration. Immutable; build via ``Config.from_env()``."""

    seed: int = 42
    n_jobs: int = 1  # always 1: Windows multiprocessing is unsupported
    lambda_fairness: float = 1.0
    n_trials: int = 30
    timeout_s: float = 120.0
    n_per_group: int = 600

    def __post_init__(self) -> None:
        if self.n_jobs != 1:
            raise ConfigError("n_jobs must be 1 (Windows multiprocessing is unsupported)")
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        if self.lambda_fairness < 0:
            raise ConfigError("lambda_fairness must be >= 0")
        if self.n_trials <= 0 or self.timeout_s <= 0:
            raise ConfigError("n_trials and timeout_s must be positive")

    @classmethod
    def from_env(cls, environ=None) -> "Config":
        environ = os.environ if environ is None else environ
        kwargs: dict = {}
        mapping = {
            "SEED": ("seed", int),
            "N_JOBS": ("n_jobs", int),
            "LAMBDA_FAIRNESS": ("lambda_fairness", float),
            "N_TRIALS": ("n_trials", int),
            "TIMEOUT_S": ("timeout_s", float),
            "N_PER_GROUP": ("n_per_group", int),
        }
        for env_key, (field_name, cast) in mapping.items():
            key = _ENV_PREFIX + env_key
            if key in environ and environ[key] != "":
                try:
                    kwargs[field_name] = cast(environ[key])
                except ValueError as exc:
                    raise ConfigError(f"invalid value for {key}: {environ[key]!r}") from exc
        return cls(**kwargs)
