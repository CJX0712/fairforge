"""FairForge: a reproducible fairness audit & mitigation toolkit.

Author: 晨星
"""

__version__ = "0.1.0"
__author__ = "晨星"

from fairforge.core.backend import available_backends
from fairforge.core.config import Config
from fairforge.core.errors import (
    BackendError,
    ConfigError,
    DataError,
    FairForgeError,
    MetricError,
    MitigatorError,
)
from fairforge.core.types import Dataset, RunResult

__all__ = [
    "Dataset",
    "RunResult",
    "Config",
    "available_backends",
    "FairForgeError",
    "ConfigError",
    "DataError",
    "MetricError",
    "MitigatorError",
    "BackendError",
    "__version__",
]
