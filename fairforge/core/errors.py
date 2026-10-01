"""Typed error hierarchy. Codes E100~E500 by layer.

Author: 晨星
"""


class FairForgeError(Exception):
    """Base class for all FairForge errors."""

    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class ConfigError(FairForgeError):
    """Invalid configuration / environment override (E1xx)."""

    code = "E100"


class DataError(FairForgeError):
    """Data generation / loading problems (E2xx)."""

    code = "E200"


class MetricError(FairForgeError):
    """Fairness metric computation problems (E3xx)."""

    code = "E300"


class MitigatorError(FairForgeError):
    """Mitigation algorithm problems, unknown mitigator, bad fit state (E4xx)."""

    code = "E400"


class BackendError(FairForgeError):
    """Backend detection / import problems (E5xx)."""

    code = "E500"
