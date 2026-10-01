"""Metrics layer: unified fairness metrics + performance."""

from fairforge.metrics import fairness, performance
from fairforge.metrics.fairness import fairness_report

__all__ = ["fairness", "performance", "fairness_report"]
