"""In-processing mitigators (fairness-aware training)."""

from fairforge.mitigation.inproc.exponentiated_gradient import ExponentiatedGradient
from fairforge.mitigation.inproc.grid_search import GridSearchReductions

__all__ = ["ExponentiatedGradient", "GridSearchReductions"]
