"""Mitigation layer: registry, base class, pre/in/post algorithm families.

Importing this package registers ALL built-in mitigators (the algorithm
modules carry @register decorators at import time).
"""

from fairforge.mitigation.base import Mitigator, MitigatorResult, create, list_mitigators
from fairforge.mitigation.inproc import ExponentiatedGradient, GridSearchReductions
from fairforge.mitigation.post import GroupThreshold, RejectOption, ThresholdOptimizer
from fairforge.mitigation.pre import DisparateImpactRemover, Reweighing
from fairforge.mitigation.pre.none import NoMitigation

__all__ = [
    "Mitigator",
    "MitigatorResult",
    "create",
    "list_mitigators",
    "Reweighing",
    "DisparateImpactRemover",
    "NoMitigation",
    "ExponentiatedGradient",
    "GridSearchReductions",
    "ThresholdOptimizer",
    "GroupThreshold",
    "RejectOption",
]
