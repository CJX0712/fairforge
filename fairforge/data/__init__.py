"""Data layer: synthetic biased generators (with counterfactual truth) + optional Adult."""

from fairforge.data.adult import load_adult
from fairforge.data.generator import SCENARIOS, generate_biased_data, generate_scenario

__all__ = ["generate_biased_data", "generate_scenario", "SCENARIOS", "load_adult"]
