"""Synthetic biased data generator with counterfactual ground truth.

Three configurable discrimination mechanisms, applied AFTER the unbiased
counterfactual label ``y_cf`` is drawn:

1. ``delta``  - direct discrimination: the privileged group's label logits
                are shifted by ``+delta`` (and unprivileged by ``-delta``).
2. ``proxy_strength`` - indirect (proxy) discrimination: a feature strongly
                correlated with A enters the label-generating process, so
                discrimination flows through X -> y, invisible in A alone.
3. ``sampling_bias`` - historical sampling bias: unprivileged *favorable*
                samples (y_cf == 1) are dropped with probability rho.

Returns ``(X, A, y, y_cf)`` inside a :class:`Dataset`; ``y_cf`` is what y
*would have been* without mechanisms 1 & 2 (mechanism 3 does not change
labels, only who survives sampling).

Author: 晨星
"""

from dataclasses import dataclass

import numpy as np

from fairforge.core.errors import DataError
from fairforge.core.types import Dataset


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class BiasParams:
    delta: float = 0.0
    proxy_strength: float = 0.0
    sampling_bias: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.sampling_bias < 1.0):
            raise DataError("sampling_bias must be in [0, 1)")
        if abs(self.delta) > 5.0 or abs(self.proxy_strength) > 5.0:
            raise DataError("delta / proxy_strength must be within [-5, 5] (logit scale)")


# Three benchmark scenarios (seed=42), per architecture spec (5):
#   mild_direct      : light direct + medium indirect (proxy) discrimination
#   strong_sampling  : strong historical sampling bias
#   mixed            : moderate mix of all three mechanisms
SCENARIOS: dict[str, BiasParams] = {
    "mild_direct": BiasParams(delta=0.9, proxy_strength=1.3, sampling_bias=0.0),
    "strong_sampling": BiasParams(delta=0.5, proxy_strength=1.0, sampling_bias=0.5),
    "mixed": BiasParams(delta=0.7, proxy_strength=1.0, sampling_bias=0.4),
}


def generate_biased_data(
    n_per_group: int = 600,
    seed: int = 42,
    feature_dim: int = 4,
    delta: float = 0.0,
    proxy_strength: float = 0.0,
    sampling_bias: float = 0.0,
    scenario: str = "custom",
) -> Dataset:
    """Generate a biased binary-classification dataset with counterfactual labels.

    Guarantees at least ``n_per_group`` samples in each sensitive group after
    the sampling-bias drop (uses oversampling internally).
    """
    params = BiasParams(delta=delta, proxy_strength=proxy_strength, sampling_bias=sampling_bias)
    if n_per_group < 1 or feature_dim < 2:
        raise DataError("n_per_group >= 1 and feature_dim >= 2 required")

    rng = np.random.default_rng(seed)
    # Oversample to survive the unprivileged-favorable drop.
    n_draw = int(n_per_group * 2 / max(1e-6, (1.0 - params.sampling_bias / 2.0)))
    n_draw = min(n_draw, 200_000)

    while True:
        A = rng.binomial(1, 0.5, size=n_draw)
        skill = rng.normal(0.0, 1.0, size=n_draw)
        noise_y = rng.normal(0.0, 0.8, size=n_draw)

        # 1) Counterfactual *unbiased* label: driven only by skill.
        p_cf = _sigmoid(1.2 * skill + noise_y)
        y_cf = rng.binomial(1, p_cf)

        # 2) Observed label = counterfactual + direct (delta on A) + proxy path.
        x_proxy = params.proxy_strength * (2.0 * A - 1.0) + rng.normal(0.0, 0.6, size=n_draw)
        logit_obs = np.log(np.clip(p_cf, 1e-6, 1 - 1e-6) / np.clip(1 - p_cf, 1e-6, 1))
        logit_obs = logit_obs + params.delta * (2.0 * A - 1.0) + 0.9 * x_proxy
        y = rng.binomial(1, _sigmoid(logit_obs))

        # 3) Historical sampling bias: drop unprivileged favorable samples.
        drop = (A == 0) & (y_cf == 1) & (rng.random(n_draw) < params.sampling_bias)
        keep = ~drop

        X = np.empty((n_draw, feature_dim))
        for j in range(feature_dim):
            X[:, j] = skill * rng.normal(1.0, 0.15) + rng.normal(0.0, 0.5, size=n_draw)
        X[:, 0] = x_proxy  # proxy feature visible in X
        X[:, 1] = X[:, 1] + 0.8 * skill

        A_k, y_k, y_cf_k, X_k = A[keep], y[keep], y_cf[keep], X[keep]
        n0, n1 = int((A_k == 0).sum()), int((A_k == 1).sum())
        if min(n0, n1) >= n_per_group:
            idx0 = np.where(A_k == 0)[0][:n_per_group]
            idx1 = np.where(A_k == 1)[0][:n_per_group]
            sel = np.sort(np.concatenate([idx0, idx1]))
            break

    ds = Dataset(
        X=X_k[sel],
        A=A_k[sel],
        y=y_k[sel],
        y_cf=y_cf_k[sel],
        feature_names=[f"x{j}" for j in range(feature_dim)],
        sensitive_name="group",
        scenario=scenario,
    )
    return ds


def generate_scenario(name: str, n_per_group: int = 600, seed: int = 42) -> Dataset:
    """Generate one of the three built-in benchmark scenarios."""
    if name not in SCENARIOS:
        raise DataError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    p = SCENARIOS[name]
    return generate_biased_data(
        n_per_group=n_per_group,
        seed=seed,
        delta=p.delta,
        proxy_strength=p.proxy_strength,
        sampling_bias=p.sampling_bias,
        scenario=name,
    )
