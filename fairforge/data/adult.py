"""Optional Adult (UCI) loader. Network failures are logged and skipped (returns None).

Author: 晨星
"""

import logging

import numpy as np

from fairforge.core.types import Dataset

logger = logging.getLogger("fairforge")


def load_adult() -> Dataset | None:
    """Load the UCI Adult dataset via OpenML. Returns None on any failure.

    Sensitive attribute: sex (male=1 privileged, female=0 unprivileged).
    Target: income > 50K -> 1.
    """
    try:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = raw.data
        target = (raw.target == ">50K").astype(int)
        sex = df["sex"].astype(str)
        A = np.where(sex.str.lower().str.startswith("male"), 1, 0)
        num = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        num = np.nan_to_num(num, nan=0.0)
        return Dataset(
            X=num,
            A=A,
            y=np.asarray(target, dtype=int),
            feature_names=list(df.select_dtypes(include=[np.number]).columns),
            sensitive_name="sex",
            scenario="adult",
        )
    except Exception as exc:  # network unreachable / parse issue -> optional
        logger.info("Adult dataset unavailable (%s) -> skipping optional comparison", exc)
        return None
