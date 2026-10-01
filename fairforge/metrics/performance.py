"""Performance metrics (numpy-only, no sklearn dependency in the hot path).

Author: 晨星
"""

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(y_true == y_pred))


def selection_rate(y_pred) -> float:
    y_pred = np.asarray(y_pred).astype(float).ravel()
    if len(y_pred) == 0:
        return np.nan
    return float(np.mean(y_pred))
