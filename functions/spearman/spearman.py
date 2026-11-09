"""Spearman correlation helpers."""

import numpy as np
from scipy.stats import spearmanr


def correlation(x, y):
    """Return Spearman rho, nominal p-value, and pairwise finite sample count."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n_samples = int(valid.sum())
    if n_samples < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan, np.nan, n_samples
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue), n_samples


def correlate_columns(x, y):
    """Correlation of every column of x with every column of y (both samples x variables, already
    centred to mean 0). Returns a (columns of x) x (columns of y) array. A column with no variation
    gives NaN.
    "No variation" includes residuals that are only rounding error (~1e-13): e.g. a gene's only
    isoform has exactly the same ranks as the gene total, so once the total is accounted for
    nothing is left, and correlating the leftover rounding noise would give a meaningless rho."""
    x_norm = np.linalg.norm(x, axis=0)
    y_norm = np.linalg.norm(y, axis=0)
    x_norm[x_norm < 1e-8] = np.nan
    y_norm[y_norm < 1e-8] = np.nan
    return (x / x_norm).T @ (y / y_norm)
