"""False discovery rate corrections.

The *_correction functions take p-values with no missing values and return
(adjusted p-values, significant). The *_adjust functions leave missing p-values missing and
return only the adjusted p-values.
"""

import numpy as np


# 4) Benjamini-Hochberg (BH) correction
def benjamini_hochberg_correction(p_values, q=0.05):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    ranked_adjusted_p_values = sorted_p_values * m / np.arange(1, m + 1)
    ranked_adjusted_p_values = np.minimum.accumulate(ranked_adjusted_p_values[::-1])[::-1]
    adjusted_p_values = np.empty(m)
    adjusted_p_values[sorted_indices] = np.minimum(ranked_adjusted_p_values, 1.0)

    significant = adjusted_p_values < q
    
    return adjusted_p_values, significant


# 5) Benjamini-Yekutieli (BY) correction
def benjamini_yekutieli_correction(p_values, q=0.05):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Calculate the harmonic number H_m
    H_m = sum(1.0 / i for i in range(1, m + 1))
    
    ranked_adjusted_p_values = sorted_p_values * m * H_m / np.arange(1, m + 1)
    ranked_adjusted_p_values = np.minimum.accumulate(ranked_adjusted_p_values[::-1])[::-1]
    adjusted_p_values = np.empty(m)
    adjusted_p_values[sorted_indices] = np.minimum(ranked_adjusted_p_values, 1.0)

    significant = adjusted_p_values < q
    
    return adjusted_p_values, significant


def benjamini_yekutieli_adjust(p_values):
    """Return BY-adjusted p-values in input order, leaving missing values missing."""
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")
    if not finite_values.size:
        raise ValueError("No finite p-values are available to adjust.")

    n_tests = finite_values.size
    order = np.argsort(finite_values)
    sorted_p_values = finite_values[order]
    harmonic_number = np.sum(1.0 / np.arange(1, n_tests + 1))
    ranked_adjusted = (
        sorted_p_values * n_tests * harmonic_number / np.arange(1, n_tests + 1)
    )
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]

    adjusted = np.full(values.shape, np.nan, dtype=float)
    adjusted_finite = np.empty(n_tests, dtype=float)
    adjusted_finite[order] = np.minimum(ranked_adjusted, 1.0)
    adjusted[finite] = adjusted_finite
    return adjusted


def benjamini_hochberg_adjust(p_values):
    """Return BH-adjusted p-values in input order, leaving missing values missing."""
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")
    if not finite_values.size:
        raise ValueError("No finite p-values are available to adjust.")

    n_tests = finite_values.size
    order = np.argsort(finite_values)
    sorted_p_values = finite_values[order]
    ranked_adjusted = sorted_p_values * n_tests / np.arange(1, n_tests + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]

    adjusted = np.full(values.shape, np.nan, dtype=float)
    adjusted_finite = np.empty(n_tests, dtype=float)
    adjusted_finite[order] = np.minimum(ranked_adjusted, 1.0)
    adjusted[finite] = adjusted_finite
    return adjusted


def adjust_p_values(p_values, method):
    """Dispatch an FDR adjustment by method name."""
    if method == "by":
        return benjamini_yekutieli_adjust(p_values)
    if method == "bh":
        return benjamini_hochberg_adjust(p_values)
    raise ValueError("MHT_METHOD must be 'by' or 'bh'.")
