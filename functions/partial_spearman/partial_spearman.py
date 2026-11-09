"""Partial Spearman helpers: removing a covariate from the ranks, and the p-value."""

import numpy as np
from scipy.stats import t as t_distribution


def ranks_within_group(gene_cpm, groups):
    """For each gene (row): rank its CPM across all samples, then subtract the mean rank of the
    sample's group (its region, or its disease). What is left says whether the gene is high or
    low *for that group*."""
    ranks = gene_cpm.rank(axis=1)   # ties get the average rank, same as spearmanr
    # Transpose so samples are rows, group the samples, get each group's mean rank
    group_mean_ranks = ranks.T.groupby(groups).transform("mean").T
    return ranks - group_mean_ranks


def partial_p_value(rho, n_samples, n_covariates):
    """Two-sided p-value for a partial correlation: t-test with n - 2 - k degrees of freedom."""
    degrees_of_freedom = n_samples - 2 - n_covariates
    t_statistic = rho * np.sqrt(degrees_of_freedom / (1 - rho ** 2))
    return 2 * t_distribution.sf(abs(t_statistic), degrees_of_freedom)


def residuals(values, covariates):
    """values: samples x variables. Fit every column on the covariates (least squares) and return
    what the covariates can't explain. covariates includes a column of 1s, so the residuals have
    mean 0."""
    coefficients = np.linalg.lstsq(covariates, values, rcond=None)[0]
    return values - covariates @ coefficients
