import time

import numpy as np


def permutation_max_statistics(
    partial_ranks,
    n_permutations=10_000,
    random_state=2026
):
    """Generate the empirical PMS maximum-statistic distribution.

    Rows of ``partial_ranks`` are genes and columns are samples. Each gene's
    rank residuals are independently permuted across samples. The maximum
    absolute pairwise correlation in each permutation forms the null
    distribution. The returned value retains the sign of the pair attaining
    that maximum.
    """
    partial_ranks = np.asarray(partial_ranks, dtype=float)
    if partial_ranks.ndim != 2:
        raise ValueError("partial_ranks must be a 2D array (genes x samples)")
    if partial_ranks.shape[0] < 2 or partial_ranks.shape[1] < 2:
        raise ValueError("PMS requires at least two genes and two samples")
    if not np.isfinite(partial_ranks).all():
        raise ValueError("partial_ranks must contain only finite values")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")

    centered_ranks = partial_ranks - partial_ranks.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(centered_ranks, axis=1)
    has_variation = row_norms > 0
    pair_rows, pair_columns = np.triu_indices(partial_ranks.shape[0], k=1)
    valid_pairs = has_variation[pair_rows] & has_variation[pair_columns]
    if not valid_pairs.any():
        raise ValueError("PMS requires at least one pair with non-constant rank residuals")

    standardized_ranks = np.zeros_like(centered_ranks)
    standardized_ranks[has_variation] = (
        centered_ranks[has_variation] / row_norms[has_variation, np.newaxis]
    )

    rng = np.random.default_rng(random_state)
    signed_max_statistics = np.empty(n_permutations)
    batch_size = 32
    n_genes, n_samples = partial_ranks.shape
    valid_rows = pair_rows[valid_pairs]
    valid_columns = pair_columns[valid_pairs]
    n_valid_pairs = int(valid_pairs.sum())
    report_every = max(1, n_permutations // 10)
    started_at = time.perf_counter()
    print(
        f"PMS: starting {n_permutations:,} permutations for "
        f"{n_genes:,} genes ({n_valid_pairs:,} valid pairs).",
        flush=True
    )

    for start in range(0, n_permutations, batch_size):
        batch_count = min(batch_size, n_permutations - start)
        permutation_order = np.argsort(rng.random((batch_count, n_genes, n_samples)), axis=2)
        permuted_ranks = np.take_along_axis(
            standardized_ranks[np.newaxis, :, :],
            permutation_order,
            axis=2
        )
        permuted_correlations = permuted_ranks @ np.swapaxes(permuted_ranks, 1, 2)
        pair_correlations = permuted_correlations[:, valid_rows, valid_columns]
        max_pair_indices = np.argmax(np.abs(pair_correlations), axis=1)
        signed_max_statistics[start:start + batch_count] = np.take_along_axis(
            pair_correlations,
            max_pair_indices[:, np.newaxis],
            axis=1
        ).ravel()
        completed = start + batch_count
        if completed == n_permutations or completed // report_every > start // report_every:
            elapsed = time.perf_counter() - started_at
            estimated_total = elapsed * n_permutations / completed
            remaining = max(0.0, estimated_total - elapsed)
            print(
                f"PMS: {completed:,}/{n_permutations:,} permutations "
                f"({completed / n_permutations:.0%}) | "
                f"elapsed {elapsed:.1f}s | estimated remaining {remaining:.1f}s",
                flush=True
            )

    return signed_max_statistics


def pms_correction(
    partial_ranks,
    alpha=0.05,
    n_permutations=10_000,
    random_state=2026
):
    """Return max-statistic adjusted p-values and permutation maxima.

    Pair results follow the upper-triangle pair order for the input gene
    order and use a plus-one Monte Carlo correction.
    """
    partial_ranks = np.asarray(partial_ranks, dtype=float)
    if partial_ranks.ndim != 2:
        raise ValueError("partial_ranks must be a 2D array (genes x samples)")
    if partial_ranks.shape[0] < 2 or partial_ranks.shape[1] < 2:
        raise ValueError("PMS requires at least two genes and two samples")
    if not np.isfinite(partial_ranks).all():
        raise ValueError("partial_ranks must contain only finite values")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")

    centered_ranks = partial_ranks - partial_ranks.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(centered_ranks, axis=1)
    has_variation = row_norms > 0
    standardized_ranks = np.zeros_like(centered_ranks)
    standardized_ranks[has_variation] = (
        centered_ranks[has_variation] / row_norms[has_variation, np.newaxis]
    )
    pair_rows, pair_columns = np.triu_indices(partial_ranks.shape[0], k=1)
    valid_pairs = has_variation[pair_rows] & has_variation[pair_columns]
    if not valid_pairs.any():
        raise ValueError("PMS requires at least one pair with non-constant rank residuals")

    observed_correlations = standardized_ranks @ standardized_ranks.T
    observed_absolute = np.abs(
        observed_correlations[pair_rows[valid_pairs], pair_columns[valid_pairs]]
    )
    max_statistics = permutation_max_statistics(
        partial_ranks,
        n_permutations=n_permutations,
        random_state=random_state
    )
    sorted_max_statistics = np.sort(np.abs(max_statistics))
    adjusted_p_values = np.full(len(pair_rows), np.nan)
    exceedances = n_permutations - np.searchsorted(
        sorted_max_statistics,
        observed_absolute,
        side="left"
    )
    adjusted_p_values[valid_pairs] = (exceedances + 1) / (n_permutations + 1)
    significant = adjusted_p_values < alpha

    return adjusted_p_values, significant, max_statistics
