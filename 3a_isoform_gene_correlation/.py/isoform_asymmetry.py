"""
Isoform asymmetry between genes: does an isoform of gene A follow the total of gene B more than
gene A's own total can explain - and is it the same the other way round?

  1. Load the matrix, keep the 45 adult samples (the 4 "_Fetal_" columns are dropped straight away,
     so every later step uses adults only - same as gene_spearman.py).
  2. Gene CPM = sum of all its isoforms' CPM, per sample.
  3. Keep only expressed genes: CPM > MIN_CPM in at least MIN_FRACTION_SAMPLES of the adult samples
     (default CPM > 1 in >= 80% -> 572 of 816 genes, the same filter as gene_spearman.py).
  4. Keep only isoforms that make up at least MIN_SHARE (1.5%) of their gene, on average across the
     adult samples. Share in a sample = isoform CPM / gene CPM; samples where the gene is 0 are skipped.
  5. For every pair of kept genes (A, B), both directions:
       each kept isoform of A against the total of B
       each kept isoform of B against the total of A
     (iso_gene says which way round a row is: the gene the isoform belongs to)
     giving per row:
       rho_raw      - plain Spearman (isoform vs partner gene total, all adult samples pooled)
       rho_partial         - partial Spearman accounting for the isoform's own gene total
       p_partial           - its p-value
       rho_partial_region  - partial Spearman accounting for the own gene total AND brain region
                             (DLPFC / Hippocampus / Caudate)
       p_partial_region    - its p-value
  6. Sort so each gene pair is one block (pair_id, then iso_gene, then isoform) and save in
     3a_isoform_gene_correlation/CSV/:
       .parquet - the full table (millions of rows; load with pd.read_parquet, needs pyarrow)
       .csv     - only the rows with p_partial OR p_partial_region < SIGNIFICANT_P, small enough
                  to open in Excel

Partial Spearman, accounting for the own gene total:
  - rank every isoform and every gene total across the samples (as Spearman does)
  - fit each rank vector on the rank of the isoform's own gene total (straight line, least squares)
    and keep what is left (the residuals)
  - do this for the isoform AND for the partner gene's total, using the same gene A total
  - correlate the two sets of residuals
  - p-value from a t-test with n - 2 - k degrees of freedom, k = 1 (own gene total)
So rho_raw = isoform of A vs gene B, and rho_partial = the same after removing the part that is
just gene A vs gene B. A high rho_partial means the isoform rises and falls with gene B in a way
that gene A's total does not explain - i.e. something specific to that isoform.
rho_partial_region is worked out the same way, but the fit also includes region (as 0/1 columns,
one region as the baseline), so k = 1 + 2 = 3. If rho_partial is high but rho_partial_region is
near 0, the link came from the isoform and gene B both differing between regions.

pair_id is the two gene names in alphabetical order ("GENE1__GENE2"), so both directions of a pair
share one ID. A few gene names
belong to more than one locus (e.g. ESR1 has 3); those get the locus added ("ESR1_G60428") so
every gene has its own label.

Run with:
    python 3a_isoform_gene_correlation/.py/isoform_asymmetry.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.data_loading.expression_data import load_matrix, gene_of, read_gene_names, region_of
from functions.partial_spearman.partial_spearman import partial_p_value, residuals
from functions.spearman.spearman import correlate_columns


# --- Settings (change these, not the code below) ---------------------------
MIN_CPM = 1                  # a gene counts as "expressed" in a sample if its CPM is above this
MIN_FRACTION_SAMPLES = 0.8   # ... and it must be expressed in at least this fraction of samples
MIN_SHARE = 0.015            # keep an isoform if it is at least this share of its gene on average (0.015 = 1.5%)
SIGNIFICANT_P = 0.05         # rows with p_partial below this also go in the small CSV

MATRIX_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv")
GTF_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf")
RESULTS_FOLDER = str(ROOT_FOLDER / "3a_isoform_gene_correlation" / "CSV")


def gene_labels(names):
    """G locus -> label used in the table: the gene name, or "name_locus" if that name belongs to
    more than one locus, so two different genes never share a label."""
    repeated = names.duplicated(keep=False)   # True for every locus whose name appears more than once
    return pd.Series(np.where(repeated, names + "_" + names.index, names), index=names.index)


def main():
    # ---- 1. Load, adults only ----
    df = load_matrix(MATRIX_PATH)
    df = df[[name for name in df.columns if "_Fetal_" not in name]]
    n_samples = df.shape[1]

    # ---- 2. Gene CPM = sum of each gene's isoforms ----
    locus_of_isoform = pd.Series([gene_of(i) for i in df.index], index=df.index)
    gene_cpm = df.groupby(locus_of_isoform, sort=False).sum()
    labels = gene_labels(read_gene_names(GTF_PATH))   # G locus -> e.g. "NRXN1"

    # ---- 3. Expressed genes only ----
    # (gene_cpm > MIN_CPM) is True/False per sample; .mean(axis=1) = fraction of samples that are True
    fraction_expressed = (gene_cpm > MIN_CPM).mean(axis=1)
    kept_genes = list(gene_cpm.index[fraction_expressed >= MIN_FRACTION_SAMPLES])
    print(f"Expression filter (CPM > {MIN_CPM} in >= {MIN_FRACTION_SAMPLES:.0%} of samples): "
          f"kept {len(kept_genes)} of {len(gene_cpm)} genes")

    # ---- 4. Isoforms that are at least MIN_SHARE of their gene on average ----
    isoforms = df[locus_of_isoform.isin(kept_genes)]
    isoform_locus = locus_of_isoform[isoforms.index]
    # Each isoform's row divided by its gene's row; gene = 0 gives 0/0 = NaN, which .mean() skips
    own_gene_cpm = gene_cpm.loc[isoform_locus].set_axis(isoforms.index)
    share = (isoforms / own_gene_cpm.replace(0, np.nan)).mean(axis=1)
    keep = share >= MIN_SHARE
    print(f"Isoforms of these genes: kept {keep.sum()} of {len(isoforms)} "
          f"(mean share >= {MIN_SHARE:.1%})")
    isoforms, isoform_locus, share = isoforms[keep], isoform_locus[keep], share[keep]
    gene_cpm = gene_cpm.loc[kept_genes]
    n_nonzero = (isoforms > 0).sum(axis=1)
    mean_cpm = isoforms.mean(axis=1)

    # ---- 5. Ranks and region ----
    # Ranks across samples (ties get the average rank, as spearmanr does), transposed to
    # samples x variables for the least-squares fits
    isoform_ranks = isoforms.rank(axis=1).T
    gene_ranks = gene_cpm.rank(axis=1).T
    intercept = np.ones((n_samples, 1))
    # Region as 0/1 columns, dropping one region (the baseline)
    regions = pd.Series([region_of(name) for name in df.columns])
    print("Samples per region:", regions.value_counts().to_dict())
    region_columns = pd.get_dummies(regions, drop_first=True).to_numpy(dtype=float)
    n_covariates = 1                                      # own gene total
    n_covariates_region = 1 + region_columns.shape[1]     # own gene total + 2 region columns = 3

    # Centred ranks, for the plain Spearman (rho_raw)
    isoform_centred = isoform_ranks - isoform_ranks.mean()
    gene_centred = gene_ranks - gene_ranks.mean()

    # ---- 6. Every gene A: its isoforms against every other gene's total ----
    # One block of rows per gene A (its isoforms x all partner genes), joined at the end
    blocks = []
    for number, locus_a in enumerate(kept_genes, start=1):
        if number % 100 == 0:   # progress, so a long run doesn't look frozen
            print(f"  {number} / {len(kept_genes)} genes done")
        isoforms_of_a = isoform_locus.index[isoform_locus == locus_a]
        if len(isoforms_of_a) == 0:
            continue
        partners = [locus for locus in kept_genes if locus != locus_a]

        # Covariates for this gene: 1s + rank of gene A's own total (+ region for the second one)
        isoform_values = isoform_ranks[isoforms_of_a].to_numpy()
        partner_values = gene_ranks[partners].to_numpy()
        covariates = np.column_stack([intercept, gene_ranks[locus_a].to_numpy()])
        covariates_region = np.column_stack([covariates, region_columns])
        partial_rhos = correlate_columns(residuals(isoform_values, covariates),
                                         residuals(partner_values, covariates))
        partial_region_rhos = correlate_columns(residuals(isoform_values, covariates_region),
                                                residuals(partner_values, covariates_region))
        raw_rhos = correlate_columns(isoform_centred[isoforms_of_a].to_numpy(),
                                     gene_centred[partners].to_numpy())

        # Rows go isoform 1 x every partner, isoform 2 x every partner, ...: np.repeat repeats each
        # isoform once per partner, np.tile repeats the partner list once per isoform, and
        # .ravel() reads the rho arrays (isoforms x partners) in that same order
        n_partners = len(partners)
        label_a = labels[locus_a]
        partner_labels = labels[partners].to_numpy()
        partner_column = np.tile(partner_labels, len(isoforms_of_a))
        a_first = label_a < partner_column   # is gene A first alphabetically in the pair?
        first = np.where(a_first, label_a, partner_column)
        second = np.where(a_first, partner_column, label_a)
        partial_column = partial_rhos.ravel()
        partial_region_column = partial_region_rhos.ravel()
        blocks.append(pd.DataFrame({
            "pair_id": pd.Series(first) + "__" + pd.Series(second),
            "iso_gene": label_a,
            "isoform_id": np.repeat(isoforms_of_a, n_partners),
            "partner_gene": partner_column,
            "partner_gene_id": np.tile(partners, len(isoforms_of_a)),   # its G-number, e.g. G35694
            "n_samples": n_samples,
            "n_nonzero": np.repeat(n_nonzero[isoforms_of_a].to_numpy(), n_partners),
            "iso_mean_cpm": np.repeat(mean_cpm[isoforms_of_a].to_numpy(), n_partners),
            "iso_share_mean": np.repeat(share[isoforms_of_a].to_numpy(), n_partners),
            "rho_raw": raw_rhos.ravel(),
            "rho_partial": partial_column,
            "p_partial": partial_p_value(partial_column, n_samples, n_covariates),
            "rho_partial_region": partial_region_column,
            "p_partial_region": partial_p_value(partial_region_column, n_samples, n_covariates_region),
        }))

    # Each gene pair as one block: the first gene's isoforms, then the second gene's
    results = pd.concat(blocks, ignore_index=True)
    results = results.sort_values(["pair_id", "iso_gene", "isoform_id"], ignore_index=True)
    print(f"\n{len(results):,} rows ({results['pair_id'].nunique():,} gene pairs)")
    print(results.head(20).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ---- 7. Save: full table as parquet, significant rows as CSV ----
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(RESULTS_FOLDER,
                        f"isoform_asymmetry_all_adults_share{MIN_SHARE}_owntotal_and_region_{timestamp}")
    results.to_parquet(path + ".parquet", index=False)
    # A row goes in the CSV if either partial correlation is significant
    significant = results[(results["p_partial"] < SIGNIFICANT_P)
                          | (results["p_partial_region"] < SIGNIFICANT_P)]
    significant.to_csv(path + f"_p{SIGNIFICANT_P}.csv", index=False)
    print(f"\nSaved: {path}.parquet ({len(results):,} rows)")
    print(f"Saved: {path}_p{SIGNIFICANT_P}.csv ({len(significant):,} rows with p_partial or "
          f"p_partial_region < {SIGNIFICANT_P})")
    print(f"  p_partial < {SIGNIFICANT_P}: {(results['p_partial'] < SIGNIFICANT_P).sum():,} rows;  "
          f"p_partial_region < {SIGNIFICANT_P}: {(results['p_partial_region'] < SIGNIFICANT_P).sum():,} rows")


if __name__ == "__main__":
    main()
