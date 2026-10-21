"""
Spearman correlation between every pair of genes in a chosen range (e.g. genes 10-40).

  1. Load the matrix, keep the 45 adult samples (fetal removed).
  2. Gene CPM = sum of all its isoforms' CPM, per sample.
     Optional filter (FILTER_LOW_EXPRESSION): keep only genes with CPM > MIN_CPM in at least
     MIN_FRACTION_SAMPLES of the adult samples (default: CPM > 1 in >= 80% -> 572 of 816 genes).
     Genes that are zero in most samples give mostly tied ranks, so their rho is unreliable.
  3. Sort the genes (by mean expression, or keep file order) and pick genes FIRST_GENE to LAST_GENE.
  4. For every pair of those genes, across the 45 samples:
       - Spearman rho and p-value (all samples pooled)
       - partial Spearman rho and p-value, accounting for brain region (DLPFC / Hippocampus / Caudate)
  5. Save a table (gene IDs, gene names, both rhos, both p-values) as CSV in Results/gene_spearman/.
  6. Optional plot (PLOT_REGION_EFFECT): normal rho vs partial rho, one dot per pair, saved next to the CSV.

Partial Spearman, accounting for region:
  - rank each gene's CPM across the samples (as Spearman does)
  - subtract the mean rank of the sample's region -> "how high is this gene for its region"
    (identical to regressing the ranks on region dummies and keeping the residuals)
  - correlate gene A's residuals with gene B's residuals
  - p-value from a t-test with n - 2 - k degrees of freedom, k = number of regions - 1 = 2
If the pooled rho is high but the partial rho is near 0, the correlation came from both genes
differing between regions, not from the genes varying together.

Note: Spearman(A, B) is exactly the same as Spearman(B, A) - correlation has no direction.

Run from the Sapal-Speke-RP-2026 folder:  python gene_spearman.py
"""

import os
import re
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as t_distribution

from describe_data import load_matrix, gene_of


# --- Settings (change these, not the code below) ---------------------------
FIRST_GENE = 1           # first gene in the range (1 = first gene)
LAST_GENE = None         # last gene in the range (included); None = every gene (572 with the filter on)
ORDER_BY = "expression"  # "expression" = gene 1 is the most highly expressed gene
                         # "file"       = genes in the order they appear in the matrix

FILTER_LOW_EXPRESSION = True   # True = drop low-expressed genes BEFORE numbering them 1, 2, 3, ...
                               # (so with the filter on, LAST_GENE can go up to 572, not 816)
MIN_CPM = 1                    # a gene counts as "expressed" in a sample if its CPM is above this
MIN_FRACTION_SAMPLES = 0.8     # ... and it must be expressed in at least this fraction of samples

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
RESULTS_FOLDER = "Results/gene_spearman"

PLOT_REGION_EFFECT = True   # True = scatter of normal Spearman rho vs region-corrected (partial) rho
                            # False = turn the plot off (CSV is still saved)
MAX_DOTS = 5000             # above this many pairs, dots pile into a blob, so the plot shows
                            # density instead (hexagons coloured by how many pairs fall in each)


def read_gene_names(gtf_path):
    """G locus -> gene name, from the GTF "transcript" lines, which look like
      transcript_id "G249.3.nnc"; gene_name "SLC45A1"; ...
    A locus can contain transcripts with more than one gene name (e.g. overlapping genes);
    use the name with the most transcripts."""
    loci, gene_names = [], []
    with open(gtf_path) as file:
        for line in file:
            columns = line.split("\t")
            if len(columns) < 9 or columns[2] != "transcript":   # skip exon / CDS lines
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', columns[8]).group(1)
            name = re.search(r'gene_name "([^"]+)"', columns[8])
            loci.append(gene_of(transcript_id))
            gene_names.append(name.group(1) if name else "")
    # For each locus, value_counts() counts each name; index[0] is the most common one
    return pd.Series(gene_names).groupby(loci).agg(lambda found: found.value_counts().index[0])


def region_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the region is the text before the first "_"."""
    return sample_name.split("_")[0]


def ranks_within_region(gene_cpm, regions):
    """For each gene (row): rank its CPM across all samples, then subtract the mean rank of the
    sample's region. What is left says whether the gene is high or low *for that region*."""
    ranks = gene_cpm.rank(axis=1)   # ties get the average rank, same as spearmanr
    # Transpose so samples are rows, group the samples by region, get each region's mean rank
    region_mean_ranks = ranks.T.groupby(regions).transform("mean").T
    return ranks - region_mean_ranks


def partial_p_value(rho, n_samples, n_covariates):
    """Two-sided p-value for a partial correlation: t-test with n - 2 - k degrees of freedom."""
    degrees_of_freedom = n_samples - 2 - n_covariates
    t_statistic = rho * np.sqrt(degrees_of_freedom / (1 - rho ** 2))
    return 2 * t_distribution.sf(abs(t_statistic), degrees_of_freedom)


def plot_region_effect(results, path, gene_range):
    """Normal Spearman rho (x) against partial rho accounting for region (y), one point per gene pair.
    Points on the dashed y = x line are unaffected by region. Points pulled towards y = 0 had a
    correlation that came (partly) from both genes differing between regions.
    Up to MAX_DOTS pairs: one dot each. More than that: a hexbin (density) plot instead."""
    fig = plt.figure(figsize=(7.5 if len(results) > MAX_DOTS else 6.5, 6.5))   # wider if a colour bar is added
    if len(results) > MAX_DOTS:
        # Split the square into hexagons; colour = number of pairs in each (log scale, as most
        # pairs sit near 0). mincnt=1 leaves empty hexagons white.
        hexagons = plt.hexbin(results["spearman_rho"], results["partial_rho_region"], gridsize=80,
                              extent=(-1, 1, -1, 1), bins="log", mincnt=1, cmap="Blues", zorder=2)
        plt.colorbar(hexagons, label="Number of gene pairs (log scale)")
    else:
        plt.scatter(results["spearman_rho"], results["partial_rho_region"], s=18, color="#2a78d6",
                    alpha=0.6, edgecolor="white", linewidth=0.4, zorder=2)
    # Reference lines drawn on top (zorder=3) so they stay visible over the hexagons
    plt.axhline(0, color="#d0d0d0", linewidth=0.8, zorder=3)
    plt.axvline(0, color="#d0d0d0", linewidth=0.8, zorder=3)
    plt.plot([-1, 1], [-1, 1], color="grey", linestyle="--", linewidth=1,
             label="y = x (region has no effect)", zorder=3)

    # Label the 3 pairs that region changed the most, using gene names (G locus if no name)
    change = (results["spearman_rho"] - results["partial_rho_region"]).abs()
    for _, row in results.loc[change.nlargest(3).index].iterrows():
        label = f"{row['gene_a_name'] or row['gene_a']}–{row['gene_b_name'] or row['gene_b']}"
        # Dots on the right get their label on the left, so it doesn't run off the plot
        on_right = row["spearman_rho"] > 0.5
        plt.annotate(label, (row["spearman_rho"], row["partial_rho_region"]),
                     xytext=(-5 if on_right else 5, 5), textcoords="offset points",
                     ha="right" if on_right else "left", fontsize=8)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Spearman rho (all adult samples pooled)")
    plt.ylabel("Partial Spearman rho (accounting for brain region)")
    plt.title(f"Effect of brain region on gene-gene correlation\n"
              f"Genes {gene_range}, {len(results):,} pairs, "
              f"median |change| = {change.median():.2f}")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()                                  # stops labels being cut off

    fig.savefig(path, dpi=200)
    print(f"Saved figure: {path}")
    plt.show()                                          # opens the figure window (script pauses until closed)


def main():
    # ---- 1. Load, adults only ----
    df = load_matrix(MATRIX_PATH)
    df = df[[name for name in df.columns if "_Fetal_" not in name]]

    # ---- 2. Gene CPM = sum of each gene's isoforms ----
    # groupby with sort=False keeps genes in the order they first appear in the file
    gene_cpm = df.groupby([gene_of(i) for i in df.index], sort=False).sum()
    names = read_gene_names(GTF_PATH)   # G locus -> gene name, e.g. G35694 -> NRXN1

    # ---- 2b. Drop low-expressed genes (turn off with FILTER_LOW_EXPRESSION = False) ----
    if FILTER_LOW_EXPRESSION:
        # (gene_cpm > MIN_CPM) is True/False per sample; .mean(axis=1) = fraction of samples that are True
        fraction_expressed = (gene_cpm > MIN_CPM).mean(axis=1)
        n_before = len(gene_cpm)
        gene_cpm = gene_cpm[fraction_expressed >= MIN_FRACTION_SAMPLES]
        print(f"Expression filter (CPM > {MIN_CPM} in >= {MIN_FRACTION_SAMPLES:.0%} of samples): "
              f"kept {len(gene_cpm)} of {n_before} genes")

    # ---- 3. Order the genes and pick the range ----
    if ORDER_BY == "expression":
        gene_cpm = gene_cpm.loc[gene_cpm.mean(axis=1).sort_values(ascending=False).index]
    elif ORDER_BY != "file":
        raise ValueError('ORDER_BY must be "expression" or "file"')
    # Python counts from 0 and excludes the end, so genes 10-40 are rows 9 to 39 -> iloc[9:40]
    # (LAST_GENE = None -> iloc[FIRST_GENE - 1:None] runs to the last gene)
    selected = gene_cpm.iloc[FIRST_GENE - 1:LAST_GENE]
    gene_range = f"{FIRST_GENE}-{FIRST_GENE + len(selected) - 1}"   # e.g. "1-572", used in titles/file names
    n_pairs = len(selected) * (len(selected) - 1) // 2
    print(f"Genes {gene_range} (ordered by {ORDER_BY}): {len(selected)} genes, "
          f"{n_pairs:,} pairs, {df.shape[1]} adult samples")

    # Region of each sample, in the same order as the columns
    regions = [region_of(name) for name in selected.columns]
    n_covariates = len(set(regions)) - 1   # 3 regions -> 2 dummy variables (one region is the baseline)
    print("Samples per region:", pd.Series(regions).value_counts().to_dict())
    # Done once for all genes, then reused for every pair
    region_ranks = ranks_within_region(selected, regions)

    # ---- 4. Spearman and partial Spearman for every pair ----
    # combinations(..., 2) gives each pair once: (g1, g2), (g1, g3), ... but never (g2, g1)
    rows = []
    for pair_number, (gene_a, gene_b) in enumerate(combinations(selected.index, 2), start=1):
        if pair_number % 20000 == 0:   # progress, so a long run doesn't look frozen
            print(f"  {pair_number:,} / {n_pairs:,} pairs done")
        rho, p_value = spearmanr(selected.loc[gene_a], selected.loc[gene_b])
        # Partial Spearman = ordinary (Pearson) correlation of the within-region ranks
        partial_rho = np.corrcoef(region_ranks.loc[gene_a], region_ranks.loc[gene_b])[0, 1]
        rows.append({"gene_a": gene_a, "gene_a_name": names.get(gene_a, ""),
                     "gene_b": gene_b, "gene_b_name": names.get(gene_b, ""),
                     "spearman_rho": rho, "p_value": p_value,
                     "partial_rho_region": partial_rho,
                     "partial_p_value": partial_p_value(partial_rho, selected.shape[1], n_covariates),
                     "n_samples": selected.shape[1]})

    # Strongest correlations (positive or negative) first
    results = pd.DataFrame(rows)
    results = results.reindex(results["spearman_rho"].abs().sort_values(ascending=False).index)
    print()
    print(results.head(20).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ---- 5. Save ----
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # "filtered" in the file name, because gene numbers mean different genes with the filter on/off
    filter_tag = "_filtered" if FILTER_LOW_EXPRESSION else ""
    path = os.path.join(RESULTS_FOLDER,
                        f"gene_spearman_{gene_range}_{ORDER_BY}{filter_tag}_{timestamp}.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # ---- 6. Plot: normal vs region-corrected rho (turn off with PLOT_REGION_EFFECT = False) ----
    if PLOT_REGION_EFFECT:
        plot_region_effect(results, path.replace(".csv", "_region_effect.png"), gene_range)


if __name__ == "__main__":
    main()
