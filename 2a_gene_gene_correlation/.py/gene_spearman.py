"""
Spearman correlation between every pair of genes in a chosen range (e.g. genes 10-40).

  1. Load the matrix, keep the 45 adult samples (fetal removed).
     The matrix has 49 samples; the 4 fetal ones (all DLPFC Control: R3450, R3427, R4269, R3461)
     are dropped straight after loading - any column with "_Fetal_" in its name - so every later
     step (gene sums, expression filter, ordering, all correlations) uses adults only.
     Fetal brain expression differs hugely from adult and they are all one region / diagnosis,
     so keeping them would create strong correlations that only reflect fetal vs adult.
  2. Gene CPM = sum of all its isoforms' CPM, per sample.
     Optional filter (FILTER_LOW_EXPRESSION): keep only genes with CPM > MIN_CPM in at least
     MIN_FRACTION_SAMPLES of the adult samples (default: CPM > 1 in >= 80% -> 572 of 816 genes).
     Genes that are zero in most samples give mostly tied ranks, so their rho is unreliable.
  3. Sort the genes (by mean expression, or keep file order) and pick genes FIRST_GENE to LAST_GENE.
  4. For every pair of those genes, across the 45 samples, depending on the analysis chosen:
       "spearman" - Spearman rho and p-value only (all samples pooled)
       "region"   - Spearman + partial Spearman accounting for brain region (DLPFC / Hippocampus / Caudate)
       "disease"  - Spearman + partial Spearman accounting for disease (Control / MDD / Bipolar / Schizo),
                    ignoring region
       "all"      - Spearman + both partial correlations, in one table
  5. Save a settings-specific CSV in 2a_gene_gene_correlation/CSV/. Running the same settings again
     recalculates and overwrites it. Options 1-4 have separate output names.
  6. Optional plot (PLOT_PARTIAL_EFFECT): normal rho vs partial rho, one dot per pair, saved in
     2a_gene_gene_correlation/Figures/ - one plot per partial correlation (none for "spearman" alone).

Partial Spearman, accounting for a grouping (region or disease):
  - rank each gene's CPM across the samples (as Spearman does)
  - subtract the mean rank of the sample's group -> "how high is this gene for its region / disease"
    (identical to regressing the ranks on group dummies and keeping the residuals)
  - correlate gene A's residuals with gene B's residuals
  - p-value from a t-test with n - 2 - k degrees of freedom, k = number of groups - 1
    (region: 3 regions -> k = 2;  disease: 4 diagnoses -> k = 3)
If the pooled rho is high but the partial rho is near 0, the correlation came from both genes
differing between the groups, not from the genes varying together.

Note: Spearman(A, B) is exactly the same as Spearman(B, A) - correlation has no direction.

Run with:
    python 2a_gene_gene_correlation/.py/gene_spearman.py            # asks which analysis to run (or uses ANALYSIS below if set)
    python 2a_gene_gene_correlation/.py/gene_spearman.py disease    # or spearman / region / all - skips the question
    python 2a_gene_gene_correlation/.py/gene_spearman.py 4          # numbered options 1-4 also work
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.data_loading.expression_data import load_matrix, gene_of, read_gene_names, region_of, disease_of
from functions.graphs.partial_effect import plot_partial_effect
from functions.partial_spearman.partial_spearman import ranks_within_group, partial_p_value


# --- Settings (change these, not the code below) ---------------------------
ANALYSIS = None          # which correlation(s) to calculate:
                         #   "spearman" = Spearman only (all samples pooled)
                         #   "region"   = Spearman + partial Spearman accounting for brain region
                         #   "disease"  = Spearman + partial Spearman accounting for disease (ignores region)
                         #   "all"      = Spearman + both partial correlations
                         #   None       = ask when the script runs
                         # A name on the command line (python gene_spearman.py disease) overrides this.
FIRST_GENE = 1          # first gene in the range (1 = first gene)
LAST_GENE = None         # last gene in the range (included); None = every gene (572 with the filter on)
ORDER_BY = "expression"  # "expression" = gene 1 is the most highly expressed gene
                         # "file"       = genes in the order they appear in the matrix

FILTER_LOW_EXPRESSION = True   # True = drop low-expressed genes BEFORE numbering them 1, 2, 3, ...
                               # (so with the filter on, LAST_GENE can go up to 572, not 816)
MIN_CPM = 1                    # a gene counts as "expressed" in a sample if its CPM is above this
MIN_FRACTION_SAMPLES = 0.8     # ... and it must be expressed in at least this fraction of samples

MATRIX_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv")
GTF_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf")
RESULTS_FOLDER = str(ROOT_FOLDER / "2a_gene_gene_correlation" / "CSV")
FIGURES_FOLDER = str(ROOT_FOLDER / "2a_gene_gene_correlation" / "Figures")

PLOT_PARTIAL_EFFECT = True  # True = scatter of normal Spearman rho vs partial rho (one plot for each
                            # partial correlation calculated: region and/or disease)
                            # False = turn the plots off (CSV is still saved)
MAX_DOTS = 5000             # above this many pairs, dots pile into a blob, so the plot shows
                            # density instead (hexagons coloured by how many pairs fall in each)
ANALYSIS_OPTIONS = {"spearman": 1, "region": 2, "disease": 3, "all": 4}


# The groupings a partial correlation can account for: name -> (function that reads the group
# from a sample name, wording used in plot labels)
COVARIATES = {"region": (region_of, "brain region"),
              "disease": (disease_of, "disease")}

# Which partial correlations each analysis choice calculates
ANALYSIS_COVARIATES = {"spearman": [], "region": ["region"], "disease": ["disease"],
                       "all": ["region", "disease"]}


def choose_analysis():
    """Which analysis to run: the command-line argument if there is one, else ANALYSIS,
    else ask. Keeps asking until one of the valid names (or its number) is typed."""
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    elif ANALYSIS is not None:
        choice = ANALYSIS
    else:
        print("Which analysis?\n"
              "  1 = spearman  (Spearman only)\n"
              "  2 = region    (Spearman + partial Spearman accounting for brain region)\n"
              "  3 = disease   (Spearman + partial Spearman accounting for disease)\n"
              "  4 = all       (Spearman + both partial correlations)")
        choice = input("Enter a number or name: ").strip().lower()
    numbers = {"1": "spearman", "2": "region", "3": "disease", "4": "all"}
    choice = numbers.get(choice, choice)   # "3" -> "disease"; names are left as they are
    while choice not in ANALYSIS_COVARIATES:
        choice = input(f"'{choice}' is not an option - type spearman, region, disease or all: ").strip().lower()
        choice = numbers.get(choice, choice)
    return choice


def main():
    # ---- 0. Which analysis: "spearman", "region", "disease" or "all" ----
    # Asked first, so the question comes before the slow loading
    analysis = choose_analysis()
    covariates = ANALYSIS_COVARIATES[analysis]   # e.g. "all" -> ["region", "disease"]
    print(f"Analysis: {analysis}")

    # ---- 1. Load, adults only ----
    df = load_matrix(MATRIX_PATH)
    df = df[[name for name in df.columns if "_Fetal_" not in name]]

    # ---- 2. Gene CPM = sum of each gene's isoforms ----
    # groupby with sort=False keeps genes in the order they first appear in the file
    gene_cpm = df.groupby([gene_of(i) for i in df.index], sort=False).sum()

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

    # A setting that changes the results also changes the file name, so those runs get separate files.
    filter_tag = "_filtered" if FILTER_LOW_EXPRESSION else ""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    descriptive_stem = (
        f"gene_spearman_{gene_range}_{ORDER_BY}{filter_tag}_"
        f"cpm{MIN_CPM:g}_frac{MIN_FRACTION_SAMPLES:.0%}_"
        f"{ANALYSIS_OPTIONS[analysis]:02d}_{analysis}_adults_"
        f"{len(selected.columns)}samples"
    )
    path = os.path.join(RESULTS_FOLDER, descriptive_stem + ".csv")
    partial_ranks_path = os.path.join(
        RESULTS_FOLDER,
        descriptive_stem + "_partial_ranks.npz"
    )

    needs_region_ranks = "region" in covariates
    within_group_ranks, n_covariates = {}, {}
    for covariate in covariates:
        group_of = COVARIATES[covariate][0]
        groups = [group_of(name) for name in selected.columns]
        n_covariates[covariate] = len(set(groups)) - 1
        print(f"Samples per {covariate}:", pd.Series(groups).value_counts().to_dict())
        within_group_ranks[covariate] = ranks_within_group(selected, groups)

    names = read_gene_names(GTF_PATH)
    # ---- 4. Spearman (and chosen partial Spearmans) for every pair ----
    n_samples = selected.shape[1]
    rows = []
    for pair_number, (gene_a, gene_b) in enumerate(combinations(selected.index, 2), start=1):
        if pair_number % 20000 == 0:
            print(f"  {pair_number:,} / {n_pairs:,} pairs done")
        rho, p_value = spearmanr(selected.loc[gene_a], selected.loc[gene_b])
        row = {"gene_a": gene_a, "gene_a_name": names.get(gene_a, ""),
               "gene_b": gene_b, "gene_b_name": names.get(gene_b, ""),
               "spearman_rho": rho, "p_value": p_value}
        for covariate in covariates:
            ranks = within_group_ranks[covariate]
            partial_rho = np.corrcoef(ranks.loc[gene_a], ranks.loc[gene_b])[0, 1]
            row[f"partial_rho_{covariate}"] = partial_rho
            row[f"partial_p_value_{covariate}"] = partial_p_value(
                partial_rho,
                n_samples,
                n_covariates[covariate]
            )
        row["n_samples"] = n_samples
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.reindex(
        results["spearman_rho"].abs().sort_values(ascending=False).index
    )
    print()
    print(results.head(20).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # A sidecar is only written for analyses containing region-adjusted ranks, which are the
    # residuals PMS consumes.
    results.to_csv(path, index=False)
    if needs_region_ranks:
        np.savez_compressed(
            partial_ranks_path,
            gene_ids=selected.index.to_numpy(dtype=str),
            partial_ranks=within_group_ranks["region"].to_numpy(dtype=float)
        )
        print(f"Saved PMS partial-rank residuals: {partial_ranks_path}")
    print(f"Saved analysis result: {path}")

    # ---- 6. Plot: normal vs partial rho, one plot per covariate (turn off with PLOT_PARTIAL_EFFECT = False) ----
    # e.g. ..._region_effect.png and/or ..._disease_effect.png; nothing for "spearman" alone
    if PLOT_PARTIAL_EFFECT:
        for covariate in covariates:
            figure_path = os.path.join(FIGURES_FOLDER, f"{descriptive_stem}_{covariate}_effect.png")
            plot_partial_effect(results, figure_path, gene_range, covariate, COVARIATES[covariate][1], MAX_DOTS)


if __name__ == "__main__":
    main()
