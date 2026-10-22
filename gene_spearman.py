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
  5. Save a table (gene IDs, gene names, rhos, p-values) as CSV in Results/gene_spearman/.
  6. Optional plot (PLOT_PARTIAL_EFFECT): normal rho vs partial rho, one dot per pair, saved next to
     the CSV - one plot per partial correlation (none for "spearman" alone).

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

Run from the Sapal-Speke-RP-2026 folder:
    python gene_spearman.py            # asks which analysis to run (or uses ANALYSIS below if set)
    python gene_spearman.py disease    # or spearman / region / all - skips the question
"""

import os
import re
import sys
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as t_distribution

from describe_data import load_matrix, gene_of


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

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
RESULTS_FOLDER = "Results/gene_spearman"

PLOT_PARTIAL_EFFECT = True  # True = scatter of normal Spearman rho vs partial rho (one plot for each
                            # partial correlation calculated: region and/or disease)
                            # False = turn the plots off (CSV is still saved)
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


def disease_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the disease is the second part ("MDD")."""
    return sample_name.split("_")[1]


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


def plot_partial_effect(results, path, gene_range, covariate):
    """Normal Spearman rho (x) against partial rho accounting for `covariate` (y), one point per
    gene pair. covariate = "region" or "disease" (a key of COVARIATES).
    Points on the dashed y = x line are unaffected by it. Points pulled towards y = 0 had a
    correlation that came (partly) from both genes differing between the groups.
    Up to MAX_DOTS pairs: one dot each. More than that: a hexbin (density) plot instead."""
    partial = results[f"partial_rho_{covariate}"]   # the y values, e.g. column "partial_rho_disease"
    wording = COVARIATES[covariate][1]              # "brain region" or "disease", for the labels
    fig = plt.figure(figsize=(7.5 if len(results) > MAX_DOTS else 6.5, 6.5))   # wider if a colour bar is added
    if len(results) > MAX_DOTS:
        # Split the square into hexagons; colour = number of pairs in each (log scale, as most
        # pairs sit near 0). mincnt=1 leaves empty hexagons white.
        hexagons = plt.hexbin(results["spearman_rho"], partial, gridsize=80,
                              extent=(-1, 1, -1, 1), bins="log", mincnt=1, cmap="Blues", zorder=2)
        plt.colorbar(hexagons, label="Number of gene pairs (log scale)")
    else:
        plt.scatter(results["spearman_rho"], partial, s=18, color="#2a78d6",
                    alpha=0.6, edgecolor="white", linewidth=0.4, zorder=2)
    # Reference lines drawn on top (zorder=3) so they stay visible over the hexagons
    plt.axhline(0, color="#d0d0d0", linewidth=0.8, zorder=3)
    plt.axvline(0, color="#d0d0d0", linewidth=0.8, zorder=3)
    plt.plot([-1, 1], [-1, 1], color="grey", linestyle="--", linewidth=1,
             label=f"y = x ({wording} has no effect)", zorder=3)

    # Label the 3 pairs that the covariate changed the most, using gene names (G locus if no name)
    change = (results["spearman_rho"] - partial).abs()
    for index in change.nlargest(3).index:
        row = results.loc[index]
        label = f"{row['gene_a_name'] or row['gene_a']}–{row['gene_b_name'] or row['gene_b']}"
        # Dots on the right get their label on the left, so it doesn't run off the plot
        on_right = row["spearman_rho"] > 0.5
        plt.annotate(label, (row["spearman_rho"], partial[index]),
                     xytext=(-5 if on_right else 5, 5), textcoords="offset points",
                     ha="right" if on_right else "left", fontsize=8)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Spearman rho (all adult samples pooled)")
    plt.ylabel(f"Partial Spearman rho (accounting for {wording})")
    plt.title(f"Effect of {wording} on gene-gene correlation\n"
              f"Genes {gene_range}, {len(results):,} pairs, "
              f"median |change| = {change.median():.2f}")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()                                  # stops labels being cut off

    fig.savefig(path, dpi=200)
    print(f"Saved figure: {path}")
    plt.show()                                          # opens the figure window (script pauses until closed)


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

    # For each partial correlation chosen (region and/or disease), work out once for all genes:
    #   - the group of each sample (in the same order as the columns)
    #   - k = number of groups - 1 (3 regions -> 2, 4 diseases -> 3: one group is the baseline)
    #   - the within-group ranks, reused for every pair
    within_group_ranks, n_covariates = {}, {}
    for covariate in covariates:
        group_of = COVARIATES[covariate][0]            # region_of or disease_of
        groups = [group_of(name) for name in selected.columns]
        n_covariates[covariate] = len(set(groups)) - 1
        print(f"Samples per {covariate}:", pd.Series(groups).value_counts().to_dict())
        within_group_ranks[covariate] = ranks_within_group(selected, groups)

    # ---- 4. Spearman (and the chosen partial Spearmans) for every pair ----
    # combinations(..., 2) gives each pair once: (g1, g2), (g1, g3), ... but never (g2, g1)
    n_samples = selected.shape[1]
    rows = []
    for pair_number, (gene_a, gene_b) in enumerate(combinations(selected.index, 2), start=1):
        if pair_number % 20000 == 0:   # progress, so a long run doesn't look frozen
            print(f"  {pair_number:,} / {n_pairs:,} pairs done")
        rho, p_value = spearmanr(selected.loc[gene_a], selected.loc[gene_b])
        row = {"gene_a": gene_a, "gene_a_name": names.get(gene_a, ""),
               "gene_b": gene_b, "gene_b_name": names.get(gene_b, ""),
               "spearman_rho": rho, "p_value": p_value}
        for covariate in covariates:
            # Partial Spearman = ordinary (Pearson) correlation of the within-group ranks
            ranks = within_group_ranks[covariate]
            partial_rho = np.corrcoef(ranks.loc[gene_a], ranks.loc[gene_b])[0, 1]
            # Columns e.g. "partial_rho_disease" and "partial_p_value_disease"
            row[f"partial_rho_{covariate}"] = partial_rho
            row[f"partial_p_value_{covariate}"] = partial_p_value(partial_rho, n_samples,
                                                                  n_covariates[covariate])
        row["n_samples"] = n_samples
        rows.append(row)

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
    # The analysis name is in the file name too, so the different runs are easy to tell apart
    path = os.path.join(RESULTS_FOLDER,
                        f"gene_spearman_{gene_range}_{ORDER_BY}{filter_tag}_{analysis}_{timestamp}.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # ---- 6. Plot: normal vs partial rho, one plot per covariate (turn off with PLOT_PARTIAL_EFFECT = False) ----
    # e.g. ..._region_effect.png and/or ..._disease_effect.png; nothing for "spearman" alone
    if PLOT_PARTIAL_EFFECT:
        for covariate in covariates:
            plot_partial_effect(results, path.replace(".csv", f"_{covariate}_effect.png"),
                                gene_range, covariate)


if __name__ == "__main__":
    main()
