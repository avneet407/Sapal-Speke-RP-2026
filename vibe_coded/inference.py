"""
Gene-gene correlation pipeline (isoform matrix -> top correlated gene pairs -> isoform plots).

Steps
  1. Load the isoform CPM matrix (61k isoforms x 49 samples) and drop the fetal samples (EXCLUDE_FETAL).
  2. Sum isoforms into genes (gene CPM = sum of its isoforms' CPM in each sample).
  3. Drop lowly expressed genes (gene CPM below MIN_GENE_CPM).
  4. Spearman correlation for every pair of remaining genes (across the 49 samples).
  5. Keep only pairs with p-value below P_VALUE_THRESHOLD.
  6. Take N_TOP_PAIRS of them: by default 20 random pairs with |rho| between 0.6 and 0.7
     (RHO_RANGE, PAIR_SELECTION), or the strongest pairs if RHO_RANGE = None and PAIR_SELECTION = "strongest".
  7. For each pair (A, B):
       a) plot CPM of A vs CPM of B
       b) table: for the N_TOP_ISOFORMS most abundant isoforms of A, the partial Spearman
          correlation of the isoform's % share of A with B's total CPM, controlling for A's total CPM
          ("does the way A is split tell us about B, beyond how much A there is?")
       c) the same table the other way round: isoforms of B vs A's total, controlling for B's total
       (optionally also the isoform scatter grids - see PLOT_ISOFORM_GRIDS)
  8. Summary graph of how many genes / pairs survive each step.

Every step prints an expected run time first, then the actual time taken.
Figures go to Figures/inference/, tables (CSV) to Results/inference/.
Run from the Sapal-Speke-RP-2026 folder:  python inference.py
"""

import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_distribution, false_discovery_control

from figure_utils import save_figure
from control import remove_fetal
from gene_comparison import (load_matrix, gene_ids, sum_by_gene,
                             region_colours_for, add_region_legend, spearman_text)


# =============================================================================
#  USER PARAMETERS - edit these
# =============================================================================

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
EXCLUDE_FETAL = True           # True = drop the 4 fetal DLPFC samples straight after loading (49 -> 45 samples),
                               # so fetal vs adult differences can't drive the correlations

# ---- Step 3: gene expression filter (X) ----
MIN_GENE_CPM = 1000            # X: drop genes whose CPM (all isoforms added up) is below this
GENE_CPM_STATISTIC = "sum"     # "sum"  = total CPM over all 49 samples
                               # "mean" = average CPM per sample (= sum / 49)
MIN_SAMPLES_EXPRESSED = 0      # Also require gene CPM > 0 in at least this many samples (0 = off).
                               # Genes that are zero in most samples give lots of tied ranks,
                               # which can make Spearman look stronger than it really is.

# ---- Step 5: significance filter (Y) ----
P_VALUE_THRESHOLD = 0.05       # Y: keep pairs with p below this
USE_FDR = True                 # True  = Benjamini-Hochberg adjusted p (recommended: with ~1000 genes
                               #         we test ~500,000 pairs, so ~25,000 would pass p < 0.05 by
                               #         chance alone using raw p-values)
                               # False = raw Spearman p-values
CORRELATION_SIGN = "both"      # "both", "positive" (rho > 0 only) or "negative" (rho < 0 only)

# ---- Step 6: which significant pairs to take forward ----
N_TOP_PAIRS = 20               # Z: number of gene pairs to take forward
RHO_RANGE = (0.6, 0.7)         # only pairs with |rho| in this range (inclusive); None = no limit.
                               # Very strong pairs (|rho| ~0.9) leave little variation beyond the gene total
                               # for isoforms to explain, so a moderate band suits the isoform tables better.
PAIR_SELECTION = "random"      # "random"    = Z random pairs from the range (fixed seed, so reproducible)
                               # "strongest" = the Z pairs with the largest |rho| in the range
RANDOM_SEED = 0                # change to draw a different random set

# ---- Step 7: isoform tables ----
N_TOP_ISOFORMS = 6             # W: isoforms per gene in the tables (most abundant by mean CPM)
CONTROL_FOR_REGION = False     # True = partial Spearman also removes brain region (DLPFC / Hippocampus /
                               # Caudate), so region differences can't create a fake isoform link
PLOT_ISOFORM_GRIDS = False     # True = also draw the isoform scatter grids (% and CPM) for each pair
LOG_CPM_AXES = False           # True = log scale on CPM axes (helps when a few samples dominate)
FULL_PERCENT_AXIS = False      # True = isoform % axis always 0-100 (easier to compare isoforms,
                               # but squashes minor isoforms at ~2%); False = zoom to the data

# ---- Output ----
SHOW_PLOTS = False             # True = open each figure window (script pauses until closed).
                               # Figures are always saved either way.
FIGURE_FOLDER = "inference"    # Figures/<FIGURE_FOLDER>/
TABLE_FOLDER = "Results/inference"
N_TABLE_ROWS = 10              # rows shown in the printed example tables

# =============================================================================


def start_step(title, expected):
    """Print a step header with the expected run time, return the start time."""
    print(f"\n{'=' * 70}\n{title}\n  Expected time: {expected}")
    return time.perf_counter()


def end_step(start):
    print(f"  Done in {time.perf_counter() - start:.1f} s")


def finish_figure(fig, name):
    """Save, then show or close (closing stops ~20 open figures piling up in memory)."""
    save_figure(fig, FIGURE_FOLDER, name)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def save_table(table, name):
    """Save a DataFrame as CSV with a timestamp, like save_figure does for figures."""
    os.makedirs(TABLE_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(TABLE_FOLDER, f"{name}_{timestamp}.csv")
    table.to_csv(path, index=False)
    print(f"Saved table: {path}")


# ---------------------------------------------------------------------------
# Step 1-2: load and sum to genes
# ---------------------------------------------------------------------------

def show_example_gene(df, gene_cpm, n_samples=4):
    """Print one gene's isoforms next to its gene total, to show what the summing does.
    Picks the first gene with 3-5 isoforms so the table stays small."""
    isoform_counts = pd.Series(gene_ids(df.index)).value_counts(sort=False)
    gene = isoform_counts[(isoform_counts >= 3) & (isoform_counts <= 5)].index[0]

    samples = df.columns[:n_samples]
    isoforms = df[[g == gene for g in gene_ids(df.index)]][samples]
    total = gene_cpm.loc[[gene], samples].rename(index={gene: f"{gene} TOTAL"})
    print(f"\nExample: isoforms of {gene} and their sum (first {n_samples} samples)")
    print(pd.concat([isoforms, total]).round(1).to_string())


# ---------------------------------------------------------------------------
# Step 3: filter genes
# ---------------------------------------------------------------------------

def gene_expression_statistic(gene_cpm):
    """The number compared against MIN_GENE_CPM for each gene."""
    if GENE_CPM_STATISTIC == "sum":
        return gene_cpm.sum(axis=1)
    if GENE_CPM_STATISTIC == "mean":
        return gene_cpm.mean(axis=1)
    raise ValueError('GENE_CPM_STATISTIC must be "sum" or "mean"')


def filter_genes(gene_cpm):
    """Keep genes with expression statistic >= MIN_GENE_CPM and enough non-zero samples."""
    statistic = gene_expression_statistic(gene_cpm)
    n_expressed = (gene_cpm > 0).sum(axis=1)
    keep = (statistic >= MIN_GENE_CPM) & (n_expressed >= MIN_SAMPLES_EXPRESSED)
    return gene_cpm[keep], statistic, keep


def plot_gene_filter(statistic, keep):
    """Histogram of gene expression (log scale) with the cut-off marked.
    Shows how many genes the threshold throws away and whether it sits in a sensible place."""
    # +1 so genes with 0 CPM can still go on a log axis
    log_values = np.log10(statistic + 1)
    bins = np.linspace(0, log_values.max(), 50)

    fig = plt.figure(figsize=(9, 5))
    plt.hist(log_values[keep], bins=bins, color="#2a78d6", edgecolor="white", label=f"Kept ({keep.sum()})")
    plt.hist(log_values[~keep], bins=bins, color="#b0b0b0", edgecolor="white", label=f"Removed ({(~keep).sum()})")
    plt.axvline(np.log10(MIN_GENE_CPM + 1), color="black", linestyle="--", label=f"X = {MIN_GENE_CPM} CPM")
    plt.xlabel(f"log10({GENE_CPM_STATISTIC} gene CPM over samples + 1)")
    plt.ylabel("Number of genes")
    plt.title(f"Step 3: gene expression filter ({GENE_CPM_STATISTIC} CPM >= {MIN_GENE_CPM}"
              f"{f', expressed in >= {MIN_SAMPLES_EXPRESSED} samples' if MIN_SAMPLES_EXPRESSED else ''})")
    plt.legend()
    plt.tight_layout()
    finish_figure(fig, "step3_gene_filter_histogram")


# ---------------------------------------------------------------------------
# Step 4: Spearman for all pairs
# ---------------------------------------------------------------------------

def spearman_all_pairs(gene_cpm):
    """Spearman rho and p-value for every pair of genes, all at once (no Python loop).

    Spearman = Pearson correlation of the ranks, so: rank each gene's 49 values, then
    np.corrcoef gives every pair's rho in one go. The p-value uses the same t-test as
    scipy.stats.spearmanr: t = rho * sqrt((n-2) / (1-rho^2)), n-2 degrees of freedom.
    This takes ~1 s for ~1000 genes, vs. several minutes looping over spearmanr."""
    n_samples = gene_cpm.shape[1]
    ranks = gene_cpm.rank(axis=1).to_numpy()   # ties (e.g. many zeros) get their average rank, as in spearmanr

    with np.errstate(divide="ignore", invalid="ignore"):
        rho_matrix = np.corrcoef(ranks)

    # Upper triangle only: each pair once, no gene paired with itself
    i, j = np.triu_indices(len(gene_cpm), k=1)
    rho = rho_matrix[i, j]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = rho * np.sqrt((n_samples - 2) / ((1 - rho) * (1 + rho)))
    p_value = 2 * t_distribution.sf(np.abs(t_stat), n_samples - 2)

    # Samples where both genes are detected (CPM > 0), via a matrix product of 0/1 detection
    detected = (gene_cpm.to_numpy() > 0).astype(np.int32)
    n_both = (detected @ detected.T)[i, j]

    genes = gene_cpm.index.to_numpy()
    results = pd.DataFrame({"gene_a": genes[i], "gene_b": genes[j], "spearman_rho": rho,
                            "p_value": p_value, "n_both_detected": n_both})
    return results.dropna(subset=["spearman_rho"])   # constant genes have no defined rho


def plot_correlation_distributions(results):
    """Left: distribution of rho over all pairs. Right: p-value histogram.
    If nothing were truly correlated, p-values would be flat (dashed line);
    the spike at p ~ 0 above that line is the real signal."""
    fig, (ax_rho, ax_p) = plt.subplots(1, 2, figsize=(14, 5))

    ax_rho.hist(results["spearman_rho"], bins=80, range=(-1, 1), color="#2a78d6", edgecolor="white")
    ax_rho.axvline(0, color="black", linewidth=0.8)
    ax_rho.set_xlabel("Spearman rho")
    ax_rho.set_ylabel("Number of gene pairs")
    ax_rho.set_title(f"Correlation of all {len(results):,} gene pairs")

    n_bins = 20
    ax_p.hist(results["p_value"], bins=n_bins, range=(0, 1), color="#2a78d6", edgecolor="white")
    ax_p.axhline(len(results) / n_bins, color="black", linestyle="--", label="Expected if no correlations")
    ax_p.set_xlabel("Raw Spearman p-value")
    ax_p.set_ylabel("Number of gene pairs")
    ax_p.set_title("Raw p-value distribution")
    ax_p.legend()

    fig.suptitle("Step 4: Spearman correlation of every gene pair")
    plt.tight_layout()
    finish_figure(fig, "step4_rho_and_p_distributions")


# ---------------------------------------------------------------------------
# Step 5-6: significance filter and top pairs
# ---------------------------------------------------------------------------

def filter_significant(results):
    """Add the p-value used for filtering, then keep pairs below the threshold with the chosen sign."""
    results = results.copy()
    results["fdr_p_value"] = false_discovery_control(results["p_value"], method="bh")
    p_column = "fdr_p_value" if USE_FDR else "p_value"

    keep = results[p_column] < P_VALUE_THRESHOLD
    if CORRELATION_SIGN == "positive":
        keep &= results["spearman_rho"] > 0
    elif CORRELATION_SIGN == "negative":
        keep &= results["spearman_rho"] < 0
    elif CORRELATION_SIGN != "both":
        raise ValueError('CORRELATION_SIGN must be "both", "positive" or "negative"')

    # Every pair is tested on the same 49 samples, so a smaller p always means a larger |rho|.
    # Sorting by |rho| gives the same order as by p, but still works when p underflows to 0.
    results["abs_rho"] = results["spearman_rho"].abs()
    results = results.sort_values("abs_rho", ascending=False)
    return results, results[keep[results.index]], p_column


def select_pairs(significant):
    """Pick the N_TOP_PAIRS pairs to take forward: restrict to |rho| in RHO_RANGE, then take
    either a random sample or the strongest. Returned sorted by |rho|, strongest first."""
    candidates = significant
    if RHO_RANGE is not None:
        low, high = RHO_RANGE
        candidates = significant[significant["abs_rho"].between(low, high)]
        print(f"  {len(candidates):,} significant pairs have |rho| between {low} and {high}")
    if candidates.empty:
        raise SystemExit("No significant pairs in RHO_RANGE - widen it.")

    if PAIR_SELECTION == "strongest":
        return candidates.head(N_TOP_PAIRS)
    if PAIR_SELECTION == "random":
        chosen = candidates.sample(n=min(N_TOP_PAIRS, len(candidates)), random_state=RANDOM_SEED)
        return chosen.sort_values("abs_rho", ascending=False)
    raise ValueError('PAIR_SELECTION must be "random" or "strongest"')


def plot_volcano(results, significant, top_pairs, p_column):
    """rho vs -log10(p) for every pair. Grey = not significant, blue = passes Y, black = the Z plotted.
    Shows where the threshold cuts and how extreme the plotted pairs are."""
    # p can be exactly 0 for very strong correlations; clip so the log is finite
    smallest = np.finfo(float).tiny
    minus_log_p = -np.log10(results[p_column].clip(lower=smallest))

    not_significant = ~results.index.isin(significant.index)
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(results.loc[not_significant, "spearman_rho"], minus_log_p[not_significant],
                s=2, color="#b0b0b0", rasterized=True, label="Not significant")
    plt.scatter(significant["spearman_rho"], minus_log_p[significant.index],
                s=2, color="#2a78d6", rasterized=True, label=f"Significant ({len(significant):,})")
    plt.scatter(top_pairs["spearman_rho"], minus_log_p[top_pairs.index],
                s=40, color="black", label=f"Selected {len(top_pairs)} (taken forward)")
    # Top pairs sit almost on top of each other, so label with the pair number (matches figure names)
    for rank, (idx, row) in enumerate(top_pairs.iterrows(), start=1):
        plt.annotate(str(rank), (row.spearman_rho, minus_log_p[idx]), fontsize=8,
                     xytext=(-12 if rank % 2 else 6, -3), textcoords="offset points")

    plt.axhline(-np.log10(P_VALUE_THRESHOLD), color="black", linestyle="--",
                label=f"Y = {P_VALUE_THRESHOLD}")
    plt.xlabel("Spearman rho")
    plt.ylabel(f"-log10({'BH-adjusted ' if USE_FDR else ''}p-value)")
    plt.title("Step 5: significance filter (each dot = one gene pair; numbers = plotted pairs)")
    legend = plt.legend(fontsize=8)
    for handle in legend.legend_handles[:3]:
        handle.set_sizes([20])   # same marker size in the legend for all three groups
    plt.tight_layout()
    finish_figure(fig, "step5_volcano")


# ---------------------------------------------------------------------------
# Step 7: per-pair plots
# ---------------------------------------------------------------------------

def plot_gene_vs_gene(cpm_a, cpm_b, gene_a, gene_b, rank):
    """CPM of gene A against gene B, one point per sample, coloured by brain region."""
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(cpm_a, cpm_b, c=region_colours_for(cpm_a.index))
    if LOG_CPM_AXES:
        plt.xscale("symlog")   # symlog copes with zeros, unlike log
        plt.yscale("symlog")
    plt.xlabel(f"{gene_a} CPM (all isoforms)")
    plt.ylabel(f"{gene_b} CPM (all isoforms)")
    plt.title(f"Pair {rank}: {gene_a} vs {gene_b}\n{spearman_text(cpm_a, cpm_b)}", fontsize=10)
    add_region_legend(plt.gca())
    plt.tight_layout()
    finish_figure(fig, f"pair{rank}_{gene_a}_vs_{gene_b}_gene_cpm")


def spearman_rho_p(x, y):
    """Spearman rho, p-value and n for two series, skipping samples where x is NaN."""
    valid = x.notna()
    n = valid.sum()
    rho = x[valid].corr(y[valid], method="spearman")
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = rho * np.sqrt((n - 2) / ((1 - rho) * (1 + rho)))
    return rho, 2 * t_distribution.sf(abs(t_stat), n - 2), n


def plot_isoforms_vs_gene(df, gene_cpm, isoform_gene, other_gene, rank):
    """For the N_TOP_ISOFORMS isoforms of isoform_gene with the highest mean CPM, make two figures
    (one subplot per isoform, other gene's total CPM on y):
      - "percent": x = % of isoform_gene's CPM coming from that isoform
      - "cpm":     x = the isoform's own CPM

    Comparing the two separates "the other gene tracks how much of this gene there is" (CPM plot
    correlated, % plot flat) from "the other gene tracks which isoform is used" (% plot correlated) -
    the second is the starting point for directionality.
    """
    isoforms, _, top_isoforms = top_isoforms_of(df, isoform_gene)

    total = gene_cpm.loc[isoform_gene]
    other_cpm = gene_cpm.loc[other_gene]
    colours = region_colours_for(other_cpm.index)

    # x values for each isoform, for both versions of the plot.
    # Samples where the gene has 0 CPM give 0/0 = NaN for %; spearman_text skips them.
    x_values = {"percent": {iso: 100 * isoforms.loc[iso] / total.replace(0, np.nan) for iso in top_isoforms},
                "cpm": {iso: isoforms.loc[iso] for iso in top_isoforms}}
    x_labels = {"percent": f"% of {isoform_gene} CPM", "cpm": "isoform CPM"}

    n_cols = math.ceil(math.sqrt(len(top_isoforms)))
    n_rows = math.ceil(len(top_isoforms) / n_cols)

    for mode in ["percent", "cpm"]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.5 * n_rows), squeeze=False)
        for isoform, ax in zip(top_isoforms, axes.flatten()):
            x = x_values[mode][isoform]
            ax.scatter(x, other_cpm, c=colours, s=15)
            if LOG_CPM_AXES:
                ax.set_yscale("symlog")
                if mode == "cpm":
                    ax.set_xscale("symlog")
            if mode == "percent" and FULL_PERCENT_AXIS:
                ax.set_xlim(-3, 103)
            ax.set_xlabel(f"{isoform}\n{x_labels[mode]}", fontsize=8)
            ax.set_ylabel(f"{other_gene} CPM (all isoforms)", fontsize=8)
            ax.set_title(spearman_text(x, other_cpm), fontsize=7)

        for ax in axes.flatten()[len(top_isoforms):]:
            ax.set_visible(False)   # unused grid cells when the gene has fewer isoforms than the grid
        add_region_legend(axes[0, 0], fontsize=7)

        fig.suptitle(f"Pair {rank}: top {len(top_isoforms)} of {len(isoforms)} {isoform_gene} isoforms "
                     f"({x_labels[mode]}) vs {other_gene} CPM")
        plt.tight_layout()
        finish_figure(fig, f"pair{rank}_{isoform_gene}_isoform_{mode}_vs_{other_gene}")


def top_isoforms_of(df, gene):
    """All isoforms (rows) of gene, their mean CPM (highest first), and the top N_TOP_ISOFORMS of them."""
    isoforms = df[[g == gene for g in gene_ids(df.index)]]
    mean_isoform_cpm = isoforms.mean(axis=1).sort_values(ascending=False)
    return isoforms, mean_isoform_cpm, mean_isoform_cpm.index[:N_TOP_ISOFORMS]


def partial_spearman(x, y, control, regions=None):
    """Partial Spearman correlation of x and y, controlling for `control` (and region if given).

    1. Rank x, y and control (samples with a NaN in any of them are dropped first).
    2. Fit rank(x) ~ rank(control) [+ region] by least squares; keep the leftovers (residuals).
       Same for rank(y). The leftovers are the parts of x and y that `control` can't explain.
    3. Correlate the two sets of leftovers -> partial rho.
    p-value: same t-test as plain Spearman, but losing one degree of freedom per covariate
    (df = n - 2 - number of covariates). Returns rho, p, n."""
    keep = x.notna() & y.notna() & control.notna()
    x_rank, y_rank, control_rank = x[keep].rank(), y[keep].rank(), control[keep].rank()

    covariates = [np.ones(keep.sum()), control_rank.to_numpy()]
    if regions is not None:
        # One 0/1 column per region except the first (the intercept stands in for it)
        dummies = pd.get_dummies(regions[keep], drop_first=True, dtype=float)
        covariates += [dummies[column].to_numpy() for column in dummies]
    design = np.column_stack(covariates)

    def leftovers(values):
        coefficients = np.linalg.lstsq(design, values.to_numpy(), rcond=None)[0]
        return values.to_numpy() - design @ coefficients

    n = int(keep.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.corrcoef(leftovers(x_rank), leftovers(y_rank))[0, 1]
        degrees_of_freedom = n - 2 - (design.shape[1] - 1)
        t_stat = rho * np.sqrt(degrees_of_freedom / ((1 - rho) * (1 + rho)))
    return rho, 2 * t_distribution.sf(abs(t_stat), degrees_of_freedom), n


def isoform_partial_table(df, gene_cpm, isoform_gene, other_gene, rank):
    """For the N_TOP_ISOFORMS most abundant isoforms of isoform_gene:
    - plain Spearman of the isoform's % share of its gene vs other_gene's total CPM
    - partial Spearman of the same, controlling for isoform_gene's total CPM (+ region if
      CONTROL_FOR_REGION), i.e. the link between isoform choice and the other gene that is
      not just "more of the gene overall".
    Share = isoform CPM / gene total CPM in that sample; samples where the gene is 0 are skipped."""
    isoforms, mean_isoform_cpm, top_isoforms = top_isoforms_of(df, isoform_gene)
    total = gene_cpm.loc[isoform_gene]
    other_cpm = gene_cpm.loc[other_gene]
    regions = pd.Series([name.split("_")[0] for name in total.index], index=total.index) if CONTROL_FOR_REGION else None

    rows = []
    for isoform in top_isoforms:
        share = 100 * isoforms.loc[isoform] / total.replace(0, np.nan)
        rho, p_value, _ = spearman_rho_p(share, other_cpm)
        partial_rho, partial_p, n = partial_spearman(share, other_cpm, total, regions)
        rows.append({"pair": rank, "isoform_gene": isoform_gene, "other_gene": other_gene,
                     "isoform": isoform, "mean_isoform_cpm": mean_isoform_cpm[isoform],
                     "mean_percent_of_gene": share.mean(),
                     "spearman_rho": rho, "spearman_p": p_value,
                     "partial_rho": partial_rho, "partial_p": partial_p, "n_samples": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 8: summary
# ---------------------------------------------------------------------------

def plot_funnel(counts):
    """Horizontal bars: how many items survive each step (log scale, numbers written on bars)."""
    labels, values = list(counts.keys()), list(counts.values())
    fig = plt.figure(figsize=(10, 4.5))
    bars = plt.barh(labels[::-1], values[::-1], color="#2a78d6")
    plt.xscale("log")
    for bar, value in zip(bars, values[::-1]):
        plt.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=9)
    plt.xlim(right=max(values) * 10)
    plt.xlabel("Count (log scale)")
    plt.title("Pipeline summary: what survives each step")
    plt.tight_layout()
    finish_figure(fig, "step8_pipeline_summary")


def main():
    pipeline_start = time.perf_counter()
    print("Parameters:")
    print(f"  X  MIN_GENE_CPM      = {MIN_GENE_CPM} ({GENE_CPM_STATISTIC} over samples), "
          f"MIN_SAMPLES_EXPRESSED = {MIN_SAMPLES_EXPRESSED}")
    print(f"  Y  P_VALUE_THRESHOLD = {P_VALUE_THRESHOLD} ({'BH FDR' if USE_FDR else 'raw p'}), "
          f"sign = {CORRELATION_SIGN}")
    print(f"  Z  N_TOP_PAIRS       = {N_TOP_PAIRS}")
    print(f"  W  N_TOP_ISOFORMS    = {N_TOP_ISOFORMS}")

    # ---- Step 1 ----
    start = start_step("Step 1: load isoform matrix",
                       "~1-10 s (61k rows x 49 samples; longer if OneDrive has to download the file)")
    df = load_matrix(MATRIX_PATH)
    if EXCLUDE_FETAL:
        n_before = df.shape[1]
        df = remove_fetal(df)
        print(f"  Removed {n_before - df.shape[1]} fetal samples")
    # Isoforms that are 0 in every remaining sample (e.g. fetal-only) carry no information
    df = df[(df > 0).any(axis=1)]
    print(f"  {df.shape[0]:,} isoforms x {df.shape[1]} samples, {100 * (df == 0).to_numpy().mean():.0f}% zeros")
    print(f"\nFirst {N_TABLE_ROWS} isoforms, first 4 samples (CPM):")
    print(df.iloc[:N_TABLE_ROWS, :4].round(1).to_string())
    end_step(start)

    # ---- Step 2 ----
    start = start_step("Step 2: sum isoforms into genes", "~1 s")
    gene_cpm = sum_by_gene(df)
    print(f"  {len(gene_cpm):,} genes")
    show_example_gene(df, gene_cpm)
    end_step(start)

    # ---- Step 3 ----
    start = start_step("Step 3: filter lowly expressed genes", "~2 s (includes saving histogram)")
    filtered_cpm, statistic, keep = filter_genes(gene_cpm)
    print(f"  Kept {keep.sum():,} of {len(gene_cpm):,} genes")
    if len(filtered_cpm) < 2:
        raise SystemExit("Fewer than 2 genes pass the filter - lower MIN_GENE_CPM.")
    plot_gene_filter(statistic, keep)
    end_step(start)

    # ---- Step 4 ----
    n_genes = len(filtered_cpm)
    n_pairs = n_genes * (n_genes - 1) // 2
    # Measured: ~1 s for 175k pairs (590 genes) including the histograms; grows with genes^2
    start = start_step(f"Step 4: Spearman correlation for all {n_pairs:,} gene pairs",
                       f"~{n_pairs / 200_000 + 1:.0f} s")
    results = spearman_all_pairs(filtered_cpm)
    print(f"\nStrongest {N_TABLE_ROWS} correlations (by |rho|):")
    print(results.reindex(results["spearman_rho"].abs().sort_values(ascending=False).index)
          .head(N_TABLE_ROWS).to_string(index=False))
    plot_correlation_distributions(results)
    end_step(start)

    # ---- Step 5-6 ----
    start = start_step("Step 5-6: significance filter and pick top pairs",
                       f"~{n_pairs / 100_000 + 1:.0f} s (volcano plot of every pair is the slow part)")
    results, significant, p_column = filter_significant(results)
    print(f"  {len(significant):,} of {len(results):,} pairs have {p_column} < {P_VALUE_THRESHOLD}"
          f" (sign: {CORRELATION_SIGN})")
    if significant.empty:
        raise SystemExit("No significant pairs - raise P_VALUE_THRESHOLD or set USE_FDR = False.")
    top_pairs = select_pairs(significant)
    print(f"\n{len(top_pairs)} pairs taken forward ({PAIR_SELECTION}"
          f"{f', |rho| {RHO_RANGE[0]}-{RHO_RANGE[1]}' if RHO_RANGE else ''}):")
    print(top_pairs.drop(columns="abs_rho").to_string(index=False))
    save_table(significant.drop(columns="abs_rho"), "significant_pairs")
    plot_volcano(results, significant, top_pairs, p_column)
    end_step(start)

    # ---- Step 7 ----
    # Tables take <0.1 s per pair; the gene-gene scatter ~1 s; isoform grids (if on) another ~4-8 s
    seconds_per_pair = (1, 2) if not PLOT_ISOFORM_GRIDS else (5, 10)
    start = start_step(f"Step 7: isoform partial Spearman tables for top {len(top_pairs)} pairs"
                       f"{' (controlling for region too)' if CONTROL_FOR_REGION else ''}",
                       f"~{seconds_per_pair[0] * len(top_pairs)}-{seconds_per_pair[1] * len(top_pairs)} s")
    isoform_tables = []
    for rank, row in enumerate(top_pairs.itertuples(), start=1):
        gene_a, gene_b = row.gene_a, row.gene_b
        print(f"\n  Pair {rank}/{len(top_pairs)}: {gene_a} vs {gene_b} (rho = {row.spearman_rho:.2f})")
        plot_gene_vs_gene(filtered_cpm.loc[gene_a], filtered_cpm.loc[gene_b], gene_a, gene_b, rank)
        for isoform_gene, other_gene in [(gene_a, gene_b), (gene_b, gene_a)]:
            table = isoform_partial_table(df, filtered_cpm, isoform_gene, other_gene, rank)
            print(f"\n  Isoforms of {isoform_gene} (% share) vs {other_gene} total CPM, "
                  f"partial = controlling for {isoform_gene} total:")
            print(table.drop(columns=["pair", "isoform_gene", "other_gene"])
                  .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
            isoform_tables.append(table)
            if PLOT_ISOFORM_GRIDS:
                plot_isoforms_vs_gene(df, filtered_cpm, isoform_gene, other_gene, rank)
    save_table(pd.concat(isoform_tables), "top_pairs_isoform_partial_spearman")
    end_step(start)

    # ---- Step 8 ----
    start = start_step("Step 8: pipeline summary", "~2 s")
    plot_funnel({"Isoforms loaded": len(df),
                 "Genes": len(gene_cpm),
                 f"Genes passing X = {MIN_GENE_CPM}": n_genes,
                 "Gene pairs tested": len(results),
                 f"Pairs with p < Y = {P_VALUE_THRESHOLD}": len(significant),
                 f"Pairs taken forward (Z = {N_TOP_PAIRS})": len(top_pairs)})
    end_step(start)

    print(f"\nTotal run time: {time.perf_counter() - pipeline_start:.0f} s")


if __name__ == "__main__":
    main()
