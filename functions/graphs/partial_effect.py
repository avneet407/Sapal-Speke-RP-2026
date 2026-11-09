"""Scatter of normal Spearman rho against partial Spearman rho."""

import matplotlib.pyplot as plt


def plot_partial_effect(results, path, gene_range, covariate, wording, max_dots):
    """Normal Spearman rho (x) against partial rho accounting for `covariate` (y), one point per
    gene pair. covariate = "region" or "disease"; wording = how the labels describe it.
    Points on the dashed y = x line are unaffected by it. Points pulled towards y = 0 had a
    correlation that came (partly) from both genes differing between the groups.
    Up to max_dots pairs: one dot each. More than that: a hexbin (density) plot instead."""
    partial = results[f"partial_rho_{covariate}"]   # the y values, e.g. column "partial_rho_disease"
    fig = plt.figure(figsize=(7.5 if len(results) > max_dots else 6.5, 6.5))   # wider if a colour bar is added
    if len(results) > max_dots:
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

    # Label the 3 pairs that the covariate changed the most, using gene names
    change = (results["spearman_rho"] - partial).abs()
    for index in change.nlargest(3).index:
        row = results.loc[index]
        label = f"{row['gene_a']}–{row['gene_b']}"
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
