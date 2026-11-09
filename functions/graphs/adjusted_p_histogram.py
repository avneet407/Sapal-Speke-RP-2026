"""Histogram of BY-adjusted p-values."""

import numpy as np
import matplotlib.pyplot as plt


def plot_adjusted_p_histogram(adjusted_p_values, fdr_threshold, histogram_path):
    """Log-scale histogram of the adjusted p-values below 1; those equal to 1 are counted in a note."""
    finite_adjusted = adjusted_p_values[np.isfinite(adjusted_p_values)]
    equal_to_one = finite_adjusted == 1.0
    below_one = np.clip(finite_adjusted[~equal_to_one], 1e-12, 1.0)
    minimum_adjusted_p = float(below_one.min()) if below_one.size else 1e-12
    histogram_bins = np.logspace(np.log10(minimum_adjusted_p), 0, 31)

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_counts, _, patches = ax.hist(
        below_one,
        bins=histogram_bins,
        color="#4c78a8",
        alpha=0.8
    )
    ax.bar_label(
        patches,
        labels=[f"{int(count):,}" if count else "" for count in bin_counts],
        padding=2,
        fontsize=8
    )
    ax.axvline(
        fdr_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"BY FDR threshold = {fdr_threshold:.0%}"
    )
    ax.set_xscale("log")
    ax.set_xlim(minimum_adjusted_p, 1.0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    ax.set_xlabel("BY-adjusted p-value below 1 (log scale)")
    ax.set_ylabel("Number of gene-pair tests")
    ax.set_title("Benjamini–Yekutieli adjusted p-value distribution")
    ax.legend()
    if equal_to_one.any():
        ax.text(
            0.98,
            0.95,
            f"p = 1: {int(equal_to_one.sum()):,}",
            transform=ax.transAxes,
            ha="right",
            va="top"
        )
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=300)
    plt.close(fig)
