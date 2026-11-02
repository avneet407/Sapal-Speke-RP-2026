"""
Apply Benjamini-Yekutieli (BY) multiple-testing correction to a gene-gene results table
and save only the tests significant at the configured FDR.

By default this uses the ordinary (non-partial) Spearman p-values from analysis option 1.
To use another p-value family, change P_VALUE_COLUMN and set INPUT_FILE to a master CSV
that contains that column. The correction is always calculated across every row in the
input before significant rows are selected.

Output rows retain the source columns and add:
  by_adjusted_p_value  BY-adjusted p-value for the configured source p-value column
  by_significant       True for rows retained in the filtered output

Run from the repository root:
    python filter_gene_gene_by.py
"""

from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Settings (change these to select the input and hypothesis family) -----
INPUT_FILE = None
# None selects the latest option-1 Spearman master CSV in INPUT_FOLDER.
# Set to a path such as "Results/gene_spearman/<master-result>.csv" to select another result.
P_VALUE_COLUMN = "p_value"
# For region-adjusted analyses, change P_VALUE_COLUMN to "partial_p_value_region"
# and INPUT_FILE to a result from option 2 or 4.
# For disease-adjusted analyses, use "partial_p_value_disease" and option 3 or 4.
FDR_THRESHOLD = 0.05

ROOT_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = ROOT_FOLDER / "Results" / "gene_spearman"
OUTPUT_FOLDER = ROOT_FOLDER / "Results" / "gene_gene_mht"

ADJUSTED_P_COLUMN = "by_adjusted_p_value"
SIGNIFICANCE_COLUMN = "by_significant"


def find_input_file():
    """Choose an explicit source file or the newest ordinary Spearman master result."""
    if INPUT_FILE is not None:
        path = Path(INPUT_FILE)
        if not path.is_absolute():
            path = ROOT_FOLDER / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured input CSV does not exist: {path}")
        return path

    candidates = list(INPUT_FOLDER.glob("gene_spearman_*_01_spearman_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No option-1 Spearman master CSV found in "
            f"{INPUT_FOLDER}. Run gene_spearman.py with option 1 or set INPUT_FILE."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def benjamini_yekutieli_adjust(p_values):
    """Return BY-adjusted p-values in the original order, preserving missing values."""
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")

    adjusted = np.full(values.shape, np.nan, dtype=float)
    if not finite_values.size:
        raise ValueError(f"No finite p-values found in column {P_VALUE_COLUMN!r}.")

    n_tests = finite_values.size
    order = np.argsort(finite_values)
    sorted_p_values = finite_values[order]
    harmonic_number = np.sum(1.0 / np.arange(1, n_tests + 1))
    ranked_adjusted = sorted_p_values * n_tests * harmonic_number / np.arange(
        1, n_tests + 1
    )
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    adjusted_finite = np.empty(n_tests, dtype=float)
    adjusted_finite[order] = np.minimum(ranked_adjusted, 1.0)
    adjusted[finite] = adjusted_finite
    return adjusted


def main():
    if not 0 <= FDR_THRESHOLD <= 1:
        raise ValueError("FDR_THRESHOLD must be between 0 and 1.")

    input_path = find_input_file()
    print(f"Reading master dataset: {input_path}")
    data = pd.read_csv(input_path)
    if P_VALUE_COLUMN not in data.columns:
        raise ValueError(
            f"Input file {input_path} has no {P_VALUE_COLUMN!r} column. "
            f"Available columns: {', '.join(data.columns)}"
        )
    if data.empty:
        raise ValueError(f"Input dataset has no gene-pair tests: {input_path}")

    # Apply BY across the complete family first; filtering before this step would change
    # the number of hypotheses and therefore the correction.
    data[ADJUSTED_P_COLUMN] = benjamini_yekutieli_adjust(data[P_VALUE_COLUMN])
    data[SIGNIFICANCE_COLUMN] = (
        data[ADJUSTED_P_COLUMN] < FDR_THRESHOLD
    )
    filtered = data.loc[data[SIGNIFICANCE_COLUMN]].copy()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    threshold_tag = f"{FDR_THRESHOLD * 100:g}".replace(".", "p")
    input_tag = sha256(input_path.name.encode("utf-8")).hexdigest()[:8]
    output_path = OUTPUT_FOLDER / (
        f"gene_pairs_{input_tag}_BY{threshold_tag}_p.csv"
    )
    filtered.to_csv(output_path, index=False)

    adjusted_p_values = data[ADJUSTED_P_COLUMN].to_numpy(dtype=float)
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
        FDR_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"BY FDR threshold = {FDR_THRESHOLD:.0%}"
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
    histogram_path = output_path.with_name(
        f"{output_path.stem}_adjp_hist.png"
    )
    fig.savefig(histogram_path, dpi=300)
    plt.close(fig)

    print(f"Input hypotheses: {len(data):,}")
    print(f"P-value family: {P_VALUE_COLUMN}")
    print(f"BY FDR threshold: {FDR_THRESHOLD:.1%}")
    print(f"Significant gene-pair tests: {len(filtered):,}")
    print(f"Filtered dataset columns: {', '.join(filtered.columns)}")
    print(f"Saved filtered dataset: {output_path}")
    print(f"Saved BY-adjusted p-value histogram: {histogram_path}")


if __name__ == "__main__":
    main()
