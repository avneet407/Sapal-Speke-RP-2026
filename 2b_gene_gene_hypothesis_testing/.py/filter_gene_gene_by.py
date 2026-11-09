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

Run with:
    python 2b_gene_gene_hypothesis_testing/.py/filter_gene_gene_by.py
"""

from pathlib import Path
import sys

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.graphs.adjusted_p_histogram import plot_adjusted_p_histogram
from functions.hypothesis_testing.fdr import benjamini_yekutieli_adjust


# --- Settings (change these to select the input and hypothesis family) -----
INPUT_FILE = None
# None selects the latest option-1 Spearman master CSV in INPUT_FOLDER.
# Set to a path such as "2a_gene_gene_correlation/CSV/<master-result>.csv" to select another result.
P_VALUE_COLUMN = "p_value"
# For region-adjusted analyses, change P_VALUE_COLUMN to "partial_p_value_region"
# and INPUT_FILE to a result from option 2 or 4.
# For disease-adjusted analyses, use "partial_p_value_disease" and option 3 or 4.
FDR_THRESHOLD = 0.05

INPUT_FOLDER = ROOT_FOLDER / "2a_gene_gene_correlation" / "CSV"
OUTPUT_FOLDER = ROOT_FOLDER / "2b_gene_gene_hypothesis_testing" / "CSV"
FIGURES_FOLDER = ROOT_FOLDER / "2b_gene_gene_hypothesis_testing" / "Figures"

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
    output_path = OUTPUT_FOLDER / f"gene_pairs_BY{threshold_tag}_p.csv"
    filtered.to_csv(output_path, index=False)

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    histogram_path = FIGURES_FOLDER / f"{output_path.stem}_adjp_hist.png"
    plot_adjusted_p_histogram(
        data[ADJUSTED_P_COLUMN].to_numpy(dtype=float),
        FDR_THRESHOLD,
        histogram_path
    )

    print(f"Input hypotheses: {len(data):,}")
    print(f"P-value family: {P_VALUE_COLUMN}")
    print(f"BY FDR threshold: {FDR_THRESHOLD:.1%}")
    print(f"Significant gene-pair tests: {len(filtered):,}")
    print(f"Filtered dataset columns: {', '.join(filtered.columns)}")
    print(f"Saved filtered dataset: {output_path}")
    print(f"Saved BY-adjusted p-value histogram: {histogram_path}")


if __name__ == "__main__":
    main()
