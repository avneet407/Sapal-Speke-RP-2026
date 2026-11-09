"""
Apply multiple-testing correction to the isoform-gene correlations and draw example matrices.

The input is the raw correlation table from 3a_isoform_gene_correlation/.py/isoform_gene_correlation.py.
All isoform-vs-gene tests share one correction; gene-total rows keep their upstream
gene-gene BY result.

Each example pair is drawn as a matrix with B components as rows and A components
as columns. Isoform-to-isoform correlations are not calculated in this version; those
cells are blank and can be added later without changing the output matrix layout.

Run with:
    python 3b_isoform_gene_hypothesis_testing/.py/filter_isoform_gene.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.graphs.pair_matrix import plot_pair_matrix
from functions.hypothesis_testing.fdr import adjust_p_values


# --- Settings ---------------------------------------------------------------
INPUT_FILE = None
# None selects the latest raw correlation table from isoform_gene_correlation.py.
N_EXAMPLE_PAIRS = 3
RANDOM_SEED = 2026
MHT_METHOD = "by"  # "by" or "bh"; correction and threshold do not invalidate raw correlations
FDR_THRESHOLD = 0.05

INPUT_FOLDER = ROOT_FOLDER / "3a_isoform_gene_correlation" / "CSV"
CSV_FOLDER = ROOT_FOLDER / "3b_isoform_gene_hypothesis_testing" / "CSV"
FIGURES_FOLDER = ROOT_FOLDER / "3b_isoform_gene_hypothesis_testing" / "Figures"


def find_input_file():
    """Return an explicitly configured input or the latest raw correlation table."""
    if INPUT_FILE is not None:
        path = Path(INPUT_FILE)
        if not path.is_absolute():
            path = ROOT_FOLDER / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured correlation CSV does not exist: {path}")
        return path

    candidates = list(INPUT_FOLDER.glob("isoform_gene_raw_correlations_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No raw correlation table found in "
            f"{INPUT_FOLDER}. Run isoform_gene_correlation.py first or set INPUT_FILE."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main():
    if not 0 <= FDR_THRESHOLD <= 1:
        raise ValueError("FDR_THRESHOLD must be between 0 and 1.")
    if MHT_METHOD not in {"by", "bh"}:
        raise ValueError("MHT_METHOD must be 'by' or 'bh'.")

    input_path = find_input_file()
    # The settings are in the raw table's file name, e.g. cpm1_frac80%_share1.5pct
    settings_tag = input_path.stem.removeprefix("isoform_gene_raw_correlations_")
    CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    mht_tag = f"{MHT_METHOD.upper()}_FDR{FDR_THRESHOLD * 100:g}".replace(".", "p")
    output_path = CSV_FOLDER / f"isoform_gene_correlations_{settings_tag}_{mht_tag}.csv"

    print(f"Loading raw correlations: {input_path}", flush=True)
    # round_trip: read back exactly the values that were calculated
    result_table = pd.read_csv(input_path, float_precision="round_trip")
    required_columns = {
        "pair_id", "gene_a", "gene_a_name", "gene_b", "gene_b_name",
        "component_a", "component_a_type", "component_b", "component_b_type",
        "spearman_rho", "p_value", "n_samples", "gene_pair_by_adjusted_p_value",
        "gene_pair_by_significant", "correction_family"
    }
    missing_columns = required_columns.difference(result_table.columns)
    if missing_columns:
        raise ValueError(
            f"Correlation table is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    print(f"Loaded {len(result_table):,} correlation rows.", flush=True)
    result_table[["gene_a_name", "gene_b_name"]] = result_table[
        ["gene_a_name", "gene_b_name"]
    ].fillna("")

    isoform_gene_tests = (
        (result_table["component_a_type"] == "isoform_percent_share")
        | (result_table["component_b_type"] == "isoform_percent_share")
    )
    correction_started_at = time.monotonic()
    print(
        f"Applying {MHT_METHOD.upper()} correction to "
        f"{int(result_table.loc[isoform_gene_tests, 'p_value'].notna().sum()):,} "
        "finite isoform-gene p-values.",
        flush=True
    )
    result_table["mht_adjusted_p_value"] = np.nan
    result_table.loc[~isoform_gene_tests, "mht_adjusted_p_value"] = (
        result_table.loc[~isoform_gene_tests, "gene_pair_by_adjusted_p_value"]
    )
    result_table.loc[isoform_gene_tests, "mht_adjusted_p_value"] = adjust_p_values(
        result_table.loc[isoform_gene_tests, "p_value"],
        MHT_METHOD
    )
    result_table["mht_method"] = np.where(
        isoform_gene_tests,
        MHT_METHOD.upper(),
        "BY_UPSTREAM"
    )
    result_table["mht_threshold"] = np.where(
        isoform_gene_tests,
        FDR_THRESHOLD,
        np.nan
    )
    result_table["mht_significant"] = False
    result_table.loc[~isoform_gene_tests, "mht_significant"] = (
        result_table.loc[~isoform_gene_tests, "gene_pair_by_significant"]
        .fillna(False)
        .astype(bool)
    )
    result_table.loc[isoform_gene_tests, "mht_significant"] = (
        result_table.loc[isoform_gene_tests, "mht_adjusted_p_value"] < FDR_THRESHOLD
    )
    print(
        f"FDR adjustment and thresholding completed in "
        f"{time.monotonic() - correction_started_at:.1f}s.",
        flush=True
    )
    if MHT_METHOD == "by":
        result_table["by_adjusted_p_value"] = result_table["mht_adjusted_p_value"]
        result_table["by_significant"] = result_table["mht_significant"]

    result_table.to_csv(output_path, index=False)
    print(f"Correlation rows saved: {len(result_table):,}")
    print(
        f"Isoform-gene p-values {MHT_METHOD.upper()}-adjusted: "
        f"{int(result_table.loc[isoform_gene_tests, 'p_value'].notna().sum()):,} finite tests; "
        f"{int(result_table.loc[isoform_gene_tests, 'mht_significant'].sum()):,} significant "
        f"at FDR {FDR_THRESHOLD:.1%}"
    )
    print(f"Used raw correlations: {input_path}", flush=True)
    print(f"Saved full correlation table: {output_path}")

    gene_pair_rows = result_table.loc[~isoform_gene_tests].drop_duplicates(
        ["gene_a", "gene_b"]
    )
    selected_pairs = gene_pair_rows.sample(
        n=min(N_EXAMPLE_PAIRS, len(gene_pair_rows)),
        random_state=RANDOM_SEED
    )
    for example_number, pair in enumerate(
        selected_pairs.itertuples(index=False),
        start=1
    ):
        pair_rows = result_table[
            (result_table["gene_a"] == pair.gene_a)
            & (result_table["gene_b"] == pair.gene_b)
        ]
        a_rows = pair_rows.loc[
            pair_rows["component_a_type"] == "isoform_percent_share"
        ].sort_values(
            "component_a",
            key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
        )
        b_rows = pair_rows.loc[
            pair_rows["component_b_type"] == "isoform_percent_share"
        ].sort_values(
            "component_b",
            key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
        )
        column_labels = ["A"] + a_rows["component_a"].tolist()
        row_labels = ["B"] + b_rows["component_b"].tolist()
        rho_matrix = np.full(
            (len(row_labels), len(column_labels)),
            np.nan,
            dtype=float
        )
        adjusted_p_matrix = np.full_like(rho_matrix, np.nan)
        gene_row = pair_rows.loc[
            (pair_rows["component_a"] == "A")
            & (pair_rows["component_b"] == "B")
        ].iloc[0]
        rho_matrix[0, 0] = gene_row["spearman_rho"]
        adjusted_p_matrix[0, 0] = gene_row["mht_adjusted_p_value"]
        for column_index, row in enumerate(a_rows.itertuples(index=False), start=1):
            rho_matrix[0, column_index] = row.spearman_rho
            adjusted_p_matrix[0, column_index] = row.mht_adjusted_p_value
        for row_index, row in enumerate(b_rows.itertuples(index=False), start=1):
            rho_matrix[row_index, 0] = row.spearman_rho
            adjusted_p_matrix[row_index, 0] = row.mht_adjusted_p_value

        filename = (
            f"{output_path.stem}_example{example_number}_"
            f"{pair.gene_a}_{pair.gene_b}.png"
        )
        plot_path = FIGURES_FOLDER / filename
        print(f"Saving example matrix {example_number}/{len(selected_pairs)}...", flush=True)
        plot_pair_matrix(
            pair,
            rho_matrix,
            adjusted_p_matrix,
            row_labels,
            column_labels,
            plot_path,
            MHT_METHOD
        )
        print(f"Saved example matrix: {plot_path}")


if __name__ == "__main__":
    main()
