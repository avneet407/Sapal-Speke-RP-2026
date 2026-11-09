"""
Take the significant isoform-gene tests from 3b, count the separate gene-gene pairs they come from,
and draw the matrix of rho and adjusted p-values for the strongest of those pairs.

The input is the corrected table from filter_isoform_gene.py. It also holds one gene-total row per
gene pair, which are significant by construction, so only isoform-vs-gene tests are counted here.

Run with:
    python 4_identifying_candidates/.py/significant_pairs.py
"""

from pathlib import Path
import sys

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.graphs.pair_matrix import plot_gene_pair_matrix


# --- Settings ---------------------------------------------------------------
INPUT_FILE = None
# None selects the latest corrected table from filter_isoform_gene.py.
N_MATRIX_PAIRS = 20
# How many pairs to draw, smallest adjusted p-value first. None = every significant pair.

INPUT_FOLDER = ROOT_FOLDER / "3b_isoform_gene_hypothesis_testing" / "CSV"
CSV_FOLDER = ROOT_FOLDER / "4_identifying_candidates" / "CSV"
FIGURES_FOLDER = ROOT_FOLDER / "4_identifying_candidates" / "Figures"


def find_input_file():
    """Return an explicitly configured input or the latest corrected table."""
    if INPUT_FILE is not None:
        path = Path(INPUT_FILE)
        if not path.is_absolute():
            path = ROOT_FOLDER / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured correlation CSV does not exist: {path}")
        return path

    candidates = list(INPUT_FOLDER.glob("isoform_gene_correlations_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No corrected table found in {INPUT_FOLDER}. Run filter_isoform_gene.py first or set INPUT_FILE."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main():
    input_path = find_input_file()
    print(f"Reading: {input_path}")
    results = pd.read_csv(input_path, low_memory=False)

    # An isoform test has an isoform share on one side; gene-total rows have gene CPM on both
    isoform_gene_tests = (
        (results["component_a_type"] == "isoform_percent_share")
        | (results["component_b_type"] == "isoform_percent_share")
    )
    significant = results.loc[isoform_gene_tests & results["mht_significant"]]
    gene_pairs = significant["pair_id"].nunique()

    print(f"Isoform-gene tests: {isoform_gene_tests.sum():,}")
    print(f"Significant isoform-gene tests: {len(significant):,}")
    print(f"Separate gene-gene pairs with a significant result: {gene_pairs:,}")

    CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = CSV_FOLDER / f"significant_{input_path.name}"
    significant.to_csv(output_path, index=False)
    print(f"Saved significant tests: {output_path}")

    # Rank each pair by its smallest adjusted p-value; ties stay in pair_id order
    strongest = significant.groupby("pair_id")["mht_adjusted_p_value"].min().sort_values(kind="stable")
    selected_pairs = strongest.index[:N_MATRIX_PAIRS]
    gene_rows = results.loc[~isoform_gene_tests].set_index("pair_id", drop=False).loc[selected_pairs]
    mht_method = results.loc[isoform_gene_tests, "mht_method"].iloc[0]

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    for rank, pair in enumerate(gene_rows.itertuples(index=False), start=1):
        plot_path = FIGURES_FOLDER / f"candidate{rank}_{pair.gene_a}_{pair.gene_b}.png"
        print(f"Saving matrix {rank}/{len(gene_rows)}...", flush=True)
        plot_gene_pair_matrix(results, pair, plot_path, mht_method)
        print(f"Saved matrix: {plot_path}")


if __name__ == "__main__":
    main()
