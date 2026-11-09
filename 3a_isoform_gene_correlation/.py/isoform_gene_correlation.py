"""
Calculate isoform-share to partner-gene correlations for BY-significant gene pairs.

The input is the filtered CSV produced by filter_gene_gene_by.py. Genes are filtered
using CPM > MIN_CPM in at least MIN_FRACTION_SAMPLES of adult samples, then isoforms
are retained when their mean share of the gene is at least MIN_SHARE.

For each retained gene pair A-B, the output contains:
  - the original A-vs-B gene-total correlation;
  - each A isoform's percentage share vs B total CPM;
  - each B isoform's percentage share vs A total CPM.

The output table holds the raw correlations only. Multiple-testing correction and the
example figures are in 3b_isoform_gene_hypothesis_testing/.py/filter_isoform_gene.py.

Run with:
    python 3a_isoform_gene_correlation/.py/isoform_gene_correlation.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.data_loading.expression_data import gene_of, load_matrix
from functions.spearman.spearman import correlation


# --- Settings ---------------------------------------------------------------
INPUT_FILE = None
# None selects the latest raw-p-value BY output from filter_gene_gene_by.py.
MIN_CPM = 1
MIN_FRACTION_SAMPLES = 0.8
MIN_SHARE = 0.015
PROGRESS_INTERVAL_SECONDS = 15

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
INPUT_FOLDER = ROOT_FOLDER / "2b_gene_gene_hypothesis_testing" / "CSV"
CSV_FOLDER = ROOT_FOLDER / "3a_isoform_gene_correlation" / "CSV"


def find_input_file():
    """Return an explicitly configured input or the latest raw-p BY-filtered CSV."""
    if INPUT_FILE is not None:
        path = Path(INPUT_FILE)
        if not path.is_absolute():
            path = ROOT_FOLDER / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured gene-pair CSV does not exist: {path}")
        return path

    candidates = list(INPUT_FOLDER.glob("gene_pairs_BY*_p.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No raw-p BY-filtered gene-pair CSV found in "
            f"{INPUT_FOLDER}. Run filter_gene_gene_by.py first or set INPUT_FILE."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def report_progress(label, completed, total, started_at, last_report_at):
    """Print progress no less often than the configured interval during pair calculations."""
    now = time.monotonic()
    if completed == total or now - last_report_at >= PROGRESS_INTERVAL_SECONDS:
        elapsed = now - started_at
        print(
            f"{label}: {completed:,}/{total:,} gene pairs "
            f"({completed / total:.1%}); elapsed {elapsed:.0f}s",
            flush=True
        )
        return now
    return last_report_at


def add_result(
    results,
    pair,
    component_a,
    component_b,
    component_a_type,
    component_b_type,
    rho,
    p_value,
    n_samples,
    isoform_a_id="",
    isoform_b_id="",
    mean_share_a_pct=np.nan,
    mean_share_b_pct=np.nan,
):
    results.append({
        "pair_id": f"{pair.gene_a}__{pair.gene_b}",
        "gene_a": pair.gene_a,
        "gene_a_name": pair.gene_a_name,
        "gene_b": pair.gene_b,
        "gene_b_name": pair.gene_b_name,
        "component_a": component_a,
        "component_a_type": component_a_type,
        "isoform_a_id": isoform_a_id,
        "mean_share_a_pct": mean_share_a_pct,
        "component_b": component_b,
        "component_b_type": component_b_type,
        "isoform_b_id": isoform_b_id,
        "mean_share_b_pct": mean_share_b_pct,
        "spearman_rho": rho,
        "p_value": p_value,
        "n_samples": n_samples,
        "gene_pair_spearman_rho": pair.spearman_rho,
        "gene_pair_p_value": pair.p_value,
        "gene_pair_by_adjusted_p_value": pair.by_adjusted_p_value,
        "gene_pair_by_significant": getattr(pair, "by_significant", np.nan),
        "correction_family": (
            "upstream_gene_gene"
            if component_a_type == "gene_cpm" and component_b_type == "gene_cpm"
            else "all_isoform_gene_tests"
        ),
    })


def main():
    if MIN_CPM < 0:
        raise ValueError("MIN_CPM must be non-negative.")
    if not 0 <= MIN_FRACTION_SAMPLES <= 1:
        raise ValueError("MIN_FRACTION_SAMPLES must be between 0 and 1.")
    if not 0 <= MIN_SHARE <= 1:
        raise ValueError("MIN_SHARE must be between 0 and 1.")

    input_path = find_input_file()
    pairs = pd.read_csv(input_path)
    required_columns = {
        "gene_a", "gene_a_name", "gene_b", "gene_b_name", "spearman_rho",
        "p_value", "by_adjusted_p_value", "by_significant"
    }
    missing_columns = required_columns.difference(pairs.columns)
    if missing_columns:
        raise ValueError(
            f"Input file {input_path} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if pairs.empty:
        raise ValueError(f"Input gene-pair dataset is empty: {input_path}")
    pairs[["gene_a_name", "gene_b_name"]] = pairs[
        ["gene_a_name", "gene_b_name"]
    ].fillna("")
    if pairs.duplicated(["gene_a", "gene_b"]).any():
        raise ValueError(f"Input has duplicate gene pairs: {input_path}")
    if "by_significant" in pairs and not pairs["by_significant"].all():
        raise ValueError(
            f"Input contains rows not marked BY-significant: {input_path}. "
            "Pass the filtered output from filter_gene_gene_by.py."
        )

    matrix_path = ROOT_FOLDER / MATRIX_PATH
    settings_tag = f"cpm{MIN_CPM:g}_frac{MIN_FRACTION_SAMPLES:.0%}_share{MIN_SHARE * 100:g}pct"
    CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = CSV_FOLDER / f"isoform_gene_raw_correlations_{settings_tag}.csv"

    expression = load_matrix(str(matrix_path))
    expression = expression[
        [sample for sample in expression.columns if "_Fetal_" not in sample]
    ]
    if expression.empty:
        raise ValueError(f"No adult samples found in {matrix_path}.")

    isoform_gene = pd.Series(
        [gene_of(str(isoform_id)) for isoform_id in expression.index],
        index=expression.index
    )
    gene_cpm = expression.groupby(isoform_gene, sort=False).sum()
    fraction_expressed = (gene_cpm > MIN_CPM).mean(axis=1)
    kept_genes = gene_cpm.index[
        fraction_expressed >= MIN_FRACTION_SAMPLES
    ]

    kept_isoforms = expression.loc[isoform_gene.isin(kept_genes)]
    kept_isoform_gene = isoform_gene.loc[kept_isoforms.index]
    own_gene_cpm = gene_cpm.loc[kept_isoform_gene].set_axis(kept_isoforms.index)
    shares = kept_isoforms.div(own_gene_cpm.where(own_gene_cpm > 0)) * 100
    mean_share_pct = shares.mean(axis=1)
    share_filter = mean_share_pct >= MIN_SHARE * 100
    shares = shares.loc[share_filter]
    mean_share_pct = mean_share_pct.loc[share_filter]
    mean_isoform_cpm = kept_isoforms.mean(axis=1).loc[share_filter]
    isoform_gene = kept_isoform_gene.loc[share_filter]

    isoforms_by_gene = {}
    for gene_id in kept_genes:
        isoforms = isoform_gene.index[isoform_gene == gene_id]
        isoforms = sorted(
            isoforms,
            key=lambda isoform_id: (
                -mean_isoform_cpm[isoform_id],
                str(isoform_id)
            )
        )
        isoforms_by_gene[gene_id] = isoforms

    genes_in_input = set(pairs["gene_a"]).union(pairs["gene_b"])
    missing_genes = genes_in_input.difference(gene_cpm.index)
    if missing_genes:
        raise ValueError(
            "The expression matrix is missing genes from the filtered "
            f"interaction file: {sorted(missing_genes)[:10]}"
        )
    eligible_mask = (
        pairs["gene_a"].isin(kept_genes)
        & pairs["gene_b"].isin(kept_genes)
    )
    eligible_pairs = pairs.loc[eligible_mask].copy().reset_index(drop=True)
    if eligible_pairs.empty:
        raise ValueError(
            "No gene pairs remain after the gene-expression filter. "
            "Check MIN_CPM, MIN_FRACTION_SAMPLES, and the input analysis."
        )

    if "n_samples" in eligible_pairs:
        observed_sample_counts = set(
            eligible_pairs["n_samples"].dropna().astype(int)
        )
        if observed_sample_counts and observed_sample_counts != {len(expression.columns)}:
            raise ValueError(
                "The filtered gene-pair results and expression matrix use "
                f"different sample counts: {sorted(observed_sample_counts)} vs "
                f"{len(expression.columns)} adult samples."
            )

    results = []
    total_pairs = len(eligible_pairs)
    started_at = time.monotonic()
    last_report_at = started_at
    for pair_number, pair in enumerate(
        eligible_pairs.itertuples(index=False),
        start=1
    ):
        if pair_number == 1:
            print(
                f"Calculating raw isoform-gene correlations for "
                f"{total_pairs:,} gene pairs.",
                flush=True
            )
        last_report_at = report_progress(
            "Correlation progress",
            pair_number,
            total_pairs,
            started_at,
            last_report_at
        )

        add_result(
            results, pair, "A", "B", "gene_cpm", "gene_cpm",
            float(pair.spearman_rho), float(pair.p_value),
            int(pair.n_samples) if hasattr(pair, "n_samples") else len(expression.columns)
        )

        for index, isoform_id in enumerate(
            isoforms_by_gene[pair.gene_a],
            start=1
        ):
            rho, p_value, n_samples = correlation(
                shares.loc[isoform_id], gene_cpm.loc[pair.gene_b]
            )
            add_result(
                results, pair, f"A{index}", "B",
                "isoform_percent_share", "gene_cpm",
                rho, p_value, n_samples,
                isoform_a_id=str(isoform_id),
                mean_share_a_pct=float(mean_share_pct[isoform_id])
            )

        for index, isoform_id in enumerate(
            isoforms_by_gene[pair.gene_b],
            start=1
        ):
            rho, p_value, n_samples = correlation(
                shares.loc[isoform_id], gene_cpm.loc[pair.gene_a]
            )
            add_result(
                results, pair, "A", f"B{index}",
                "gene_cpm", "isoform_percent_share",
                rho, p_value, n_samples,
                isoform_b_id=str(isoform_id),
                mean_share_b_pct=float(mean_share_pct[isoform_id])
            )

    result_table = pd.DataFrame(results)
    result_table.to_csv(output_path, index=False)
    print(f"Saved raw correlations: {output_path}", flush=True)


if __name__ == "__main__":
    main()
