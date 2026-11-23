"""
For the genes that are the asymmetric gene in several candidate pairs, how consistently is each of
their isoforms significant across those pairs?

For each of the top N_GENES asymmetric genes (most candidate pairs first, at least MIN_PAIRS pairs):
    - take every candidate pair where it is the asymmetric gene
    - for each of its isoforms, the percentage of those pairs where that isoform's share was
      significant against the partner (silent) gene's total
      (100% = significant in every pair, 0% = tested in every pair but never significant)

Isoform numbers match the isoform visualisation (1_data_processing/.py/isoform_visualisation.py):
all isoforms of the gene ranked by mean % share of the gene across the 45 adult samples (the same
measure as the 1.5% share filter), 1 = highest share = top row of that figure. Because the ranking
and the filter use the same measure, the tested isoforms are exactly isoforms 1 to N with no gaps.
Blank means "not tested" (or the gene has fewer isoforms), not "never significant".

Outputs (in 4_identifying_candidates/CSV):
    isoform_consistency_<settings>.csv      - rows = asymmetric gene, columns = isoform number,
                                              cells = % of the gene's candidate pairs significant
    isoform_consistency_key_<settings>.csv  - one row per gene and tested isoform: its number, ID,
                                              mean CPM, mean share and the counts behind each %

Run with:
    python 4_identifying_candidates/.py/checking_across_pairs.py
"""

from pathlib import Path
import sys

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.data_loading.expression_data import (
    load_matrix, mean_isoform_share, rank_isoforms_by_share, read_isoform_genes,
)


# --- Settings ---------------------------------------------------------------
CANDIDATES_FILE = None
# None selects the latest promising_candidate_pairs_*.csv from promising_candidates.py.
N_GENES = 10
# How many asymmetric genes to include, most candidate pairs first.
MIN_PAIRS = 2
# A gene must be the asymmetric gene in at least this many pairs to be included.

MATRIX_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv"
GTF_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf"
CORRELATIONS_FOLDER = ROOT_FOLDER / "3b_isoform_gene_hypothesis_testing" / "CSV"
CSV_FOLDER = ROOT_FOLDER / "4_identifying_candidates" / "CSV"


def find_candidates_file():
    """Return an explicitly configured candidates file or the latest one."""
    if CANDIDATES_FILE is not None:
        path = Path(CANDIDATES_FILE)
        if not path.is_absolute():
            path = ROOT_FOLDER / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured candidates CSV does not exist: {path}")
        return path

    candidates = list(CSV_FOLDER.glob("promising_candidate_pairs_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No candidate pairs found in {CSV_FOLDER}. Run promising_candidates.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def isoform_numbers(genes):
    """Isoform number for every isoform of the given genes, as in the isoform visualisation:
    all of the gene's isoforms ranked by mean % share of the gene across the adult samples,
    1 = highest share. Uses the same shared functions as the visualisation, so the numbers match.
    Returns (number, mean CPM), both indexed by isoform ID."""
    matrix = load_matrix(str(MATRIX_PATH))
    matrix = matrix[[name for name in matrix.columns if "_Fetal_" not in name]]
    mean_cpm = matrix.mean(axis=1)

    isoform_gene = read_isoform_genes(str(GTF_PATH))   # in GTF order, like the visualisation
    mean_share = mean_isoform_share(matrix, isoform_gene)
    numbers = {}
    for gene in genes:
        isoforms = list(isoform_gene.index[isoform_gene == gene])
        for number, isoform_id in enumerate(rank_isoforms_by_share(isoforms, mean_share), start=1):
            numbers[isoform_id] = number
    numbers = pd.Series(numbers, name="isoform_number")
    return numbers, mean_cpm.reindex(numbers.index)


def main():
    candidates_path = find_candidates_file()
    settings_tag = candidates_path.stem.removeprefix("promising_candidate_pairs_")
    correlations_path = CORRELATIONS_FOLDER / f"isoform_gene_correlations_{settings_tag}.csv"
    print(f"Reading candidates: {candidates_path}")
    print(f"Reading isoform tests: {correlations_path}")
    candidates = pd.read_csv(candidates_path)
    results = pd.read_csv(correlations_path, low_memory=False)

    # Top asymmetric genes by number of candidate pairs
    pair_counts = candidates["asymmetric_gene"].value_counts()
    pair_counts = pair_counts[pair_counts >= MIN_PAIRS]
    pair_counts = pair_counts.sort_index().sort_values(ascending=False, kind="stable")   # ties by name
    top_genes = pair_counts.index[:N_GENES]
    print(f"Asymmetric genes in at least {MIN_PAIRS} pairs: {len(pair_counts):,}; using the top {len(top_genes)}")

    # Every tested isoform share, from either side, as: pair, gene the isoform belongs to, isoform, significant
    tested = results.loc[results["spearman_rho"].notna()]
    sides = []
    for side in ["a", "b"]:
        rows = tested.loc[tested[f"component_{side}_type"] == "isoform_percent_share"]
        sides.append(pd.DataFrame({
            "pair_id": rows["pair_id"].values,
            "gene": rows[f"gene_{side}"].values,
            "isoform_id": rows[f"isoform_{side}_id"].astype(str).values,
            "mean_share_pct": rows[f"mean_share_{side}_pct"].values,
            "significant": rows["mht_significant"].astype(bool).values,
        }))
    isoform_tests = pd.concat(sides, ignore_index=True)

    # Keep only the asymmetric side of each candidate pair, for the top genes
    asymmetric_sides = candidates.loc[candidates["asymmetric_gene"].isin(top_genes), ["pair_id", "asymmetric_gene"]]
    asymmetric_sides = asymmetric_sides.rename(columns={"asymmetric_gene": "gene"})
    isoform_tests = isoform_tests.merge(asymmetric_sides, on=["pair_id", "gene"])

    # Per gene and isoform: in how many of the gene's candidate pairs it was tested / significant
    key = isoform_tests.groupby(["gene", "isoform_id"]).agg(
        mean_share_pct=("mean_share_pct", "first"),
        n_pairs_tested=("significant", "size"),
        n_pairs_significant=("significant", "sum"),
    ).reset_index()
    key["percent_significant"] = 100 * key["n_pairs_significant"] / key["n_pairs_tested"]

    numbers, mean_cpm = isoform_numbers(top_genes)
    key["isoform_number"] = key["isoform_id"].map(numbers).astype(int)
    key["mean_cpm"] = key["isoform_id"].map(mean_cpm)
    key["n_candidate_pairs"] = key["gene"].map(pair_counts)
    key["gene"] = pd.Categorical(key["gene"], categories=top_genes, ordered=True)
    key = key.sort_values(["gene", "isoform_number"])[[
        "gene", "n_candidate_pairs", "isoform_number", "isoform_id", "mean_cpm", "mean_share_pct",
        "n_pairs_tested", "n_pairs_significant", "percent_significant",
    ]]

    # Table: rows = gene, columns = isoform number, cells = % significant (blank = not tested)
    table = key.pivot(index="gene", columns="isoform_number", values="percent_significant").round(1)
    table = table.reindex(columns=range(1, int(key["isoform_number"].max()) + 1))
    table.columns = [str(number) for number in table.columns]
    table.insert(0, "n_candidate_pairs", pair_counts.loc[top_genes].values)
    table.index.name = "asymmetric_gene"

    CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    table_path = CSV_FOLDER / f"isoform_consistency_{settings_tag}.csv"
    key_path = CSV_FOLDER / f"isoform_consistency_key_{settings_tag}.csv"
    table.to_csv(table_path)
    key.to_csv(key_path, index=False)
    print(f"Saved table: {table_path}")
    print(f"Saved key: {key_path}")

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n% of each gene's candidate pairs in which the isoform is significant (blank = not tested):")
        print(table.fillna("").to_string())


if __name__ == "__main__":
    main()
