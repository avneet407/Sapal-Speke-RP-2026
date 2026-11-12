"""
Find gene pairs with isoform asymmetry: the promising candidates.

Every pair in the 3b table is already significant at the gene-gene stage (gene A and gene B are
correlated). For each pair there are two sides:
    side A = each isoform share of gene A tested against the total of gene B
    side B = each isoform share of gene B tested against the total of gene A

A pair is a promising candidate when one side is asymmetric and the other is silent:
    - the asymmetric side has at least one significant isoform, but not all of its isoforms are
      significant (so the isoform mix matters, not just how much of the gene there is), and
    - the silent side has isoforms that were tested, and none of them are significant.

Only isoforms that were tested count (mean share >= 1.5%, and a rho could be calculated).
A side with no tested isoforms is not evidence of anything, so such pairs are skipped.

Outputs (in 4_identifying_candidates/CSV):
    promising_candidate_pairs_<input settings>.csv  - one row per candidate pair
    promising_candidate_genes_<input settings>.csv  - one row per gene, how often it is the
                                                      asymmetric or the silent gene

Run with:
    python 4_identifying_candidates/.py/promising_candidates.py
"""

from pathlib import Path

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder


# --- Settings ---------------------------------------------------------------
INPUT_FILE = None
# None selects the latest corrected table from filter_isoform_gene.py (all tests, not only significant).
N_PRINT = 20
# How many candidate pairs to print, smallest adjusted p-value first.

INPUT_FOLDER = ROOT_FOLDER / "3b_isoform_gene_hypothesis_testing" / "CSV"
CSV_FOLDER = ROOT_FOLDER / "4_identifying_candidates" / "CSV"


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


def summarise_side(tests, isoform_column):
    """Count tested and significant isoforms per pair for one side (A or B)."""
    significant = tests.loc[tests["mht_significant"]]
    summary = tests.groupby("pair_id").agg(n_tested=(isoform_column, "size"))
    summary["n_significant"] = significant.groupby("pair_id").size()
    summary["n_significant"] = summary["n_significant"].fillna(0).astype(int)

    # Details of the significant isoforms, for the asymmetric side
    summary["significant_isoforms"] = significant.groupby("pair_id")[isoform_column].agg(";".join)
    summary["significant_rhos"] = significant.groupby("pair_id")["spearman_rho"].agg(
        lambda rhos: ";".join(f"{rho:.3f}" for rho in rhos)
    )
    summary["min_adjusted_p"] = significant.groupby("pair_id")["mht_adjusted_p_value"].min()
    return summary


def main():
    input_path = find_input_file()
    settings_tag = input_path.stem.removeprefix("isoform_gene_correlations_")
    print(f"Reading: {input_path}")
    results = pd.read_csv(input_path, low_memory=False)

    # One gene-total row per pair holds the gene names and the gene-gene correlation
    gene_rows = results.loc[
        (results["component_a_type"] == "gene_cpm") & (results["component_b_type"] == "gene_cpm"),
        ["pair_id", "gene_a", "gene_b", "spearman_rho", "mht_adjusted_p_value"],
    ].rename(columns={"spearman_rho": "gene_gene_rho", "mht_adjusted_p_value": "gene_gene_adjusted_p"})
    gene_rows = gene_rows.set_index("pair_id")

    # Only isoform tests with a rho count as "tested" (single-isoform shares give blanks)
    tested = results.loc[results["spearman_rho"].notna()]
    side_a = tested.loc[tested["component_a_type"] == "isoform_percent_share"]
    side_b = tested.loc[tested["component_b_type"] == "isoform_percent_share"]

    summary_a = summarise_side(side_a, "isoform_a_id")
    summary_b = summarise_side(side_b, "isoform_b_id")
    pairs = gene_rows.join(summary_a.add_suffix("_a")).join(summary_b.add_suffix("_b"))
    for column in ["n_tested_a", "n_significant_a", "n_tested_b", "n_significant_b"]:
        pairs[column] = pairs[column].fillna(0).astype(int)

    # Asymmetric = some but not all tested isoforms significant; silent = tested, none significant
    asymmetric_a = (pairs["n_significant_a"] >= 1) & (pairs["n_significant_a"] < pairs["n_tested_a"])
    asymmetric_b = (pairs["n_significant_b"] >= 1) & (pairs["n_significant_b"] < pairs["n_tested_b"])
    silent_a = (pairs["n_tested_a"] >= 1) & (pairs["n_significant_a"] == 0)
    silent_b = (pairs["n_tested_b"] >= 1) & (pairs["n_significant_b"] == 0)

    print(f"Gene pairs (all gene-gene significant): {len(pairs):,}")
    print(f"Pairs with any significant isoform test: {((pairs['n_significant_a'] + pairs['n_significant_b']) > 0).sum():,}")
    print(f"Pairs failing because both sides have significant isoforms: "
          f"{((pairs['n_significant_a'] > 0) & (pairs['n_significant_b'] > 0)).sum():,}")
    print(f"Pairs failing because every tested isoform on the significant side is significant: "
          f"{(((pairs['n_significant_a'] > 0) & ~asymmetric_a & silent_b) | ((pairs['n_significant_b'] > 0) & ~asymmetric_b & silent_a)).sum():,}")
    print(f"Pairs failing because the other side has no tested isoforms: "
          f"{(((pairs['n_significant_a'] > 0) & (pairs['n_tested_b'] == 0)) | ((pairs['n_significant_b'] > 0) & (pairs['n_tested_a'] == 0))).sum():,}")

    # Write each candidate with the asymmetric gene first, whichever of A or B it is
    rows = []
    for side, other, keep in [("a", "b", asymmetric_a & silent_b), ("b", "a", asymmetric_b & silent_a)]:
        chosen = pairs.loc[keep]
        rows.append(pd.DataFrame({
            "pair_id": chosen.index,
            "asymmetric_gene": chosen[f"gene_{side}"].values,
            "silent_gene": chosen[f"gene_{other}"].values,
            "gene_gene_rho": chosen["gene_gene_rho"].values,
            "gene_gene_adjusted_p": chosen["gene_gene_adjusted_p"].values,
            "asymmetric_isoforms_tested": chosen[f"n_tested_{side}"].values,
            "asymmetric_isoforms_significant": chosen[f"n_significant_{side}"].values,
            "silent_isoforms_tested": chosen[f"n_tested_{other}"].values,
            "significant_isoforms": chosen[f"significant_isoforms_{side}"].values,
            "significant_rhos": chosen[f"significant_rhos_{side}"].values,
            "min_isoform_adjusted_p": chosen[f"min_adjusted_p_{side}"].values,
        }))
    candidates = pd.concat(rows, ignore_index=True).sort_values(
        ["min_isoform_adjusted_p", "pair_id"], kind="stable"
    )

    # Per gene: how many candidate pairs it is the asymmetric gene in, and as the silent gene
    genes = pd.DataFrame({
        "n_pairs_as_asymmetric_gene": candidates["asymmetric_gene"].value_counts(),
        "n_pairs_as_silent_gene": candidates["silent_gene"].value_counts(),
    }).fillna(0).astype(int)
    genes["silent_partners"] = candidates.groupby("asymmetric_gene")["silent_gene"].agg(
        lambda names: ";".join(sorted(names))
    )
    genes = genes.rename_axis("gene").reset_index().sort_values(
        ["n_pairs_as_asymmetric_gene", "n_pairs_as_silent_gene", "gene"], ascending=[False, False, True]
    )

    print(f"\nPromising candidate pairs: {len(candidates):,}")
    print(f"Separate asymmetric genes: {candidates['asymmetric_gene'].nunique():,}")
    print(f"Separate silent genes: {candidates['silent_gene'].nunique():,}")

    CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    pairs_path = CSV_FOLDER / f"promising_candidate_pairs_{settings_tag}.csv"
    genes_path = CSV_FOLDER / f"promising_candidate_genes_{settings_tag}.csv"
    candidates.to_csv(pairs_path, index=False)
    genes.to_csv(genes_path, index=False)
    print(f"Saved pairs: {pairs_path}")
    print(f"Saved genes: {genes_path}")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\nTop {N_PRINT} candidate pairs:")
        print(candidates[[
            "asymmetric_gene", "silent_gene", "gene_gene_rho", "asymmetric_isoforms_significant",
            "asymmetric_isoforms_tested", "silent_isoforms_tested", "min_isoform_adjusted_p",
        ]].head(N_PRINT).to_string(index=False))
        print(f"\nTop {N_PRINT} asymmetric genes:")
        print(genes.drop(columns="silent_partners").head(N_PRINT).to_string(index=False))


if __name__ == "__main__":
    main()
