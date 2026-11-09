"""
Isoform shares: for every gene, each isoform's CPM and its percentage of the gene's total CPM.

  1. Load the matrix (adults only by default - the 4 fetal DLPFC samples are dropped, as in
     gene_spearman.py).
  2. Gene = gene name from the GTF (e.g. SLC45A1).
  3. Gene CPM = sum of all its isoforms' CPM, per sample.
  4. Isoform % = isoform CPM / gene CPM * 100, per sample. The isoforms of a gene add up to 100%
     in every sample. If the gene has CPM 0 in a sample, the % is left blank (0 / 0 is undefined).
  5. Save one row per isoform as CSV in 1_data_processing/CSV/:
       gene, isoform,
       mean_cpm, mean_pct            - summary over the samples (mean_pct = isoform's share of the
                                       gene's mean CPM, so it still adds up to 100% within a gene)
       <sample>_cpm, <sample>_pct    - one pair of columns for every sample
     Genes are ordered by mean CPM (highest first), and isoforms within a gene by mean CPM.

Run with:
    python 1_data_processing/.py/isoform_analysis.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.data_loading.expression_data import load_matrix, read_isoform_genes


# --- Settings (change these, not the code below) ---------------------------
ADULTS_ONLY = True    # True = drop the 4 fetal samples (columns with "_Fetal_" in the name)
MIN_GENE_CPM = 0      # only keep genes whose mean CPM is above this (0 = keep every expressed gene)

MATRIX_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv")
GTF_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf")
RESULTS_FOLDER = str(ROOT_FOLDER / "1_data_processing" / "CSV")


def main():
    # ---- 1. Load ----
    df = load_matrix(MATRIX_PATH)
    if ADULTS_ONLY:
        df = df[[name for name in df.columns if "_Fetal_" not in name]]
    samples = list(df.columns)

    # ---- 2. Which gene each isoform belongs to ----
    genes = read_isoform_genes(GTF_PATH)[df.index]

    # ---- 3. Gene CPM per sample, repeated on every isoform row of that gene ----
    # transform("sum") gives each isoform row its gene's total, so the division below lines up
    gene_cpm = df.groupby(genes).transform("sum")

    # ---- 4. Isoform % of its gene, per sample ----
    # Where the gene is 0 in a sample, 0 / 0 gives NaN, which is saved as a blank cell
    pct = df / gene_cpm.where(gene_cpm > 0) * 100

    # Summary columns: mean CPM over the samples, and the share of the gene's mean CPM
    mean_cpm = df.mean(axis=1)
    gene_mean_cpm = mean_cpm.groupby(genes).transform("sum")
    mean_pct = mean_cpm / gene_mean_cpm.where(gene_mean_cpm > 0) * 100

    # ---- 5. Build the table: CPM and % next to each other for every sample ----
    table = pd.DataFrame({"gene": genes,
                          "isoform": df.index,
                          "mean_cpm": mean_cpm,
                          "mean_pct": mean_pct,
                          "gene_mean_cpm": gene_mean_cpm})
    for sample in samples:
        table[f"{sample}_cpm"] = df[sample]
        table[f"{sample}_pct"] = pct[sample]

    # Optional filter on gene expression
    n_genes_before = table["gene"].nunique()
    table = table[table["gene_mean_cpm"] > MIN_GENE_CPM]
    print(f"Genes with mean CPM > {MIN_GENE_CPM}: {table['gene'].nunique()} of {n_genes_before} "
          f"({len(table):,} isoforms, {len(samples)} samples)")

    # Highest-expressed genes first; inside each gene, its highest-expressed isoform first
    table = table.sort_values(["gene_mean_cpm", "gene", "mean_cpm"], ascending=[False, True, False])
    table = table.drop(columns="gene_mean_cpm")

    print()
    print(table[["gene", "isoform", "mean_cpm", "mean_pct"]].head(15)
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ---- 6. Save ----
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    age_tag = "adults" if ADULTS_ONLY else "all_samples"
    path = os.path.join(RESULTS_FOLDER, f"isoform_shares_{age_tag}.csv")
    table.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
