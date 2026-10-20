"""
Step 0: describe the data (the numbers in DATA.md).

Prints:
  1. how many isoforms, genes and samples there are
  2. what kind of isoforms they are (known vs novel)
  3. how many isoforms each gene has (+ histogram figure)
  4. what the samples are (region x condition x age table)
  5. what the expression values look like (CPM, zeros)
  6. how much expression each isoform category carries (why .nnc isoforms are kept)

Run from the Sapal-Speke-RP-2026 folder:  python describe_data.py
"""

# --- Imports ---------------------------------------------------------------
import os                          # making folders and building file paths
from datetime import datetime      # timestamp in figure file names, so old figures aren't overwritten

import numpy as np                 # maths on arrays (log, NaN, infinity)
import pandas as pd                # tables: DataFrame (2D table) and Series (one labelled column)
import matplotlib.pyplot as plt    # plotting


# --- Settings (change these, not the code below) ---------------------------
MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"   # isoform x sample expression table
CDS_PATH = "Long_Read_Data/cds_output.fa"                       # coding sequences (FASTA format)
FIGURE_FOLDER = "Figures/describe_data"                         # where figures are saved
MAJOR_SHARE = 0.10   # an isoform is "major" if it makes up at least this share of its gene on average


def load_matrix(path):
    """Rows = isoforms (transcript IDs), columns = samples, values = CPM."""
    # sep="\t"     -> the file is tab-separated (.tsv)
    # index_col=0  -> use the first column (transcript_id) as row names, not as data
    return pd.read_csv(path, sep="\t", index_col=0)


def gene_of(isoform_id):
    """Gene = part of the isoform ID before the first "." or "_".
    G249.3.nnc -> G249,  G249.7_ENST00000471889.7 -> G249"""
    # .split(".") breaks the text at every "."  ("G249.3.nnc" -> ["G249", "3", "nnc"])
    # [0] takes the first piece. The second split handles IDs like "G249_..." just in case.
    return isoform_id.split(".")[0].split("_")[0]


def isoform_category(isoform_id):
    """SQANTI category from the end of the ID: known (_ENST), ism, nic or nnc.
    A few IDs end in a plain number (e.g. G10726.122) - no category given."""
    # Known Ensembl transcripts have "_ENST" in the ID (e.g. G249.7_ENST00000471889.7)
    if "_ENST" in isoform_id:
        return "known (_ENST)"
    # Otherwise the category is the last piece after a "."  ("G249.3.nnc" -> "nnc")
    suffix = isoform_id.split(".")[-1]
    # .isdigit() is True if the piece is only numbers, i.e. there is no category label
    return "no suffix" if suffix.isdigit() else suffix


def count_cds(path):
    """Number of sequences in the FASTA file (each starts with a '>' line)."""
    # A FASTA file looks like:
    #   >name_of_sequence_1
    #   ATGGCC...
    #   >name_of_sequence_2
    #   ...
    # so counting the lines that start with ">" counts the sequences.
    # "with open(...)" opens the file and closes it automatically afterwards.
    with open(path) as file:
        return sum(1 for line in file if line.startswith(">"))


def plot_isoforms_per_gene(isoforms_per_gene):
    """Histogram of isoforms per gene. Log x-axis because it ranges from 1 to ~1,400."""
    # Bin edges spaced evenly on a log scale (1, ~1.3, ~1.6, ... up to the max).
    # With normal, evenly spaced bins, all the genes with 1-20 isoforms would be squashed into the first bar.
    bins = np.logspace(0, np.log10(isoforms_per_gene.max() + 1), 30)

    fig = plt.figure(figsize=(9, 5))                    # new figure, 9 x 5 inches
    plt.hist(isoforms_per_gene, bins=bins, color="#2a78d6", edgecolor="white")
    plt.xscale("log")                                   # log x-axis to match the log-spaced bins
    # Dashed vertical line at the median, so the "typical" gene is visible
    plt.axvline(isoforms_per_gene.median(), color="black", linestyle="--",
                label=f"Median = {isoforms_per_gene.median():.0f}")
    plt.xlabel("Number of isoforms in the gene (log scale)")
    plt.ylabel("Number of genes")
    plt.title(f"Isoforms per gene ({len(isoforms_per_gene)} genes)")
    plt.legend()
    plt.tight_layout()                                  # stops labels being cut off

    # Save the figure: make the folder if needed, add a timestamp to the file name
    os.makedirs(FIGURE_FOLDER, exist_ok=True)           # exist_ok=True: no error if the folder already exists
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(FIGURE_FOLDER, f"isoforms_per_gene_{timestamp}.png")
    fig.savefig(path, dpi=200)                          # dpi = resolution
    print(f"\nSaved figure: {path}")
    plt.show()                                          # opens the figure window (script pauses until closed)


def expression_by_category(df):
    """Table: how much expression each isoform category (known / ism / nic / nnc) carries.

    Share of gene = isoform CPM / gene CPM in a sample, averaged over the samples where
    the gene is expressed. "Major" isoforms have an average share of at least MAJOR_SHARE."""
    # Two labels for every isoform (row), lined up with df by using the same index (row names):
    #   genes:      G249.3.nnc -> "G249"
    #   categories: G249.3.nnc -> "nnc"
    genes = pd.Series([gene_of(i) for i in df.index], index=df.index)
    categories = pd.Series([isoform_category(i) for i in df.index], index=df.index)

    # Total CPM of each isoform, added up over all samples (axis=1 = sum across the columns)
    isoform_total = df.sum(axis=1)

    # Gene CPM in each sample = sum of its isoforms' CPM.
    # groupby(genes) puts all rows with the same gene label together, .sum() adds them up.
    # Result: one row per gene, one column per sample.
    gene_cpm = df.groupby(genes).sum()

    # Make a table the same shape as df, where each isoform's row holds its GENE's CPM.
    # gene_cpm.loc[genes] looks up the gene row for every isoform (so a gene with 10 isoforms
    # appears 10 times); set_index puts the isoform names back as row names.
    gene_cpm_per_isoform = gene_cpm.loc[genes].set_index(df.index)

    # Share of gene = isoform CPM / gene CPM, sample by sample (e.g. 0.25 = 25% of the gene).
    # If the gene is 0 in a sample, 0/0 is undefined: replace(0, np.nan) turns those into NaN
    # ("not a number"), and .mean() skips NaN. So the mean is over samples where the gene is expressed.
    # .mean(axis=1) = average across the samples -> one number per isoform.
    share_of_gene = (df / gene_cpm_per_isoform.replace(0, np.nan)).mean(axis=1)

    # For each gene: the category of its most-used isoform.
    # idxmax() gives the NAME of the isoform with the largest share in each gene;
    # categories[...] looks up those isoforms' categories; value_counts() counts each category.
    top_isoform_of_gene = share_of_gene.groupby(genes).idxmax()
    top_category = categories[top_isoform_of_gene].value_counts()

    # Build the table: one row per category, one column per statistic.
    # Each groupby(categories) splits the isoforms by category and summarises each group.
    by_category = categories.groupby(categories)
    table = pd.DataFrame({
        # How many isoforms are in each category
        "isoforms": by_category.size(),
        # Category's CPM as a % of ALL CPM in the dataset
        "% of all expression": 100 * isoform_total.groupby(categories).sum() / isoform_total.sum(),
        # The typical (median) isoform's share of its gene, in %
        "median % of gene": 100 * share_of_gene.groupby(categories).median(),
        # (share >= 0.10) is True/False per isoform; summing True/False counts the Trues
        f"major (>= {MAJOR_SHARE:.0%} of gene)": (share_of_gene >= MAJOR_SHARE).groupby(categories).sum(),
        # How many genes have an isoform of this category as their top isoform
        "genes where it is the top isoform": top_category,
    }).fillna(0).sort_values("% of all expression", ascending=False)
    # fillna(0): a category that is never the top isoform would be missing (NaN) -> show 0 instead
    # sort_values: biggest share of expression at the top

    # Genes made up ONLY of .nnc isoforms: count each gene's non-.nnc isoforms, and keep genes where that count is 0
    only_nnc = (categories != "nnc").groupby(genes).sum() == 0
    return table, only_nnc.sum()      # only_nnc is True/False per gene; .sum() counts the Trues


def main():
    # Load the matrix and label every isoform (row) with its gene
    df = load_matrix(MATRIX_PATH)
    genes = pd.Series([gene_of(i) for i in df.index], index=df.index)

    # ---- 1. Headline numbers ----
    n_cds = count_cds(CDS_PATH)
    print("=== 1. Headline numbers ===")
    # df.shape = (number of rows, number of columns) = (isoforms, samples)
    # {:,} in an f-string prints thousands separators, e.g. 61,117
    print(f"Isoforms: {df.shape[0]:,}")
    print(f"Genes:    {genes.nunique():,}")          # nunique() = number of distinct gene labels
    print(f"Samples:  {df.shape[1]}")
    print(f"Isoforms with a CDS: {n_cds:,} ({100 * n_cds / df.shape[0]:.0f}%)")

    # ---- 2. Isoform categories ----
    # Label every isoform with its category, then count how many of each (value_counts)
    print("\n=== 2. Isoform categories ===")
    categories = pd.Series([isoform_category(i) for i in df.index]).value_counts()
    print(categories.to_string())

    # ---- 3. Isoforms per gene ----
    # value_counts() on the gene labels = how many isoforms (rows) each gene has, largest first
    print("\n=== 3. Isoforms per gene ===")
    isoforms_per_gene = genes.value_counts()
    print(f"Median {isoforms_per_gene.median():.0f}, mean {isoforms_per_gene.mean():.0f}, "
          f"min {isoforms_per_gene.min()}, max {isoforms_per_gene.max()}")

    # Group the genes into ranges (1, 2, 3-5, ...) and count genes per range.
    # pd.cut puts each value into a bin; bins include their right edge, so (2, 5] = 3-5.
    # reindex(bin_labels) keeps the bins in order (value_counts would sort them by count).
    bin_edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, np.inf]
    bin_labels = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", ">500"]
    binned = pd.cut(isoforms_per_gene, bins=bin_edges, labels=bin_labels).value_counts().reindex(bin_labels)
    print("\nIsoforms per gene -> number of genes")
    print(binned.to_string())

    # value_counts() already sorted largest first, so head(5) = the 5 genes with the most isoforms
    print("\nGenes with the most isoforms:")
    print(isoforms_per_gene.head(5).to_string())

    # ---- 4. Samples ----
    # Column names look like Caudate_MDD_Adult_R17353 = region_condition_age_sampleID.
    # Splitting each name at "_" gives the four pieces, which become a small table (one row per sample).
    print("\n=== 4. Samples ===")
    samples = pd.DataFrame([name.split("_") for name in df.columns],
                           columns=["region", "condition", "age", "sample_id"])
    # crosstab counts samples for every combination: rows = region + age, columns = condition.
    # margins=True adds the "Total" row and column.
    print(pd.crosstab([samples["region"], samples["age"]], samples["condition"], margins=True, margins_name="Total"))
    print(f"\nUnique sample IDs: {samples['sample_id'].nunique()}")

    # ---- 5. Expression values ----
    print("\n=== 5. Expression values ===")
    # Sum of each column (sample). For CPM (counts per million) each should be ~1,000,000.
    column_sums = df.sum()
    print(f"Column sums (should be ~1,000,000 for CPM): {column_sums.min():,.0f} to {column_sums.max():,.0f}")
    # (df == 0) is a True/False table; the mean of True/False = fraction of True = fraction of zeros
    print(f"Zero entries: {100 * (df == 0).to_numpy().mean():.0f}%")
    print(f"Isoforms that are zero in every sample: {(df.sum(axis=1) == 0).sum()}")

    # ---- 6. Expression by isoform category ----
    # Adult samples only (the analysis excludes fetal), and isoforms seen in at least one of them
    print("\n=== 6. Expression by isoform category (45 adult samples) ===")
    # Keep only the columns whose name does not contain "_Fetal_" (49 -> 45 samples)
    adult = df[[name for name in df.columns if "_Fetal_" not in name]]
    # Drop isoforms that are 0 in every adult sample (fetal-only isoforms).
    # (adult > 0).any(axis=1) is True for a row if ANY sample is above 0.
    adult = adult[(adult > 0).any(axis=1)]
    table, n_only_nnc = expression_by_category(adult)
    print(table.round(2).to_string())
    print(f"\nGenes whose isoforms are ALL .nnc (would vanish if .nnc were removed): {n_only_nnc}")

    # Histogram last, because plt.show() pauses the script until the window is closed
    plot_isoforms_per_gene(isoforms_per_gene)


# Only run main() when this file is run directly (python describe_data.py),
# not when another script imports functions from it.
if __name__ == "__main__":
    main()
