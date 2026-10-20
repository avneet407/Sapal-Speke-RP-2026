import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from figure_utils import save_figure


def load_matrix(path="Long_Read_Data/expressed_isoforms_matrix.tsv"):
    """Loading Expression Matrix (isoform expression levels for 49 samples)"""
    return pd.read_csv(path,
                       sep="\t",
                       index_col=0,
                      )


def gene_ids(transcript_ids):
    """Gene ID is the part of the transcript ID before the first "." (e.g. G249.3.nnc -> G249)"""
    return [name.split(".")[0] for name in transcript_ids]


def sum_by_gene(df):
    """Sum CPM of all isoforms of each gene (one row per gene, one column per sample).
    sort=False keeps genes in the order they appear in the file."""
    return df.groupby(gene_ids(df.index), sort=False).sum()


def average_cpm_per_gene(gene_cpm):
    """Average CPM per sample = row sum / number of samples, sorted highest first."""
    n_samples = gene_cpm.shape[1]
    average_cpm = gene_cpm.sum(axis=1) / n_samples
    return average_cpm.sort_values(ascending=False)


def plot_average_cpm(average_cpm, n_samples):
    """Plot average CPM for each gene."""
    fig = plt.figure(figsize=(14, 5))
    plt.bar(average_cpm.index, average_cpm.values, width=1.0)
    plt.yscale("log") # A few genes are very highly expressed, so log scale keeps the rest visible.
    plt.xticks([]) # 816 genes, too many to label.
    plt.xlabel(f"Gene ({len(average_cpm)} genes, sorted by average CPM)")
    plt.ylabel(f"Average CPM per sample (sum / {n_samples})")
    plt.title("Average CPM per sample for each gene")

    plt.tight_layout()
    save_figure(fig, "average_cpm", "average_cpm_per_gene")
    #plt.show()


def filter_by_average_cpm(df, average_cpm, threshold=1000):
    """Keep only isoforms whose gene has average CPM greater than threshold."""
    return df[pd.Series(gene_ids(df.index), index=df.index).map(average_cpm) > threshold]


def spearman_all_pairs(gene_cpm):
    """Spearman correlation for every pair of genes, compared across the 49 samples."""
    results = []
    for gene_a, gene_b in combinations(gene_cpm.index, 2):
        rho, p_value = spearmanr(gene_cpm.loc[gene_a], gene_cpm.loc[gene_b])

        both_detected = (gene_cpm.loc[gene_a] > 0) & (gene_cpm.loc[gene_b] > 0)
        n_both = both_detected.sum()

        results.append({"gene_a": gene_a, "gene_b": gene_b, "spearman_rho": rho, "p_value": p_value, "n_both_detected": n_both})

    return pd.DataFrame(results)


def plot_p_value_histogram(results, name="p_value_histogram", n_bins=20):
    """Plot histogram of raw p-values for all gene pairs.
    If nothing were correlated, p-values would be flat between 0 and 1 (dashed line).
    A spike near 0 above that line = more correlated pairs than expected by chance."""
    p_values = results["p_value"].dropna()
    fig = plt.figure(figsize=(8, 5))
    plt.hist(p_values, bins=n_bins, range=(0, 1), edgecolor="white")
    plt.axhline(len(p_values) / n_bins, color="black", linestyle="--", label="Expected if no correlations")
    plt.xlabel("Spearman p-value")
    plt.ylabel("Number of gene pairs")
    plt.title(f"P-value distribution ({len(p_values)} gene pairs)")
    plt.legend()

    plt.tight_layout()
    save_figure(fig, "average_cpm", name)
    plt.show()


def main():
    threshold = 1000

    df = load_matrix()
    gene_cpm = sum_by_gene(df)
    average_cpm = average_cpm_per_gene(gene_cpm)

    plot_average_cpm(average_cpm, gene_cpm.shape[1])

    filtered_df = filter_by_average_cpm(df, average_cpm, threshold=threshold)
    gene_cpm = sum_by_gene(filtered_df)
    print(gene_cpm.head())

    print("Calculating Spearman correlation for all gene pairs...")
    results = spearman_all_pairs(gene_cpm)
    final_result = results.sort_values(by="p_value", ascending=True)
    print(final_result)

    plot_p_value_histogram(results, name=f"p_value_histogram_cpm{threshold}")


if __name__ == "__main__":
    main()
