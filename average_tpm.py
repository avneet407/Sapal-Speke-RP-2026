import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr


def load_matrix(path="expressed_isoforms_matrix.tsv"):
    """Loading Expression Matrix (isoform expression levels for 49 samples)"""
    return pd.read_csv(path,
                       sep="\t",
                       index_col=0,
                      )


def gene_ids(transcript_ids):
    """Gene ID is the part of the transcript ID before the first "." (e.g. G249.3.nnc -> G249)"""
    return [name.split(".")[0] for name in transcript_ids]


def sum_by_gene(df):
    """Sum TPM of all isoforms of each gene (one row per gene, one column per sample).
    sort=False keeps genes in the order they appear in the file."""
    return df.groupby(gene_ids(df.index), sort=False).sum()


def average_tpm_per_gene(gene_tpm):
    """Average TPM per sample = row sum / number of samples, sorted highest first."""
    n_samples = gene_tpm.shape[1]
    average_tpm = gene_tpm.sum(axis=1) / n_samples
    return average_tpm.sort_values(ascending=False)


def plot_average_tpm(average_tpm, n_samples):
    """Plot average TPM for each gene."""
    plt.figure(figsize=(14, 5))
    plt.bar(average_tpm.index, average_tpm.values, width=1.0)
    plt.yscale("log") # A few genes are very highly expressed, so log scale keeps the rest visible.
    plt.xticks([]) # 816 genes, too many to label.
    plt.xlabel(f"Gene ({len(average_tpm)} genes, sorted by average TPM)")
    plt.ylabel(f"Average TPM per sample (sum / {n_samples})")
    plt.title("Average TPM per sample for each gene")

    plt.tight_layout()
    #plt.show()


def filter_by_average_tpm(df, average_tpm, threshold=1000):
    """Keep only isoforms whose gene has average TPM greater than threshold."""
    return df[pd.Series(gene_ids(df.index), index=df.index).map(average_tpm) > threshold]


def spearman_all_pairs(gene_tpm):
    """Spearman correlation for every pair of genes, compared across the 49 samples."""
    results = []
    for gene_a, gene_b in combinations(gene_tpm.index, 2):
        rho, p_value = spearmanr(gene_tpm.loc[gene_a], gene_tpm.loc[gene_b])

        both_detected = (gene_tpm.loc[gene_a] > 0) & (gene_tpm.loc[gene_b] > 0)
        n_both = both_detected.sum()

        results.append({"gene_a": gene_a, "gene_b": gene_b, "spearman_rho": rho, "p_value": p_value, "n_both_detected": n_both})

    return pd.DataFrame(results)


def plot_p_value_histogram(results, n_bins=20):
    """Plot histogram of raw p-values for all gene pairs.
    If nothing were correlated, p-values would be flat between 0 and 1 (dashed line).
    A spike near 0 above that line = more correlated pairs than expected by chance."""
    p_values = results["p_value"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(p_values, bins=n_bins, range=(0, 1), edgecolor="white")
    plt.axhline(len(p_values) / n_bins, color="black", linestyle="--", label="Expected if no correlations")
    plt.xlabel("Spearman p-value")
    plt.ylabel("Number of gene pairs")
    plt.title(f"P-value distribution ({len(p_values)} gene pairs)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    df = load_matrix()
    gene_tpm = sum_by_gene(df)
    average_tpm = average_tpm_per_gene(gene_tpm)

    plot_average_tpm(average_tpm, gene_tpm.shape[1])

    filtered_df = filter_by_average_tpm(df, average_tpm, threshold=1000)
    gene_tpm = sum_by_gene(filtered_df)
    print(gene_tpm.head())

    print("Calculating Spearman correlation for all gene pairs...")
    results = spearman_all_pairs(gene_tpm)
    final_result = results.sort_values(by="p_value", ascending=True)
    print(final_result)

    plot_p_value_histogram(results)


if __name__ == "__main__":
    main()
