import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr

# Loading Expression Matrix (isoform expression levels for 49 samples)
df = pd.read_csv("expressed_isoforms_matrix.tsv",
                 sep="\t",
                 index_col=0,
                )

# Gene ID is the part of the transcript ID before the first "." (e.g. G249.3.nnc -> G249)
genes = [name.split(".")[0] for name in df.index]

# Sum TPM of all isoforms of each gene (one row per gene, one column per sample).
gene_tpm = df.groupby(genes, sort=False).sum()

# Average TPM per sample = row sum / number of samples.
n_samples = gene_tpm.shape[1]
average_tpm = gene_tpm.sum(axis=1) / n_samples
average_tpm = average_tpm.sort_values(ascending=False)

"""
# Plot average TPM for each gene.
plt.figure(figsize=(14, 5))
plt.bar(average_tpm.index, average_tpm.values, width=1.0)
plt.yscale("log") # A few genes are very highly expressed, so log scale keeps the rest visible.
plt.xticks([]) # 816 genes, too many to label.
plt.xlabel(f"Gene ({len(average_tpm)} genes, sorted by average TPM)")
plt.ylabel(f"Average TPM per sample (sum / {n_samples})")
plt.title("Average TPM per sample for each gene")

plt.tight_layout()
#plt.show()
"""


filtered_df = df[pd.Series(genes, index=df.index).map(average_tpm) > 1000]
# Gene ID is the part of the transcript ID before the first "." (e.g. G249.3.nnc -> G249)
genes = [name.split(".")[0] for name in filtered_df.index]

# Sum TPM of all isoforms of each gene (ignores which isoform it is).
# sort=False keeps genes in the order they appear in the file.
gene_tpm = filtered_df.groupby(genes, sort=False).sum()
print(gene_tpm.head())

print("Calculating Spearman correlation for all gene pairs...")

# Only the first three genes for now -> 3 pairs
all_genes = gene_tpm.index

results = []
for gene_a, gene_b in combinations(all_genes, 2):
    rho, p_value = spearmanr(gene_tpm.loc[gene_a], gene_tpm.loc[gene_b]) # Compares across the 49 samples.

    both_detected = (gene_tpm.loc[gene_a] > 0) & (gene_tpm.loc[gene_b] > 0)
    n_both = both_detected.sum()

    results.append({"gene_a": gene_a, "gene_b": gene_b, "spearman_rho": rho, "p_value": p_value, "n_both_detected": n_both})    

results = pd.DataFrame(results)
final_result = results.sort_values(by="p_value", ascending=True)
print(final_result)

# Plot histogram of raw p-values for all gene pairs.
# If nothing were correlated, p-values would be flat between 0 and 1 (dashed line).
# A spike near 0 above that line = more correlated pairs than expected by chance.
p_values = results["p_value"].dropna()
n_bins = 20
plt.figure(figsize=(8, 5))
plt.hist(p_values, bins=n_bins, range=(0, 1), edgecolor="white")
plt.axhline(len(p_values) / n_bins, color="black", linestyle="--", label="Expected if no correlations")
plt.xlabel("Spearman p-value")
plt.ylabel("Number of gene pairs")
plt.title(f"P-value distribution ({len(p_values)} gene pairs)")
plt.legend()

plt.tight_layout()
plt.show()
