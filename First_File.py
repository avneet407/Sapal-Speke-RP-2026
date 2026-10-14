import pandas as pd

# Loading Expression Matrix (isoform expression levels for 49 samples)
df = pd.read_csv("expressed_isoforms_matrix.tsv", 
                 sep="\t", 
                 index_col=0,
                )



import matplotlib.pyplot as plt

column_sums = df.sum(axis=0)
print(column_sums)

detected = df > 0 # True when an isoform is detected.
genes = [name.split(".")[0] for name in detected.index]
gene_detected = detected.groupby(genes).any()
genes_per_sample = gene_detected.sum(axis=0) 
genes_per_sample = genes_per_sample.sort_values()

detected_per_sample = detected.sum(axis=0) # Adding down each column.
detected_per_sample = detected_per_sample.sort_values()

# Colour bars by brain region
region_colours = {"DLPFC": '#2a78d6', "Hippocampus": '#eb6834', "Caudate": "#1baf7a"}
regions = [name.split("_")[0] for name in detected_per_sample.index]
colours = [region_colours[r] for r in regions]   
"""
 Plot graph showing isoform numbers detected per sample.
plt.figure(figsize=(12, 5))
plt.bar(detected_per_sample.index, detected_per_sample.values, color=colours)
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("Number of isoforms detected (TPM>0)")
plt.title("Isoforms detected per sample")
# Legend
for region, colour in region_colours.items():
    plt.bar(0, 0, color=colour, label=region)
    plt.legend()

plt.tight_layout()
"""

# Plot graph showing isoform numbers detected per sample.
plt.figure(figsize=(12, 5))
plt.bar(genes_per_sample.index, genes_per_sample.values)
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("Number of genes detected (TPM>0)")
plt.title("Genes detected per sample")


plt.tight_layout()
plt.show()
