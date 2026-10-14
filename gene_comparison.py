import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Colour points by brain region
REGION_COLOURS = {"DLPFC": '#2a78d6', "Hippocampus": '#eb6834', "Caudate": "#1baf7a"}


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
    """Sum TPM of all isoforms of each gene (one row per gene, one column per sample)."""
    return df.groupby(gene_ids(df.index), sort=False).sum()


def region_colours_for(samples):
    """One colour per sample, based on the brain region at the start of its name."""
    regions = [name.split("_")[0] for name in samples]
    return [REGION_COLOURS[r] for r in regions]


def add_region_legend(ax, fontsize=None):
    """Legend with one entry per brain region."""
    for region, colour in REGION_COLOURS.items():
        ax.scatter([], [], color=colour, label=region)
    ax.legend(fontsize=fontsize)


def plot_gene_vs_gene(tpm_a, tpm_b, gene_a, gene_b):
    """Plot TPM of gene_a against gene_b, one point per sample."""
    rho, p_value = spearmanr(tpm_a, tpm_b) # Compares across the 49 samples.

    plt.figure(figsize=(7, 6))
    plt.scatter(tpm_a, tpm_b, c=region_colours_for(tpm_a.index))
    plt.xlabel(f"{gene_a} TPM")
    plt.ylabel(f"{gene_b} TPM")
    plt.title(f"{gene_a} vs {gene_b} across {len(tpm_a)} samples\nSpearman rho = {rho:.2f}, p = {p_value:.2g}")
    add_region_legend(plt.gca())

    plt.tight_layout()
    plt.show()


def top_isoforms_of_gene(df, gene, n=10):
    """All isoforms (rows) belonging to gene, plus the top n by average TPM across the 49 samples."""
    isoforms = df[[g == gene for g in gene_ids(df.index)]]
    average_isoform_tpm = isoforms.mean(axis=1).sort_values(ascending=False)
    return isoforms, average_isoform_tpm, average_isoform_tpm.index[:n]


def plot_isoforms_vs_gene(isoforms_b, average_isoform_tpm, top_isoforms, tpm_a, gene_a, gene_b):
    """One subplot per isoform: isoform TPM on x, gene_a total TPM on y.
    Returns a table of isoform, Spearman rho and p-value."""
    colours = region_colours_for(tpm_a.index)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    isoform_results = []
    for isoform, ax in zip(top_isoforms, axes.flatten()):
        tpm_isoform = isoforms_b.loc[isoform]
        rho, p_value = spearmanr(tpm_isoform, tpm_a) # Compares across the 49 samples.

        isoform_results.append({"isoform": isoform, "average_tpm": average_isoform_tpm[isoform], "spearman_rho": rho, "p_value": p_value})

        ax.scatter(tpm_isoform, tpm_a, c=colours, s=15)
        ax.set_xlabel(f"{isoform} TPM", fontsize=8)
        ax.set_ylabel(f"{gene_a} TPM (all isoforms)", fontsize=8)
        ax.set_title(f"rho = {rho:.2f}, p = {p_value:.2g}", fontsize=9)

    # Legend (once, for the whole figure)
    add_region_legend(axes[0, 0], fontsize=7)

    fig.suptitle(f"Top {len(top_isoforms)} {gene_b} isoforms vs {gene_a}")
    plt.tight_layout()

    return pd.DataFrame(isoform_results)


def main():
    gene_a = "G60740"
    gene_b = "G49035"

    df = load_matrix()
    gene_tpm = sum_by_gene(df)
    tpm_a = gene_tpm.loc[gene_a]
    tpm_b = gene_tpm.loc[gene_b]

    plot_gene_vs_gene(tpm_a, tpm_b, gene_a, gene_b)

    # ---- Each isoform of gene_b against the whole of gene_a ----
    isoforms_b, average_isoform_tpm, top_isoforms = top_isoforms_of_gene(df, gene_b, n=10)
    print(f"{gene_b} has {len(isoforms_b)} isoforms, plotting the top {len(top_isoforms)} by average TPM")

    isoform_results = plot_isoforms_vs_gene(isoforms_b, average_isoform_tpm, top_isoforms, tpm_a, gene_a, gene_b)

    # Table of isoform, Spearman rho and p-value
    print(isoform_results)

    plt.show()


if __name__ == "__main__":
    main()
