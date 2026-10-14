import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from figure_utils import save_figure

# Colour points by brain region
REGION_COLOURS = {"DLPFC": '#2a78d6', "Hippocampus": '#eb6834', "Caudate": "#1baf7a"}


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
    """Sum CPM of all isoforms of each gene (one row per gene, one column per sample)."""
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


def spearman_text(x, y):
    """Spearman rho and p for all samples, then within each brain region, one line each.
    Samples with NaN (e.g. 0/0 fractions) are skipped; n = samples used."""
    regions = pd.Series([name.split("_")[0] for name in x.index], index=x.index)
    groups = [("All", regions.index)] + [(region, regions.index[regions == region]) for region in REGION_COLOURS]

    lines = []
    for label, samples in groups:
        x_group, y_group = x[samples], y[samples]
        n = (x_group.notna() & y_group.notna()).sum()
        rho, p_value = spearmanr(x_group, y_group, nan_policy="omit")
        lines.append(f"{label}: rho = {rho:.2f}, p = {p_value:.2g} (n={n})")
    return "\n".join(lines)


def plot_gene_vs_gene(cpm_a, cpm_b, gene_a, gene_b):
    """Plot CPM of gene_a against gene_b, one point per sample."""
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(cpm_a, cpm_b, c=region_colours_for(cpm_a.index))
    plt.xlabel(f"{gene_a} CPM")
    plt.ylabel(f"{gene_b} CPM")
    plt.title(f"{gene_a} vs {gene_b}\n{spearman_text(cpm_a, cpm_b)}", fontsize=10)
    add_region_legend(plt.gca())

    plt.tight_layout()
    save_figure(fig, "gene_comparison", f"{gene_a}_vs_{gene_b}_scatter")
    plt.show()


def top_isoforms_of_gene(df, gene, n=10):
    """All isoforms (rows) belonging to gene, plus the top n by average CPM across the 49 samples."""
    isoforms = df[[g == gene for g in gene_ids(df.index)]]
    average_isoform_cpm = isoforms.mean(axis=1).sort_values(ascending=False)
    return isoforms, average_isoform_cpm, average_isoform_cpm.index[:n]


def plot_isoforms_vs_gene(isoforms_b, average_isoform_cpm, top_isoforms, cpm_a, cpm_b, gene_a, gene_b):
    """One subplot per isoform: isoform fraction of gene_b on x, gene_a total CPM on y.
    Fraction = isoform CPM / total gene_b CPM in the same sample (0 to 1).
    Returns a table of isoform, mean fraction, Spearman rho and p-value."""
    colours = region_colours_for(cpm_a.index)

    fig, axes = plt.subplots(2, 5, figsize=(22, 11))
    isoform_results = []
    for isoform, ax in zip(top_isoforms, axes.flatten()):
        # Samples where gene_b is not expressed at all give 0/0 = NaN; spearmanr skips them.
        fraction_isoform = isoforms_b.loc[isoform] / cpm_b
        rho, p_value = spearmanr(fraction_isoform, cpm_a, nan_policy="omit") # Compares across the 49 samples.

        isoform_results.append({"isoform": isoform, "average_cpm": average_isoform_cpm[isoform],
                                "mean_fraction": fraction_isoform.mean(), "spearman_rho": rho, "p_value": p_value})

        ax.scatter(fraction_isoform, cpm_a, c=colours, s=15)
        ax.set_xlabel(f"{isoform}\nfraction of {gene_b} CPM", fontsize=8)
        ax.set_ylabel(f"{gene_a} CPM (all isoforms)", fontsize=8)
        ax.set_title(spearman_text(fraction_isoform, cpm_a), fontsize=8)

    # Legend (once, for the whole figure)
    add_region_legend(axes[0, 0], fontsize=7)

    fig.suptitle(f"Top {len(top_isoforms)} {gene_b} isoforms (fraction of gene) vs {gene_a}")
    plt.tight_layout()
    save_figure(fig, "gene_comparison", f"{gene_b}_top{len(top_isoforms)}_isoform_fractions_vs_{gene_a}")

    return pd.DataFrame(isoform_results)


def main():
    gene_a = "G49035"
    gene_b = "G60740"

    df = load_matrix()
    gene_cpm = sum_by_gene(df)
    cpm_a = gene_cpm.loc[gene_a]
    cpm_b = gene_cpm.loc[gene_b]

    plot_gene_vs_gene(cpm_a, cpm_b, gene_a, gene_b)

    # ---- Each isoform of gene_b against the whole of gene_a ----
    isoforms_b, average_isoform_cpm, top_isoforms = top_isoforms_of_gene(df, gene_b, n=10)
    print(f"{gene_b} has {len(isoforms_b)} isoforms, plotting the top {len(top_isoforms)} by average CPM")

    isoform_results = plot_isoforms_vs_gene(isoforms_b, average_isoform_cpm, top_isoforms, cpm_a, cpm_b, gene_a, gene_b)

    # Table of isoform, Spearman rho and p-value
    print(isoform_results)

    plt.show()


if __name__ == "__main__":
    main()
