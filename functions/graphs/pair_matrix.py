"""Matrix figure for one gene pair: its isoforms against the partner gene."""

import matplotlib.pyplot as plt
import numpy as np


def plot_pair_matrix(
    pair, rho_matrix, adjusted_p_matrix, row_labels, column_labels, path, mht_method
):
    """Save a matrix coloured by adjusted p-value and labelled with rho and p."""
    n_rows, n_columns = rho_matrix.shape
    fig, ax = plt.subplots(
        figsize=(max(8, 0.42 * n_columns + 3), max(6, 0.36 * n_rows + 2))
    )
    colour_map = plt.get_cmap("RdYlGn").copy()
    colour_map.set_bad("#eeeeee")
    image = ax.imshow(
        np.ma.masked_invalid(adjusted_p_matrix),
        cmap=colour_map,
        vmin=0,
        vmax=1,
        aspect="auto"
    )
    fig.colorbar(
        image,
        ax=ax,
        label=f"{mht_method.upper()}-adjusted p-value (colour)"
    )

    ax.set_xticks(np.arange(n_columns), labels=column_labels)
    ax.set_yticks(np.arange(n_rows), labels=row_labels)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(f"A components: {pair.gene_a_name} ({pair.gene_a})")
    ax.set_ylabel(f"B components: {pair.gene_b_name} ({pair.gene_b})")
    ax.set_title(
        f"Gene pair {pair.gene_a_name} ({pair.gene_a}) vs "
        f"{pair.gene_b_name} ({pair.gene_b})\n"
        f"BY-adjusted gene-pair p = {pair.by_adjusted_p_value:.3g}"
    )

    for row_index in range(n_rows):
        for column_index in range(n_columns):
            rho = rho_matrix[row_index, column_index]
            adjusted_p = adjusted_p_matrix[row_index, column_index]
            if np.isfinite(rho):
                p_label = f"{adjusted_p:.2g}" if np.isfinite(adjusted_p) else "NA"
                text_colour = "white" if np.isfinite(adjusted_p) and adjusted_p < 0.01 else "black"
                ax.text(
                    column_index,
                    row_index,
                    f"rho={rho:.2f}\n{mht_method.upper()} p={p_label}",
                    ha="center",
                    va="center",
                    color=text_colour,
                    fontsize=7
                )

    ax.text(
        0,
        -0.16,
        f"Cells show rho and {mht_method.upper()}-adjusted p. Colour encodes adjusted p "
        "(red = smaller/more significant; green = larger). A x B uses the upstream "
        "gene-gene BY p-value; all "
        "isoform-vs-gene tests share one BY correction. Isoform labels follow descending "
        "mean CPM. Blank cells are isoform-to-isoform correlations, not calculated here. "
        "See the CSV for the isoform-ID mapping.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        wrap=True
    )
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
