"""Matrix figure for one gene pair: its isoforms against the partner gene."""

import matplotlib.pyplot as plt
import numpy as np


def plot_pair_matrix(
    pair, rho_matrix, adjusted_p_matrix, row_labels, column_labels, path, mht_method, isoform_key
):
    """Save a matrix coloured by adjusted p-value and labelled with rho and p.
    isoform_key = lines of text beside the matrix saying which isoform each label stands for."""
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
    ax.set_xlabel(f"A components: {pair.gene_a}")
    ax.set_ylabel(f"B components: {pair.gene_b}")
    ax.set_title(
        f"Gene pair {pair.gene_a} vs {pair.gene_b}\n"
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
        "mean CPM. Blank cells are isoform-to-isoform correlations, not calculated here.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        wrap=True
    )
    fig.tight_layout()
    # Added after the layout, just past the figure's right edge, so the matrix keeps its size;
    # bbox_inches="tight" widens the saved image to include it
    fig.text(
        1.0,
        ax.get_position().y1,
        "\n".join(isoform_key),
        ha="left",
        va="top",
        family="monospace",
        fontsize=7
    )
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_gene_pair_matrix(result_table, pair, path, mht_method):
    """Build the rho and adjusted-p matrices for one gene pair and save the figure.

    result_table is the corrected table from filter_isoform_gene.py; pair is that gene
    pair's gene-total row. The gene-total cell is A x B, A's isoforms fill the first row
    and B's isoforms fill the first column."""
    pair_rows = result_table[
        (result_table["gene_a"] == pair.gene_a)
        & (result_table["gene_b"] == pair.gene_b)
    ]
    a_rows = pair_rows.loc[
        pair_rows["component_a_type"] == "isoform_percent_share"
    ].sort_values(
        "component_a",
        key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
    )
    b_rows = pair_rows.loc[
        pair_rows["component_b_type"] == "isoform_percent_share"
    ].sort_values(
        "component_b",
        key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
    )
    column_labels = ["A"] + a_rows["component_a"].tolist()
    row_labels = ["B"] + b_rows["component_b"].tolist()
    rho_matrix = np.full(
        (len(row_labels), len(column_labels)),
        np.nan,
        dtype=float
    )
    adjusted_p_matrix = np.full_like(rho_matrix, np.nan)
    gene_row = pair_rows.loc[
        (pair_rows["component_a"] == "A")
        & (pair_rows["component_b"] == "B")
    ].iloc[0]
    rho_matrix[0, 0] = gene_row["spearman_rho"]
    adjusted_p_matrix[0, 0] = gene_row["mht_adjusted_p_value"]
    for column_index, row in enumerate(a_rows.itertuples(index=False), start=1):
        rho_matrix[0, column_index] = row.spearman_rho
        adjusted_p_matrix[0, column_index] = row.mht_adjusted_p_value
    for row_index, row in enumerate(b_rows.itertuples(index=False), start=1):
        rho_matrix[row_index, 0] = row.spearman_rho
        adjusted_p_matrix[row_index, 0] = row.mht_adjusted_p_value

    isoform_key = [
        "Isoform key",
        "",
        f"A = {pair.gene_a}",
        *[
            f"{row.component_a:<4}{row.isoform_a_id}"
            for row in a_rows.itertuples(index=False)
        ],
        "",
        f"B = {pair.gene_b}",
        *[
            f"{row.component_b:<4}{row.isoform_b_id}"
            for row in b_rows.itertuples(index=False)
        ]
    ]
    plot_pair_matrix(
        pair,
        rho_matrix,
        adjusted_p_matrix,
        row_labels,
        column_labels,
        path,
        mht_method,
        isoform_key
    )
