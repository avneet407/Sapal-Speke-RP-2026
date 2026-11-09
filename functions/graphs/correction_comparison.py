"""The figure pages comparing the multiple-testing corrections.

One figure with a page switcher (< and > buttons); every page is also saved as a PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle


def threshold_label(error_rate, threshold):
    return f"{error_rate} = {threshold:.0%}"


def thresholds_by_significant_count(methods_to_plot):
    thresholds_by_method = {}
    for method, adjusted_p_values, _ in methods_to_plot:
        adjusted_p_values = np.asarray(adjusted_p_values, dtype=float)
        finite_adjusted_p_values = np.sort(
            adjusted_p_values[np.isfinite(adjusted_p_values)]
        )
        if not finite_adjusted_p_values.size:
            raise ValueError(f"No finite adjusted p-values for {method}")

        candidate_thresholds = np.unique(np.concatenate((
            np.array([0.0, 1.0]),
            np.nextafter(np.unique(finite_adjusted_p_values), np.inf)
        )))
        candidate_thresholds = candidate_thresholds[
            (candidate_thresholds >= 0) & (candidate_thresholds <= 1)
        ]
        counts = np.searchsorted(
            finite_adjusted_p_values,
            candidate_thresholds,
            side="left"
        )
        thresholds_for_count = {}
        for threshold, count in zip(candidate_thresholds, counts):
            thresholds_for_count.setdefault(int(count), float(threshold))
        thresholds_by_method[method] = thresholds_for_count

    common_counts = sorted(set.intersection(*(
        set(thresholds_for_count)
        for thresholds_for_count in thresholds_by_method.values()
    )))
    if not common_counts:
        raise ValueError("Methods have no common attainable significant-test counts")

    if len(common_counts) > 15:
        selected_indices = np.linspace(0, len(common_counts) - 1, 15, dtype=int)
        common_counts = [common_counts[index] for index in selected_indices]

    return common_counts, thresholds_by_method


def plot_correction_comparison(
    data,
    max_statistics,
    raw_max_statistics,
    bonferroni_alpha,
    holm_alpha,
    pms_alpha,
    bh_q,
    by_q,
    plot_output_dir
):
    """Draw and save every page, then show the figure (the script pauses until it is closed)."""
    # Visualising Difference in Significance Determination Between Methods

    # Plot of the number of significant results determined by each method
    methods = ['Bonferroni', 'Holm-Bonferroni', 'PMS', 'Benjamini-Hochberg', 'Benjamini-Yekutieli']
    counts = [
        data['bonferroni_significant'].sum(),
        data['holm_bonferroni_significant'].sum(),
        data['pms_significant'].sum(),
        data['bh_significant'].sum(),
        data['by_significant'].sum()
    ]
    raw_counts = [
        data['bonferroni_raw_significant'].sum(),
        data['holm_bonferroni_raw_significant'].sum(),
        data['pms_raw_significant'].sum(),
        data['bh_raw_significant'].sum(),
        data['by_raw_significant'].sum()
    ]
    adjusted_p_columns = {
        'Holm-Bonferroni': ('holm_bonferroni_adjusted_p', 'FWER'),
        'PMS': ('pms_adjusted_p', 'FWER'),
        'Benjamini-Hochberg': ('bh_adjusted_p', 'FDR'),
        'Benjamini-Yekutieli': ('by_adjusted_p', 'FDR')
    }
    comparison_methods = list(adjusted_p_columns)
    jaccard_comparisons = [
        {
            'title': f'Jaccard overlap: FWER={holm_alpha:.0%}, FDR={bh_q:.0%}',
            'thresholds': {
                'Holm-Bonferroni': holm_alpha,
                'PMS': pms_alpha,
                'Benjamini-Hochberg': bh_q,
                'Benjamini-Yekutieli': by_q
            },
            'filename': 'jaccard_similarity_fwer50_fdr5.png'
        },
        {
            'title': f'Jaccard overlap: FWER={holm_alpha:.0%}, FDR=1%',
            'thresholds': {
                'Holm-Bonferroni': holm_alpha,
                'PMS': pms_alpha,
                'Benjamini-Hochberg': 0.01,
                'Benjamini-Yekutieli': 0.01
            },
            'filename': 'jaccard_similarity_fwer50_fdr1.png'
        }
    ]

    for comparison in jaccard_comparisons:
        comparison['significance_sets'] = {}
        comparison['counts'] = {}
        for method in comparison_methods:
            adjusted_p_values = np.asarray(data[adjusted_p_columns[method][0]], dtype=float)
            selected = np.isfinite(adjusted_p_values) & (
                adjusted_p_values < comparison['thresholds'][method]
            )
            comparison['significance_sets'][method] = selected
            comparison['counts'][method] = int(np.count_nonzero(selected))

        matrix = np.full(
            (len(comparison_methods), len(comparison_methods)),
            np.nan,
            dtype=float
        )
        for row, method_a in enumerate(comparison_methods):
            significant_a = comparison['significance_sets'][method_a]
            for column, method_b in enumerate(comparison_methods):
                significant_b = comparison['significance_sets'][method_b]
                union_size = np.count_nonzero(np.logical_or(significant_a, significant_b))
                if union_size:
                    intersection_size = np.count_nonzero(
                        np.logical_and(significant_a, significant_b)
                    )
                    matrix[row, column] = intersection_size / union_size
        comparison['matrix'] = matrix

    # Prepare a single figure with a simple view switcher.
    fig = plt.figure(figsize=(12, 7))
    fig.subplots_adjust(bottom=0.12)

    # Overall comparison panel
    overall_ax = fig.add_subplot(111)
    overall_ax.set_position([0.08, 0.24, 0.84, 0.66])
    thresholds = [
        threshold_label("FWER", bonferroni_alpha),
        threshold_label("FWER", holm_alpha),
        threshold_label("FWER", pms_alpha),
        threshold_label("FDR", bh_q),
        threshold_label("FDR", by_q)
    ]
    bar_positions = np.arange(len(methods))
    bar_width = 0.38
    adjusted_bars = overall_ax.bar(
        bar_positions - bar_width / 2,
        counts,
        width=bar_width,
        color="#4c78a8",
        label="Region-adjusted partial Spearman"
    )
    raw_bars = overall_ax.bar(
        bar_positions + bar_width / 2,
        raw_counts,
        width=bar_width,
        color="#d62728",
        label="Raw Spearman"
    )
    overall_ax.bar_label(adjusted_bars, padding=2, fontsize=8)
    overall_ax.bar_label(raw_bars, padding=2, fontsize=8)
    max_count = max([*counts, *raw_counts], default=0)
    overall_ax.set_ylim(0, max_count * 1.16 if max_count else 1)
    overall_ax.set_xticks(
        bar_positions,
        [f"{method}\n{threshold}" for method, threshold in zip(methods, thresholds)],
        rotation=25,
        ha="right"
    )
    overall_ax.set_xlabel('Correction Method')
    overall_ax.set_ylabel('Number of Significant Results')
    overall_ax.set_title('Significant tests: region-adjusted partial vs raw Spearman')
    overall_ax.legend(loc="upper left", fontsize=8)

    # Histogram panel for all methods: 2 rows x 3 columns with independent axes.
    all_adjusted = {
        'Bonferroni': data['bonferroni_adjusted_p'],
        'Holm-Bonferroni': data['holm_bonferroni_adjusted_p'],
        'PMS': data['pms_adjusted_p'],
        'Benjamini-Hochberg': data['bh_adjusted_p'],
        'Benjamini-Yekutieli': data['by_adjusted_p']
    }
    all_raw_adjusted = {
        'Bonferroni': data['bonferroni_raw_adjusted_p'],
        'Holm-Bonferroni': data['holm_bonferroni_raw_adjusted_p'],
        'PMS': data['pms_raw_adjusted_p'],
        'Benjamini-Hochberg': data['bh_raw_adjusted_p'],
        'Benjamini-Yekutieli': data['by_raw_adjusted_p']
    }

    finite_values = [
        np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))]
        for values in [*all_adjusted.values(), *all_raw_adjusted.values()]
    ]
    values_below_one = [values[values < 1.0] for values in finite_values]
    global_min_p_value = min(
        (float(np.clip(values, 1e-12, 1.0).min()) for values in values_below_one if values.size),
        default=1e-12
    )
    histogram_bins = np.logspace(np.log10(global_min_p_value), 0, 31)

    hist_axes = fig.subplots(
        2,
        3,
        sharex=False,
        sharey=False,
        gridspec_kw={'left': 0.06, 'right': 0.94, 'bottom': 0.08, 'top': 0.92, 'wspace': 0.20, 'hspace': 0.28}
    )
    for idx, (method, values) in enumerate(all_adjusted.items()):
        row, col = divmod(idx, 3)
        ax = hist_axes[row, col]
        p_values = np.asarray(values, dtype=float)
        p_values = p_values[np.isfinite(p_values)]
        equal_to_one = p_values == 1.0
        below_one = np.clip(p_values[~equal_to_one], 1e-12, 1.0)

        # Keep the spike at exactly 1 out of the histogram scale and report it separately.
        bin_counts, bin_edges, patches = ax.hist(
            below_one,
            bins=histogram_bins,
            alpha=0.7
        )
        ax.bar_label(patches, labels=[f'{int(count)}' if count else '' for count in bin_counts], padding=2, fontsize=7)
        ax.set_xscale('log')
        ax.set_xlim(global_min_p_value, 1.0)
        ax.set_title(method)
        ax.set_xlabel('Adjusted p-value below 1 (log scale)')
        ax.set_ylabel('Frequency')
        ax.text(
            0.97,
            0.95,
            f'p = 1: {equal_to_one.sum():,}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'}
        )

    # There are five methods, so leave the sixth grid position empty.
    hist_axes[1, 2].set_axis_off()

    raw_hist_axes = fig.subplots(
        2,
        3,
        sharex=False,
        sharey=False,
        gridspec_kw={
            'left': 0.06,
            'right': 0.94,
            'bottom': 0.08,
            'top': 0.92,
            'wspace': 0.20,
            'hspace': 0.28
        }
    )
    for idx, (method, values) in enumerate(all_raw_adjusted.items()):
        row, col = divmod(idx, 3)
        ax = raw_hist_axes[row, col]
        p_values = np.asarray(values, dtype=float)
        p_values = p_values[np.isfinite(p_values)]
        equal_to_one = p_values == 1.0
        below_one = np.clip(p_values[~equal_to_one], 1e-12, 1.0)

        bin_counts, _, patches = ax.hist(
            below_one,
            bins=histogram_bins,
            alpha=0.7,
            color="#d62728"
        )
        ax.bar_label(
            patches,
            labels=[f'{int(count)}' if count else '' for count in bin_counts],
            padding=2,
            fontsize=7
        )
        ax.set_xscale('log')
        ax.set_xlim(global_min_p_value, 1.0)
        ax.set_title(f"{method}\nRaw Spearman")
        ax.set_xlabel('Adjusted p-value (log scale)', fontsize=9)
        ax.set_ylabel('Frequency')
        ax.text(
            0.97,
            0.95,
            f'p = 1: {equal_to_one.sum():,}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'}
        )
    raw_hist_axes[1, 2].set_axis_off()

    partial_rhos = np.asarray(data["partial_rho_region"], dtype=float)
    partial_rhos = partial_rhos[np.isfinite(partial_rhos)]
    if not partial_rhos.size:
        raise ValueError("No finite partial Spearman values are available to plot")

    comparison_axes = fig.subplots(
        2,
        2,
        sharex="col",
        gridspec_kw={
            'left': 0.08,
            'right': 0.94,
            'bottom': 0.10,
            'top': 0.90,
            'wspace': 0.28,
            'hspace': 0.35
        }
    )
    comparison_axes[0, 0].hist(partial_rhos, bins=120, range=(-1, 1), alpha=0.75, color="#2a78d6")
    comparison_axes[0, 0].set_xlim(-1, 1)
    comparison_axes[0, 0].set_title("Observed partial Spearman values")
    comparison_axes[0, 0].set_xlabel("Partial Spearman rho")
    comparison_axes[0, 0].set_ylabel("Number of gene pairs")

    comparison_axes[0, 1].hist(
        np.abs(partial_rhos), bins=120, range=(0, 1), alpha=0.75, color="#2a78d6"
    )
    critical_value = float(np.quantile(np.abs(max_statistics), 1 - pms_alpha))
    comparison_axes[0, 1].axvline(
        critical_value,
        color="black",
        linestyle="--",
        label=f"{1 - pms_alpha:.0%} null quantile = {critical_value:.3f}"
    )
    comparison_axes[0, 1].set_xlim(0, 1)
    comparison_axes[0, 1].set_title("Absolute observed partial Spearman values")
    comparison_axes[0, 1].set_xlabel("|Partial Spearman rho|")
    comparison_axes[0, 1].set_ylabel("Number of gene pairs")
    comparison_axes[0, 1].legend()

    comparison_axes[1, 0].hist(
        max_statistics, bins=120, range=(-1, 1), alpha=0.75, color="#e08b36"
    )
    comparison_axes[1, 0].set_xlim(-1, 1)
    comparison_axes[1, 0].set_title("Signed PMS permutation maxima")
    comparison_axes[1, 0].set_xlabel("Partial Spearman rho at maximum |rho|")
    comparison_axes[1, 0].set_ylabel("Number of permutations")

    comparison_axes[1, 1].hist(
        np.abs(max_statistics), bins=120, range=(0, 1), alpha=0.75, color="#e08b36"
    )
    comparison_axes[1, 1].axvline(
        critical_value,
        color="black",
        linestyle="--",
        label=f"{1 - pms_alpha:.0%} null quantile = {critical_value:.3f}"
    )
    comparison_axes[1, 1].set_xlim(0, 1)
    comparison_axes[1, 1].set_title("Absolute PMS permutation maxima")
    comparison_axes[1, 1].set_xlabel("Maximum |partial Spearman rho|")
    comparison_axes[1, 1].set_ylabel("Number of permutations")
    comparison_axes[1, 1].legend()

    raw_rhos = np.asarray(data["spearman_rho"], dtype=float)
    raw_rhos = raw_rhos[np.isfinite(raw_rhos)]
    if not raw_rhos.size:
        raise ValueError("No finite raw Spearman values are available to plot")

    raw_comparison_axes = fig.subplots(
        2,
        2,
        sharex="col",
        gridspec_kw={
            'left': 0.08,
            'right': 0.94,
            'bottom': 0.10,
            'top': 0.90,
            'wspace': 0.28,
            'hspace': 0.35
        }
    )
    raw_comparison_axes[0, 0].hist(
        raw_rhos, bins=120, range=(-1, 1), alpha=0.75, color="#d62728"
    )
    raw_comparison_axes[0, 0].set_xlim(-1, 1)
    raw_comparison_axes[0, 0].set_title("Observed raw Spearman values")
    raw_comparison_axes[0, 0].set_xlabel("Spearman rho")
    raw_comparison_axes[0, 0].set_ylabel("Number of gene pairs")

    raw_comparison_axes[0, 1].hist(
        np.abs(raw_rhos), bins=120, range=(0, 1), alpha=0.75, color="#d62728"
    )
    raw_critical_value = float(np.quantile(np.abs(raw_max_statistics), 1 - pms_alpha))
    raw_comparison_axes[0, 1].axvline(
        raw_critical_value,
        color="black",
        linestyle="--",
        label=f"{1 - pms_alpha:.0%} null quantile = {raw_critical_value:.3f}"
    )
    raw_comparison_axes[0, 1].set_xlim(0, 1)
    raw_comparison_axes[0, 1].set_title("Absolute observed raw Spearman values")
    raw_comparison_axes[0, 1].set_xlabel("|Spearman rho|")
    raw_comparison_axes[0, 1].set_ylabel("Number of gene pairs")
    raw_comparison_axes[0, 1].legend()

    raw_comparison_axes[1, 0].hist(
        raw_max_statistics, bins=120, range=(-1, 1), alpha=0.75, color="#2ca02c"
    )
    raw_comparison_axes[1, 0].set_xlim(-1, 1)
    raw_comparison_axes[1, 0].set_title("Signed raw PMS permutation maxima")
    raw_comparison_axes[1, 0].set_xlabel("Spearman rho at maximum |rho|")
    raw_comparison_axes[1, 0].set_ylabel("Number of permutations")

    raw_comparison_axes[1, 1].hist(
        np.abs(raw_max_statistics), bins=120, range=(0, 1), alpha=0.75, color="#2ca02c"
    )
    raw_comparison_axes[1, 1].axvline(
        raw_critical_value,
        color="black",
        linestyle="--",
        label=f"{1 - pms_alpha:.0%} null quantile = {raw_critical_value:.3f}"
    )
    raw_comparison_axes[1, 1].set_xlim(0, 1)
    raw_comparison_axes[1, 1].set_title("Absolute raw PMS permutation maxima")
    raw_comparison_axes[1, 1].set_xlabel("Maximum |Spearman rho|")
    raw_comparison_axes[1, 1].set_ylabel("Number of permutations")
    raw_comparison_axes[1, 1].legend()

    fwer_methods = [
        ("Bonferroni", data["bonferroni_adjusted_p"], bonferroni_alpha),
        ("Holm-Bonferroni", data["holm_bonferroni_adjusted_p"], holm_alpha),
        ("PMS", data["pms_adjusted_p"], pms_alpha)
    ]
    fdr_methods = [
        ("Benjamini-Hochberg", data["bh_adjusted_p"], bh_q),
        ("Benjamini-Yekutieli", data["by_adjusted_p"], by_q)
    ]


    shared_counts, shared_thresholds = thresholds_by_significant_count(
        fwer_methods + fdr_methods
    )
    fwer_counts = shared_counts
    fdr_counts = shared_counts
    fwer_thresholds = shared_thresholds
    fdr_thresholds = shared_thresholds
    sensitivity_ax = fig.add_axes([0.08, 0.14, 0.86, 0.74])
    fdr_ax = sensitivity_ax.twinx()
    method_colours = {
        "Bonferroni": "#1f77b4",
        "Holm-Bonferroni": "#ff7f0e",
        "PMS": "#2ca02c",
        "Benjamini-Hochberg": "#d62728",
        "Benjamini-Yekutieli": "#9467bd"
    }

    for method, _, _ in fwer_methods:
        sensitivity_ax.plot(
            fwer_counts,
            [fwer_thresholds[method][count] for count in fwer_counts],
            color=method_colours[method],
            marker="o",
            label=f"{method} (FWER)"
        )
    for method, _, _ in fdr_methods:
        fdr_ax.plot(
            fdr_counts,
            [fdr_thresholds[method][count] for count in fdr_counts],
            color=method_colours[method],
            marker="s",
            label=f"{method} (FDR)"
        )

    sensitivity_ax.axhline(
        0.05,
        color="black",
        linestyle="--",
        linewidth=1,
        label="FWER = 5%"
    )
    sensitivity_ax.set_xlabel("Number of significant hypotheses")
    sensitivity_ax.set_ylabel("FWER threshold")
    fdr_ax.set_ylabel("FDR threshold")
    sensitivity_ax.set_ylim(0, 1)
    fdr_values = np.asarray([
        fdr_thresholds[method][count]
        for method, _, _ in fdr_methods
        for count in fdr_counts
    ], dtype=float)
    fdr_min = float(fdr_values.min())
    fdr_max = float(fdr_values.max())
    fdr_padding = (fdr_max - fdr_min) * 0.08
    if fdr_padding == 0:
        fdr_padding = max(fdr_max * 0.08, 1e-6)
    fdr_lower = max(0.0, fdr_min - fdr_padding)
    fdr_upper = min(1.0, fdr_max + fdr_padding)
    fdr_ax.set_ylim(fdr_lower, fdr_upper)
    if fdr_lower <= 0.05 <= fdr_upper:
        fdr_ax.axhline(
            0.05,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label="FDR = 5%"
        )
    handles, labels = sensitivity_ax.get_legend_handles_labels()
    fdr_handles, fdr_labels = fdr_ax.get_legend_handles_labels()
    sensitivity_ax.legend(
        handles + fdr_handles,
        labels + fdr_labels,
        loc="best",
        fontsize=8
    )
    sensitivity_axes = (sensitivity_ax, fdr_ax)

    jaccard_short_names = {
        'Holm-Bonferroni': 'Holm',
        'PMS': 'PMS',
        'Benjamini-Hochberg': 'BH',
        'Benjamini-Yekutieli': 'BY'
    }
    jaccard_page_axes = []
    for comparison in jaccard_comparisons:
        jaccard_ax = fig.add_axes([0.23, 0.25, 0.58, 0.62])
        jaccard_colorbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.62])
        jaccard_image = jaccard_ax.imshow(
            comparison['matrix'],
            cmap="YlGnBu",
            vmin=0,
            vmax=1
        )
        jaccard_tick_labels = [
            (
                f"{jaccard_short_names[method]}\n"
                f"{adjusted_p_columns[method][1]}="
                f"{comparison['thresholds'][method]:.0%}\n"
                f"n={comparison['counts'][method]}"
            )
            for method in comparison_methods
        ]
        jaccard_ax.set_xticks(
            np.arange(len(comparison_methods)),
            labels=jaccard_tick_labels,
            rotation=0
        )
        jaccard_ax.set_yticks(
            np.arange(len(comparison_methods)),
            labels=jaccard_tick_labels
        )
        jaccard_ax.set_title(comparison['title'])
        jaccard_ax.set_xlabel(
            "Method (Holm = Holm-Bonferroni; BH = Benjamini-Hochberg; "
            "BY = Benjamini-Yekutieli)"
        )
        jaccard_ax.set_ylabel("Method, threshold, and significant-test count")
        jaccard_colorbar = fig.colorbar(jaccard_image, cax=jaccard_colorbar_ax)
        jaccard_colorbar.set_label("Jaccard coefficient")
        for row in range(len(comparison_methods)):
            for column in range(len(comparison_methods)):
                value = comparison['matrix'][row, column]
                label = f"{value:.3f}" if np.isfinite(value) else "N/A"
                text_colour = "white" if np.isfinite(value) and value >= 0.55 else "black"
                jaccard_ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_colour
                )
        jaccard_page_axes.append((jaccard_ax, jaccard_colorbar_ax))

    venn_page_axes = []
    venn_colours = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]
    venn_method_labels = {
        'Bonferroni': 'Bonferroni',
        'Holm-Bonferroni': 'Holm-Bonferroni',
        'PMS': 'PMS',
        'Benjamini-Hochberg': 'BH',
        'Benjamini-Yekutieli': 'BY'
    }
    for comparison in jaccard_comparisons:
        venn_ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
        method_sets = []
        for method in ['Bonferroni'] + comparison_methods:
            adjusted_column = (
                'bonferroni_adjusted_p'
                if method == 'Bonferroni'
                else adjusted_p_columns[method][0]
            )
            adjusted_p_values = np.asarray(data[adjusted_column], dtype=float)
            selected = np.isfinite(adjusted_p_values) & (
                adjusted_p_values < comparison['thresholds'].get(
                    method,
                    comparison['thresholds']['Holm-Bonferroni']
                )
            )
            threshold = comparison['thresholds'].get(
                method,
                comparison['thresholds']['Holm-Bonferroni']
            )
            method_sets.append((method, threshold, selected))

        grouped_sets = []
        for method, threshold, selected in method_sets:
            matching_group = next(
                (
                    group for group in grouped_sets
                    if np.array_equal(group['selected'], selected)
                ),
                None
            )
            if matching_group is None:
                grouped_sets.append({
                    'methods': [(method, threshold)],
                    'selected': selected
                })
            else:
                matching_group['methods'].append((method, threshold))
        grouped_sets.sort(
            key=lambda group: int(np.count_nonzero(group['selected'])),
            reverse=True
        )
        is_nested = all(
            np.all(~group_a['selected'] | group_b['selected'])
            or np.all(~group_b['selected'] | group_a['selected'])
            for index, group_a in enumerate(grouped_sets)
            for group_b in grouped_sets[index + 1:]
        )

        if is_nested and any(np.count_nonzero(group['selected']) for group in grouped_sets):
            largest_count = max(
                int(np.count_nonzero(group['selected']))
                for group in grouped_sets
            )
            max_radius = 3.0
            circle_radii = []
            for index, group in enumerate(grouped_sets):
                count = int(np.count_nonzero(group['selected']))
                radius = max_radius * np.sqrt(count / largest_count) if count else 0
                circle_radii.append(radius)
                circle = Circle(
                    (0, 0),
                    radius,
                    facecolor=venn_colours[index % len(venn_colours)],
                    edgecolor=venn_colours[index % len(venn_colours)],
                    alpha=0.16,
                    linewidth=2.2
                )
                venn_ax.add_patch(circle)

            label_y_positions = np.linspace(2.2, -2.2, len(grouped_sets))
            for group, radius, label_y in zip(
                grouped_sets,
                circle_radii,
                label_y_positions
            ):
                count = int(np.count_nonzero(group['selected']))
                method_thresholds = [
                    f"{venn_method_labels[method]} "
                    f"{adjusted_p_columns.get(method, ('', 'FWER'))[1]}={threshold:.0%}"
                    for method, threshold in group['methods']
                ]
                label = " + ".join(method_thresholds) + f"\nn={count:,}"
                venn_ax.annotate(
                    label,
                    xy=(radius, 0),
                    xytext=(3.35, label_y),
                    ha="left",
                    va="center",
                    fontsize=9,
                    arrowprops={'arrowstyle': '-', 'color': '#555555', 'lw': 0.9}
                )
            venn_ax.set_xlim(-3.3, 8.0)
            venn_ax.set_ylim(-3.45, 3.45)
            venn_ax.set_aspect('equal')
            venn_ax.set_title(
                f"Nested significance sets: {comparison['title']}\n"
                "Circle area is proportional to significant tests; coincident Bonferroni/Holm sets share a boundary"
            )
        else:
            patterns = {}
            for index in range(len(data)):
                members = tuple(
                    method
                    for method, _, selected in method_sets
                    if selected[index]
                )
                if members:
                    patterns[members] = patterns.get(members, 0) + 1
            ordered_patterns = sorted(patterns.items(), key=lambda item: item[1], reverse=True)
            venn_ax.bar(
                np.arange(len(ordered_patterns)),
                [count for _, count in ordered_patterns],
                color="#4c78a8"
            )
            venn_ax.set_xticks(
                np.arange(len(ordered_patterns)),
                labels=[
                    " + ".join(venn_method_labels[method] for method in members)
                    for members, _ in ordered_patterns
                ],
                rotation=45,
                ha='right'
            )
            venn_ax.set_ylabel("Number of gene-pair tests")
            venn_ax.set_title(
                f"Exact significant-method intersections: {comparison['title']}\n"
                "Sets are not nested, so intersections are shown as exact membership groups"
            )
            venn_ax.bar_label(venn_ax.containers[0], padding=2, fontsize=8)
        venn_page_axes.append(venn_ax)

    raw_p_values = np.asarray(data['partial_p_value'], dtype=float)
    p_value_plot_methods = [
        ('Bonferroni', 'bonferroni_adjusted_p', (0.50,)),
        ('Holm-Bonferroni', 'holm_bonferroni_adjusted_p', (0.50,)),
        ('PMS', 'pms_adjusted_p', (0.50,)),
        ('Benjamini-Hochberg', 'bh_adjusted_p', (0.05, 0.01)),
        ('Benjamini-Yekutieli', 'by_adjusted_p', (0.05, 0.01))
    ]
    plot_p_value_arrays = [
        np.asarray(data[column], dtype=float)
        for _, column, _ in p_value_plot_methods
    ]
    finite_positive_p_values = [
        values[np.isfinite(values) & (values > 0)]
        for values in [raw_p_values, *plot_p_value_arrays]
    ]
    plot_p_min = min(float(values.min()) for values in finite_positive_p_values if values.size)
    p_value_axes = fig.subplots(
        2,
        3,
        sharex=True,
        sharey=True,
        gridspec_kw={
            'left': 0.08,
            'right': 0.96,
            'bottom': 0.12,
            'top': 0.91,
            'wspace': 0.28,
            'hspace': 0.34
        }
    )
    for index, (method, _, thresholds_for_method) in enumerate(p_value_plot_methods):
        row, column = divmod(index, 3)
        ax = p_value_axes[row, column]
        adjusted_p_values = plot_p_value_arrays[index]
        valid = (
            np.isfinite(raw_p_values)
            & np.isfinite(adjusted_p_values)
            & (raw_p_values > 0)
            & (adjusted_p_values > 0)
        )
        ax.scatter(
            raw_p_values[valid],
            adjusted_p_values[valid],
            s=3,
            alpha=0.12,
            color=method_colours[method],
            rasterized=True
        )
        ax.plot(
            [plot_p_min, 1],
            [plot_p_min, 1],
            color='black',
            linestyle=':',
            linewidth=1,
            label='Adjusted p = raw p'
        )
        for threshold in thresholds_for_method:
            ax.axhline(
                threshold,
                linestyle='--',
                linewidth=1,
                label=f"Adjusted p = {threshold:.0%}"
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(plot_p_min, 1)
        ax.set_ylim(plot_p_min, 1)
        ax.set_title(method)
        ax.set_xlabel('Raw partial Spearman p-value')
        ax.set_ylabel('Adjusted p-value')
        ax.legend(fontsize=7, loc='lower right')
    p_value_axes[1, 2].set_axis_off()
    p_value_axes[1, 2].text(
        0.5,
        0.5,
        "Each point = one gene pair\n\n"
        "Log-log axes\n"
        "Dotted: adjusted = raw\n"
        "Dashed: significance thresholds\n"
        "BH/BY: FDR 5% and 1%",
        transform=p_value_axes[1, 2].transAxes,
        ha='center',
        va='center',
        fontsize=11
    )
    p_value_axes = tuple(p_value_axes.flat)

    # Hide graph pages initially and show the comparison plot.
    for ax in hist_axes.flat:
        ax.set_visible(False)
    for ax in raw_hist_axes.flat:
        ax.set_visible(False)
    for ax in comparison_axes.flat:
        ax.set_visible(False)
    for ax in raw_comparison_axes.flat:
        ax.set_visible(False)
    for ax in sensitivity_axes:
        ax.set_visible(False)
    for axes in jaccard_page_axes:
        for ax in axes:
            ax.set_visible(False)
    for ax in venn_page_axes:
        ax.set_visible(False)
    for ax in p_value_axes:
        ax.set_visible(False)
    view_index = 0

    # Arrow buttons to switch between the comparison plot and histograms.
    button_prev_ax = fig.add_axes([0.02, 0.48, 0.03, 0.05])
    button_next_ax = fig.add_axes([0.95, 0.48, 0.03, 0.05])
    button_prev = Button(button_prev_ax, '<')
    button_next = Button(button_next_ax, '>')


    def show_view(index):
        nonlocal view_index
        view_index = index % 11
        overall_ax.set_visible(view_index == 0)
        for ax in hist_axes.flat:
            ax.set_visible(view_index == 1)
        for ax in raw_hist_axes.flat:
            ax.set_visible(view_index == 2)
        for ax in comparison_axes.flat:
            ax.set_visible(view_index == 3)
        for ax in raw_comparison_axes.flat:
            ax.set_visible(view_index == 4)
        for ax in sensitivity_axes:
            ax.set_visible(view_index == 5)
        for page_index, axes in enumerate(jaccard_page_axes):
            for ax in axes:
                ax.set_visible(view_index == page_index + 6)
        for page_index, ax in enumerate(venn_page_axes):
            ax.set_visible(view_index == page_index + 8)
        for ax in p_value_axes:
            ax.set_visible(view_index == 10)
        fig.canvas.draw_idle()

    button_prev.on_clicked(lambda event: show_view(view_index - 1))
    button_next.on_clicked(lambda event: show_view(view_index + 1))

    show_view(0)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = [
        "significant_tests_by_method.png",
        "adjusted_p_value_histograms.png",
        "raw_spearman_adjusted_p_value_histograms.png",
        "partial_spearman_and_pms_histograms.png",
        "raw_spearman_and_pms_histograms.png",
        "fwer_fdr_thresholds_by_significant_tests.png",
        *[comparison['filename'] for comparison in jaccard_comparisons],
        "significant_methods_venn_fwer50_fdr5.png",
        "significant_methods_venn_fwer50_fdr1.png",
        "raw_vs_adjusted_p_values.png"
    ]
    button_prev_ax.set_visible(False)
    button_next_ax.set_visible(False)
    for page, filename in enumerate(plot_files):
        show_view(page)
        fig.savefig(plot_output_dir / filename, dpi=300)
        print(f"Saved plot: {plot_output_dir / filename}")
    button_prev_ax.set_visible(True)
    button_next_ax.set_visible(True)
    show_view(0)
    plt.show()
