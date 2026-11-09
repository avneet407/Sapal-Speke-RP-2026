"""Bar chart of how many isoforms each gene has."""

import os                          # making folders and building file paths
from datetime import datetime      # timestamp in figure file names, so old figures aren't overwritten

import numpy as np                 # maths on arrays (log, NaN, infinity)
import matplotlib.pyplot as plt    # plotting


def plot_isoforms_per_gene(binned, max_isoforms, bin_edges, bin_labels, figure_folder):
    """Histogram of isoforms per gene on a log x-axis, one bar per range (1, 2, 3-5, 6-10, ...),
    the same ranges as the printed table.
    Each bar spans its range on the log axis, so wider ranges (e.g. 201-500) give wider bars.
    Log scale because genes have from 1 to ~1,400 isoforms; on a normal axis the 1-20 bars
    would be squashed into a sliver at the left.
    binned       = number of genes in each range (a Series indexed by bin_labels)
    max_isoforms = the largest gene's isoform count, used as the right edge of the last bar (">500")."""
    # Bar edges: the counts are whole numbers, so put each edge half-way between two numbers.
    # The range "3-5" then runs from 2.5 to 5.5 and "1" from 0.5 to 1.5 - no gaps, no overlaps.
    # (log(0) is impossible, so the first edge must be above 0 anyway.)
    edges = np.array(bin_edges[:-1] + [max_isoforms]) + 0.5   # replace the last edge (infinity) with the max
    left, right = edges[:-1], edges[1:]

    fig = plt.figure(figsize=(9, 5))                    # new figure, 9 x 5 inches
    # align="edge" + width = right - left: each bar starts at its range's left edge and ends at its right edge
    bars = plt.bar(left, binned.values, width=right - left, align="edge",
                   color="#2a78d6", edgecolor="white")
    plt.xscale("log")
    # Write the number of genes on top of each bar
    plt.bar_label(bars, padding=2, fontsize=9)

    # Label each bar with its range, at the bar's middle. On a log axis the middle is the
    # geometric mean sqrt(left * right), not (left + right) / 2.
    plt.xticks(np.sqrt(left * right), bin_labels, fontsize=8)
    plt.minorticks_off()                                # hide the log axis's own small tick marks
    plt.xlim(left[0], right[-1])
    plt.xlabel("Number of isoforms in the gene (log scale)")
    plt.ylabel("Number of genes")
    plt.title(f"Isoforms per gene ({binned.sum()} genes)")
    plt.tight_layout()                                  # stops labels being cut off

    # Save the figure: make the folder if needed, add a timestamp to the file name
    os.makedirs(figure_folder, exist_ok=True)           # exist_ok=True: no error if the folder already exists
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(figure_folder, f"isoforms_per_gene_{timestamp}.png")
    fig.savefig(path, dpi=200)                          # dpi = resolution
    print(f"\nSaved figure: {path}")
    plt.show()                                          # opens the figure window (script pauses until closed)
