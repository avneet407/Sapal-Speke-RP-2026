"""
Isoform structure viewer: draw every isoform of a gene, one row per isoform, to compare their structure.

  - exons  = blocks (grey = untranslated / UTR part, red = coding / CDS part)
  - introns = thin black lines joining the exons
  - a coordinate ruler above the isoforms labels every distinct exon start / end once, with a
    dotted guide line down through every row (exon edges on the same line = shared splice site)
  - each isoform has 5' and 3' written at its ends (which side is 5' depends on the strand)
  - isoforms are ordered by mean CPM (most abundant at the top), with the CPM in the row label

Structure comes from Long_Read_Data/expressed_isoforms.gtf ("exon" and "CDS" rows),
abundance from Long_Read_Data/expressed_isoforms_matrix.tsv (same 61,117 transcript IDs).
Non-coding isoforms (~15.6k, no CDS rows) are drawn all grey.

A gene can be given as:
    gene name        SLC45A1
    Ensembl gene ID  ENSG00000162426 (version .16 optional)
    PacBio locus     G249            (the part of the transcript ID before the first ".")

Usage (from the Sapal-Speke-RP-2026 folder):
    python isoform_visualisation.py SLC45A1          # one gene, top MAX_ISOFORMS isoforms
    python isoform_visualisation.py SLC45A1 all      # every isoform of the gene
    python isoform_visualisation.py SLC45A1 40       # top 40 isoforms
    python isoform_visualisation.py SLC45A1 G70506   # several genes, one figure each
    python isoform_visualisation.py                  # asks for gene names one at a time
Or from Python / a notebook:
    from isoform_visualisation import plot_gene
    plot_gene("SLC45A1")
"""

import os
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath


# --- Settings (change these, not the code below) ---------------------------
GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
FIGURES_FOLDER = "Figures/isoform_visualisation"

MAX_ISOFORMS = 20        # show only the N most abundant isoforms; None = show all
                         # (some genes have ~100 isoforms, which makes a very tall figure)
COMPRESS_INTRONS = True  # True = shrink long introns so the exons are wide enough to see.
                         # Exon lengths stay to scale; the coordinate labels are still the
                         # real genomic positions. False = everything to true genomic scale.
MAX_INTRON_FRACTION = 0.05  # with COMPRESS_INTRONS, no intron is drawn longer than this
                            # fraction of the total exonic length (shorter introns keep their size)
SAVE_FIGURE = True          # True = also save a PNG in FIGURES_FOLDER (timestamped, never overwritten)

FIGURE_WIDTH_INCHES = 14
ROW_HEIGHT_INCHES = 0.5     # vertical space per isoform
RULER_LINES_INCHES = 0.4    # height of the ruler's connector lines; the coordinate labels sit above
                            # them and get exactly as much room as the longest label needs
TITLE_SPACE_INCHES = 0.6    # room for the two-line title above the labels
EXON_HEIGHT = 0.4        # height of an exon block (each isoform row is 1 unit tall)
EXON_COLOUR = "lightgrey"
CDS_COLOUR = "red"
LABEL_FONT_SIZE = 5


# Data is read once and kept here, so plotting several genes doesn't re-read the 110 MB GTF
_cache = {}


def read_gtf(gtf_path=GTF_PATH):
    """Read the GTF into {transcript_id: info}, where info is a dict with
         chrom, strand, gene_name, gene_id,
         exons: list of (start, end), cds: list of (start, end)   [1-based, both ends included]
    Every line has the attributes  transcript_id "..."; gene_name "..."; gene_id "...";
    Takes a few seconds (~900k lines)."""
    attribute = re.compile(r'(\w+) "([^"]*)"')
    transcripts = defaultdict(lambda: {"exons": [], "cds": []})
    with open(gtf_path) as gtf:
        for line in gtf:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, feature, start, end, strand = fields[0], fields[2], int(fields[3]), int(fields[4]), fields[6]
            attributes = dict(attribute.findall(fields[8]))
            info = transcripts[attributes["transcript_id"]]
            if feature == "transcript":
                info.update(chrom=chrom, strand=strand,
                            gene_name=attributes.get("gene_name", ""),
                            gene_id=attributes.get("gene_id", ""))
            elif feature == "exon":
                info["exons"].append((start, end))
            elif feature == "CDS":
                info["cds"].append((start, end))
    for info in transcripts.values():
        info["exons"].sort()
        info["cds"].sort()
    return dict(transcripts)


def read_mean_cpm(matrix_path=MATRIX_PATH):
    """Mean CPM of each isoform across all samples, as a pandas Series indexed by transcript_id.
    Only used to order the isoforms and label them, so if the matrix is missing return None."""
    if not os.path.exists(matrix_path):
        print(f"Note: {matrix_path} not found - isoforms will be ordered by position, not abundance")
        return None
    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
    return matrix.mean(axis=1)


def load_data():
    """Read the GTF and matrix the first time they are needed, then reuse them."""
    if not _cache:
        print("Reading GTF and expression matrix...")
        _cache["transcripts"] = read_gtf()
        _cache["mean_cpm"] = read_mean_cpm()
    return _cache["transcripts"], _cache["mean_cpm"]


def find_isoforms(gene, transcripts):
    """All transcript IDs belonging to a gene given by name, Ensembl ID or PacBio locus (see top).
    Returns an empty list if nothing matches.
    (Every gene in expressed_isoforms.gtf lies on a single chromosome and strand, so all the
    matches can be drawn on one shared axis.)"""
    query = gene.strip()
    query_no_version = query.split(".")[0]   # ENSG00000162426.16 -> ENSG00000162426
    return [transcript_id for transcript_id, info in transcripts.items()
            if info.get("gene_name", "").upper() == query.upper()
            or info.get("gene_id", "").split(".")[0] == query_no_version
            or transcript_id.split(".")[0] == query]


def make_x_axis(isoforms, transcripts):
    """Return a function converting a genomic position to an x position on the plot.

    Without compression this is just x = position. With compression:
      1. merge all exons of all shown isoforms into non-overlapping exonic blocks
      2. lay the blocks out left to right at their true length
      3. the gap between two blocks (intronic in every isoform) is drawn at its true length,
         but capped at MAX_INTRON_FRACTION of the total exonic length
    Every exon / CDS boundary lies inside a block, so each one maps exactly."""
    if not COMPRESS_INTRONS:
        return lambda position: position

    # 1. merge overlapping / touching exons
    blocks = []
    for start, end in sorted(exon for t in isoforms for exon in transcripts[t]["exons"]):
        if blocks and start <= blocks[-1][1] + 1:
            blocks[-1][1] = max(blocks[-1][1], end)
        else:
            blocks.append([start, end])

    # 2-3. work out where each block starts on the plot
    exonic_length = sum(end - start + 1 for start, end in blocks)
    max_gap = MAX_INTRON_FRACTION * exonic_length
    block_starts, block_shifts = [], []   # x = position + shift of the block it falls in
    x = 0
    for i, (start, end) in enumerate(blocks):
        if i > 0:
            x += min(start - blocks[i - 1][1] - 1, max_gap)
        block_starts.append(start)
        block_shifts.append(x - start)
        x += end - start + 1

    def to_x(position):
        block = bisect_right(block_starts, position) - 1   # last block starting at or before position
        return position + block_shifts[block]
    return to_x


def spread_labels(targets, gap, low, high):
    """Positions for labels that want to sit at `targets` (sorted x values) but must be at least
    `gap` apart. Two passes:
      forward  - go left to right, pushing each label right of the previous one if too close
      backward - go right to left, pushing each label left of the next one if too close
    Both results keep every pair >= gap apart, and so does their average, which centres each
    crowded group on its true positions instead of shoving the whole group to one side."""
    forward = []
    for x in targets:
        forward.append(max(x, low) if not forward else max(x, forward[-1] + gap))
    backward = [0.0] * len(targets)
    for i in reversed(range(len(targets))):
        backward[i] = min(targets[i], high) if i == len(targets) - 1 else min(targets[i], backward[i + 1] - gap)
    return [(f + b) / 2 for f, b in zip(forward, backward)]


def exon_boundaries(isoforms, transcripts, to_x):
    """Every distinct exon boundary of the shown isoforms: sorted x positions and their labels.
    Exon starts sit at the left edge of a block (to_x(start)), exon ends at the right edge
    (to_x(end) + 1). Different coordinates at the same x (e.g. one isoform's exon ends at 100,
    another's starts at 101) share one label, "100/101"."""
    boundaries = defaultdict(set)
    for transcript_id in isoforms:
        for start, end in transcripts[transcript_id]["exons"]:
            boundaries[to_x(start)].add(start)
            boundaries[to_x(end) + 1].add(end)
    xs = sorted(boundaries)
    texts = ["/".join(f"{c:,}" for c in sorted(boundaries[x])) for x in xs]
    return xs, texts


def draw_ruler(fig, ruler_ax, ax, xs, texts):
    """Label every exon boundary once, in the ruler above the isoform rows. Labels are spread
    out sideways so they never overlap, with a line pointing each label to its true position.
    Returns how tall the labels are in inches, so the layout can leave exactly that much room.

    The spacing depends on how wide the axes are in inches RIGHT NOW, so this is called again
    whenever the window is resized (a figure too big for the screen gets shrunk to fit, but
    text keeps its size, so labels spaced for the full-size figure would collide).

    Ruler layout (y from 0 at the bottom to 1 at the top of ruler_ax):
      0   -> 0.4  vertical tick at the boundary's true x position
      0.4 -> 1    slanted line to the label's (possibly shifted) x position
      above 1     the coordinate, written vertically, starting 2 points above the axes.
                  It is placed in points (not axes units) so its size never depends on how
                  tall the ruler happens to be."""
    ruler_ax.clear()
    x_low, x_high = ax.get_xlim()
    ruler_ax.set_xlim(x_low, x_high)
    ruler_ax.set_ylim(0, 1)
    ruler_ax.set_autoscale_on(False)   # otherwise plotting the connectors re-fits the limits and
                                       # the ruler no longer lines up with the isoform rows
    ruler_ax.axis("off")

    # Vertical text is about one font-height wide, so labels need ~1.3 x font size (in points)
    # between them. If there are too many boundaries to fit at LABEL_FONT_SIZE, shrink the font.
    width_inches = ruler_ax.get_position().width * fig.get_size_inches()[0]
    x_units_per_inch = (x_high - x_low) / width_inches
    font_size = min(LABEL_FONT_SIZE, 72 * width_inches / (1.3 * len(xs)))
    gap = 1.3 * font_size / 72 * x_units_per_inch
    label_xs = spread_labels(xs, gap, x_low, x_high)

    # x in data units, y in axes units (0 = bottom of the ruler, 1 = top)
    x_data_y_axes = ruler_ax.get_xaxis_transform()
    for x, label_x, text in zip(xs, label_xs, texts):
        ruler_ax.plot([x, x, label_x], [0, 0.4, 1], color="grey", linewidth=0.5)
        ruler_ax.annotate(text, xy=(label_x, 1), xycoords=x_data_y_axes,
                          xytext=(0, 2), textcoords="offset points",
                          rotation=90, ha="center", va="bottom", fontsize=font_size)

    # Length of the longest label (= its height, as it's vertical): TextPath measures the text
    # in points at this font size without needing to draw it
    longest = max(TextPath((0, 0), text, size=font_size).get_extents().width for text in texts)
    return (longest + 2) / 72


def place_axes(fig, ax, ruler_ax, title, label_inches, left_inches):
    """Position everything in inches, measured from the edges of the figure's current size:
         top:     title (TITLE_SPACE_INCHES), then the ruler labels (label_inches)
         then:    the ruler's connector lines (RULER_LINES_INCHES)
         then:    the isoform rows - they get whatever height is left
         bottom:  0.5 inch for the x-axis label
         left:    left_inches for the row labels (transcript ID + CPM)
         right:   0.4 inch for the 3' / 5' marks
    Because text gets fixed room in inches, nothing overlaps or gets cut off even when the
    window shrinks the figure; only the drawing area itself gets smaller."""
    width_inches, height_inches = fig.get_size_inches()
    bottom, right = 0.5, 0.4
    top = TITLE_SPACE_INCHES + label_inches
    rows = max(height_inches - bottom - RULER_LINES_INCHES - top, 0.1)
    left = left_inches / width_inches                                  # as fractions of the figure
    width = max(width_inches - left_inches - right, 0.1) / width_inches
    ax.set_position([left, bottom / height_inches, width, rows / height_inches])
    ruler_ax.set_position([left, (bottom + rows) / height_inches,
                           width, RULER_LINES_INCHES / height_inches])
    title.set_y(1 - 0.1 / height_inches)


def plot_gene(gene, max_isoforms=MAX_ISOFORMS, save=SAVE_FIGURE, show=True):
    """Draw the isoform structures of one gene. Returns the matplotlib figure (None if not found)."""
    transcripts, mean_cpm = load_data()
    isoforms = find_isoforms(gene, transcripts)
    if not isoforms:
        print(f"No isoforms found for '{gene}' - check the gene name / ID")
        return None

    # Order: most abundant first (or by start position if there is no matrix), then keep the top N
    if mean_cpm is not None:
        isoforms.sort(key=lambda t: -mean_cpm.get(t, 0))
    else:
        isoforms.sort(key=lambda t: transcripts[t]["exons"][0][0])
    n_total = len(isoforms)
    if max_isoforms is not None:
        isoforms = isoforms[:max_isoforms]

    first = transcripts[isoforms[0]]
    chrom, strand = first["chrom"], first["strand"]
    gene_name = first["gene_name"] or gene
    to_x = make_x_axis(isoforms, transcripts)

    # Two axes placed by hand (not tight_layout) so the ruler sits exactly above the rows:
    # ax = isoform rows, ruler_ax = coordinate ruler. Their final positions are set by
    # place_axes() at the end, once the label height is known; these are placeholders.
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, ROW_HEIGHT_INCHES * len(isoforms) + 3))
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.6])
    ruler_ax = fig.add_axes([0.2, 0.7, 0.7, 0.1])
    half = EXON_HEIGHT / 2
    x_min, x_max = float("inf"), float("-inf")

    for row, transcript_id in enumerate(isoforms):
        info = transcripts[transcript_id]
        exons, cds = info["exons"], info["cds"]
        left, right = to_x(exons[0][0]), to_x(exons[-1][1]) + 1
        x_min, x_max = min(x_min, left), max(x_max, right)

        # Intron line: one line under the whole isoform; the exon blocks are drawn on top of it
        ax.plot([left, right], [row, row], color="black", linewidth=0.8, zorder=1)

        for exon_start, exon_end in exons:
            x0, x1 = to_x(exon_start), to_x(exon_end) + 1
            # Exon block (grey); any coding part is painted red over it below
            ax.add_patch(Rectangle((x0, row - half), x1 - x0, EXON_HEIGHT,
                                   facecolor=EXON_COLOUR, edgecolor="black", linewidth=0.5, zorder=2))

        # CDS segments in red, inside the exon blocks (CDS rows are already split per exon)
        for cds_start, cds_end in cds:
            x0, x1 = to_x(cds_start), to_x(cds_end) + 1
            ax.add_patch(Rectangle((x0, row - half), x1 - x0, EXON_HEIGHT,
                                   facecolor=CDS_COLOUR, edgecolor="none", zorder=3))
        # Redraw the exon outlines on top so the red doesn't hide the exon edges
        for exon_start, exon_end in exons:
            x0, x1 = to_x(exon_start), to_x(exon_end) + 1
            ax.add_patch(Rectangle((x0, row - half), x1 - x0, EXON_HEIGHT,
                                   facecolor="none", edgecolor="black", linewidth=0.5, zorder=4))

        # 5' / 3' at the ends: on the + strand 5' is on the left, on the - strand it is on the right
        left_end, right_end = ("5'", "3'") if strand == "+" else ("3'", "5'")
        ax.text(left, row, left_end + " ", ha="right", va="center", fontsize=7, fontweight="bold")
        ax.text(right, row, " " + right_end, ha="left", va="center", fontsize=7, fontweight="bold")

    # Row labels: transcript ID, mean CPM, and "non-coding" if there is no CDS
    labels = []
    for transcript_id in isoforms:
        label = transcript_id
        if mean_cpm is not None:
            label += f"  ({mean_cpm.get(transcript_id, 0):.1f} CPM)"
        if not transcripts[transcript_id]["cds"]:
            label += "  non-coding"
        labels.append(label)
    ax.set_yticks(range(len(isoforms)))
    ax.set_yticklabels(labels, fontsize=7)
    # Left margin = longest row label (measured in points at font size 7) + room for the
    # tick padding and the 5'/3' mark that sits just left of each isoform
    left_inches = max(TextPath((0, 0), label, size=7).get_extents().width for label in labels) / 72 + 0.45
    ax.set_ylim(len(isoforms) - 0.4, -0.6)   # flipped: first (most abundant) isoform at the top

    # x axis: real coordinates only make sense uncompressed; compressed, the labels carry them
    padding = 0.03 * (x_max - x_min)
    ax.set_xlim(x_min - padding, x_max + padding)
    if COMPRESS_INTRONS:
        ax.set_xticks([])
        ax.set_xlabel(f"{chrom} ({strand} strand) - introns compressed, exons to scale")
    else:
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlabel(f"{chrom} position ({strand} strand)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_autoscale_on(False)   # keep these x limits; the ruler copies them

    # Faint dotted guide line down through every isoform row at each exon boundary
    xs, texts = exon_boundaries(isoforms, transcripts, to_x)
    for x in xs:
        ax.axvline(x, color="grey", linestyle=":", linewidth=0.5, zorder=0)

    shown = f"top {len(isoforms)} of {n_total}" if len(isoforms) < n_total else f"all {n_total}"
    title = fig.suptitle(f"{gene_name} ({first['gene_id']}) - {shown} isoforms, most abundant first\n"
                         f"grey = exon (UTR), red = CDS, line = intron", va="top", fontsize=11)

    # Final layout: draw the ruler labels to find out how tall they are, set the figure height
    # so every isoform row gets ROW_HEIGHT_INCHES, then place the axes and title
    # (place_axes runs first so the ruler's width in inches is right before its labels are spaced)
    place_axes(fig, ax, ruler_ax, title, 0, left_inches)
    label_inches = draw_ruler(fig, ruler_ax, ax, xs, texts)
    fig.set_size_inches(FIGURE_WIDTH_INCHES, 0.5 + ROW_HEIGHT_INCHES * len(isoforms)
                        + RULER_LINES_INCHES + label_inches + TITLE_SPACE_INCHES)
    place_axes(fig, ax, ruler_ax, title, label_inches, left_inches)

    # If the window changes size (e.g. shrunk to fit the screen), re-space the labels for the
    # new width and redo the layout, so nothing overlaps on screen either
    def on_resize(event):
        place_axes(fig, ax, ruler_ax, title, label_inches, left_inches)   # new width first
        new_label_inches = draw_ruler(fig, ruler_ax, ax, xs, texts)
        place_axes(fig, ax, ruler_ax, title, new_label_inches, left_inches)
    fig.canvas.mpl_connect("resize_event", on_resize)

    if save:
        os.makedirs(FIGURES_FOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = os.path.join(FIGURES_FOLDER, f"{gene_name}_{timestamp}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {path}")
    if show:
        plt.show()
    return fig


def main():
    """Genes from the command line; a number or "all" sets how many isoforms to show.
    With no genes given, keep asking for gene names until a blank line is entered."""
    max_isoforms = MAX_ISOFORMS
    genes = []
    for arg in sys.argv[1:]:
        if arg.lower() == "all":
            max_isoforms = None
        elif arg.isdigit():
            max_isoforms = int(arg)
        else:
            genes.append(arg)

    if genes:
        for gene in genes:
            plot_gene(gene, max_isoforms=max_isoforms)
        return

    load_data()   # read the files before the first prompt, so later genes appear quickly
    while True:
        gene = input("Gene name / ID (blank to quit): ").strip()
        if not gene:
            break
        plot_gene(gene, max_isoforms=max_isoforms)


if __name__ == "__main__":
    main()
