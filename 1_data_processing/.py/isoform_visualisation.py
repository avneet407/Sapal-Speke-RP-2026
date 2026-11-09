"""
Isoform structure viewer: draw every isoform of a gene, one row per isoform, to compare their structure.

  - exons  = blocks (grey = untranslated / UTR part, red = coding / CDS part)
  - introns = thin black lines joining the exons
  - a coordinate ruler above the isoforms labels every distinct exon start / end once, with a
    dotted guide line down through every row (exon edges on the same line = shared splice site)
  - each isoform has 5' and 3' written at its ends (which side is 5' depends on the strand)
  - isoforms are ordered by mean CPM (most abundant at the top), with the CPM in the row label

Structure comes from Long_Read_Data/expressed_isoforms.gtf ("exon" and "CDS" rows),
abundance from Long_Read_Data/expressed_isoforms_matrix.tsv (same 61,117 transcript IDs),
using the 45 adult samples only (the 4 "_Fetal_" columns are dropped straight after loading).
Non-coding isoforms (~15.6k, no CDS rows) are drawn all grey.

A gene can be given as:
    gene name        SLC45A1
    Ensembl gene ID  ENSG00000162426 (version .16 optional)

Usage:
    python 1_data_processing/.py/isoform_visualisation.py SLC45A1          # one gene, top MAX_ISOFORMS isoforms
    python 1_data_processing/.py/isoform_visualisation.py SLC45A1 all      # every isoform of the gene
    python 1_data_processing/.py/isoform_visualisation.py SLC45A1 40       # top 40 isoforms
    python 1_data_processing/.py/isoform_visualisation.py SLC45A1 GABBR2   # several genes, one figure each
    python 1_data_processing/.py/isoform_visualisation.py                  # asks for gene names one at a time
The drawing itself (and its settings: intron compression, colours, sizes) is in
functions/graphs/isoform_structure.py.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.graphs.isoform_structure import plot_isoform_structures


# --- Settings (change these, not the code below) ---------------------------
GTF_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf")
MATRIX_PATH = str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv")
FIGURES_FOLDER = str(ROOT_FOLDER / "1_data_processing" / "Figures")

MAX_ISOFORMS = 20        # show only the N most abundant isoforms; None = show all
                         # (some genes have ~100 isoforms, which makes a very tall figure)
SAVE_FIGURE = True          # True = also save a PNG in FIGURES_FOLDER (timestamped, never overwritten)


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
    """Mean CPM of each isoform across the 45 adult samples, as a pandas Series indexed by transcript_id.
    Only used to order the isoforms and label them, so if the matrix is missing return None."""
    if not os.path.exists(matrix_path):
        print(f"Note: {matrix_path} not found - isoforms will be ordered by position, not abundance")
        return None
    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
    # Drop the 4 fetal samples straight after loading, as in gene_spearman.py
    matrix = matrix[[name for name in matrix.columns if "_Fetal_" not in name]]
    return matrix.mean(axis=1)


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

    # Read the files once, so plotting several genes doesn't re-read the 110 MB GTF
    print("Reading GTF and expression matrix...")
    transcripts = read_gtf()
    mean_cpm = read_mean_cpm()

    if genes:
        for gene in genes:
            plot_isoform_structures(gene, transcripts, mean_cpm, FIGURES_FOLDER, max_isoforms, SAVE_FIGURE)
        return

    while True:
        gene = input("Gene name / ID (blank to quit): ").strip()
        if not gene:
            break
        plot_isoform_structures(gene, transcripts, mean_cpm, FIGURES_FOLDER, max_isoforms, SAVE_FIGURE)


if __name__ == "__main__":
    main()
