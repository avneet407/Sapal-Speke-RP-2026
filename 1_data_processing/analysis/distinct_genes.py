"""Find and visualise G-loci assigned to two distinct named genes.

For each example, show the highest-mean-CPM isoform of each named gene, followed
only by other transcripts with exons overlapping both top isoforms' genomic spans.
Rows are labelled by mean CPM across adult samples, matching the isoform viewer.

Run from any directory with:
    python 1_data_processing/analysis/distinct_genes.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_FOLDER))

from functions.graphs.isoform_structure import plot_isoform_structures


GTF_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf"
MATRIX_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv"
FIGURES_FOLDER = ROOT_FOLDER / "1_data_processing" / "Figures"

MAX_EXAMPLES = 3
# Keep the chosen figures readable without omitting any overlapping transcripts.
MAX_PLOTTED_ISOFORMS = 60
SAVE_FIGURES = True


def read_gtf(gtf_path=GTF_PATH):
    """Read transcript metadata and exon/CDS coordinates in the plotter's format."""
    attribute = re.compile(r'(\w+) "([^"]*)"')
    transcripts = defaultdict(lambda: {"exons": [], "cds": []})
    with open(gtf_path, encoding="utf-8") as gtf:
        for line in gtf:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            attributes = dict(attribute.findall(fields[8]))
            transcript_id = attributes.get("transcript_id")
            if not transcript_id:
                continue
            info = transcripts[transcript_id]
            if feature == "transcript":
                info.update(
                    chrom=fields[0],
                    strand=fields[6],
                    start=int(fields[3]),
                    end=int(fields[4]),
                    gene_name=attributes.get("gene_name", ""),
                    gene_id=attributes.get("gene_id", ""),
                )
            elif feature == "exon":
                info["exons"].append((int(fields[3]), int(fields[4])))
            elif feature == "CDS":
                info["cds"].append((int(fields[3]), int(fields[4])))

    for info in transcripts.values():
        info["exons"].sort()
        info["cds"].sort()
    return dict(transcripts)


def read_adult_mean_cpm(matrix_path=MATRIX_PATH):
    """Return each transcript's mean CPM across non-fetal samples."""
    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
    adult_columns = [column for column in matrix.columns if "_Fetal_" not in column]
    if not adult_columns:
        raise ValueError(f"No adult samples found in expression matrix: {matrix_path}")
    return matrix[adult_columns].mean(axis=1)


def find_examples(transcripts, mean_cpm):
    """Find manageable two-gene loci with transcripts crossing both top-isoform spans."""
    loci = defaultdict(list)
    for transcript_id, info in transcripts.items():
        if transcript_id not in mean_cpm.index or not info.get("exons"):
            continue
        locus = transcript_id.split(".")[0].split("_")[0]
        loci[locus].append(transcript_id)

    examples = []
    for locus, transcript_ids in loci.items():
        gene_names = {
            transcripts[transcript_id].get("gene_name", "")
            for transcript_id in transcript_ids
        } - {""}
        if len(gene_names) != 2 or any(name.startswith("ENSG") for name in gene_names):
            continue

        chrom_strand = {
            (transcripts[transcript_id]["chrom"], transcripts[transcript_id]["strand"])
            for transcript_id in transcript_ids
        }
        if len(chrom_strand) != 1:
            continue

        top_isoforms = {
            name: max(
                (transcript_id for transcript_id in transcript_ids
                 if transcripts[transcript_id].get("gene_name") == name),
                key=lambda transcript_id: mean_cpm.get(transcript_id, 0),
            )
            for name in gene_names
        }
        top_infos = [transcripts[top_isoforms[name]] for name in sorted(gene_names)]
        spanning = [
            transcript_id for transcript_id in transcript_ids
            if transcript_id not in top_isoforms.values()
            and all(
                any(
                    exon_start <= top_info["end"] and top_info["start"] <= exon_end
                    for exon_start, exon_end in transcripts[transcript_id]["exons"]
                )
                for top_info in top_infos
            )
        ]
        if not spanning:
            continue
        if len(spanning) + len(top_isoforms) > MAX_PLOTTED_ISOFORMS:
            continue

        # The two representative isoforms lead the plot; cross-domain transcripts
        # follow in descending adult mean CPM.
        ordered_isoforms = [top_isoforms[name] for name in sorted(gene_names)]
        ordered_isoforms.extend(
            sorted(spanning, key=lambda transcript_id: -mean_cpm.get(transcript_id, 0))
        )
        examples.append({
            "locus": locus,
            "gene_names": sorted(gene_names),
            "top_isoforms": top_isoforms,
            "spanning": spanning,
            "plot_isoforms": ordered_isoforms,
            "mean_cpm": sum(mean_cpm.get(transcript_id, 0)
                            for transcript_id in top_isoforms.values()),
        })

    return sorted(examples, key=lambda example: example["mean_cpm"], reverse=True)


def main():
    print("Reading GTF and expression matrix...")
    transcripts = read_gtf()
    mean_cpm = read_adult_mean_cpm()
    examples = find_examples(transcripts, mean_cpm)
    selected_examples = examples[:MAX_EXAMPLES]
    print(
        f"Found {len(examples)} eligible two-gene loci with at most "
        f"{MAX_PLOTTED_ISOFORMS} plotted transcripts (top isoforms and "
        "exon-overlapping transcripts); "
        f"visualising {len(selected_examples)}."
    )

    for example in selected_examples:
        locus = example["locus"]
        names = example["gene_names"]
        print(
            f"{locus}: {' + '.join(names)}; "
            f"{len(example['spanning'])} additional transcripts with exons in both domains"
        )
        for name in names:
            transcript_id = example["top_isoforms"][name]
            print(f"  top {name}: {transcript_id} ({mean_cpm.get(transcript_id, 0):.1f} CPM)")
        title = (
            f"{locus}: {' + '.join(names)} - top isoform per gene, then "
            "transcripts with exons in both domains\n"
            f"Mean adult CPM; grey = exon (UTR), red = CDS, line = intron"
        )
        figure = plot_isoform_structures(
            locus,
            transcripts,
            mean_cpm,
            FIGURES_FOLDER,
            max_isoforms=None,
            save=SAVE_FIGURES,
            show=False,
            isoform_ids=example["plot_isoforms"],
            plot_title=title,
            include_gene_names=True,
            filename_prefix=locus,
            preserve_isoform_order=True,
        )
        if figure is not None:
            plt.close(figure)


if __name__ == "__main__":
    main()