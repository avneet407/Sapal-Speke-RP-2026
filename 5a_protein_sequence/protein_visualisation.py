"""
Visualise completed InterProScan proteins and annotations along amino-acid coordinates.

Completed scan mappings come from the InterPro job database, so proteins with
no annotation hits are still shown. Annotation CSV exports are used when
available; otherwise annotations are read from the database.

Usage:
    python 5a_protein_sequence/protein_visualisation.py IRF3
    python 5a_protein_sequence/protein_visualisation.py IRF3 all
    python 5a_protein_sequence/protein_visualisation.py IRF3 40
    python 5a_protein_sequence/protein_visualisation.py
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.textpath import TextPath


def _find_project_root(script_path: Path) -> Path:
    for candidate in script_path.resolve().parents:
        if (
            (candidate / "Long_Read_Data").is_dir()
            and (candidate / "functions" / "graphs" / "isoform_structure.py").is_file()
        ):
            return candidate
    raise RuntimeError(f"Could not locate project root from {script_path}")


ROOT_FOLDER = _find_project_root(Path(__file__))
for import_folder in (
    ROOT_FOLDER,
    ROOT_FOLDER / "1_data_processing" / ".py",
    ROOT_FOLDER / "5a_protein_sequence",
):
    if str(import_folder) not in sys.path:
        sys.path.insert(0, str(import_folder))

from functions.graphs.isoform_structure import (
    FIGURE_WIDTH_INCHES,
    ROW_HEIGHT_INCHES,
)
from isoform_visualisation import find_isoforms, read_gtf
from submitting_to_InterPro import (
    DEFAULT_DB_PATH,
    load_annotations,
    load_completed_isoforms,
)

GTF_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf"
CSV_FOLDER = ROOT_FOLDER / "5a_protein_sequence" / "CSV"
FIGURES_FOLDER = ROOT_FOLDER / "5a_protein_sequence" / "Figures"

MAX_PROTEINS = 20
SAVE_FIGURE = True
PROTEIN_BAR_HEIGHT = 0.22
DOMAIN_BAR_HEIGHT = 0.12
DOMAIN_LANE_STEP = 0.45
DOMAIN_LABEL_FONT_SIZE = 5
DOMAIN_LABEL_GAP = 0.08
PROTEIN_ROW_GAP = 0.28
UTR_MARKER_FRACTION = 0.015


def _safe_gene_token(gene_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", gene_name).strip("._-")
    return token or "gene"


def _find_annotation_csv(
    gene_name: str,
    entered_gene: str,
    explicit_path: Path | None,
) -> Path | None:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {explicit_path}")
        return explicit_path

    candidates = []
    for candidate_gene in (gene_name, entered_gene):
        candidate = CSV_FOLDER / (
            f"interproscan_annotations_{_safe_gene_token(candidate_gene)}.csv"
        )
        if candidate not in candidates:
            candidates.append(candidate)
    candidates.append(CSV_FOLDER / "interproscan_annotations_all_genes.csv")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def _normalise_protein(sequence: Any) -> str:
    if pd.isna(sequence):
        return ""
    return "".join(str(sequence).split()).upper()


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _feature_rows_by_protein(
    annotations: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    features_by_protein: dict[str, list[dict[str, Any]]] = {}
    seen_by_protein: dict[str, set[tuple[Any, ...]]] = {}

    for _, row in annotations.iterrows():
        protein = _normalise_protein(row.get("protein_sequence"))
        start = _optional_int(row.get("start"))
        end = _optional_int(row.get("end"))
        if not protein or start is None or end is None or start < 1 or end < start:
            continue

        accession = row.get("interpro_accession")
        if pd.isna(accession) or not str(accession).strip():
            accession = row.get("signature_accession")
        accession_text = "" if pd.isna(accession) else str(accession).strip()

        description = row.get("interpro_description")
        if pd.isna(description) or not str(description).strip():
            description = row.get("signature_description")
        description_text = "" if pd.isna(description) else str(description).strip()

        feature = {
            "start": start,
            "end": end,
            "accession": accession_text or "InterPro match",
            "description": description_text,
        }
        key = (start, end, feature["accession"], description_text)
        seen = seen_by_protein.setdefault(protein, set())
        if key in seen:
            continue
        seen.add(key)
        features_by_protein.setdefault(protein, []).append(feature)

    for features in features_by_protein.values():
        features.sort(key=lambda feature: (feature["start"], feature["end"], feature["accession"]))
    return features_by_protein


def _build_protein_groups(
    scan_mappings: pd.DataFrame,
    transcripts: dict[str, dict[str, Any]],
    max_proteins: int | None,
    features_by_protein: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    proteins: dict[str, dict[str, Any]] = {}
    rank_by_transcript: dict[str, int] = {}
    missing_structures = []

    for row in scan_mappings.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        if transcript_id not in transcripts or not transcripts[transcript_id]["cds"]:
            missing_structures.append(transcript_id)
            continue

        protein = _normalise_protein(row.protein_sequence)
        if not protein:
            raise ValueError(
                f"Finished InterProScan mapping for {transcript_id!r} has no protein sequence"
            )
        rank = int(row.isoform_rank)
        rank_by_transcript[transcript_id] = min(
            rank, rank_by_transcript.get(transcript_id, rank)
        )
        group = proteins.setdefault(
            protein,
            {
                "sequence": protein,
                "transcripts": [],
                "first_rank": rank,
                "features": features_by_protein.get(protein, []),
            },
        )
        if transcript_id not in group["transcripts"]:
            group["transcripts"].append(transcript_id)
        group["first_rank"] = min(group["first_rank"], rank)

    if missing_structures:
        examples = ", ".join(dict.fromkeys(missing_structures[:5]))
        print(
            f"Note: {len(missing_structures)} completed scan mapping(s) have no "
            f"coding structure in the current GTF and will be omitted; examples: {examples}"
        )

    groups = list(proteins.values())
    for group in groups:
        group["transcripts"].sort(
            key=lambda transcript_id: (
                rank_by_transcript[transcript_id],
                transcript_id,
            )
        )
    groups.sort(key=lambda group: group["first_rank"])
    if max_proteins is not None:
        groups = groups[:max_proteins]
    return groups, rank_by_transcript


def _terminal_utr_lengths(info: dict[str, Any]) -> tuple[int, int]:
    """Return UTR lengths only within the exons containing the first/last CDS bases."""
    strand = info["strand"]
    cds = sorted(info["cds"], reverse=strand == "-")
    if not cds:
        return 0, 0

    exons = sorted(info["exons"], reverse=strand == "-")
    first_cds_start, first_cds_end = cds[0]
    last_cds_start, last_cds_end = cds[-1]
    five_prime_length = 0
    three_prime_length = 0

    for exon_start, exon_end in exons:
        if exon_start <= first_cds_start <= exon_end and exon_start <= first_cds_end <= exon_end:
            if strand == "-":
                five_prime_length = exon_end - first_cds_end
            else:
                five_prime_length = first_cds_start - exon_start
            break

    for exon_start, exon_end in exons:
        if exon_start <= last_cds_start <= exon_end and exon_start <= last_cds_end <= exon_end:
            if strand == "-":
                three_prime_length = last_cds_start - exon_start
            else:
                three_prime_length = exon_end - last_cds_end
            break

    return max(five_prime_length, 0), max(three_prime_length, 0)


def _annotation_label(feature: dict[str, Any]) -> str:
    return " ".join(feature["description"].split()) or feature["accession"]


def _assign_annotation_lanes(
    features: list[dict[str, Any]],
    amino_acids_per_inch: float,
) -> tuple[list[dict[str, Any]], int]:
    """Assign the minimum number of lanes needed for intervals and their labels."""
    positioned = []
    for feature in features:
        label = _annotation_label(feature)
        label_width = (
            TextPath((0, 0), label, size=DOMAIN_LABEL_FONT_SIZE)
            .get_extents()
            .width
            / 72
            * amino_acids_per_inch
        )
        start = feature["start"] - 1
        end = feature["end"]
        midpoint = (start + end) / 2
        positioned.append(
            {
                **feature,
                "label": label,
                "lane_start": min(start, midpoint - label_width / 2),
                "lane_end": max(end, midpoint + label_width / 2),
            }
        )

    positioned.sort(key=lambda feature: (feature["lane_start"], feature["lane_end"]))
    lane_ends: list[float] = []
    for feature in positioned:
        for lane, lane_end in enumerate(lane_ends):
            if lane_end < feature["lane_start"]:
                feature["lane"] = lane
                lane_ends[lane] = feature["lane_end"]
                break
        else:
            feature["lane"] = len(lane_ends)
            lane_ends.append(feature["lane_end"])
    return positioned, len(lane_ends)


def _draw_protein_plot(
    gene_name: str,
    gene_id: str,
    groups: list[dict[str, Any]],
    transcripts: dict[str, dict[str, Any]],
    rank_by_transcript: dict[str, int],
    show: bool = True,
):
    track_transcripts = list(
        dict.fromkeys(
            transcript_id
            for group in groups
            for transcript_id in group["transcripts"]
        )
    )
    labels = []
    for group in groups:
        members = ", ".join(
            f"{rank_by_transcript[transcript_id]} ({transcript_id})"
            for transcript_id in group["transcripts"]
        )
        labels.append(textwrap.fill(f"Isoforms: {members}", width=48))
    label_font_size = 6
    longest_line = max(
        (
            line
            for label in labels
            for line in label.splitlines()
        ),
        key=lambda line: TextPath((0, 0), line, size=label_font_size).get_extents().width,
    )
    left_inches = (
        TextPath((0, 0), longest_line, size=label_font_size).get_extents().width / 72
        + 0.45
    )

    max_protein_length = max(len(group["sequence"]) for group in groups)
    utr_marker_width = max(1.0, max_protein_length * UTR_MARKER_FRACTION)
    plot_width_inches = max(FIGURE_WIDTH_INCHES - left_inches - 0.6, 1.0)
    max_label_width = max(
        (
            TextPath(
                (0, 0),
                _annotation_label(feature),
                size=DOMAIN_LABEL_FONT_SIZE,
            ).get_extents().width
            for group in groups
            for feature in group["features"]
        ),
        default=0.0,
    )
    initial_units_per_inch = (
        max_protein_length + 2 * utr_marker_width
    ) / plot_width_inches
    plot_padding = max(
        2.0,
        max_label_width / 72 * initial_units_per_inch / 2 + 1,
    )
    x_min = -utr_marker_width - plot_padding
    x_max = max_protein_length + utr_marker_width + plot_padding
    amino_acids_per_inch = (x_max - x_min) / plot_width_inches

    total_y_units = 0.0
    for group in groups:
        group["features"], group["lane_count"] = _assign_annotation_lanes(
            group["features"], amino_acids_per_inch
        )
        group["protein_y"] = (
            total_y_units
            + group["lane_count"] * DOMAIN_LANE_STEP
            + DOMAIN_LABEL_GAP
            + PROTEIN_BAR_HEIGHT / 2
        )
        total_y_units = (
            group["protein_y"]
            + PROTEIN_BAR_HEIGHT / 2
            + PROTEIN_ROW_GAP
        )

    figure_height = max(3.2, 1.4 + ROW_HEIGHT_INCHES * total_y_units)
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    ax = fig.add_axes(
        [
            left_inches / FIGURE_WIDTH_INCHES,
            0.55 / figure_height,
            plot_width_inches / FIGURE_WIDTH_INCHES,
            (figure_height - 1.2) / figure_height,
        ]
    )
    half_bar_height = PROTEIN_BAR_HEIGHT / 2
    domain_keys = sorted(
        {
            (feature["accession"], feature["description"])
            for group in groups
            for feature in group["features"]
        }
    )
    palette = plt.get_cmap("tab20")
    domain_colors = {
        key: palette(index % palette.N) for index, key in enumerate(domain_keys)
    }

    for group in groups:
        protein_y = group["protein_y"]
        protein_length = len(group["sequence"])
        representative = group["transcripts"][0]
        five_prime_utr, three_prime_utr = _terminal_utr_lengths(
            transcripts[representative]
        )
        ax.add_patch(
            Rectangle(
                (0, protein_y - half_bar_height),
                protein_length,
                PROTEIN_BAR_HEIGHT,
                facecolor="#dceaf5",
                edgecolor="#333333",
                linewidth=0.5,
                zorder=2,
            )
        )
        if five_prime_utr:
            ax.add_patch(
                Rectangle(
                    (-utr_marker_width, protein_y - half_bar_height),
                    utr_marker_width,
                    PROTEIN_BAR_HEIGHT,
                    facecolor="#d3d3d3",
                    edgecolor="#555555",
                    hatch="///",
                    linewidth=0.4,
                    zorder=2,
                )
            )
        if three_prime_utr:
            ax.add_patch(
                Rectangle(
                    (protein_length, protein_y - half_bar_height),
                    utr_marker_width,
                    PROTEIN_BAR_HEIGHT,
                    facecolor="#d3d3d3",
                    edgecolor="#555555",
                    hatch="///",
                    linewidth=0.4,
                    zorder=2,
                )
            )
        for feature in group["features"]:
            color_key = (feature["accession"], feature["description"])
            color = domain_colors[color_key]
            domain_y = (
                protein_y
                - half_bar_height
                - DOMAIN_LABEL_GAP
                - (feature["lane"] + 0.5) * DOMAIN_LANE_STEP
            )
            domain_start = feature["start"] - 1
            domain_end = feature["end"]
            ax.add_patch(
                Rectangle(
                    (domain_start, domain_y - DOMAIN_BAR_HEIGHT / 2),
                    domain_end - domain_start,
                    DOMAIN_BAR_HEIGHT,
                    facecolor=color,
                    edgecolor="#222222",
                    linewidth=0.4,
                    zorder=3,
                )
            )
            ax.text(
                (domain_start + domain_end) / 2,
                domain_y - DOMAIN_BAR_HEIGHT / 2 - 0.01,
                feature["label"],
                ha="center",
                va="bottom",
                fontsize=DOMAIN_LABEL_FONT_SIZE,
                clip_on=False,
                zorder=4,
            )

    max_length = max_protein_length
    tick_positions = sorted({round(max_length * fraction / 4) for fraction in range(5)})
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{position:,}" for position in tick_positions], fontsize=7)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(total_y_units + 0.1, -0.1)
    ax.set_yticks([group["protein_y"] for group in groups])
    ax.set_yticklabels(labels, fontsize=label_font_size)
    ax.set_xlabel("Amino-acid position (CDS to scale)")
    ax.grid(axis="x", color="#e5e5e5", linestyle=":", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    title = fig.suptitle(
        f"{gene_name} ({gene_id}) - {len(track_transcripts)} scanned isoforms, "
        f"{len(groups)} unique proteins\nFINISHED scans only; zero-hit sequences are included",
        va="top",
        fontsize=11,
    )
    title.set_y(0.98)

    legend_handles = [
        Patch(facecolor="#dceaf5", edgecolor="#333333", label="Protein sequence"),
        Patch(
            facecolor="#d3d3d3",
            edgecolor="#555555",
            hatch="///",
            label="5' UTR fragment (left marker)",
        ),
        Patch(
            facecolor="#d3d3d3",
            edgecolor="#555555",
            hatch="///",
            label="3' UTR fragment (right marker)",
        ),
        Patch(facecolor="#777777", edgecolor="#222222", label="Annotation (label above)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=6,
        frameon=False,
    )

    if SAVE_FIGURE:
        FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
        output = FIGURES_FOLDER / f"{_safe_gene_token(gene_name)}_proteins.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output}")
    if show:
        plt.show()
    return fig


def plot_gene(
    gene: str,
    transcripts: dict[str, dict[str, Any]],
    max_proteins: int | None,
    annotation_csv: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    *,
    show: bool = True,
):
    matching_transcripts = find_isoforms(gene, transcripts)
    if not matching_transcripts:
        print(f"No isoforms found for '{gene}' - check the gene name / ID")
        return None

    gene_name = transcripts[matching_transcripts[0]]["gene_name"] or gene
    gene_id = transcripts[matching_transcripts[0]]["gene_id"]
    scan_mappings = load_completed_isoforms(db_path, gene_name=gene_name)
    if scan_mappings.empty:
        print(f"No FINISHED InterProScan jobs were found for '{gene_name}'.")
        return None

    csv_path = _find_annotation_csv(gene_name, gene, annotation_csv)
    if csv_path is None:
        annotations = load_annotations(db_path, gene_name=gene_name)
    else:
        annotations = pd.read_csv(csv_path, low_memory=False)
    required_columns = {"transcript_id", "protein_sequence", "start", "end"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(
            f"Annotation data is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    mapped_transcripts = set(scan_mappings["transcript_id"].astype(str))
    annotations = annotations[
        annotations["transcript_id"].astype(str).isin(mapped_transcripts)
    ]
    features_by_protein = _feature_rows_by_protein(annotations)

    scan_mappings = scan_mappings[
        scan_mappings["transcript_id"].astype(str).isin(set(transcripts))
    ]
    groups, rank_by_transcript = _build_protein_groups(
        scan_mappings,
        transcripts,
        max_proteins,
        features_by_protein,
    )
    if not groups:
        print(f"No FINISHED protein scans with current coding structures for '{gene_name}'.")
        return None

    return _draw_protein_plot(
        gene_name, gene_id, groups, transcripts, rank_by_transcript, show=show
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualise unique proteins and InterPro annotations for a gene."
    )
    parser.add_argument(
        "items",
        nargs="*",
        help="Gene names/IDs; use 'all' or a number to set the protein limit.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional annotation CSV override; by default annotations are loaded from the InterPro database.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="InterPro job database (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    max_proteins = MAX_PROTEINS
    genes = []
    for item in args.items:
        if item.casefold() == "all":
            max_proteins = None
        elif item.isdigit():
            max_proteins = int(item)
        else:
            genes.append(item)

    print("Reading GTF and completed InterProScan mappings...")
    transcripts = read_gtf(GTF_PATH)

    if genes:
        for gene in genes:
            plot_gene(
                gene,
                transcripts,
                max_proteins,
                args.csv,
                args.db,
            )
        return 0

    while True:
        gene = input("Gene name / ID (blank to quit): ").strip()
        if not gene:
            break
        plot_gene(
            gene,
            transcripts,
            max_proteins,
            args.csv,
            args.db,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
