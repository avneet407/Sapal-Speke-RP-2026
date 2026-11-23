"""
Build an isoform-by-exon inclusion matrix for genes in the filtered interesting-pairs table.

The default input is the latest significant isoform-gene-pairs CSV from
4_identifying_candidates/CSV. Genes are taken from the union of its gene_a and gene_b columns.
Set INPUT_FILE to use a different filtered pairs table with those columns.

Isoforms are retained using the same rule as isoform_gene_correlation.py: among adult samples,
calculate each isoform's share of its gene CPM per sample, skip samples where the gene has zero
CPM, and keep isoforms whose mean share is at least 1.5%.

For each gene, save a CSV with one row per retained isoform and one binary column per exact
GTF exon interval (chromosome, start, end, strand). With CODING_ONLY enabled, include an exon
column if that interval overlaps a CDS feature for at least one retained isoform of that gene.

Run with:
    python 5a_identifying_exons/.py/exon_inclusion_matrix.py
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_FOLDER))

from functions.data_loading.expression_data import load_matrix, read_isoform_genes


# --- Settings ---------------------------------------------------------------
INPUT_FILE = None
MIN_SHARE = 0.015
ADULTS_ONLY = True
CODING_ONLY = False

INPUT_FOLDER = ROOT_FOLDER / "4_identifying_candidates" / "CSV"
MATRIX_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv"
GTF_PATH = ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf"
RESULTS_FOLDER = ROOT_FOLDER / "5a_identifying_exons" / "CSV"

TRANSCRIPT_ID_PATTERN = re.compile(r'transcript_id "([^"]+)"')
Exon = tuple[str, int, int, str]


def find_input_file():
    """Return a configured input or the latest significant-pairs CSV."""
    if INPUT_FILE is not None:
        path = Path(INPUT_FILE)
        if not path.is_absolute():
            path = ROOT_FOLDER / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured interesting-pairs CSV does not exist: {path}")
        return path

    candidates = list(INPUT_FOLDER.glob("significant_isoform_gene_correlations_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No significant-pairs CSV found in "
            f"{INPUT_FOLDER}. Run the interesting-pair filter first or set INPUT_FILE."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def read_interesting_genes(path):
    """Read the unique gene names from the gene_a/gene_b columns."""
    pairs = pd.read_csv(path, usecols=["gene_a", "gene_b"], low_memory=False)
    genes = pd.unique(pairs[["gene_a", "gene_b"]].to_numpy().ravel())
    genes = sorted(str(gene) for gene in genes if pd.notna(gene) and str(gene).strip())
    if not genes:
        raise ValueError(f"No gene names found in gene_a/gene_b columns of {path}.")
    return genes


def read_exon_annotations(gtf_path, transcript_ids):
    """Read exact exon and CDS intervals for the requested transcripts."""
    exons_by_isoform = {}
    cds_by_isoform = {}
    transcript_ids = set(transcript_ids)

    with open(gtf_path, encoding="utf-8") as gtf:
        for line_number, line in enumerate(gtf, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in {"exon", "CDS"}:
                continue

            match = TRANSCRIPT_ID_PATTERN.search(fields[8])
            if match is None:
                raise ValueError(
                    f"Missing transcript_id on GTF line {line_number} ({fields[2]} feature)."
                )
            isoform = match.group(1)
            if isoform not in transcript_ids:
                continue

            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError as error:
                raise ValueError(f"Invalid coordinates on GTF line {line_number}.") from error
            if start > end:
                raise ValueError(f"Start is after end on GTF line {line_number}.")

            interval = (fields[0], start, end, fields[6])
            target = exons_by_isoform if fields[2] == "exon" else cds_by_isoform
            target.setdefault(isoform, set()).add(interval)

    missing_exons = transcript_ids.difference(exons_by_isoform)
    if missing_exons:
        examples = ", ".join(sorted(missing_exons)[:5])
        raise ValueError(
            f"No exon annotations found for {len(missing_exons)} retained isoform(s), "
            f"including: {examples}"
        )
    return exons_by_isoform, cds_by_isoform


def exon_overlaps_cds(exon, cds_intervals):
    """Return whether an exon interval overlaps any CDS interval on the same sequence/strand."""
    chrom, start, end, strand = exon
    return any(
        chrom == cds_chrom
        and strand == cds_strand
        and start <= cds_end
        and cds_start <= end
        for cds_chrom, cds_start, cds_end, cds_strand in cds_intervals
    )


def exon_column_name(exon):
    """Create a coordinate-labelled CSV column for a GTF exon interval."""
    chrom, start, end, strand = exon
    return f"exon_{chrom}:{start}-{end}:{strand}"


def safe_gene_filename(gene):
    """URL-encode gene names so they can be used safely as Windows filenames."""
    return f"exon_matrix_{quote(gene, safe='._-')}.csv"


def build_gene_matrix(gene, isoforms, exons_by_isoform, cds_by_isoform, coding_only=False):
    """Build one binary row-per-isoform, column-per-exon matrix for a gene."""
    exon_features = {
        exon
        for isoform in isoforms.index
        for exon in exons_by_isoform[isoform]
        if not coding_only
        or exon_overlaps_cds(exon, cds_by_isoform.get(isoform, set()))
    }
    ordered_exons = sorted(exon_features)
    if not ordered_exons:
        raise ValueError(
            f"No {'coding ' if coding_only else ''}exon features found for gene {gene}."
        )

    columns = {
        "gene": [gene] * len(isoforms),
        "isoform_id": isoforms.index.to_list(),
        "mean_share_pct": (isoforms["mean_share"] * 100).to_list(),
    }
    for exon in ordered_exons:
        columns[exon_column_name(exon)] = [
            int(exon in exons_by_isoform[isoform])
            for isoform in isoforms.index
        ]
    return pd.DataFrame(columns)


def main():
    if not 0 <= MIN_SHARE <= 1:
        raise ValueError("MIN_SHARE must be between 0 and 1.")

    input_path = find_input_file()
    interesting_genes = read_interesting_genes(input_path)
    print(f"Reading filtered gene pairs: {input_path}")
    print(f"Genes in the union of gene_a/gene_b: {len(interesting_genes):,}")

    expression = load_matrix(MATRIX_PATH)
    if ADULTS_ONLY:
        adult_samples = [sample for sample in expression.columns if "_Fetal_" not in sample]
        if not adult_samples:
            raise ValueError("No adult samples were found in the expression matrix.")
        expression = expression[adult_samples]
    print(f"Using {expression.shape[1]} {'adult ' if ADULTS_ONLY else ''}samples.")

    gene_of_isoform = read_isoform_genes(GTF_PATH).reindex(expression.index)
    if gene_of_isoform.isna().any():
        missing = expression.index[gene_of_isoform.isna()]
        raise ValueError(
            f"No gene annotation found for {len(missing)} expression isoform(s), "
            f"including: {', '.join(missing[:5])}"
        )

    selected = gene_of_isoform.isin(interesting_genes)
    expression = expression.loc[selected]
    gene_of_isoform = gene_of_isoform.loc[selected]
    if expression.empty:
        raise ValueError("None of the input gene names matched genes in the GTF.")

    own_gene_cpm = expression.groupby(gene_of_isoform, sort=False).transform("sum")
    mean_share = (expression / own_gene_cpm.where(own_gene_cpm > 0)).mean(axis=1)
    retained = mean_share >= MIN_SHARE
    expression = expression.loc[retained]
    gene_of_isoform = gene_of_isoform.loc[retained]
    mean_share = mean_share.loc[retained]
    print(
        f"Isoforms retained at mean share >= {MIN_SHARE:.1%}: "
        f"{len(expression):,} of {int(selected.sum()):,}"
    )

    if expression.empty:
        raise ValueError(
            f"No isoforms met the {MIN_SHARE:.1%} mean-share threshold in the selected samples."
        )

    exons_by_isoform, cds_by_isoform = read_exon_annotations(GTF_PATH, expression.index)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    mode = "coding_exons" if CODING_ONLY else "all_exons"
    age = "adults" if ADULTS_ONLY else "all_samples"
    output_folder = RESULTS_FOLDER / f"isoform_exon_matrices_{age}_share{MIN_SHARE}_{mode}_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    skipped_genes = 0
    for gene in interesting_genes:
        isoforms = expression.loc[gene_of_isoform == gene]
        if isoforms.empty:
            skipped_genes += 1
            print(f"Skipping {gene}: no isoforms met the share threshold.")
            continue

        gene_table = pd.DataFrame({
            "mean_share": mean_share.loc[isoforms.index]
        }).sort_values("mean_share", ascending=False)
        gene_table = build_gene_matrix(
            gene,
            gene_table,
            exons_by_isoform,
            cds_by_isoform,
            coding_only=CODING_ONLY,
        )
        output_path = output_folder / safe_gene_filename(gene)
        gene_table.to_csv(output_path, index=False)
        exon_count = len(gene_table.columns) - 3
        manifest_rows.append({
            "gene": gene,
            "isoform_count": len(gene_table),
            "exon_count": exon_count,
            "file": output_path.name,
        })
        print(
            f"Saved {gene}: {len(gene_table):,} isoforms x {exon_count:,} exon features."
        )

    manifest_path = output_folder / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(
        f"\nSaved matrices for {len(manifest_rows):,} genes "
        f"({skipped_genes:,} genes had no retained isoforms)."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
