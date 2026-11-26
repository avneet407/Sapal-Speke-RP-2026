"""
Build protein sequences for expression-ranked isoforms.

Examples:
    python 5a_protein_sequence/.py
    python 5a_protein_sequence/.py SLC45A1 1 2 3
    python 5a_protein_sequence/.py --gene SLC45A1 --isoforms 1 2 --all-samples

With no gene or isoform arguments, the script prompts for them repeatedly.
Enter a blank gene name or blank isoform-rank list to quit.

Isoforms are numbered exactly as in 1_data_processing/.py/isoform_visualisation.py:
all of the gene's isoforms ranked by mean % share of the gene across the adult
samples (1 = highest share), using the shared functions in
functions/data_loading/expression_data.py. A ranking CSV is only used if one is
given explicitly with --rank-order-csv.

The script then extracts the CDS and translates it into amino-acid sequence
suitable for InterProScan submission.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOLDER))

from functions.data_loading.expression_data import (
    load_matrix, mean_isoform_share, rank_isoforms_by_share, read_isoform_genes,
)

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _parse_gtf_attributes(text):
    return dict(re.findall(r'(\S+)\s+"([^"]*)"', text))


def _read_fasta(path):
    sequences = {}
    sequence_id = None
    chunks = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_id is not None:
                    sequences[sequence_id] = "".join(chunks).upper()
                sequence_id = line[1:].split()[0]
                chunks = []
            elif sequence_id is None:
                raise ValueError(f"FASTA sequence encountered before a header in {path}")
            else:
                chunks.append(line)
    if sequence_id is not None:
        sequences[sequence_id] = "".join(chunks).upper()
    return sequences


def _reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def _translate_dna_to_protein(sequence):
    seq = "".join(str(sequence).split()).upper()
    if len(seq) % 3:
        raise ValueError(f"CDS length is not divisible by 3: {len(seq)} nt")

    protein = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        aa = CODON_TABLE.get(codon)
        if aa is None:
            raise ValueError(f"Unrecognised codon {codon!r} in the sequence")
        protein.append(aa)

    protein = "".join(protein)
    if "*" in protein[:-1]:
        raise ValueError("The CDS contains an internal stop codon")
    if protein.endswith("*"):
        protein = protein[:-1]
    return protein


def _ranked_isoforms_from_previous_csv(gene_name, csv_path):
    """Use the earlier adult-only isoform-ranking CSV when available."""
    if csv_path is None:
        return None
    path = Path(csv_path)
    if not path.is_file():
        return None

    table = pd.read_csv(path, low_memory=False)
    if "gene" not in table.columns or "isoform" not in table.columns:
        return None

    gene_rows = table[
        table["gene"].astype(str).str.casefold() == str(gene_name).casefold()
    ]
    if gene_rows.empty:
        return None

    if "mean_cpm" in gene_rows.columns:
        ranked = gene_rows.sort_values("mean_cpm", ascending=False)["isoform"].tolist()
    elif "mean_pct" in gene_rows.columns:
        ranked = gene_rows.sort_values("mean_pct", ascending=False)["isoform"].tolist()
    else:
        ranked = gene_rows["isoform"].tolist()
    return ranked


def share_ranked_isoforms(genes, matrix_path, gtf_path, *, adult_only=True):
    """{gene: [transcript IDs, isoform 1 first]} for the given gene names (case-insensitive).
    Same numbering as the isoform visualisation: every isoform of the gene, in GTF order,
    ranked by mean % share of the gene across the (adult) samples, 1 = highest share."""
    expression = load_matrix(matrix_path)
    if adult_only:
        expression = expression[[sample for sample in expression.columns if "_Fetal_" not in sample]]
    gene_of_isoform = read_isoform_genes(gtf_path)   # in GTF order, like the visualisation
    mean_share = mean_isoform_share(expression, gene_of_isoform)
    gene_folded = gene_of_isoform.astype(str).str.casefold()
    ranked = {}
    for gene in genes:
        isoforms = list(gene_of_isoform.index[gene_folded == str(gene).casefold()])
        ranked[gene] = rank_isoforms_by_share(isoforms, mean_share)
    return ranked


def _ranked_isoforms_for_gene(gene_name, matrix_path, gtf_path, *, adult_only=True, rank_csv_path=None):
    previous_rank = _ranked_isoforms_from_previous_csv(gene_name, rank_csv_path)
    if previous_rank is not None:
        return previous_rank

    ranked = share_ranked_isoforms([gene_name], matrix_path, gtf_path, adult_only=adult_only)[gene_name]
    if not ranked:
        raise ValueError(f"No isoforms for gene {gene_name!r} were found in {gtf_path}")
    return ranked


def _cds_from_gtf(gene_name, transcript_ids, gtf_path, genome_fasta_path, *, gene_name_attribute="gene_name"):
    transcript_ids = set(transcript_ids)
    cds_parts = defaultdict(list)

    with open(gtf_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Malformed GTF row at line {line_number}")
            seqname, _, feature, start, end, _, strand, _, raw_attributes = fields
            if feature != "CDS":
                continue
            attributes = _parse_gtf_attributes(raw_attributes)
            if (attributes.get(gene_name_attribute) or "").casefold() != str(gene_name).casefold():
                continue
            transcript_id = attributes.get("transcript_id")
            if transcript_id in transcript_ids:
                cds_parts[transcript_id].append((seqname, int(start), int(end), strand))

    genome = _read_fasta(genome_fasta_path)
    cds_by_transcript = {}
    for transcript_id in transcript_ids:
        parts = cds_parts.get(transcript_id)
        if not parts:
            continue
        seqnames = {part[0] for part in parts}
        strands = {part[3] for part in parts}
        if len(seqnames) != 1 or len(strands) != 1:
            raise ValueError(f"Inconsistent CDS chromosome or strand for transcript {transcript_id!r}")
        seqname = parts[0][0]
        if seqname not in genome:
            raise ValueError(f"GTF sequence {seqname!r} is absent from the FASTA")
        reverse = parts[0][3] == "-"
        ordered_parts = sorted(parts, key=lambda part: part[1], reverse=reverse)
        nucleotide_sequence = "".join(
            genome[seqname][start - 1 : end] for _, start, end, _ in ordered_parts
        )
        cds_by_transcript[transcript_id] = _reverse_complement(nucleotide_sequence) if reverse else nucleotide_sequence
    return cds_by_transcript


def get_ranked_isoform_proteins(
    gene_name,
    isoform_numbers,
    *,
    matrix_path,
    gtf_path,
    genome_fasta_path=None,
    cds_fasta_path=None,
    adult_only=True,
    gene_name_attribute="gene_name",
    rank_csv_path=None,
):
    """Return a dict of expression-rank -> (transcript_id, protein_sequence)."""
    ranks = list(isoform_numbers)
    if not gene_name or not ranks:
        raise ValueError("gene_name and at least one isoform number are required")
    if any((not isinstance(rank, int)) or rank < 1 for rank in ranks):
        raise ValueError("isoform numbers must be positive integers")
    if len(set(ranks)) != len(ranks):
        raise ValueError("isoform numbers must not contain duplicates")

    transcript_ids = _ranked_isoforms_for_gene(
        gene_name,
        matrix_path,
        gtf_path,
        adult_only=adult_only,
        rank_csv_path=rank_csv_path,
    )
    if max(ranks) > len(transcript_ids):
        raise ValueError(
            f"Requested rank {max(ranks)} but only {len(transcript_ids)} isoforms were found for {gene_name!r}"
        )

    transcript_by_rank = {rank: transcript_ids[rank - 1] for rank in ranks}

    cds_by_transcript = {}
    if cds_fasta_path and Path(cds_fasta_path).is_file():
        direct = _read_fasta(cds_fasta_path)
        for transcript_id in transcript_by_rank.values():
            if transcript_id in direct:
                cds_by_transcript[transcript_id] = direct[transcript_id]

    missing = [tid for tid in transcript_by_rank.values() if tid not in cds_by_transcript]
    if missing:
        if genome_fasta_path is None:
            raise ValueError(
                "Missing CDS records for requested transcripts and no genome FASTA was supplied. "
                "Pass --genome or provide the project cds_output.fa."
            )
        if not Path(genome_fasta_path).is_file():
            raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta_path}")
        recovered = _cds_from_gtf(
            gene_name,
            missing,
            gtf_path,
            genome_fasta_path,
            gene_name_attribute=gene_name_attribute,
        )
        cds_by_transcript.update(recovered)

    missing_after_recovery = [tid for tid in transcript_by_rank.values() if tid not in cds_by_transcript]
    if missing_after_recovery:
        missing_list = ", ".join(missing_after_recovery[:5])
        raise ValueError(
            f"No CDS sequence could be recovered for {len(missing_after_recovery)} transcript(s), "
            f"including: {missing_list}"
        )

    proteins = {}
    for rank, transcript_id in transcript_by_rank.items():
        cds = cds_by_transcript[transcript_id]
        proteins[rank] = (transcript_id, _translate_dna_to_protein(cds))
    return proteins


def _format_transcript_mapping(gene_name, proteins):
    lines = [f"Transcript IDs for {gene_name}:"]
    lines.extend(
        f"Isoform {rank}: {transcript_id}"
        for rank, (transcript_id, _) in sorted(proteins.items())
    )
    return "\n".join(lines)


def _format_protein_fasta(proteins):
    fasta_lines = []
    for rank, (_, protein) in sorted(proteins.items()):
        fasta_lines.append(f">Isoform {rank}")
        fasta_lines.append(protein)
    return "\n".join(fasta_lines) + "\n"


def _run_interactive(args):
    fasta_outputs = []
    while True:
        try:
            gene_name = input("Gene name (blank to quit): ").strip()
        except EOFError:
            print()
            break
        if not gene_name:
            break

        while True:
            try:
                ranks_text = input(
                    "Isoform ranks (e.g. 1 2 3; blank to quit): "
                ).strip()
            except EOFError:
                print()
                return fasta_outputs
            if not ranks_text:
                return fasta_outputs

            try:
                ranks = [int(value) for value in re.split(r"[,\s]+", ranks_text)]
            except ValueError:
                print("Enter positive integer ranks, separated by spaces or commas.")
                continue
            if any(rank < 1 for rank in ranks):
                print("Isoform ranks must be positive integers.")
                continue
            if len(set(ranks)) != len(ranks):
                print("Do not enter duplicate isoform ranks.")
                continue
            break

        try:
            proteins = get_ranked_isoform_proteins(
                gene_name,
                ranks,
                matrix_path=str(args.matrix),
                gtf_path=str(args.gtf),
                genome_fasta_path=str(args.genome) if args.genome else None,
                cds_fasta_path=str(args.cds) if args.cds else None,
                adult_only=args.adult_only,
                rank_csv_path=args.rank_order_csv,
            )
        except ValueError as error:
            print(f"Could not process {gene_name!r}: {error}")
            continue

        output_text = _format_protein_fasta(proteins)
        print(_format_transcript_mapping(gene_name, proteins))
        print("InterProScan FASTA:")
        print(output_text, end="")
        fasta_outputs.append(output_text)

    return fasta_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Return amino-acid sequences for selected gene isoforms for InterProScan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("gene_name", nargs="?", help="Gene name, for example SLC45A1")
    parser.add_argument("isoform_numbers", nargs="*", type=int, help="Expression ranks to retrieve, for example 1 2 3")
    parser.add_argument("--gene", dest="gene_name_flag", help="Alternative to the positional gene name")
    parser.add_argument("--isoforms", nargs="*", type=int, help="Alternative to the positional isoform numbers")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv",
        help="Expression matrix TSV",
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf",
        help="GTF file",
    )
    parser.add_argument(
        "--genome",
        type=Path,
        default=None,
        help="Genome FASTA used only if the CDS FASTA is unavailable",
    )
    parser.add_argument(
        "--cds",
        type=Path,
        default=ROOT_FOLDER / "Long_Read_Data" / "cds_output.fa",
        help="Project CDS FASTA; used when available",
    )
    parser.add_argument(
        "--rank-order-csv",
        type=Path,
        default=None,
        help="Optional isoform ranking CSV to use instead of the default mean-share ranking",
    )
    parser.add_argument(
        "--all-samples",
        dest="adult_only",
        action="store_false",
        help="Use all samples rather than just adult samples to rank isoforms",
    )
    parser.set_defaults(adult_only=True)
    parser.add_argument("--output", type=Path, help="Optional FASTA output file to save in addition to terminal output")

    args = parser.parse_args(argv)

    gene_name = args.gene_name_flag or args.gene_name
    isoform_numbers = args.isoforms if args.isoforms is not None else args.isoform_numbers

    if not args.matrix.is_file():
        raise FileNotFoundError(f"Expression matrix not found: {args.matrix}")
    if not args.gtf.is_file():
        raise FileNotFoundError(f"GTF not found: {args.gtf}")

    if not gene_name and not isoform_numbers:
        fasta_outputs = _run_interactive(args)
        if args.output and fasta_outputs:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(fasta_outputs), encoding="utf-8")
            print(f"Saved InterProScan FASTA to: {args.output}")
        return 0

    if not gene_name:
        parser.error("A gene name is required, for example: SLC45A1")
    if not isoform_numbers:
        parser.error("At least one isoform rank is required, for example: 1 2 3")

    proteins = get_ranked_isoform_proteins(
        gene_name,
        isoform_numbers,
        matrix_path=str(args.matrix),
        gtf_path=str(args.gtf),
        genome_fasta_path=str(args.genome) if args.genome else None,
        cds_fasta_path=str(args.cds) if args.cds else None,
        adult_only=args.adult_only,
        rank_csv_path=args.rank_order_csv,
    )

    output_text = _format_protein_fasta(proteins)
    print(_format_transcript_mapping(gene_name, proteins))
    print("InterProScan FASTA:")
    print(output_text, end="")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Saved InterProScan FASTA to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
