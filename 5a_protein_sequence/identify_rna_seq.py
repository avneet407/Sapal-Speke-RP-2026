"""
Build mature RNA sequences (spliced exons, 5' to 3') for expression-ranked isoforms.

The RNA counterpart of identify_protein_seq.py, with the same inputs and output layout.

Examples:
    python 5a_protein_sequence/identify_rna_seq.py
    python 5a_protein_sequence/identify_rna_seq.py SLC45A1 1 2 3
    python 5a_protein_sequence/identify_rna_seq.py --gene SLC45A1 --isoforms 1 2 --all-samples
    python 5a_protein_sequence/identify_rna_seq.py SLC45A1 1 --dna --output rna.fa

With no gene or isoform arguments, the script prompts for them repeatedly.
Enter a blank gene name or blank isoform-rank list to quit.

Isoforms are numbered exactly as in identify_protein_seq.py (and the isoform
visualisation): all of the gene's isoforms ranked by mean % share of the gene across
the adult samples, 1 = highest share. Unlike the protein script, non-coding isoforms
work too, since every isoform has exons.

The project has no transcript or genome FASTA, so each isoform's exons (from the GTF)
are fetched from the Ensembl REST API (GRCh38) and joined; minus-strand isoforms are
reverse-complemented. Pass --genome to use a local GRCh38 FASTA instead (no network).
Where the isoform has a CDS in cds_output.fa, the script checks that the CDS lies
inside the assembled RNA, which catches a wrong genome build or assembly error.

Output is in the RNA alphabet (U); use --dna for T.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from identify_protein_seq import (  # noqa: E402
    _parse_gtf_attributes,
    _ranked_isoforms_for_gene,
    _read_fasta,
    _reverse_complement,
)

ENSEMBL_REGION_URL = "https://rest.ensembl.org/sequence/region/human"
ENSEMBL_MAX_REGIONS = 50  # the API's limit per POST request
ENSEMBL_TIMEOUT_SECONDS = 60
ENSEMBL_RETRIES = 3


def _exons_from_gtf(transcript_ids, gtf_path):
    """{transcript_id: [(seqname, start, end, strand), ...]} for the given transcripts."""
    transcript_ids = set(transcript_ids)
    exons = defaultdict(list)
    with open(gtf_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Malformed GTF row at line {line_number}")
            seqname, _, feature, start, end, _, strand, _, raw_attributes = fields
            if feature != "exon" or '"' not in raw_attributes:
                continue
            transcript_id = _parse_gtf_attributes(raw_attributes).get("transcript_id")
            if transcript_id in transcript_ids:
                exons[transcript_id].append((seqname, int(start), int(end), strand))
    return dict(exons)


def _ensembl_region(seqname, start, end):
    """GTF coordinates -> Ensembl region string on the forward strand (e.g. 1:100..200:1)."""
    chromosome = seqname.removeprefix("chr")
    if chromosome == "M":
        chromosome = "MT"
    return f"{chromosome}:{start}..{end}:1"


def _fetch_ensembl_regions(regions):
    """{region: forward-strand DNA} from the Ensembl REST API, in batches of 50."""
    sequences = {}
    unique_regions = list(dict.fromkeys(regions))
    for batch_start in range(0, len(unique_regions), ENSEMBL_MAX_REGIONS):
        batch = unique_regions[batch_start : batch_start + ENSEMBL_MAX_REGIONS]
        for attempt in range(ENSEMBL_RETRIES):
            try:
                response = requests.post(
                    ENSEMBL_REGION_URL,
                    json={"regions": batch},
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=ENSEMBL_TIMEOUT_SECONDS,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
                status = getattr(error.response, "status_code", None) if hasattr(error, "response") else None
                if status is not None and status < 500 and status != 429:
                    raise ValueError(f"Ensembl rejected the sequence request: {error}") from error
                if attempt + 1 == ENSEMBL_RETRIES:
                    raise ValueError(
                        f"Could not fetch exon sequences from Ensembl ({error}). "
                        "Check your internet connection, or pass --genome with a local GRCh38 FASTA."
                    ) from error
                retry_after = response.headers.get("Retry-After") if status == 429 else None
                time.sleep(float(retry_after) if retry_after else 2 ** attempt)
        for record in response.json():
            sequences[record["query"]] = record["seq"].upper()
    missing = [region for region in unique_regions if region not in sequences]
    if missing:
        raise ValueError(f"Ensembl returned no sequence for {len(missing)} exon(s), e.g. {missing[0]}")
    return sequences


def _assemble_transcripts(exons_by_transcript, genome_fasta_path=None):
    """{transcript_id: spliced DNA, 5'->3' on the transcript's strand}."""
    for transcript_id, exons in exons_by_transcript.items():
        if len({exon[0] for exon in exons}) != 1 or len({exon[3] for exon in exons}) != 1:
            raise ValueError(f"Inconsistent exon chromosome or strand for transcript {transcript_id!r}")

    if genome_fasta_path is not None:
        genome = _read_fasta(genome_fasta_path)

        def exon_sequence(seqname, start, end):
            if seqname not in genome:
                raise ValueError(f"GTF sequence {seqname!r} is absent from {genome_fasta_path}")
            return genome[seqname][start - 1 : end]
    else:
        fetched = _fetch_ensembl_regions(
            _ensembl_region(seqname, start, end)
            for exons in exons_by_transcript.values()
            for seqname, start, end, _ in exons
        )

        def exon_sequence(seqname, start, end):
            return fetched[_ensembl_region(seqname, start, end)]

    transcripts = {}
    for transcript_id, exons in exons_by_transcript.items():
        ordered = sorted(exons, key=lambda exon: exon[1])
        dna = "".join(exon_sequence(seqname, start, end) for seqname, start, end, _ in ordered)
        expected_length = sum(end - start + 1 for _, start, end, _ in ordered)
        if len(dna) != expected_length:
            raise ValueError(
                f"Assembled {len(dna)} nt for {transcript_id!r} but its exons span {expected_length} nt"
            )
        transcripts[transcript_id] = _reverse_complement(dna) if ordered[0][3] == "-" else dna
    return transcripts


def _check_against_cds(transcripts, cds_fasta_path):
    """Warnings for transcripts whose project CDS is not inside the assembled RNA."""
    if not cds_fasta_path or not Path(cds_fasta_path).is_file():
        return []
    cds = _read_fasta(cds_fasta_path)
    return [
        f"warning - the CDS of {transcript_id} in {Path(cds_fasta_path).name} is not inside "
        "the assembled RNA (genome build or exon coordinates may not match)"
        for transcript_id, dna in transcripts.items()
        if transcript_id in cds and cds[transcript_id] not in dna
    ]


def get_ranked_isoform_rnas(
    gene_name,
    isoform_numbers,
    *,
    matrix_path,
    gtf_path,
    genome_fasta_path=None,
    cds_fasta_path=None,
    adult_only=True,
    rank_csv_path=None,
    as_dna=False,
):
    """Return a dict of expression-rank -> (transcript_id, RNA sequence), plus warnings."""
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

    exons = _exons_from_gtf(transcript_by_rank.values(), gtf_path)
    missing = [tid for tid in transcript_by_rank.values() if tid not in exons]
    if missing:
        raise ValueError(f"No exons found in the GTF for: {', '.join(missing[:5])}")
    if genome_fasta_path is not None and not Path(genome_fasta_path).is_file():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta_path}")

    transcripts = _assemble_transcripts(exons, genome_fasta_path)
    warnings = _check_against_cds(transcripts, cds_fasta_path)

    rnas = {}
    for rank, transcript_id in transcript_by_rank.items():
        sequence = transcripts[transcript_id]
        rnas[rank] = (transcript_id, sequence if as_dna else sequence.replace("T", "U"))
    return rnas, warnings


def _format_transcript_mapping(gene_name, rnas):
    lines = [f"Transcript IDs for {gene_name}:"]
    lines.extend(
        f"Isoform {rank}: {transcript_id} ({len(sequence):,} nt)"
        for rank, (transcript_id, sequence) in sorted(rnas.items())
    )
    return "\n".join(lines)


def _format_rna_fasta(rnas):
    fasta_lines = []
    for rank, (_, sequence) in sorted(rnas.items()):
        fasta_lines.append(f">Isoform {rank}")
        fasta_lines.append(sequence)
    return "\n".join(fasta_lines) + "\n"


def _report(gene_name, rnas, warnings, as_dna):
    output_text = _format_rna_fasta(rnas)
    print(_format_transcript_mapping(gene_name, rnas))
    for warning in warnings:
        print(warning)
    print("cDNA FASTA:" if as_dna else "RNA FASTA:")
    print(output_text, end="")
    return output_text


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
            rnas, warnings = get_ranked_isoform_rnas(
                gene_name,
                ranks,
                matrix_path=str(args.matrix),
                gtf_path=str(args.gtf),
                genome_fasta_path=str(args.genome) if args.genome else None,
                cds_fasta_path=str(args.cds) if args.cds else None,
                adult_only=args.adult_only,
                rank_csv_path=args.rank_order_csv,
                as_dna=args.dna,
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"Could not process {gene_name!r}: {error}")
            continue

        fasta_outputs.append(_report(gene_name, rnas, warnings, args.dna))

    return fasta_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Return mature RNA sequences (spliced exons) for selected gene isoforms.",
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
        help="Local GRCh38 genome FASTA; if omitted, exon sequences are fetched from Ensembl",
    )
    parser.add_argument(
        "--cds",
        type=Path,
        default=ROOT_FOLDER / "Long_Read_Data" / "cds_output.fa",
        help="Project CDS FASTA, used only to check the assembled RNAs",
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
    parser.add_argument("--dna", action="store_true", help="Output T instead of U (cDNA rather than RNA alphabet)")
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
            print(f"Saved RNA FASTA to: {args.output}")
        return 0

    if not gene_name:
        parser.error("A gene name is required, for example: SLC45A1")
    if not isoform_numbers:
        parser.error("At least one isoform rank is required, for example: 1 2 3")

    rnas, warnings = get_ranked_isoform_rnas(
        gene_name,
        isoform_numbers,
        matrix_path=str(args.matrix),
        gtf_path=str(args.gtf),
        genome_fasta_path=str(args.genome) if args.genome else None,
        cds_fasta_path=str(args.cds) if args.cds else None,
        adult_only=args.adult_only,
        rank_csv_path=args.rank_order_csv,
        as_dna=args.dna,
    )
    output_text = _report(gene_name, rnas, warnings, args.dna)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Saved RNA FASTA to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
