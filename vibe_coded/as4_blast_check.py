"""
as4_blast_check.py

Confirm which exon in the long-read annotation matches the BLAST hit for the NRXN1 AS4 peptide.

The AS4 peptide comes from UniProt Q9ULB1, alternative sequence VSP_041355
(a 30-residue insertion after residue 1239, in the 6th laminin G-like domain):
    GNNDNERLAIARQRIPYRLGRVVDEWLLDK
TBLASTN of this peptide against the human genome (GRCh38, Ensembl) gave a 100%-identity hit at
chr2:50,054,956-50,055,045 (reverse strand).

This script finds every exon (and CDS) line in the GTF that overlaps that hit, so we can check:
    - which exon(s) it is, and which gene they belong to
    - how many isoforms contain it
This identifies AS4 from outside annotation only, without using any correlation with KHDRBS3.

Run from the Sapal-Speke-RP-2026 folder: python as4_blast_check.py
"""
import re
import pandas as pd

GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
CHROMOSOME = "chr2"
BLAST_HIT = (50054956, 50055045)  # TBLASTN hit of the UniProt AS4 peptide (start, end)


def read_gtf_features(gtf_path=GTF_PATH, chromosome=CHROMOSOME, features=("exon", "CDS")):
    """Read GTF lines on one chromosome whose feature type is in `features`.
    GTF columns: chromosome, source, feature, start, end, score, strand, frame, attributes
    Returns one row per GTF line, with transcript_id, gene_name and gene_id taken from the attributes."""
    rows = []
    with open(gtf_path) as gtf:
        for line in gtf:
            fields = line.split("\t")
            if len(fields) < 9 or fields[0] != chromosome or fields[2] not in features:
                continue

            attributes = fields[8]
            transcript_id = re.search(r'transcript_id "([^"]+)"', attributes)
            gene_name = re.search(r'gene_name "([^"]+)"', attributes)
            gene_id = re.search(r'gene_id "([^"]+)"', attributes)

            rows.append({"chromosome": fields[0],
                         "feature": fields[2],
                         "start": int(fields[3]),
                         "end": int(fields[4]),
                         "strand": fields[6],
                         "transcript_id": transcript_id.group(1) if transcript_id else "",
                         "gene_name": gene_name.group(1) if gene_name else "",
                         "gene_id": gene_id.group(1) if gene_id else ""})
    return pd.DataFrame(rows)


def overlapping(rows, hit_start, hit_end):
    """Keep rows that share at least one base with the hit.
    Two ranges overlap if each one starts before the other ends."""
    return rows[(rows["start"] <= hit_end) & (rows["end"] >= hit_start)]


def summarise_overlaps(rows, hit_start, hit_end):
    """One row per distinct (feature, start, end, strand, gene): how many transcripts contain it,
    its length, and how many of its bases are shared with the hit."""
    summary = (rows.groupby(["feature", "start", "end", "strand", "gene_name"])["transcript_id"]
                   .nunique()
                   .reset_index(name="n_transcripts"))
    summary["length"] = summary["end"] - summary["start"] + 1
    # Shared bases = from the later of the two starts to the earlier of the two ends
    summary["overlap_bp"] = (summary["end"].clip(upper=hit_end)
                             - summary["start"].clip(lower=hit_start) + 1)
    return summary.sort_values(["feature", "start"])


def main():
    hit_start, hit_end = BLAST_HIT

    # 1. Load exon and CDS lines on chr2, and keep those overlapping the BLAST hit
    rows = read_gtf_features()
    hits = overlapping(rows, hit_start, hit_end)

    # 2. Summary table of distinct exons / CDS segments overlapping the hit
    summary = summarise_overlaps(hits, hit_start, hit_end)
    pd.set_option("display.width", 200)
    print(f"GTF features overlapping BLAST hit {CHROMOSOME}:{hit_start}-{hit_end}\n")
    print(summary[["feature", "start", "end", "length", "strand", "gene_name",
                   "n_transcripts", "overlap_bp"]].to_string(index=False))

    # 3. Verdict, using exon lines only (CDS lines repeat the coordinates for coding isoforms)
    exons = summary[summary["feature"] == "exon"]
    n_exon_lines = (hits["feature"] == "exon").sum()
    n_cds_lines = (hits["feature"] == "CDS").sum()

    print(f"\nDistinct exons overlapping the hit: {len(exons)}")
    print(f"Genes involved: {', '.join(sorted(hits['gene_name'].unique()))}")
    for _, exon in exons.iterrows():
        print(f"  Exon {exon['start']}-{exon['end']} ({exon['length']} nt, {exon['strand']} strand, "
              f"{exon['gene_name']}): in {exon['n_transcripts']} transcripts, "
              f"{exon['overlap_bp']}/{exon['length']} bases shared with the hit")
        # The hit is shifted relative to the exon because the first codon of the peptide
        # starts in the previous exon; BLAST completes it with the neighbouring intron base.
        print(f"    Offset (hit - exon): start {hit_start - exon['start']:+d}, end {hit_end - exon['end']:+d}")

    # Ctrl+F for the start coordinate counts both exon and CDS lines
    print(f"\nGTF lines overlapping the hit: {n_exon_lines} exon + {n_cds_lines} CDS = {n_exon_lines + n_cds_lines}")


if __name__ == "__main__":
    main()
