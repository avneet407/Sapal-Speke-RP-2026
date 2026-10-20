"""
control.py

Table of every NRXN1 isoform with its CPM in each adult sample,
plus a column saying whether the isoform contains the AS4 exon.

Run from the Sapal-Speke-RP-2026 folder: python control.py
"""
import re
from collections import defaultdict

import pandas as pd

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
TARGET = "G35694"                 # NRXN1
BLAST_HIT = (50054956, 50055045)  # TBLASTN hit of the UniProt AS4 peptide (chr2)


def remove_fetal(df):
    """Drop the fetal sample columns."""
    return df[[sample for sample in df.columns if "Fetal" not in sample]]


def read_gene_exons(gene=TARGET):
    """Exons of every isoform of one gene: {transcript_id: [(start, end), ...]}."""
    exon_lists = defaultdict(list)
    with open(GTF_PATH) as gtf:
        for line in gtf:
            if f'"{gene}.' not in line:
                continue
            fields = line.split("\t")
            if fields[2] != "exon":
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', fields[8]).group(1)
            exon_lists[transcript_id].append((int(fields[3]), int(fields[4])))
    return exon_lists


def find_as4_exon(exon_lists):
    """AS4 = the exon overlapping the BLAST hit that is found in the most isoforms."""
    counts = defaultdict(int)
    for exons in exon_lists.values():
        for start, end in exons:
            if start <= BLAST_HIT[1] and end >= BLAST_HIT[0]:
                counts[(start, end)] += 1
    return max(counts, key=counts.get)


def nrxn1_isoform_table(df, as4, exon_lists):
    """One row per NRXN1 isoform: CPM in each sample, plus contains_AS4 (True/False)."""
    isoforms = df[[t.split(".")[0] == TARGET for t in df.index]].copy()
    isoforms["contains_AS4"] = [as4 in exon_lists.get(t, []) for t in isoforms.index]
    return isoforms


def main():
    df = remove_fetal(pd.read_csv(MATRIX_PATH, sep="\t", index_col=0))
    exon_lists = read_gene_exons()
    as4 = find_as4_exon(exon_lists)

    table = nrxn1_isoform_table(df, as4, exon_lists)
    print(f"AS4 exon: chr2:{as4[0]}-{as4[1]}")
    print(f"{len(table)} NRXN1 isoforms x {df.shape[1]} adult samples; "
          f"{table['contains_AS4'].sum()} contain AS4")
    print(table)
    return table


if __name__ == "__main__":
    main()
