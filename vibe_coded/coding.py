"""
coding.py

How many genes in the dataset code for proteins?

The GTF has no gene-type (biotype) field, so each gene is classified from two pieces of evidence:
  1. CDS: does any of its isoforms have a CDS record in the GTF (= an entry in cds_output.fa)?
  2. Name: does its gene_name follow a non-coding naming convention?
       LINC...                      long intergenic non-coding RNA
       ...-AS1 / -DT / -IT / -OT    antisense / divergent / intronic / overlapping lncRNA
       MIR...HG, SNHG...            microRNA / snoRNA host genes
       ENSG... (no symbol)          unnamed, mostly novel lncRNAs (e.g. G8359)
     lncRNAs can still get a predicted CDS (short ORFs, e.g. LINC02552 has 4), so a CDS alone
     isn't enough - a gene counts as protein-coding only if it has a CDS AND a normal gene name.

Also reports the share of each gene's expression that comes from its CDS-carrying isoforms:
real protein-coding genes are mostly expressed as coding isoforms; lncRNAs with stray ORFs aren't.

Run from the Sapal-Speke-RP-2026 folder: python coding.py   (~5-10 s)
Writes Results/coding/gene_coding_summary_<timestamp>.csv (one row per gene).
"""
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

from gene_comparison import load_matrix
from control import remove_fetal

GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
MIN_GENE_CPM = 1000      # same expression filter as inference.py (X), to report the usable set too
TABLE_FOLDER = "Results/coding"

NON_CODING_NAME = re.compile(r"^(LINC\d|LOC\d|SNHG\d|MIR\d+.*HG)|-(AS|DT|IT|OT)\d*$")
UNNAMED = re.compile(r"^ENSG\d+")


def read_transcripts(path=GTF_PATH):
    """From the GTF: {transcript_id: gene_name} and the set of transcripts with a CDS."""
    names, has_cds = {}, set()
    with open(path) as gtf:
        for line in gtf:
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', fields[8]).group(1)
            if fields[2] == "transcript":
                names[transcript_id] = re.search(r'gene_name "([^"]+)"', fields[8]).group(1)
            elif fields[2] == "CDS":
                has_cds.add(transcript_id)
    return names, has_cds


def classify(gene_name, n_coding_isoforms):
    """Category for one gene from its name and whether any isoform has a CDS."""
    if UNNAMED.match(gene_name):
        return "unnamed (Ensembl ID only)"
    if NON_CODING_NAME.search(gene_name):
        return "lncRNA (by name)"
    if n_coding_isoforms > 0:
        return "protein-coding"
    return "named, no CDS"


def gene_table(names, has_cds, isoform_cpm):
    """One row per gene (G-number): name, isoform counts, % expression from CDS isoforms, category.
    isoform_cpm = each isoform's CPM summed over all samples (isoforms missing from the matrix count as 0)."""
    isoforms_by_gene = defaultdict(list)
    for transcript_id in names:
        isoforms_by_gene[transcript_id.split(".")[0]].append(transcript_id)

    rows = []
    for gene, isoforms in isoforms_by_gene.items():
        # A few genes have isoforms labelled with different names; use the most common one
        gene_name = Counter(names[t] for t in isoforms).most_common(1)[0][0]
        coding = [t for t in isoforms if t in has_cds]
        total_cpm = isoform_cpm.reindex(isoforms).sum()
        coding_cpm = isoform_cpm.reindex(coding).sum()
        rows.append({"gene": gene, "gene_name": gene_name, "n_isoforms": len(isoforms),
                     "n_coding_isoforms": len(coding),
                     "percent_cpm_from_coding": 100 * coding_cpm / total_cpm if total_cpm > 0 else float("nan"),
                     "total_cpm_all_samples": total_cpm,
                     "category": classify(gene_name, len(coding))})
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def summarise(table, label):
    """Print counts per category, with how many in each have any CDS and the median coding share."""
    summary = table.groupby("category").agg(
        genes=("gene", "size"),
        with_any_cds=("n_coding_isoforms", lambda n: (n > 0).sum()),
        median_percent_cpm_coding=("percent_cpm_from_coding", "median"))
    summary = summary.sort_values("genes", ascending=False)
    summary.loc["TOTAL"] = [summary["genes"].sum(), summary["with_any_cds"].sum(), float("nan")]
    print(f"\n{label}")
    print(summary.to_string(float_format=lambda v: f"{v:.0f}"))


def main():
    names, has_cds = read_transcripts()
    df = remove_fetal(load_matrix())          # adults only, as in inference.py
    table = gene_table(names, has_cds, df.sum(axis=1))
    n_coding = (table["category"] == "protein-coding").sum()
    print(f"{n_coding} of {len(table)} genes are protein-coding "
          f"(have a CDS and a normal gene name)")
    summarise(table, "All genes:")

    passing = table[table["total_cpm_all_samples"] >= MIN_GENE_CPM]
    print(f"\n{(passing['category'] == 'protein-coding').sum()} of {len(passing)} genes passing "
          f"inference.py's filter (total CPM >= {MIN_GENE_CPM}) are protein-coding")
    summarise(passing, f"Genes passing the CPM >= {MIN_GENE_CPM} filter:")

    print("\nExamples of each category:")
    for category, group in table.groupby("category"):
        print(f"  {category}: {', '.join(group['gene_name'].head(8))}")

    os.makedirs(TABLE_FOLDER, exist_ok=True)
    path = os.path.join(TABLE_FOLDER, f"gene_coding_summary_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
    table.to_csv(path, index=False)
    print(f"\nSaved table: {path}")


if __name__ == "__main__":
    main()
