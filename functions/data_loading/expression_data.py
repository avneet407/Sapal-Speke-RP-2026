"""Reading the expression matrix and GTF, and the naming rules for genes and samples."""

import re

import pandas as pd


def load_matrix(path):
    """Rows = isoforms (transcript IDs), columns = samples, values = CPM."""
    # sep="\t"     -> the file is tab-separated (.tsv)
    # index_col=0  -> use the first column (transcript_id) as row names, not as data
    return pd.read_csv(path, sep="\t", index_col=0)


def read_isoform_genes(gtf_path):
    """Isoform (transcript ID) -> gene name, from the GTF "transcript" lines, which look like
      transcript_id "G249.3.nnc"; gene_name "SLC45A1"; ...
    Genes are defined by gene name, not by the G number in the ID: one G number can hold several
    different genes (G2476 = NEGR1 + ZRANB2), and one gene can be split over several G numbers."""
    isoforms, gene_names = [], []
    with open(gtf_path) as file:
        for line in file:
            columns = line.split("\t")
            if len(columns) < 9 or columns[2] != "transcript":   # skip exon / CDS lines
                continue
            isoforms.append(re.search(r'transcript_id "([^"]+)"', columns[8]).group(1))
            gene_names.append(re.search(r'gene_name "([^"]+)"', columns[8]).group(1))
    return pd.Series(gene_names, index=isoforms)


def region_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the region is the text before the first "_"."""
    return sample_name.split("_")[0]


def disease_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the disease is the second part ("MDD")."""
    return sample_name.split("_")[1]
