"""Reading the expression matrix and GTF, and the naming rules for genes and samples."""

import re

import pandas as pd


def load_matrix(path):
    """Rows = isoforms (transcript IDs), columns = samples, values = CPM."""
    # sep="\t"     -> the file is tab-separated (.tsv)
    # index_col=0  -> use the first column (transcript_id) as row names, not as data
    return pd.read_csv(path, sep="\t", index_col=0)


def gene_of(isoform_id):
    """Gene = part of the isoform ID before the first "." or "_".
    G249.3.nnc -> G249,  G249.7_ENST00000471889.7 -> G249"""
    # .split(".") breaks the text at every "."  ("G249.3.nnc" -> ["G249", "3", "nnc"])
    # [0] takes the first piece. The second split handles IDs like "G249_..." just in case.
    return isoform_id.split(".")[0].split("_")[0]


def read_gene_names(gtf_path):
    """G locus -> gene name, from the GTF "transcript" lines, which look like
      transcript_id "G249.3.nnc"; gene_name "SLC45A1"; ...
    A locus can contain transcripts with more than one gene name (e.g. overlapping genes);
    use the name with the most transcripts."""
    loci, gene_names = [], []
    with open(gtf_path) as file:
        for line in file:
            columns = line.split("\t")
            if len(columns) < 9 or columns[2] != "transcript":   # skip exon / CDS lines
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', columns[8]).group(1)
            name = re.search(r'gene_name "([^"]+)"', columns[8])
            loci.append(gene_of(transcript_id))
            gene_names.append(name.group(1) if name else "")
    # For each locus, value_counts() counts each name; index[0] is the most common one
    return pd.Series(gene_names).groupby(loci).agg(lambda found: found.value_counts().index[0])


def region_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the region is the text before the first "_"."""
    return sample_name.split("_")[0]


def disease_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the disease is the second part ("MDD")."""
    return sample_name.split("_")[1]
