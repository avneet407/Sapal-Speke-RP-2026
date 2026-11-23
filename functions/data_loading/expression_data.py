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


def mean_isoform_share(matrix, isoform_gene):
    """Mean % share of each isoform within its own gene, across the samples in `matrix`.

    Per sample: isoform CPM / gene total CPM * 100, where the gene total is the sum of ALL the
    gene's isoforms (gene = gene name in `isoform_gene`). Samples where the gene total is 0 are
    skipped (a gene that is 0 in every sample gives 0). This is the same mean share used by the
    1.5% filter in isoform_gene_correlation.py, and it is the measure isoforms are numbered by
    (1 = highest mean share)."""
    isoform_gene = isoform_gene.reindex(matrix.index)
    gene_total = matrix.groupby(isoform_gene, sort=False).sum()
    own_total = gene_total.loc[isoform_gene].set_axis(matrix.index)
    shares = matrix.div(own_total.where(own_total > 0)) * 100
    return shares.mean(axis=1).fillna(0)


def rank_isoforms_by_share(isoforms, mean_share):
    """Order isoform IDs by mean share, highest first. Ties (e.g. isoforms with share 0) keep the
    order they were given in, so the numbering is the same wherever this is used."""
    return sorted(isoforms, key=lambda isoform_id: -mean_share.get(isoform_id, 0))


def region_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the region is the text before the first "_"."""
    return sample_name.split("_")[0]


def disease_of(sample_name):
    """Sample names look like "Caudate_MDD_Adult_R17353" - the disease is the second part ("MDD")."""
    return sample_name.split("_")[1]
