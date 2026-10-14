import re
import pandas as pd
from scipy.stats import spearmanr


def load_matrix(path="expressed_isoforms_matrix.tsv"):
    """Loading Expression Matrix (isoform expression levels for 49 samples)"""
    return pd.read_csv(path,
                       sep="\t",
                       index_col=0,
                      )


def gene_ids(transcript_ids):
    """Gene ID is the part of the transcript ID before the first "." (e.g. G249.3.nnc -> G249)"""
    return [name.split(".")[0] for name in transcript_ids]


def sum_by_gene(df):
    """Sum TPM of all isoforms of each gene (one row per gene, one column per sample)."""
    return df.groupby(gene_ids(df.index), sort=False).sum()


def top_isoforms_of_gene(df, gene, n=10):
    """All isoforms (rows) belonging to gene, plus the top n by average TPM across the 49 samples."""
    isoforms = df[[g == gene for g in gene_ids(df.index)]]
    average_isoform_tpm = isoforms.mean(axis=1).sort_values(ascending=False)
    return isoforms, average_isoform_tpm, list(average_isoform_tpm.index[:n])


def read_exons(gene, isoforms, gtf_path="expressed_isoforms.gtf"):
    """Read only the exon lines of the given isoforms from the GTF.
    GTF columns: chromosome, source, feature, start, end, score, strand, frame, attributes
    Returns {isoform: [(start, end), ...]} and the gene's strand."""
    exons = {isoform: [] for isoform in isoforms}
    strand = "+"
    with open(gtf_path) as gtf:
        for line in gtf:
            if f'"{gene}.' not in line:
                continue
            fields = line.split("\t")
            if fields[2] != "exon":
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', fields[8]).group(1)
            if transcript_id in exons:
                exons[transcript_id].append((int(fields[3]), int(fields[4])))
                strand = fields[6]
    return exons, strand


def label_exons(exons, strand):
    """Give every distinct exon (start, end) across the isoforms a label E1, E2, ...
    numbered in genomic order (reversed on "-" strand so E1 is always the 5' end).
    Exons with slightly different start/end (alternative splice sites) count as different exons."""
    all_exons = sorted({exon for exon_list in exons.values() for exon in exon_list}, reverse=(strand == "-"))
    return {exon: f"E{i + 1}" for i, exon in enumerate(all_exons)}


def isoform_structure_table(isoforms_b, average_isoform_tpm, top_isoforms, tpm_a, exons, exon_labels, strand):
    """Table: isoform, Spearman rho, p-value, structure, and one 0/1 column per exon."""
    results = []
    for isoform in top_isoforms:
        rho, p_value = spearmanr(isoforms_b.loc[isoform], tpm_a) # Compares across the 49 samples.

        isoform_exons = sorted(exons[isoform], reverse=(strand == "-"))
        row = {"isoform": isoform,
               "average_tpm": average_isoform_tpm[isoform],
               "spearman_rho": rho,
               "p_value": p_value,
               "n_exons": len(isoform_exons),
               "structure": "-".join(exon_labels[exon] for exon in isoform_exons)}

        # One column per exon: 1 if the isoform contains it, 0 if not.
        for exon, label in exon_labels.items():
            row[label] = int(exon in isoform_exons)
        results.append(row)

    return pd.DataFrame(results)


def exon_key_table(exon_labels):
    """Exon coordinates, so you can look up what each label means."""
    return pd.DataFrame([{"exon": label, "start": start, "end": end, "length": end - start + 1}
                         for (start, end), label in exon_labels.items()])


def main():
    gene_a = "G60740"
    gene_b = "G49035"

    df = load_matrix()
    gene_tpm = sum_by_gene(df)
    tpm_a = gene_tpm.loc[gene_a]

    isoforms_b, average_isoform_tpm, top_isoforms = top_isoforms_of_gene(df, gene_b, n=10)

    # ---- Exon structure of each isoform from the GTF ----
    exons, strand = read_exons(gene_b, top_isoforms)
    exon_labels = label_exons(exons, strand)

    # ---- Table: isoform, Spearman rho, p-value, structure ----
    results = isoform_structure_table(isoforms_b, average_isoform_tpm, top_isoforms, tpm_a, exons, exon_labels, strand)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(results[["isoform", "average_tpm", "spearman_rho", "p_value", "n_exons", "structure"]])

    # Exon presence table (1 = exon present in isoform)
    print(results.set_index("isoform")[list(exon_labels.values())])

    print(exon_key_table(exon_labels))


if __name__ == "__main__":
    main()
