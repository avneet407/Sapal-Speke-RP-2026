"""
Look up the coding sequence (CDS) of an isoform, or of every isoform of a gene.

Sequences come from Long_Read_Data/cds_output.fa (DNA, start codon to stop codon inclusive).
Only protein-coding isoforms have a CDS: ~45k of the 61k isoforms. The rest are reported as
"no CDS (non-coding)" rather than silently skipped.

Usage (from the Sapal-Speke-RP-2026 folder):
    python gene_finder.py G70506.252_ENST00000259455.4      # one isoform
    python gene_finder.py G70506                            # every isoform of a gene
    python gene_finder.py G70506 G49035.38_ENST00000600801.7  # several at once
With no names given, it looks up QUERIES in main().
"""

import sys

from gene_comparison import load_matrix, gene_ids

CDS_PATH = "Long_Read_Data/cds_output.fa"
LINE_WIDTH = 70   # bases per line when printing, same as the input FASTA


def read_fasta(path=CDS_PATH):
    """Read a FASTA file into {name: sequence}.
    Each record is a ">name" line followed by the sequence split over several lines,
    so lines are collected until the next ">" and then joined. Takes ~1-2 s for the CDS file."""
    sequences = {}
    name, chunks = None, []
    with open(path) as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def isoform_sequence(isoform, sequences, all_isoforms):
    """CDS of one isoform, e.g. "G70506.252_ENST00000259455.4".
    Returns None if the isoform exists but is non-coding; raises KeyError if the name is unknown."""
    if isoform in sequences:
        return sequences[isoform]
    if isoform in all_isoforms:
        return None
    raise KeyError(f"{isoform} is not in the expression matrix or the CDS file - check the name")


def gene_sequences(gene, sequences, df):
    """CDS of every isoform of a gene, e.g. "G70506".
    Returns a list of (isoform, mean CPM, sequence or None), most abundant isoform first
    (the same order inference.py uses to pick its top W isoforms)."""
    isoforms = df[[g == gene for g in gene_ids(df.index)]]
    if isoforms.empty:
        raise KeyError(f"No isoforms found for gene {gene} - check the name")
    mean_cpm = isoforms.mean(axis=1).sort_values(ascending=False)
    return [(isoform, cpm, sequences.get(isoform)) for isoform, cpm in mean_cpm.items()]


def print_fasta(name, sequence, description=""):
    """Print one record in FASTA format (can be pasted straight into BLAST etc.)."""
    print(f">{name} {description}".rstrip())
    for start in range(0, len(sequence), LINE_WIDTH):
        print(sequence[start:start + LINE_WIDTH])


def look_up(query, sequences, df):
    """Isoform names contain a "." (G70506.252_...); gene names don't (G70506)."""
    if "." in query:
        sequence = isoform_sequence(query, sequences, df.index)
        if sequence is None:
            print(f"# {query}: no CDS (non-coding isoform)")
        else:
            print_fasta(query, sequence, f"length={len(sequence)}")
    else:
        results = gene_sequences(query, sequences, df)
        n_coding = sum(sequence is not None for _, _, sequence in results)
        print(f"# {query}: {len(results)} isoforms, {n_coding} with a CDS (most abundant first)")
        for isoform, cpm, sequence in results:
            if sequence is None:
                print(f"# {isoform}: no CDS (non-coding), mean CPM = {cpm:.1f}")
            else:
                print_fasta(isoform, sequence, f"mean_cpm={cpm:.1f} length={len(sequence)}")
    print()


def main():
    # Edit this list, or pass names on the command line instead
    QUERIES = ["G70506.252_ENST00000259455.4"]
    queries = sys.argv[1:] or QUERIES

    sequences = read_fasta()
    df = load_matrix()   # needed for the full isoform list (incl. non-coding) and mean CPM

    for query in queries:
        try:
            look_up(query, sequences, df)
        except KeyError as error:
            print(f"# {error.args[0]}\n")


if __name__ == "__main__":
    main()
