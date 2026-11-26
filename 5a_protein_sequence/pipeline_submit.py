"""
Submit the proteins of every asymmetric gene from step 4 to InterProScan.

For each gene with n_pairs_as_asymmetric_gene > 0 in the step-4 genes table, the
isoforms tested in step 3b (mean share >= 1.5%) are translated from their CDS and
handed to submitting_to_InterPro.submit_sequences. That function records every
isoform in the job database, de-duplicates identical proteins (including ones
already scanned), queues them, and sends the first batch (max 30 jobs at once).

The remaining queued jobs are sent, and results collected, by the normal check step:
    python 5a_protein_sequence/submitting_to_InterPro.py check --watch
and exported with:
    python 5a_protein_sequence/submitting_to_InterPro.py export

Run with:
    python 5a_protein_sequence/pipeline_submit.py --dry-run   # counts only, nothing sent
    python 5a_protein_sequence/pipeline_submit.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from identify_protein_seq import (  # noqa: E402
    DEFAULT_RANKING_CSV,
    _read_fasta,
    _translate_dna_to_protein,
)
from submitting_to_InterPro import (  # noqa: E402
    DEFAULT_DB_PATH,
    InterProConfig,
    initialize_database,
    submit_sequences,
)


# --- Settings ---------------------------------------------------------------
EMAIL = os.environ.get("INTERPROSCAN_EMAIL", "tom.speke@outlook.com")
ISOFORMS = "tested"
# "tested": every isoform tested in 3b for the gene (needed to compare isoforms with
#           each other, since asymmetry is about some isoforms responding and others not)
# "significant": only the isoforms significant in at least one candidate pair

STEP4_CSV = ROOT_FOLDER / "4_identifying_candidates" / "CSV"
GENES_CSV = STEP4_CSV / "promising_candidate_genes_cpm1_frac80%_share1.5pct_BY_FDR5.csv"
PAIRS_CSV = STEP4_CSV / "promising_candidate_pairs_cpm1_frac80%_share1.5pct_BY_FDR5.csv"
TESTS_FOLDER = ROOT_FOLDER / "3b_isoform_gene_hypothesis_testing" / "CSV"
CDS_FASTA = ROOT_FOLDER / "Long_Read_Data" / "cds_output.fa"


def asymmetric_genes() -> list[str]:
    genes = pd.read_csv(GENES_CSV)
    return genes.loc[genes["n_pairs_as_asymmetric_gene"] > 0, "gene"].tolist()


def isoforms_per_gene(genes: list[str]) -> dict[str, set[str]]:
    """Isoform IDs to submit for each gene, according to ISOFORMS."""
    if ISOFORMS == "significant":
        pairs = pd.read_csv(PAIRS_CSV, usecols=["asymmetric_gene", "significant_isoforms"])
        pairs = pairs.assign(isoform=pairs["significant_isoforms"].str.split(";")).explode("isoform")
        table = pairs.rename(columns={"asymmetric_gene": "gene"})[["gene", "isoform"]]
    elif ISOFORMS == "tested":
        # Same choice of table as step 4: the latest corrected 3b table (all tests)
        tests_csv = max(
            TESTS_FOLDER.glob("isoform_gene_correlations_*.csv"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        tests = pd.read_csv(tests_csv, usecols=["gene_a", "isoform_a_id", "gene_b", "isoform_b_id"])
        table = pd.concat(
            [
                tests[["gene_a", "isoform_a_id"]].set_axis(["gene", "isoform"], axis=1),
                tests[["gene_b", "isoform_b_id"]].set_axis(["gene", "isoform"], axis=1),
            ]
        )
    else:
        raise ValueError("ISOFORMS must be 'tested' or 'significant'")

    table = table.dropna().drop_duplicates()
    table = table[table["gene"].isin(genes)]
    return table.groupby("gene")["isoform"].agg(set).to_dict()


def isoform_ranks(genes: list[str]) -> dict[str, dict[str, int]]:
    """Expression rank of each isoform, matching identify_protein_seq's ranking."""
    ranking = pd.read_csv(DEFAULT_RANKING_CSV, usecols=["gene", "isoform", "mean_cpm"])
    ranks = {}
    for gene in genes:
        gene_rows = ranking[ranking["gene"].astype(str).str.casefold() == gene.casefold()]
        ordered = gene_rows.sort_values("mean_cpm", ascending=False)["isoform"].tolist()
        ranks[gene] = {isoform: rank for rank, isoform in enumerate(ordered, start=1)}
    return ranks


def database_gene_names(db_path: Path) -> dict[str, str]:
    """Spelling already used in the database for each gene (e.g. 'mef2c'), so a
    re-submitted gene is not recorded twice under different capitalisation."""
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        names = [row[0] for row in connection.execute("SELECT DISTINCT gene_name FROM isoforms")]
    return {name.casefold(): name for name in names}


def build_submissions(genes: list[str], db_path: Path) -> tuple[dict, list[str]]:
    """Return {gene: {rank: (transcript_id, protein)}} and notes on skipped isoforms."""
    wanted = isoforms_per_gene(genes)
    ranks = isoform_ranks(genes)
    cds = _read_fasta(CDS_FASTA)
    known_names = database_gene_names(db_path)

    submissions, notes = {}, []
    for gene in genes:
        proteins = {}
        for isoform in sorted(wanted.get(gene, ())):
            rank = ranks[gene].get(isoform)
            if rank is None:
                notes.append(f"{gene}: {isoform} has no expression rank, skipped")
            elif isoform not in cds:
                continue  # non-coding isoform: no CDS, nothing to scan
            else:
                try:
                    proteins[rank] = (isoform, _translate_dna_to_protein(cds[isoform]))
                except ValueError as error:
                    notes.append(f"{gene}: {isoform} not translated ({error}), skipped")
        if proteins:
            submissions[known_names.get(gene.casefold(), gene)] = proteins
        else:
            notes.append(f"{gene}: no protein-coding tested isoforms, skipped")
    return submissions, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be submitted.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    genes = asymmetric_genes()
    submissions, notes = build_submissions(genes, args.db)
    n_isoforms = sum(len(proteins) for proteins in submissions.values())
    unique = {protein for proteins in submissions.values() for _, protein in proteins.values()}
    print(f"Asymmetric genes: {len(genes)}; with protein-coding isoforms: {len(submissions)}")
    print(f"Isoforms ({ISOFORMS}): {n_isoforms}; unique protein sequences: {len(unique)}")
    for note in notes:
        print(f"  note - {note}")

    if args.dry_run:
        print("Dry run: nothing submitted.")
        return 0

    config = InterProConfig()
    totals = {"submitted_now": 0, "cached_or_already_recorded": 0}
    for index, (gene, proteins) in enumerate(submissions.items(), start=1):
        report = submit_sequences(gene, proteins, email=EMAIL, db_path=args.db, config=config)
        for key in totals:
            totals[key] += report[key]
        print(
            f"[{index}/{len(submissions)}] {gene}: {len(proteins)} isoforms, "
            f"{report['submitted_now']} sent now, {report['queued']} queued in total"
        )

    print(
        f"\nSent now: {totals['submitted_now']}; already scanned/recorded: "
        f"{totals['cached_or_already_recorded']}; queued: {report['queued']}."
    )
    print("Send the rest and collect results with:")
    print("    python 5a_protein_sequence/submitting_to_InterPro.py check --watch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
