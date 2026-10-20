# Data description

All data files are in `Long_Read_Data/` (git-ignored, not in the repository).

| File | What it is |
|---|---|
| `expressed_isoforms_matrix.tsv` | Expression matrix: one row per isoform, one column per sample, values in CPM |
| `expressed_isoforms.gtf` | Genome coordinates (exons, CDS) of every isoform in the matrix |
| `cds_output.fa` | Coding sequence (DNA) of each isoform that has a predicted open reading frame |

## Headline numbers

| | Count |
|---|---|
| Isoforms (transcripts) | **61,117** |
| Genes (loci) | **816** |
| Samples | **49** (45 adult + 4 fetal) |
| Isoforms with a coding sequence (CDS) | 45,479 (74%) |
| Zero entries in the matrix | 76% |

## Isoforms

The isoforms come from PacBio long-read sequencing, so each read covers a full-length
transcript. Isoform IDs look like `G249.3.nnc`:

- `G249` is the gene (locus) the isoform belongs to
- `3` is the isoform number within that gene
- the suffix is the SQANTI category, which says how the isoform compares with the
  reference annotation (Ensembl):

| Suffix | Meaning | Isoforms |
|---|---|---|
| `_ENST...` | Matches a known Ensembl transcript exactly | 4,868 |
| `.ism` | Incomplete splice match: a fragment of a known transcript | 981 |
| `.nic` | Novel in catalogue: new combination of known splice sites | 6,573 |
| `.nnc` | Novel not in catalogue: uses at least one new splice site | 48,654 |
| (none) | ID ends in a number only, e.g. `G10726.122`; no category given | 41 |

So about 90% of the isoforms are not in the reference annotation.

### Why `.nnc` isoforms are not removed

`.nnc` isoforms use at least one splice site that has never been seen before, so some
of them may be sequencing or alignment errors. Removing the whole category is still a
bad idea, because `.nnc` isoforms carry a large share of the expression:

| Category | Isoforms | % of all expression | Median % of its gene | Major isoforms (≥10% of gene) | Genes where it is the top isoform |
|---|---|---|---|---|---|
| known (`_ENST`) | 4,862 | 47.9 | 0.69 | 840 | 464 |
| `.nnc` | 48,569 | **36.9** | 0.04 | **710** | **293** |
| `.nic` | 6,559 | 12.7 | 0.11 | 99 | 42 |
| `.ism` | 980 | 2.5 | 0.16 | 35 | 15 |
| (none) | 41 | 0.03 | 0.74 | 3 | 1 |

These figures use the 45 adult samples. An isoform's **% of its gene** is its CPM
divided by its gene's total CPM in each sample, averaged over the samples where the
gene is expressed. An isoform is **major** if that average is at least 10%. Section 6
of `describe_data.py` computes this table.

- `.nnc` isoforms make up **37% of all expression**, second only to known transcripts.
- In **293 of 815 genes**, the most-used isoform is `.nnc`.
- **138 genes have only `.nnc` isoforms.** Removing `.nnc` would delete these genes
  entirely and lower the totals of many others.
- Brain tissue is known for extensive novel splicing, so many of these isoforms are
  probably real.

The problem is **low expression, not the category**. The *typical* `.nnc` isoform is
tiny (median 0.04% of its gene), and only 710 of the 48,569 are major. The rare ones are
where errors are most likely. So isoforms are filtered **by expression**, using the same
rule for every category: for example, keep isoforms detected in at least ~10 samples
or making up at least 10% of their gene. This removes most dubious `.nnc` isoforms and
keeps the well-supported ones. Gene totals are calculated from all isoforms *before*
filtering, so the % shares still reflect the gene's true expression.

**Robustness check:** re-run key results using only known + `.nic` isoforms. If they
still hold, they don't depend on novel isoforms.

## Genes

Genes are defined by the `G` prefix of the isoform ID. The isoforms of a gene are
summed to give the gene's CPM. There are 816 genes.

A `G` locus does not always correspond to exactly one named gene. In the GTF, 660 loci
map to a single gene name. The other 156 contain transcripts with more than one gene
name, for example overlapping or read-through genes. The GTF also lists 1,036
distinct Ensembl gene IDs, and 212 transcripts have no `gene_id`.

### Isoforms per gene

The number of isoforms per gene varies a lot: median 11, mean 75, range 1 to 1,439.

| Isoforms per gene | Number of genes |
|---|---|
| 1 | 121 |
| 2 | 79 |
| 3–5 | 108 |
| 6–10 | 96 |
| 11–20 | 74 |
| 21–50 | 102 |
| 51–100 | 67 |
| 101–200 | 75 |
| 201–500 | 70 |
| >500 | 24 |

- 121 genes (15%) have only one isoform. They have no splicing choice to study, but
  they can still act as a "regulator" gene B.
- The most isoform-rich genes are G21721 (1,439), G54577 (1,396), G70076 (1,271),
  G61543 (993) and G6716 (991). Many of these isoforms are probably rare.

## Samples

The 49 samples are human brain tissue, most likely post-mortem (the design looks like
the LIBD brain collection, but this has not been confirmed with Earlham). Each sample
has a different ID (49 unique IDs). These probably correspond to different donors. Column names follow
`Region_Condition_Age_SampleID`, e.g. `Caudate_MDD_Adult_R17353`.

- **Regions:** DLPFC (dorsolateral prefrontal cortex), Hippocampus, Caudate
- **Conditions:** Control, Bipolar disorder, MDD (major depressive disorder),
  Schizophrenia
- **Age:** Adult, or Fetal (4 fetal DLPFC controls)

| Region | Age | Control | Bipolar | MDD | Schizo | Total |
|---|---|---|---|---|---|---|
| Caudate | Adult | 3 | 4 | 3 | 4 | 14 |
| DLPFC | Adult | 4 | 4 | 4 | 3 | 15 |
| DLPFC | Fetal | 4 | 0 | 0 | 0 | 4 |
| Hippocampus | Adult | 4 | 4 | 4 | 4 | 16 |
| **Total** | | **15** | **12** | **11** | **11** | **49** |

The 4 fetal samples are usually excluded, because fetal and adult brains differ a lot
and that difference could dominate correlations. That leaves 45 adult samples, spread
almost evenly across 3 regions × 4 conditions (3–4 samples per group).

## Expression values

- Values are CPM (counts per million): each sample's column adds up to about 1,000,000
  (980,882 to 992,519).
- Every isoform is expressed in at least one sample (no all-zero rows), but 76% of all
  entries are zero, so most isoforms are detected in only a few samples.
