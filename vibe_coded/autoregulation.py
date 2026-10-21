"""
Negative autoregulation of one splicing regulator, measured across the 45 adult samples.

Background: many splicing regulators control their own level. When there is too much of
the protein, it splices its own RNA into an "unproductive" isoform that carries a premature
stop codon, and the cell destroys that RNA (nonsense-mediated decay, NMD). Extra RNA is
shunted into the unproductive isoforms, so the productive RNA (which makes protein) is
held steadier than the gene total. This is negative feedback.

Steps
  1. Classify every isoform:
       productive = has a CDS whose stop codon is in the last exon, or <= 50 nt before the last
                    exon-exon junction (the standard rule for escaping NMD)
       NMD        = has a CDS, but its stop codon is > 50 nt before the last junction
                    (premature stop codon -> the RNA is destroyed)
       no CDS     = no coding sequence predicted (e.g. retained intron, or a missed prediction)
  2. For GENE, add up productive and unproductive CPM in each sample.
  3. Feedback signatures (prediction under negative autoregulation):
       a) sensitivity S = slope of log(productive) vs log(total)    -> S < 1 (productive is buffered)
       b) unproductive fraction vs total (Spearman)                 -> positive (excess goes to NMD)
       c) variability (CV) of productive vs total                   -> productive varies less
     Each is calculated with all samples, and "within region" (each sample's region average
     removed first), so differences between brain regions can't create the pattern.
  4. Baseline: the same numbers for every other gene that has unproductive isoforms. Some
     of the pattern appears even without feedback (measurement noise in the unproductive part
     lowers S, for example), so the question is whether GENE is MORE extreme than most genes.
  5. Save: isoform table, per-sample table, baseline table (CSV) and a figure.

Caveat: unproductive RNA is destroyed, so we only see what is left at the time of sampling.
The measured unproductive fraction underestimates how much RNA was spliced that way.

Run from the Sapal-Speke-RP-2026 folder:  python autoregulation.py
"""

import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from describe_data import load_matrix, gene_of


# --- Settings (change these, not the code below) ---------------------------
GENE = "SRSF3"                 # gene name or G locus ID. Known autoregulated genes in the panel include
                               # SRSF1, SRSF2, SRSF3, SRSF5, SRSF7 (Lareau 2007), SFPQ, PTBP1 (Pervouchine 2019)
NMD_DISTANCE = 50              # stop codon more than this many nt before the last junction -> NMD
UNPRODUCTIVE = ["NMD", "no CDS"]   # which classes count as unproductive
MIN_BASELINE_CPM = 10          # baseline genes: mean CPM per sample at least this
MIN_BASELINE_UNPRODUCTIVE = 0.05   # baseline genes: mean unproductive fraction at least this

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
FIGURE_FOLDER = "Figures/autoregulation"
RESULTS_FOLDER = "Results/autoregulation"
REGION_COLOURS = {"DLPFC": "#2a78d6", "Hippocampus": "#eb6834", "Caudate": "#1baf7a"}


# ---------------------------------------------------------------------------
# 1. Read the GTF and classify isoforms
# ---------------------------------------------------------------------------

def read_gtf(gtf_path):
    """Exons, CDS pieces, strand and gene name for every transcript.
    GTF columns used: 2 = feature (transcript / exon / CDS), 3-4 = start-end, 6 = strand, 8 = attributes."""
    exons, cds, strand, gene_name = defaultdict(list), defaultdict(list), {}, {}
    with open(gtf_path) as file:
        for line in file:
            columns = line.split("\t")
            if len(columns) < 9:
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', columns[8]).group(1)
            start, end = int(columns[3]), int(columns[4])
            if columns[2] == "exon":
                exons[transcript_id].append((start, end))
            elif columns[2] == "CDS":
                cds[transcript_id].append((start, end))
            elif columns[2] == "transcript":
                strand[transcript_id] = columns[6]
                name = re.search(r'gene_name "([^"]+)"', columns[8])
                gene_name[transcript_id] = name.group(1) if name else ""
    return exons, cds, strand, gene_name


def position_in_transcript(genome_position, exons_in_order, strand):
    """Convert a genome coordinate into a position along the spliced transcript (1 = first base).
    Walk through the exons in transcript order, adding up exon lengths until we reach the position."""
    passed = 0
    for start, end in exons_in_order:
        if start <= genome_position <= end:
            offset = genome_position - start if strand == "+" else end - genome_position
            return passed + offset + 1
        passed += end - start + 1
    return None


def classify_isoform(exons, cds, strand):
    """productive / NMD / no CDS, using the 50-nt rule.

    The cell marks each exon-exon junction while splicing. During the first round of translation
    the ribosome removes the marks it passes. If it stops (stop codon) more than ~50 nt before
    the last junction, a mark is left behind and the RNA is destroyed by NMD."""
    if not cds:
        return "no CDS"
    if len(exons) == 1:
        return "productive"        # no junctions, so nothing can be left behind

    # Exons in transcript order: left to right on + strand, right to left on - strand
    exons_in_order = sorted(exons, reverse=(strand == "-"))

    # The stop codon is at the end of the CDS: its highest coordinate on +, lowest on -
    stop_end = max(end for _, end in cds) if strand == "+" else min(start for start, _ in cds)
    stop_position = position_in_transcript(stop_end, exons_in_order, strand)

    # Last junction = where the last exon begins in transcript coordinates
    transcript_length = sum(end - start + 1 for start, end in exons)
    last_exon = exons_in_order[-1]
    last_junction = transcript_length - (last_exon[1] - last_exon[0] + 1)

    if stop_position is not None and last_junction - stop_position > NMD_DISTANCE:
        return "NMD"
    return "productive"


# ---------------------------------------------------------------------------
# 3. Feedback signatures
# ---------------------------------------------------------------------------

def remove_region(values, region):
    """Subtract each region's average, so only the variation WITHIN regions is left."""
    return values - values.groupby(region).transform("mean")


def signatures(productive, unproductive, region):
    """Feedback signatures for one gene, from per-sample productive and unproductive CPM.
    Uses log10(CPM + 1) so that samples with 0 still work. Returns a dict of numbers."""
    total = productive + unproductive
    log_p, log_t = np.log10(productive + 1), np.log10(total + 1)
    fraction = unproductive / total.replace(0, np.nan)
    result = {"mean_total_cpm": total.mean(), "mean_unproductive_fraction": fraction.mean()}

    for label, x, y, f in [("all", log_t, log_p, fraction),
                           ("within_region", remove_region(log_t, region), remove_region(log_p, region),
                            remove_region(fraction, region))]:
        # a) sensitivity S = least-squares slope of log productive on log total
        result[f"S_{label}"] = np.polyfit(x, y, 1)[0]
        # b) does the unproductive fraction rise with the total?
        keep = f.notna()
        result[f"fraction_vs_total_rho_{label}"] = spearmanr(x[keep], f[keep])[0]
        # c) variability: standard deviation of the logs (log scale -> comparable to a CV)
        result[f"sd_ratio_{label}"] = y.std() / x.std()   # < 1 = productive varies less than total
    return result


# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------

def plot_gene(per_sample, name, result, baseline):
    """Left: log productive vs log total (slope = S). Middle: unproductive fraction vs total.
    Right: where this gene's S sits among the baseline genes."""
    colours = [REGION_COLOURS[r] for r in per_sample["region"]]
    log_t = np.log10(per_sample["total_cpm"] + 1)
    log_p = np.log10(per_sample["productive_cpm"] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(log_t, log_p, c=colours, s=25)
    line = np.linspace(log_t.min(), log_t.max(), 10)
    slope, intercept = np.polyfit(log_t, log_p, 1)
    ax.plot(line, slope * line + intercept, color="black", label=f"fit: S = {slope:.2f}")
    # Reference: what S = 1 (no buffering) would look like through the middle of the data
    ax.plot(line, line - log_t.mean() + log_p.mean(), color="grey", linestyle="--", label="S = 1 (no buffering)")
    ax.set_xlabel(f"log10 total {name} CPM")
    ax.set_ylabel(f"log10 productive {name} CPM")
    ax.set_title("a) Sensitivity of productive RNA to total")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(per_sample["total_cpm"], per_sample["unproductive_fraction"], c=colours, s=25)
    ax.set_xscale("log")
    ax.set_xlabel(f"total {name} CPM (log scale)")
    ax.set_ylabel("unproductive fraction")
    ax.set_title(f"b) Unproductive fraction vs total\n"
                 f"rho = {result['fraction_vs_total_rho_all']:.2f} "
                 f"(within region {result['fraction_vs_total_rho_within_region']:.2f})")
    for region, colour in REGION_COLOURS.items():
        ax.scatter([], [], color=colour, label=region)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(baseline["S_within_region"].dropna(), bins=40, color="#b0b0b0", edgecolor="white")
    ax.axvline(result["S_within_region"], color="black", linewidth=2, label=f"{name}")
    ax.axvline(1, color="grey", linestyle="--", label="S = 1")
    ax.set_xlabel("S (within region)")
    ax.set_ylabel("number of genes")
    ax.set_title(f"c) {name} vs {len(baseline)} genes with unproductive isoforms")
    ax.legend(fontsize=8)

    fig.suptitle(f"Autoregulation signatures for {name} (45 adult samples)")
    plt.tight_layout()
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    path = os.path.join(FIGURE_FOLDER, f"{name}_autoregulation_{timestamp()}.png")
    fig.savefig(path, dpi=200)
    print(f"Saved figure: {path}")
    plt.show()


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def save_csv(table, name):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    path = os.path.join(RESULTS_FOLDER, f"{name}_{timestamp()}.csv")
    table.to_csv(path, index=False)
    print(f"Saved table: {path}")


def main():
    # ---- Load, adults only ----
    df = load_matrix(MATRIX_PATH)
    df = df[[name for name in df.columns if "_Fetal_" not in name]]
    df = df[(df > 0).any(axis=1)]
    region = pd.Series([name.split("_")[0] for name in df.columns], index=df.columns)
    loci = pd.Series([gene_of(i) for i in df.index], index=df.index)

    # ---- 1. Classify every isoform ----
    exons, cds, strand, gene_name = read_gtf(GTF_PATH)
    classes = pd.Series({i: classify_isoform(exons[i], cds.get(i, []), strand[i]) for i in df.index})
    unproductive = classes.isin(UNPRODUCTIVE)
    print("Isoform classes (all genes):")
    print(classes.value_counts().to_string())

    # G locus -> gene name (most common name among its transcripts)
    names = pd.Series(gene_name).reindex(df.index).groupby(loci).agg(lambda n: n.value_counts().index[0])
    locus = GENE if re.fullmatch(r"G\d+", GENE) else names[names == GENE].index[0]
    name = names[locus]
    print(f"\nGene: {name} ({locus})")

    # ---- 2. Productive and unproductive CPM per sample ----
    productive_cpm = df[~unproductive].groupby(loci[~unproductive]).sum()
    unproductive_cpm = df[unproductive].groupby(loci[unproductive]).sum()
    productive_cpm = productive_cpm.reindex(loci.unique(), fill_value=0)
    unproductive_cpm = unproductive_cpm.reindex(loci.unique(), fill_value=0)

    isoforms = df[loci == locus]
    isoform_table = pd.DataFrame({
        "isoform": isoforms.index,
        "class": classes[isoforms.index].to_numpy(),
        "mean_cpm": isoforms.mean(axis=1).to_numpy(),
        "mean_share_of_gene": (isoforms / isoforms.sum()).mean(axis=1).to_numpy(),
    }).sort_values("mean_cpm", ascending=False)
    print(f"{len(isoform_table)} isoforms; share of {name} expression by class:")
    print(isoform_table.groupby("class")["mean_share_of_gene"].sum().round(3).to_string())
    print("\nMost abundant isoforms:")
    print(isoform_table.head(10).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    per_sample = pd.DataFrame({
        "sample": df.columns, "region": region.to_numpy(),
        "productive_cpm": productive_cpm.loc[locus].to_numpy(),
        "unproductive_cpm": unproductive_cpm.loc[locus].to_numpy()})
    per_sample["total_cpm"] = per_sample["productive_cpm"] + per_sample["unproductive_cpm"]
    per_sample["unproductive_fraction"] = per_sample["unproductive_cpm"] / per_sample["total_cpm"]

    # ---- 3. Signatures for this gene ----
    result = signatures(productive_cpm.loc[locus], unproductive_cpm.loc[locus], region)

    # ---- 4. Baseline: same numbers for every gene with unproductive isoforms ----
    rows = []
    for g in loci.unique():
        total = productive_cpm.loc[g] + unproductive_cpm.loc[g]
        fraction = (unproductive_cpm.loc[g] / total.replace(0, np.nan)).mean()
        if total.mean() >= MIN_BASELINE_CPM and fraction >= MIN_BASELINE_UNPRODUCTIVE \
                and (productive_cpm.loc[g] > 0).sum() >= 10:
            rows.append({"gene": g, "gene_name": names[g],
                         **signatures(productive_cpm.loc[g], unproductive_cpm.loc[g], region)})
    baseline = pd.DataFrame(rows)

    print(f"\nFeedback signatures for {name} (prediction under negative feedback in brackets),")
    print(f"with its percentile among {len(baseline)} genes with >= {MIN_BASELINE_UNPRODUCTIVE:.0%} unproductive RNA:")
    for key, prediction, low_is_feedback in [
            ("S_all", "< 1", True), ("S_within_region", "< 1", True),
            ("fraction_vs_total_rho_all", "> 0", False), ("fraction_vs_total_rho_within_region", "> 0", False),
            ("sd_ratio_all", "< 1", True), ("sd_ratio_within_region", "< 1", True)]:
        # Percentile = % of baseline genes that look LESS like feedback than this gene
        if low_is_feedback:
            percentile = 100 * (baseline[key] > result[key]).mean()
        else:
            percentile = 100 * (baseline[key] < result[key]).mean()
        print(f"  {key:38s} {result[key]:6.2f}   (feedback: {prediction})   "
              f"more feedback-like than {percentile:.0f}% of genes")
    print(f"  mean unproductive fraction: {result['mean_unproductive_fraction']:.2f}")

    # ---- 5. Save ----
    print()
    safe_name = re.sub(r"\W", "_", name)
    save_csv(isoform_table, f"{safe_name}_isoforms")
    save_csv(per_sample, f"{safe_name}_per_sample")
    save_csv(baseline, "baseline_all_genes")
    plot_gene(per_sample, name, result, baseline)


if __name__ == "__main__":
    main()
