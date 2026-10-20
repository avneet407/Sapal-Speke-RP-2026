"""
verification.py

Positive control: SLM2 (KHDRBS3) represses inclusion of NRXN1 AS4.
If the data recovers this, AS4 inclusion (PSI) should be negatively correlated with KHDRBS3 expression,
and more negatively than for most other alternative exons in the dataset (background comparison).

Reverse-direction check: KHDRBS3 regulates NRXN1 splicing, not the other way round, so
  forward: NRXN1 isoform usage should track KHDRBS3 total (beyond NRXN1's own total)
  reverse: KHDRBS3 isoform usage should NOT track NRXN1 total (beyond KHDRBS3's own total)
If forward >> reverse, the isoform-level asymmetry points the right way.

Run from the Sapal-Speke-RP-2026 folder: python verification.py
"""
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from as4_blast_check import read_gtf_features, overlapping, BLAST_HIT
from figure_utils import save_figure
from inference import partial_spearman

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
GTF_PATH = "Long_Read_Data/expressed_isoforms.gtf"
REGULATOR = "G68150"   # KHDRBS3 (SLM2)
TARGET = "G35694"      # NRXN1
REGIONS = ["DLPFC", "Hippocampus", "Caudate"]
REGION_COLOURS = {"DLPFC": '#2a78d6', "Hippocampus": '#eb6834', "Caudate": "#1baf7a"}

# Reverse-direction check
N_ISOFORMS = 6          # top isoforms (by mean adult CPM) tested per gene - same number on both sides,
                        # so neither direction gets more chances to find a large rho by luck
N_PERMUTATIONS = 1000   # shuffles for the permutation p-value (~20-30 s for 1000)
RANDOM_SEED = 0


def find_as4_exon():
    """AS4 = the NRXN1 exon overlapping the BLAST hit that is found in the most isoforms."""
    exons = read_gtf_features(features=("exon",))
    exons = exons[exons["gene_name"] == "NRXN1"]
    hits = overlapping(exons, *BLAST_HIT)
    counts = hits.groupby(["start", "end"])["transcript_id"].nunique()
    return counts.idxmax()


def read_all_exons():
    """Every exon in the GTF, grouped by transcript: {transcript_id: sorted [(start, end), ...]}."""
    exon_lists = defaultdict(list)
    with open(GTF_PATH) as gtf:
        for line in gtf:
            fields = line.split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            transcript_id = re.search(r'transcript_id "([^"]+)"', fields[8]).group(1)
            exon_lists[transcript_id].append((int(fields[3]), int(fields[4])))
    return {t: sorted(e) for t, e in exon_lists.items()}


def split_by_exon(exon, transcripts, exon_lists):
    """Isoforms that include the exon, and isoforms that skip it (exons on both sides but not this one)."""
    including, skipping = [], []
    for t in transcripts:
        e = exon_lists[t]
        if exon in e:
            including.append(t)
        elif any(a[1] < exon[0] and b[0] > exon[1] for a, b in zip(e, e[1:])):
            skipping.append(t)
    return including, skipping


def calculate_psi(df, including, skipping):
    """PSI per sample = including CPM / (including + skipping CPM)."""
    included_cpm = df.loc[including].sum()
    skipped_cpm = df.loc[skipping].sum()
    return included_cpm / (included_cpm + skipped_cpm)


def rho_or_nan(x, y, min_samples=5):
    """Spearman rho, skipping NaN samples; NaN if too few samples are left."""
    usable = x.notna() & y.notna()
    if usable.sum() < min_samples:
        return np.nan
    return spearmanr(x[usable], y[usable])[0]


def within_region_rho(x, y, region, adult):
    """Average of the Spearman rho calculated separately in each brain region (adults only)."""
    rhos = [rho_or_nan(x[(region == name) & adult], y[(region == name) & adult]) for name in REGIONS]
    return np.nanmean(rhos) if not np.all(np.isnan(rhos)) else np.nan


def plot_correlations(khdrbs3, psi, region, adult):
    """KHDRBS3 CPM vs AS4 PSI: all adults (fetal shown as grey crosses, not in stats), then each region."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for name, colour in REGION_COLOURS.items():
        in_region = (region == name) & adult
        axes[0].scatter(khdrbs3[in_region], psi[in_region], color=colour, s=20, label=name)
    axes[0].scatter(khdrbs3[~adult], psi[~adult], color="grey", marker="x", s=30, label="Fetal (excluded)")
    rho, p_value = spearmanr(khdrbs3[adult], psi[adult], nan_policy="omit")
    axes[0].set_title(f"All adults\nrho = {rho:.2f}, p = {p_value:.2g}, n = {adult.sum()}")
    axes[0].legend(fontsize=8)

    for ax, name in zip(axes[1:], REGIONS):
        in_region = (region == name) & adult
        ax.scatter(khdrbs3[in_region], psi[in_region], color=REGION_COLOURS[name], s=20)
        rho, p_value = spearmanr(khdrbs3[in_region], psi[in_region], nan_policy="omit")
        ax.set_title(f"{name}\nrho = {rho:.2f}, p = {p_value:.2g}, n = {in_region.sum()}")

    for ax in axes:
        ax.set_xlabel("KHDRBS3 CPM (all isoforms)")
    axes[0].set_ylabel("NRXN1 AS4 PSI (fraction including AS4)")

    fig.suptitle("SLM2 (KHDRBS3) vs NRXN1 AS4 inclusion (expect negative rho)")
    plt.tight_layout()
    save_figure(fig, "verification", "KHDRBS3_vs_NRXN1_AS4_PSI")


# ---- Reverse-direction check ----

def isoform_shares(df, gene, n=N_ISOFORMS):
    """% share of the gene's CPM for each of its top n isoforms (by mean CPM), plus the gene total.
    Samples where the gene is 0 give NaN shares, which partial_spearman skips."""
    isoforms = df[[t.split(".")[0] == gene for t in df.index]]
    total = isoforms.sum()
    top = isoforms.mean(axis=1).sort_values(ascending=False).index[:n]
    return {isoform: 100 * isoforms.loc[isoform] / total.replace(0, np.nan) for isoform in top}, total


def direction_table(features, own_total, other_total, region, direction):
    """Partial Spearman of each isoform feature (share or PSI) vs the OTHER gene's total,
    controlling for its OWN gene's total - without and with brain region as an extra control."""
    rows = []
    for name, feature in features.items():
        rho, p_value, n = partial_spearman(feature, other_total, own_total)
        rho_region, p_region, _ = partial_spearman(feature, other_total, own_total, region)
        rows.append({"direction": direction, "feature": name,
                     "partial_rho": rho, "partial_p": p_value,
                     "partial_rho_region": rho_region, "partial_p_region": p_region, "n_samples": n})
    return pd.DataFrame(rows)


def max_rho_permutation(features, own_total, other_total, region, rng):
    """Largest |partial rho| (region-controlled) over the isoforms, and how often chance beats it.

    Null: shuffle the other gene's total across samples WITHIN each region, which breaks any real
    link but keeps region differences. Recompute the largest |rho| each time. Taking the max over
    the isoforms in both the real data and every shuffle accounts for testing several isoforms.
    p = (1 + shuffles >= observed) / (1 + N_PERMUTATIONS)."""
    def largest_abs_rho(other):
        return max(abs(partial_spearman(f, other, own_total, region)[0]) for f in features.values())

    observed = largest_abs_rho(other_total)
    null = [largest_abs_rho(other_total.groupby(region).transform(lambda v: rng.permutation(v.to_numpy())))
            for _ in range(N_PERMUTATIONS)]
    return observed, (1 + sum(value >= observed for value in null)) / (1 + N_PERMUTATIONS)


def reverse_direction_check(df, psi, region):
    """Forward  (expected): NRXN1 isoform usage (AS4 PSI + top isoform shares) vs KHDRBS3 total.
       Reverse (expected ~0): KHDRBS3 top isoform shares vs NRXN1 total.
    Adults only; df, psi and region must already be restricted to adult samples."""
    target_shares, target_total = isoform_shares(df, TARGET)
    regulator_shares, regulator_total = isoform_shares(df, REGULATOR)

    forward_features = {"AS4 PSI": psi, **target_shares}
    forward = direction_table(forward_features, target_total, regulator_total, region,
                              "forward: NRXN1 isoforms vs KHDRBS3 total")
    reverse = direction_table(regulator_shares, regulator_total, target_total, region,
                              "reverse: KHDRBS3 isoforms vs NRXN1 total")

    # Permutation test on the isoform shares only (same N_ISOFORMS each side, AS4 PSI left out)
    rng = np.random.default_rng(RANDOM_SEED)
    forward_max, forward_p = max_rho_permutation(target_shares, target_total, regulator_total, region, rng)
    reverse_max, reverse_p = max_rho_permutation(regulator_shares, regulator_total, target_total, region, rng)
    summary = {"forward": (forward_max, forward_p), "reverse": (reverse_max, reverse_p)}
    return forward, reverse, summary


def plot_direction_check(forward, reverse, summary):
    """Bar per isoform feature: region-controlled partial rho. Forward on the left, reverse on the right.
    If the method recovers direction, the left panel should have the long bars and the right should not."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, table, key, colour in [(axes[0], forward, "forward", "#2a78d6"), (axes[1], reverse, "reverse", "#eb6834")]:
        labels = table["feature"][::-1]
        bars = ax.barh(labels, table["partial_rho_region"][::-1], color=colour)
        for bar, p_value in zip(bars, table["partial_p_region"][::-1]):
            x = bar.get_width()
            ax.text(x + (0.02 if x >= 0 else -0.02), bar.get_y() + bar.get_height() / 2, f"p={p_value:.2g}",
                    va="center", ha="left" if x >= 0 else "right", fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Partial Spearman rho (controlling for own gene total + region)")
        largest, p_perm = summary[key]
        ax.set_title(f"{table['direction'].iloc[0]}\nmax |rho| over isoforms = {largest:.2f}, "
                     f"permutation p = {p_perm:.3g}", fontsize=9)

    fig.suptitle("Reverse-direction check (adults): expect a signal on the left, none on the right")
    plt.tight_layout()
    save_figure(fig, "verification", "KHDRBS3_NRXN1_direction_check")


# ---- Background comparison ----

def internal_exons(transcripts, exon_lists):
    """Exons of one gene that are never the first or last exon of any of its isoforms."""
    internal, terminal = set(), set()
    for t in transcripts:
        e = exon_lists[t]
        terminal.update([e[0], e[-1]])
        internal.update(e[1:-1])
    return internal - terminal


def background_correlations(df, exon_lists, khdrbs3, region, adult, min_cpm=5):
    """Repeat the KHDRBS3 vs PSI test for every alternative internal exon in the dataset.
    Keeps exons where including and skipping isoforms both average >= min_cpm across adults."""
    transcripts_by_gene = defaultdict(list)
    for t in df.index:
        if t in exon_lists:
            transcripts_by_gene[t.split(".")[0]].append(t)

    rows = []
    for gene, transcripts in transcripts_by_gene.items():
        for exon in internal_exons(transcripts, exon_lists):
            including, skipping = split_by_exon(exon, transcripts, exon_lists)
            if not including or not skipping:
                continue
            if df.loc[including, adult].sum().mean() < min_cpm or df.loc[skipping, adult].sum().mean() < min_cpm:
                continue
            psi = calculate_psi(df, including, skipping)
            rows.append({"gene": gene, "start": exon[0], "end": exon[1],
                         "rho_adults": rho_or_nan(khdrbs3[adult], psi[adult]),
                         "rho_within_region": within_region_rho(khdrbs3, psi, region, adult)})
    return pd.DataFrame(rows)


def plot_background(background, as4_rho_adults, as4_rho_within):
    """Histogram of rho for all other alternative exons, with NRXN1 AS4 marked in red."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, column, as4_rho, label in [(axes[0], "rho_adults", as4_rho_adults, "all adults"),
                                       (axes[1], "rho_within_region", as4_rho_within, "average within region")]:
        values = background[column].dropna()
        percent = (values <= as4_rho).mean() * 100
        ax.hist(values, bins=40, color="lightgrey", edgecolor="white")
        ax.axvline(as4_rho, color="red", linewidth=2, label=f"NRXN1 AS4 (rho = {as4_rho:.2f})")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Spearman rho: KHDRBS3 CPM vs exon PSI ({label})")
        ax.set_ylabel("Number of exons")
        ax.set_title(f"{len(values)} alternative exons; {percent:.1f}% as negative as AS4")
        ax.legend(fontsize=8)

    fig.suptitle("Is NRXN1 AS4 unusual among alternative exons?")
    plt.tight_layout()
    save_figure(fig, "verification", "KHDRBS3_exon_PSI_background")


def main():
    # Check the BLAST hit and identify AS4
    as4 = find_as4_exon()
    print(f"BLAST hit chr2:{BLAST_HIT[0]}-{BLAST_HIT[1]} -> AS4 exon chr2:{as4[0]}-{as4[1]} ({as4[1] - as4[0] + 1} nt)")

    # AS4 inclusion (PSI) per sample
    df = pd.read_csv(MATRIX_PATH, sep="\t", index_col=0)
    exon_lists = read_all_exons()
    nrxn1_transcripts = [t for t in df.index if t.split(".")[0] == TARGET and t in exon_lists]
    including, skipping = split_by_exon(as4, nrxn1_transcripts, exon_lists)
    print(f"{len(including)} isoforms include AS4, {len(skipping)} skip it")
    psi = calculate_psi(df, including, skipping)

    # KHDRBS3 total expression per sample
    khdrbs3 = df[[t.split(".")[0] == REGULATOR for t in df.index]].sum()

    # Spearman correlation: adults only (fetal samples excluded), then within each region
    adult = pd.Series(["Fetal" not in sample for sample in df.columns], index=df.columns)
    region = pd.Series([sample.split("_")[0] for sample in df.columns], index=df.columns)

    print("\nKHDRBS3 CPM vs AS4 PSI (expect negative rho)")
    rho, p_value = spearmanr(khdrbs3[adult], psi[adult], nan_policy="omit")
    print(f"  All adults: rho = {rho:.2f}, p = {p_value:.2g}, n = {adult.sum()}")
    for name in REGIONS:
        in_region = (region == name) & adult
        rho, p_value = spearmanr(khdrbs3[in_region], psi[in_region], nan_policy="omit")
        print(f"  {name}: rho = {rho:.2f}, p = {p_value:.2g}, n = {in_region.sum()}")

    plot_correlations(khdrbs3, psi, region, adult)

    # Reverse-direction check (adults only)
    print(f"\nReverse-direction check: top {N_ISOFORMS} isoforms per gene, {N_PERMUTATIONS} permutations "
          f"(expected ~20-30 s)...")
    forward, reverse, summary = reverse_direction_check(df.loc[:, adult], psi[adult], region[adult])
    for table in (forward, reverse):
        print(f"\n  {table['direction'].iloc[0]}")
        print(table.drop(columns="direction").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    for key, (largest, p_perm) in summary.items():
        print(f"  {key}: max |partial rho| (region-controlled) = {largest:.2f}, permutation p = {p_perm:.3g}")
    plot_direction_check(forward, reverse, summary)

    # Background: same test for every other alternative exon in the dataset
    print("\nCalculating background over all alternative exons (may take a minute)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # exons with constant PSI in a region give harmless warnings
        background = background_correlations(df, exon_lists, khdrbs3, region, adult)
        as4_rho_adults = rho_or_nan(khdrbs3[adult], psi[adult])
        as4_rho_within = within_region_rho(khdrbs3, psi, region, adult)

    is_as4 = (background["gene"] == TARGET) & (background["start"] == as4[0]) & (background["end"] == as4[1])
    background = background[~is_as4]

    print(f"Background: {len(background)} alternative exons")
    for column, as4_rho in [("rho_adults", as4_rho_adults), ("rho_within_region", as4_rho_within)]:
        values = background[column].dropna()
        print(f"  {column}: AS4 rho = {as4_rho:.2f}, {(values <= as4_rho).mean() * 100:.1f}% of exons "
              f"are as negative (background median {values.median():.2f})")

    plot_background(background, as4_rho_adults, as4_rho_within)
    plt.show()


if __name__ == "__main__":
    main()
