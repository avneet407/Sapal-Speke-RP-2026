"""
Calculate isoform-share to partner-gene correlations for BY-significant gene pairs.

The input is the filtered CSV produced by filter_gene_gene_by.py. Genes are filtered
using CPM > MIN_CPM in at least MIN_FRACTION_SAMPLES of adult samples, then isoforms
are retained when their mean share of the gene is at least MIN_SHARE.

For each retained gene pair A-B, the output contains:
  - the original A-vs-B gene-total correlation;
  - each A isoform's percentage share vs B total CPM;
  - each B isoform's percentage share vs A total CPM.

Each pair is also represented as a matrix with B components as rows and A components
as columns. Isoform-to-isoform correlations are not calculated in this version; those
cells are blank and can be added later without changing the output matrix layout.

Run from the repository root:
    python filter_isoform_gene.py
"""

from hashlib import sha256
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from describe_data import gene_of, load_matrix


# --- Settings ---------------------------------------------------------------
INPUT_FILE = None
# None selects the latest raw-p-value BY output from filter_gene_gene_by.py.
MIN_CPM = 1
MIN_FRACTION_SAMPLES = 0.8
MIN_SHARE = 0.015
N_EXAMPLE_PAIRS = 3
RANDOM_SEED = 2026
MHT_METHOD = "by"  # "by" or "bh"; correction and threshold do not invalidate raw correlations
FDR_THRESHOLD = 0.05
PROGRESS_INTERVAL_SECONDS = 15
CACHE_VERSION = 1

MATRIX_PATH = "Long_Read_Data/expressed_isoforms_matrix.tsv"
ROOT_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = ROOT_FOLDER / "Results" / "gene_gene_mht"
OUTPUT_FOLDER = ROOT_FOLDER / "Results" / "isoform_gene"
CACHE_FOLDER = OUTPUT_FOLDER / "cache"


def find_input_file():
    """Return an explicitly configured input or the latest raw-p BY-filtered CSV."""
    root = Path(__file__).resolve().parent
    if INPUT_FILE is not None:
        path = Path(INPUT_FILE)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured gene-pair CSV does not exist: {path}")
        return path

    candidates = list(INPUT_FOLDER.glob("gene_pairs_*_BY*_p.csv"))
    candidates.extend(INPUT_FOLDER.glob("*_BY_FDR*_p_value_significant.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No raw-p BY-filtered gene-pair CSV found in "
            f"{INPUT_FOLDER}. Run filter_gene_gene_by.py first or set INPUT_FILE."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def correlation(x, y):
    """Return Spearman rho, nominal p-value, and pairwise finite sample count."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n_samples = int(valid.sum())
    if n_samples < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan, np.nan, n_samples
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue), n_samples


def benjamini_yekutieli_adjust(p_values):
    """Return BY-adjusted p-values in input order, leaving missing values missing."""
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")
    if not finite_values.size:
        raise ValueError("No finite isoform-gene p-values are available to adjust.")

    n_tests = finite_values.size
    order = np.argsort(finite_values)
    sorted_p_values = finite_values[order]
    harmonic_number = np.sum(1.0 / np.arange(1, n_tests + 1))
    ranked_adjusted = (
        sorted_p_values * n_tests * harmonic_number / np.arange(1, n_tests + 1)
    )
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]

    adjusted = np.full(values.shape, np.nan, dtype=float)
    adjusted_finite = np.empty(n_tests, dtype=float)
    adjusted_finite[order] = np.minimum(ranked_adjusted, 1.0)
    adjusted[finite] = adjusted_finite
    return adjusted


def benjamini_hochberg_adjust(p_values):
    """Return BH-adjusted p-values in input order, leaving missing values missing."""
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")
    if not finite_values.size:
        raise ValueError("No finite isoform-gene p-values are available to adjust.")

    n_tests = finite_values.size
    order = np.argsort(finite_values)
    sorted_p_values = finite_values[order]
    ranked_adjusted = sorted_p_values * n_tests / np.arange(1, n_tests + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]

    adjusted = np.full(values.shape, np.nan, dtype=float)
    adjusted_finite = np.empty(n_tests, dtype=float)
    adjusted_finite[order] = np.minimum(ranked_adjusted, 1.0)
    adjusted[finite] = adjusted_finite
    return adjusted


def adjust_p_values(p_values, method):
    """Dispatch an FDR adjustment by method name."""
    if method == "by":
        return benjamini_yekutieli_adjust(p_values)
    if method == "bh":
        return benjamini_hochberg_adjust(p_values)
    raise ValueError("MHT_METHOD must be 'by' or 'bh'.")


def file_fingerprint(path):
    """Identify an input by its resolved path, byte size, and nanosecond mtime."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def content_fingerprint(path):
    """Identify a results CSV by content so renaming it does not invalidate its cache."""
    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def make_cache_paths(input_path, matrix_path):
    """Build raw-correlation cache paths, deliberately excluding MHT settings."""
    cache_settings = {
        "version": CACHE_VERSION,
        "input": content_fingerprint(input_path),
        "matrix": file_fingerprint(matrix_path),
        "min_cpm": MIN_CPM,
        "min_fraction_samples": MIN_FRACTION_SAMPLES,
        "min_share": MIN_SHARE,
        "exclude_fetal": True,
        "correlation": "spearman_isoform_share_vs_partner_gene_total",
    }
    key = sha256(
        json.dumps(cache_settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_FOLDER / f"isoform_gene_raw_correlations_{key}.csv",
        CACHE_FOLDER / f"isoform_gene_raw_correlations_{key}.json",
        cache_settings,
    )


def report_progress(label, completed, total, started_at, last_report_at):
    """Print progress no less often than the configured interval during pair calculations."""
    now = time.monotonic()
    if completed == total or now - last_report_at >= PROGRESS_INTERVAL_SECONDS:
        elapsed = now - started_at
        print(
            f"{label}: {completed:,}/{total:,} gene pairs "
            f"({completed / total:.1%}); elapsed {elapsed:.0f}s",
            flush=True
        )
        return now
    return last_report_at


def add_result(
    results,
    pair,
    component_a,
    component_b,
    component_a_type,
    component_b_type,
    rho,
    p_value,
    n_samples,
    isoform_a_id="",
    isoform_b_id="",
    mean_share_a_pct=np.nan,
    mean_share_b_pct=np.nan,
):
    results.append({
        "pair_id": f"{pair.gene_a}__{pair.gene_b}",
        "gene_a": pair.gene_a,
        "gene_a_name": pair.gene_a_name,
        "gene_b": pair.gene_b,
        "gene_b_name": pair.gene_b_name,
        "component_a": component_a,
        "component_a_type": component_a_type,
        "isoform_a_id": isoform_a_id,
        "mean_share_a_pct": mean_share_a_pct,
        "component_b": component_b,
        "component_b_type": component_b_type,
        "isoform_b_id": isoform_b_id,
        "mean_share_b_pct": mean_share_b_pct,
        "spearman_rho": rho,
        "p_value": p_value,
        "n_samples": n_samples,
        "gene_pair_spearman_rho": pair.spearman_rho,
        "gene_pair_p_value": pair.p_value,
        "gene_pair_by_adjusted_p_value": pair.by_adjusted_p_value,
        "gene_pair_by_significant": getattr(pair, "by_significant", np.nan),
        "correction_family": (
            "upstream_gene_gene"
            if component_a_type == "gene_cpm" and component_b_type == "gene_cpm"
            else "all_isoform_gene_tests"
        ),
    })


def plot_pair_matrix(
    pair, rho_matrix, adjusted_p_matrix, row_labels, column_labels, path
):
    """Save a matrix coloured by adjusted p-value and labelled with rho and p."""
    n_rows, n_columns = rho_matrix.shape
    fig, ax = plt.subplots(
        figsize=(max(8, 0.42 * n_columns + 3), max(6, 0.36 * n_rows + 2))
    )
    colour_map = plt.get_cmap("RdYlGn").copy()
    colour_map.set_bad("#eeeeee")
    image = ax.imshow(
        np.ma.masked_invalid(adjusted_p_matrix),
        cmap=colour_map,
        vmin=0,
        vmax=1,
        aspect="auto"
    )
    fig.colorbar(
        image,
        ax=ax,
        label=f"{MHT_METHOD.upper()}-adjusted p-value (colour)"
    )

    ax.set_xticks(np.arange(n_columns), labels=column_labels)
    ax.set_yticks(np.arange(n_rows), labels=row_labels)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(f"A components: {pair.gene_a_name} ({pair.gene_a})")
    ax.set_ylabel(f"B components: {pair.gene_b_name} ({pair.gene_b})")
    ax.set_title(
        f"Gene pair {pair.gene_a_name} ({pair.gene_a}) vs "
        f"{pair.gene_b_name} ({pair.gene_b})\n"
        f"BY-adjusted gene-pair p = {pair.by_adjusted_p_value:.3g}"
    )

    for row_index in range(n_rows):
        for column_index in range(n_columns):
            rho = rho_matrix[row_index, column_index]
            adjusted_p = adjusted_p_matrix[row_index, column_index]
            if np.isfinite(rho):
                p_label = f"{adjusted_p:.2g}" if np.isfinite(adjusted_p) else "NA"
                text_colour = "white" if np.isfinite(adjusted_p) and adjusted_p < 0.01 else "black"
                ax.text(
                    column_index,
                    row_index,
                    f"rho={rho:.2f}\n{MHT_METHOD.upper()} p={p_label}",
                    ha="center",
                    va="center",
                    color=text_colour,
                    fontsize=7
                )

    ax.text(
        0,
        -0.16,
        f"Cells show rho and {MHT_METHOD.upper()}-adjusted p. Colour encodes adjusted p "
        "(red = smaller/more significant; green = larger). A x B uses the upstream "
        "gene-gene BY p-value; all "
        "isoform-vs-gene tests share one BY correction. Isoform labels follow descending "
        "mean CPM. Blank cells are isoform-to-isoform correlations, not calculated here. "
        "See the CSV for the isoform-ID mapping.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        wrap=True
    )
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    if MIN_CPM < 0:
        raise ValueError("MIN_CPM must be non-negative.")
    if not 0 <= MIN_FRACTION_SAMPLES <= 1:
        raise ValueError("MIN_FRACTION_SAMPLES must be between 0 and 1.")
    if not 0 <= MIN_SHARE <= 1:
        raise ValueError("MIN_SHARE must be between 0 and 1.")
    if not 0 <= FDR_THRESHOLD <= 1:
        raise ValueError("FDR_THRESHOLD must be between 0 and 1.")
    if MHT_METHOD not in {"by", "bh"}:
        raise ValueError("MHT_METHOD must be 'by' or 'bh'.")

    input_path = find_input_file()
    pairs = pd.read_csv(input_path)
    required_columns = {
        "gene_a", "gene_a_name", "gene_b", "gene_b_name", "spearman_rho",
        "p_value", "by_adjusted_p_value", "by_significant"
    }
    missing_columns = required_columns.difference(pairs.columns)
    if missing_columns:
        raise ValueError(
            f"Input file {input_path} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if pairs.empty:
        raise ValueError(f"Input gene-pair dataset is empty: {input_path}")
    pairs[["gene_a_name", "gene_b_name"]] = pairs[
        ["gene_a_name", "gene_b_name"]
    ].fillna("")
    if pairs.duplicated(["gene_a", "gene_b"]).any():
        raise ValueError(f"Input has duplicate gene pairs: {input_path}")
    if "by_significant" in pairs and not pairs["by_significant"].all():
        raise ValueError(
            f"Input contains rows not marked BY-significant: {input_path}. "
            "Pass the filtered output from filter_gene_gene_by.py."
        )

    matrix_path = ROOT_FOLDER / MATRIX_PATH
    raw_cache_path, cache_manifest_path, cache_settings = make_cache_paths(
        input_path,
        matrix_path
    )
    settings_tag = (
        f"cpm{MIN_CPM:g}_frac{MIN_FRACTION_SAMPLES:.0%}_"
        f"share{MIN_SHARE * 100:g}pct"
    )
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    mht_tag = f"{MHT_METHOD.upper()}_FDR{FDR_THRESHOLD * 100:g}".replace(".", "p")
    output_path = OUTPUT_FOLDER / (
        f"isoform_gene_correlations_{raw_cache_path.stem[-12:]}_"
        f"{settings_tag}_{mht_tag}.csv"
    )

    cache_hit = raw_cache_path.is_file() and cache_manifest_path.is_file()
    if cache_hit:
        with cache_manifest_path.open(encoding="utf-8") as manifest_file:
            cached_settings = json.load(manifest_file)
        cache_hit = cached_settings == cache_settings

    if cache_hit:
        print(f"Loading cached raw correlations: {raw_cache_path}", flush=True)
        result_table = pd.read_csv(raw_cache_path)
        required_cache_columns = {
            "pair_id", "gene_a", "gene_a_name", "gene_b", "gene_b_name",
            "component_a", "component_a_type", "component_b", "component_b_type",
            "spearman_rho", "p_value", "n_samples", "gene_pair_by_adjusted_p_value",
            "correction_family"
        }
        missing_cache_columns = required_cache_columns.difference(result_table.columns)
        if missing_cache_columns:
            raise ValueError(
                f"Correlation cache is missing required columns: "
                f"{', '.join(sorted(missing_cache_columns))}"
            )
        print(f"Loaded {len(result_table):,} cached correlation rows.", flush=True)
    else:
        print("Correlation cache missing or stale; recalculating raw correlations.", flush=True)
        expression = load_matrix(str(matrix_path))
        expression = expression[
            [sample for sample in expression.columns if "_Fetal_" not in sample]
        ]
        if expression.empty:
            raise ValueError(f"No adult samples found in {matrix_path}.")

        isoform_gene = pd.Series(
            [gene_of(str(isoform_id)) for isoform_id in expression.index],
            index=expression.index
        )
        gene_cpm = expression.groupby(isoform_gene, sort=False).sum()
        fraction_expressed = (gene_cpm > MIN_CPM).mean(axis=1)
        kept_genes = gene_cpm.index[
            fraction_expressed >= MIN_FRACTION_SAMPLES
        ]

        kept_isoforms = expression.loc[isoform_gene.isin(kept_genes)]
        kept_isoform_gene = isoform_gene.loc[kept_isoforms.index]
        own_gene_cpm = gene_cpm.loc[kept_isoform_gene].set_axis(kept_isoforms.index)
        shares = kept_isoforms.div(own_gene_cpm.where(own_gene_cpm > 0)) * 100
        mean_share_pct = shares.mean(axis=1)
        share_filter = mean_share_pct >= MIN_SHARE * 100
        shares = shares.loc[share_filter]
        mean_share_pct = mean_share_pct.loc[share_filter]
        mean_isoform_cpm = kept_isoforms.mean(axis=1).loc[share_filter]
        isoform_gene = kept_isoform_gene.loc[share_filter]

        isoforms_by_gene = {}
        for gene_id in kept_genes:
            isoforms = isoform_gene.index[isoform_gene == gene_id]
            isoforms = sorted(
                isoforms,
                key=lambda isoform_id: (
                    -mean_isoform_cpm[isoform_id],
                    str(isoform_id)
                )
            )
            isoforms_by_gene[gene_id] = isoforms

        genes_in_input = set(pairs["gene_a"]).union(pairs["gene_b"])
        missing_genes = genes_in_input.difference(gene_cpm.index)
        if missing_genes:
            raise ValueError(
                "The expression matrix is missing genes from the filtered "
                f"interaction file: {sorted(missing_genes)[:10]}"
            )
        eligible_mask = (
            pairs["gene_a"].isin(kept_genes)
            & pairs["gene_b"].isin(kept_genes)
        )
        eligible_pairs = pairs.loc[eligible_mask].copy().reset_index(drop=True)
        if eligible_pairs.empty:
            raise ValueError(
                "No gene pairs remain after the gene-expression filter. "
                "Check MIN_CPM, MIN_FRACTION_SAMPLES, and the input analysis."
            )

        if "n_samples" in eligible_pairs:
            observed_sample_counts = set(
                eligible_pairs["n_samples"].dropna().astype(int)
            )
            if observed_sample_counts and observed_sample_counts != {len(expression.columns)}:
                raise ValueError(
                    "The filtered gene-pair results and expression matrix use "
                    f"different sample counts: {sorted(observed_sample_counts)} vs "
                    f"{len(expression.columns)} adult samples."
                )

        results = []
        total_pairs = len(eligible_pairs)
        started_at = time.monotonic()
        last_report_at = started_at
        for pair_number, pair in enumerate(
            eligible_pairs.itertuples(index=False),
            start=1
        ):
            if pair_number == 1:
                print(
                    f"Calculating raw isoform-gene correlations for "
                    f"{total_pairs:,} gene pairs.",
                    flush=True
                )
            last_report_at = report_progress(
                "Correlation progress",
                pair_number,
                total_pairs,
                started_at,
                last_report_at
            )

            add_result(
                results, pair, "A", "B", "gene_cpm", "gene_cpm",
                float(pair.spearman_rho), float(pair.p_value),
                int(pair.n_samples) if hasattr(pair, "n_samples") else len(expression.columns)
            )

            for index, isoform_id in enumerate(
                isoforms_by_gene[pair.gene_a],
                start=1
            ):
                rho, p_value, n_samples = correlation(
                    shares.loc[isoform_id], gene_cpm.loc[pair.gene_b]
                )
                add_result(
                    results, pair, f"A{index}", "B",
                    "isoform_percent_share", "gene_cpm",
                    rho, p_value, n_samples,
                    isoform_a_id=str(isoform_id),
                    mean_share_a_pct=float(mean_share_pct[isoform_id])
                )

            for index, isoform_id in enumerate(
                isoforms_by_gene[pair.gene_b],
                start=1
            ):
                rho, p_value, n_samples = correlation(
                    shares.loc[isoform_id], gene_cpm.loc[pair.gene_a]
                )
                add_result(
                    results, pair, "A", f"B{index}",
                    "gene_cpm", "isoform_percent_share",
                    rho, p_value, n_samples,
                    isoform_b_id=str(isoform_id),
                    mean_share_b_pct=float(mean_share_pct[isoform_id])
                )

        result_table = pd.DataFrame(results)
        temporary_cache = raw_cache_path.with_suffix(".csv.tmp")
        result_table.to_csv(temporary_cache, index=False)
        temporary_cache.replace(raw_cache_path)
        temporary_manifest = cache_manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(cache_settings, indent=2, sort_keys=True),
            encoding="utf-8"
        )
        temporary_manifest.replace(cache_manifest_path)
        print(f"Saved raw correlation cache: {raw_cache_path}", flush=True)

    if "gene_pair_by_significant" not in result_table.columns:
        upstream_significance = pairs.set_index(
            ["gene_a", "gene_b"]
        )["by_significant"].to_dict()
        gene_total_rows = (
            (result_table["component_a_type"] == "gene_cpm")
            & (result_table["component_b_type"] == "gene_cpm")
        )
        result_table["gene_pair_by_significant"] = pd.Series(
            index=result_table.index,
            dtype="object"
        )
        result_table.loc[gene_total_rows, "gene_pair_by_significant"] = [
            upstream_significance[(gene_a, gene_b)]
            for gene_a, gene_b in zip(
                result_table.loc[gene_total_rows, "gene_a"],
                result_table.loc[gene_total_rows, "gene_b"]
            )
        ]

    isoform_gene_tests = (
        (result_table["component_a_type"] == "isoform_percent_share")
        | (result_table["component_b_type"] == "isoform_percent_share")
    )
    correction_started_at = time.monotonic()
    print(
        f"Applying {MHT_METHOD.upper()} correction to "
        f"{int(result_table.loc[isoform_gene_tests, 'p_value'].notna().sum()):,} "
        "finite isoform-gene p-values.",
        flush=True
    )
    result_table["mht_adjusted_p_value"] = np.nan
    result_table.loc[~isoform_gene_tests, "mht_adjusted_p_value"] = (
        result_table.loc[~isoform_gene_tests, "gene_pair_by_adjusted_p_value"]
    )
    result_table.loc[isoform_gene_tests, "mht_adjusted_p_value"] = adjust_p_values(
        result_table.loc[isoform_gene_tests, "p_value"],
        MHT_METHOD
    )
    result_table["mht_method"] = np.where(
        isoform_gene_tests,
        MHT_METHOD.upper(),
        "BY_UPSTREAM"
    )
    result_table["mht_threshold"] = np.where(
        isoform_gene_tests,
        FDR_THRESHOLD,
        np.nan
    )
    result_table["mht_significant"] = False
    result_table.loc[~isoform_gene_tests, "mht_significant"] = (
        result_table.loc[~isoform_gene_tests, "gene_pair_by_significant"]
        .fillna(False)
        .astype(bool)
    )
    result_table.loc[isoform_gene_tests, "mht_significant"] = (
        result_table.loc[isoform_gene_tests, "mht_adjusted_p_value"] < FDR_THRESHOLD
    )
    print(
        f"FDR adjustment and thresholding completed in "
        f"{time.monotonic() - correction_started_at:.1f}s.",
        flush=True
    )
    if MHT_METHOD == "by":
        result_table["by_adjusted_p_value"] = result_table["mht_adjusted_p_value"]
        result_table["by_significant"] = result_table["mht_significant"]

    result_table.to_csv(output_path, index=False)
    print(f"Correlation rows saved: {len(result_table):,}")
    print(
        f"Isoform-gene p-values {MHT_METHOD.upper()}-adjusted: "
        f"{int(result_table.loc[isoform_gene_tests, 'p_value'].notna().sum()):,} finite tests; "
        f"{int(result_table.loc[isoform_gene_tests, 'mht_significant'].sum()):,} significant "
        f"at FDR {FDR_THRESHOLD:.1%}"
    )
    print(f"Used raw correlation cache: {raw_cache_path}", flush=True)
    print(f"Saved full correlation table: {output_path}")

    gene_pair_rows = result_table.loc[~isoform_gene_tests].drop_duplicates(
        ["gene_a", "gene_b"]
    )
    selected_pairs = gene_pair_rows.sample(
        n=min(N_EXAMPLE_PAIRS, len(gene_pair_rows)),
        random_state=RANDOM_SEED
    )
    for example_number, pair in enumerate(
        selected_pairs.itertuples(index=False),
        start=1
    ):
        pair_rows = result_table[
            (result_table["gene_a"] == pair.gene_a)
            & (result_table["gene_b"] == pair.gene_b)
        ]
        a_rows = pair_rows.loc[
            pair_rows["component_a_type"] == "isoform_percent_share"
        ].sort_values(
            "component_a",
            key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
        )
        b_rows = pair_rows.loc[
            pair_rows["component_b_type"] == "isoform_percent_share"
        ].sort_values(
            "component_b",
            key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
        )
        column_labels = ["A"] + a_rows["component_a"].tolist()
        row_labels = ["B"] + b_rows["component_b"].tolist()
        rho_matrix = np.full(
            (len(row_labels), len(column_labels)),
            np.nan,
            dtype=float
        )
        adjusted_p_matrix = np.full_like(rho_matrix, np.nan)
        gene_row = pair_rows.loc[
            (pair_rows["component_a"] == "A")
            & (pair_rows["component_b"] == "B")
        ].iloc[0]
        rho_matrix[0, 0] = gene_row["spearman_rho"]
        adjusted_p_matrix[0, 0] = gene_row["mht_adjusted_p_value"]
        for column_index, row in enumerate(a_rows.itertuples(index=False), start=1):
            rho_matrix[0, column_index] = row.spearman_rho
            adjusted_p_matrix[0, column_index] = row.mht_adjusted_p_value
        for row_index, row in enumerate(b_rows.itertuples(index=False), start=1):
            rho_matrix[row_index, 0] = row.spearman_rho
            adjusted_p_matrix[row_index, 0] = row.mht_adjusted_p_value

        filename = (
            f"{output_path.stem}_example{example_number}_"
            f"{pair.gene_a}_{pair.gene_b}.png"
        )
        plot_path = OUTPUT_FOLDER / filename
        print(f"Saving example matrix {example_number}/{len(selected_pairs)}...", flush=True)
        plot_pair_matrix(
            pair,
            rho_matrix,
            adjusted_p_matrix,
            row_labels,
            column_labels,
            plot_path
        )
        print(f"Saved example matrix: {plot_path}")


if __name__ == "__main__":
    main()
