# Calculating adjusted p-values and determining significance using...

# 1) Bonferroni correction
# 2) Holm-Bonferroni correction
# 3) Permutation Max Statistic (PMS) correction
# 4) Benjamini-Hochberg (BH) correction
# 5) Benjamini-Yekutieli (BY) correction

from itertools import combinations
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT_FOLDER = Path(__file__).resolve().parents[2]   # the Sapal-Speke-RP-2026 folder
sys.path.insert(0, str(ROOT_FOLDER))                # so "from functions..." works from any folder

from functions.data_loading.expression_data import load_matrix, read_isoform_genes
from functions.graphs.correction_comparison import plot_correction_comparison, threshold_label
from functions.hypothesis_testing.fdr import benjamini_hochberg_correction, benjamini_yekutieli_correction
from functions.hypothesis_testing.fwer import bonferroni_correction, holm_bonferroni_correction
from functions.hypothesis_testing.pms import pms_correction

DATA_FILE = None
# None selects the latest option-2 (region) result from gene_spearman.py in INPUT_FOLDER.
INPUT_FOLDER = ROOT_FOLDER / "2a_gene_gene_correlation" / "CSV"
FIGURES_FOLDER = ROOT_FOLDER / "2b_gene_gene_hypothesis_testing" / "Figures"
PMS_ALPHA = 0.5
PMS_N_PERMUTATIONS = 10_000
PMS_RANDOM_STATE = 2026
BONFERRONI_ALPHA = 0.5
HOLM_ALPHA = 0.5
BH_Q = 0.05
BY_Q = 0.05

# Read data from CSV file
def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

def find_data_file():
    if DATA_FILE is not None:
        return str(ROOT_FOLDER / DATA_FILE)
    candidates = list(INPUT_FOLDER.glob("gene_spearman_*_02_region_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No option-2 region result found in {INPUT_FOLDER}. "
            "Run gene_spearman.py with option 2 or set DATA_FILE."
        )
    return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))

data_file_path = find_data_file()
data = read_data(data_file_path)
# gene_spearman.py names this column partial_p_value_region
data = data.rename(columns={"partial_p_value_region": "partial_p_value"})

# 1) Bonferroni correction
data['bonferroni_adjusted_p'], data['bonferroni_significant'] = bonferroni_correction(
    data['partial_p_value'],
    alpha=BONFERRONI_ALPHA
)

# 2) Holm-Bonferroni correction
data['holm_bonferroni_adjusted_p'], data['holm_bonferroni_significant'] = holm_bonferroni_correction(
    data['partial_p_value'],
    alpha=HOLM_ALPHA
)

# 3) Permutation Max Statistic (PMS) correction
pms_ranks_path = data_file_path.replace(".csv", "_partial_ranks.npz")
with np.load(pms_ranks_path, allow_pickle=False) as pms_data:
    pms_gene_ids = pms_data["gene_ids"].astype(str)
    partial_ranks = pms_data["partial_ranks"]

if len(set(pms_gene_ids)) != len(pms_gene_ids):
    raise ValueError(f"Duplicate gene IDs in PMS rank file: {pms_ranks_path}")
if partial_ranks.ndim != 2 or partial_ranks.shape[0] != len(pms_gene_ids):
    raise ValueError(f"Gene IDs do not match the rank matrix in {pms_ranks_path}")
gene_indices = {gene_id: index for index, gene_id in enumerate(pms_gene_ids)}
missing_genes = set(data["gene_a"]).union(data["gene_b"]).difference(gene_indices)
if missing_genes:
    raise ValueError(
        f"PMS rank file is missing genes from the results: {sorted(missing_genes)[:5]}"
    )

pair_adjusted_p, pair_significant, max_statistics = pms_correction(
    partial_ranks,
    alpha=PMS_ALPHA,
    n_permutations=PMS_N_PERMUTATIONS,
    random_state=PMS_RANDOM_STATE
)
pair_index = {
    (gene_a, gene_b): index
    for index, (gene_a, gene_b) in enumerate(combinations(pms_gene_ids, 2))
}

def get_pms_pair_result(gene_a, gene_b):
    pair_position = pair_index.get((gene_a, gene_b))
    if pair_position is None:
        pair_position = pair_index.get((gene_b, gene_a))
    if pair_position is None:
        raise ValueError(f"PMS rank file has no gene pair: {gene_a}, {gene_b}")
    return pair_adjusted_p[pair_position], pair_significant[pair_position]

pair_results = [
    get_pms_pair_result(str(gene_a), str(gene_b))
    for gene_a, gene_b in zip(data["gene_a"], data["gene_b"])
]
data["pms_adjusted_p"] = [result[0] for result in pair_results]
data["pms_significant"] = [result[1] for result in pair_results]

# 4) Benjamini-Hochberg (BH) correction
data['bh_adjusted_p'], data['bh_significant'] = benjamini_hochberg_correction(
    data['partial_p_value'],
    q=BH_Q
)

# 5) Benjamini-Yekutieli (BY) correction
data['by_adjusted_p'], data['by_significant'] = benjamini_yekutieli_correction(
    data['partial_p_value'],
    q=BY_Q
)

# Calculate the same correction methods from ordinary Spearman p-values.
data['bonferroni_raw_adjusted_p'], data['bonferroni_raw_significant'] = bonferroni_correction(
    data['p_value'],
    alpha=BONFERRONI_ALPHA
)
data['holm_bonferroni_raw_adjusted_p'], data['holm_bonferroni_raw_significant'] = (
    holm_bonferroni_correction(data['p_value'], alpha=HOLM_ALPHA)
)
data['bh_raw_adjusted_p'], data['bh_raw_significant'] = benjamini_hochberg_correction(
    data['p_value'],
    q=BH_Q
)
data['by_raw_adjusted_p'], data['by_raw_significant'] = benjamini_yekutieli_correction(
    data['p_value'],
    q=BY_Q
)

# PMS needs a permutation null built from the same raw gene ranks being tested.
raw_expression = load_matrix(str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv"))
raw_expression = raw_expression[
    [sample for sample in raw_expression.columns if "_Fetal_" not in sample]
]
raw_gene_expression = raw_expression.groupby(
    read_isoform_genes(str(ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf"))[raw_expression.index],
    sort=False
).sum()
missing_raw_genes = set(pms_gene_ids).difference(raw_gene_expression.index)
if missing_raw_genes:
    raise ValueError(
        "Raw expression matrix is missing genes from the PMS rank file: "
        f"{sorted(missing_raw_genes)[:5]}"
    )
raw_gene_ranks = raw_gene_expression.loc[pms_gene_ids].rank(axis=1).to_numpy(dtype=float)
raw_pair_adjusted_p, raw_pair_significant, raw_max_statistics = pms_correction(
    raw_gene_ranks,
    alpha=PMS_ALPHA,
    n_permutations=PMS_N_PERMUTATIONS,
    random_state=PMS_RANDOM_STATE
)
raw_pair_results = [
    (
        raw_pair_adjusted_p[pair_index[(gene_a, gene_b)]]
        if (gene_a, gene_b) in pair_index
        else raw_pair_adjusted_p[pair_index[(gene_b, gene_a)]],
        raw_pair_significant[pair_index[(gene_a, gene_b)]]
        if (gene_a, gene_b) in pair_index
        else raw_pair_significant[pair_index[(gene_b, gene_a)]]
    )
    for gene_a, gene_b in zip(data["gene_a"], data["gene_b"])
]
data["pms_raw_adjusted_p"] = [result[0] for result in raw_pair_results]
data["pms_raw_significant"] = [result[1] for result in raw_pair_results]

## Methods Evaluation ######################################################################################################################





## Results Visualisation ######################################################################################################################


# Produce a summary table of the number of significant results, proportion of total tests for each method
summary_table = pd.DataFrame({
    'Method': ['Bonferroni', 'Holm-Bonferroni', 'PMS', 'Benjamini-Hochberg', 'Benjamini-Yekutieli'],
    'Threshold': [
        threshold_label("FWER", BONFERRONI_ALPHA),
        threshold_label("FWER", HOLM_ALPHA),
        threshold_label("FWER", PMS_ALPHA),
        threshold_label("FDR", BH_Q),
        threshold_label("FDR", BY_Q)
    ],
    'Number of Significant Results': [
        data['bonferroni_significant'].sum(),
        data['holm_bonferroni_significant'].sum(),
        data['pms_significant'].sum(),
        data['bh_significant'].sum(),
        data['by_significant'].sum()
    ],
    'Proportion of Total Tests': [
        data['bonferroni_significant'].sum() / len(data),
        data['holm_bonferroni_significant'].sum() / len(data),
        data['pms_significant'].sum() / len(data),
        data['bh_significant'].sum() / len(data),
        data['by_significant'].sum() / len(data)
    ]
})

print(summary_table)

plot_correction_comparison(
    data,
    max_statistics,
    raw_max_statistics,
    BONFERRONI_ALPHA,
    HOLM_ALPHA,
    PMS_ALPHA,
    BH_Q,
    BY_Q,
    FIGURES_FOLDER
)
