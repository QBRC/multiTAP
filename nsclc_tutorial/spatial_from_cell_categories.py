"""Compute NSCLC spatial interaction coefficients at the cell-category level."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph as skgraph


BASE_CSV_DIR = Path("/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work")
DEFAULT_OUTPUT_DIR = BASE_CSV_DIR / "05-08-26_results"
DEFAULT_ACCUMUL_TYPE = "sum"
DEFAULT_K_NEIGHBORS = 15
CATEGORY_DUMMY_COLS = [
    "cell_category_tumor",
    "cell_category_immune",
    "cell_category_vessel",
    "cell_category_CAF",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compute_roi_interaction(one_roi_spatial: pd.DataFrame, k_neighbors: int) -> pd.DataFrame:
    """Compute one ROI's category interaction matrix using the tutorial formula."""
    n_cells = one_roi_spatial.shape[0]
    if n_cells < 2:
        raise ValueError("ROI has fewer than two categorized cells")

    n_neighbors = min(k_neighbors, n_cells - 1)
    neighbor_matrix = skgraph(
        np.array(one_roi_spatial.loc[:, ["coordinate_x", "coordinate_y"]]),
        n_neighbors=n_neighbors,
        mode="distance",
    )

    i_index, j_index, distances = sp.find(neighbor_matrix)
    keep_index = distances > 0
    i_keep, j_keep = i_index[keep_index], j_index[keep_index]
    if len(i_keep) == 0:
        raise ValueError("ROI has no non-zero-distance neighbor pairs")

    df_pos_exp_val = one_roi_spatial[CATEGORY_DUMMY_COLS].values
    cell_category_pos_list = [np.where(cell)[0] for cell in df_pos_exp_val]

    n_categories = len(CATEGORY_DUMMY_COLS)
    observed_counts = np.zeros((n_categories, n_categories))
    expected_marker_count_1d = np.zeros(n_categories)

    for i_cell, j_cell in zip(i_keep, j_keep):
        category_index_pair1 = cell_category_pos_list[i_cell]
        category_index_pair2 = cell_category_pos_list[j_cell]

        for coords in product(category_index_pair1, category_index_pair2):
            observed_counts[coords] += 1

        category_index_both_neighbor_pair = np.union1d(category_index_pair1, category_index_pair2)
        expected_marker_count_1d[category_index_both_neighbor_pair] += 1

    n_neighbor_pairs = len(i_keep)
    expected_counts = np.outer(expected_marker_count_1d, expected_marker_count_1d)
    expected_prob = expected_counts / n_neighbor_pairs**2
    observed_prob = observed_counts / n_neighbor_pairs

    epsilon = 1e-6
    edge_percentage_norm = np.log10(observed_prob / (expected_prob + epsilon) + epsilon)
    edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0

    return pd.DataFrame(edge_percentage_norm, index=CATEGORY_DUMMY_COLS, columns=CATEGORY_DUMMY_COLS)


def build_category_spatial_table(category_df: pd.DataFrame, k_neighbors: int, test_run: bool = False) -> pd.DataFrame:
    categorized_df = category_df.dropna(subset=["cell_category"]).copy()
    categorized_df["cell_category"] = pd.Categorical(
        categorized_df["cell_category"],
        categories=["tumor", "immune", "vessel", "CAF"],
        ordered=True,
    )
    categorized_df = categorized_df.dropna(subset=["cell_category"]).copy()
    categorized_df = pd.get_dummies(categorized_df, columns=["cell_category"], dtype=int)

    for col in CATEGORY_DUMMY_COLS:
        if col not in categorized_df.columns:
            categorized_df[col] = 0

    roi_ids = np.unique(categorized_df["roi_id"])
    if test_run:
        roi_ids = roi_ids[:2]

    print(timestamp(), f"{len(roi_ids)} ROIs identified")
    roi_matrices = []
    skipped = []

    for one_roi in roi_ids:
        print(f"Processing ROI {one_roi}")
        try:
            one_roi_spatial = categorized_df.loc[categorized_df["roi_id"] == one_roi].reset_index(drop=True)
            edge_perc_remapped = compute_roi_interaction(one_roi_spatial, k_neighbors)
            edge_perc_remapped["roi_id"] = one_roi
            roi_matrices.append(edge_perc_remapped)
        except Exception as exc:
            skipped.append({"roi_id": one_roi, "reason": str(exc)})
            print(f"ROI {one_roi} not processed due to error {exc}")

    if not roi_matrices:
        raise RuntimeError("No ROI spatial matrices were computed.")

    combined_df = pd.concat(roi_matrices, axis=0)
    if skipped:
        print(f"Skipped {len(skipped)} ROIs")
        print(pd.DataFrame(skipped).head(20).to_string(index=False))
    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--accuml-type", default=DEFAULT_ACCUMUL_TYPE, choices=["sum", "ave"])
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite the output CSV if it already exists.")
    args = parser.parse_args()

    input_path = args.output_dir / f"cell_categories_accuml_{args.accuml_type}.csv"
    output_path = args.output_dir / f"cell_category_spatial_kneighbor{args.k_neighbors}_{args.accuml_type}.csv"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists; pass --force to overwrite: {output_path}")

    category_df = pd.read_csv(input_path)
    combined_df = build_category_spatial_table(category_df, args.k_neighbors, args.test_run)
    combined_df.to_csv(output_path)

    print(timestamp(), f"Saved category spatial interactions to {output_path}")
    print(f"Output shape: {combined_df.shape}")
    print(f"Unique ROIs: {combined_df['roi_id'].nunique()}")


if __name__ == "__main__":
    main()
