"""Assign NSCLC cells to broad cell categories for spatial survival analysis.

This script mirrors ``marker_to_cell_subtypes.py`` but keeps tumor cells and
adds the four requested analysis categories: tumor, immune, vessel, and CAF.
It reuses the existing per-save-group coordinate/binary marker CSV files and
writes a new combined table under the dated results directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SAVED_GROUPS = [86, 87, 88, 175, 176, 178]
BASE_CSV_DIR = Path("/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work")
DEFAULT_OUTPUT_DIR = BASE_CSV_DIR / "05-08-26_results"
DEFAULT_ACCUMUL_TYPE = "sum"

CATEGORY_BY_FINAL_CELL_TYPE = {
    "Epithelial": "tumor",
    "CD4 Treg": "immune",
    "IDO_CD4": "immune",
    "IDO_CD8": "immune",
    "Ki67_CD4": "immune",
    "Ki67_CD8": "immune",
    "TCF1/7 CD4": "immune",
    "TCF1/7 CD8": "immune",
    "PD1 CD4": "immune",
    "CD4": "immune",
    "CD8": "immune",
    "myeloid": "immune",
    "neutrophils": "immune",
    "B cells": "immune",
    "lymphatic endothelial": "vessel",
    "high endothelial venules (HEV)": "vessel",
    "endothelial": "vessel",
    "mCAF": "CAF",
    "iCAF": "CAF",
    "tCAF": "CAF",
    "hypoxic CAF": "CAF",
    "hypoxic tCAF": "CAF",
    "IFN CAF": "CAF",
    "vCAF": "CAF",
    "dCAF": "CAF",
    "PDPN CAF": "CAF",
    "SMA CAF": "CAF",
    "Collagen CAF": "CAF",
}


def marker_col(marker_stem: str, cell_suffix: str) -> str:
    return f"{marker_stem}_{cell_suffix}"


def marker_bool(df: pd.DataFrame, marker_name: str) -> pd.Series:
    """Return a robust boolean marker-positive series from a CSV-loaded column."""
    if marker_name not in df.columns:
        raise KeyError(f"Required marker column not found: {marker_name}")

    series = df[marker_name]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def assign_non_tumor_subtypes(non_tumor_cells: pd.DataFrame, cell_suffix: str) -> pd.DataFrame:
    """Assign final non-tumor cell subtypes using the existing tutorial rules."""
    cd3_marker = marker_col("CD3_1841((3363))Sm152-Sm152", cell_suffix)
    cd20_marker = marker_col("CD20_36((3369))Sm149-Sm149", cell_suffix)

    cd3_pos = marker_bool(non_tumor_cells, cd3_marker)
    cd20_pos = marker_bool(non_tumor_cells, cd20_marker)
    non_tumor_cleaned = non_tumor_cells.loc[~(cd3_pos & cd20_pos)].copy()
    non_tumor_cleaned["final_cell_type"] = np.nan

    def is_pos(marker_stem: str) -> pd.Series:
        return marker_bool(non_tumor_cleaned, marker_col(marker_stem, cell_suffix))

    cd3 = is_pos("CD3_1841((3363))Sm152-Sm152")
    cd20 = is_pos("CD20_36((3369))Sm149-Sm149")
    cd4 = is_pos("CD4_2293((3000))Yb171-Yb171")
    cd8 = is_pos("CD8a_1718((2991))Er166-Er166")
    foxp3 = is_pos("FOXP3_115((2911))Dy163-Dy163")
    ido = is_pos("Indolea_2281((3014))Eu151-Eu151")
    ki67 = is_pos("Ki-67_142((3418))Pt194-Pt194")
    tcf7 = is_pos("TCF1TCF_2221((3415))Gd160-Gd160")
    pd1 = is_pos("CD279(P_1743((3414))Gd155-Gd155")
    hla = is_pos("HLA-DR_1849((3362))Nd143-Nd143")
    cd68 = is_pos("CD68_77((3413))Nd150-Nd150")
    mpo = is_pos("Myelope_276((2996))Y89-Y89")
    mmp9 = is_pos("MMP9_2241((2912))Gd158-Gd158")
    lyve = is_pos("LYVE-1_1982((2881))Er168-Er168")
    ccl21 = is_pos("CCL21 6_2177((2889))Yb174-Yb174")
    pnad = is_pos("PNAd_1981((3323))Ho165-Ho165")
    cd146 = is_pos("CD146_22((3259))Nd144-Nd144")
    cd31_vwf = is_pos("CD31_1859((3370))Yb172-Yb172")
    fap = is_pos("fap_323((3412))Nd142-Nd142")
    mmp11 = is_pos("MMP11_2925((3364))Sm154-Sm154")
    collagen = is_pos("Collage_1360((2568))Sm147-Sm147")
    sma = is_pos("SMA_174((3277))In115-In115")
    cd34 = is_pos("CD34_2254((3337))Er170-Er170")
    cd248 = is_pos("CD248 E_2178((2830))Er167-Er167")
    cd10 = is_pos("CD10_2546((3029))Dy161-Dy161")
    cd73 = is_pos("CD73_2193((3319))Gd156-Gd156")
    caix = is_pos("Carboni_2443((2757))Nd146-Nd146")
    pdpn = is_pos("Podopla_1463((2619))Eu153-Eu153")

    def unassigned() -> pd.Series:
        return non_tumor_cleaned["final_cell_type"].isna()

    assignments = [
        ("CD4 Treg", cd3 & cd4 & foxp3),
        ("IDO_CD4", cd3 & cd4 & ido),
        ("IDO_CD8", cd3 & cd8 & ido),
        ("Ki67_CD4", cd3 & cd4 & ki67),
        ("Ki67_CD8", cd3 & cd8 & ki67),
        ("TCF1/7 CD4", cd3 & cd4 & tcf7),
        ("TCF1/7 CD8", cd3 & cd8 & tcf7),
        ("PD1 CD4", cd3 & cd4 & pd1),
        ("CD4", cd3 & cd4),
        ("CD8", cd3 & cd8),
        ("myeloid", hla & cd68),
        ("neutrophils", mpo & mmp9),
        ("B cells", cd20),
        ("lymphatic endothelial", lyve & ccl21),
    ]

    for cell_type, mask in assignments:
        non_tumor_cleaned.loc[mask & unassigned(), "final_cell_type"] = cell_type

    lymphatic_index = non_tumor_cleaned["final_cell_type"].eq("lymphatic endothelial")
    non_tumor_cleaned.loc[pnad & lymphatic_index, "final_cell_type"] = "high endothelial venules (HEV)"

    ido_all_non_tumor = marker_bool(non_tumor_cells, marker_col("Indolea_2281((3014))Eu151-Eu151", cell_suffix))
    cd146_all_non_tumor = marker_bool(non_tumor_cells, marker_col("CD146_22((3259))Nd144-Nd144", cell_suffix))
    cd34_all_non_tumor = marker_bool(non_tumor_cells, marker_col("CD34_2254((3337))Er170-Er170", cell_suffix))
    ki67_all_non_tumor = marker_bool(non_tumor_cells, marker_col("Ki-67_142((3418))Pt194-Pt194", cell_suffix))

    more_assignments = [
        ("endothelial", cd146 & cd31_vwf & ~ccl21),
        ("mCAF", sma & mmp11 & collagen),
        ("iCAF", cd34 & cd248),
        ("hypoxic tCAF", cd10 & caix),
        ("tCAF", cd10 & cd73),
        ("hypoxic CAF", caix),
        ("IFN CAF", ido_all_non_tumor),
        ("vCAF", cd146_all_non_tumor & ~cd34_all_non_tumor),
        ("dCAF", ki67_all_non_tumor),
        ("PDPN CAF", pdpn),
        ("SMA CAF", sma & ~fap & ~mmp11 & ~collagen),
        ("Collagen CAF", collagen & ~mmp11 & ~sma),
    ]

    for cell_type, mask in more_assignments:
        non_tumor_cleaned.loc[mask & unassigned(), "final_cell_type"] = cell_type

    return non_tumor_cleaned


def load_combined_binary_df(base_csv_dir: Path, accumul_type: str) -> pd.DataFrame:
    dfs = []
    for prefix in SAVED_GROUPS:
        csv_path = (
            base_csv_dir
            / f"nsclc_save_group{prefix}"
            / f"nsclc_save_group{prefix}_coordinates_binary_expr_df_{accumul_type}2.csv"
        )
        print(f"Reading {csv_path}")
        dfs.append(pd.read_csv(csv_path))
    return pd.concat(dfs, axis=0, ignore_index=True)


def build_cell_category_table(base_csv_dir: Path, accumul_type: str) -> pd.DataFrame:
    cell_suffix = f"cell_{accumul_type}"
    combined_binary_df = load_combined_binary_df(base_csv_dir, accumul_type)

    tumor_marker = marker_col("panCyto_234((2745))Lu175-Lu175", cell_suffix)
    tumor_index = marker_bool(combined_binary_df, tumor_marker)

    tumor_cells = combined_binary_df.loc[tumor_index].reset_index(drop=True)
    tumor_cells["final_cell_type"] = "Epithelial"

    non_tumor_cells = combined_binary_df.loc[~tumor_index].reset_index(drop=True)
    non_tumor_cells = assign_non_tumor_subtypes(non_tumor_cells, cell_suffix)

    category_df = pd.concat([tumor_cells, non_tumor_cells], axis=0, ignore_index=True)
    keep_cols = ["pt_id", "roi_id", "coordinate_x", "coordinate_y", "final_cell_type"]
    category_df = category_df[keep_cols].copy()
    category_df["cell_category"] = category_df["final_cell_type"].map(CATEGORY_BY_FINAL_CELL_TYPE)

    return category_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv-dir", type=Path, default=BASE_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--accuml-type", default=DEFAULT_ACCUMUL_TYPE, choices=["sum", "ave"])
    parser.add_argument("--force", action="store_true", help="Overwrite the output CSV if it already exists.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"cell_categories_accuml_{args.accuml_type}.csv"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists; pass --force to overwrite: {output_path}")

    category_df = build_cell_category_table(args.base_csv_dir, args.accuml_type)
    category_df.to_csv(output_path, index=False)

    print(f"Saved {category_df.shape[0]} cells to {output_path}")
    print("Cell category counts:")
    print(category_df["cell_category"].value_counts(dropna=False).to_string())
    print(f"Unique ROIs: {category_df['roi_id'].nunique()}")
    print(f"Unique patients: {category_df['pt_id'].nunique()}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
