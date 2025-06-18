# this file runs the spatial interactions for individual roi, and returns a long format values

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import skimage
import yaml
from typing import Union, Optional, Type, Tuple, List, Dict
import sys
from skimage.color import label2rgb
import json
# import nrrd

import pandas as pd
import seaborn as sns
# Project Root
# used for searching packages and functions
# TODO: enter your project root dir here
ROOT_DIR = '/project/Xie_Lab/zgu/xiao_multiplex/multiTAP/image_cytof'

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'image_cytof'))
from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import save_multi_channel_img, check_feature_distribution
from cytof.classes import CytofImageTiff
from cytof.classes import CytofCohort



def flatten_interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens a non-symmetric spatial interaction matrix into a single-row
    wide-format DataFrame. Each column is a directional pair (e.g., 'GeneA_to_GeneB').

    Parameters:
        df (pd.DataFrame): A square DataFrame of spatial interaction values.

    Returns:
        pd.DataFrame: A single-row wide-format DataFrame with directional interaction pairs.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("Input DataFrame must be square.")
    if not df.columns.equals(df.index):
        raise ValueError("DataFrame must have matching row and column labels.")

    colnames = [f"{row}_to_{col}" for row in df.index for col in df.columns]
    values = df.values.flatten()

    return pd.DataFrame([values], columns=colnames)

TEST_RUN = False # only runs a small batch

# spatial arguments
METHOD = "k-neighbor" # ["distance", "k-neighbor"]
THRESHOLD = 50

# single .pkl would be too alrge to store all 1000pt, 2000+ ROI 
SAVED_GROUPS = [88, 178] if TEST_RUN else [86, 87, 88, 175, 176, 178]
BASE_PKL_DIR = "/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work"
df_feature_name = 'df_feature_75normed'
accumul_type = 'sum'

# contains the final list of patient included after quality check
# for exclusion criteria see redistribute_save_groups.ipynb
roi_pt_id_mapping = pd.read_csv('/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/roi_pt_id_mapping.csv')

for prefix in SAVED_GROUPS:
    prefix_pt_roi_path = f'/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/nsclc_save_group{prefix}/nsclc_save_group{prefix}.pkl'
    cytof_cohort_whole_slide = pkl.load(open(prefix_pt_roi_path, 'rb'))
    pt_prefix_rois = roi_pt_id_mapping[roi_pt_id_mapping['ROI'].str.startswith(f'{prefix}_')].reset_index(drop=True)

    if TEST_RUN: pt_prefix_rois = pt_prefix_rois.head(2)
    
    prefix_pt_ids = np.unique([pt_prefix_rois['Patient_ID']])

    print(f"\n{len(prefix_pt_ids)} unique patient IDs identified in 'nsclc_save_group{prefix}.pkl' file")
    
    # flatten each marker x marker matrix
    flat_marker_roi_list = []

    # process for each ROI
    for index, row in pt_prefix_rois.iterrows():
        print('\nProcessing ROI', row['ROI'])

        try:
            # create key to access in save groups
            new_key = f"{row['SLIDE']}_{row['ROI']}"
            cytof_img_roi = cytof_cohort_whole_slide.cytof_images[new_key]
            
            # df_cohort not saved, creating one automatically from CytofCohort
            per_roi_cohort = CytofCohort(cytof_images={new_key:cytof_img_roi}, dir_out=None)
            per_roi_cohort.batch_process_feature()
            per_roi_cohort.generate_summary() # writes new attributes into cytof_img_roi needed for spatial analysis
            
            # run spatial interaction
            df_expected_prob, df_cell_interaction_prob = cytof_img_roi.roi_interaction_graphs(feature_name='75normed', accumul_type='sum', method=METHOD, threshold=THRESHOLD)

            # do some post processing
            marker_all = df_expected_prob.columns
            epsilon = 1e-6

            # Normalize and fix Nan
            edge_percentage_norm = np.log10(df_cell_interaction_prob.values / (df_expected_prob.values+epsilon) + epsilon)

            # if observed/expected = 0, then log odds ratio will have log10(epsilon)
            # no observed means interaction cannot be determined, does not mean strong negative interaction
            edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0
            edge_perc_remapped = pd.DataFrame(edge_percentage_norm, index=marker_all, columns=marker_all)

            # flatten the matrix
            flat_edge_perc_remapped = flatten_interaction_matrix(edge_perc_remapped)
            flat_edge_perc_remapped['ROI_ID'] = row['ROI']
            flat_marker_roi_list.append(flat_edge_perc_remapped)

        except Exception as e:
            print(f"ROI {row['ROI']} not processed due to error {e}")

    # save to csv to each .pkl
    combined_df = pd.concat(flat_marker_roi_list, axis=0, ignore_index=True)

    save_path = os.path.join(BASE_PKL_DIR, f'nsclc_save_group{prefix}', f'nsclc_save_group{prefix}_flattened_spatial_thresh{THRESHOLD}_method_{METHOD}.csv')
    combined_df.to_csv(save_path, index=False)
    print('flattened co-exp csv saved to', save_path)

print('process completed.')