# this file runs the co-expression for individual roi, and returns a long format co-exp values

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



def flatten_coexpression_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens a symmetric co-expression matrix (including the diagonal)
    into a single-row wide-format DataFrame.
    
    Parameters:
        df (pd.DataFrame): A square symmetric DataFrame of co-expression values.
        
    Returns:
        pd.DataFrame: A single-row wide-format DataFrame where each column is a
                      unique pair (e.g., 'GeneA_GeneB') and the value is the
                      corresponding co-expression score.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("Input DataFrame must be square.")
    if not df.columns.equals(df.index):
        raise ValueError("DataFrame must have matching row and column labels.")

    # Get upper triangle indices (including diagonal)
    mask = np.triu(np.ones(df.shape), k=0).astype(bool)
    i, j = np.where(mask)

    # Build column names and extract values
    colnames = [f"{df.index[r]}_{df.columns[c]}" for r, c in zip(i, j)]
    values = df.values[i, j]

    # Return as single-row wide-format DataFrame
    return pd.DataFrame([values], columns=colnames)


TEST_RUN = False # only runs a small batch

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

    # process for each patient
    for pt_id in prefix_pt_ids:
        print('\nProcessing Patient ID', pt_id)
        per_pt_roi_dict = dict() # to be pass into CytofCohort later

        # load the pt's ROIs
        df_to_load = pt_prefix_rois[pt_prefix_rois['Patient_ID']==pt_id]
        print(len(df_to_load), 'ROIs identified for patient', pt_id)

        try:
            # load all of this pt's ROI into a new dict
            for index, row in df_to_load.iterrows():
                new_key = f"{row['SLIDE']}_{row['ROI']}"
                per_pt_roi_dict[new_key] = cytof_cohort_whole_slide.cytof_images[new_key]

            # df_cohort not saved, creating one automatically from CytofCohort
            per_pt_cohort = CytofCohort(cytof_images=per_pt_roi_dict, dir_out=None)
            per_pt_cohort.batch_process_feature()
            per_pt_cohort.generate_summary()

            # compute co-expression
            slide_co_expression_dict = per_pt_cohort.co_expression_analysis()
            edge_percentage_norm, column_names = slide_co_expression_dict['NSCLC_ALL']
            edge_perc_remapped = pd.DataFrame(edge_percentage_norm, index=column_names, columns=column_names)
            
            # flatten the matrix
            flat_edge_perc_remapped = flatten_coexpression_matrix(edge_perc_remapped)
            flat_edge_perc_remapped['Patient_ID'] = pt_id

            flat_marker_roi_list.append(flat_edge_perc_remapped)

        except Exception as e:
            print(f'pt_id {pt_id} not processed due to error {e}')

    # save to csv to each .pkl
    combined_df = pd.concat(flat_marker_roi_list, axis=0, ignore_index=True)
    combined_df.to_csv(os.path.join(BASE_PKL_DIR, f'nsclc_save_group{prefix}', f'nsclc_save_group{prefix}_flattened_coexp.csv'), index=False)
    print('flattened co-exp csv saved to', os.path.join(BASE_PKL_DIR, f'nsclc_save_group{prefix}', f'nsclc_save_group{prefix}_flattened_coexp.csv'))

print('process completed.')