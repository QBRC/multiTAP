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
from datetime import datetime

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

TEST_RUN = False

SAVED_GROUPS = [88, 178] if TEST_RUN else [86, 87, 88, 175, 176, 178]
BASE_PKL_DIR = "/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work"

# spatial arguments
METHOD = "k-neighbor" # ["distance", "k-neighbor"]
THRESHOLD = 15

roi_pt_id_mapping = pd.read_csv('/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/roi_pt_id_mapping.csv')

all_tumor_cells = 0
all_nontumor_cells = 0
accumul_type = 'sum'
feature_name = "75normed"
df_feature_name = f"df_feature_{feature_name}"


for prefix in SAVED_GROUPS:
    prefix_pt_roi_path = f'/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/nsclc_save_group{prefix}/nsclc_save_group{prefix}.pkl'
    cytof_cohort_whole_slide = pkl.load(open(prefix_pt_roi_path, 'rb'))
    pt_prefix_rois = roi_pt_id_mapping[roi_pt_id_mapping['ROI'].str.startswith(f'{prefix}_')].reset_index(drop=True)
    prefix_pt_ids = np.unique([pt_prefix_rois['Patient_ID']])
    
    if TEST_RUN: prefix_pt_ids = prefix_pt_ids[:2]

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_datetime, f"{len(prefix_pt_ids)} unique patient IDs identified in 'nsclc_save_group{prefix}.pkl' file")

    save_group_2d_spatial_list = list()

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
            per_pt_cohort.generate_summary(accumul_type=accumul_type)
                
            # go through each roi, get their binary marker-cell expression
            for key, cytof_img in per_pt_cohort.cytof_images.items():
                                
                # run spatial interaction
                df_expected_prob, df_cell_interaction_prob = cytof_img.roi_interaction_graphs(feature_name='75normed', accumul_type=accumul_type, method=METHOD, threshold=THRESHOLD)

                # do some post processing
                marker_all = df_expected_prob.columns
                epsilon = 1e-6

                # Normalize and fix Nan
                edge_percentage_norm = np.log10(df_cell_interaction_prob.values / (df_expected_prob.values+epsilon) + epsilon)

                # if observed/expected = 0, then log odds ratio will have log10(epsilon)
                # no observed means interaction cannot be determined, does not mean strong negative interaction
                edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0
                edge_perc_remapped = pd.DataFrame(edge_percentage_norm, index=marker_all, columns=marker_all)
                edge_perc_remapped['roi_id'] = key

                save_group_2d_spatial_list.append(edge_perc_remapped)

        except Exception as e:
            print(f"ROI {row['ROI']} not processed due to error {e}")

    # save to csv to each .pkl
    combined_df = pd.concat(save_group_2d_spatial_list)

    # save to file
    save_path = os.path.join(BASE_PKL_DIR, f'nsclc_save_group{prefix}', f'nsclc_save_group{prefix}_2d_spatial_interaction_thresh_{THRESHOLD}_method_{METHOD}_accuml_{accumul_type}.csv')
    combined_df.to_csv(save_path)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_datetime, f"File saved to {save_path}")
    
print('process completed.')