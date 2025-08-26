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


cytof_cohort_whole_slide = pkl.load(open("/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/nsclc_save_group88/nsclc_save_group88.pkl", 'rb'))
SAVED_GROUPS = [86, 87, 88, 175, 176, 178]
BASE_PKL_DIR = "/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work"
roi_pt_id_mapping = pd.read_csv('/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/roi_pt_id_mapping.csv')

all_tumor_cells = 0
all_nontumor_cells = 0
accumul_type = 'ave'
feature_name = "75normed"
df_feature_name = f"df_feature_{feature_name}"


for prefix in SAVED_GROUPS:
    prefix_pt_roi_path = f'/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/nsclc_save_group{prefix}/nsclc_save_group{prefix}.pkl'
    cytof_cohort_whole_slide = pkl.load(open(prefix_pt_roi_path, 'rb'))
    pt_prefix_rois = roi_pt_id_mapping[roi_pt_id_mapping['ROI'].str.startswith(f'{prefix}_')].reset_index(drop=True)
    prefix_pt_ids = np.unique([pt_prefix_rois['Patient_ID']])

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_datetime, f"{len(prefix_pt_ids)} unique patient IDs identified in 'nsclc_save_group{prefix}.pkl' file")

    save_group_df_list = list()

    # process for each patient
    for pt_id in prefix_pt_ids:
        print('\nProcessing Patient ID', pt_id)

        # load the pt's ROIs
        df_to_load = pt_prefix_rois[pt_prefix_rois['Patient_ID']==pt_id]
        print(len(df_to_load), 'ROIs identified for patient', pt_id)

        # save individually for each ROI
        for index, row in df_to_load.iterrows():
            print('\nProcessing ROI', row['ROI'])

            try:
                # create key to access in save groups
                new_key = f"{row['SLIDE']}_{row['ROI']}"
                cytof_img_roi = cytof_cohort_whole_slide.cytof_images[new_key]
            
                # df_cohort not saved, creating one automatically from CytofCohort
                per_pt_cohort = CytofCohort(cytof_images={new_key:cytof_img_roi}, dir_out=None)
                per_pt_cohort.batch_process_feature()
                per_pt_cohort.generate_summary(accumul_type=accumul_type)
                

                # get the feature extraction result
                df_feature = getattr(cytof_img_roi, df_feature_name)
                cell_coords = df_feature[["coordinate_x", "coordinate_y"]].copy()
                cell_coords['roi_id'] = new_key
                cell_coords['pt_id'] = row['Patient_ID']
            
                # get the binary expression df
                df_binary_pos_exp = cytof_img_roi.get_binary_pos_express_df(feature_name, accumul_type)
                
                # concatenate the two df
                df_binary_w_coords = pd.concat([cell_coords, df_binary_pos_exp], axis=1)
                save_group_df_list.append(df_binary_w_coords)
            
            except Exception as e:
                print(f'pt_id {pt_id} not processed due to error {e}')

    
    # concatenate to one df at the group level
    save_group_binary_df = pd.concat(save_group_df_list).reset_index(drop=True)

    # save to file
    save_path = os.path.join(BASE_PKL_DIR, f'nsclc_save_group{prefix}', f'nsclc_save_group{prefix}_coordinates_binary_expr_df_{accumul_type}.csv')
    save_group_binary_df.to_csv(save_path, index=False)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_datetime, f"File saved to {save_path}")
    
print('process completed.')