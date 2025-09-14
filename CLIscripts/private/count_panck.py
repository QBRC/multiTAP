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
accumul_type = 'sum'

for prefix in SAVED_GROUPS:
    prefix_pt_roi_path = f'/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/nsclc_save_group{prefix}/nsclc_save_group{prefix}.pkl'
    cytof_cohort_whole_slide = pkl.load(open(prefix_pt_roi_path, 'rb'))
    pt_prefix_rois = roi_pt_id_mapping[roi_pt_id_mapping['ROI'].str.startswith(f'{prefix}_')].reset_index(drop=True)
    prefix_pt_ids = np.unique([pt_prefix_rois['Patient_ID']])
    print(f"\n{len(prefix_pt_ids)} unique patient IDs identified in 'nsclc_save_group{prefix}.pkl' file")

    save_group_df_list = list()
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
                
                # get the mean expression
                pt_binary_df = cytof_img.get_binary_pos_express_df(feature_name='75normed', accumul_type=accumul_type)
                
                # save the binary expression df for each ROI
                save_binary_df = pt_binary_df.copy()
                save_binary_df['pt_id'] = pt_id
                save_binary_df['roi_id'] = key
                save_group_df_list.append(save_binary_df)

                # mean panCK expression used to identify tumor cells
                # then count number of cells with positively expressed panCK
                panck_col = f"panCyto_234((2745))Lu175-Lu175_cell_{accumul_type}"
                per_pt_tumor_cells = np.sum(pt_binary_df[panck_col])
                per_pt_nontumor_cells = len(pt_binary_df) - per_pt_tumor_cells

                # tally up
                all_tumor_cells += per_pt_tumor_cells
                all_nontumor_cells += per_pt_nontumor_cells


        except Exception as e:
            print(f'pt_id {pt_id} not processed due to error {e}')

    
    print("# of tumor cells so far:", all_tumor_cells)
    print("# of nontumor cells so far:", all_nontumor_cells)

    # concatenate to one df at the group level
    save_group_binary_df = pd.concat(save_group_df_list)

    # save to file
    save_path = os.path.join(BASE_PKL_DIR, f'nsclc_save_group{prefix}', f'nsclc_save_group{prefix}_binary_expr_df_{accumul_type}.csv')
    save_group_binary_df.to_csv(save_path, index=False)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_datetime, f"File saved to {save_path}")
    
print('process completed.')