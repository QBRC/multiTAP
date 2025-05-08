# This file is designed to run by shell scripts to extracts all specified slides and ROIs 
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

# Project Root, used for searching packages and functions
ROOT_DIR = '/project/Xie_Lab/zgu/xiao_multiplex/multiTAP/image_cytof'
sys.path.append(ROOT_DIR)

from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import save_multi_channel_img, check_feature_distribution
from cytof.classes import CytofCohort

os.environ['OPENBLAS_NUM_THREADS'] = '64'

class SetParameters():
    def __init__(self, 
                 filename: str, 
                 outdir: str, 
                 label_marker_file: str, 
                 slide: Optional[str] = 'slide1', 
                 roi: Optional[str] = 'roi1', 
                 quality_control_thres: Optional[int] = 50,
                 channels_remove: Optional[List] = None, 
                 channels_dict: Optional[Dict] = None,
                 use_membrane: Optional[bool] = True,
                 cell_radius: Optional[int] = 5, 
                 normalize_qs: Optional[List[int]] = [75, 99]):
        
        self.filename = filename
        self.outdir   = outdir
        self.slide    = slide
        self.roi      = roi
        self.quality_control_thres = quality_control_thres
        self.label_marker_file     = label_marker_file
        self.channels_remove = channels_remove if channels_remove is not None else []
        self.channels_dict   = channels_dict if channels_dict is not None else {}
        self.use_membrane    = use_membrane
        self.cell_radius     = cell_radius
        self.normalize_qs    = normalize_qs

##### generates a pandas df to save as cohorts#####
# one_slide = 'BaselTMA_SP43_25'
# IMC_FOLDER = '/archive/DPDS/Xiao_lab/shared/shidan/hyperion/The Single-Cell Pathology Landscape of Breast Cancer/OMEandSingleCellMasks/OMEnMasks/ome/ome'
# cohort_file_list = []
# roi_name = []
# # find all samples with the above slide number
# for file in glob.glob(os.path.join(IMC_FOLDER, f"{one_slide}*")):
#   cohort_file_list.append(file)
#   roi_name.append("_".join(file.split('_')[-6:-2]))
# slides = [one_slide] * len(cohort_file_list)
# fs_input = cohort_file_list.copy()

# # during batch processing, CytofCohort always expect three inputs. 
# # First input is for naming purposes (name of the slide of cohort)
# # Second input is the varying ROI/TMA that you want to analyze together as one cohort
# # Third input is the corresponding file path
# # df is required to have three keys for downstream analysis: 'Slide', 'ROI', 'input file'
# df_cohort_to_load = pd.DataFrame({"Slide": slides, "ROI": roi_name, "input file": fs_input}) 
# df_cohort_to_load.to_csv('df_cohort_to_load.csv', index=False) #debug purposes
####################################################

##### reading csv in manually#####
# # still required to contain the three keys (see above section)
filename = '/project/Xie_Lab/zgu/xiao_multiplex/breast_cancer_multiTAP_work/testing2.csv'
df_cohort_to_load = pd.read_csv(filename)
##################################

print(f'{len(df_cohort_to_load)} instances identified for cohort processing')

dir_out = '/project/Xie_Lab/zgu/xiao_multiplex/breast_cancer_multiTAP_work'
# dir_out creates an output folder. set dir_out=None to disable.
cytof_slide_cohort = CytofCohort(cytof_images=None, df_cohort=df_cohort_to_load, cohort_name='BaselTMA_testing_roi4', dir_out=dir_out)

channel_dict = {
        'nuclei': ['DNA1-Ir191', 'DNA2-Ir193'],
        'membrane': ['Vimentin-Sm149', 'c-erbB-2 - Her2-Eu151', 'pan Cytokeratin-Keratin Epithelial-Lu175', 
                     'CD44-Gd160','Fibronectin-Nd142'], 
    }

params_cohort = {
  'label_marker_file': "/project/Xie_Lab/zgu/xiao_multiplex/breast_cancer_multiTAP_work/marker_labels.txt",
  'channels_remove': ['nan1-nan1', 'nan2-nan2', 'nan3-nan3', 'nan4-nan4', 'nan5-nan5'],
  'channels_dict': channel_dict,
  'use_membrane': True
}

# computes features individually for all ROI/TMA in the defined cohort
cytof_slide_cohort.batch_process(params=params_cohort)

# # scale feature across ROI images, if needed
# cytof_slide_cohort.batch_process_feature()
# cytof_slide_cohort.generate_summary()

# save cohort
save_path = cytof_slide_cohort.save_cytof_cohort()

print(f'Program completed. Results saved to {save_path}')