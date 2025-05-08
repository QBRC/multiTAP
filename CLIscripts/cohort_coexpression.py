# This file analyzes co-expression at the slide level. This script is used to generate figure 4b.

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

# used for searching packages and functions
ROOT_DIR = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cytof/image_cytof'

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'image_cytof'))
from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import save_multi_channel_img, check_feature_distribution
from cytof.classes import CytofCohort

os.environ['OPENBLAS_NUM_THREADS'] = '64'
one_slide = 'BaselTMA_SP43_25'

## TODO: Load your cohort files here
filename = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cytof/multiTAP_public/multiTAP/CLIscripts/BaselTMA_SP43_25_verify/BaselTMA_SP43_25_verify.pkl'
# filename = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cytof/multiTAP_public/multiTAP/CLIscripts/BaselTMA_PTNM_T4/BaselTMA_PTNM_T4.pkl'
# cytof_cohort_whole_slide = pkl.load(open(os.path.join(ROOT_DIR, 'CLIscripts', 'SlideBaselTMA_SP43_25_final.pkl'), 'rb'))
# cytof_cohort_whole_slide = pkl.load(open(os.path.join(ROOT_DIR, 'CLIscripts', 'SlideBaselTMA_SP43_25_rerun.pkl'), 'rb'))
cytof_cohort_whole_slide = pkl.load(open(filename, 'rb'))

print(f'{cytof_cohort_whole_slide} successfully loaded')
slide_co_expression_dict = cytof_cohort_whole_slide.co_expression_analysis()
edge_percentage_norm, column_names = slide_co_expression_dict[one_slide]

# post processing
column_names_clean = [m.replace('_cell_sum', '') for m in column_names]
epsilon = 1e-6 # avoid divide by 0 or log(0)
clustergrid = sns.clustermap(edge_percentage_norm,
                            # clustergrid = sns.clustermap(edge_percentage_norm,
                            center=np.log10(1 + epsilon), cmap='RdBu_r', vmin=-1, vmax=3,
                            xticklabels=column_names_clean, yticklabels=column_names_clean)
plt.title(one_slide)
plt.savefig('figure-slide-coexp-no-summary.pdf', format='pdf', dpi=300, bbox_inches='tight')

print('Program completed.')