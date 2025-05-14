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
filename = '/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/nsclc_all.csv'
df_nsclc_all = pd.read_csv(filename)
##################################

channel_dict = {
        'nuclei': ['Iridium_1033((1253))Ir191-Ir191', 'Iridium_1033((1254))Ir193-Ir193'],
        
        'membrane': ['HLA-DR_1849((3362))Nd143-Nd143', # HLA-DR — MHC class II, antigen-presenting cell surface
                'CD146_22((3259))Nd144-Nd144',         # CD146 — cell adhesion molecule (endothelial, melanoma)
                'Cadheri_2088((2893))Nd145-Nd145',     # Cadherin — cell-cell adhesion
                'VCAM1_1986((3332))Nd148-Nd148',       # VCAM-1 — endothelial cell adhesion molecule
                'CD20_36((3369))Sm149-Sm149',          # CD20 — B cell marker, membrane localized
                'CD3_1841((3363))Sm152-Sm152',         # CD3 — T cell co-receptor, membrane localized
                'CD279(P_1743((3414))Gd155-Gd155',     # PD-1 — immune checkpoint, membrane localized
                'CD73_2193((3319))Gd156-Gd156',        # CD73 — surface enzyme, membrane-bound
                'CD10_2546((3029))Dy161-Dy161',        # CD10 — membrane metalloproteinase
                'CD45RA_732((2896))Dy164-Dy164',       # CD45RA — isoform of CD45, T cell membrane protein
                'CD8a_1718((2991))Er166-Er166',        # CD8 — cytotoxic T cell marker, membrane-bound
                'CD4_2293((3000))Yb171-Yb171',         # CD4 — helper T cell marker, membrane-bound
                'CD31_1859((3370))Yb172-Yb172',        # CD31 (PECAM-1) — endothelial adhesion molecule
                'CD34_2254((3337))Er170-Er170',        # CD34 — progenitor/endothelial cell surface marker
                'CD140b(_1938((2914))Tm169-Tm169',     # PDGFRB — membrane receptor tyrosine kinase
                'LYVE-1_1982((2881))Er168-Er168',      # LYVE-1 — lymphatic vessel endothelial hyaluronan receptor
                'K-Cadhe_2600((3417))Yb176-Yb176',     # K-Cadherin — membrane adhesion protein
                'panCyto_234((2745))Lu175-Lu175',      # Pan-Cytokeratin — cytoplasmic, but outlines epithelial cells well
                'Vimenti_655((1939))Dy162-Dy162'       # used in BrCa IMC dataset
                ]
        }

params_cohort = {
  'label_marker_file': "/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/marker_labels.txt",
  'channels_remove': ['208Pb-Pb208', '207Pb-Pb207', '205Tl-Tl205', '204Pb-Pb204', '203Tl-Tl203', '202Hg-Hg202', '201Hg-Hg201', '200Hg-Hg200', '199Hg-Hg199', '198Hg-Hg198', '197Au-Au197', '196Pt-Pt196', '192Pt-Pt192', '190Os-Os190', '189Os-Os189', '188Os-Os188', '187Os-Os187', '186Os-Os186', '185Re-Re185', '184W-W184', '183W-W183', '182W-W182', '181Ta-Ta181', '180Hf-Hf180', '179Hf-Hf179', '178Hf-Hf178', '177Hf-Hf177', '157Gd-Gd157', '140Ce-Ce140', '139La-La139', '138Ba-Ba138', '137Ba-Ba137', '136Ba-Ba136', '135Ba-Ba135', '134Ba-Ba134', '133Cs-Cs133', '132Xe-Xe132', '131Xe-Xe131', '130Xe-Xe130', '129Xe-Xe129', '128Xe-Xe128', '127I-I127', '126Te-Te126', '125Te-Te125', '124Te-Te124', '123Te-Te123', '122Te-Te122', '121Sb-Sb121', '120Sn-Sn120', '119Sn-Sn119', '118Sn-Sn118', '116Sn-Sn116', '114Cd-Cd114', '112Cd-Cd112', '111Cd-Cd111', '110Cd-Cd110', '109Ag-Ag109', '108Cd-Cd108', '107Ag-Ag107', '106Pd-Pd106', '105Pd-Pd105', '104Pd-Pd104', '103Rh-Rh103', '102Ru-Ru102', '101Ru-Ru101', '100Ru-Ru100', '99Ru-Ru99', '98Mo-Mo98', '97Mo-Mo97', '96Mo-Mo96', '95Mo-Mo95', '94Mo-Mo94', '93Nb-Nb93', '92Zr-Zr92', '91Zr-Zr91', '90Zr-Zr90', '88Sr-Sr88', '87Sr-Sr87', '86Sr-Sr86', '85Rb-Rb85', '84Sr-Sr84', '83Kr-Kr83', '82Kr-Kr82', '81Br-Br81', '80ArAr-ArAr80', '78Se-Se78', '77Se-Se77', '76Se-Se76', '75As-As75'],
  'channels_dict': channel_dict,
  'use_membrane': True
}

dir_out = '/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work'

# number of files too large to be saved into one .pkl
save_groups = np.unique(df_nsclc_all['save_group'])
for save_group in save_groups:

  df_cohort_to_load = df_nsclc_all[df_nsclc_all['save_group']==save_group].reset_index(drop=True)
  df_cohort_to_load = df_cohort_to_load.drop(columns=['save_group']) # for consistency
  print(f'{len(df_cohort_to_load)} instances identified for cohort processing')

  # dir_out creates an output folder. set dir_out=None to disable.
  cytof_slide_cohort = CytofCohort(cytof_images=None, df_cohort=df_cohort_to_load, cohort_name=f'nsclc_save_group{save_group}', dir_out=dir_out)

  # computes features individually for all ROI/TMA in the defined cohort
  cytof_slide_cohort.batch_process(params=params_cohort)

  # # scale feature across ROI images, if needed
  # cytof_slide_cohort.batch_process_feature()
  # cytof_slide_cohort.generate_summary()

  # save cohort
  save_path = cytof_slide_cohort.save_cytof_cohort()

  print(f'Results saved to {save_path}')

print(f'Program completed.')