# This file showcases on reading inflated data collected from 
# (Cords, L., Engler, S., Haberecker, M., Rüschoff, J.-H., Moch, H., de Souza, N., & Bodenmiller, B. (2023). Cancer-associated fibroblast phenotypes predict patient outcome in non-small cell lung cancer (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7961844)
# and converts all acquisitions inside .mcd to .tiff

import numpy as np
import os
import pickle as pkl
import yaml
import sys
import re
import tifffile
from readimc import MCDFile

# ######Specific to the NSCLC cohort from Cords, Lena, et al. Cancer cell 42.3 (2024): 396-412. ######
# search_folder = '/project/Xie_Lab/zgu/xiao_multiplex/data/zenodo_7961844' # the parent folder that contains the data folder
# pattern = re.compile(r'.*_LC_NSCLC_TMA.*') # identifiers for the folder that contains .mcd files
# ##########################################

######General cases######
search_folder = '/project/Xie_Lab/zgu/xiao_multiplex'  # the parent folder that contains the data folder
pattern = re.compile(r'.*sclc_data_mdacc*') # identifiers for the folder that contains .mcd files
##########################################

matched_folders = [name for name in os.listdir(search_folder) if pattern.match(name)]
print(matched_folders)

output_dir = '/project/Xie_Lab/zgu/xiao_multiplex/sclc_data_mdacc/tiffs'

for matched_folder in matched_folders:
    print(f'========Processing folder {matched_folder}========')
    for mcd_paths in os.listdir(os.path.join(search_folder, matched_folder)):
        group = mcd_paths.split('.')[0] # e.g. 2020115_LC_NSCLC_TMA_86_A

        ######Specific to the NSCLC######
        # accession_find = re.search(r'_([0-9]+_[A-Z])\.mcd$', mcd_paths) # find by digits then a capital letter then .mcd
        #################################

        # general case
        accession_find = re.search(r'\.mcd$', mcd_paths, re.IGNORECASE)
        
        if not accession_find: continue # skip iteration if no match

        # path to the actual mcd file
        mcd_file = os.path.join(search_folder, matched_folder, mcd_paths)
        with MCDFile(mcd_file) as f:
            for slide in f.slides: # len(f.slides)=1

                # loop through individual ROIs
                for roi in slide.acquisitions:
                    roi_id = roi.description
                    roi_id = roi_id.replace(',', '_')

                    try:
                        ind_roi = f.read_acquisition(roi)
                        save_path = os.path.join(output_dir, f'{group}_{roi_id}.tiff')
                        tifffile.imwrite(save_path, ind_roi, metadata=roi.metadata)
                        print('TIFF saved to', save_path)
                    except OSError as e: # sometimes individual images are corrupt
                        print(f'OSError for {group}_{roi_id}', e)

print('Process completed.')
