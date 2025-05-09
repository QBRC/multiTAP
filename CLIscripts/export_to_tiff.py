# this file reads inflates data collected from https://zenodo.org/records/7961844
import numpy as np
import os
import pickle as pkl
import yaml
import sys
import re
import tifffile
from readimc import MCDFile

search_folder = '/project/Xie_Lab/zgu/xiao_multiplex/data/zenodo_7961844'
pattern = re.compile(r'.*_LC_NSCLC_TMA.*')
matched_folders = [name for name in os.listdir(search_folder) if pattern.match(name)]
print(matched_folders)

# matched_folder = matched_folders[0] # test case


output_dir = '/project/Xie_Lab/zgu/xiao_multiplex/nsclc_tiff_data'
for matched_folder in matched_folders:
    print(f'========Processing folder {matched_folder}========')
    for mcd_paths in os.listdir(os.path.join(search_folder, matched_folder)):
        group = mcd_paths.split('.')[0] # e.g. 2020115_LC_NSCLC_TMA_86_A

        accession_find = re.search(r'_([0-9]+_[A-Z])\.mcd$', mcd_paths) # find by digits then a capital letter then .mcd
        if accession_find:
            accession = accession_find.group(1)

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
