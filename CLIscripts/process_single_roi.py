import os
import argparse
import pickle as pkl
import yaml
from typing import Union, Optional, Type, Tuple, List, Dict
import sys

ROOT_DIR = os.path.dirname(sys.path[0])

# add root dir and src dir to path
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'image_cytof'))

from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import check_feature_distribution, save_multi_channel_img
from cytof import classes
from classes import CytofCohort, CytofImage, CytofImageTiff

class SetParameters():
    def __init__(self,
                 filename: str,
                 outdir: str,
                 label_marker_file: Optional[str] = None,
                 slide: str = 'slide1',
                 roi: str = 'roi1',
                 quality_control_thres: int = 50,
                 channels_remove: Optional[List] = None,
                 channels_dict: Optional[Dict] = None,
                 use_membrane: bool = True,
                 cell_radius: int = 5,
                 normalize_qs: List[int] = [75, 99],
                 iltype: Optional[str] = None
                ):
        self.filename = filename
        self.outdir = outdir
        self.slide = slide
        self.roi = roi
        self.quality_control_thres = quality_control_thres
        self.label_marker_file = label_marker_file
        self.channels_remove = channels_remove if channels_remove is not None else []
        self.channels_dict = channels_dict if channels_dict is not None else {}
        self.use_membrane = use_membrane
        self.cell_radius = cell_radius
        self.normalize_qs = normalize_qs
        self.iltype = iltype

def image_to_feature(cytof_img, params: SetParameters):
    '''
    Given a cytof_img, performs 1) loading marker and preprocess, 2) remove unwanted channels,
    3) define special (nuclei and/or membrane) channels, 4) cell segmentation
    5) feature extractions
    '''
    # step 1) loading marker
    if isinstance(cytof_img, CytofImageTiff):
    # if type(cytof_img) is CytofImageTiff:
        try:
            labels_markers = yaml.load(open(params.label_marker_file, "rb"), Loader=yaml.Loader)
            cytof_img.set_markers(**labels_markers)
        except AttributeError: # could not find the label_marker_file attribute
            print('A `label_marker_file` is a mandatory input when analyzing a TIFF image!!')
            return cytof_img
    
    else: # preprocess if instance CytofImage
        cytof_img.get_markers()
        cytof_img.preprocess()
        cytof_img.get_image()


    # step 2) remove unwanted channels
    # will not remove if not specified. channels_remove deault to None
    if params.channels_remove: # this removes channels like 'nan3-nan3'
        cytof_img.remove_special_channels(params.channels_remove)

    # step 3) define special channels
    channels_rm = cytof_img.define_special_channels(params.channels_dict, rm_key='nuclei')
    cytof_img.remove_special_channels(channels_rm) # this removes channels after they have been defined as another channel
    
    # step 4) nuclei and cell segmentation
    nuclei_seg, cell_seg = cytof_img.get_seg(use_membrane=params.use_membrane, radius=params.cell_radius,
                                                show_process=False)

    # step 5) feature extractions
    cytof_img.extract_features(filename=cytof_img.filename)
    cytof_img.feature_quantile_normalization(qs=params.normalize_qs, vis_compare=False) 

    return cytof_img 

def process_single(params: SetParameters, 
                  downstream_analysis: bool = False, 
                  verbose: bool = False) -> Union[Union[CytofImage, CytofImageTiff], None]: 
    
    # read in the data, handles TXT or TIFF based in image_to_feature()
    cytof_img, cols = cytof_read_data_roi(params.filename, params.slide, params.roi, iltype=params.iltype)

    print('Processing', params.slide, 'ROI', params.roi)

    cytof_img.quality_control(thres=params.quality_control_thres)
    if not cytof_img.keep and (not downstream_analysis):
        cytof_img = image_to_feature(cytof_img, params)
        return cytof_img




def main():
    parser = argparse.ArgumentParser(description="CyTOF single ROI image processing.")
    parser.add_argument("param", type=str, help="full file path of the parameters file")
    args = parser.parse_args()
    with open(args.param, 'r') as file:
        fparams = yaml.safe_load(file)
    params = SetParameters(**fparams)

    # read in the data and extracts the features
    cytof_img = process_single(params)
    cytof_img.save_cytof(os.path.join(params.outdir, f'{params.slide}_ROI_{params.roi}.pkl'))

if __name__ == "__main__":
    main()
