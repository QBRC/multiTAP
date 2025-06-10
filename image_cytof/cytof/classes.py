import itertools
import re
import warnings
import os
import sys
import copy
import pickle as pkl
import numpy as np
import pandas as pd
import skimage
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import matplotlib.pyplot
matplotlib.pyplot.switch_backend('Agg') 

import seaborn as sns
import phenograph

# suppress numba deprecation warning
# ref: https://github.com/Arize-ai/phoenix/pull/799
with warnings.catch_warnings():
    from numba.core.errors import NumbaWarning

    warnings.simplefilter("ignore", category=NumbaWarning)
    import umap
    from umap import UMAP


from typing import Union, Optional, Type, Tuple, List, Dict
from collections.abc import Callable
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph as skgraph  # , DistanceMetric
from sklearn.metrics import DistanceMetric
from sklearn.cluster import KMeans
from itertools import product


## added for test
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # cytof root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from hyperion_segmentation import cytof_nuclei_segmentation, cytof_cell_segmentation, visualize_segmentation
from cytof.utils import (save_multi_channel_img, generate_color_dict, show_color_table, 
visualize_scatter, visualize_expression, _get_thresholds, _generate_summary)

def get_name(dfrow):
    return os.path.join(dfrow['path'], dfrow['ROI'])


class CytofImage():
    morphology = ["area", "convex_area", "eccentricity", "extent",
                  "filled_area", "major_axis_length", "minor_axis_length",
                  "orientation", "perimeter", "solidity", "pa_ratio"]

    def __init__(self, df: Optional[pd.DataFrame] = None, slide: str = "", roi: str = "", filename: str = ""):
        self.df       = df
        self.slide    = slide
        self.roi      = roi
        self.filename = filename
        self.columns  = None # column names in original cytof data (dataframe)
        self.markers  = None # protein markers
        self.labels   = None # metal isotopes used to tag protein

        self.image    = None
        self.channels = None # channel names correspond to each channel of self.image

        self.features = None


    def copy(self):
        '''
        Creates a deep copy of the current CytofImage object and return it
        '''
        new_instance = type(self)(self.df.copy(), self.slide, self.roi, self.filename)
        new_instance.columns = copy.deepcopy(self.columns)
        new_instance.markers = copy.deepcopy(self.markers)
        new_instance.labels = copy.deepcopy(self.labels)
        new_instance.image = copy.deepcopy(self.image)
        new_instance.channels = copy.deepcopy(self.channels)
        new_instance.features = copy.deepcopy(self.features)
        return new_instance


    def __str__(self):
        return f"CytofImage slide {self.slide}, ROI {self.roi}"

    def __repr__(self):
        return f"CytofImage(slide={self.slide}, roi={self.roi})"

    def save_cytof(self, savename: str):
        directory = os.path.dirname(savename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        pkl.dump(self, open(savename, "wb"))

    def get_markers(self, imarker0: Optional[str] = None):
        """ 
        Get     (1) the channel names correspond to each image channel
                (2) a list of protein markers used to obtain the CyTOF image
                (3) a list of labels tagged to each of the protein markers
        """
        self.columns = list(self.df.columns)
        if imarker0 is not None:  # if the index of the 1st marker provided
            self.raw_channels = self.columns[imarker0:]
        else:  # assumption: channel names have the common expression: marker(label*)
            pattern = "\w+.*\(\w+\)"
            self.raw_channels = [re.findall(pattern, t)[0] for t in self.columns if len(re.findall(pattern, t)) > 0]

        self.raw_markers = [x.split('(')[0] for x in self.raw_channels]
        self.raw_labels  = [x.split('(')[-1].split(')')[0] for x in self.raw_channels]

        self.channels = self.raw_channels.copy()
        self.markers  = self.raw_markers.copy()
        self.labels   = self.raw_labels.copy()
        
    def export_feature(self, feat_name: str, savename: Optional[str] = None):
        """ Export a set of specified feature """
        savename = savename if savename else f"{feat_name}.csv"
        savename = savename if savename.endswith(".csv") else f"{feat_name}.csv"
        df = getattr(self, feat_name)
        df.to_csv(savename)

    def preprocess(self):
        nrow = int(max(self.df['Y'].values)) + 1
        ncol = int(max(self.df['X'].values)) + 1
        n = len(self.df)
        if nrow * ncol > n:
            df2 = pd.DataFrame(np.zeros((nrow * ncol - n, len(self.df.columns)), dtype=int),
                               columns=self.df.columns)
            self.df = pd.concat([self.df, df2])

    def quality_control(self, thres: int = 50) -> None:
        setattr(self, "keep", False)
        if (max(self.df['X']) < thres) \
                or (max(self.df['Y']) < thres):
            print("At least one dimension of the image {}-{} is smaller than {}, exclude from analyzing" \
                  .format(self.slide, self.roi, thres))
            self.keep = False

    def check_channels(self,
                       channels: Optional[List] = None,
                       xlim: Optional[List] = None,
                       ylim: Optional[List] = None,
                       ncols: int = 5,
                       vis_q: float = 0.9,
                       colorbar: bool = False,
                       savedir: Optional[str] = None,
                       savename: str = "check_channels"
                       ):# -> Optional[matplotlib.figure.Figure]:
        """
        xlim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ylim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ncols = number of subplots per row (default=5)
        vis_q = percentile q used to normalize image before visualization  (default=0.9)
        """
        show = True if savedir is None else False
        if channels is not None:
            if not all([cl.lower() in self.channels for cl in channels]):
                print("At least one of the channels not available, visualizing all channels instead!")
                channels = None
        if channels is None:  # if no desired channels specified, check all channels
            channels = self.channels
        nrow = max(self.df['Y'].values) + 1
        ncol = max(self.df['X'].values) + 1
        if len(channels) <= ncols:
            ax_nrow = 1
            ax_ncol = len(channels)
        else:
            ax_ncol = ncols
            ax_nrow = int(np.ceil(len(channels) / ncols))

        fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3 * ax_ncol, 3 * ax_nrow))
        if ax_nrow == 1:
            axes = np.array([axes])
            if ax_ncol == 1:
                axes = np.expand_dims(axes, axis=1)
        for i, _ in enumerate(channels):
            _ax_nrow = int(np.floor(i / ax_ncol))
            _ax_ncol = i % ax_ncol
            image = self.df[_].values.reshape(nrow, ncol)
            percentile_q = np.quantile(image, vis_q) if np.quantile(image, vis_q)!= 0 else 1
            image = np.clip(image / percentile_q, 0, 1)
            axes[_ax_nrow, _ax_ncol].set_title(_)
            if xlim is not None:
                image = image[:, xlim[0]:xlim[1]]
            if ylim is not None:
                image = image[ylim[0]:ylim[1], :]
            im = axes[_ax_nrow, _ax_ncol].imshow(image, cmap="gray")
            if colorbar:
                fig.colorbar(im, ax=axes[_ax_nrow, _ax_ncol])
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(savedir, f"{savename}.png"))
            return fig


    def get_image(self, channels: List =None, inplace: bool = True, verbose=False):
        """ 
        Get channel images based on provided channels. By default, get channel images correspond to all channels
        """
        if channels is not None:
            if not all([cl in self.channels for cl in channels]):
                print("At least one of the channels not available, using default all channels instead!")
                channels = self.channels
                inplace = True
        else:
            channels = self.channels
            inplace = True
        nc = len(channels)
        nrow = max(self.df['Y'].values) + 1
        ncol = max(self.df['X'].values) + 1
        if verbose:
            print("Output image shape: [{}, {}, {}]".format(nrow, ncol, nc))

        target_image = np.zeros([nrow, ncol, nc], dtype=float)
        for _nc in range(nc):
            target_image[..., _nc] = self.df[channels[_nc]].values.reshape(nrow, ncol)
        if inplace:
            self.image = target_image
        else:
            return target_image

    def visualize_single_channel(self,
                                 channel_name: str,
                                 color: str,
                                 quantile: float = None,
                                 visualize: bool = False):
        """
        Visualize one channel of the multi-channel image, with a specified color from red, green, and blue
        """
        channel_id = self.channels.index(channel_name)
        if quantile is None:  # calculate 99th percentile by default
            quantile = np.quantile(self.image[..., channel_id], 0.99)

        channel_id_ = ["red", "green", "blue"].index(color)  # channel index

        vis_im = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        gs = np.clip(self.image[..., channel_id] / quantile, 0, 1)  # grayscale
        vis_im[..., channel_id_] = gs
        vis_im = (vis_im * 255).astype(np.uint8)

        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(vis_im)
            plt.show()
        return vis_im

    def visualize_channels(self,
                           channel_ids: Optional[List]=None,
                           channel_names: Optional[List]=None,
                           quantiles: Optional[List]=None,
                           visualize: Optional[bool]=False,
                           show_colortable: Optional[bool]=False
                           ):
        """
        Visualize multiple channels simultaneously
        """
        assert channel_ids or channel_names, 'At least one should be provided, either "channel_ids" or "channel_names"!'
        if channel_ids is None:
            channel_ids = [self.channels.index(n) for n in channel_names]
        else:
            channel_names = [self.channels[i] for i in channel_ids]
        assert len(channel_ids) <= 7, "No more than 6 channels can be visualized simultaneously!"
        if len(channel_ids) > 3:
            warnings.warn(
                "Visualizing more than 3 channels the same time results in deteriorated visualization. \
                It is not recommended!")

        print("Visualizing channels: {}".format(', '.join(channel_names)))
        full_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white']
        color_values = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                        (0, 1, 1), (1, 0, 1), (1, 1, 0),
                        (1, 1, 1)]
        info = ["{} in {}\n".format(marker, c) for (marker, c) in \
                zip([self.channels[i] for i in channel_ids], full_colors[:len(channel_ids)])]
        print("Visualizing... \n{}".format(''.join(info)))
        merged_im = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        if quantiles is None:
            quantiles = [np.quantile(self.image[..., _], 0.99) for _ in channel_ids]

        # max_vals = []
        for _ in range(min(len(channel_ids), 3)):  # first 3 channels, assign colors R, G, B
            gs = np.clip(self.image[..., channel_ids[_]] / quantiles[_], 0, 1)  # grayscale
            merged_im[..., _] = gs * 255
            max_val = [0, 0, 0]
            max_val[_] = gs.max() * 255
            # max_vals.append(max_val)

        chs = [[1, 2], [0, 2], [0, 1], [0, 1, 2]]
        chs_id = 0
        while _ < len(channel_ids) - 1:
            _ += 1
            max_val = [0, 0, 0]
            for j in chs[chs_id]:
                gs = np.clip(self.image[..., channel_ids[_]] / quantiles[_], 0, 1)
                merged_im[..., j] += gs * 255  # /2
                merged_im[..., j] = np.clip(merged_im[..., j], 0, 255)
                max_val[j] = gs.max() * 255
            chs_id += 1
            # max_vals.append(max_val)
        merged_im = merged_im.astype(np.uint8)
        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(merged_im)
            plt.show()

        vis_markers = [self.markers[i] if i < len(self.markers) else self.channels[i] for i in channel_ids]

        color_dict = dict((n, c) for (n, c) in zip(vis_markers, color_values[:len(channel_ids)]))
        if show_colortable:
            show_color_table(color_dict=color_dict, title="color dictionary", emptycols=3, sort_names=True)
        return merged_im, quantiles, color_dict

    def remove_special_channels(self, channels: List):
        """
        Given a list of channels, remove them from the class. This typically happens when users define certain channels to be the nuclei for special processing.
        """
        for channel in channels:
            if channel not in self.channels:
                print("Channel {} not available, escaping...".format(channel))
                continue
            idx = self.channels.index(channel)
            self.channels.pop(idx)
            self.markers.pop(idx)
            self.labels.pop(idx)
            self.df.drop(columns=channel, inplace=True)

    def define_special_channels(self, channels_dict: Dict, verbose=False, rm_key: str = 'nuclei'):
        '''
        Special channels (antibodies) commonly found to define cell componenets (e.g. nuclei or membranes)
        '''
        channels_rm = []
        for new_name, old_names in channels_dict.items():

            if len(old_names) == 0:
                continue

            old_nms = []
            for i, old_name in enumerate(old_names):
                if old_name not in self.channels:
                    warnings.warn('{} is not available!'.format(old_name))
                    continue
                old_nms.append(old_name)
            if verbose:
                print("Defining channel '{}' by summing up channels: {}.".format(new_name, ', '.join(old_nms)))
            if len(old_nms) > 0:
                # only add channels to removal list if matching remove key
                if new_name == rm_key:
                    channels_rm += old_nms
                for i, old_name in enumerate(old_nms):
                    if i == 0:
                        self.df[new_name] = self.df[old_name]
                    else:
                        self.df[new_name] += self.df[old_name]
                if new_name not in self.channels:
                    self.channels.append(new_name)

        self.get_image(verbose=verbose)
        if hasattr(self, "defined_channels"):
            for key in channels_dict.keys():
                self.defined_channels.add(key)
        else:
            setattr(self, "defined_channels", set(list(channels_dict.keys())))   
        return channels_rm

    def get_seg(
        self, 
        use_membrane: bool = True, 
        radius: int = 5, 
        sz_hole: int = 1, 
        sz_obj: int = 3,
        min_distance: int = 2, 
        fg_marker_dilate: int = 2, 
        bg_marker_dilate: int = 2, 
        show_process: bool = False,
        verbose: bool = False):
        channels = [x.lower() for x in self.channels]
        assert 'nuclei' in channels, "a 'nuclei' channel is required for segmentation!"
        nuclei_img = self.image[..., self.channels.index('nuclei')]

        if show_process:
            print("Nuclei segmentation...")
        # else:
        #     print("Not showing segmentation process")
        nuclei_seg, color_dict = cytof_nuclei_segmentation(nuclei_img, show_process=show_process,
                                                           size_hole=sz_hole, size_obj=sz_obj,
                                                           fg_marker_dilate=fg_marker_dilate,
                                                           bg_marker_dilate=bg_marker_dilate,
                                                           min_distance=min_distance)

        membrane_img = self.image[..., self.channels.index('membrane')] \
            if (use_membrane and 'membrane' in self.channels) else None
        if show_process:
            print("Cell segmentation...")
        cell_seg, _ = cytof_cell_segmentation(nuclei_seg, radius, membrane_channel=membrane_img,
                                              show_process=show_process, colors=color_dict)

        self.nuclei_seg = nuclei_seg
        self.cell_seg   = cell_seg
        return nuclei_seg, cell_seg

    def visualize_seg(self, segtype: str = "cell", seg=None, show: bool = False, bg_label: int = 1):
        assert segtype in ["nuclei", "cell"], f"segtype {segtype} not supported. Accepted cell type: ['nuclei', 'cell']"
        # nuclei in red, membrane in green
        if "membrane" in self.channels:
            channel_ids = [self.channels.index(_) for _ in ["nuclei", "membrane"]]
        else:

            # visualize one marker channel and nuclei channel
            channel_ids = [self.channels.index("nuclei"), 0]

        if seg is None:
            if segtype == "cell":
                seg = self.cell_seg
                '''# membrane in red, nuclei in green
                channel_ids = [self.channels.index(_) for _ in ["membrane", "nuclei"]]'''
            else:
                seg = self.nuclei_seg

        # mark distinct membrane or nuclei boundary colors
        if segtype == 'cell':
            marked_image = visualize_segmentation(self.image, self.channels, seg, channel_ids=channel_ids, bound_color=(1, 1, 1), show=show, bg_label=bg_label)
        else: # marking nucleus boundaries as blue
            marked_image = visualize_segmentation(self.image, self.channels, seg, channel_ids=channel_ids, bound_color=(1, 1, 0), show=show, bg_label=bg_label)

        seg_color = 'yellow' if segtype=='nuclei' else 'white'
        print(f"{segtype} boundary marked by {seg_color}")
        return marked_image

    def extract_features(self, filename, use_parallel=True, show_sample=False):
        from cytof.utils import extract_feature

        # channel indices correspond to pure markers
        '''pattern = "\w+.*\(\w+\)"
        marker_idx      = [i for (i,x) in enumerate(self.channels) if len(re.findall(pattern, x))>0] '''
        marker_idx = [i for (i, x) in enumerate(self.channels) if x not in self.defined_channels]

        marker_channels = [self.channels[i] for i in marker_idx]  # pure marker channels
        marker_image = self.image[..., marker_idx]  # channel images correspond to pure markers
        morphology = self.morphology
        self.features = {
            "nuclei_morphology": [_ + '_nuclei' for _ in morphology],  # morphology - nuclei level
            "cell_morphology": [_ + '_cell' for _ in morphology],  # morphology - cell level
            "cell_sum": [_ + '_cell_sum' for _ in marker_channels],
            "cell_ave": [_ + '_cell_ave' for _ in marker_channels],
            "nuclei_sum": [_ + '_nuclei_sum' for _ in marker_channels],
            "nuclei_ave": [_ + '_nuclei_ave' for _ in marker_channels],
        }
        self.df_feature = extract_feature(marker_channels, marker_image,
                                          self.nuclei_seg, self.cell_seg,
                                          filename, use_parallel=use_parallel, 
                                          show_sample=show_sample)

    def calculate_quantiles(self, qs: Union[List, int] = 75, savename: Optional[str] = None, verbose: bool = False):
        """
        Calculate the q-quantiles of each marker with cell level summation given the q values
        """
        qs = [qs] if isinstance(qs, int) else qs
        _expressions_cell_sum = []
        quantiles = {}
        colors = cm.rainbow(np.linspace(0, 1, len(qs)))
        for feature_name in self.features["cell_sum"]:  # all cell sum features except for nuclei_cell_sum and membrane_cell_sum
            if feature_name.startswith("nuclei") or feature_name.startswith("membrane"):
                continue
            _expressions_cell_sum.extend(self.df_feature[feature_name])

        plt.hist(np.log2(np.array(_expressions_cell_sum) + 0.0001), 100, density=True)
        for q, c in zip(qs, colors):
            quantiles[q] = np.quantile(_expressions_cell_sum, q / 100)
            plt.axvline(np.log2(quantiles[q]), label=f"{q}th percentile", c=c)
            if verbose:
                print(f"{q}th percentile: {quantiles[q]}")
        plt.xlim(-15, 15)
        plt.xlabel("log2(expression of all markers)")
        plt.legend()
        if savename is not None:
            plt.savefig(savename)
        plt.show()
        # attach quantile dictionary to self
        self.dict_quantiles = quantiles

        # return quantiles

    def _vis_normalization(self, savename: Optional[str] = None):
        """
        Compare before and after normalization
        """
        expressions = {}
        expressions["original"] = []

        ## before normalization
        for key, features in self.features.items():
            if key.endswith("morphology"):
                continue
            for feature_name in features:
                if feature_name.startswith('nuclei') or feature_name.startswith('membrane'):
                    continue
                expressions["original"].extend(self.df_feature[feature_name])
        log_exp = np.log2(np.array(expressions['original']) + 0.0001)
        plt.hist(log_exp, 100, density=True, label='before normalization')

        for q in self.dict_quantiles.keys():
            n_attr = f"df_feature_{q}normed"
            expressions[f"{q}_normed"] = []

            for key, features in self.features.items():
                if key.endswith("morphology"):
                    continue
                for feature_name in features:
                    if feature_name.startswith('nuclei') or feature_name.startswith('membrane'):
                        continue
                    expressions[f"{q}_normed"].extend(getattr(self, n_attr)[feature_name])
            plt.hist(expressions[f"{q}_normed"], 100, density=True, label=f"after {q}th percentile normalization")

        plt.legend()
        plt.xlabel('log2(expressions of all markers)')
        plt.ylabel('Frequency')
        if savename is not None:
            plt.savefig(savename)
        plt.show()
        return expressions

    def feature_quantile_normalization(self,
                                       qs: Union[List[int], int] = 75,
                                       vis_compare: bool = True,
                                       savedir: Optional[str] = None):
        """
        Normalize all features with given quantiles except for morphology features
        Args:
            qs: value (int) or values (list of int) of for q-th percentile normalization
            vis_compare: a boolean flag indicating whether or not visualize comparison before and after normalization
            (default=True)
            savedir: saving directory for comparison and percentiles;
            if not None, visualizations of percentiles and comparison before and after normalization will be saved in savedir
            (default=None)

        """
        qs = [qs] if isinstance(qs, int) else qs
        if savedir is not None:
            savename_quantile = os.path.join(savedir, "{}_{}_percentiles.png".format(self.slide, self.roi))
            savename_compare  = os.path.join(savedir, "{}_{}_comparison.png".format(self.slide, self.roi))
        else:
            savename_quantile, savename_compare = None, None
        self.calculate_quantiles(qs, savename=savename_quantile)
        for q, quantile_val in self.dict_quantiles.items():
            n_attr = f"df_feature_{q}normed" # attribute name
            log_normed = copy.deepcopy(self.df_feature)
            for key, features in self.features.items():
                if key.endswith("morphology"):
                    continue
                for feature_name in features:
                    if feature_name.startswith("nuclei") or feature_name.startswith("membrane"):
                        continue
                    # log-quantile normalization
                    log_normed.loc[:, feature_name] = np.log2(log_normed.loc[:, feature_name] / quantile_val + 0.0001)
            setattr(self, n_attr, log_normed)
        if vis_compare:
            _ = self._vis_normalization(savename=savename_compare)


    def save_channel_images(self, savedir: str, channels: Optional[List] = None, ext: str = ".png", quantile_norm: int = 99):
        """
        Save channel images
        """
        if channels is not None:
            if not all([cl in self.channels for cl in channels]):
                print("At least one of the channels not available, saving all channels instead!")
                channels = self.channels
        else:
            channels = self.channels
        '''assert all([x.lower() in channels_temp for x in channels]), "Not all provided channels are available!"'''
        for chn in channels:
            savename = os.path.join(savedir, f"{chn}{ext}")
            #         i = channels_temp.index(chn.lower())
            i = self.channels.index(chn)
            im_temp = self.image[..., i]
            quantile_temp = np.quantile(im_temp, quantile_norm / 100) \
                if np.quantile(im_temp, quantile_norm / 100) != 0 else 1

            im_temp_ = np.clip(im_temp / quantile_temp, 0, 1)
            save_multi_channel_img((im_temp_ * 255).astype(np.uint8), savename)

    def marker_positive(self, feature_type: str = "normed", accumul_type: str = "sum", normq: int = 75):
        assert feature_type in ["original", "normed", "scaled"], 'accepted feature types are "original", "normed", "scaled"'
        if feature_type == "original":
            feat_name = ""
        elif feature_type == "normed":
            feat_name = f"_{normq}normed"
        else:
            feat_name = f"_{normq}normed_scaled"

        n_attr     = f"df_feature{feat_name}"  # class attribute name for feature table
        count_attr = f"cell_count{feat_name}_{accumul_type}"  # class attribute name for feature summary table

        df_feat  = getattr(self, n_attr)
        df_thres = getattr(self, count_attr)

        thresholds_cell_marker = dict((x, y) for (x, y) in zip(df_thres["feature"], df_thres["threshold"]))

        columns = ["id"] + [marker for marker in self.markers]
        df_marker_positive = pd.DataFrame(columns=columns, 
                                          data=np.zeros((len(df_feat), len(self.markers) + 1), type=np.int32))
        df_marker_positive["id"] = df_feat["id"]
        for im, marker in enumerate(self.markers):
            channel_ = f"{self.channels[im]}_cell_{accumul_type}"
            df_marker_positive.loc[df_feat[channel_] > thresholds_cell_marker[channel_], marker] = 1
        setattr(self, f"df_marker_positive{feat_name}", df_marker_positive)


    def marker_positive_summary(self,
                                thresholds: Dict,
                                feat_type: str = "normed", 
                                normq: int = 75,
                                accumul_type: str = "sum"
                                ):

        """
        Generate marker positive summary for CytofImage: 
        Output rendered: f"cell_count_{feat_name}_{aggre}" and f"marker_positive_{feat_name}_{aggre}"
        """

        assert feat_type in ["normed_scaled", "normed", ""], f"feature type {feat_type} not supported!"
        feat_name = f"{feat_type}" if feat_type=="" else f"{normq}{feat_type}" # the attribute name to achieve from cytof_img
        n_attr = f"df_feature{feat_name}" if feat_type=="" else f"df_feature_{feat_name}" # the attribute name to achieve from cytof_img

        df_thres = pd.DataFrame({"feature": thresholds.keys(), "threshold": thresholds.values()})
        df_marker_pos_sum = getattr(self, n_attr).copy()

        keep_feat_set = f"cell_{accumul_type}"

        for key, feat_set in getattr(self, "features").items():
            if key == keep_feat_set:
                marker_set = self.markers
                df_marker_pos_sum_ = df_marker_pos_sum[feat_set].copy().transpose()
                
                comp_cols = list(df_marker_pos_sum_.columns)
                df_marker_pos_sum_.reset_index(names='feature', inplace=True)
                merged  = df_marker_pos_sum_.merge(df_thres, on="feature", how="left")
                df_temp = merged[comp_cols].ge(merged["threshold"], axis=0)
                df_temp.index = merged['feature']
                df_marker_pos_sum[feat_set] = df_temp.transpose()[feat_set]
                map_rename = dict((k, v) for (k,v) in zip(feat_set, marker_set))
                df_marker_pos_sum.rename(columns=map_rename, inplace=True)
            else:        
                df_marker_pos_sum.drop(columns=feat_set, inplace=True)

        df_thres['total number']    = df_temp.count(axis=1).values
        df_thres['positive counts'] = df_temp.sum(axis=1).values
        df_thres['positive ratio']  = df_thres['positive counts'] / df_thres['total number']

        attr_cell_count = f"cell_count_{feat_name}_{accumul_type}"
        attr_marker_pos = f"df_marker_positive_{feat_name}_{accumul_type}"
        setattr(self, attr_cell_count, df_thres)
        setattr(self, attr_marker_pos, df_marker_pos_sum)

        return f"{feat_name}_{accumul_type}"


    def visualize_marker_positive(self,
                                  marker: str,
                                  feature_type: str,
                                  accumul_type: str = "sum",
                                  normq: int = 99,
                                  show_boundary: bool = True,
                                  color_list: List[Tuple] = [(0,0,1), (0,1,0)], # negative, positive
                                  color_bound: Tuple = (0,0,0),
                                  show_colortable: bool=False
                                  ):
        assert feature_type in ["original", "normed",
                                "scaled"], 'accepted feature types are "original", "normed", "scaled"'
        if feature_type == "original":
            feat_name = ""
        elif feature_type == "normed":
            feat_name = f"_{normq}normed"
        else:
            feat_name = f"_{normq}normed_scaled"

        # self.marker_positive(feature_type=feature_type, accumul_type=accumul_type, normq=normq)
        df_marker_positive_original = getattr(self, f"df_marker_positive{feat_name}_{accumul_type}")
        df_marker_positive = df_marker_positive_original.copy()

        # exclude the channels accordingly
        if 'membrane' in self.channels:
            channels_wo_special = self.channels[:-2] # excludes nuclei and membrane channel
        else:
            channels_wo_special = self.channels[:-1] # excludes nuclei channel only

        # the original four location info + marker/channel names
        reconstructed_marker_channel = ['filename', 'id', 'coordinate_x', 'coordinate_y'] + channels_wo_special
        
        assert len(reconstructed_marker_channel) == len(df_marker_positive_original.columns)
        df_marker_positive.columns = reconstructed_marker_channel

        color_dict = dict((key, v) for (key, v) in zip(['negative', 'positive'], color_list))
        if show_colortable:
            show_color_table(color_dict=color_dict, title="color dictionary", emptycols=3)
        color_ids = []

        stain_nuclei = np.zeros((self.nuclei_seg.shape[0], self.nuclei_seg.shape[1], 3)) + 1
        for i in range(2, np.max(self.nuclei_seg) + 1):
            color_id = df_marker_positive[marker][df_marker_positive['id'] == i].values[0]
            if color_id not in color_ids:
                color_ids.append(color_id)
            stain_nuclei[self.nuclei_seg == i] = color_list[color_id][:3]
        # add boundary
        if show_boundary:
            stain_nuclei = mark_boundaries(stain_nuclei,
                                       self.nuclei_seg, mode="inner", color=color_bound)

        # stained Cell image
        stain_cell = np.zeros((self.cell_seg.shape[0], self.cell_seg.shape[1], 3)) + 1
        for i in range(2, np.max(self.cell_seg) + 1):
            color_id = df_marker_positive[marker][df_marker_positive['id'] == i].values[0]
            stain_cell[self.cell_seg == i] = color_list[color_id][:3]
        if show_boundary:
            stain_cell = mark_boundaries(stain_cell,
                                     self.cell_seg, mode="inner", color=color_bound)
        return stain_nuclei, stain_cell, color_dict

    def visualize_pheno(self, key_pheno: str,
                        color_dict: Optional[dict] = None,
                        show: bool = False,
                        show_colortable: bool = False):
        assert key_pheno in self.phenograph, "Pheno-Graph with {} not available!".format(key_pheno)
        phenograph = self.phenograph[key_pheno]
        communities = phenograph['communities']  # phenograph clustering community IDs
        seg_id = self.df_feature['id']  # nuclei / cell segmentation IDs

        if color_dict is None:
            color_dict = dict((_, plt.cm.get_cmap('tab20').colors[_ % 20]) \
                              for _ in np.unique(communities))
        #     rgba_colors   = np.array([color_dict[_] for _ in communities])

        if show_colortable:
            show_color_table(color_dict=color_dict,
                             title="phenograph clusters",
                             emptycols=3, dpi=60)

        # Create image with nuclei / cells stained by PhenoGraph clustering output
        # stain rule: same color for same cluster, stain nuclei
        stain_nuclei = np.zeros((self.nuclei_seg.shape[0], self.nuclei_seg.shape[1], 3)) + 1
        stain_cell = np.zeros((self.cell_seg.shape[0], self.cell_seg.shape[1], 3)) + 1

        for i in range(2, np.max(self.nuclei_seg) + 1):
            commu_id = communities[seg_id == i][0]
            stain_nuclei[self.nuclei_seg == i] = color_dict[commu_id]  # rgba_colors[communities[seg_id == i]][:3] #
            stain_cell[self.cell_seg == i] = color_dict[commu_id]  # rgba_colors[communities[seg_id == i]][:3] #
        if show:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            axs[0].imshow(stain_nuclei)
            axs[1].imshow(stain_cell)

        return stain_nuclei, stain_cell, color_dict

    def get_binary_pos_express_df(self, feature_name, accumul_type):
        """
        returns a dataframe in the form marker1, marker2, ... vs. cell1, cell2; indicating whether each cell is positively expressed in each marker
        """
        df_feature_name = f"df_feature_{feature_name}"

        # get the feature extraction result
        df_feature = getattr(self , df_feature_name)

        # select only markers with desired accumulation type
        marker_col_all = [x for x in df_feature.columns if f"cell_{accumul_type}" in x]

        # subset feature
        df_feature_of_interst = df_feature[marker_col_all]

        # reports each marker's threshold to be considered positively expressed, number of positive cells, etc
        df_cell_count_info = getattr(self, f"cell_count_{feature_name}_{accumul_type}")
        thresholds = df_cell_count_info.threshold

        # returns a binary dataframe of whether each cell at each marker passes the positive threshold
        df_binary_pos_exp = df_feature_of_interst.apply(lambda column: apply_threshold_to_column(column, threshold=thresholds[df_feature_of_interst.columns.get_loc(column.name)]))

        return df_binary_pos_exp
    
    def roi_co_expression(self, feature_name, accumul_type, return_components=False):
        """
        Performs the co-expression analysis at the single ROI level.
        Can return components for cohort analysis if needed
        """
        from itertools import product

        # returns a binary dataframe of whether each cell at each marker passes the positive threshold
        df_binary_pos_exp = self.get_binary_pos_express_df(feature_name, accumul_type)

        n_cells, n_markers = df_binary_pos_exp.shape
        df_pos_exp_val = df_binary_pos_exp.values

        # list all pair-wise combinations of the markers
        column_combinations = list(product(range(n_markers), repeat=2))

        # step to the numerator of the log odds ratio
        co_positive_count_matrix = np.zeros((n_markers, n_markers))

        # step to the denominator of the log odds ratio
        expected_count_matrix = np.zeros((n_markers, n_markers))

        for combo in column_combinations:
            marker1, marker2 = combo

            # count cells that positively expresses in both marker 1 and 2
            positive_prob_marker1_and_2 = np.sum(np.logical_and(df_pos_exp_val[:, marker1], df_pos_exp_val[:, marker2]))
            co_positive_count_matrix[marker1, marker2] = positive_prob_marker1_and_2
            
            # pair (A,B) counts is the same as pair (B,A) counts
            co_positive_count_matrix[marker2, marker1] = positive_prob_marker1_and_2

            # count expected cells if marker 1 and 2 are independently expressed
            # p(A and B) = p(A) * p(B) = num_pos_a * num_pos_b / (num_cells * num_cells)
            # p(A) = number of positive cells / number of cells
            exp_prob_in_marker1_and_2 = np.sum(df_pos_exp_val[:, marker1]) * np.sum(df_pos_exp_val[:, marker2])
            expected_count_matrix[marker1, marker2] = exp_prob_in_marker1_and_2
            expected_count_matrix[marker2, marker1] = exp_prob_in_marker1_and_2

        # theta(i_pos and j_pos)
        df_co_pos = pd.DataFrame(co_positive_count_matrix, index=df_binary_pos_exp.columns, columns=df_binary_pos_exp.columns)

        # E(x)
        df_expected = pd.DataFrame(expected_count_matrix, index=df_binary_pos_exp.columns, columns=df_binary_pos_exp.columns)

        if return_components:
            # hold off on calculating probabilites. Need the components from other ROIs to calculate the co-expression
            return df_co_pos, df_expected, n_cells
        
        # otherwise, return the probabilies
        df_co_pos_prob = df_co_pos / n_cells
        df_expected_prob = df_expected / n_cells**2
        return df_co_pos_prob, df_expected_prob

    def roi_interaction_graphs(self, feature_name, accumul_type, method: str = "distance", threshold=50, return_components=False):
        """ Performs spatial interaction at the ROI level. 
        Finds if two positive markers are in proximity with each other. Proximity can be defined either with k-nearest neighbor or distance thresholding.
        Args:
            key_pheno: dictionary key for a specific phenograph output
            method: method to construct the adjacency matrix, choose from "distance" and "kneighbor"
            threshold: either the number of neighbors or euclidean distance to qualify as neighborhood pairs. Default is 50 for distance and 20 for k-neighbor.
            **kwargs: used to specify distance threshold (thres) for "distance" method or number of neighbors (k)
            for "kneighbor" method
        Output:
            network: (dict) ROI level network that will be used for cluster interaction analysis
        """
        assert method in ["distance", "k-neighbor"], "Method can be either 'distance' or 'k-neighbor'!"
        print(f'Calculating spatial interaction with method "{method}" and threshold at {threshold}')

        df_feature_name = f"df_feature_{feature_name}"

        # get the feature extraction result
        df_feature = getattr(self , df_feature_name)

        # select only markers with desired accumulation type
        marker_col_all = [x for x in df_feature.columns if f"cell_{accumul_type}" in x]

        # subset feature
        df_feature_of_interst = df_feature[marker_col_all]

        n_cells, n_markers = df_feature_of_interst.shape

        networks = {}
        if method == "distance":
            dist = DistanceMetric.get_metric('euclidean')
            neighbor_matrix = dist.pairwise(df_feature.loc[:, ['coordinate_x', 'coordinate_y']].values)
            
            # returns nonzero elements of the matrix
            # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.find.html
            I, J, V = sp.find(neighbor_matrix)
            # finds index of values less than the distance threshold
            v_keep_index = V < threshold

        elif method == "k-neighbor":
            neighbor_matrix = skgraph(np.array(df_feature.loc[:, ['coordinate_x', 'coordinate_y']]), n_neighbors=threshold, mode='distance')
            # returns nonzero elements of the matrix
            # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.find.html
            I, J, V = sp.find(neighbor_matrix)
            v_keep_index = V > 0 # any non-zero distance neighbor qualifies

        # finds index of values less than the distance threshold
        i_keep, j_keep = I[v_keep_index], J[v_keep_index]
        assert len(i_keep) == len(j_keep) # these are paired indexes for the cell. must equal in length.

        n_neighbor_pairs = len(i_keep)

        # (i,j) now tells you the index of the two cells that are in close proximity (within {thres} distance of each other)
        # now we need a list that tells you the positive expressed marker index in each cell
        
        # returns a binary dataframe of whether each cell at each marker passes the positive threshold
        df_binary_pos_exp = self.get_binary_pos_express_df(feature_name, accumul_type)
        df_pos_exp_val = df_binary_pos_exp.values # convert to matrix operation

        # cell-marker positive list, 1-D. len = n_cells. Each element indicates the positively expressed marker of that cell index
        # only wants where the x condition is True. x refers to the docs x, not the actual array direction
        # ref: https://numpy.org/doc/stable/reference/generated/numpy.where.html
        cell_marker_pos_list = [np.where(cell)[0] for cell in df_pos_exp_val]

        cell_interaction_in_markers_counts = np.zeros((n_markers, n_markers))
        
        # used to calculate E(x)
        expected_marker_count_1d = np.zeros(n_markers)

        # go through each close proxmity cell pair
        for i, j in zip(i_keep, j_keep):
            # locate the cell via index, then 
            marker_index_neighbor_pair1 = cell_marker_pos_list[i]
            marker_index_neighbor_pair2 = cell_marker_pos_list[j]

            # within each neighbor pair (i.e. pairs of cells) contains the positively expressed markers index in that cell
            # the product of these markers index from each cell indicates interaction pair
            marker_matrix_update_coords = list(product(marker_index_neighbor_pair1, marker_index_neighbor_pair2))
            
            # update the counts between each marker interaction pair
            # example coords: (pos_marker_index_in_cell1, pos_marker_index_in_cell2)
            for coords in marker_matrix_update_coords:
                cell_interaction_in_markers_counts[coords] += 1

            # find the marker index that appeared in both pairs of the neighbor cells
            markers_index_both_neighbor_pair = np.union1d(marker_index_neighbor_pair1, marker_index_neighbor_pair2)
            expected_marker_count_1d[markers_index_both_neighbor_pair] += 1 # increase the markers that appears in either neighborhood pair


        # expected counts
        # expected_marker_count_1d = np.sum(df_pos_exp_val, axis=0)
        # ref: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        expected_counts = np.outer(expected_marker_count_1d, expected_marker_count_1d)

        # expected and observed needs to match dimension to perform element-wise operation
        assert expected_counts.shape == cell_interaction_in_markers_counts.shape

        df_expected_counts = pd.DataFrame(expected_counts, index=df_feature_of_interst.columns, columns=df_feature_of_interst.columns)
        df_cell_interaction_counts = pd.DataFrame(cell_interaction_in_markers_counts, index=df_feature_of_interst.columns, columns=df_feature_of_interst.columns)
        if return_components:
            return df_expected_counts, df_cell_interaction_counts, n_neighbor_pairs
        
        # calculates percentage within function if not return compoenents
        # df_expected_prob = df_expected_counts / n_cells**2
        df_expected_prob = df_expected_counts / n_neighbor_pairs**2

        # theta(i_pos and j_pos)
        df_cell_interaction_prob = df_cell_interaction_counts / n_neighbor_pairs
        
        return df_expected_prob, df_cell_interaction_prob
    

class CytofImageTiff(CytofImage):
    """ 
    CytofImage for Tiff images, inherit from Cytofimage
    """

    def __init__(self, image, slide="", roi="", filename=""):
        self.image = image

        self.markers = None  # markers
        self.labels = None  # labels
        self.slide = slide
        self.roi = roi
        self.filename = filename

        self.channels = None  # ["{}({})".format(marker, label) for (marker, label) in zip(self.markers, self.labels)]

    def copy(self):
        '''
        Creates a deep copy of the current CytofImageTIFF object and return it
        '''
        new_instance = type(self)(self.image.copy(), self.slide, self.roi, self.filename)
        new_instance.markers = copy.deepcopy(self.markers)
        new_instance.labels = copy.deepcopy(self.labels)
        new_instance.channels = copy.deepcopy(self.channels)
        return new_instance
    
    def quality_control(self,  thres: int = 50) -> None:
        setattr(self, "keep", False)
        if any([x < thres for x in self.image.shape]):
            print(f"At least one dimension of the image {self.slide}-{self.roi} is smaller than {thres}, \
                hence exclude from analyzing" )
            self.keep = False

    def set_channels(self, markers: List, labels: List):
        self.markers = markers
        self.labels = labels
        self.channels = ["{}({})".format(marker, label) for (marker, label) in zip(self.markers, self.labels)]
        
    def set_markers(self, 
                markers: list, 
                labels: list, 
                channels: Optional[list] = None
               ):
        """This deprecates set_channels """
        self.raw_markers = markers
        self.raw_labels  = labels
        if channels is not None:
            self.raw_channels = channels
        else:
            self.raw_channels = [f"{marker}-{label}" for (marker, label) in zip(markers, labels)]
        self.channels = self.raw_channels.copy()
        self.markers  = self.raw_markers.copy()
        self.labels   = self.raw_labels.copy()


    def check_channels(self, 
                       channels: Optional[List] = None, 
                       xlim: Optional[List] = None, 
                       ylim: Optional[List] = None, 
                       ncols: int = 5, vis_q: int = 0.9, 
                       colorbar: bool = False, 
                       savedir: Optional[str] = None, 
                       savename: str = "check_channels"):
        """
        xlim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ylim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ncols = number of subplots per row (default=5)
        vis_q = percentile q used to normalize image before visualization  (default=0.9)
        """
        show = True if savedir is None else False
        if channels is not None:
            if not all([cl in self.channels for cl in channels]):
                print("At least one of the channels not available, visualizing all channels instead!")
                channels = None
        if channels is None:  # if no desired channels specified, check all channels
            channels = self.channels
        if len(channels) <= ncols:
            ax_nrow = 1
            ax_ncol = len(channels)
        else:
            ax_ncol = ncols
            ax_nrow = int(np.ceil(len(channels) / ncols))
        fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3 * ax_ncol, 3 * ax_nrow))
        # fig, axes = plt.subplots(ax_nrow, ax_ncol)
        if ax_nrow == 1:
            axes = np.array([axes])
            if ax_ncol == 1:
                axes = np.expand_dims(axes, axis=1)
        for i, _ in enumerate(channels):
            _ax_nrow = int(np.floor(i / ax_ncol))
            _ax_ncol = i % ax_ncol
            _i = self.channels.index(_)
            image = self.image[..., _i]
            percentile_q = np.quantile(image, vis_q) if np.quantile(image, vis_q) != 0 else 1
            image = np.clip(image / percentile_q, 0, 1)
            axes[_ax_nrow, _ax_ncol].set_title(_)
            if xlim is not None:
                image = image[:, xlim[0]:xlim[1]]
            if ylim is not None:
                image = image[ylim[0]:ylim[1], :]
            im = axes[_ax_nrow, _ax_ncol].imshow(image, cmap="gray")
            if colorbar:
                fig.colorbar(im, ax=axes[_ax_nrow, _ax_ncol])
        plt.tight_layout(pad=1.2)
        # axes.axis('scaled')
        if show:
            plt.show()
        else:
            # plt.savefig(os.path.join(savedir, f"{savename}.png"))
            return fig

    def remove_special_channels(self, channels: List):
        for channel in channels:
            if channel not in self.channels:
                print("Channel {} not available, escaping...".format(channel))
                continue
            idx = self.channels.index(channel)
            self.channels.pop(idx)
            self.markers.pop(idx)
            self.labels.pop(idx)
            self.image = np.delete(self.image, idx, axis=2)

            if hasattr(self, "df"):
                self.df.drop(columns=channel, inplace=True)

    def define_special_channels(
        self, 
        channels_dict: Dict, 
        q: float = 0.95, 
        overwrite: bool = False, 
        verbose: bool = False,
        rm_key: str = 'nuclei'):
        channels_rm = []

        # new_name is the key from channels_dict, old_names contains a list of existing channel names
        for new_name, old_names in channels_dict.items():
            if len(old_names) == 0:
                continue
            if new_name in self.channels and (not overwrite):
                print("Warning: {} is already present, skipping...".format(new_name))
                continue
            if new_name in self.channels and overwrite:
                print("Warning: {} is already present, overwriting...".format(new_name))
                idx = self.channels.index(new_name)
                self.image = np.delete(self.image, idx, axis=2)
                self.channels.pop(idx)


            old_nms = []
            for i, old_name in enumerate(old_names):
                if old_name not in self.channels:
                    # warnings.warn('{} is not available!'.format(old_name['marker_name']))
                    warnings.warn('{} is not available!'.format(old_name))

                    continue
                old_nms.append(old_name)
            if verbose:
                print("Defining channel '{}' by summing up channels: {}.".format(new_name, ', '.join(old_nms)))

            if len(old_nms) > 0:

                # only add channels to removal list if matching remove key
                if new_name == rm_key:
                    channels_rm += old_nms
                for i, old_name in enumerate(old_nms):
                    _i = self.channels.index(old_name)
                    _image = self.image[..., _i]
                    percentile_q = np.quantile(_image, q) if np.quantile(_image, q) != 0 else 1
                    _image = np.clip(_image / percentile_q, 0, 1)  # quantile normalization
                    if i == 0:
                        image = _image
                    else:
                        image += _image
                if verbose:
                    print(f"Original image shape: {self.image.shape}")
                self.image = np.dstack([self.image, image[:, :, None]])
                if verbose:
                    print(f"Image shape after defining special channel(s) {self.image.shape}")
                
                if new_name not in self.channels:
                    self.channels.append(new_name)

        if hasattr(self, "defined_channels"):
            for key in channels_dict.keys():
                self.defined_channels.add(key)
        else:
            setattr(self, "defined_channels", set(list(channels_dict.keys())))           
        return channels_rm

# Define a function to apply the threshold and convert to binary
def apply_threshold_to_column(column, threshold):
    """
    Apply a threshold to a column of data and convert it to binary.

    @param column: The input column of data to be thresholded.
    @param threshold: The threshold value to compare the elements in the column.
    
    @return: A binary array where True represents values meeting or exceeding the threshold,
             and False represents values below the threshold.
    """
    return (column >= threshold)
    
class CytofCohort():
    def __init__(self, cytof_images: Optional[dict] = None, 
                 df_cohort: Optional[pd.DataFrame] = None, 
                 dir_out: Optional[str] = "./", 
                 cohort_name: str = "cohort1"):
        """
        cytof_images: 
        df_cohort: Slide | ROI | input file
        """
        self.cytof_images = cytof_images or {}
        self.df_cohort    = df_cohort# or None# pd.read_csv(file_cohort) # the slide-ROI
        self.feat_sets = {
            "all": ["cell_sum", "cell_ave", "cell_morphology"],
            "cell_sum": ["cell_sum", "cell_morphology"],
            "cell_ave": ["cell_ave", "cell_morphology"],
            "cell_sum_only": ["cell_sum"],
            "cell_ave_only": ["cell_ave"]
        }
        
        self.name = cohort_name
        self.dir_out = os.path.join(dir_out, self.name) if isinstance(dir_out, str) else None 
        if self.dir_out:
            os.makedirs(self.dir_out, exist_ok=True)
            print('Output folder created:', self.dir_out)        

    def __getitem__(self, key):
        'Extracts a particular cytof image from the cohort'
        return self.cytof_images[key]
        
    def __str__(self):
        return f"CytofCohort {self.name}"

    def __repr__(self):
        return f"CytofCohort(name={self.name})"
    
    def save_cytof_cohort(self):
        if self.dir_out:
            save_path = f'{os.path.join(self.dir_out, self.name)}.pkl'
            pkl.dump(self, open(save_path, "wb"))

            return save_path
        else:
            raise FileNotFoundError('self.dir_out not specified')


    def batch_process_feature(self):
        """
        Batch process: if the CytofCohort is initialized by a dictionary of CytofImages
        """
        
        slides, rois, fs_input = [], [], []
        for n, cytof_img in self.cytof_images.items():
            if not hasattr(self, "dict_feat"):
                setattr(self, "dict_feat", cytof_img.features)
            if not hasattr(self, "markers"):
                setattr(self, "markers", cytof_img.markers)

            try:
                qs &= set(list(cytof_img.dict_quantiles.keys()))
            except:
                qs = set(list(cytof_img.dict_quantiles.keys()))

            slides.append(cytof_img.slide)
            rois.append(cytof_img.roi)
            fs_input.append(cytof_img.filename) #df_feature['filename'].unique()[0])
        
        setattr(self, "normqs", qs)
        # scale feature (in a batch)
        df_scale_params = self.scale_feature()
        setattr(self, "df_scale_params", df_scale_params)
        if self.df_cohort is None:
            self.df_cohort = pd.DataFrame({"Slide": slides, "ROI": rois, "input file": fs_input})        


    def batch_process(self, params: Dict):
        sys.path.append("../CLIscripts")
        from process_single_roi import process_single, SetParameters
        
        success_rows = []
        for i, (slide, roi, fname) in self.df_cohort.iterrows():
            paramsi = SetParameters(filename=fname,
                outdir=self.dir_out,
                label_marker_file=params.get('label_marker_file', None),
                slide=slide,
                roi=roi,
                quality_control_thres=params.get("quality_control_thres", 50),
                channels_remove=params.get("channels_remove", None),
                channels_dict=params.get("channels_dict", None),
                use_membrane=params.get("use_membrane",True),
                cell_radius=params.get("cell_radius", 5),
                normalize_qs=params.get("normalize_qs", 75),
                iltype=params.get('iltype', None))

            try:
                cytof_img = process_single(paramsi, downstream_analysis=False, verbose=False)
                self.cytof_images[f"{slide}_{roi}"] = cytof_img
                
                # image successfully processed, record index
                success_rows.append(i)
            
            except Exception as e:
                print(f"Skipping {slide}_{roi} due to error: {e}")
                continue
        
        # update df_cohort to contain only successfully calculated rows
        self.df_cohort = self.df_cohort.loc[success_rows].reset_index(drop=True)
            

    def get_feature(self, 
                    normq: int = 75, 
                    feat_type: str = "normed_scaled", 
                    verbose: bool = False):
        """ 
        Get a specific set of feature for the cohort
        The set is defined by `normq` and `feat_type`
        """
        
        assert feat_type in ["normed_scaled", "normed", ""], f"feature type {feat_type} not supported!"
        
        if feat_type != "" and not hasattr(self, "df_feature"):
            orig_dfs = {}
            for f_roi, cytof_img in self.cytof_images.items():
                orig_dfs[f_roi] = getattr(cytof_img, "df_feature")
            setattr(self, "df_feature", pd.concat([_ for key, _ in orig_dfs.items()]).reset_index(drop=True))

        feat_name = feat_type if feat_type=="" else f"_{normq}{feat_type}"
        n_attr    = f"df_feature{feat_name}"
              
        dfs = {}
        for f_roi, cytof_img in self.cytof_images.items():
            dfs[f_roi] = getattr(cytof_img, n_attr)
        setattr(self, n_attr, pd.concat([_ for key, _ in dfs.items()]).reset_index(drop=True))
        if verbose:
            print("The attribute name of the feature: {}".format(n_attr))
        
    def scale_feature(self):
        """Scale features for all normalization q values"""
        cytof_img = list(self.cytof_images.values())[0]
        # features to be scaled
        s_features = [col for key, features in cytof_img.features.items() \
                              for f in features \
                              for col in cytof_img.df_feature.columns if col.startswith(f)]

        for normq in self.normqs:
            n_attr = f"df_feature_{normq}normed"
            n_attr_scaled = f"df_feature_{normq}normed_scaled"

            if not hasattr(self, n_attr):
                self.get_feature(normq=normq, feat_type="normed")

            df_feature = getattr(self, n_attr)

            # calculate scaling parameters
            df_scale_params = df_feature[s_features].mean().to_frame(name="mean").transpose()
            df_scale_params = pd.concat([df_scale_params, df_feature[s_features].std().to_frame(name="std").transpose()])

            # 
            m = df_scale_params[df_scale_params.columns].iloc[0] # mean
            s = df_scale_params[df_scale_params.columns].iloc[1] # std.dev

            df_feature_scale = copy.deepcopy(df_feature)

            assert len([x for x in df_scale_params.columns if x not in df_scale_params.columns]) == 0

            # scale
            df_feature_scale[df_scale_params.columns] = (df_feature_scale[df_scale_params.columns] - m) / s
            setattr(self, n_attr_scaled, df_feature_scale)
        return df_scale_params
    
    def _get_feature_subset(self, 
                           normq: int = 75, 
                           feat_type: str = "normed_scaled", 
                           feat_set: str = "all", 
                           markers: str = "all", 
                           verbose: bool = False):

        assert feat_type in ["normed_scaled", "normed", ""], f"feature type {feat_type} not supported!"
        assert (markers == "all" or isinstance(markers, list))
        assert feat_set in self.feat_sets.keys(), f"feature set {feat_set} not supported!"
        
        description = "original" if feat_type=="" else f"{normq}{feat_type}"
        n_attr      = f"df_feature{feat_type}" if feat_type=="" else f"df_feature_{normq}{feat_type}" # the attribute name to achieve from cytof_img
        
        if not hasattr(self, n_attr):
            self.get_feature(normq, feat_type)
        if verbose:
            print("\nThe attribute name of the feature: {}".format(n_attr))

        feat_names = [] # a list of feature names
        for y in self.feat_sets[feat_set]:
            if "morphology" in y:
                feat_names += self.dict_feat[y]
            else:
                if markers == "all": # features extracted from all markers are kept
                    feat_names += self.dict_feat[y]
                    markers = self.markers
                else: # only features correspond to markers kept (markers are a subset of self.markers)
                    ids = [self.markers.index(x) for x in markers] # TODO: the case where marker in markers not in self.markers???
                    feat_names += [self.dict_feat[y][x] for x in ids]
        
        df_feature = getattr(self, n_attr)[feat_names]
        return df_feature, markers, feat_names, description, n_attr
    
    ###############################################################
    ################## PhenoGraph Clustering ######################
    ###############################################################
    def clustering_phenograph(self, 
                              normq:int = 75, 
                              feat_type:str = "normed_scaled", 
                              feat_set: str = "all", 
                              pheno_markers: Union[str, List] = "all", 
                              k: int = None, 
                              save_vis: bool = False,
                              verbose:bool = True):
        
        if pheno_markers == "all":
            pheno_markers_ = "_all"
        else:
            pheno_markers_ = "_subset1"

        assert feat_type in ["normed_scaled", "normed", ""], f"feature type {feat_type} not supported!"
        df_feature, pheno_markers, feat_names, description, n_attr = self._get_feature_subset(normq=normq,
                                                                                          feat_type=feat_type,
                                                                                          feat_set=feat_set,
                                                                                          markers=pheno_markers,
                                                                                          verbose=verbose)
        # set number of nearest neighbors k and run PhenoGraph for phenotype clustering
        k = k if k else int(df_feature.shape[0] / 100) 
        if k < 10:
            k = min(df_feature.shape[0]-1, 10)

            # perform k-means algorithm for small k
            kmeans = KMeans(n_clusters=k, random_state=42).fit(df_feature)
            communities = kmeans.labels_
        else: 
            communities, graph, Q = phenograph.cluster(df_feature, k=k, n_jobs=-1)   # run PhenoGraph

        # project to 2D using UMAP
        umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap_2d.fit_transform(df_feature)
        
        if not hasattr(self, "phenograph"):
            setattr(self, "phenograph", {})
        key_pheno  = f"{description}_{feat_set}_feature_{k}"
        key_pheno += f"{pheno_markers_}_markers" 
            
            
        N = len(np.unique(communities))
        self.phenograph[key_pheno] = {
            "data": df_feature,
            "markers": pheno_markers,
            "features": feat_names,
            "description": {"normalization": description, "feature_set": feat_set}, # normalization and/or scaling | set of feature (in self.feat_sets)
            "communities": communities, 
            "proj_2d": proj_2d, 
            "N": N,
            "feat_attr": n_attr
        }
        
        if verbose:
            print(f"\n{N} communities found. The dictionary key for phenograph: {key_pheno}.")
        return key_pheno

    def _gather_roi_pheno(self, key_pheno):
        """Split whole df into df for each ROI"""
        df_slide_roi    = self.df_cohort
        pheno_out       = self.phenograph[key_pheno]
        df_feat_all     = getattr(self, pheno_out['feat_attr']) # original feature (to use the slide/ roi /filename info) data
        df_pheno_all    = pheno_out['data'] # phenograph data
        proj_2d_all     = pheno_out['proj_2d']
        communities_all = pheno_out['communities']  

        df_feature_roi, proj_2d_roi, communities_roi = {}, {}, {}        
        for i in self.df_cohort.index:       # Slide | ROI | input file
#             path_i = df_slide_roi.loc[i, "path"]
            roi_i  = df_slide_roi.loc[i, "ROI"]
            f_in   = df_slide_roi.loc[i, "input file"]# os.path.join(path_i, roi_i)
            cond   = df_feat_all["filename"] == f_in
            df_feature_roi[roi_i] = df_pheno_all.loc[cond, :] 
            proj_2d_roi[roi_i] = proj_2d_all[cond, :] 
            communities_roi[roi_i] = communities_all[cond] 
        return df_feature_roi, proj_2d_roi, communities_roi

    def vis_phenograph(self,
                       key_pheno: str,
                       level: str = "cohort",
                       accumul_type: Union[List[str], str] = "cell_sum",  # ["cell_sum", "cell_ave"]
                       normalize: bool = False,
                       save_vis: bool = False,
                       show_plots: bool = False,
                       plot_together: bool = True,
                       fig_width: int = 5 # only when plot_together is True
                       ):
        assert level.upper() in ["COHORT", "SLIDE", "ROI"], "Only 'cohort', 'slide' and 'roi' are accetable values for level"
        this_pheno = self.phenograph[key_pheno]
        feat_names = this_pheno['features']
        descrip = this_pheno['description']
        n_community = this_pheno['N']
        markers = this_pheno['markers']
        feat_set = self.feat_sets[descrip['feature_set']]

        if save_vis:
            vis_savedir = os.path.join(self.dir_out, "phenograph", key_pheno + f"-{n_community}clusters")
            if not os.path.exists(vis_savedir):
                os.makedirs(vis_savedir)
        else:
            vis_savedir = None

        if accumul_type is None:  # by default, visualize all accumulation types
            accumul_type = [_ for _ in feat_set if "morphology" not in _]
        if isinstance(accumul_type, str):
            accumul_type = [accumul_type]

        proj_2d = this_pheno['proj_2d']
        df_feature = this_pheno['data']
        communities = this_pheno['communities']

        if level.upper() == "COHORT":
            proj_2ds = {"cohort": proj_2d}
            df_feats = {"cohort": df_feature}
            commus = {"cohort": communities}
        else:
            df_feats, proj_2ds, commus = self._gather_roi_pheno(key_pheno)
            if level.upper() == "SLIDE":
                for slide in self.df_cohort["Slide"].unique():  # for each slide

                    f_rois = [roi_i.replace(".txt", "") for roi_i in
                              self.df_cohort.loc[self.df_cohort["Slide"] == slide, "ROI"]]
                    df_feats[slide] = pd.concat([df_feats[f_roi] for f_roi in f_rois])
                    proj_2ds[slide] = np.concatenate([proj_2ds[f_roi] for f_roi in f_rois])
                    commus[slide] = np.concatenate([commus[f_roi] for f_roi in f_rois])
                    for f_roi in f_rois:
                        df_feats.pop(f_roi)
                        proj_2ds.pop(f_roi)
                        commus.pop(f_roi)
        
        figs = {} # if plot_together

        figs_scatter = {} # if not plot_together
        figs_exps    = {}

        cluster_protein_exps = {}
        for key, df_feature in df_feats.items():
            if plot_together:
                ncol = len(accumul_type)+1
                fig, axs = plt.subplots(1,ncol, figsize=(ncol*fig_width, fig_width))
            proj_2d = proj_2ds[key]
            commu = commus[key]
            # Visualize 1: plot 2d projection together
            print("Visualization in 2d - {}-{}".format(level, key))
            savename = os.path.join(vis_savedir, f"cluster_scatter_{level}_{key}.png") if (save_vis and not plot_together) else None
            ax = axs[0] if plot_together else None
            fig_scatter = visualize_scatter(data=proj_2d, communities=commu, n_community=n_community, 
                                            title=key, savename=savename, show=show_plots, ax=ax)
            figs_scatter[key] = fig_scatter
            
            figs_exps[key]    = {}
            # Visualize 2: protein expression
            for axid, acm_tpe in enumerate(accumul_type):
                ids = [i for (i, x) in enumerate(feat_names) if re.search(".{}".format(acm_tpe), x)]
                feat_names_ = [feat_names[i] for i in ids]

                cluster_protein_exp = np.zeros((n_community, len(markers)))

                group_ids = np.arange(len(np.unique(communities)))
                for cluster in range(len(np.unique(communities))):  # for each (global) community
                    df_sub = df_feature.loc[commu == cluster]
                    if df_sub.shape[0] == 0:
                        group_ids = np.delete(group_ids, group_ids == cluster)
                        continue
                    
                    # number of markers should match # of features extracted.
                    for i, feat in enumerate(feat_names_):
                        cluster_protein_exp[cluster, i] = np.average(df_sub[feat])

                # get rid of non-exist clusters
                '''cluster_protein_exp = cluster_protein_exp[group_ids, :]'''
                if normalize:
                    cluster_protein_exp_norm = cluster_protein_exp - np.median(cluster_protein_exp, axis=0)
                    # or set non-exist cluster to be inf
                    rid = set(np.arange(len(np.unique(communities)))) - set(group_ids)
                    if len(rid) > 0:
                        rid = np.array(list(rid))
                        cluster_protein_exp_norm[rid, :] = np.nan
                        group_ids = np.arange(len(np.unique(communities)))
                savename = os.path.join(vis_savedir, f"protein_expression_{level}_{acm_tpe}_{key}.png") \
                    if (save_vis and not plot_together) else None
                vis_exp = cluster_protein_exp_norm if normalize else cluster_protein_exp
                ax = axs[axid+1] if plot_together else None
                fig_exps = visualize_expression(data=vis_exp, markers=markers,
                                                group_ids=group_ids, title="{} - {}-{}".format(level, acm_tpe, key), 
                                                savename=savename, show=show_plots, ax=ax)
                figs_exps[key][acm_tpe]   = fig_exps
                cluster_protein_exps[key] = vis_exp
            plt.tight_layout()
            if plot_together:
                figs[key] = fig
                if save_vis:
                    plt.savefig(os.path.join(vis_savedir, f"phenograph_{level}_{acm_tpe}_{key}.png"), dpi=300)
                if show_plots:
                    plt.show()
            if not show_plots:
                plt.close("all")
        return df_feats, commus, cluster_protein_exps, figs, figs_scatter, figs_exps


    def attach_individual_roi_pheno(self, key_pheno, override=False):
        """ Attach PhenoGraph outputs to each individual CytofImage (roi) and update each saved CytofImage
        """
        assert key_pheno in self.phenograph.keys(), "Pheno-Graph with {} not available!".format(key_pheno)
        phenograph = self.phenograph[key_pheno]  # data, markers, features, description, communities, proj_2d, N
        
        for n, cytof_img in self.cytof_images.items():
            if not hasattr(cytof_img, "phenograph"):
                setattr(cytof_img, "phenograph", {})
            if key_pheno in cytof_img.phenograph and not override:
                print("\n{} already attached for {}-{}, skipping ... ".format(key_pheno, cytof_img.slide, cytof_img.roi))
                continue

            cond = self.df_feature['filename'] == cytof_img.filename  # cytof_img.filename: original file name
            data = phenograph['data'].loc[cond, :]

            communities = phenograph['communities'][cond.values]
            proj_2d = phenograph['proj_2d'][cond.values]

            # phenograph for this image
            this_phenograph = {"data": data,
                               "markers": phenograph["markers"],
                               "features": phenograph["features"],
                               "description": phenograph["description"],
                               "communities": communities,
                               "proj_2d": proj_2d,
                               "N": phenograph["N"]
                               }

            cytof_img.phenograph[key_pheno] = this_phenograph



    def _gather_roi_kneighbor_graphs(self, key_pheno: str, method: str = "distance", **kwars: dict) -> dict:
        """ Define adjacency community for each cell based on either k-nearest neighbor or distance
        Args:
            key_pheno: dictionary key for a specific phenograph output
            method: method to construct the adjacency matrix, choose from "distance" and "kneighbor"
            **kwargs: used to specify distance threshold (thres) for "distance" method or number of neighbors (k)
            for "kneighbor" method
        Output:
            network: (dict) ROI level network that will be used for cluster interaction analysis
        """
        assert method in ["distance", "kneighbor"], "Method can be either 'distance' or 'kneighbor'!"
        default_thres = {
            "thres": 50,
            "k": 8
        }
        _ = "k" if method == "kneighbor" else "thres"
        thres = kwars.get(_, default_thres[_])
        print("{}: {}".format(_, thres))
        df_pheno_feat = getattr(self, self.phenograph[key_pheno]['feat_attr'])
        n_cluster = self.phenograph[key_pheno]['N']
        cluster = self.phenograph[key_pheno]['communities']
        df_slide_roi = getattr(self, "df_cohort")

        networks = {}
        if method == "kneighbor":  # construct K-neighbor graph
            for i, row in df_slide_roi.iterrows(): #for i in df_slide_roi.index:       # Slide | ROI | input file
                slide, roi, f_in = row["Slide"], row["ROI"], row["input file"]
                cond = df_pheno_feat['filename'] == f_in
                if cond.sum() == 0:
                    continue
                _cluster = cluster[cond.values]
                df_sub = df_pheno_feat.loc[cond, :]
                graph = skgraph(np.array(df_sub.loc[:, ['coordinate_x', 'coordinate_y']]),
                                n_neighbors=thres, mode='distance')
                graph.toarray()
                I, J, V = sp.find(graph)
                networks[roi] = dict()
                networks[roi]['I'] = I  # from cell
                networks[roi]['J'] = J  # to cell
                networks[roi]['V'] = V  # distance value
                networks[roi]['network'] = graph

                # Edge type summary
                edge_nums = np.zeros((n_cluster, n_cluster))
                for _i, _j in zip(I, J):
                    edge_nums[_cluster[_i], _cluster[_j]] += 1
                networks[roi]['edge_nums'] = edge_nums

                expected_percentage = np.zeros((n_cluster, n_cluster))
                for _i in range(n_cluster):
                    for _j in range(n_cluster):
                        expected_percentage[_i, _j] = sum(_cluster == _i) * sum(_cluster == _j)  # / len(df_sub)**2
                networks[roi]['expected_percentage'] = expected_percentage
                networks[roi]['num_cell'] = len(df_sub)
        else:  # construct neighborhood matrix using distance cut-off
            cal_dist = DistanceMetric.get_metric('euclidean')
            for i, row in df_slide_roi.iterrows(): #for i in df_slide_roi.index:       # Slide | ROI | input file
                slide, roi, f_in = row["Slide"], row["ROI"], row["input file"]
                cond = df_pheno_feat['filename'] == f_in
                if cond.sum() == 0:
                    continue
                networks[roi] = dict()
                _cluster = cluster[cond.values]
                df_sub = df_pheno_feat.loc[cond, :]
                dist = cal_dist.pairwise(df_sub.loc[:, ['coordinate_x', 'coordinate_y']].values)
                networks[roi]['dist'] = dist

                # expected percentage
                expected_percentage = np.zeros((n_cluster, n_cluster))
                for _i in range(n_cluster):
                    for _j in range(n_cluster):
                        expected_percentage[_i, _j] = sum(_cluster == _i) * sum(_cluster == _j)  # / len(df_sub)**2
                networks[roi]['expected_percentage'] = expected_percentage
                n_cells = len(df_sub)

                # edge num
                edge_nums = np.zeros_like(expected_percentage)
                for _i in range(n_cells):
                    for _j in range(n_cells):
                        if dist[_i, _j] > 0 and dist[_i, _j] < thres:
                            edge_nums[_cluster[_i], _cluster[_j]] += 1
                networks[roi]['edge_nums'] = edge_nums
                networks[roi]['num_cell'] = n_cells
        return networks

    def cluster_interaction_analysis(self, key_pheno, method="distance", level="slide", clustergrid=None, viz=False, **kwars):
        """Interaction analysis for clusters

        """
        assert method in ["distance", "kneighbor"], "Method can be either 'distance' or 'kneighbor'!"
        assert level in ["slide", "roi"], "Level can be either 'slide' or 'roi'!"
        default_thres = {
            "thres": 50,
            "k": 8
        }
        _ = "k" if method == "kneighbor" else "thres"
        thres = kwars.get(_, default_thres[_])
        """print("{}: {}".format(_, thres))"""
        networks = self._gather_roi_kneighbor_graphs(key_pheno, method=method, **{_: thres})

        if level == "slide":
            keys = ['edge_nums', 'expected_percentage', 'num_cell']
            for slide in self.df_cohort['Slide'].unique():
                cond = self.df_cohort['Slide'] == slide
                df_slide = self.df_cohort.loc[cond, :]
                rois = df_slide['ROI'].values
                '''keys = list(networks.values())[0].keys()'''
                networks[slide] = {}
                for key in keys:
                    networks[slide][key] = sum([networks[roi][key] for roi in rois if roi in networks])
                for roi in rois:
                    if roi in networks:
                        networks.pop(roi)

        interacts = {}
        epsilon = 1e-6
        for key, item in networks.items():
            edge_percentage = item['edge_nums'] / np.sum(item['edge_nums'])
            expected_percentage = item['expected_percentage'] / item['num_cell'] ** 2
            
            # Normalize
            interact_norm = np.log10(edge_percentage / (expected_percentage+epsilon) + epsilon)
            interact_norm[interact_norm == np.log10(epsilon)] = 0
            interacts[key] = interact_norm

        # plot
        for f_key, interact in interacts.items():
            plt.figure(figsize=(6, 6))
            ax = sns.heatmap(interact, center=np.log10(1 + epsilon),
                             cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_aspect('equal')
            plt.title(f_key)
            plt.show()

            if clustergrid is None:
                plt.figure()
                clustergrid = sns.clustermap(interact, center=np.log10(1 + epsilon),
                                             cmap='RdBu_r', vmin=-1, vmax=1,
                                             xticklabels=np.arange(interact.shape[0]),
                                             yticklabels=np.arange(interact.shape[0]),
                                             figsize=(6, 6))

                plt.title(f_key)
                plt.show()

            plt.figure()
            sns.clustermap(interact[clustergrid.dendrogram_row.reordered_ind, :] \
                               [:, clustergrid.dendrogram_row.reordered_ind],
                           center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=1,
                           xticklabels=clustergrid.dendrogram_row.reordered_ind,
                           yticklabels=clustergrid.dendrogram_row.reordered_ind,
                           figsize=(6, 6), row_cluster=False, col_cluster=False)
            plt.title(f_key)
            plt.show()

        # IMPORTANT: attch to individual ROIs
        self.attach_individual_roi_pheno(key_pheno, override=True)
        return interacts, clustergrid
    
    
    ###############################################################
    ###################### Marker Level ###########################
    ###############################################################

    def generate_summary(self, 
                        feat_type: str = "normed", 
                        normq: int = 75, 
                        vis_thres: bool = False, 
                        accumul_type: Union[List[str], str] = "sum",
                        verbose: bool = False,
                        get_thresholds: Callable = _get_thresholds,
                        ) -> List:

        """ Generate marker positive summaries and attach to each individual CyTOF image in the cohort
        """
        accumul_type = [accumul_type] if isinstance(accumul_type, str) else accumul_type
        assert feat_type in ["normed_scaled", "normed", ""], f"feature type {feat_type} not supported!"
        feat_name = f"{feat_type}" if feat_type=="" else f"{normq}{feat_type}" # the attribute name to achieve from cytof_img
        n_attr = f"df_feature{feat_name}" if feat_type=="" else f"df_feature_{feat_name}" # the attribute name to achieve from cytof_img
        df_feat = getattr(self, n_attr)

        # get thresholds
        thres = getattr(self, "marker_thresholds", {})
        thres[f"{normq}_{feat_type}"] = {}
        for _ in accumul_type: # for either marker sum or marker average
            print(f"Getting thresholds for cell {_} of all markers.")
            thres[f"{normq}_{feat_type}"][f"cell_{_}"] = get_thresholds(df_feature=df_feat, 
                                                                        features=self.dict_feat[f"cell_{_}"], 
                                                                        visualize=vis_thres,
                                                                        verbose=verbose)
        setattr(self, "marker_thresholds", thres)

        # split to each ROI
        _attr_marker_pos, seen = [], 0
        self.df_cohort['Slide_ROI'] = self.df_cohort[['Slide', 'ROI']].agg('_'.join, axis=1) 
        for n, cytof_img in self.cytof_images.items(): # ({slide}_{roi}, CytofImage)
            if not hasattr(cytof_img, n_attr): # cytof_img object instance may not contain _scaled feature
                cond = self.df_cohort['Slide_ROI'] == n
                input_file = self.df_cohort.loc[self.df_cohort['Slide_ROI'] == n, 'input file'].values[0]
                _df_feat = df_feat.loc[df_feat['filename'] == input_file].reset_index(drop=True)
                setattr(cytof_img, n_attr, _df_feat)
            else:
                _df_feat = getattr(cytof_img, n_attr)
            for _ in accumul_type: #["sum", "ave"]: # for either marker sum or marker average accumulation
                
                attr_marker_pos = cytof_img.marker_positive_summary(
                    thresholds=thres[f"{normq}_{feat_type}"][f"cell_{_}"], 
                    feat_type=feat_type,
                    normq=normq,
                    accumul_type=_
                )  
                if seen == 0:
                    _attr_marker_pos.append(attr_marker_pos)
            seen += 1
        return _attr_marker_pos

    def co_expression_analysis(self, 
                                normq: int = 75,
                                feat_type: str = "normed", 
                                co_exp_markers: Union[str, List] = "all", 
                                accumul_type: Union[str, List[str]] = "sum", 
                                verbose: bool = False,
                                clustergrid=None):


        # parameter checks and preprocess for analysis
        assert feat_type in ["original", "normed", "scaled"]
        if feat_type == "original":
            feat_name = ""
        elif feat_type == "normed":
            feat_name = f"{normq}normed"
        else:
            feat_name = f"{normq}normed_scaled" 

        # go through each roi, get their binary marker-cell expression
        roi_binary_express_dict = dict()
        for i, cytof_img in enumerate(self.cytof_images.values()):
            slide, roi = cytof_img.slide, cytof_img.roi
            df_binary_pos_exp = cytof_img.get_binary_pos_express_df(feat_name, accumul_type)
            roi_binary_express_dict[roi] = df_binary_pos_exp
        
        df_slide_roi = self.df_cohort

        # in cohort analysis, co-expression is always analyzed per Slide.
        # per ROI analysis can be done by calling the cytof_img individually
        slide_binary_express_dict = dict()

        # concatenate all ROIs into one, for each slide
        for slide in df_slide_roi["Slide"].unique():
            rois_of_one_slide = df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"]

            for i, filename_roi in enumerate(rois_of_one_slide):
                ind_roi = filename_roi.replace('.txt', '')

                if ind_roi not in roi_binary_express_dict:
                    print(f'ROI {ind_roi} in self.df_cohort, but not found in co-expression dicts')
                    continue
                
                try: # adding to existing slide key
                    # append dataframe row-wise, then perform co-expression analysis at the slide level 
                    slide_binary_express_dict[slide] = pd.concat([slide_binary_express_dict[slide], roi_binary_express_dict[ind_roi]], ignore_index=True)
                except KeyError: # # first iteration writing to slide, couldn't find the slide key
                    slide_binary_express_dict[slide] = roi_binary_express_dict[ind_roi].copy()

        slide_co_expression_dict = dict()

        # for each slide, perform co-expression analysis
        for slide_key, large_binary_express in slide_binary_express_dict.items():

            n_cells, n_markers = large_binary_express.shape
            df_pos_exp_val = large_binary_express.values

            # list all pair-wise combinations of the markers
            column_combinations = list(product(range(n_markers), repeat=2))

            # step to the numerator of the log odds ratio
            co_positive_prob_matrix = np.zeros((n_markers, n_markers))

            # step to the denominator of the log odds ratio
            expected_prob_matrix = np.zeros((n_markers, n_markers))

            for combo in column_combinations:
                marker1, marker2 = combo

                # count cells that positively expresses in both marker 1 and 2
                positive_prob_marker1_and_2 = np.sum(np.logical_and(df_pos_exp_val[:, marker1], df_pos_exp_val[:, marker2])) / n_cells
                co_positive_prob_matrix[marker1, marker2] = positive_prob_marker1_and_2
                
                # pair (A,B) counts is the same as pair (B,A) counts
                co_positive_prob_matrix[marker2, marker1] = positive_prob_marker1_and_2

                # count expected cells if marker 1 and 2 are independently expressed
                # p(A and B) = p(A) * p(B) = num_pos_a * num_pos_b / (num_cells * num_cells)
                # p(A) = number of positive cells / number of cells
                exp_prob_in_marker1_and_2 = np.sum(df_pos_exp_val[:, marker1]) * np.sum(df_pos_exp_val[:, marker2]) / n_cells**2
                expected_prob_matrix[marker1, marker2] = exp_prob_in_marker1_and_2
                expected_prob_matrix[marker2, marker1] = exp_prob_in_marker1_and_2

            # theta(i_pos and j_pos)
            df_co_pos = pd.DataFrame(co_positive_prob_matrix, index=df_binary_pos_exp.columns, columns=df_binary_pos_exp.columns)

            # E(x)
            df_expected = pd.DataFrame(expected_prob_matrix, index=df_binary_pos_exp.columns, columns=df_binary_pos_exp.columns)
            
            epsilon = 1e-6 # avoid divide by 0 or log(0)

            # Normalize and fix Nan
            edge_percentage_norm = np.log10(df_co_pos.values / (df_expected.values+epsilon) + epsilon)
            
            # if observed/expected = 0, then log odds ratio will have log10(epsilon)
            # no observed means co-expression cannot be determined, does not mean strong negative co-expression
            edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0
    
            slide_co_expression_dict[slide_key] = (edge_percentage_norm, df_expected.columns)

        return slide_co_expression_dict
