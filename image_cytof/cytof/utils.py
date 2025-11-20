import os
import pickle as pkl
import skimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import scipy
from typing import Union, Optional, Type, Tuple, List, Dict
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from readimc import MCDFile, TXTFile
import warnings



def load_CytofImage(savename):
    cytof_img = pkl.load(open(savename, "rb"))
    return cytof_img

def load_CytofCohort(savename):
    cytof_cohort = pkl.load(open(savename, "rb"))
    return cytof_cohort


def process_mcd(filename: str, 
                params: Dict):

    """
    A function to process a whole slide .mcd file
    """


    from classes import CytofImageTiff, CytofCohort
    quality_control_thres = params.get("quality_control_thres", None)
    channels_remove       = params.get("channels_remove", None)
    channels_dict         = params.get("channels_dict", None)
    use_membrane          = params.get("use_membrane", False)
    cell_radius           = params.get("cell_radius", 5)
    normalize_qs          = params.get("normalize_qs", 75)
    
    df_cohort    = pd.DataFrame(columns = ['Slide', 'ROI', 'input file'])
    cytof_images = {}
    corrupted    = []
    with MCDFile(filename) as f:
        for slide in f.slides:
            sid = f"{slide.description}{slide.id}"
            print(sid)
            for roi in slide.acquisitions:
                rid = roi.description
                print(f'processing slide_id-roi: {sid}-{rid}')

                if roi.metadata["DataStartOffset"] < roi.metadata["DataEndOffset"]:
                    img_roi = f.read_acquisition(roi)  # array, shape: (c, y, x), dtype: float3
                    img_roi = np.transpose(img_roi, (1, 2, 0))
                    cytof_img = CytofImageTiff(slide=sid, roi = rid, image=img_roi, filename=f"{sid}-{rid}")
                    
                    # cytof_img.quality_control(thres=quality_control_thres)
                    channels = [f"{mk}({cn})" for (mk, cn) in zip(roi.channel_labels, roi.channel_names)]
                    cytof_img.set_markers(markers=roi.channel_labels, labels=roi.channel_names, channels=channels) # targets, metals
                    
                    # known corrupted channels, e.g. nan-nan1
                    if channels_remove is not None and len(channels_remove) > 0:
                        cytof_img.remove_special_channels(channels_remove)

                    # maps channel names to nuclei/membrane
                    if channels_dict is not None:

                        # remove nuclei channel for segmentation
                        channels_rm = cytof_img.define_special_channels(channels_dict, rm_key='nuclei')
                        cytof_img.remove_special_channels(channels_rm)
                        cytof_img.get_seg(radius=cell_radius, use_membrane=use_membrane)
                        cytof_img.extract_features(cytof_img.filename)
                        cytof_img.feature_quantile_normalization(qs=normalize_qs)

                    df_cohort = pd.concat([df_cohort, pd.DataFrame.from_dict([{'Slide': sid, 
                                                                   'ROI': rid, 
                                                                   'input file': filename}])])
                    cytof_images[f"{sid}-{rid}"] = cytof_img
                else:
                    corrupted.append(f"{sid}-{rid}")
    print(f"This cohort now contains {len(cytof_images)} ROIs, after excluding {len(corrupted)} corrupted ones from the original MCD.")
    
    cytof_cohort = CytofCohort(cytof_images=cytof_images, df_cohort=df_cohort)
    if channels_dict is not None:
        cytof_cohort.batch_process_feature()
    else:
        warnings.warn("Feature extraction is not done as no nuclei channels defined by 'channels_dict'!")
    return corrupted, cytof_cohort#, cytof_images


def save_multi_channel_img(img, savename):
    """
    A helper function to save multi-channel images
    """
    skimage.io.imsave(savename, img)

    
def generate_color_dict(names: List,
                        sort_names: bool = True,
                       ):
    """
    Randomly generate a dictionary of colors based on provided "names"
    """
    if sort_names:
        names.sort()

    color_dict = dict((n, plt.cm.get_cmap('tab20').colors[i]) for (i, n) in enumerate(names))
    return color_dict

def show_color_table(color_dict: dict, # = None,
                   # names: List = ['1'],
                   title: str = "",
                   maxcols: int = 4,
                   emptycols: int = 0,
                   # sort_names: bool = True,
                   dpi: int = 72,
                   cell_width: int = 212,
                   cell_height: int = 22,
                   swatch_width: int = 48,
                   margin: int = 12,
                   topmargin: int = 40,
                   show: bool = True
                   ):
    """
    Show color dictionary
    Generate the color table for visualization.
    If "color_dict" is provided, show color_dict;
    otherwise, randomly generate color_dict based on "names"
    reference: https://matplotlib.org/stable/gallery/color/named_colors.html
    args:
        color_dict (optional) = a dictionary of colors. key: color legend name - value: RGB representation of color
        names (optional) = names for each color legend (default=["1"])
        title (optional) = title for the color table (default="")
        maxcols = maximum number of columns in visualization
        emptycols (optional) = number of empty columns for a maxcols-column figure,
            i.e. maxcols=4 and emptycols=3 means presenting single column plot (default=0)
        sort_names (optional) = a flag indicating whether sort colors based on names (default=True)
    """

#     if sort_names:
#         names.sort()

#     if color_pool is None:
#         color_pool = dict((n, plt.cm.get_cmap('tab20').colors[i]) for (i, n) in enumerate(names))
#     else:
    names = color_dict.keys()

    n = len(names)
    ncols = maxcols - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    #     width  = cell_width * 4 + 2 * margin
    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + margin + topmargin

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    #     ax.set_xlim(0, cell_width * 4)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, loc="left", pad=10)

    for i, n in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, n, fontsize=12,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=color_dict[n], edgecolor='0.7')
        )
        


def _extract_feature_one_nuclei(nuclei_id, nuclei_seg, cell_seg, filename, morphology, nuclei_morphology, cell_morphology, 
                       channels, raw_image, sum_exp_nuclei, ave_exp_nuclei, sum_exp_cell, ave_exp_cell):
    regions = skimage.measure.regionprops((nuclei_seg == nuclei_id) * 1) 
    if len(regions) >= 1:
        this_nucleus = regions[0]
    else:
        return {}
    regions = skimage.measure.regionprops((cell_seg == nuclei_id) * 1)  # , coordinates='xy') (deprecated)
    if len(regions) >= 1:
        this_cell = regions[0]
    else:
        return {}

    centroid_y, centroid_x = this_nucleus.centroid  # y: rows; x: columnsb
    res = {"filename": filename,
           "id": nuclei_id,
           "coordinate_x": centroid_x,
           "coordinate_y": centroid_y}

    # morphology
    for i, feature in enumerate(morphology[:-1]):
        res[nuclei_morphology[i]] = getattr(this_nucleus, feature)
        res[cell_morphology[i]]   = getattr(this_cell, feature)
    res[nuclei_morphology[-1]]    = 1.0 * this_nucleus.perimeter ** 2 / this_nucleus.filled_area
    res[cell_morphology[-1]]      = 1.0 * this_cell.perimeter ** 2 / this_cell.filled_area


    # markers
    for ch, marker in enumerate(channels):
        res[sum_exp_nuclei[ch]] = np.sum(raw_image[nuclei_seg == nuclei_id, ch])
        res[ave_exp_nuclei[ch]] = np.average(raw_image[nuclei_seg == nuclei_id, ch])
        res[sum_exp_cell[ch]]   = np.sum(raw_image[cell_seg == nuclei_id, ch])
        res[ave_exp_cell[ch]]   = np.average(raw_image[cell_seg == nuclei_id, ch])
    return res


def extract_feature(channels: List, 
                    raw_image: np.ndarray, 
                    nuclei_seg: np.ndarray, 
                    cell_seg: np.ndarray, 
                    filename: str, 
                    use_parallel: bool = True, 
                    show_sample: bool = False) -> pd.DataFrame:
    """ Extract nuclei and cell level feature from cytof image based on nuclei segmentation and cell segmentation
        results
    Inputs:
        channels   = channels to extract feature from
        raw_image  = raw cytof image
        nuclei_seg = nuclei segmentation result
        cell_seg   = cell segmentation result
        filename   = filename of current cytof image
    Returns:
        feature_summary_df = a dataframe containing summary of extracted features
        morphology         = names of morphology features extracted

    :param channels: list
    :param raw_image: numpy.ndarray
    :param nuclei_seg: numpy.ndarray
    :param cell_seg: numpy.ndarray
    :param filename: string
    :param morpholoty: list
    :return feature_summary_df: pandas.core.frame.DataFrame
    """
    assert (len(channels) == raw_image.shape[-1])

    # morphology features to be extracted
    morphology = ["area", "convex_area", "eccentricity", "extent",
                "filled_area", "major_axis_length", "minor_axis_length",
                "orientation", "perimeter", "solidity", "pa_ratio"]

    ## morphology features
    nuclei_morphology = [_ + '_nuclei' for _ in morphology]  # morphology - nuclei level
    cell_morphology = [_ + '_cell' for _ in morphology]  # morphology - cell level

    ## single cell features
    # nuclei level
    sum_exp_nuclei = [_ + '_nuclei_sum' for _ in channels]  # sum expression over nuclei
    ave_exp_nuclei = [_ + '_nuclei_ave' for _ in channels]  # average expression over nuclei

    # cell level
    sum_exp_cell   = [_ + '_cell_sum' for _ in channels]  # sum expression over cell
    ave_exp_cell   = [_ + '_cell_ave' for _ in channels]  # average expression over cell

    # column names of final result dataframe
    column_names       = ["filename", "id", "coordinate_x", "coordinate_y"] + \
                         sum_exp_nuclei + ave_exp_nuclei + nuclei_morphology + \
                         sum_exp_cell + ave_exp_cell + cell_morphology

    # Initiate
    n_nuclei = np.max(nuclei_seg)
    feature_summary_df = pd.DataFrame(columns=column_names)
    
    if use_parallel:
        nuclei_ids = range(2, n_nuclei + 1)     
        with Pool() as mp_pool:
            res = mp_pool.starmap(_extract_feature_one_nuclei, 
                                      zip(nuclei_ids, 
                                          itertools.repeat(nuclei_seg), 
                                          itertools.repeat(cell_seg),
                                          itertools.repeat(filename),
                                          itertools.repeat(morphology),
                                          itertools.repeat(nuclei_morphology),
                                          itertools.repeat(cell_morphology),
                                          itertools.repeat(channels),
                                          itertools.repeat(raw_image),
                                          itertools.repeat(sum_exp_nuclei), 
                                          itertools.repeat(ave_exp_nuclei), 
                                          itertools.repeat(sum_exp_cell), 
                                          itertools.repeat(ave_exp_cell)
                                         ))
            # print(len(res), n_nuclei)

    else:
        res = []
        for nuclei_id in tqdm(range(2, n_nuclei + 1), position=0, leave=True):
            res.append(_extract_feature_one_nuclei(nuclei_id, nuclei_seg, cell_seg, filename, 
                                                   morphology, nuclei_morphology, cell_morphology, 
                                                   channels, raw_image, 
                                                   sum_exp_nuclei, ave_exp_nuclei, sum_exp_cell, ave_exp_cell))


    feature_summary_df = pd.DataFrame(res)
    if show_sample:
        print(feature_summary_df.sample(5))

    return feature_summary_df



def check_feature_distribution(feature_summary_df, features):
    """ Visualize feature distribution for each feature
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = features to check distribution
    Returns:
        None

    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    """

    for feature in features:
        print(feature)
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(np.log2(feature_summary_df[feature] + 0.0001), 100)
        ax.set_xlim(-15, 15)
        plt.show()


# def visualize_scatter(data, communities, n_community, title, figsize=(4,4), savename=None, show=False):
#     """
#     data = data to visualize (N, 2)
#     communities = group indices correspond to each sample in data (N, 1) or (N, )
#     n_community = total number of groups in the cohort (n_community >= unique number of communities)
#     """
#     fig, ax = plt.subplots(1,1, figsize=figsize)
#     ax.set_title(title)
#     sns.scatterplot(x=data[:,0], y=data[:,1], hue=communities, palette='tab20',
#                     hue_order=np.arange(n_community))
#                     #                 legend=legend,
#                     # hue_order=np.arange(n_community))
#     plt.axis('tight')
#     plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#     if savename is not None:
#         print("saving plot to {}".format(savename))
#         plt.savefig(savename)
#     if show:
#         plt.show()
#         return None
#     return fig

def visualize_scatter(data, communities, n_community, title, scatter_dot_size, figsize=(5,5), savename=None, show=False, ax=None):
    """
    data = data to visualize (N, 2)
    communities = group indices correspond to each sample in data (N, 1) or (N, )
    n_community = total number of groups in the cohort (n_community >= unique number of communities)
    """
    clos = not show and ax is None
    show = show and ax is None
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = None
    ax.set_title(title)
    sns.scatterplot(x=data[:,0], 
                    y=data[:,1], 
                    hue=communities, 
                    palette='tab20', 
                    s=scatter_dot_size, 
                    alpha=0.9, 
                    linewidth=0,
                    hue_order=np.arange(n_community), ax=ax
                    )
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    # plt.axis('tight')
    if savename is not None:
        print("saving plot to {}".format(savename))
        plt.tight_layout()
        plt.savefig(savename)
    if show:
        plt.show()
    if clos:
        plt.close('all')
    return fig, ax

def visualize_expression(data, markers, group_ids, title, figsize=(5,5), savename=None, show=False, ax=None):
    clos = not show and ax is None
    show = show and ax is None
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = None

    sns.heatmap(data,
                cmap='magma',
                xticklabels=markers,
                yticklabels=group_ids,
                ax=ax
               )
    # ax.set_xlabel("Markers")
    ax.set_ylabel("PhenoGraph clusters")
    ax.set_title("normalized expression - {}".format(title))
    ax.xaxis.set_tick_params(labelsize=8)
    if savename is not None:
        plt.tight_layout()
        plt.savefig(savename)
    if show:
        plt.show()
    if clos:
        plt.close('all')
    return fig, ax

def _get_thresholds(df_feature: pd.DataFrame,
                    features: List[str],
                    thres_bg: float = 0.3,
                    visualize: bool = True,
                    verbose: bool = False):
    """Calculate thresholds for each feature by assuming a Gaussian Mixture Model
    Inputs:
        df_feature = dataframe of extracted feature summary
        features   = a list of features to calculate thresholds from
        thres_bg   = a threshold such that the component with the mixing weight greater than the threshold would
                            be considered as background. (Default=0.3)
        visualize  = a flag indicating whether to visualize the feature distributions and thresholds or not.
                            (Default=True)
        verbose    = a flag indicating whether to print calculated values on screen or not. (Default=False)
    Outputs:
        thresholds = a dictionary of calculated threshold values
    :param df_feature: pandas.core.frame.DataFrame
    :param features: list
    :param visualize: bool
    :param verbose: bool
    :return thresholds: dict
    """
    thresholds = {}
    for f, feat_name in enumerate(features):
        X = df_feature[feat_name].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0, n_init=2).fit(X)
        mu = np.min(gm.means_[gm.weights_ > thres_bg])
        which_component = np.argmax(gm.means_ == mu)

        if verbose:
            print(f"GMM mean values: {gm.means_}")
            print(f"GMM weights: {gm.weights_}")
            print(f"GMM covariances: {gm.covariances_}")

        X     = df_feature[feat_name].values
        hist  = np.histogram(X, 150)
        sigma = np.sqrt(gm.covariances_[which_component, 0, 0])
        background_ratio = gm.weights_[which_component]
        thres = sigma * 2.5 + mu
        thresholds[feat_name] = thres

        n = sum(X > thres)
        percentage = n / len(X)

        ## visualize
        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.hist(X, 150, density=True)
            ax.set_xlabel("log2({})".format(feat_name))
            ax.plot(hist[1], scipy.stats.norm.pdf(hist[1], mu, sigma) * background_ratio, c='red')

            _which_component = np.argmin(gm.means_ == mu)
            _mu = gm.means_[_which_component]
            _sigma = np.sqrt(gm.covariances_[_which_component, 0, 0])
            ax.plot(hist[1], scipy.stats.norm.pdf(hist[1], _mu, _sigma) * (1 - background_ratio), c='orange')

            ax.axvline(x=thres, c='red')
            ax.text(0.7, 0.9, "n={}, percentage={}".format(n, np.round(percentage, 3)), ha='center', va='center',
                    transform=ax.transAxes)
            ax.text(0.3, 0.9, "mu={}, sigma={}".format(np.round(mu, 2), np.round(sigma, 2)), ha='center', va='center',
                    transform=ax.transAxes)
            ax.text(0.3, 0.8, "background ratio={}".format(np.round(background_ratio, 2)), ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(feat_name)
            plt.show()
    return thresholds

def _generate_summary(df_feature: pd.DataFrame, features: List[str], thresholds: dict) -> pd.DataFrame:
    """Generate (cell level) summary table for each feature in features: feature name, total number (of cells),
        calculated GMM threshold for this feature, number of individuals (cells) with greater than threshold values,
        ratio of individuals (cells) with greater than threshold values
    Inputs:
        df_feature = dataframe of extracted feature summary
        features   = a list of features to generate summary table
        thresholds = (calculated GMM-based) thresholds for each feature
    Outputs:
        df_info    = summary table for each feature

    :param df_feature: pandas.core.frame.DataFrame
    :param features: list
    :param thresholds: dict
    :return df_info: pandas.core.frame.DataFrame
    """

    df_info = pd.DataFrame(columns=['feature', 'total number', 'threshold', 'positive counts', 'positive ratio'])

    for feature in features:  # loop over each feature
        thres = thresholds[feature]  # fetch threshold for the feature
        X = df_feature[feature].values
        n = sum(X > thres)
        N = len(X)

        df_new_row = pd.DataFrame({'feature': feature, 'total number': N, 'threshold': thres,
                                   'positive counts': n, 'positive ratio': n / N}, index=[0])
        df_info = pd.concat([df_info, df_new_row])
    return df_info.reset_index(drop=True)