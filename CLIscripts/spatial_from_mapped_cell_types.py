# this files reads the assigned nontumor cell subtypes (CD4, iCAF, etc)
# performs spatial interaction on the cell types, and store as interaction matrix for each ROI
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.neighbors import kneighbors_graph as skgraph  # , DistanceMetric
from scipy import sparse as sp
from itertools import product
from datetime import datetime

cell_type_cols = ['final_cell_type_B cells', 'final_cell_type_CD4',
       'final_cell_type_CD4 Treg', 'final_cell_type_CD8',
       'final_cell_type_Collagen CAF', 'final_cell_type_IDO_CD4',
       'final_cell_type_IDO_CD8', 'final_cell_type_IFN CAF',
       'final_cell_type_Ki67_CD4', 'final_cell_type_Ki67_CD8',
       'final_cell_type_PD1 CD4', 'final_cell_type_PDPN CAF',
       'final_cell_type_SMA CAF', 'final_cell_type_TCF1/7 CD4',
       'final_cell_type_TCF1/7 CD8', 'final_cell_type_dCAF',
       'final_cell_type_endothelial',
       'final_cell_type_high endothelial venules (HEV)',
       'final_cell_type_hypoxic CAF', 'final_cell_type_hypoxic tCAF',
       'final_cell_type_iCAF', 'final_cell_type_lymphatic endothelial',
       'final_cell_type_mCAF', 'final_cell_type_myeloid',
       'final_cell_type_neutrophils', 'final_cell_type_tCAF',
       'final_cell_type_vCAF']

BASE_CSV_DIR = "/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work"
TEST_RUN = False
accuml_type = 'sum' # ['sum', 'ave']
 
# load nontumor cell types
nontumor_pairwise_survival = pd.read_csv(os.path.join(BASE_CSV_DIR, f"nontumor_cell_subtypes_accuml_{accuml_type}.csv"))
nontumor_pairwise_survival = pd.get_dummies(nontumor_pairwise_survival, columns=['final_cell_type'], dtype=int)

marker_roi_list = list()

roi_ids = np.unique(nontumor_pairwise_survival['roi_id'])

if TEST_RUN: roi_ids = roi_ids[:2]

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_datetime, f'{len(roi_ids)} ROIs identified')

# loop through each roi for spatial
for one_roi in roi_ids:

    print(f"Processing ROI {one_roi}")

    try:

        # finds the roi
        one_roi_spatial = nontumor_pairwise_survival[nontumor_pairwise_survival['roi_id']==one_roi].reset_index(drop=True)

        # performing k-neighbor
        threshold = 15
        neighbor_matrix = skgraph(np.array(one_roi_spatial.loc[:, ['coordinate_x', 'coordinate_y']]), n_neighbors=threshold, mode='distance')

        I, J, V = sp.find(neighbor_matrix)
        v_keep_index = V > 0 # any non-zero distance neighbor qualifies

        # finds index of values less than the distance threshold
        i_keep, j_keep = I[v_keep_index], J[v_keep_index]
        assert len(i_keep) == len(j_keep) # these are paired indexes for the cell. must equal in length.

        n_neighbor_pairs = len(i_keep)
        n_markers = len(cell_type_cols)
        # (i,j) now tells you the index of the two cells that are in close proximity (within {thres} distance of each other)
        # now we need a list that tells you the positive expressed marker index in each cell

        # returns a binary dataframe of whether each cell at each marker passes the positive threshold
        df_binary_pos_exp = one_roi_spatial[cell_type_cols].copy()
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

        df_expected_counts = pd.DataFrame(expected_counts, index=cell_type_cols, columns=cell_type_cols)
        df_cell_interaction_counts = pd.DataFrame(cell_interaction_in_markers_counts, index=cell_type_cols, columns=cell_type_cols)

        # calculates percentage within function if not return compoenents
        # df_expected_prob = df_expected_counts / n_cells**2
        df_expected_prob = df_expected_counts / n_neighbor_pairs**2

        # theta(i_pos and j_pos)
        df_cell_interaction_prob = df_cell_interaction_counts / n_neighbor_pairs


        # do some post processing
        marker_all = df_expected_prob.columns
        epsilon = 1e-6

        # Normalize and fix Nan
        edge_percentage_norm = np.log10(df_cell_interaction_prob.values / (df_expected_prob.values+epsilon) + epsilon)

        # if observed/expected = 0, then log odds ratio will have log10(epsilon)
        # no observed means interaction cannot be determined, does not mean strong negative interaction
        edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0

        edge_perc_remapped = pd.DataFrame(edge_percentage_norm, index=marker_all, columns=marker_all)
        edge_perc_remapped["roi_id"] = one_roi
        marker_roi_list.append(edge_perc_remapped)


    except Exception as e:
        print(f"ROI {one_roi} not processed due to error {e}")


# concatenate all pt df
combined_df = pd.concat(marker_roi_list, axis=0)

combined_df.to_csv(os.path.join(BASE_CSV_DIR, f"nontumor_spatial_kneighbor15_{accuml_type}.csv"))

# Get the current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_datetime, 'process completed.')

