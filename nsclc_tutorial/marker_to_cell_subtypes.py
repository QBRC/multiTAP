import pandas as pd
import numpy as np
import os

SAVED_GROUPS = [86, 87, 88, 175, 176, 178]
BASE_CSV_DIR = "/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work"
accuml_type = 'sum'
cell_suffix = f"cell_{accuml_type}"

# pt id and ROI id mapping
roi_pt_id_mapping = pd.read_csv('/project/Xie_Lab/zgu/xiao_multiplex/nsclc_multiTAP_work/roi_pt_id_mapping.csv')

bin_df_list = list()

# merge the binary dfs
for prefix in SAVED_GROUPS:
    csv_path = os.path.join(BASE_CSV_DIR, f"nsclc_save_group{prefix}", f"nsclc_save_group{prefix}_coordinates_binary_expr_df_{accuml_type}2.csv")
    df = pd.read_csv(csv_path)
    bin_df_list.append(df)

combined_binary_df = pd.concat(bin_df_list, axis=0, ignore_index=True)

# first find the tumor cells by panCK
tumor_marker = f"panCyto_234((2745))Lu175-Lu175_{cell_suffix}"
tumor_cells = combined_binary_df[combined_binary_df[tumor_marker]].reset_index(drop=True)
non_tumor_cells = combined_binary_df[~combined_binary_df[tumor_marker]].reset_index(drop=True)
print(len(tumor_cells), "tumors cells identified")

# remove cells with both CD3 and CD20 positive
cd3_marker = f"CD3_1841((3363))Sm152-Sm152_{cell_suffix}"
cd20_marker = f"CD20_36((3369))Sm149-Sm149_{cell_suffix}"
non_tumor_cleaned = non_tumor_cells[~(non_tumor_cells[cd3_marker] & non_tumor_cells[cd20_marker])].copy()


non_tumor_cleaned["final_cell_type"] = np.nan

############ assign immune cells ############
# assign CD4 Treg (CD3+/CD4+/FOXP3+)
cd4_marker = f"CD4_2293((3000))Yb171-Yb171_{cell_suffix}"
foxp3_marker = f"FOXP3_115((2911))Dy163-Dy163_{cell_suffix}"

cd4_t_reg_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd4_marker] & non_tumor_cleaned[foxp3_marker]
non_tumor_cleaned.loc[cd4_t_reg_index, "final_cell_type"] = "CD4 Treg"
print(np.sum(cd4_t_reg_index), "CD4 T reg identified")

# assign IDO positive subsets
ido_marker = f"Indolea_2281((3014))Eu151-Eu151_{cell_suffix}"
ido_cd4_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd4_marker] & non_tumor_cleaned[ido_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[ido_cd4_index, "final_cell_type"] = "IDO_CD4"
print(np.sum(ido_cd4_index), "IDO_CD4 identified")

cd8_marker = f"CD8a_1718((2991))Er166-Er166_{cell_suffix}"
ido_cd8_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd8_marker] & non_tumor_cleaned[ido_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[ido_cd8_index, "final_cell_type"] = "IDO_CD8"
print(np.sum(ido_cd8_index), "IDO_CD8 identified")

# assign proliferation (Ki-67) subsets
ki67_marker = f"Ki-67_142((3418))Pt194-Pt194_{cell_suffix}"
ki67_cd4_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd4_marker] & non_tumor_cleaned[ki67_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[ki67_cd4_index, "final_cell_type"] = "Ki67_CD4"
print(np.sum(ki67_cd4_index), "Ki67 CD4 identified")

ki67_cd8_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd8_marker] & non_tumor_cleaned[ki67_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[ki67_cd8_index, "final_cell_type"] = "Ki67_CD8"
print(np.sum(ki67_cd8_index), "Ki67 CD8 identified")

# assign TCF1/7 subsets
# TCF7 gene also known as TCF-1 
tcf7_marker = f"TCF1TCF_2221((3415))Gd160-Gd160_{cell_suffix}"
tcf7_cd4_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd4_marker] & non_tumor_cleaned[tcf7_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[tcf7_cd4_index, "final_cell_type"] = "TCF1/7 CD4"
print(np.sum(tcf7_cd4_index), "TCF1/7 CD4 identified")

tcf7_cd8_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd8_marker] & non_tumor_cleaned[tcf7_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[tcf7_cd8_index, "final_cell_type"] = "TCF1/7 CD8"
print(np.sum(tcf7_cd8_index), "TCF1/7 CD8 identified")

# assign PD-1 subsets
pd1_marker = f"CD279(P_1743((3414))Gd155-Gd155_{cell_suffix}"
pd1_cd4_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd4_marker] & non_tumor_cleaned[pd1_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[pd1_cd4_index, "final_cell_type"] = "PD1 CD4"
print(np.sum(pd1_cd4_index), "PD1 CD4 identified")

# assign general CD4 and CD8
cd4_only_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd4_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[cd4_only_index, "final_cell_type"] = "CD4"
print(np.sum(cd4_only_index), "CD4 general identified")

cd8_only_index = non_tumor_cleaned[cd3_marker] & non_tumor_cleaned[cd8_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[cd8_only_index, "final_cell_type"] = "CD8"
print(np.sum(cd8_only_index), "CD8 general identified")

# above assignments for T cells only
# allow assignment only in np.nan, so by defn it excludes T cells
hla_marker = f"HLA-DR_1849((3362))Nd143-Nd143_{cell_suffix}"
cd68_marker = f"CD68_77((3413))Nd150-Nd150_{cell_suffix}"
myeloid_index = non_tumor_cleaned[hla_marker] & non_tumor_cleaned[cd68_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[myeloid_index, "final_cell_type"] = "myeloid"
print(np.sum(myeloid_index), "myeloid cells identified")

# assign neutrophils
mpo_marker = f"Myelope_276((2996))Y89-Y89_{cell_suffix}"
mmp9_marker = f"MMP9_2241((2912))Gd158-Gd158_{cell_suffix}"
neutrophils_index = non_tumor_cleaned[mpo_marker] & non_tumor_cleaned[mmp9_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[neutrophils_index, "final_cell_type"] = "neutrophils"
print(np.sum(neutrophils_index), "neutrophils identified")

# assign B cells
b_cell_index = non_tumor_cleaned[cd20_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[b_cell_index, "final_cell_type"] = "B cells"
print(np.sum(b_cell_index), "B cells identified")
################################################

################subset vessel cells############# 
# first find lymphatic endothelial
lyve_marker = f"LYVE-1_1982((2881))Er168-Er168_{cell_suffix}"
ccl21_marker = f"CCL21 6_2177((2889))Yb174-Yb174_{cell_suffix}"
lymphatic_index = non_tumor_cleaned[lyve_marker] & non_tumor_cleaned[ccl21_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[lymphatic_index, "final_cell_type"] = "lymphatic endothelial"
print(np.sum(lymphatic_index), "lymphatic cells identified")

# then overwrite lymphatic with HEV if PNAd+
pnad_marker = f"PNAd_1981((3323))Ho165-Ho165_{cell_suffix}"
hev_cells_index = non_tumor_cleaned[pnad_marker] & lymphatic_index
non_tumor_cleaned.loc[hev_cells_index, "final_cell_type"] = "high endothelial venules (HEV)"
print(np.sum(hev_cells_index), "HEVs identified")

# get general endothelial cells (CD146+/CD31+/vWF+/CCL21-)
cd146_marker = f"CD146_22((3259))Nd144-Nd144_{cell_suffix}"
cd31_vwf_marker = f"CD31_1859((3370))Yb172-Yb172_{cell_suffix}"
endothelial_index = non_tumor_cleaned[cd146_marker] & non_tumor_cleaned[cd31_vwf_marker] & (~non_tumor_cleaned[ccl21_marker]) & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[endothelial_index, "final_cell_type"] = "endothelial"
print(np.sum(endothelial_index), "endothelial cells identified")
################################################


##################CAF assignments##################### 
# mCAF
fap_marker = f"fap_323((3412))Nd142-Nd142_{cell_suffix}"
mmp11_marker = f"MMP11_2925((3364))Sm154-Sm154_{cell_suffix}"
collagen_marker = f"Collage_1360((2568))Sm147-Sm147_{cell_suffix}"
sma_marker = f"SMA_174((3277))In115-In115_{cell_suffix}"

mcaf_index = non_tumor_cleaned[sma_marker] & non_tumor_cleaned[mmp11_marker] & non_tumor_cleaned[collagen_marker] & (non_tumor_cleaned['final_cell_type'].isna())
# mcaf_index = non_tumor_cleaned[fap_marker] & non_tumor_cleaned[mmp11_marker] & non_tumor_cleaned[collagen_marker] & (non_tumor_cleaned['final_cell_type'].isna())

non_tumor_cleaned.loc[mcaf_index, "final_cell_type"] = "mCAF"
print(np.sum(mcaf_index), "mCAF identified")

# iCAF
cd34_marker = f"CD34_2254((3337))Er170-Er170_{cell_suffix}"
cd248_marker = f"CD248 E_2178((2830))Er167-Er167_{cell_suffix}"
icaf_index = non_tumor_cleaned[cd34_marker] & non_tumor_cleaned[cd248_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[icaf_index, "final_cell_type"] = "iCAF"
print(np.sum(icaf_index), "iCAF identified")

# hypoxic tCAF
cd10_marker = f"CD10_2546((3029))Dy161-Dy161_{cell_suffix}"
cd73_marker = f"CD73_2193((3319))Gd156-Gd156_{cell_suffix}"
caix_marker = f"Carboni_2443((2757))Nd146-Nd146_{cell_suffix}"
hypoxic_tcaf_index = non_tumor_cleaned[cd10_marker] & non_tumor_cleaned[caix_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[hypoxic_tcaf_index, "final_cell_type"] = "hypoxic tCAF"
print(np.sum(hypoxic_tcaf_index), "hypoxic tCAF identified")

# tCAF
tcaf_index = non_tumor_cleaned[cd10_marker] & non_tumor_cleaned[cd73_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[tcaf_index, "final_cell_type"] = "tCAF"
print(np.sum(tcaf_index), "tCAF identified")

# hypoxic CAF
hypoxic_caf_index = non_tumor_cleaned[caix_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[hypoxic_caf_index, "final_cell_type"] = "hypoxic CAF"
print(np.sum(hypoxic_caf_index), "hypoxic CAF identified")

#ifnCAF
ifn_caf_index = non_tumor_cells[ido_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[ifn_caf_index, "final_cell_type"] = "IFN CAF"
print(np.sum(ifn_caf_index), "IFN CAF identified")

# vCAF
vcaf_index = non_tumor_cells[cd146_marker] & (~non_tumor_cells[cd34_marker]) & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[vcaf_index, "final_cell_type"] = "vCAF"
print(np.sum(vcaf_index), "vCAF identified")

# dCAF
dcaf_index = non_tumor_cells[ki67_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[dcaf_index, "final_cell_type"] = "dCAF"
print(np.sum(dcaf_index), "dCAF identified")

# PDPN CAF
pdpn_marker = f"Podopla_1463((2619))Eu153-Eu153_{cell_suffix}"
pdpn_caf_index = non_tumor_cleaned[pdpn_marker] & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[pdpn_caf_index, "final_cell_type"] = "PDPN CAF"
print(np.sum(pdpn_caf_index), "PDPN CAF identified")

# SMA CAF
sma_caf_index = non_tumor_cleaned[sma_marker] & (~non_tumor_cleaned[fap_marker]) & (~non_tumor_cleaned[mmp11_marker]) & (~non_tumor_cleaned[collagen_marker]) & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[sma_caf_index, "final_cell_type"] = "SMA CAF"
print(np.sum(sma_caf_index), "SMA CAF identified")

# collagen CAF
collagen_caf_index = non_tumor_cleaned[collagen_marker] & (~non_tumor_cleaned[mmp11_marker]) & (~non_tumor_cleaned[sma_marker])  & (non_tumor_cleaned['final_cell_type'].isna())
non_tumor_cleaned.loc[collagen_caf_index, "final_cell_type"] = "Collagen CAF"
print(np.sum(collagen_caf_index), "Collagen CAF identified")
################################################

# subset needed columns
nontumor_pairwise_survival = non_tumor_cleaned[["pt_id", "roi_id", "coordinate_x", "coordinate_y", "final_cell_type"]].copy()
save_path = os.path.join(BASE_CSV_DIR, f"nontumor_cell_subtypes_accuml_{accuml_type}.csv")
nontumor_pairwise_survival.to_csv(save_path, index=False)

print("Cell_subtypes saved to", save_path)
