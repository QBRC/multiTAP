# Analyzing a large cohort of IMC cores
Using a publicly available dataset (Cords, Lena, et al. "Cancer-associated fibroblast phenotypes are associated with patient outcome in non-small cell lung cancer." Cancer cell 42.3 (2024): 396-412.) containing 2,070 cores from 1,070 subjects, we constructed a spatial interaction matrix fo 27 cell subtypes averaged across all ROIs. We then flattened the pairwise interaction terms and compared patient outcomes in those stratified groups. This directory documents the detailed steps for spatial interaction and survival analysis.

Steps:
1. `CLIscripts/mcd_to_tiff.py` reads in the original MCD files downloaded from Zenodo and converts acquisitions to TIFF files. All images will be stored in `output_dir`.

1. `CLIscripts/batch_process_feature.py` reads in a `.csv` file to process IMC cores in batch. The CSV file needs to contain slides, ROI, TIFF file path. A template is shown in `CLIscripts/templates/example_cohort.csv`.
> [Note]
> The `save_group` argument is used here as a single .pkl with 2000+ IMC cores were too memory intensive. 

`channel_dict` defines the **required** `nuclei` channels and the *optional* membrane channels. Here we used transmembrane proteins as membrane markers. For other parameters, first refer to `tutorial-MCD-explore.ipynb` for documentations. Optionally, `tutorial-Single-ROI-tiff.ipynb` contains advanced parameters if needed. After batch processing, the specified `dir_out` should have the following structure:
 ```text
    .
    ├── dir_out/
    │   ├── nsclc_save_group86/
    │   │   └── nsclc_save_group86.pkl
    │   ├── nsclc_save_group87/
    │   │   └── nsclc_save_group87.pkl
    ...
    │   ├── nsclc_save_group178/
    │   │   └── nsclc_save_group178.pkl
```

1. `nsclc_tutorial/save_df_feature.py` reads in saved `.pkl` files and generates marker level summary. For each image, the binary positive expression dataframe and  cell coordinates are extracted. All patients with the same prefix (i.e. within the same save group) are saved to one `.csv`.

1. `nsclc_tutorial/marker_to_cell_subtypes.py` reads in the the preprocessed binary marker expression `.csv` files, separates cells into tumor and nontuomor, then separates the nontumor into immune, vessel, and cancer-associated fibroblasts (CAFs). This creates a single `.csv` for all nontumor cells in all patients.

1. `nsclc_tutorial/spatial_from_mapped_cell_types.py` reads in the subtype classification file, and constructs a (k=15)-neighbor network graph based on the cell coordinates and assigned subtypes within each ROI. The spatial interactions within the 27 cell subtypes are computed. A dataframe with shape 27*len(roi) x 28 is saved. The columns of the dataframe contains 27 markers and one roi_id. An example is available in `notebooks/preprocessed/nontumor_spatial_kneighbor15_sum.csv`

1. `notebooks/tutorial-figures.ipynb` In header `Figure 6B`, the dataframe is averaged across all ROIs and an `sns.clustermap() is constructed`. In `Figure 6C`, each ROI-level spatial interaction matrix was flattened, and patient survival data is merged. Log-rank tests were computed for each pairwise interaction in 953 eligible patients, and FDR-adjusted p-values are reported.