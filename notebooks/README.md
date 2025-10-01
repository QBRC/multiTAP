This subdirectory contains Jupyter notebook tutorials that demonstrate how to process high-dimensional data using the analysis pipeline.

- `manuscript-figures.ipynb` 
Provides step-by-step instructions to reproduce the figures presented in the manuscript. This includes both single-ROI analyses and cohort-level (multi-ROI) analyses.

- `tutorial-[FileFormat]-explore.ipynb` (FileFormat=[MCD, QPTIFF])
Demonstrates helper functions for early-stage data exploration. These notebooks enable visualization of channels and, when supported by the input format, automatic extraction of marker names from metadata. Users can flag low-quality channels and include them in the `channels_remove` parameter. For MCD files containing batch acquisitions, a DataFrame formatted for the `CytofCohort` batch processing pipeline is also generated.

- `tutorial-Single-ROI-[tiff, txt].ipynb` 
Builds upon the initial exploratory step and introduces the core analysis pipeline. These notebooks provide tutorials on running segmentation, marker quantification, phenotyping, and spatial interaction analyses on individual acquisitions.