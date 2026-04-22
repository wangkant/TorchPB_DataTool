# EPB Benchmark Analysis

This repository contains a small set of Jupyter notebooks for cleaning, aligning, and comparing EPB-related benchmark results. The workflow focuses on preparing raw experiment outputs, matching benchmark data with processed results, and analyzing differences in EPB values and runtime-related metrics.

## Repository Structure

- `data_cleaning.ipynb`  
  Cleans and standardizes raw EPB experiment data.  
  Main tasks include:
  - loading raw CSV data
  - extracting molecule names and solver information from file paths
  - renaming columns into a consistent schema
  - inspecting missing values and intermediate outputs

- `epb_analysis_graph.ipynb`  
  Merges benchmark data with processed EPB data and performs comparison analysis.  
  Main tasks include:
  - loading benchmark and processed datasets
  - aligning shared keys across datasets
  - converting columns to numeric format
  - computing EPB absolute differences and normalized differences
  - preparing data for downstream analysis and visualization

- `comparison.ipynb`  
  Performs a more direct normalized comparison between benchmark and processed datasets.  
  Main tasks include:
  - standardizing numeric fields
  - merging datasets on shared metadata
  - creating matched comparison tables

## Data

The notebooks expect input CSV files, but the raw datasets are not included in this repository by default.

You will need to provide your own input files and update the file paths in the notebooks accordingly.

Suggested local structure:

```text
project_root/
├── data/
│   ├── benchmark.csv
│   ├── processed_epb.csv
├── data_cleaning.ipynb
├── epb_analysis_graph.ipynb
├── comparison.ipynb
└── README.md
