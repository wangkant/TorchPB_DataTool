# EPB Benchmark Analysis

A small set of Jupyter notebooks for cleaning, aligning, and comparing EPB (electrostatic Poisson-Boltzmann) benchmark results. The workflow takes raw experiment outputs, matches them against benchmark data, and analyzes differences in EPB values and runtime metrics.

## Notebooks

- **`Data_cleaning.ipynb`** — Cleans and standardizes raw EPB experiment data.
  - Loads raw CSV data
  - Extracts molecule names and solver information from file paths
  - Renames columns into a consistent schema
  - Inspects missing values and intermediate outputs

- **`EPB_analysis_graph.ipynb`** — Merges benchmark data with processed EPB data and runs comparison analysis.
  - Loads benchmark and processed datasets
  - Aligns shared keys across datasets
  - Converts columns to numeric format
  - Computes EPB absolute and normalized differences
  - Prepares data for downstream analysis and visualization

- **`Comparison.ipynb`** — More direct normalized comparison between benchmark and processed datasets.
  - Standardizes numeric fields
  - Merges datasets on shared metadata
  - Produces matched comparison tables

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Jupyter (Notebook or Lab)

```bash
pip install -r requirements.txt
```

## Data

The notebooks expect input CSV files; the raw datasets are not included in this repository. Provide your own input files and update the `pd.read_csv("")` paths at the top of each notebook accordingly. `Data_cleaning.ipynb` also writes its result via `to_csv("")` in its final cell — set that output path too.

Suggested local layout:

```text
project_root/
├── data/
│   ├── benchmark.csv
│   └── processed_epb.csv
├── Data_cleaning.ipynb
├── EPB_analysis_graph.ipynb
├── Comparison.ipynb
└── README.md
```

## Running

```bash
jupyter notebook                 # or: jupyter lab
```

Open each notebook in order — `Data_cleaning.ipynb` first (cleans the raw experiment CSV; its final cell exports one cleaned per-solver slice to the `to_csv` path you set), then `EPB_analysis_graph.ipynb` or `Comparison.ipynb`. The two analysis notebooks each read two CSVs (a benchmark table and a processed EPB table) — point their `pd.read_csv("")` calls at your benchmark file and the cleaned data.
