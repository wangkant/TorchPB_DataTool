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

Bring your own data. The raw datasets are intentionally **not** committed to
this repository. The notebooks read their inputs from default paths under
`data/`; point those paths at your own files (or overwrite the files at those
paths). The notebook read/write cells are:

- `Data_cleaning.ipynb` — reads `data/sample_raw.csv` (raw experiment output),
  writes `data/cleaned_sor.csv` (cleaned SOR-solver slice).
- `EPB_analysis_graph.ipynb` and `Comparison.ipynb` — each read
  `data/benchmark.csv` (benchmark table) and `data/processed_epb.csv`
  (processed TorchPB table).

Expected local layout:

```text
project_root/
├── data/
│   ├── sample_raw.csv        # raw input for Data_cleaning.ipynb
│   ├── benchmark.csv         # benchmark table for the analysis notebooks
│   └── processed_epb.csv     # processed TorchPB table for the analysis notebooks
├── scripts/
│   └── generate_sample_data.py
├── Data_cleaning.ipynb
├── EPB_analysis_graph.ipynb
├── Comparison.ipynb
└── README.md
```

### Sample data for a dry run

If you just want to exercise the notebooks end to end without real data, you
can generate small synthetic CSVs that match the exact schema each notebook
expects (only `pandas` and `numpy` are needed):

```bash
python scripts/generate_sample_data.py            # writes into data/
python scripts/generate_sample_data.py --out-dir data --seed 0
```

This produces `data/sample_raw.csv`, `data/benchmark.csv`, and
`data/processed_epb.csv` at the default paths above. The synthetic values are
meaningless — this is only for a plumbing check, not real analysis. All
generated CSVs are git-ignored.

## Running

```bash
jupyter notebook                 # or: jupyter lab
```

Open each notebook in order — `Data_cleaning.ipynb` first (cleans the raw experiment CSV; its final cell exports one cleaned per-solver slice to `data/cleaned_sor.csv`), then `EPB_analysis_graph.ipynb` or `Comparison.ipynb`. The two analysis notebooks each read two CSVs (a benchmark table and a processed EPB table) from `data/benchmark.csv` and `data/processed_epb.csv`. Adjust those default paths at the top of each notebook if your files live elsewhere.
