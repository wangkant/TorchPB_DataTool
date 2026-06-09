# EPB Benchmark Analysis

Notebooks and a small Python package that clean, match and compare the EPB benchmark runs of
[AmberTorchPB](https://doi.org/10.1021/acs.jctc.6c00085) against AMBER PBSA (R², MAPE, speedup)
and redraw the paper's figures.

- `Data_cleaning.ipynb`: raw run output → cleaned tables, one per solver
- `EPB_analysis_graph.ipynb`: paper figures and R² / MAPE / speedup tables
- `Comparison.ipynb`: normalized per-setting comparison
- `torchpb_datatool/`: shared schema, cleaning, merging, metrics and plots
- `scripts/make_figures.py`: all figures and tables without Jupyter

## Usage

Raw data is not included. Put your tables at `data/sample_raw.csv`, `data/benchmark.csv`
and `data/processed_epb.csv`, or generate synthetic ones for a dry run:

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/make_figures.py   # -> results/
```

Tests: `pip install -r requirements-dev.txt && python -m pytest`

## Citation

Y. Wu, Q. Wang, R. Jiang, R. Luo. *AmberTorchPB: A Unified Framework for Poisson–Boltzmann-Based
Reaction Field Energy Calculation via Tensor Computation.* J. Chem. Theory Comput. **2026**, 22 (7), 3554–3570.
