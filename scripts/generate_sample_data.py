"""Generate small synthetic CSVs matching the schema the notebooks expect.

This is a convenience for a dry run only. Bring-your-own-data remains the
primary workflow; the real raw datasets are intentionally not committed.

Three files are produced in the output directory (default: ``data/``):

1. ``sample_raw.csv``     -- raw experiment output consumed by
   ``Data_cleaning.ipynb`` (pre-rename schema).
2. ``benchmark.csv``      -- cleaned/standardized benchmark table consumed
   as ``benchmark`` by ``EPB_analysis_graph.ipynb`` and ``Comparison.ipynb``.
3. ``processed_epb.csv``  -- cleaned/standardized TorchPB table consumed as
   ``pb_data`` by the same two analysis notebooks.

The two cleaned tables share merge keys (Split, Mol Name, Atom Number,
Grid Size, CUDA Used, Tolerance) so the notebooks' inner joins yield rows.

Only pandas and numpy are required.
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants (mirror the notebook code exactly)
# ---------------------------------------------------------------------------

# Raw column names, before Data_cleaning.ipynb renames them.
RAW_COLUMNS = [
    "file",        # path string; Mol Name + Solver Name are extracted from it
    "system",      # -> Split
    "NATOM",       # -> Atom Number
    "grid",        # -> Grid Size
    "EPB",         # -> EPB Value
    "Total_time",  # -> Elapsed Time
    "init_lr",     # -> Initial Learning Rate
    "final_lr",    # -> Final Learning Rate
    "CUDA_used",   # -> CUDA Used
    "Run_Done",    # -> Succeeded
    "accept",      # -> Tolerance
]

# Cleaned column names, as the analysis notebooks read them.
CLEANED_COLUMNS = [
    "Split",
    "Mol Name",
    "Atom Number",
    "Grid Size",
    "Solver Name",
    "Tolerance",
    "EPB Value",
    "Elapsed Time",
    "CUDA Used",
    "Succeeded",
]

# Raw "system" tokens -> cleaned "Split" labels (Data_cleaning.ipynb mapping).
SYSTEM_TO_SPLIT = {
    "nucleic_acid_test": "Nucleic Acid",
    "protein_test": "Protein",
    "protein_complex_test": "Protein Complex",
    "small_molecule_test": "Small Molecule",
}

# Raw solver index -> cleaned solver name (Data_cleaning.ipynb mapping).
SOLVER_INDEX_TO_NAME = {
    0: "TorchPBCGSolver",
    1: "TorchPBBiCGSolver",
    4: "TorchPBAdamSolver",
    6: "TorchPBSorSolver",
}

SPLITS = list(SYSTEM_TO_SPLIT.keys())
GRIDS = [64, 128]
TOLERANCES = [1e-6, 1e-5, 1e-4]
CUDA_VALUES = [0, 1]

# A couple of molecule names per split, so path regexes have something to
# match. Names deliberately include a ".p22" suffix; Data_cleaning strips it.
MOLS_PER_SPLIT = {
    "nucleic_acid_test": ["1d20", "2gis"],
    "protein_test": ["1ubq", "1crn"],
    "protein_complex_test": ["1a2k", "3hfm"],
    "small_molecule_test": ["benz", "phen"],
}


def _base_records(seed=0):
    """Build the shared (split, mol, grid, tol, cuda) records both cleaned
    tables are derived from, so their merge keys line up."""
    rng = np.random.default_rng(seed)
    records = []
    for system in SPLITS:
        for mol in MOLS_PER_SPLIT[system]:
            atom_number = int(rng.integers(200, 5000))
            for grid in GRIDS:
                for tol in TOLERANCES:
                    for cuda in CUDA_VALUES:
                        # A physically meaningless but stable EPB baseline.
                        epb = float(rng.normal(-500.0, 120.0))
                        records.append(
                            {
                                "system": system,
                                "mol": mol,
                                "atom_number": atom_number,
                                "grid": grid,
                                "tol": tol,
                                "cuda": cuda,
                                "epb": epb,
                            }
                        )
    return records


def build_raw_frame(seed=0):
    """Raw experiment output for Data_cleaning.ipynb.

    ``file`` paths encode the solver index (``/solver_<d>/``) and the molecule
    name + grid (``<mol>_g<NNN>``) so the notebook's regex extractions work.
    """
    rng = np.random.default_rng(seed + 1)
    rows = []
    for rec in _base_records(seed):
        solver_idx = int(rng.choice(list(SOLVER_INDEX_TO_NAME.keys())))
        mol = rec["mol"]
        grid = rec["grid"]
        # e.g. /runs/solver_0/1ubq.p22_g064.out
        file_path = f"/runs/solver_{solver_idx}/{mol}.p22_g{grid:03d}.out"
        succeeded = int(rng.integers(0, 2))
        epb = rec["epb"] if succeeded else np.nan
        total_time = float(rng.uniform(0.5, 60.0)) if succeeded else np.nan
        rows.append(
            {
                "file": file_path,
                "system": rec["system"],
                "NATOM": rec["atom_number"],
                "grid": grid,
                "EPB": epb,
                "Total_time": total_time,
                "init_lr": round(float(rng.uniform(0.01, 0.5)), 4),
                "final_lr": round(float(rng.uniform(1e-4, 0.01)), 6),
                "CUDA_used": rec["cuda"],
                "Run_Done": succeeded,
                "accept": rec["tol"],
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def build_cleaned_frame(seed, epb_jitter, solver_name="TorchPBCGSolver"):
    """A cleaned/standardized table in the analysis-notebook schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for rec in _base_records():
        epb = rec["epb"] + float(rng.normal(0.0, epb_jitter))
        rows.append(
            {
                "Split": SYSTEM_TO_SPLIT[rec["system"]],
                "Mol Name": rec["mol"],
                "Atom Number": rec["atom_number"],
                "Grid Size": rec["grid"],
                "Solver Name": solver_name,
                "Tolerance": rec["tol"],
                "EPB Value": round(epb, 4),
                "Elapsed Time": round(float(rng.uniform(0.5, 60.0)), 2),
                "CUDA Used": rec["cuda"],
                "Succeeded": 1,
            }
        )
    return pd.DataFrame(rows, columns=CLEANED_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="data",
        help="directory to write the sample CSVs into (default: data)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed (default: 0)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw = build_raw_frame(seed=args.seed)
    # Benchmark = classical reference; processed = TorchPB run (small offset).
    benchmark = build_cleaned_frame(seed=args.seed + 10, epb_jitter=0.0)
    processed = build_cleaned_frame(seed=args.seed + 20, epb_jitter=2.0)

    raw_path = os.path.join(args.out_dir, "sample_raw.csv")
    bench_path = os.path.join(args.out_dir, "benchmark.csv")
    proc_path = os.path.join(args.out_dir, "processed_epb.csv")

    raw.to_csv(raw_path, index=False)
    benchmark.to_csv(bench_path, index=False)
    processed.to_csv(proc_path, index=False)

    print(f"Wrote {raw_path} ({len(raw)} rows, raw schema)")
    print(f"Wrote {bench_path} ({len(benchmark)} rows, cleaned schema)")
    print(f"Wrote {proc_path} ({len(processed)} rows, cleaned schema)")


if __name__ == "__main__":
    main()
