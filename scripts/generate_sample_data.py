"""Generate small synthetic CSVs matching the schema the notebooks expect.

This is a convenience for a dry run only. Bring-your-own-data remains the
primary workflow; the real raw datasets are intentionally not committed.

The grid mirrors the benchmark design of the AmberTorchPB paper (JCTC 2026,
22, 3554): the nucleic acid and protein splits, grid spacings of 0.35 / 0.55 /
0.75 / 0.95 A, tolerances of 1e-4 / 1e-5 / 1e-6, CPU and GPU, and the five
solvers CG, BiCG, GMRES, SOR and RB-SOR. The four representative molecules of
the paper (1P4Z, 351D, 1b16_A, 1ds1_A) are included by name so the
representative-molecule figures have rows; every number attached to them is
synthetic.

Three files are produced in the output directory (default: ``data/``):

1. ``sample_raw.csv``     -- raw experiment output consumed by
   ``Data_cleaning.ipynb`` (pre-rename schema, one row per AmberTorchPB run).
2. ``benchmark.csv``      -- AMBER PBSA reference table consumed as
   ``benchmark`` by ``EPB_analysis_graph.ipynb`` and ``Comparison.ipynb``.
3. ``processed_epb.csv``  -- cleaned AmberTorchPB table consumed as
   ``pb_data`` by the same notebooks (the cleaned form of ``sample_raw.csv``).

The two cleaned tables share merge keys (Split, Mol Name, Atom Number,
Grid Size, CUDA Used, Tolerance) so the notebooks' inner joins yield rows.

Only pandas and numpy are required.
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants (mirror torchpb_datatool/schema.py; tests check they match)
# ---------------------------------------------------------------------------

# Raw column names, before Data_cleaning.ipynb renames them.
RAW_COLUMNS = [
    "file",        # path string; Mol Name + Solver Name are extracted from it
    "system",      # -> Split
    "NATOM",       # -> Atom Number
    "grid",        # -> Grid Size (grid spacing, angstrom)
    "EPB",         # -> EPB Value (kcal/mol)
    "Total_time",  # -> Elapsed Time (s)
    "init_lr",     # -> Initial Learning Rate
    "final_lr",    # -> Final Learning Rate
    "CUDA_used",   # -> CUDA Used
    "Run_Done",    # -> Succeeded
    "accept",      # -> Tolerance
    "itn",         # -> Iterations (optional in real data)
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

# Raw "system" tokens -> cleaned "Split" labels (the two splits in the paper).
SYSTEM_TO_SPLIT = {
    "nucleic_acid_test": "Nucleic Acid",
    "protein_test": "Protein",
}

# Raw solver index (`solver_opt`) -> solver class, for the paper's solvers.
SOLVER_INDEX_TO_NAME = {
    0: "TorchPBCGSolver",
    1: "TorchPBBiCGSolver",
    2: "TorchPBGMRESSolver",
    6: "TorchPBSorSolver",
    7: "TorchPBRedBlackSorSolver",
}
BENCH_SOLVER_NAME = "AMBER PBSA CG"

SPLITS = list(SYSTEM_TO_SPLIT.keys())
GRID_SPACINGS = [0.35, 0.55, 0.75, 0.95]
TOLERANCES = [1e-4, 1e-5, 1e-6]
CUDA_VALUES = [0, 1]

# Names deliberately carry the ".p22" suffix in raw paths; Data_cleaning strips it.
MOLS_PER_SPLIT = {
    "nucleic_acid_test": ["1P4Z", "351D", "syn_na_01", "syn_na_02", "syn_na_03", "syn_na_04"],
    "protein_test": ["1b16_A", "1ds1_A", "syn_pr_01", "syn_pr_02", "syn_pr_03", "syn_pr_04"],
}
# The paper's systems span ~250 to 8,254 atoms.
ATOM_RANGE = {"nucleic_acid_test": (250, 1000), "protein_test": (800, 8254)}
ENERGY_PER_ATOM = {"nucleic_acid_test": (3.0, 6.0), "protein_test": (0.8, 2.5)}

# Synthetic solver behaviour, loosely shaped like the paper's Figs. 9, 11, 19:
# iterations = (a * atoms + b) * lognormal(0, sigma); seconds per iteration
# per 1e6 grid-point units; chance of not converging.
ITERATIONS = {0: (0.30, 150, 0.10), 1: (0.30, 150, 0.35), 2: (0.02, 25, 0.10), 6: (0.08, 60, 0.10), 7: (0.09, 70, 0.10)}
ITER_COST = {0: 1.0, 1: 1.0, 2: 30.0, 6: 10.0, 7: 3.0}
FAIL_RATE = {2: 0.05, 6: 0.05}
GPU_FACTOR = 0.1
SETUP_TIME = {0: 0.05, 1: 0.02}  # seconds, by CUDA Used


def _simulate(seed=0):
    """Return (runs, refs): one dict per AmberTorchPB run and per PBSA reference."""
    rng = np.random.default_rng(seed)
    runs, refs = [], []
    for system in SPLITS:
        for mol in MOLS_PER_SPLIT[system]:
            atoms = int(rng.integers(*ATOM_RANGE[system]))
            base = -atoms * float(rng.uniform(*ENERGY_PER_ATOM[system]))
            for h in GRID_SPACINGS:
                # Coarser grids shift the energy slightly (cf. Fig. 6).
                e_grid = base * (1.0 + 0.04 * (h - 0.35))
                points = atoms * (0.55 / h) ** 3
                for tol in TOLERANCES:
                    tighter = 1.0 + 0.15 * np.log10(1e-4 / tol)
                    for cuda in CUDA_VALUES:
                        ref_epb = e_grid + abs(e_grid) * tol * float(rng.normal(0.0, 0.05))
                        cg_time = None
                        for idx, name in SOLVER_INDEX_TO_NAME.items():
                            a, b, sigma = ITERATIONS[idx]
                            itn = max(1, int((a * atoms + b) * (0.55 / h) * tighter * rng.lognormal(0.0, sigma)))
                            elapsed = SETUP_TIME[cuda] + itn * ITER_COST[idx] * points * 1e-6 * (GPU_FACTOR if cuda else 1.0)
                            elapsed *= float(rng.lognormal(0.0, 0.1))
                            if idx == 0:
                                cg_time = elapsed
                            failed = rng.random() < FAIL_RATE.get(idx, 0.0)
                            runs.append({
                                "system": system,
                                "mol": mol,
                                "atoms": atoms,
                                "h": h,
                                "tol": tol,
                                "cuda": cuda,
                                "solver_idx": idx,
                                "solver": name,
                                "epb": np.nan if failed else e_grid + abs(e_grid) * float(rng.normal(0.0, 2e-5)),
                                "time": np.nan if failed else elapsed,
                                "itn": np.nan if failed else itn,
                            })
                        # AMBER PBSA CG: ~2x slower than AmberTorchPB CG (cf. Fig. 17).
                        speedup = float(rng.uniform(1.7, 2.1) if cuda else rng.uniform(2.0, 2.6))
                        refs.append({
                            "system": system,
                            "mol": mol,
                            "atoms": atoms,
                            "h": h,
                            "tol": tol,
                            "cuda": cuda,
                            "epb": ref_epb,
                            "time": cg_time * speedup,
                        })
    return runs, refs


def build_raw_frame(seed=0):
    """Raw experiment output for Data_cleaning.ipynb.

    ``file`` paths encode the solver index (``/solver_<d>/``) and the molecule
    name + grid spacing (``<mol>.p22_g<NNN>``, spacing x 100) so the notebook's
    regex extractions work.
    """
    runs, _ = _simulate(seed)
    rows = []
    for r in runs:
        rows.append({
            # e.g. /runs/solver_0/1P4Z.p22_g055.out
            "file": f"/runs/solver_{r['solver_idx']}/{r['mol']}.p22_g{round(r['h'] * 100):03d}.out",
            "system": r["system"],
            "NATOM": r["atoms"],
            "grid": r["h"],
            "EPB": r["epb"],
            "Total_time": r["time"],
            # AmberTorchPB defaults; only the optimizer-based solvers use them.
            "init_lr": 1e-3,
            "final_lr": 1e-6,
            "CUDA_used": r["cuda"],
            "Run_Done": int(not np.isnan(r["epb"])),
            "accept": r["tol"],
            "itn": r["itn"],
        })
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def build_processed_frame(seed=0):
    """Cleaned AmberTorchPB table (what Data_cleaning.ipynb makes of the raw frame)."""
    runs, _ = _simulate(seed)
    rows = []
    for r in runs:
        rows.append({
            "Split": SYSTEM_TO_SPLIT[r["system"]],
            "Mol Name": r["mol"],
            "Atom Number": r["atoms"],
            "Grid Size": r["h"],
            "Solver Name": r["solver"],
            "Tolerance": r["tol"],
            "EPB Value": round(r["epb"], 4),
            "Elapsed Time": round(r["time"], 2),
            "CUDA Used": r["cuda"],
            "Succeeded": int(not np.isnan(r["epb"])),
            "Iterations": r["itn"],
        })
    return pd.DataFrame(rows, columns=CLEANED_COLUMNS + ["Iterations"])


def build_benchmark_frame(seed=0):
    """AMBER PBSA reference table in the analysis-notebook schema."""
    _, refs = _simulate(seed)
    rows = []
    for r in refs:
        rows.append({
            "Split": SYSTEM_TO_SPLIT[r["system"]],
            "Mol Name": r["mol"],
            "Atom Number": r["atoms"],
            "Grid Size": r["h"],
            "Solver Name": BENCH_SOLVER_NAME,
            "Tolerance": r["tol"],
            "EPB Value": round(r["epb"], 4),
            "Elapsed Time": round(r["time"], 2),
            "CUDA Used": r["cuda"],
            "Succeeded": 1,
        })
    return pd.DataFrame(rows, columns=CLEANED_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    benchmark = build_benchmark_frame(seed=args.seed)
    processed = build_processed_frame(seed=args.seed)

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
