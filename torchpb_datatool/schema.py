"""Column names, label maps, and the benchmark settings used in the paper.

Wu, Wang, Jiang, Luo. "AmberTorchPB: A Unified Framework for Poisson-Boltzmann-
Based Reaction Field Energy Calculation via Tensor Computation."
J. Chem. Theory Comput. 2026, 22 (7), 3554-3570. doi:10.1021/acs.jctc.6c00085
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw experiment output -> cleaned schema (Data_cleaning.ipynb)
# ---------------------------------------------------------------------------

# Raw column -> cleaned column. `itn` (the iteration count AmberTorchPB hands
# back to PBSA) is optional; it is only needed for the iteration figures.
RAW_TO_CLEAN = {
    "system": "Split",
    "NATOM": "Atom Number",
    "grid": "Grid Size",
    "EPB": "EPB Value",
    "Total_time": "Elapsed Time",
    "init_lr": "Initial Learning Rate",
    "final_lr": "Final Learning Rate",
    "CUDA_used": "CUDA Used",
    "Run_Done": "Succeeded",
    "accept": "Tolerance",
    "itn": "Iterations",
}

# Raw "system" token -> Split label. The paper reports the nucleic acid and
# protein splits of the AMBER PBSA benchmark suite; the other two are kept so
# older raw files still clean.
SYSTEM_TO_SPLIT = {
    "nucleic_acid_test": "Nucleic Acid",
    "protein_test": "Protein",
    "protein_complex_test": "Protein Complex",
    "small_molecule_test": "Small Molecule",
}
PAPER_SPLITS = ("Nucleic Acid", "Protein")

# `solver_opt` index (the `/solver_<d>/` path component) -> solver class, as
# documented in the AmberTorchPB wrapper. Index 4 keeps the short name used by
# earlier processed tables.
SOLVER_INDEX_TO_NAME = {
    "0": "TorchPBCGSolver",
    "1": "TorchPBBiCGSolver",
    "2": "TorchPBGMRESSolver",
    "3": "TorchPBSgdGradientOptimSolver",
    "4": "TorchPBAdamSolver",
    "5": "TorchPBNeuralNetworkSolver",
    "6": "TorchPBSorSolver",
    "7": "TorchPBRedBlackSorSolver",
}

# Solver class -> the name used in the paper's text and figures.
SOLVER_LABELS = {
    "TorchPBCGSolver": "CG",
    "TorchPBBiCGSolver": "BiCG",
    "TorchPBGMRESSolver": "GMRES",
    "TorchPBSorSolver": "SOR",
    "TorchPBRedBlackSorSolver": "RB-SOR",
    "TorchPBSgdGradientOptimSolver": "SGD",
    "TorchPBAdamSolver": "Adam",
    "TorchPBAdamGradientOptimSolver": "Adam",
    "TorchPBNeuralNetworkSolver": "NN",
}
# The five iterative solvers benchmarked in the paper, in its order.
PAPER_SOLVERS = ("CG", "BiCG", "GMRES", "SOR", "RB-SOR")
# The convergence analysis (Figs. 9-11) covers these four.
ITERATION_SOLVERS = ("CG", "BiCG", "GMRES", "SOR")

# Cleaned table written by Data_cleaning.ipynb.
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
    "Initial Learning Rate",
    "Final Learning Rate",
]

# Columns the analysis notebooks read from the benchmark and processed tables.
ANALYSIS_COLUMNS = [
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

# Keys a benchmark (AMBER PBSA) row and a processed (AmberTorchPB) row share.
MERGE_KEYS = ["Split", "Mol Name", "Atom Number", "Grid Size", "CUDA Used", "Tolerance"]

# ---------------------------------------------------------------------------
# Benchmark settings (paper, Section 3.1.1)
# ---------------------------------------------------------------------------

# "Grid Size" holds the uniform grid spacing h in angstrom.
GRID_SPACINGS = (0.35, 0.55, 0.75, 0.95)
DEFAULT_SPACING = 0.55

# Convergence tolerance on |Ax - b| / |b| (l1 norm).
TOLERANCES = (1e-4, 1e-5, 1e-6)
DEFAULT_TOLERANCE = 1e-4
# Fig. 7 measures every tolerance against AMBER PBSA at 1e-6.
REFERENCE_TOLERANCE = 1e-6

# Molecules used for the grid-spacing, tolerance, and convergence figures.
REPRESENTATIVE_MOLECULES = ("1P4Z", "351D", "1b16_A", "1ds1_A")

DEVICE_LABELS = {0: "CPU", 1: "GPU"}

# Size groups of the multithreading analysis (Fig. 18b).
SIZE_GROUPS = ("Small (<500)", "Medium (500-5000)", "Large (>5000)")

BENCH_LABEL = "AMBER PBSA"
DATA_LABEL = "AmberTorchPB"
ENERGY_UNIT = "kcal/mol"


def solver_label(name):
    """Paper name for a solver class (unknown names pass through)."""
    return SOLVER_LABELS.get(name, name)


def solver_order(labels):
    """Paper solvers first, in the paper's order, then any others sorted."""
    present = {label for label in labels if isinstance(label, str)}
    return [s for s in PAPER_SOLVERS if s in present] + sorted(present - set(PAPER_SOLVERS))


def device_label(cuda_used):
    return DEVICE_LABELS.get(cuda_used, cuda_used)


def size_group(atom_number):
    """Bin atom counts into the paper's Small / Medium / Large groups."""
    return pd.cut(
        pd.to_numeric(atom_number, errors="coerce").astype(float),
        bins=[-np.inf, 499, 5000, np.inf],
        labels=list(SIZE_GROUPS),
    )
