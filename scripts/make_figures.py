"""Write the paper's accuracy and efficiency tables and figures from two CSVs.

Reads a benchmark (AMBER PBSA) table and a processed (AmberTorchPB) table in
the cleaned schema and writes PNG figures and CSV tables, named after the
figure of the AmberTorchPB paper they correspond to:

    python scripts/make_figures.py
    python scripts/make_figures.py --benchmark data/benchmark.csv \
        --data data/processed_epb.csv --out results

Figures whose inputs are missing (e.g. no ``Iterations`` column, or no GPU
rows) are skipped with a message.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchpb_datatool import (  # noqa: E402
    accuracy_table,
    merge_runs,
    normalize_runs,
    speedup_table,
    subset,
    tolerance_mape_table,
)
from torchpb_datatool import plots  # noqa: E402
from torchpb_datatool.schema import (  # noqa: E402
    DEFAULT_SPACING,
    DEFAULT_TOLERANCE,
    REFERENCE_TOLERANCE,
    REPRESENTATIVE_MOLECULES,
)


def save(fig, path, what):
    if fig is None:
        print(f"skipped {what}: no rows for this setting")
        return
    fig = getattr(fig, "figure", fig)  # JointGrid -> Figure
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def write_table(table, path):
    table.to_csv(path, index=False)
    print(f"wrote {path} ({len(table)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="data/benchmark.csv", help="AMBER PBSA table (default: data/benchmark.csv)")
    parser.add_argument("--data", default="data/processed_epb.csv", help="AmberTorchPB table (default: data/processed_epb.csv)")
    parser.add_argument("--out", default="results", help="output directory (default: results)")
    parser.add_argument("--bench-solver", default=None, help="reference solver to keep when the benchmark holds several")
    args = parser.parse_args(argv)

    fig_dir = os.path.join(args.out, "figures")
    tab_dir = os.path.join(args.out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    benchmark = pd.read_csv(args.benchmark)
    pb_data = pd.read_csv(args.data)
    df = merge_runs(benchmark, pb_data, bench_solver=args.bench_solver)
    runs = normalize_runs(pb_data)
    print(f"merged {len(df)} AmberTorchPB rows against {len(benchmark)} reference rows")

    # Tables
    default = subset(df, spacing=DEFAULT_SPACING, tolerance=DEFAULT_TOLERANCE)
    write_table(accuracy_table(default), os.path.join(tab_dir, "accuracy_default_setting.csv"))
    write_table(
        accuracy_table(df, by=("Solver", "Device", "Split", "Grid Size", "Tolerance")),
        os.path.join(tab_dir, "accuracy_by_setting.csv"),
    )
    tol_table = tolerance_mape_table(benchmark, pb_data, REPRESENTATIVE_MOLECULES, bench_solver=args.bench_solver)
    write_table(tol_table, os.path.join(tab_dir, "tolerance_mape.csv"))
    write_table(speedup_table(default, by=("Device",)), os.path.join(tab_dir, "speedup.csv"))
    write_table(speedup_table(default, by=("Device", "Size Group")), os.path.join(tab_dir, "speedup_by_size.csv"))

    # Section 3.2: accuracy against AMBER PBSA
    for fig_no, solvers in (("fig04", ("CG", "BiCG")), ("fig08", ("GMRES", "SOR"))):
        for solver in solvers:
            save(plots.solver_parity_plot(df, solver), os.path.join(fig_dir, f"{fig_no}_parity_{solver.lower()}.png"), f"{fig_no} {solver}")
    for solver in ("CG", "BiCG"):
        save(plots.device_consistency_plot(runs, solver), os.path.join(fig_dir, f"fig05_cpu_gpu_{solver.lower()}.png"), f"fig05 {solver}")
    save(plots.grid_spacing_plot(df, REPRESENTATIVE_MOLECULES), os.path.join(fig_dir, "fig06_grid_spacing_cg.png"), "fig06")
    save(plots.tolerance_mape_plot(tol_table), os.path.join(fig_dir, "fig07_tolerance_mape_cg.png"), f"fig07 (reference: tol {REFERENCE_TOLERANCE:.0e})")

    # Section 3.3: convergence behaviour (needs an Iterations column)
    if "Iterations" not in runs.columns:
        print("skipped fig09/fig11: the processed table has no Iterations column")
    else:
        save(plots.iteration_hist(runs), os.path.join(fig_dir, "fig09_iterations_hist.png"), "fig09")
        save(plots.iterations_vs_atoms(runs), os.path.join(fig_dir, "fig11_iterations_vs_atoms.png"), "fig11")

    # Section 3.5: efficiency
    save(plots.elapsed_time_plot(df), os.path.join(fig_dir, "fig17_19_elapsed_time.png"), "fig17/19")

    # Supplementary: |dEPB| distributions (CG)
    save(plots.log_diff_violin(df, x="Grid Size", solver="CG"), os.path.join(fig_dir, "supp_log_diff_by_grid_cg.png"), "supp grid")
    save(plots.log_diff_violin(df, x="Tolerance", solver="CG", legend_loc="upper right"),
         os.path.join(fig_dir, "supp_log_diff_by_tolerance_cg.png"), "supp tolerance")


if __name__ == "__main__":
    main()
