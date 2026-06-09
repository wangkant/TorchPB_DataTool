"""Accuracy and efficiency metrics as defined in the paper (Section 3.1.2)."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .compare import normalize_runs, subset
from .schema import (
    ANALYSIS_COLUMNS,
    BENCH_LABEL,
    DATA_LABEL,
    DEFAULT_SPACING,
    REFERENCE_TOLERANCE,
    solver_order,
)


def _finite_pairs(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def r2(y_true, y_pred):
    """R^2 of AmberTorchPB energies against AMBER PBSA reference energies (Eq. 12)."""
    t, p = _finite_pairs(y_true, y_pred)
    return float(r2_score(t, p)) if len(t) >= 2 else np.nan


def mape(y_true, y_pred):
    """Mean absolute percentage error in percent (Eq. 13)."""
    t, p = _finite_pairs(y_true, y_pred)
    nonzero = t != 0
    if not nonzero.any():
        return np.nan
    return float(100 * np.mean(np.abs((p[nonzero] - t[nonzero]) / t[nonzero])))


def _sort_groups(table, by):
    """Sort a per-group table by ``by``, with solvers in the paper's order."""
    if table.empty:
        return table
    rank = {}
    if "Solver" in by:
        rank = {s: i for i, s in enumerate(solver_order(table["Solver"]))}
    return table.sort_values(
        list(by), key=lambda s: s.map(rank) if s.name == "Solver" else s
    ).reset_index(drop=True)


def accuracy_table(df, by=("Solver", "Device", "Split")):
    """Convergence rate, R^2, MAPE and MAE per group of a merge_runs() table.

    R^2 / MAPE / MAE use only rows where both the reference and the
    AmberTorchPB run converged.
    """
    by = list(by)
    rows = []
    for key, g in df.groupby(by, observed=True):
        ok = g[(g["Succeeded_Bench"] == 1) & (g["Succeeded_Data"] == 1)]
        t, p = _finite_pairs(ok["EPB_Bench"], ok["EPB_Data"])
        rows.append({
            **dict(zip(by, key)),
            "Runs": len(g),
            "Converged (%)": 100 * float((g["Succeeded_Data"] == 1).mean()),
            "N": len(t),
            "R2": r2(t, p),
            "MAPE (%)": mape(t, p),
            "MAE (kcal/mol)": float(np.mean(np.abs(p - t))) if len(t) else np.nan,
        })
    return _sort_groups(pd.DataFrame(rows), by)


def tolerance_mape_table(benchmark, pb_data, molecules, solver="CG", device="CPU",
                         spacing=DEFAULT_SPACING, ref_tolerance=REFERENCE_TOLERANCE,
                         bench_solver=None):
    """Relative difference of AmberTorchPB energies at each tolerance (Fig. 7).

    The reference is the AMBER PBSA energy of the same molecule at
    ``ref_tolerance`` (1e-6 in the paper).
    """
    bench = normalize_runs(benchmark[ANALYSIS_COLUMNS])
    if bench_solver is not None:
        bench = bench[(bench["Solver Name"] == bench_solver) | (bench["Solver"] == bench_solver)]
    ref = subset(bench, device=device, spacing=spacing, tolerance=ref_tolerance, molecules=molecules)
    ref = ref[ref["Succeeded"] == 1][["Split", "Mol Name", "Grid Size", "CUDA Used", "EPB Value"]]
    ref = ref.rename(columns={"EPB Value": "EPB_Ref"})

    data = subset(normalize_runs(pb_data), solver=solver, device=device, spacing=spacing, molecules=molecules)
    data = data[data["Succeeded"] == 1]

    t = data.merge(ref, on=["Split", "Mol Name", "Grid Size", "CUDA Used"], how="inner")
    t["APE (%)"] = 100 * (t["EPB Value"] - t["EPB_Ref"]).abs() / t["EPB_Ref"].abs()
    t = t[["Split", "Mol Name", "Solver", "Device", "Grid Size", "Tolerance", "EPB_Ref", "EPB Value", "APE (%)"]]
    order = {m: i for i, m in enumerate(molecules)}
    return t.sort_values(
        ["Mol Name", "Tolerance"], ascending=[True, False], key=lambda s: s.map(order) if s.name == "Mol Name" else s
    ).reset_index(drop=True)


def speedup_table(df, by=("Device",), solver="CG"):
    """AMBER PBSA elapsed time over AmberTorchPB elapsed time, per group (Fig. 17)."""
    d = df[(df["Solver"] == solver) & (df["Succeeded_Bench"] == 1) & (df["Succeeded_Data"] == 1)]
    d = d.assign(Speedup=d["Elapsed Time_Bench"] / d["Elapsed Time_Data"])
    d = d[np.isfinite(d["Speedup"])]
    return d.groupby(list(by), observed=True).agg(**{
        "N": ("Speedup", "size"),
        f"{BENCH_LABEL} mean time (s)": ("Elapsed Time_Bench", "mean"),
        f"{DATA_LABEL} {solver} mean time (s)": ("Elapsed Time_Data", "mean"),
        "Median speedup": ("Speedup", "median"),
        "Mean speedup": ("Speedup", "mean"),
    }).reset_index()
