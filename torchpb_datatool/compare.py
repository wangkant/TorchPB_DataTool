"""Normalize run tables and join AMBER PBSA reference rows with AmberTorchPB rows."""

import warnings

import numpy as np
import pandas as pd

from .schema import (
    ANALYSIS_COLUMNS,
    MERGE_KEYS,
    device_label,
    size_group,
    solver_label,
)

# ---------------------------------------------------------------------------
# Value parsers (tolerant of thousands separators, Unicode minus, stray spaces)
# ---------------------------------------------------------------------------


def to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("−", "-")
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_int01(x):
    v = to_float(x)
    return np.nan if pd.isna(v) else int(v)


def clean_str(x):
    if pd.isna(x):
        return np.nan
    return " ".join(str(x).split())


def normalize_tolerance(val):
    """Parse 1e-04 / 1e-4 / 0.0001 alike; 0 means "not set" and becomes NaN.

    Rounded to 6 significant digits so a float32 1e-4 still equals 1e-4.
    """
    if pd.isna(val):
        return np.nan
    v = to_float(str(val).replace(" ", ""))
    if pd.isna(v) or v == 0:
        return np.nan
    return float(f"{v:.6g}")


# ---------------------------------------------------------------------------
# Comparison.ipynb: normalized long tables
# ---------------------------------------------------------------------------


def normalize_to_schema(df, source, tol_order=(1e-6, 1e-5, 1e-4)):
    t = df.copy()

    t["Split"] = t["Split"].map(clean_str)
    t["MolName"] = t["Mol Name"].map(clean_str)
    if "Solver Name" in t.columns:
        t["Solver"] = t["Solver Name"].map(clean_str)
    else:
        t["Solver"] = np.nan

    t["AtomNum"] = t["Atom Number"].apply(to_float)
    t["Grid"] = t["Grid Size"].apply(to_float)
    t["Tol_num"] = t["Tolerance"].map(normalize_tolerance)
    t["CUDA"] = t["CUDA Used"].apply(to_int01) if "CUDA Used" in t.columns else np.nan
    t["Succeeded"] = t["Succeeded"].apply(to_int01) if "Succeeded" in t.columns else np.nan
    t["EPB"] = t["EPB Value"].apply(to_float) if "EPB Value" in t.columns else np.nan
    t["Time"] = t["Elapsed Time"].apply(to_float) if "Elapsed Time" in t.columns else np.nan

    t["Tol_cat"] = pd.Categorical(t["Tol_num"], categories=list(tol_order), ordered=True)

    cols = ["Split", "MolName", "AtomNum", "Grid", "Tol_num", "Tol_cat", "CUDA", "Succeeded", "Solver", "EPB", "Time"]
    t = t[cols].copy()

    t = t.dropna(subset=["Split", "MolName", "AtomNum", "Grid", "Tol_num"], how="any")

    t["source"] = source
    return t


def merge_normalized(bench_norm, data_norm):
    """Inner-join two normalize_to_schema() tables; adds |dEPB| and |dEPB| per atom."""
    merge_keys = ["Split", "MolName", "AtomNum", "Grid", "Tol_num", "CUDA"]
    df = pd.merge(
        bench_norm.add_suffix("_Bench"),
        data_norm.add_suffix("_Data"),
        left_on=[k + "_Bench" for k in merge_keys],
        right_on=[k + "_Data" for k in merge_keys],
        how="inner",
    )

    df["EPB_abs_diff"] = (df["EPB_Data"] - df["EPB_Bench"]).abs()
    df["EPB_abs_diff_norm"] = df["EPB_abs_diff"] / df["AtomNum_Bench"]

    return df[[
        "Split_Bench", "MolName_Bench", "AtomNum_Bench", "Grid_Bench", "Tol_num_Bench", "Tol_cat_Bench", "CUDA_Bench",
        "Solver_Data", "EPB_Bench", "EPB_Data", "Time_Bench", "Time_Data", "Succeeded_Bench", "Succeeded_Data",
        "EPB_abs_diff", "EPB_abs_diff_norm",
    ]].rename(columns={
        "Split_Bench": "Split",
        "MolName_Bench": "Mol Name",
        "AtomNum_Bench": "Atom Number",
        "Grid_Bench": "Grid Size",
        "Tol_num_Bench": "Tolerance",
        "Tol_cat_Bench": "Tolerance_cat",
        "CUDA_Bench": "CUDA Used",
        "Solver_Data": "Solver Name",
    })


# ---------------------------------------------------------------------------
# EPB_analysis_graph.ipynb: wide tables in the cleaned schema
# ---------------------------------------------------------------------------


def normalize_runs(df):
    """Coerce a cleaned run table to comparable dtypes and add paper labels.

    Adds ``Solver`` (CG, BiCG, ...), ``Device`` (CPU/GPU) and ``Size Group``.
    """
    t = df.copy()
    t["Split"] = t["Split"].map(clean_str)
    t["Mol Name"] = t["Mol Name"].map(clean_str)
    t["Atom Number"] = t["Atom Number"].map(to_float).round().astype("Int64")
    t["Grid Size"] = t["Grid Size"].map(to_float).round(4)
    t["Tolerance"] = t["Tolerance"].map(normalize_tolerance)
    t["CUDA Used"] = t["CUDA Used"].map(to_float).round().astype("Int64")
    for col in ("EPB Value", "Elapsed Time", "Iterations"):
        if col in t.columns:
            t[col] = t[col].map(to_float)
    # Where the flag is missing, a run succeeded if it produced an energy
    # (the rule Data_cleaning.ipynb applies).
    succeeded = t["Succeeded"].map(to_float)
    t["Succeeded"] = succeeded.fillna(t["EPB Value"].notna().astype(float)).round().astype(int)
    t["Solver"] = t["Solver Name"].map(solver_label)
    t["Device"] = t["CUDA Used"].map(device_label)
    t["Size Group"] = size_group(t["Atom Number"])
    return t


def merge_runs(benchmark, pb_data, bench_solver=None):
    """Join AMBER PBSA rows (``benchmark``) with AmberTorchPB rows (``pb_data``).

    Rows match on Split, Mol Name, Atom Number, Grid Size, CUDA Used and
    Tolerance. Each AmberTorchPB solver is compared with the same reference
    row. When the benchmark holds more than one reference solver, pick one
    with ``bench_solver`` (class name or paper label).
    """
    bench = normalize_runs(benchmark[ANALYSIS_COLUMNS])
    if bench_solver is not None:
        bench = bench[(bench["Solver Name"] == bench_solver) | (bench["Solver"] == bench_solver)]
    if bench.duplicated(MERGE_KEYS).any():
        warnings.warn(
            "benchmark has more than one row per merge key; pass bench_solver= "
            "to choose the reference solver",
            stacklevel=2,
        )

    extra = ["Iterations"] if "Iterations" in pb_data.columns else []
    data = normalize_runs(pb_data[ANALYSIS_COLUMNS + extra])

    df = pd.merge(
        bench[ANALYSIS_COLUMNS],
        data[ANALYSIS_COLUMNS + extra],
        on=MERGE_KEYS,
        how="inner",
        suffixes=("_Bench", "_Data"),
    ).rename(columns={
        "EPB Value_Bench": "EPB_Bench",
        "EPB Value_Data": "EPB_Data",
        "Solver Name_Data": "Solver Name",
    }).drop(columns="Solver Name_Bench")

    df = df[[
        "Split", "Mol Name", "Atom Number", "Solver Name", "Grid Size", "Tolerance", "CUDA Used",
        "EPB_Bench", "EPB_Data", "Succeeded_Bench", "Succeeded_Data",
        "Elapsed Time_Bench", "Elapsed Time_Data",
    ] + extra].copy()
    df["Solver"] = df["Solver Name"].map(solver_label)
    df["Device"] = df["CUDA Used"].map(device_label)
    df["Size Group"] = size_group(df["Atom Number"])
    df["EPB_abs_diff"] = (df["EPB_Bench"] - df["EPB_Data"]).abs()
    df["EPB_ape"] = 100 * df["EPB_abs_diff"] / df["EPB_Bench"].abs()
    return df.sort_values(["Split", "Mol Name", "CUDA Used"], ignore_index=True)


def subset(df, solver=None, device=None, spacing=None, tolerance=None, molecules=None):
    """Filter by paper solver label(s), CPU/GPU, grid spacing, tolerance, molecules."""
    mask = pd.Series(True, index=df.index)
    if solver is not None:
        mask &= df["Solver"].isin([solver] if isinstance(solver, str) else list(solver))
    if device is not None:
        mask &= df["Device"] == device
    if spacing is not None:
        mask &= np.isclose(df["Grid Size"].astype(float), spacing)
    if tolerance is not None:
        mask &= np.isclose(df["Tolerance"].astype(float), tolerance, rtol=1e-3, atol=0)
    if molecules is not None:
        mask &= df["Mol Name"].isin(list(molecules))
    return df[mask]
