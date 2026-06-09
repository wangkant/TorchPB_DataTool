"""Figures of the paper, drawn from merge_runs() / normalize_runs() tables.

Every function returns the matplotlib Figure (or seaborn JointGrid), or None
when the requested subset has no rows, so callers can save or skip.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
import pandas as pd
import seaborn as sns

from .compare import subset
from .metrics import r2
from .schema import (
    BENCH_LABEL,
    DATA_LABEL,
    DEFAULT_SPACING,
    DEFAULT_TOLERANCE,
    ENERGY_UNIT,
    ITERATION_SOLVERS,
    PAPER_SOLVERS,
    solver_order,
)

ENERGY = r"$\Delta G_{\mathrm{PB}}$"


def energy_label(who):
    return f"{who} {ENERGY} ({ENERGY_UNIT})"


def _levels(series):
    return sorted(series.dropna().unique())


def _palette(levels):
    return dict(zip(levels, sns.color_palette("deep", len(levels))))


def _r2_labels(ax, d, x, y, hue, levels, palette):
    # Stacked bottom-right, first level on top, colored like its points.
    x0, y0, dy = 0.95, 0.05, 0.06
    for i, level in enumerate(reversed(levels)):
        sub = d[d[hue] == level]
        ax.text(
            x0, y0 + i * dy,
            f"R²: {r2(sub[x], sub[y]):.4f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=14,
            color=palette[level],
        )


def parity_plot(df, x="EPB_Bench", y="EPB_Data", hue="Split", xlabel=None, ylabel=None, title=None, height=6):
    """Scatter with marginals, the y = x line, and R^2 per hue level (Figs. 4, 8, 15)."""
    d = df[np.isfinite(df[x].astype(float)) & np.isfinite(df[y].astype(float))]
    if d.empty:
        return None
    levels = _levels(d[hue])
    palette = _palette(levels)

    g = sns.jointplot(
        data=d,
        x=x, y=y,
        hue=hue, hue_order=levels, palette=palette,
        kind="scatter",
        marginal_ticks=False,
        height=height,
        alpha=0.8,
    )
    g.set_axis_labels(xlabel or x, ylabel or y)
    g.ax_joint.axline((0, 0), (1, 1), linestyle="--", color="gray", linewidth=1)
    _r2_labels(g.ax_joint, d, x, y, hue, levels, palette)
    if title:
        g.figure.suptitle(title, y=1.02)
    return g


def solver_parity_plot(df, solver, device="CPU", spacing=DEFAULT_SPACING, tolerance=DEFAULT_TOLERANCE, hue="Split"):
    """AmberTorchPB ``solver`` against AMBER PBSA at one setting (Fig. 4 / Fig. 8 panel)."""
    d = subset(df, solver=solver, device=device, spacing=spacing, tolerance=tolerance)
    d = d[(d["Succeeded_Bench"] == 1) & (d["Succeeded_Data"] == 1)]
    return parity_plot(
        d,
        hue=hue,
        xlabel=energy_label(f"{BENCH_LABEL} ({device})"),
        ylabel=energy_label(f"{DATA_LABEL} {solver} ({device})"),
        title=f"{solver}, h = {spacing} Å, tol = {tolerance:.0e}",
    )


def device_consistency_plot(runs, solver, spacing=DEFAULT_SPACING, tolerance=DEFAULT_TOLERANCE):
    """AmberTorchPB energies on GPU against CPU for one solver (Fig. 5).

    ``runs`` is a normalize_runs() table of AmberTorchPB rows.
    """
    d = subset(runs, solver=solver, spacing=spacing, tolerance=tolerance)
    d = d[d["Succeeded"] == 1]
    if d.empty:
        return None
    levels = _levels(d["Split"])
    palette = _palette(levels)
    keys = ["Split", "Mol Name", "Grid Size", "Tolerance"]
    wide = d.pivot_table(index=keys, columns="Device", values="EPB Value", aggfunc="first").reset_index()

    fig, (ax_scatter, ax_violin) = plt.subplots(1, 2, figsize=(12, 5))
    if {"CPU", "GPU"} <= set(wide.columns):
        sns.scatterplot(data=wide, x="CPU", y="GPU", hue="Split", hue_order=levels, palette=palette, alpha=0.8, ax=ax_scatter)
        ax_scatter.axline((0, 0), (1, 1), linestyle="--", color="gray", linewidth=1)
        _r2_labels(ax_scatter, wide, "CPU", "GPU", "Split", levels, palette)
    ax_scatter.set(xlabel=energy_label(f"{DATA_LABEL} {solver} (CPU)"), ylabel=energy_label(f"{DATA_LABEL} {solver} (GPU)"))

    devices = [dev for dev in ("CPU", "GPU") if dev in set(d["Device"])]
    sns.violinplot(
        data=d, x="Split", y="EPB Value", order=levels,
        hue="Device", hue_order=devices, split=len(devices) == 2, inner="quart", ax=ax_violin,
    )
    ax_violin.set(xlabel="", ylabel=energy_label(DATA_LABEL))
    fig.suptitle(f"{solver}: CPU vs GPU, h = {spacing} Å, tol = {tolerance:.0e}")
    fig.tight_layout()
    return fig


def grid_spacing_plot(df, molecules, solver="CG", device="CPU", tolerance=DEFAULT_TOLERANCE):
    """Energy against grid spacing, AMBER PBSA (a) and AmberTorchPB (b) (Fig. 6)."""
    d = subset(df, solver=solver, device=device, tolerance=tolerance, molecules=molecules)
    if d.empty:
        return None
    present = [m for m in molecules if m in set(d["Mol Name"])]
    spacings = sorted(d["Grid Size"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col, who in zip(axes, ("EPB_Bench", "EPB_Data"), (f"{BENCH_LABEL} {solver}", f"{DATA_LABEL} {solver}")):
        sns.lineplot(data=d, x="Grid Size", y=col, hue="Mol Name", hue_order=present, marker="o", errorbar=None, ax=ax)
        ax.set(title=who, xlabel="Grid spacing (Å)", ylabel=energy_label(""), xticks=spacings)
    axes[1].get_legend().remove()
    fig.tight_layout()
    return fig


def tolerance_mape_plot(table):
    """Relative difference per molecule and tolerance, from tolerance_mape_table() (Fig. 7)."""
    if table.empty:
        return None
    t = table.assign(Tol=table["Tolerance"].map(lambda v: f"{v:.0e}"))
    tol_order = [f"{v:.0e}" for v in sorted(table["Tolerance"].unique(), reverse=True)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=t, x="Mol Name", y="APE (%)", hue="Tol", hue_order=tol_order, errorbar=None, ax=ax)
    ax.axhline(0.01, linestyle="--", color="gray", linewidth=1, label="0.01%")
    ax.set(xlabel="", ylabel="Relative difference (%)")
    ax.legend(title="Tolerance")
    fig.tight_layout()
    return fig


def iteration_hist(runs, solvers=ITERATION_SOLVERS, device="CPU", spacing=DEFAULT_SPACING, tolerance=DEFAULT_TOLERANCE):
    """Proportional histogram of converged iteration counts per split (Fig. 9)."""
    if "Iterations" not in runs.columns:
        return None
    d = subset(runs, solver=solvers, device=device, spacing=spacing, tolerance=tolerance)
    d = d[(d["Succeeded"] == 1) & (d["Iterations"] > 0)]
    if d.empty:
        return None
    splits = _levels(d["Split"])
    order = solver_order(d["Solver"])
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.5), squeeze=False, sharey=True)
    for ax, split in zip(axes[0], splits):
        sns.histplot(
            data=d[d["Split"] == split], x="Iterations", hue="Solver", hue_order=order,
            stat="probability", common_norm=False, log_scale=True, element="step", ax=ax,
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(split)
    fig.tight_layout()
    return fig


def iterations_vs_atoms(runs, solvers=ITERATION_SOLVERS, device="CPU", spacing=DEFAULT_SPACING, tolerance=DEFAULT_TOLERANCE):
    """Converged iteration count against atom number with a linear fit per solver (Fig. 11)."""
    if "Iterations" not in runs.columns:
        return None
    d = subset(runs, solver=solvers, device=device, spacing=spacing, tolerance=tolerance)
    d = d[d["Succeeded"] == 1].assign(Atoms=lambda t: t["Atom Number"].astype(float))
    if d.empty:
        return None
    order = solver_order(d["Solver"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for solver, color in zip(order, sns.color_palette("deep", len(order))):
        sub = d[d["Solver"] == solver]
        slope = np.polyfit(sub["Atoms"], sub["Iterations"], 1)[0] if len(sub) > 1 else np.nan
        sns.regplot(
            data=sub, x="Atoms", y="Iterations", ci=None, color=color, ax=ax,
            scatter_kws={"s": 12, "alpha": 0.6}, label=f"{solver} (slope {slope:.3g})",
        )
    ax.set(xlabel="Atom number", ylabel="Iterations to converge")
    ax.legend()
    fig.tight_layout()
    return fig


def elapsed_time_plot(df, solvers=PAPER_SOLVERS, bench_solver="CG", spacing=DEFAULT_SPACING, tolerance=DEFAULT_TOLERANCE):
    """Elapsed time of AMBER PBSA and each AmberTorchPB solver on CPU and GPU (Figs. 17, 19).

    Bars are means with standard-error whiskers on a log axis.
    """
    base = subset(df, spacing=spacing, tolerance=tolerance)
    bench = base[(base["Solver"] == bench_solver) & (base["Succeeded_Bench"] == 1)]
    data = base[base["Solver"].isin(list(solvers)) & (base["Succeeded_Data"] == 1)]
    bench_name = f"{BENCH_LABEL} {bench_solver}"
    long = pd.concat([
        pd.DataFrame({"Method": bench_name, "Device": bench["Device"], "Time": bench["Elapsed Time_Bench"]}),
        pd.DataFrame({"Method": data["Solver"], "Device": data["Device"], "Time": data["Elapsed Time_Data"]}),
    ], ignore_index=True)
    long = long[np.isfinite(long["Time"].astype(float)) & (long["Time"] > 0)]
    if long.empty:
        return None
    order = ([bench_name] if not bench.empty else []) + solver_order(data["Solver"])
    devices = [dev for dev in ("CPU", "GPU") if dev in set(long["Device"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long, x="Method", y="Time", order=order, hue="Device", hue_order=devices, errorbar="se", ax=ax)
    ax.set_yscale("log")
    ax.set(xlabel="", ylabel="Elapsed time (s)", title=f"h = {spacing} Å, tol = {tolerance:.0e}")
    fig.tight_layout()
    return fig


def log_diff_violin(df, x="Grid Size", hue="Split", solver=None, legend_loc="lower left"):
    """Distribution of log10 |EPB_Bench - EPB_Data| across ``x`` and ``hue``."""
    d = df if solver is None else subset(df, solver=solver)
    d = d[d["Succeeded_Data"] == 1].copy()
    d["log_abs_diff"] = np.log10(d["EPB_abs_diff"].replace(0, np.nan))
    d = d[np.isfinite(d["log_abs_diff"])]
    if d.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=d, x=x, y="log_abs_diff", hue=hue, hue_order=_levels(d[hue]), density_norm="width", ax=ax)
    ax.set_ylabel("log(|ΔEPB|)")
    ax.set_title(f"distribution across {x} & {hue}" + (f" ({solver})" if solver else ""))
    ax.legend(loc=legend_loc)
    return fig
