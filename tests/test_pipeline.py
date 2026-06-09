"""End-to-end checks on the synthetic sample data."""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchpb_datatool import (  # noqa: E402
    accuracy_table,
    clean_raw,
    mape,
    merge_runs,
    r2,
    subset,
    tolerance_mape_table,
)
from torchpb_datatool import schema  # noqa: E402
from torchpb_datatool.compare import normalize_tolerance  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_script("generate_sample_data")


@pytest.fixture(scope="module")
def sample():
    return gen.build_raw_frame(), gen.build_benchmark_frame(), gen.build_processed_frame()


def test_generator_mirrors_schema():
    assert gen.CLEANED_COLUMNS == schema.ANALYSIS_COLUMNS
    assert set(gen.RAW_COLUMNS) - {"file"} == set(schema.RAW_TO_CLEAN)
    for token, split in gen.SYSTEM_TO_SPLIT.items():
        assert schema.SYSTEM_TO_SPLIT[token] == split
    for index, name in gen.SOLVER_INDEX_TO_NAME.items():
        assert schema.SOLVER_INDEX_TO_NAME[str(index)] == name
    assert tuple(gen.GRID_SPACINGS) == schema.GRID_SPACINGS
    assert sorted(gen.TOLERANCES) == sorted(schema.TOLERANCES)
    names = {m for mols in gen.MOLS_PER_SPLIT.values() for m in mols}
    assert set(schema.REPRESENTATIVE_MOLECULES) <= names


def test_solver_labels_follow_the_paper():
    labels = [schema.solver_label(schema.SOLVER_INDEX_TO_NAME[i]) for i in "01267"]
    assert labels == list(schema.PAPER_SOLVERS)
    assert schema.solver_label("SomethingElse") == "SomethingElse"


def test_clean_raw_matches_processed(sample):
    raw, _, processed = sample
    cleaned = clean_raw(raw)
    assert list(cleaned.columns) == schema.CLEANED_COLUMNS + ["Iterations", "Sor Omega", "Other Comments"]
    columns = schema.ANALYSIS_COLUMNS + ["Iterations"]
    pd.testing.assert_frame_equal(cleaned[columns], processed[columns], check_dtype=False)


def test_clean_raw_without_iterations(sample):
    raw, _, _ = sample
    cleaned = clean_raw(raw.drop(columns="itn"))
    assert "Iterations" not in cleaned.columns
    assert cleaned["Mol Name"].notna().all()


def test_metrics():
    y = np.array([-100.0, -200.0, -300.0])
    assert r2(y, y) == 1.0
    assert mape(y, y * 1.01) == pytest.approx(1.0)
    assert np.isnan(r2([1.0], [1.0]))
    # zero references and NaN pairs are skipped
    assert mape([0.0, -100.0, np.nan], [5.0, -110.0, 1.0]) == pytest.approx(10.0)


@pytest.mark.parametrize("text", ["1e-04", "1e-4", "0.0001", " 1E-4 ", "1e−4", 9.99999974737875e-05])
def test_normalize_tolerance(text):
    assert normalize_tolerance(text) == 1e-4


@pytest.mark.parametrize("text", [0, "0.0", "abc", None])
def test_normalize_tolerance_unset(text):
    assert np.isnan(normalize_tolerance(text))


def test_size_group():
    groups = schema.size_group(pd.Series([250, 499, 500, 5000, 5001]))
    assert list(groups) == [
        "Small (<500)", "Small (<500)", "Medium (500-5000)", "Medium (500-5000)", "Large (>5000)",
    ]


def test_merge_runs(sample):
    _, bench, processed = sample
    df = merge_runs(bench, processed)
    assert len(df) == len(processed)  # every run has a reference row
    assert set(df["Solver"]) == set(schema.PAPER_SOLVERS)
    assert set(df["Device"]) == {"CPU", "GPU"}

    table = accuracy_table(subset(df, spacing=schema.DEFAULT_SPACING, tolerance=schema.DEFAULT_TOLERANCE))
    assert list(table["Solver"].unique()) == list(schema.PAPER_SOLVERS)
    assert (table["R2"] > 0.999).all()


def test_merge_runs_tolerates_formatting(sample):
    _, bench, processed = sample
    messy = bench.copy()
    messy["Atom Number"] = messy["Atom Number"].map(lambda n: f"{n:,}")
    messy["Tolerance"] = messy["Tolerance"].map(lambda t: f"{t:.0e}")
    assert len(merge_runs(messy, processed)) == len(processed)


def test_merge_runs_with_two_reference_solvers(sample):
    _, bench, processed = sample
    doubled = pd.concat([bench, bench.assign(**{"Solver Name": "AMBER PBSA BiCG"})])
    with pytest.warns(UserWarning):
        merge_runs(doubled, processed)
    assert len(merge_runs(doubled, processed, bench_solver="AMBER PBSA CG")) == len(processed)


def test_tolerance_mape_table(sample):
    _, bench, processed = sample
    t = tolerance_mape_table(bench, processed, schema.REPRESENTATIVE_MOLECULES)
    assert list(t["Mol Name"].unique()) == list(schema.REPRESENTATIVE_MOLECULES)
    assert len(t) == len(schema.REPRESENTATIVE_MOLECULES) * len(schema.TOLERANCES)
    assert (t["APE (%)"] < 0.01).all()


def _run_make_figures(tmp_path, bench, processed):
    bench.to_csv(tmp_path / "benchmark.csv", index=False)
    processed.to_csv(tmp_path / "processed_epb.csv", index=False)
    load_script("make_figures").main([
        "--benchmark", str(tmp_path / "benchmark.csv"),
        "--data", str(tmp_path / "processed_epb.csv"),
        "--out", str(tmp_path / "results"),
    ])
    return {p.name for p in (tmp_path / "results" / "figures").glob("*.png")}


def test_make_figures(tmp_path, sample):
    _, bench, processed = sample
    figures = _run_make_figures(tmp_path, bench, processed)
    assert {
        "fig04_parity_cg.png", "fig04_parity_bicg.png", "fig05_cpu_gpu_cg.png",
        "fig06_grid_spacing_cg.png", "fig07_tolerance_mape_cg.png", "fig08_parity_gmres.png",
        "fig08_parity_sor.png", "fig09_iterations_hist.png", "fig11_iterations_vs_atoms.png",
        "fig17_19_elapsed_time.png",
    } <= figures
    tables = {p.name for p in (tmp_path / "results" / "tables").glob("*.csv")}
    assert {"accuracy_default_setting.csv", "tolerance_mape.csv", "speedup.csv"} <= tables


def test_make_figures_without_iterations(tmp_path, sample):
    _, bench, processed = sample
    figures = _run_make_figures(tmp_path, bench, processed.drop(columns="Iterations"))
    assert "fig04_parity_cg.png" in figures
    assert "fig09_iterations_hist.png" not in figures
