"""Raw run output -> cleaned table (the steps of Data_cleaning.ipynb)."""

import pandas as pd

from .schema import CLEANED_COLUMNS, RAW_TO_CLEAN, SOLVER_INDEX_TO_NAME, SYSTEM_TO_SPLIT

# e.g. /runs/solver_0/1P4Z.p22_g055.out -> molecule "1P4Z.p22", solver index "0"
MOL_NAME_PATTERN = r"/([^/]+)_g\d{2,4}"
SOLVER_INDEX_PATTERN = r"/solver_(\d)/"


def clean_raw(raw):
    """Return the cleaned table for a raw experiment CSV loaded as ``raw``."""
    df = raw.copy()
    df["Mol Name"] = df["file"].str.extract(MOL_NAME_PATTERN, expand=False)
    df["Solver Name"] = df["file"].str.extract(SOLVER_INDEX_PATTERN, expand=False)
    df = df.rename(columns=RAW_TO_CLEAN)

    # A run counts as succeeded when it produced an energy.
    df["Succeeded"] = (~df["EPB Value"].isna()).astype(int)
    df["Split"] = df["Split"].replace(SYSTEM_TO_SPLIT)
    df["Solver Name"] = df["Solver Name"].replace(SOLVER_INDEX_TO_NAME)

    columns = CLEANED_COLUMNS + (["Iterations"] if "Iterations" in df.columns else [])
    df = df[columns].copy()
    df["Mol Name"] = df["Mol Name"].str.replace(".p22", "", regex=False)
    df["Elapsed Time"] = df["Elapsed Time"].round(2)
    df["EPB Value"] = df["EPB Value"].round(4)
    df["Sor Omega"] = pd.NA
    df["Other Comments"] = pd.NA
    return df
