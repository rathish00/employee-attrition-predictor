"""Data loading, validation, and cleaning.

Keeping this separate from modeling means the same validated, cleaned
frame is guaranteed to reach both the EDA scripts and the training
pipeline — no risk of them silently diverging.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from attrition_predictor.config import Config
from attrition_predictor.exceptions import DataValidationError
from attrition_predictor.logging_config import get_logger

logger = get_logger(__name__)

# Columns the real IBM HR Attrition dataset always has. Used to fail fast
# with a clear error if someone points this pipeline at the wrong file.
REQUIRED_COLUMNS = {
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate it against the expected schema.

    Args:
        path: Path to a CSV with the IBM HR Attrition column schema.

    Returns:
        The raw, unmodified DataFrame.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        DataValidationError: if required columns are missing, the target
            column has unexpected values, or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No data file at {path}. Run `python3 data/generate_data.py` "
            "or drop the real Kaggle CSV at that path."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError(f"{path} loaded but contains 0 rows.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"{path} is missing required columns: {sorted(missing)}. "
            "This doesn't look like the IBM HR Attrition schema."
        )

    bad_target_values = set(df["Attrition"].unique()) - {"Yes", "No"}
    if bad_target_values:
        raise DataValidationError(
            f"Attrition column has unexpected values: {bad_target_values}. Expected only 'Yes'/'No'."
        )

    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df


def clean_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Drop non-informative columns and add a numeric target flag.

    Args:
        df: Raw, validated DataFrame (see ``load_raw_data``).
        config: Loaded pipeline config (drives which columns get dropped).

    Returns:
        Cleaned DataFrame with constant/ID columns removed and an
        ``AttritionFlag`` (0/1) column added.
    """
    drop_cols = [c for c in config.constant_columns + config.id_columns if c in df.columns]
    cleaned = df.drop(columns=drop_cols).copy()
    cleaned["AttritionFlag"] = (cleaned[config.target_column] == "Yes").astype(int)

    n_missing = cleaned.isnull().sum().sum()
    if n_missing:
        logger.warning("%d missing values found after cleaning — filling numeric with median, categorical with mode", n_missing)
        for col in cleaned.columns:
            if cleaned[col].isnull().any():
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                else:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])

    logger.info(
        "Cleaned data: %d rows, %d columns, attrition rate %.1f%%, dropped %s",
        len(cleaned), cleaned.shape[1], cleaned["AttritionFlag"].mean() * 100, drop_cols,
    )
    return cleaned


def split_feature_columns(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns), excluding the raw target
    and its numeric flag."""
    exclude = {target_column, "AttritionFlag"}
    feature_df = df.drop(columns=[c for c in exclude if c in df.columns])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols
