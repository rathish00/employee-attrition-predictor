"""Model training, selection, evaluation, and persistence.

Trains several candidate classifiers, selects the best by ROC-AUC on a
held-out test set, and exposes a single ``AttritionModel`` wrapper used
identically by the training script, the test suite, and the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition_predictor.config import Config
from attrition_predictor.exceptions import ModelNotFoundError, ModelNotTrainedError
from attrition_predictor.logging_config import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


@dataclass
class TrainingResult:
    """Everything the training script needs to report and plot."""

    best_model_name: str
    best_pipeline: Pipeline
    all_scores: dict[str, float]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    feature_names: list[str]


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Standard scale numerics, one-hot encode categoricals (unknowns ignored
    at inference time rather than raising)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def _candidate_models(config: Config, y_train: pd.Series) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            random_state=config.model.random_state, **config.model.random_forest_params
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=config.model.random_state, **config.model.gradient_boosting_params
        ),
    }
    if _HAS_XGB:
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=config.model.random_state,
            scale_pos_weight=neg / max(pos, 1),
        )
    else:
        logger.info("xgboost not installed — training without it (LogReg/RF/GBM only)")
    return candidates


def train_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: Config,
) -> TrainingResult:
    """Train every candidate model, evaluate on a held-out split, and return
    the best one by ROC-AUC along with full evaluation artifacts.

    Raises:
        ValueError: if X and y have mismatched lengths or y isn't binary.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} — must match.")
    if set(y.unique()) - {0, 1}:
        raise ValueError(f"y must be binary (0/1), got values: {set(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size,
        random_state=config.model.random_state, stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    scores: dict[str, float] = {}
    pipelines: dict[str, Pipeline] = {}

    for name, clf in _candidate_models(config, y_train).items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        scores[name] = auc
        pipelines[name] = pipe
        logger.info("%-20s ROC-AUC: %.4f", name, auc)

    best_name = max(scores, key=scores.get)
    best_pipe = pipelines[best_name]
    logger.info("Selected best model: %s (ROC-AUC=%.4f)", best_name, scores[best_name])

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    feature_names = list(best_pipe.named_steps["prep"].get_feature_names_out())

    return TrainingResult(
        best_model_name=best_name,
        best_pipeline=best_pipe,
        all_scores=scores,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=feature_names,
    )


def get_feature_importance(result: TrainingResult, top_n: int = 15) -> pd.Series:
    """Extract feature importances (tree models) or |coefficients| (linear
    models), sorted descending. Returns an empty Series if the model exposes
    neither."""
    clf = result.best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=result.feature_names).sort_values(ascending=False).head(top_n)


def evaluation_report(result: TrainingResult) -> str:
    """Human-readable text report: score comparison + sklearn classification report."""
    lines = [f"Best model: {result.best_model_name}", "", "ROC-AUC comparison:"]
    for name, auc in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:20s} {auc:.4f}")
    lines.append("")
    lines.append(classification_report(result.y_test, result.y_pred, target_names=["Stayed", "Left"]))
    return "\n".join(lines)


class AttritionModel:
    """Thin, safe wrapper around the persisted sklearn pipeline + schema.

    This is the ONLY class the app and any external caller should touch —
    it hides pickle/joblib details and validates inputs before they ever
    reach sklearn, turning cryptic sklearn errors into clear ones.
    """

    def __init__(self, pipeline: Pipeline, numeric_cols: list[str],
                 categorical_cols: list[str], categories: dict[str, list[str]]):
        self.pipeline = pipeline
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.categories = categories

    @property
    def required_columns(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols

    @classmethod
    def load(cls, config: Config) -> "AttritionModel":
        """Load a trained model + schema from disk.

        Raises:
            ModelNotFoundError: if either artifact file is missing.
        """
        model_path = config.paths.model_file
        schema_path = config.paths.schema_file
        if not model_path.exists() or not schema_path.exists():
            raise ModelNotFoundError(
                f"Model artifacts not found ({model_path}, {schema_path}). "
                "Run notebooks/02_train_model.py first."
            )
        pipeline = joblib.load(model_path)
        schema = joblib.load(schema_path)
        return cls(
            pipeline=pipeline,
            numeric_cols=schema["numeric_cols"],
            categorical_cols=schema["categorical_cols"],
            categories=schema["categories"],
        )

    def save(self, config: Config) -> None:
        config.paths.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, config.paths.model_file)
        joblib.dump(
            {
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
                "categories": self.categories,
            },
            config.paths.schema_file,
        )
        logger.info("Saved model -> %s, schema -> %s", config.paths.model_file, config.paths.schema_file)

    def _validate(self, df: pd.DataFrame) -> None:
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            from attrition_predictor.exceptions import DataValidationError
            raise DataValidationError(f"Input is missing required columns: {sorted(missing)}")
        if not self.pipeline:
            raise ModelNotTrainedError("Model pipeline is not loaded.")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(attrition=Yes) for each row.

        Raises:
            DataValidationError: if required columns are missing.
            ModelNotTrainedError: if called before a model is loaded.
        """
        self._validate(df)
        return self.pipeline.predict_proba(df[self.required_columns])[:, 1]

    def explain_row(self, df: pd.DataFrame) -> pd.Series:
        """Feature importances for the underlying model (global, not
        per-row SHAP — the model doesn't ship a per-row explainer to keep
        the dependency footprint small). Used by the app to show top drivers."""
        clf = self.pipeline.named_steps["clf"]
        feature_names = self.pipeline.named_steps["prep"].get_feature_names_out()
        if hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            values = np.abs(clf.coef_[0])
        else:
            return pd.Series(dtype=float)
        return pd.Series(values, index=feature_names).sort_values(ascending=False)
