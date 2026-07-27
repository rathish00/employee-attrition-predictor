"""Cold-start bootstrap for deployment platforms (e.g. Streamlit Community
Cloud) that only run ``app/app.py`` — they never invoke
``notebooks/01_eda.py`` or ``02_train_model.py`` as a separate build step.

Committing trained model binaries to git is deliberately avoided (bloats
the repo, can't be diffed, goes stale silently). Instead, the app checks
for the expected artifacts on startup and builds them in-process if
they're missing — a few seconds on first load, then cached for the life
of the container via ``st.cache_resource`` in app.py.
"""
from __future__ import annotations

from typing import Callable

from attrition_predictor.config import Config
from attrition_predictor.data import clean_data, load_raw_data, split_feature_columns
from attrition_predictor.logging_config import get_logger
from attrition_predictor.model import AttritionModel, train_and_select
from attrition_predictor.synthetic_data import generate_synthetic_dataset

logger = get_logger(__name__)

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def ensure_model_ready(config: Config, on_progress: ProgressFn = _noop) -> AttritionModel:
    """Guarantee a trained model exists on disk, building it if necessary,
    then return it loaded.

    Order of operations:
      1. If raw data is missing, generate the synthetic stand-in dataset.
      2. If the trained model is missing, clean the data and train it.
      3. Load and return the model.

    This makes cold starts self-sufficient: a fresh clone with only
    source code (no committed CSVs/pickles) still produces a working app.

    Args:
        config: loaded pipeline config.
        on_progress: optional callback for status messages (e.g. wired to
            st.spinner text in the app); called with human-readable stage
            descriptions.

    Returns:
        A ready-to-use AttritionModel.
    """
    if not config.paths.raw_data.exists():
        on_progress("No dataset found — generating data...")
        logger.info("Raw data missing at %s — generating synthetic dataset", config.paths.raw_data)
        df = generate_synthetic_dataset()
        config.paths.raw_data.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.paths.raw_data, index=False)

    if not (config.paths.model_file.exists() and config.paths.schema_file.exists()):
        on_progress("No trained model found — training now (first run only)...")
        logger.info("Model artifacts missing — training from scratch")
        raw_df = load_raw_data(config.paths.raw_data)
        cleaned_df = clean_data(raw_df, config)
        numeric_cols, categorical_cols = split_feature_columns(cleaned_df, config.target_column)
        X = cleaned_df[numeric_cols + categorical_cols]
        y = cleaned_df["AttritionFlag"]

        result = train_and_select(X, y, numeric_cols, categorical_cols, config)
        categories = {c: sorted(X[c].dropna().unique().tolist()) for c in categorical_cols}
        model = AttritionModel(
            pipeline=result.best_pipeline,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            categories=categories,
        )
        model.save(config)
        logger.info("Bootstrap training complete: %s, ROC-AUC=%.4f",
                     result.best_model_name, result.all_scores[result.best_model_name])

    on_progress("Loading model...")
    return AttritionModel.load(config)
