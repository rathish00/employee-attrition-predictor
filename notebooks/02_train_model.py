"""Step 2: Train, compare, and select the attrition model.

Run: python3 notebooks/02_train_model.py [--config config.yaml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from attrition_predictor.config import Config
from attrition_predictor.data import clean_data, load_raw_data, split_feature_columns
from attrition_predictor.exceptions import AttritionPredictorError
from attrition_predictor.logging_config import get_logger
from attrition_predictor.model import (
    AttritionModel,
    evaluation_report,
    get_feature_importance,
    train_and_select,
)

logger = get_logger(__name__)


def run(config: Config) -> None:
    out_dir = config.paths.model_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.paths.cleaned_data.exists():
        logger.info("No cleaned data found — loading and cleaning raw data first")
        df = clean_data(load_raw_data(config.paths.raw_data), config)
    else:
        df = pd.read_csv(config.paths.cleaned_data)

    numeric_cols, categorical_cols = split_feature_columns(df, config.target_column)
    X = df[numeric_cols + categorical_cols]
    y = df["AttritionFlag"]
    logger.info("Numeric features (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)

    result = train_and_select(X, y, numeric_cols, categorical_cols, config)

    # Text report
    report_text = evaluation_report(result)
    (out_dir / "model_report.txt").write_text(report_text, encoding="utf-8")
    logger.info("\n%s", report_text)

    # Confusion matrix
    cm = confusion_matrix(result.y_test, result.y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Left"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Reds", colorbar=False)
    plt.title(f"Confusion Matrix — {result.best_model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(result.y_test, result.y_proba)
    auc = result.all_scores[result.best_model_name]
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{result.best_model_name} (AUC={auc:.3f})", linewidth=2, color="#d62728")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    # Feature importance
    importances = get_feature_importance(result, top_n=15)
    if not importances.empty:
        plt.figure(figsize=(8, 6))
        plt.barh(importances.index[::-1], importances.values[::-1], color="#d62728")
        plt.title(f"Top 15 Feature Importances — {result.best_model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png", dpi=150)
        plt.close()
        logger.info("Top 10 drivers:\n%s", importances.head(10).to_string())

    # Persist model + schema
    categories = {c: sorted(X[c].dropna().unique().tolist()) for c in categorical_cols}
    model = AttritionModel(
        pipeline=result.best_pipeline,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categories=categories,
    )
    model.save(config)
    logger.info("Training complete. Best model: %s", result.best_model_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    args = parser.parse_args()

    try:
        config = Config.load(args.config) if args.config else Config.load()
        run(config)
    except AttritionPredictorError as e:
        logger.error("Training failed: %s", e)
        return 1
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
