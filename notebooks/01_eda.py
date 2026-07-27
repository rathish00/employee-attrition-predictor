"""Step 1: Data Cleaning & Exploratory Data Analysis.

Run: python3 notebooks/01_eda.py [--config config.yaml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from attrition_predictor.config import Config
from attrition_predictor.data import clean_data, load_raw_data
from attrition_predictor.exceptions import AttritionPredictorError
from attrition_predictor.logging_config import get_logger

logger = get_logger(__name__)
sns.set_style("whitegrid")


def run(config: Config) -> None:
    out_dir = config.paths.eda_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(config.paths.raw_data)
    df = clean_data(df, config)

    # 1. Attrition by OverTime
    plt.figure(figsize=(6, 4))
    rates = df.groupby("OverTime")["AttritionFlag"].mean().sort_values(ascending=False)
    sns.barplot(x=rates.index, y=rates.values, hue=rates.index, palette="Reds_r", legend=False)
    plt.ylabel("Attrition Rate")
    plt.title("Attrition Rate by Overtime Status")
    for i, v in enumerate(rates.values):
        plt.text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "01_attrition_by_overtime.png", dpi=150)
    plt.close()

    # 2. Monthly income distribution by attrition
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition",
                palette={"Yes": "#d62728", "No": "#2ca02c"}, legend=False)
    plt.title("Monthly Income vs Attrition")
    plt.tight_layout()
    plt.savefig(out_dir / "02_income_vs_attrition.png", dpi=150)
    plt.close()

    # 3. Distance from home
    plt.figure(figsize=(7, 4.5))
    sns.kdeplot(data=df, x="DistanceFromHome", hue="Attrition", fill=True, common_norm=False, alpha=0.4)
    plt.title("Distance From Home Density by Attrition")
    plt.tight_layout()
    plt.savefig(out_dir / "03_distance_vs_attrition.png", dpi=150)
    plt.close()

    # 4. Attrition rate by job role
    plt.figure(figsize=(8, 5))
    role_rates = df.groupby("JobRole")["AttritionFlag"].mean().sort_values(ascending=False)
    sns.barplot(x=role_rates.values, y=role_rates.index, hue=role_rates.index, palette="Reds_r", legend=False)
    plt.xlabel("Attrition Rate")
    plt.title("Attrition Rate by Job Role")
    plt.tight_layout()
    plt.savefig(out_dir / "04_attrition_by_role.png", dpi=150)
    plt.close()

    # 5. Satisfaction heatmap
    plt.figure(figsize=(6.5, 5))
    pivot = df.pivot_table(values="AttritionFlag", index="JobSatisfaction", columns="WorkLifeBalance", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds")
    plt.title("Attrition Rate: Job Satisfaction x Work-Life Balance")
    plt.tight_layout()
    plt.savefig(out_dir / "05_satisfaction_heatmap.png", dpi=150)
    plt.close()

    # 6. Correlation with attrition
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()["AttritionFlag"].drop("AttritionFlag").sort_values()
    plt.figure(figsize=(7, 8))
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in corr.values]
    plt.barh(corr.index, corr.values, color=colors)
    plt.title("Correlation of Numeric Features with Attrition")
    plt.tight_layout()
    plt.savefig(out_dir / "06_correlation_with_attrition.png", dpi=150)
    plt.close()

    config.paths.cleaned_data.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.paths.cleaned_data, index=False)
    logger.info("Saved cleaned data -> %s", config.paths.cleaned_data)
    logger.info("Saved 6 EDA plots -> %s", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    args = parser.parse_args()

    try:
        config = Config.load(args.config) if args.config else Config.load()
        run(config)
    except AttritionPredictorError as e:
        logger.error("EDA failed: %s", e)
        return 1
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
