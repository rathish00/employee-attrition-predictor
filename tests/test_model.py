import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from attrition_predictor.config import Config
from attrition_predictor.data import clean_data, load_raw_data, split_feature_columns
from attrition_predictor.exceptions import DataValidationError, ModelNotFoundError
from attrition_predictor.model import AttritionModel, get_feature_importance, train_and_select

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_small_dataset(config: Config, n: int = 300):
    raw_path = config.paths.raw_data
    df = clean_data(load_raw_data(raw_path), config)
    df = df.sample(n=min(n, len(df)), random_state=0)
    numeric_cols, categorical_cols = split_feature_columns(df, config.target_column)
    X = df[numeric_cols + categorical_cols]
    y = df["AttritionFlag"]
    return X, y, numeric_cols, categorical_cols


class TestTrainAndSelect(unittest.TestCase):
    def setUp(self):
        self.config = Config.load()
        if not self.config.paths.raw_data.exists():
            self.skipTest("Run data/generate_data.py first")
        self.X, self.y, self.numeric_cols, self.categorical_cols = _load_small_dataset(self.config)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            train_and_select(self.X, self.y.iloc[:-1], self.numeric_cols, self.categorical_cols, self.config)

    def test_rejects_non_binary_target(self):
        bad_y = self.y.copy()
        bad_y.iloc[0] = 2
        with self.assertRaises(ValueError):
            train_and_select(self.X, bad_y, self.numeric_cols, self.categorical_cols, self.config)

    def test_selects_best_by_auc_and_returns_predictions(self):
        result = train_and_select(self.X, self.y, self.numeric_cols, self.categorical_cols, self.config)
        self.assertIn(result.best_model_name, result.all_scores)
        self.assertEqual(result.all_scores[result.best_model_name], max(result.all_scores.values()))
        self.assertEqual(len(result.y_pred), len(result.X_test))
        self.assertTrue(((result.y_proba >= 0) & (result.y_proba <= 1)).all())

    def test_feature_importance_is_nonempty_and_sorted(self):
        result = train_and_select(self.X, self.y, self.numeric_cols, self.categorical_cols, self.config)
        importances = get_feature_importance(result, top_n=10)
        self.assertGreater(len(importances), 0)
        self.assertTrue((importances.values[:-1] >= importances.values[1:]).all())


class TestAttritionModel(unittest.TestCase):
    """Exercises the save/load/predict contract used by the app."""

    def setUp(self):
        self.config = Config.load()
        if not self.config.paths.raw_data.exists():
            self.skipTest("Run data/generate_data.py first")

    def test_load_raises_when_missing(self):
        fake_config = Config.load()
        object.__setattr__(fake_config.paths, "model_file", PROJECT_ROOT / "models" / "does_not_exist.pkl")
        with self.assertRaises(ModelNotFoundError):
            AttritionModel.load(fake_config)

    def test_predict_proba_rejects_missing_columns(self):
        if not self.config.paths.model_file.exists():
            self.skipTest("Train a model first (notebooks/02_train_model.py)")
        model = AttritionModel.load(self.config)
        bad_input = pd.DataFrame([{"Age": 30}])
        with self.assertRaises(DataValidationError):
            model.predict_proba(bad_input)

    def test_predict_proba_returns_valid_probability(self):
        if not self.config.paths.model_file.exists():
            self.skipTest("Train a model first (notebooks/02_train_model.py)")
        model = AttritionModel.load(self.config)
        df = load_raw_data(self.config.paths.raw_data)
        row = df.iloc[[0]]
        proba = model.predict_proba(row)
        self.assertEqual(len(proba), 1)
        self.assertTrue(0.0 <= proba[0] <= 1.0)

    def test_high_and_low_risk_profiles_are_ordered_correctly(self):
        """A textbook high-risk profile should score meaningfully higher
        than a textbook low-risk one — catches silently-broken training."""
        if not self.config.paths.model_file.exists():
            self.skipTest("Train a model first (notebooks/02_train_model.py)")
        model = AttritionModel.load(self.config)
        defaults = dict(
            DailyRate=800, HourlyRate=65, MonthlyRate=14000, PercentSalaryHike=15,
            PerformanceRating=3, RelationshipSatisfaction=3, JobInvolvement=3,
            Education=3, TrainingTimesLastYear=2, YearsInCurrentRole=4, YearsWithCurrManager=4,
            Department="Sales", JobRole="Sales Executive", MaritalStatus="Single",
            BusinessTravel="Travel_Frequently", Gender="Male", EducationField="Marketing",
        )
        high_risk = pd.DataFrame([{
            **defaults, "Age": 28, "MonthlyIncome": 2200, "DistanceFromHome": 25,
            "NumCompaniesWorked": 6, "TotalWorkingYears": 5, "JobSatisfaction": 1,
            "EnvironmentSatisfaction": 1, "WorkLifeBalance": 1, "JobLevel": 1,
            "StockOptionLevel": 0, "YearsSinceLastPromotion": 8, "YearsAtCompany": 2,
            "OverTime": "Yes",
        }])
        low_risk = pd.DataFrame([{
            **defaults, "Age": 45, "MonthlyIncome": 9500, "DistanceFromHome": 2,
            "NumCompaniesWorked": 1, "TotalWorkingYears": 20, "JobSatisfaction": 4,
            "EnvironmentSatisfaction": 4, "WorkLifeBalance": 3, "JobLevel": 4,
            "StockOptionLevel": 2, "YearsSinceLastPromotion": 1, "YearsAtCompany": 15,
            "OverTime": "No",
        }])
        p_high = model.predict_proba(high_risk)[0]
        p_low = model.predict_proba(low_risk)[0]
        self.assertGreater(p_high, p_low)


if __name__ == "__main__":
    unittest.main()
