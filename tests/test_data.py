import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from attrition_predictor.config import Config
from attrition_predictor.data import clean_data, load_raw_data, split_feature_columns
from attrition_predictor.exceptions import DataValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestLoadRawData(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_raw_data(PROJECT_ROOT / "data" / "does_not_exist.csv")

    def test_missing_required_columns_raises(self):
        bad = PROJECT_ROOT / "tests" / "_tmp_bad.csv"
        pd.DataFrame({"Age": [30, 40], "Attrition": ["Yes", "No"]}).to_csv(bad, index=False)
        try:
            with self.assertRaises(DataValidationError):
                load_raw_data(bad)
        finally:
            bad.unlink(missing_ok=True)

    def test_bad_target_values_raise(self):
        cols = {c: [1, 1] for c in [
            "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
            "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
            "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
            "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
            "OverTime", "PercentSalaryHike", "PerformanceRating",
            "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
            "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
            "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
        ]}
        cols["Attrition"] = ["Maybe", "Maybe"]
        bad = PROJECT_ROOT / "tests" / "_tmp_bad_target.csv"
        pd.DataFrame(cols).to_csv(bad, index=False)
        try:
            with self.assertRaises(DataValidationError):
                load_raw_data(bad)
        finally:
            bad.unlink(missing_ok=True)

    def test_real_data_loads(self):
        raw_path = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
        if not raw_path.exists():
            self.skipTest("Run data/generate_data.py first")
        df = load_raw_data(raw_path)
        self.assertGreater(len(df), 0)
        self.assertIn("Attrition", df.columns)


class TestCleanData(unittest.TestCase):
    def setUp(self):
        self.config = Config.load()
        raw_path = self.config.paths.raw_data
        if not raw_path.exists():
            self.skipTest("Run data/generate_data.py first")
        self.df = load_raw_data(raw_path)

    def test_drops_constant_and_id_columns(self):
        cleaned = clean_data(self.df, self.config)
        for col in self.config.constant_columns + self.config.id_columns:
            self.assertNotIn(col, cleaned.columns)

    def test_adds_attrition_flag(self):
        cleaned = clean_data(self.df, self.config)
        self.assertIn("AttritionFlag", cleaned.columns)
        self.assertTrue(set(cleaned["AttritionFlag"].unique()).issubset({0, 1}))

    def test_no_missing_values_after_cleaning(self):
        cleaned = clean_data(self.df, self.config)
        self.assertEqual(cleaned.isnull().sum().sum(), 0)

    def test_split_feature_columns_excludes_target(self):
        cleaned = clean_data(self.df, self.config)
        numeric, categorical = split_feature_columns(cleaned, self.config.target_column)
        self.assertNotIn("Attrition", numeric + categorical)
        self.assertNotIn("AttritionFlag", numeric + categorical)


if __name__ == "__main__":
    unittest.main()
