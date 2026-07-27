import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition_predictor.predict import assess


class TestAssess(unittest.TestCase):
    def test_tier_thresholds(self):
        self.assertEqual(assess({}, 0.05).tier, "Low")
        self.assertEqual(assess({}, 0.45).tier, "Moderate")
        self.assertEqual(assess({}, 0.75).tier, "High")

    def test_boundary_values(self):
        self.assertEqual(assess({}, 0.3).tier, "Moderate")
        self.assertEqual(assess({}, 0.6).tier, "High")
        self.assertEqual(assess({}, 0.2999).tier, "Low")

    def test_flags_overtime_as_risk_factor(self):
        result = assess({"OverTime": "Yes"}, 0.5)
        self.assertIn("Works overtime", result.risk_factors)

    def test_no_overtime_not_flagged(self):
        result = assess({"OverTime": "No"}, 0.5)
        self.assertNotIn("Works overtime", result.risk_factors)

    def test_low_income_flagged(self):
        result = assess({"MonthlyIncome": 2000}, 0.5)
        self.assertIn("Below-median monthly income", result.risk_factors)

    def test_high_income_not_flagged(self):
        result = assess({"MonthlyIncome": 8000}, 0.5)
        self.assertNotIn("Below-median monthly income", result.risk_factors)

    def test_senior_and_tenured_are_protective(self):
        result = assess({"JobLevel": 5, "YearsAtCompany": 12}, 0.2)
        self.assertIn("Senior job level", result.protective_factors)
        self.assertIn("Long tenure", result.protective_factors)

    def test_always_returns_at_least_one_action(self):
        result = assess({}, 0.5)
        self.assertGreaterEqual(len(result.recommended_actions), 1)

    def test_missing_fields_do_not_raise(self):
        # employee dict with no keys at all should not raise a KeyError
        result = assess({}, 0.5)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
