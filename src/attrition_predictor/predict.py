"""Business-rule explanations layered on top of the model's raw probability.

The model gives a number; HR managers need a reason and an action. This
module encodes the same domain rules used to generate/validate the
training data's risk drivers, kept in one place so the app doesn't embed
this logic inline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskAssessment:
    """A scored employee with a human-readable explanation."""

    probability: float
    tier: str  # "High" | "Moderate" | "Low"
    risk_factors: list[str]
    protective_factors: list[str]
    recommended_actions: list[str]


def _tier(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.3:
        return "Moderate"
    return "Low"


def assess(employee: dict, probability: float) -> RiskAssessment:
    """Build a full risk assessment for one employee.

    Args:
        employee: dict of raw feature values (same keys as the model's
            required columns — this function only reads a subset of them).
        probability: model's predicted P(attrition=Yes) for this employee.

    Returns:
        A RiskAssessment with tier, contributing factors, and suggested
        HR actions.
    """
    risk_factors: list[str] = []
    protective_factors: list[str] = []

    if employee.get("OverTime") == "Yes":
        risk_factors.append("Works overtime")
    if employee.get("MonthlyIncome", float("inf")) < 3000:
        risk_factors.append("Below-median monthly income")
    if employee.get("DistanceFromHome", 0) > 15:
        risk_factors.append("Long commute (>15 miles)")
    if employee.get("JobSatisfaction", 4) <= 2:
        risk_factors.append("Low job satisfaction")
    if employee.get("EnvironmentSatisfaction", 4) <= 2:
        risk_factors.append("Low environment satisfaction")
    if employee.get("WorkLifeBalance", 4) == 1:
        risk_factors.append("Poor work-life balance")
    if employee.get("YearsSinceLastPromotion", 0) > 5:
        risk_factors.append("No promotion in 5+ years")
    if employee.get("Age", 100) < 30:
        risk_factors.append("Early-career (higher baseline mobility)")
    if employee.get("NumCompaniesWorked", 0) >= 5:
        risk_factors.append("History of frequent job changes")
    if employee.get("StockOptionLevel", 1) == 0:
        risk_factors.append("No equity/stock options")

    if employee.get("JobLevel", 0) >= 4:
        protective_factors.append("Senior job level")
    if employee.get("YearsAtCompany", 0) > 10:
        protective_factors.append("Long tenure")
    if employee.get("StockOptionLevel", 0) >= 2:
        protective_factors.append("Meaningful equity stake")

    actions: list[str] = []
    if "Works overtime" in risk_factors:
        actions.append("Review workload/staffing to reduce overtime dependence")
    if "Below-median monthly income" in risk_factors:
        actions.append("Benchmark salary against market rate for this role")
    if "Low job satisfaction" in risk_factors or "Low environment satisfaction" in risk_factors:
        actions.append("Schedule a 1:1 check-in on job/team satisfaction")
    if "No promotion in 5+ years" in risk_factors:
        actions.append("Discuss career progression and promotion timeline")
    if "Poor work-life balance" in risk_factors:
        actions.append("Explore flexible scheduling or remote-work options")
    if not actions:
        actions.append("No immediate action flagged — maintain regular engagement check-ins")

    return RiskAssessment(
        probability=probability,
        tier=_tier(probability),
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        recommended_actions=actions,
    )
