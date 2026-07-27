"""Synthetic dataset generator matching the real IBM HR Attrition schema.

Extracted as a function (not just a script) so both the CLI
(``data/generate_data.py``) and the app's cold-start bootstrap
(``attrition_predictor.bootstrap_pipeline``) can call it without
shelling out to a subprocess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]
DEPT_PROBS = [0.31, 0.65, 0.04]
JOB_ROLES = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": [
        "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Research Director", "Manager",
    ],
    "Human Resources": ["Human Resources", "Manager"],
}
EDUCATION_FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
MARITAL = ["Single", "Married", "Divorced"]
TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
TRAVEL_PROBS = [0.10, 0.71, 0.19]


def generate_synthetic_dataset(n: int = 1470, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic dataset with the IBM HR Attrition dataset's exact
    35-column schema, with effect sizes calibrated so overtime, income,
    satisfaction, and commute distance come out as top attrition drivers —
    matching published HR attrition research directionally.

    Args:
        n: number of rows to generate (real dataset has 1470).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with the same columns/dtypes as
        WA_Fn-UseC_-HR-Employee-Attrition.csv.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        dept = rng.choice(DEPARTMENTS, p=DEPT_PROBS)
        role = rng.choice(JOB_ROLES[dept])
        age = int(np.clip(rng.normal(37, 9), 18, 60))
        distance = int(np.clip(rng.exponential(8), 1, 29))
        overtime = rng.choice(["Yes", "No"], p=[0.28, 0.72])
        job_level = rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.30, 0.15, 0.12, 0.08])
        monthly_income = int(np.clip(rng.normal(2000 + job_level * 3200, 1800), 1009, 20000))
        job_sat = rng.choice([1, 2, 3, 4])
        env_sat = rng.choice([1, 2, 3, 4])
        work_life = rng.choice([1, 2, 3, 4], p=[0.05, 0.25, 0.55, 0.15])
        years_at_company = int(np.clip(rng.exponential(5), 0, 40))
        years_since_promo = int(np.clip(rng.exponential(2), 0, years_at_company))
        total_working_years = int(np.clip(years_at_company + rng.exponential(4), 0, 40))
        stock_option = rng.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.2, 0.05])
        num_companies = rng.choice(range(0, 10))
        training_times = rng.choice(range(0, 7))
        perf_rating = rng.choice([3, 4], p=[0.85, 0.15])

        risk = (
            2.2 * (overtime == "Yes")
            + 1.8 * (monthly_income < 3000)
            + 1.1 * (distance > 15)
            + 1.3 * (job_sat <= 2)
            + 1.0 * (env_sat <= 2)
            + 1.2 * (work_life == 1)
            + 0.8 * (years_since_promo > 5)
            + 0.7 * (age < 30)
            + 0.8 * (num_companies >= 5)
            + 0.5 * (stock_option == 0)
            - 1.1 * (job_level >= 4)
            - 0.9 * (years_at_company > 10)
        )
        prob = 1 / (1 + np.exp(-(risk - 4.3)))
        attrition = "Yes" if rng.rand() < prob else "No"

        rows.append(dict(
            Age=age, Attrition=attrition,
            BusinessTravel=rng.choice(TRAVEL, p=TRAVEL_PROBS),
            DailyRate=int(rng.randint(102, 1500)),
            Department=dept, DistanceFromHome=distance,
            Education=rng.choice([1, 2, 3, 4, 5], p=[0.12, 0.19, 0.39, 0.27, 0.03]),
            EducationField=rng.choice(EDUCATION_FIELDS, p=[0.41, 0.15, 0.16, 0.10, 0.06, 0.12]),
            EmployeeCount=1, EmployeeNumber=i + 1,
            EnvironmentSatisfaction=env_sat,
            Gender=rng.choice(["Male", "Female"], p=[0.6, 0.4]),
            HourlyRate=int(rng.randint(30, 100)),
            JobInvolvement=rng.choice([1, 2, 3, 4], p=[0.09, 0.26, 0.59, 0.06]),
            JobLevel=job_level, JobRole=role, JobSatisfaction=job_sat,
            MaritalStatus=rng.choice(MARITAL, p=[0.32, 0.46, 0.22]),
            MonthlyIncome=monthly_income,
            MonthlyRate=int(rng.randint(2094, 27000)),
            NumCompaniesWorked=num_companies, Over18="Y", OverTime=overtime,
            PercentSalaryHike=int(rng.randint(11, 25)),
            PerformanceRating=perf_rating,
            RelationshipSatisfaction=rng.choice([1, 2, 3, 4]),
            StandardHours=80, StockOptionLevel=stock_option,
            TotalWorkingYears=total_working_years,
            TrainingTimesLastYear=training_times, WorkLifeBalance=work_life,
            YearsAtCompany=years_at_company,
            YearsInCurrentRole=int(np.clip(years_at_company * rng.uniform(0.2, 0.8), 0, years_at_company)),
            YearsSinceLastPromotion=years_since_promo,
            YearsWithCurrManager=int(np.clip(years_at_company * rng.uniform(0.2, 0.9), 0, years_at_company)),
        ))

    return pd.DataFrame(rows)
