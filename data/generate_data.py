"""
Generates a synthetic dataset with the EXACT column schema of the real
IBM HR Analytics Employee Attrition dataset (Kaggle: pavansubhash/ibm-hr-analytics-attrition-dataset,
file: WA_Fn-UseC_-HR-Employee-Attrition.csv).

WHY THIS EXISTS: without internet/Kaggle access, this builds a realistic
stand-in with the same 35 columns and believable relationships (overtime,
low income, long commute, few promotions -> higher attrition risk) so
every downstream script (EDA, training, app) runs correctly.

>>> TO USE THE REAL DATA <<<
1. Download WA_Fn-UseC_-HR-Employee-Attrition.csv from Kaggle:
   https://www.kaggle.com/datasets/pavansubhash/ibm-hr-analytics-attrition-dataset
2. Drop it into data/WA_Fn-UseC_-HR-Employee-Attrition.csv
3. Nothing else changes — every script downstream reads that same filename
   and column names, so the real data drops in with zero code edits.

Run: python3 data/generate_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition_predictor.synthetic_data import generate_synthetic_dataset

if __name__ == "__main__":
    df = generate_synthetic_dataset()
    out_path = Path(__file__).resolve().parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows, {df.shape[1]} columns -> {out_path}")
    print(f"Attrition rate: {(df.Attrition == 'Yes').mean():.1%}")
