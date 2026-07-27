# Employee Attrition & Retention Risk Predictor

An end-to-end machine learning project that predicts whether an employee is likely to leave a company, explains why, and gives HR teams a live tool to score retention risk — behind a login gate, with tests, CI, and Docker throughout.

## What it does

- Predicts an individual employee's probability of leaving
- Sorts that prediction into a Low / Moderate / High risk tier, with a visual gauge
- Explains the specific factors driving the score (overtime, income, satisfaction, tenure, and more)
- Suggests concrete retention actions for HR to take
- Scores an entire team at once via CSV upload, ranked by risk, with summary KPIs
- Shows which factors the model weighs most heavily overall (Model Insights page)
- Sits behind a simple login screen

## Project structure

```
employee-attrition-predictor/
  config.yaml                 project configuration (paths, hyperparameters)
  pyproject.toml              package metadata
  requirements.txt            dependencies
  runtime.txt                 pins Python version for deployment

  src/attrition_predictor/
    config.py                 typed config loader
    data.py                   data loading and schema validation
    model.py                  training, model selection, persistence
    predict.py                risk factors and recommended actions
    auth.py                   login credential checking
    bootstrap_pipeline.py     self-provisions data/model on first run
    synthetic_data.py         synthetic dataset generator
    exceptions.py             custom error types
    logging_config.py         shared logging setup

  notebooks/
    01_eda.py                 exploratory data analysis
    02_train_model.py         model training and evaluation

  app/
    app.py                    Streamlit application (login + 3 pages)

  tests/
    test_data.py
    test_model.py
    test_predict.py
    test_bootstrap.py
    test_auth.py
    smoke_test_app.py

  Dockerfile
  .github/workflows/ci.yml
```

## How the pipeline works

1. Load and validate the raw employee data against the expected schema
2. Clean it and add a numeric attrition flag
3. Run exploratory analysis, generating six charts (overtime, income, distance, department, satisfaction, correlations)
4. Train Logistic Regression, Random Forest, and Gradient Boosting classifiers (XGBoost too, if installed)
5. Compare them by ROC-AUC and automatically select the best one
6. Save the trained pipeline, then serve it through the Streamlit app

## Login

The app is behind a simple username/password gate.

Demo credentials (documented here on purpose — change these before sharing the app):
```
username: admin
password: attrition2026
```

To set your own, add this to Streamlit Cloud's app settings under **Secrets** (or a local `.streamlit/secrets.toml` for local runs):
```toml
[credentials]
username = "your-username"
password_hash = "sha256-hex-digest-of-your-password"
```
Generate the hash locally:
```bash
python3 -c "from src.attrition_predictor.auth import hash_password; print(hash_password('your-password'))"
```

This is single-shared-credential auth, appropriate for gating a demo/portfolio app — not a substitute for real per-user authentication on anything handling actual employee data.

## Dataset

This currently runs on a synthetic dataset built with the same 35-column schema as the real IBM HR Analytics Attrition dataset from Kaggle, since the real file wasn't accessible during development. The synthetic data is calibrated so the same real-world drivers show up — overtime, low income, long commute, and low satisfaction all raise attrition risk, matching published HR research.

To switch to the real dataset:

1. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from [Kaggle: IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhash/ibm-hr-analytics-attrition-dataset)
2. Replace `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` with it, same filename
3. Delete `models/*.pkl` so the app retrains on the new data, or re-run the pipeline manually

The model performance numbers below are from the synthetic data. Retrain on real data before using these numbers anywhere they need to be accurate.

## Model performance

Best model: **Gradient Boosting Classifier**

| Model | ROC-AUC |
|---|---|
| Gradient Boosting | 0.854 |
| Random Forest | 0.833 |
| Logistic Regression | 0.827 |

Top factors driving predicted attrition risk: overtime status, monthly income, job satisfaction, number of companies worked, environment satisfaction, distance from home.

## Setup and run

```bash
pip install -r requirements.txt

python3 data/generate_data.py
python3 notebooks/01_eda.py
python3 notebooks/02_train_model.py
python3 -m unittest discover -s tests

streamlit run app/app.py
```

## Deployment

The app is self-sufficient on platforms like Streamlit Community Cloud that only run `app/app.py` with no separate build step. On first load, it checks whether the data and trained model exist; if not, it generates the data and trains the model automatically before serving the first prediction. After that first run, it's cached for the life of the container via `st.cache_resource`.

`runtime.txt` pins Python to 3.11, since some deployment platforms default to the newest available version, and very new Python releases can lack prebuilt wheels for some packages. `requirements.txt` doesn't hard-pin `xgboost` for the same reason — the training code already falls back to Gradient Boosting/Random Forest if it's not importable.

Docker:
```bash
docker build -t attrition-predictor .
docker run -p 8501:8501 attrition-predictor
```

## Engineering practices

- Config-driven: paths and hyperparameters live in `config.yaml`, not hardcoded
- Schema validation at the data boundary, with clear custom exceptions
- One model wrapper (`AttritionModel`) used consistently by the app, scripts, and tests
- Auth logic separated from the UI layer and unit tested independently
- 38 automated unit tests plus an app-logic smoke test covering the login flow, all three pages, and both batch-upload outcomes
- GitHub Actions CI running the full pipeline and test suite on every push
- Dockerized for deployment anywhere

## Resume framing

- Built an end-to-end employee attrition prediction system (Python, scikit-learn, Streamlit) with a tested, config-driven ML pipeline, achieving 0.85 ROC-AUC and deployed as an interactive, authenticated tool for HR users to score retention risk in real time
- Designed a modular ML architecture with schema validation, custom exceptions, and a single model-serving contract, backed by 38 unit tests and a CI pipeline
- Identified the top drivers of voluntary turnover through feature importance analysis and translated them into a rules-based recommendation layer for HR action
