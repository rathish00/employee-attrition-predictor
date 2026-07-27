Employee Attrition & Retention Risk Predictor

An end-to-end machine learning project that predicts whether an employee is likely to leave a company and helps HR teams understand the possible reasons behind that risk.

The project covers the complete ML workflow — data processing, analysis, model training, evaluation, prediction, testing, and deployment using Streamlit.

The goal is to help organizations identify employees who may be at risk of leaving and take better retention decisions using data-driven insights.

Project Overview

Employee attrition can impact productivity, hiring costs, and business growth. This project uses employee information such as overtime, income, satisfaction levels, job role, and work experience to predict attrition risk.

The system provides:

Employee attrition risk prediction
Risk score generation
Important factors influencing the prediction
Suggested retention actions
Interactive Streamlit dashboard for HR users

Project Architecture
employee-attrition-predictor/

├── config.yaml                 # Project configuration
├── pyproject.toml              # Package configuration
├── bootstrap.py                # Makes source modules accessible

├── src/attrition_predictor/
│   ├── config.py               # Configuration handling
│   ├── data.py                 # Data loading and validation
│   ├── model.py                # Training and prediction pipeline
│   ├── predict.py              # Risk analysis and recommendations
│   ├── exceptions.py            # Custom error handling
│   └── logging_config.py        # Logging setup

├── notebooks/
│   ├── 01_eda.py               # Exploratory data analysis
│   └── 02_train_model.py       # Model training and evaluation

├── app/
│   └── app.py                  # Streamlit application

├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_predict.py
│   └── smoke_test_app.py

├── models/                     # Saved ML models
├── outputs/                    # Generated analysis results
├── data/                       # Dataset files

├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml

Key Features
Data Pipeline
Validates incoming employee data
Handles preprocessing automatically
Checks missing or incorrect values before training
Machine Learning

Compared multiple models:

Logistic Regression
Random Forest
Gradient Boosting

The best-performing model is selected automatically.

Prediction System

The application provides:

Attrition probability score
Employee risk category
Factors contributing to risk
Possible retention suggestions
Production Practices

This project goes beyond a simple notebook:

Modular ML pipeline
Configuration-based design
Custom error handling
Automated tests
GitHub Actions CI
Docker support
Streamlit deployment
Machine Learning Approach

The workflow:

Load and validate employee data
Clean and preprocess the data
Perform exploratory analysis
Train multiple machine learning models
Compare performance using ROC-AUC
Select the best model
Deploy the prediction system using Streamlit
Dataset Information

The project currently uses a synthetic dataset created with the same structure as the IBM HR Analytics Attrition dataset.

The synthetic data was created because the original dataset was not available during development.

The generated data follows realistic patterns where factors like:

Overtime
Monthly income
Job satisfaction
Work environment
Distance from home

influence employee attrition.

Note: The model performance shown below is based on synthetic data. For real-world usage, the model should be retrained using actual HR data.

Model Performance

Best Model:

Gradient Boosting Classifier

Performance:

ROC-AUC Score: 0.85
Accuracy: 83%

Model comparison:

Gradient Boosting     0.854
Random Forest         0.833
Logistic Regression   0.827

Important factors identified:

Overtime status
Monthly income
Job satisfaction
Number of companies worked
Environment satisfaction
Distance from home