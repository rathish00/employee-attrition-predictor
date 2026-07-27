"""Employee Attrition Risk Predictor — Streamlit app.

Run: streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from attrition_predictor.bootstrap_pipeline import ensure_model_ready
from attrition_predictor.config import Config
from attrition_predictor.exceptions import AttritionPredictorError, DataValidationError, ModelNotFoundError
from attrition_predictor.model import AttritionModel
from attrition_predictor.predict import assess

st.set_page_config(page_title="Attrition Risk Predictor", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
.risk-card {
    padding: 1.25rem 1.5rem;
    border-radius: 0.6rem;
    margin-bottom: 0.75rem;
}
.risk-high   { background-color: #fde8e8; border-left: 6px solid #d62728; }
.risk-mod    { background-color: #fff8e6; border-left: 6px solid #e6a700; }
.risk-low    { background-color: #e8f7ee; border-left: 6px solid #2ca02c; }
.risk-title  { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.15rem; }
.risk-pct    { font-size: 2.2rem; font-weight: 800; }
.factor-list li { margin-bottom: 0.25rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Setting up the model (first run trains it — this takes a few seconds)...")
def load_model() -> AttritionModel:
    config = Config.load()
    return ensure_model_ready(config)


def render_risk_card(probability: float, tier: str) -> None:
    css_class = {"High": "risk-high", "Moderate": "risk-mod", "Low": "risk-low"}[tier]
    emoji = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}[tier]
    st.markdown(
        f"""<div class="risk-card {css_class}">
                <div class="risk-title">{emoji} {tier} Risk</div>
                <div class="risk-pct">{probability:.0%}</div>
                <div>probability of leaving within the model's prediction horizon</div>
            </div>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("📊 Employee Attrition & Retention Risk Predictor")
    st.caption("Type in an employee's profile to get a live quit-risk score, the factors driving it, and suggested retention actions.")

    try:
        model = load_model()
    except ModelNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("Run `python3 notebooks/01_eda.py` then `python3 notebooks/02_train_model.py` first, then reload this page.")
        st.stop()
    except Exception as e:  # last-resort guard — never show a raw traceback to an HR user
        st.error(f"⚠️ Unexpected error loading the model: {e}")
        st.stop()

    categories = model.categories

    # ---------------- Sidebar: batch scoring ----------------
    with st.sidebar:
        st.header("Batch scoring")
        st.caption("Upload a CSV with the same columns as the training data to score many employees at once.")
        uploaded = st.file_uploader("Employee CSV", type="csv")
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                probs = model.predict_proba(batch_df)
                batch_df = batch_df.copy()
                batch_df["Risk Score"] = probs
                batch_df = batch_df.sort_values("Risk Score", ascending=False)
                st.success(f"Scored {len(batch_df)} employees")
                st.dataframe(
                    batch_df[["Risk Score"] + model.required_columns].head(20),
                    use_container_width=True,
                )
                st.download_button(
                    "Download scored CSV",
                    batch_df.to_csv(index=False).encode(),
                    "scored_employees.csv",
                    "text/csv",
                )
            except DataValidationError as e:
                st.error(f"⚠️ {e}")
            except pd.errors.ParserError:
                st.error("⚠️ Couldn't parse that file as CSV. Check the format and try again.")
            except Exception as e:
                st.error(f"⚠️ Couldn't score that file: {e}")

    # ---------------- Single employee form ----------------
    st.subheader("Score a single employee")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 32)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 4500, step=100)
        distance = st.slider("Distance From Home (miles)", 1, 30, 8)
        overtime = st.selectbox("OverTime", categories.get("OverTime", ["Yes", "No"]))
        num_companies = st.slider("Num Companies Worked", 0, 9, 2)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)

    with col2:
        job_sat = st.select_slider("Job Satisfaction", [1, 2, 3, 4], value=3)
        env_sat = st.select_slider("Environment Satisfaction", [1, 2, 3, 4], value=3)
        work_life = st.select_slider("Work-Life Balance", [1, 2, 3, 4], value=3)
        job_level = st.select_slider("Job Level", [1, 2, 3, 4, 5], value=2)
        stock_option = st.select_slider("Stock Option Level", [0, 1, 2, 3], value=0)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)

    with col3:
        department = st.selectbox("Department", categories.get("Department", ["Sales"]))
        job_role = st.selectbox("Job Role", categories.get("JobRole", ["Sales Executive"]))
        marital = st.selectbox("Marital Status", categories.get("MaritalStatus", ["Single"]))
        business_travel = st.selectbox("Business Travel", categories.get("BusinessTravel", ["Travel_Rarely"]))
        gender = st.selectbox("Gender", categories.get("Gender", ["Male", "Female"]))
        education_field = st.selectbox("Education Field", categories.get("EducationField", ["Life Sciences"]))

    years_at_company = st.slider("Years At Company", 0, 40, max(1, total_working_years // 2))

    defaults = dict(
        DailyRate=800, HourlyRate=65, MonthlyRate=14000, PercentSalaryHike=15,
        PerformanceRating=3, RelationshipSatisfaction=3, JobInvolvement=3,
        Education=3, TrainingTimesLastYear=2,
        YearsInCurrentRole=min(years_at_company, 4),
        YearsWithCurrManager=min(years_at_company, 4),
    )
    employee = {
        "Age": age, "MonthlyIncome": monthly_income, "DistanceFromHome": distance,
        "NumCompaniesWorked": num_companies, "TotalWorkingYears": total_working_years,
        "JobSatisfaction": job_sat, "EnvironmentSatisfaction": env_sat,
        "WorkLifeBalance": work_life, "JobLevel": job_level, "StockOptionLevel": stock_option,
        "YearsSinceLastPromotion": years_since_promo, "YearsAtCompany": years_at_company,
        "OverTime": overtime, "Department": department, "JobRole": job_role,
        "MaritalStatus": marital, "BusinessTravel": business_travel, "Gender": gender,
        "EducationField": education_field, **defaults,
    }

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        try:
            input_df = pd.DataFrame([employee])
            probability = float(model.predict_proba(input_df)[0])
            result = assess(employee, probability)
        except AttritionPredictorError as e:
            st.error(f"⚠️ Couldn't score this employee: {e}")
            st.stop()

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            render_risk_card(result.probability, result.tier)
        with c2:
            st.markdown("**What's driving this score**")
            if result.risk_factors:
                st.markdown(
                    "<ul class='factor-list'>" + "".join(f"<li>{f}</li>" for f in result.risk_factors) + "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("- No strong risk factors detected")
            if result.protective_factors:
                st.markdown("**Protective factors**")
                st.markdown(
                    "<ul class='factor-list'>" + "".join(f"<li>{f}</li>" for f in result.protective_factors) + "</ul>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("**Suggested retention actions for HR**")
        for a in result.recommended_actions:
            st.markdown(f"- {a}")

    st.divider()
    st.caption(
        "Model trained on the IBM HR Analytics Attrition dataset schema. "
        "Predictions are decision support, not a substitute for HR judgment."
    )


if __name__ == "__main__":
    main()