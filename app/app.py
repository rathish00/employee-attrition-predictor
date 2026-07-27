"""Employee Attrition Risk Predictor — Streamlit app.

Run: streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from attrition_predictor.auth import resolve_credentials, verify_credentials
from attrition_predictor.bootstrap_pipeline import ensure_model_ready
from attrition_predictor.config import Config
from attrition_predictor.exceptions import AttritionPredictorError, DataValidationError
from attrition_predictor.model import AttritionModel
from attrition_predictor.predict import assess

st.set_page_config(page_title="Attrition Risk Predictor", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}

.block-container { padding-top: 2rem; }

.app-header {
    display: flex; align-items: center; gap: 0.75rem;
    padding-bottom: 0.25rem; margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}
.app-header h1 { font-size: 1.6rem; margin: 0; }

.section-card {
    background: rgba(127,127,127,0.06);
    border: 1px solid rgba(127,127,127,0.15);
    border-radius: 0.75rem;
    padding: 1.1rem 1.3rem 0.4rem 1.3rem;
    margin-bottom: 1rem;
}
.section-title {
    font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem;
}

.risk-card {
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 0.75rem;
}
.risk-high   { background-color: #fde8e8; border-left: 6px solid #d62728; }
.risk-mod    { background-color: #fff8e6; border-left: 6px solid #e6a700; }
.risk-low    { background-color: #e8f7ee; border-left: 6px solid #2ca02c; }
.risk-title  { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.1rem; }
.risk-pct    { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
.risk-gauge-track {
    background: rgba(127,127,127,0.2); border-radius: 6px;
    height: 10px; margin-top: 0.6rem; overflow: hidden;
}
.risk-gauge-fill { height: 10px; border-radius: 6px; }

.factor-list { margin: 0.2rem 0 0 0; padding-left: 1.2rem; }
.factor-list li { margin-bottom: 0.25rem; }

.login-wrap {
    max-width: 380px; margin: 3rem auto 0 auto;
    padding: 2rem; border-radius: 1rem;
    background: rgba(127,127,127,0.06);
    border: 1px solid rgba(127,127,127,0.15);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RISK_COLORS = {"High": "#d62728", "Moderate": "#e6a700", "Low": "#2ca02c"}
RISK_CSS_CLASS = {"High": "risk-high", "Moderate": "risk-mod", "Low": "risk-low"}
RISK_EMOJI = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}


# --------------------------------------------------------------------------
# Model loading (cached — no Streamlit UI elements may be created inside a
# @st.cache_resource function, since a replayed cache hit can't recreate
# them; ensure_model_ready() only logs, it never touches st.*)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up the model (first run trains it — a few seconds)...")
def load_model() -> AttritionModel:
    config = Config.load()
    return ensure_model_ready(config)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def login_screen() -> None:
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown("### 🔐 HR Analytics Portal")
    st.caption("Sign in to access the Employee Attrition & Retention Risk Predictor")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", type="primary", width="stretch")
    if submitted:
        try:
            secrets = st.secrets.get("credentials", None)
        except Exception:
            secrets = None
        valid_username, valid_hash = resolve_credentials(secrets)
        if verify_credentials(username, password, valid_username, valid_hash):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("⚠️ Invalid username or password")
    st.info("Demo credentials: **admin** / **attrition2026** — set your own via Streamlit secrets before sharing this app publicly (see README).")
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Risk rendering
# --------------------------------------------------------------------------
def render_risk_card(probability: float, tier: str) -> None:
    css_class = RISK_CSS_CLASS[tier]
    color = RISK_COLORS[tier]
    emoji = RISK_EMOJI[tier]
    pct = round(probability * 100)
    st.markdown(
        f"""<div class="risk-card {css_class}">
                <div class="risk-title">{emoji} {tier} Risk</div>
                <div class="risk-pct">{pct}%</div>
                <div>probability of leaving</div>
                <div class="risk-gauge-track">
                    <div class="risk-gauge-fill" style="width:{pct}%; background:{color};"></div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Page: single employee
# --------------------------------------------------------------------------
def page_single_employee(model: AttritionModel) -> None:
    categories = model.categories
    st.subheader("Score a single employee")

    st.markdown('<div class="section-card"><div class="section-title">👤 Personal & Role</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 60, 32)
        gender = st.selectbox("Gender", categories.get("Gender", ["Male", "Female"]))
    with c2:
        department = st.selectbox("Department", categories.get("Department", ["Sales"]))
        job_role = st.selectbox("Job Role", categories.get("JobRole", ["Sales Executive"]))
    with c3:
        marital = st.selectbox("Marital Status", categories.get("MaritalStatus", ["Single"]))
        education_field = st.selectbox("Education Field", categories.get("EducationField", ["Life Sciences"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">💰 Compensation & Work</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 4500, step=100)
        overtime = st.selectbox("OverTime", categories.get("OverTime", ["Yes", "No"]))
    with c2:
        job_level = st.select_slider("Job Level", [1, 2, 3, 4, 5], value=2)
        stock_option = st.select_slider("Stock Option Level", [0, 1, 2, 3], value=0)
    with c3:
        distance = st.slider("Distance From Home (miles)", 1, 30, 8)
        business_travel = st.selectbox("Business Travel", categories.get("BusinessTravel", ["Travel_Rarely"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">😊 Satisfaction & Tenure</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        job_sat = st.select_slider("Job Satisfaction", [1, 2, 3, 4], value=3)
        env_sat = st.select_slider("Environment Satisfaction", [1, 2, 3, 4], value=3)
    with c2:
        work_life = st.select_slider("Work-Life Balance", [1, 2, 3, 4], value=3)
        num_companies = st.slider("Num Companies Worked", 0, 9, 2)
    with c3:
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        years_at_company = st.slider("Years At Company", 0, 40, 4)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
    st.markdown("</div>", unsafe_allow_html=True)

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

    if st.button("🔍 Predict Risk", type="primary", width="stretch"):
        try:
            input_df = pd.DataFrame([employee])
            probability = float(model.predict_proba(input_df)[0])
            result = assess(employee, probability)
        except AttritionPredictorError as e:
            st.error(f"⚠️ Couldn't score this employee: {e}")
            return

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
                st.markdown("No strong risk factors detected.")
            if result.protective_factors:
                st.markdown("**Protective factors**")
                st.markdown(
                    "<ul class='factor-list'>" + "".join(f"<li>{f}</li>" for f in result.protective_factors) + "</ul>",
                    unsafe_allow_html=True,
                )

        st.markdown("**Suggested retention actions for HR**")
        for a in result.recommended_actions:
            st.markdown(f"- {a}")


# --------------------------------------------------------------------------
# Page: batch scoring
# --------------------------------------------------------------------------
def page_batch_scoring(model: AttritionModel) -> None:
    st.subheader("Score a team from CSV")
    st.caption("Upload a CSV with the same columns as the training data to score many employees at once.")
    uploaded = st.file_uploader("Employee CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV to get started.")
        return

    try:
        batch_df = pd.read_csv(uploaded)
        probs = model.predict_proba(batch_df)
    except DataValidationError as e:
        st.error(f"⚠️ {e}")
        return
    except pd.errors.ParserError:
        st.error("⚠️ Couldn't parse that file as CSV. Check the format and try again.")
        return
    except Exception as e:
        st.error(f"⚠️ Couldn't score that file: {e}")
        return

    batch_df = batch_df.copy()
    batch_df["Risk Score"] = probs
    batch_df["Risk Tier"] = pd.cut(
        probs, bins=[-0.01, 0.3, 0.6, 1.01], labels=["Low", "Moderate", "High"]
    )
    batch_df = batch_df.sort_values("Risk Score", ascending=False)

    counts = batch_df["Risk Tier"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees scored", len(batch_df))
    c2.metric("🔴 High risk", int(counts.get("High", 0)))
    c3.metric("🟡 Moderate risk", int(counts.get("Moderate", 0)))
    c4.metric("🟢 Low risk", int(counts.get("Low", 0)))

    st.bar_chart(counts.reindex(["High", "Moderate", "Low"]).fillna(0))

    st.markdown("**Highest-risk employees**")
    st.dataframe(
        batch_df[["Risk Score", "Risk Tier"] + model.required_columns].head(25),
        width="stretch",
    )
    st.download_button(
        "Download full scored CSV",
        batch_df.to_csv(index=False).encode(),
        "scored_employees.csv",
        "text/csv",
    )


# --------------------------------------------------------------------------
# Page: model insights
# --------------------------------------------------------------------------
def page_model_insights(model: AttritionModel) -> None:
    st.subheader("What the model learned")
    importances = model.global_feature_importance(top_n=12)
    if importances.empty:
        st.info("This model type doesn't expose feature importances.")
        return

    clean_index = [
        name.replace("num__", "").replace("cat__", "").replace("_", " ")
        for name in importances.index
    ]
    chart_df = pd.Series(importances.values, index=clean_index)
    st.caption("Top factors the model weighs most heavily when predicting attrition risk.")
    st.bar_chart(chart_df)
    st.caption(
        "Note: this reflects patterns in the current training data (synthetic, "
        "unless you've swapped in the real Kaggle dataset — see README)."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_screen()
        return

    try:
        model = load_model()
    except Exception as e:  # last-resort guard — never show a raw traceback to an HR user
        st.error(f"⚠️ Unexpected error loading the model: {e}")
        st.stop()

    with st.sidebar:
        st.markdown("### 📊 Attrition Predictor")
        st.caption("HR Analytics Portal")
        page = st.radio(
            "Navigate",
            ["Single Employee", "Batch Scoring", "Model Insights"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("🚪 Logout", width="stretch"):
            st.session_state.authenticated = False
            st.rerun()

    st.markdown('<div class="app-header"><h1>📊 Employee Attrition & Retention Risk Predictor</h1></div>', unsafe_allow_html=True)

    if page == "Single Employee":
        page_single_employee(model)
    elif page == "Batch Scoring":
        page_batch_scoring(model)
    else:
        page_model_insights(model)

    st.divider()
    st.caption(
        "Model trained on the IBM HR Analytics Attrition dataset schema. "
        "Predictions are decision support, not a substitute for HR judgment."
    )


if __name__ == "__main__":
    main()