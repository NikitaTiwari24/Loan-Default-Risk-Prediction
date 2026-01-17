import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_xgb_final.pkl")

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🏦 Loan Default Risk Prediction")

st.write("Enter applicant details to predict loan default risk.")

# -------- User Inputs --------
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income", 10000, 1000000, 500000)
emp_exp = st.number_input("Employment Experience (years)", 0, 40, 5)
loan_amnt = st.number_input("Loan Amount", 1000, 1000000, 200000)
int_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 12.5)
loan_pct_income = st.slider("Loan % of Income", 0.0, 1.0, 0.3)
cred_hist = st.number_input("Credit History Length (years)", 0, 40, 8)
credit_score = st.number_input("Credit Score", 300, 850, 720)

gender = st.selectbox("Gender", ["male", "female"])
education = st.selectbox("Education", ["High School", "Graduate", "Postgraduate"])
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
prev_default = st.selectbox("Previous Loan Default", ["Yes", "No"])

# -------- Predict --------
if st.button("Predict Loan Status"):
    input_data = {
        "person_age": age,
        "person_income": income,
        "person_emp_exp": emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": loan_pct_income,
        "cb_person_cred_hist_length": cred_hist,
        "credit_score": credit_score,
        "person_gender": gender,
        "person_education": education,
        "person_home_ownership": home,
        "loan_intent": intent,
        "previous_loan_defaults_on_file": prev_default
    }

    df = pd.DataFrame([input_data])

    # One-hot encoding
    df_encoded = pd.get_dummies(df)

    # Align columns with training data
    trained_cols = model.get_booster().feature_names
    df_encoded = df_encoded.reindex(columns=trained_cols, fill_value=0)

    prob = model.predict_proba(df_encoded)[0][1]
    threshold = 0.4

    status = "DEFAULT" if prob >= threshold else "NON-DEFAULT"

    st.subheader("Prediction Result")
    st.write(f"**Default Probability:** {prob:.2%}")

    if status == "DEFAULT":
        st.error("⚠️ High Risk: Loan Default Likely")
    else:
        st.success("✅ Low Risk: Loan Can Be Approved")
