from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("loan_xgb_final.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "person_age": int(request.form["person_age"]),
        "person_income": float(request.form["person_income"]),
        "person_emp_exp": int(request.form["person_emp_exp"]),
        "loan_amnt": float(request.form["loan_amnt"]),
        "loan_int_rate": float(request.form["loan_int_rate"]),
        "loan_percent_income": float(request.form["loan_percent_income"]),
        "cb_person_cred_hist_length": int(request.form["cb_person_cred_hist_length"]),
        "credit_score": int(request.form["credit_score"]),
        "person_gender": request.form["person_gender"],
        "person_education": request.form["person_education"],
        "person_home_ownership": request.form["person_home_ownership"],
        "loan_intent": request.form["loan_intent"],
        "previous_loan_defaults_on_file": request.form["previous_loan_defaults_on_file"]
    }

    df = pd.DataFrame([input_data])

    df_encoded = pd.get_dummies(df)
    trained_cols = model.get_booster().feature_names
    df_encoded = df_encoded.reindex(columns=trained_cols, fill_value=0)

    prob = model.predict_proba(df_encoded)[0][1]

    if prob >= 0.4:
        result = f"⚠ High Risk (Default Probability: {prob:.2%})"
    else:
        result = f"✅ Low Risk (Default Probability: {prob:.2%})"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)






# from flask import Flask, render_template, request
# import joblib
# import numpy as np

# app = Flask(__name__)

# model = joblib.load("loan_xgb_final.pkl")

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         income = float(request.form['income'])
#         loan_amount = float(request.form['loan_amount'])
#         credit_score = float(request.form['credit_score'])

#         features = np.array([[income, loan_amount, credit_score]])
#         prediction = model.predict(features)[0]

#         if prediction == 1:
#             result = "Loan Approved ✅"
#         else:
#             result = "Loan Rejected ❌"

#         return render_template("index.html", prediction_text=result)

#     except:
#         return render_template("index.html", prediction_text="Error in input")

# if __name__ == "__main__":
#     app.run(debug=True)
