from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("loan_xgb_final.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = float(request.form['credit_score'])

        features = np.array([[income, loan_amount, credit_score]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)
