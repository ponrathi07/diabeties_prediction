from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

data = pd.read_csv("diabetes.csv")

X = data[["Glucose", "BloodPressure", "BMI", "Age"]]
y = data["Outcome"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    glucose = float(request.form["Glucose"])
    bp = float(request.form["BloodPressure"])
    bmi = float(request.form["BMI"])
    age = float(request.form["Age"])

    prediction = model.predict([[glucose, bp, bmi, age]])

    result = "The patient has diabetes" if prediction[0] == 1 else "The patient has no diabetes"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)