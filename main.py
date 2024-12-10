import numpy as np
from flask import Flask, render_template, request

from model.Model import load_model, predict

app = Flask(__name__)

# Load the saved model and vectorizer
model_data, vectorizer = load_model("./model/model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]

        # Transform the input text using loaded vectorizer
        X_vec = vectorizer.transform([email_text])
        X = X_vec.toarray().T

        # Make prediction
        Y_pred = predict(model_data["w"], model_data["b"], X)
        prediction = "Phishing Email" if Y_pred[0][0] == 1 else "Safe Email"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
