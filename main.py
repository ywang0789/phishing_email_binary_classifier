import numpy as np
from flask import Flask, render_template, request

from model.Model import load_model, predict, sigmoid

app = Flask(__name__)

# Load the saved model and vectorizer
model_data, vectorizer = load_model("./model/model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence_level = None
    if request.method == "POST":
        email_text = request.form["email_text"]

        # Transform the input text using loaded vectorizer
        X_vec = vectorizer.transform([email_text])
        X = X_vec.toarray().T

        # Make prediction
        A = sigmoid(np.dot(model_data["w"].T, X) + model_data["b"])
        Y_pred = (A > 0.5).astype(int)
        prediction = "Phishing Email" if Y_pred[0][0] == 1 else "Safe Email"
        confidence_level = round(A[0][0] * 100, 2)

    return render_template(
        "index.html", prediction=prediction, confidence_level=confidence_level
    )


if __name__ == "__main__":
    app.run(debug=True)
