import numpy as np
from flask import Flask, render_template, request

from model.formulas import sigmoid
from model.Model import *

app = Flask(__name__)

model_data, vectorizer = load_model("./model/model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    A_value = None
    if request.method == "POST":
        email_text = request.form["email_text"]

        X_vec = vectorizer.transform([email_text])
        X = X_vec.toarray().T

        A = sigmoid(np.dot(model_data["w"].T, X) + model_data["b"])
        Y_pred = (A > 0.5).astype(int)
        prediction = "Phishing Email" if Y_pred[0][0] == 1 else "Safe Email"
        A_value = round(A[0][0], 2)

    return render_template("index.html", prediction=prediction, A_value=A_value)


if __name__ == "__main__":
    app.run(debug=True)
