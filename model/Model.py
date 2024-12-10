import copy
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def sigmoid(x):

    s = 1 / (1 + np.exp(-x))

    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def loss(y_true, y_pred):
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def cost(y_true, y_pred, m):
    cost = 1 / m * np.sum(loss(y_true, y_pred))
    return cost


def cost_w_derivative(X, y_true, y_pred, m):
    dw = 1 / m * np.dot(X, (y_pred - y_true).T)
    return dw


def cost_b_derivative(y_true, y_pred, m):
    db = 1 / m * np.sum(y_pred - y_true)
    return db


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_iterations=2000,
    learning_rate=0.5,
    print_cost=False,
):

    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost
    )

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print(
            "train accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
            )
        )
        print(
            "test accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
            )
        )

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }

    return d


def save_model(model, vectorizer, file_path):
    data = {"model": model, "vectorizer": vectorizer}
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_model(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data["model"], data["vectorizer"]


class Preprocessor:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=1000, stop_words="english")

    def load_and_process_data(self, file_path):
        data = pd.read_csv(file_path)
        data["Email Type"] = data["Email Type"].map(
            {"Safe Email": 0, "Phishing Email": 1}
        )
        data["Email Text"].fillna(
            "hello", inplace=True
        )  # Replace NaN with 'hello' TODO: fix later
        X = data["Email Text"]
        Y = data["Email Type"].values
        return X, Y

    def fit_transform(self, X_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        return X_train_vec

    def transform(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return X_test_vec


if __name__ == "__main__":
    DATASET_TRAIN_FILE_PATH = "./data/dataset_train.csv"
    DATASET_TEST_FILE_PATH = "./data/dataset_test.csv"
    preprocessor = Preprocessor()
    X_train_text, Y_train = preprocessor.load_and_process_data(DATASET_TRAIN_FILE_PATH)
    X_test_text, Y_test = preprocessor.load_and_process_data(DATASET_TEST_FILE_PATH)

    X_train_vec = preprocessor.fit_transform(X_train_text)
    X_test_vec = preprocessor.transform(X_test_text)

    X_train = X_train_vec.toarray().T
    X_test = X_test_vec.toarray().T
    Y_train = Y_train.reshape(1, -1)
    Y_test = Y_test.reshape(1, -1)

    d = model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_iterations=1000,
        learning_rate=0.5,
        print_cost=True,
    )

    # save
    MODEL_FILE_PATH = "./model/model.pkl"
    save_model(d, preprocessor.vectorizer, MODEL_FILE_PATH)
    print(f"Model saved to {MODEL_FILE_PATH}")

    loaded_model, loaded_vectorizer = load_model(MODEL_FILE_PATH)
    print("Model and vectorizer loaded successfully")
