import copy
import pickle

import numpy
import pandas
from sklearn.feature_extraction.text import CountVectorizer

from model.formulas import cost_b_derivative, cost_func, cost_w_derivative, sigmoid


def initialize(feature_dimension):
    w = numpy.zeros((feature_dimension, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y_true):
    num_dataset = X.shape[1]
    Y_pred = sigmoid(numpy.dot(w.T, X) + b)

    cost = cost_func(Y_true, Y_pred, num_dataset)

    dw = cost_w_derivative(X, Y_true, Y_pred, num_dataset)
    db = cost_b_derivative(Y_true, Y_pred, num_dataset)

    cost = numpy.squeeze(numpy.array(cost))  # scalar

    gradients = {"dw": dw, "db": db}

    return gradients, cost


def optimize(w, b, X, Y, num_iterations, learning_rate):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):

        gradients, cost = propagate(w, b, X, Y)

        dw = gradients["dw"]
        db = gradients["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    gradients = {"dw": dw, "db": db}

    return params, gradients, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = numpy.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    Y_pred = sigmoid(numpy.dot(w.T, X) + b)

    for i in range(Y_pred.shape[1]):

        if Y_pred[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_iterations,
    learning_rate,
):

    w, b = initialize(X_train.shape[0])

    params, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate
    )

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    train_acc = 100 - numpy.mean(numpy.abs(Y_prediction_train - Y_train)) * 100
    test_acc = 100 - numpy.mean(numpy.abs(Y_prediction_test - Y_test)) * 100
    print(f"train accuracy: {train_acc} %")
    print(f"test accuracy: {test_acc} %")

    data = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }

    return data


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
        data = pandas.read_csv(file_path)
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

    X_train_tokens = preprocessor.fit_transform(X_train_text)
    X_test_tokens = preprocessor.transform(X_test_text)

    X_train = X_train_tokens.toarray().T
    X_test = X_test_tokens.toarray().T
    Y_train = Y_train.reshape(1, -1)
    Y_test = Y_test.reshape(1, -1)

    d = model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_iterations=1000,
        learning_rate=0.5,
    )

    # save
    MODEL_FILE_PATH = "./model/model.pkl"
    save_model(d, preprocessor.vectorizer, MODEL_FILE_PATH)
    print(f"Model saved to {MODEL_FILE_PATH}")

    loaded_model, loaded_vectorizer = load_model(MODEL_FILE_PATH)
    print("Model and vectorizer loaded successfully")
