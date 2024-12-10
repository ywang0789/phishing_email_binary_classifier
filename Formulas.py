import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
