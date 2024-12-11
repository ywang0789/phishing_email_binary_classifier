import numpy


def sigmoid(x):
    s = 1 / (1 + numpy.exp(-x))
    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def loss(y_true, y_pred):
    loss = -(y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred))
    return loss


def cost_func(y_true, y_pred, num_dataset):
    cost = 1 / num_dataset * numpy.sum(loss(y_true, y_pred))
    return cost


def cost_w_derivative(X, y_true, y_pred, num_dataset):
    dw = 1 / num_dataset * numpy.dot(X, (y_pred - y_true).T)
    return dw


def cost_b_derivative(y_true, y_pred, num_dataset):
    db = 1 / num_dataset * numpy.sum(y_pred - y_true)
    return db
