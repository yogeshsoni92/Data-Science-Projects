import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Logistic Regression

def sigmoid(z):
    return 1.0/(1.0 + np.exp2(-z))

def softmax(a):
    aexpa = np.exp(a)
    return aexpa/aexpa.sum(axis = 1, keepdims = True)

def compute_prediction(x, weights):
    z = np.dot(x,weights)
    y_hat = softmax(z)
    return y_hat

def weight_updates(X, y, learning_rate, weights):
    prediction = compute_prediction(X, weights)
    weight_delta = np.dot(X.T, y - prediction)
    m = X.shape[0]
    weights += learning_rate/m * weight_delta
    return weights

def weight_updates_sgd(X, y, learning_rate, weights):
    for X_each, y_each in zip(X, y):
        prediction = compute_prediction(X_each, weights)
        weight_delta = np.dot(X_each.T, y_each - prediction)
        weights += learning_rate * weight_delta
    return weights

def compute_cost(X,y,weights):
    prediction = compute_prediction(X, weights)
    cost = np.mean(-y * np.log(prediction)-(1-y) * np.log(1 - prediction))
    return cost

def train_logistic_regression(x_train, y_train, max_iter, learning_rate, fit_intercept = False):
    if fit_intercept:
        intercept = np.ones((x_train.shape[0], 1))
        x_train = np.hstack((intercept, x_train))
    weights = np.zeros((x_train.shape[1], y_train.shape[1]))
    for iteration in range(max_iter):
        weights = weight_updates(x_train, y_train, learning_rate, weights)
        if iteration % 100 == 0:
            print(compute_cost(x_train, y_train, weights))
    return weights

def predict_prob(x, w):
    if x.shape[1] == w.shape[0] - 1:
        intercept = np.ones((x.shape[0],1))
        x = np.hstack((intercept, x))
    return compute_prediction(x,w)

def predict(x, w):
    if x.shape[1] == w.shape[0] - 1:
        intercept = np.ones((x.shape[0],1))
        x = np.hstack((intercept, x))
    return np.argmax(compute_prediction(x,w), axis=1)


# Regularization Method for logistic regression:
def weight_updates_regularization(X, y, learning_rate, weights, lambda_):
    prediction = compute_prediction(X, weights)
    weight_delta = np.dot(X.T, y - prediction)
    m = X.shape[0]
    m = float(m)
    weights += learning_rate * (1/m * weight_delta + (lambda_/m)*np.sum(weights**2))
    return weights

def compute_cost_regularization(X,y,w,lambda_):
    m = y.size
    temp = w
    prediction = compute_prediction(X, w)
    #J = np.mean(-y.dot(np.log(prediction)) - (1 - y).dot(np.log(1 - prediction))) #+ (lambda_ /m)*np.sum(temp**2)
    J = np.mean(-y * np.log(prediction)-(1-y) * np.log(1 - prediction)) + (lambda_/m)*np.sum(temp**2)
    #grad = (1/m)*(prediction-y).dot(X)
    #grad = grad + (lambda_/m)*temp
    return J

def train_logistic_regression_regularization(x_train, y_train, max_iter, learning_rate, lambda_, fit_intercept = False):
    if fit_intercept:
        intercept = np.ones((x_train.shape[0], 1))
        x_train = np.hstack((intercept, x_train))
    weights = np.zeros((x_train.shape[1], y_train.shape[1]))
    for iteration in range(max_iter):
        weights = weight_updates_regularization(x_train, y_train, learning_rate, weights,lambda_= lambda_)
        if iteration % 100 == 0:
            print(compute_cost_regularization(x_train, y_train, weights, lambda_= lambda_))
    return weights

#Function to convert Y data of digit Recognizer to one hot encoder 10 columns for each digit 
def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    K = y.max()+1
    ind = np.zeros((N,K))
    for n in range(N):
        k = y.iloc[n]
        ind[n, k] = 1
    return ind

#Function to plot 1 digit of digit

def print_mnist_data(data, label):
    image = data.reshape(28, 28)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
    plt.show()

