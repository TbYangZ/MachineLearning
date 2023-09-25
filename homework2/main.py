import pandas as pd
import numpy as np
df = pd.read_csv("./homework2/data.csv")
X = [list(df['密度']), list(df['含糖率'])]

Y = list(df['好瓜'])

def DataTransform(X: list, Y: list):
    X = np.vstack((X, np.ones((1, len(X[0]))))).T
    Y = np.array(Y).T.reshape((X.shape[0], 1))
    return X, Y

X, Y = DataTransform(X, Y)

def sigmoid(beta, X):
    return np.sum(1 / (1 + np.exp(beta * X)), 1).reshape((X.shape[0], 1))

def Loglikelyhood(beta, X, Y):
    p0 = sigmoid(beta=beta, X=X)
    return np.sum(np.log(Y * (1 - p0) + (1 - Y) * p0))

def Gradient(beta, X, Y):
    p1 = 1 - sigmoid(beta=beta, X=X)
    return -(X.T @ (Y - p1)).T

learning_rate = 0.02
num_iteration = 100000
beta = np.random.uniform(0, 1, size=(X.shape[1],))

for i in range(num_iteration):
    gradient = Gradient(beta, X, Y)
    beta = beta - learning_rate * gradient
    pass

maxival = -Loglikelyhood(beta=beta, X=X, Y=Y)
print(beta)
print(maxival)