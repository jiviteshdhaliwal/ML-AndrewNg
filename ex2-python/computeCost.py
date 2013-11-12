#########################################
# Author:   Jivitesh Singh Dhaliwal
# Date:     07-11-2013
# Problem:  Compute Cost from Grad Des
#########################################

from pylab import *
from sigmoid import * 

def cost(theta, X, y):
    '''Return the gradient for given hypothesis function
       @Params: X, y, theta'''
    m = float(len(X))                            # Number of training examples. The float is essential.
    cost = (-1/m) * sum ((y * log(sigmoid(X.dot(theta)))) + ((1 - y) * log(1 - sigmoid(X.dot(theta)))))
    grad = (1/m) * (X.T).dot((sigmoid(X.dot(theta)) - y))
    return cost, grad

def predict(theta, X):
    p = sigmoid(X.dot(theta)) >= 0.5
    return p

