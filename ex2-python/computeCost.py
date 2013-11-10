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
    m = float(len(X))                                    # Number of training examples. The float is essential.
    print (-1.0/float(m)) * sum((y*log(sigmoid(X.dot(theta))) + (1- y)*log(1 - (sigmoid(X.dot(theta)))))) 
    return (-1.0/float(m)) * sum((y*log(sigmoid(X.dot(theta))) + (1- y)*log(1 - (sigmoid(X.dot(theta)))))) 
                         # notice the dot and * operations

def gradDes(theta,X,y):
    return (1/float(len(X)) * sum(X.T.dot((sigmoid(X.dot(theta)) - y))))


def predict(theta, X):
    p = zeros((len(X), 1))
    p[find(sigmoid(X.dot(theta)) >= 0.5)] = 1
    return p
