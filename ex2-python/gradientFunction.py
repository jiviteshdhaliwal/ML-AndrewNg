################################################
# Author:   Jivitesh Singh Dhaliwal
# Date:     09-11-2013
# Program:  Compute Cost for Logistic Regression
################################################

from computeCost import *

def gradientFunction(X, y, theta, alpha, numIter):
    '''Return the gradient for given hypothesis function
       @Params: X, y, theta'''
    costHistory = [0]
    m = len(X)                                        # Number of training examples
    for i in range(numIter):
    return (1/m) * sum((X.T.dot(X.dot(theta) - y))) 
