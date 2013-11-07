##############################################
# Author:     Jivitesh Singh Dhaliwal
# Date:       07-11-2013
# Description:Gradient Descent Algorithm 
#             without regularization
##############################################

from pylab import *

def normalEquation(X, y, theta):
    ''' Perform Linear Regression using Normal Equation'''
    return linalg.pinv((X.T.dot(X))).dot(X.T.dot(y))
