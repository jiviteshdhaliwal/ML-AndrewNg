##############################################
# Author:     Jivitesh Singh Dhaliwal
# Date:       07-11-2013
# Description:Gradient Descent Algorithm 
#             without regularization
##############################################

from pylab import * 
from sigmoid import *

def gradient(theta, X, y):
    m = float(len(y))
    print 'shape sigmoid(X.dot(theta)): ', shape(sigmoid(X.dot(theta)))
    grad = (1/m) * (X.T).dot((sigmoid(X.dot(theta)) - y))
    return grad
