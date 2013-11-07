#########################################
# Author:   Jivitesh Singh Dhaliwal
# Date:     07-11-2013
# Problem:  Compute Cost from Grad Des
#########################################

from pylab import *

def computeCost(X, y, theta, m):
    return (1/(2*m))* sum((X.dot(theta) - y)**2)         # notice the dot and * operations
