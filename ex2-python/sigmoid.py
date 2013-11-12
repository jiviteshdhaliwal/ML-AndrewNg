####################################
# Author:   Jivitesh Singh Dhaliwal
# Date:     09-11-2013
# Program:  Linear Regression Sigmoid
#           Function
####################################

from pylab import *

def sigmoid(z):
    return (1/(1+exp(-1*z)))
