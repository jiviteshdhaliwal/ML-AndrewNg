##############################################
# Author:     Jivitesh Singh Dhaliwal
# Date:       07-11-2013
# Description:Gradient Descent Algorithm 
#             without regularization
##############################################

from computeCost import *

def gradientDescent(X, y, theta, alpha, numIter):
    ''' Perform Linear Regression using Gradient Descent'''
    costHistory = [0]
    m = len(X)                                        # Number of training examples
    for i in range(numIter):
        temp = theta[:]                               # Copy theta vector to a temporary vector for update
        temp = (X.T.dot((X.dot(theta) - y)))          # Performing the entire operation of grad des using 
                                                      #     matrix multiplication
        theta = theta - ((alpha/ m) * temp)

        costHistory.append(computeCost(X, y, theta, m))
   
        if costHistory[-1] > costHistory[-2]:
            print 'Cost is not converging. Please check alpha. Exiting'
            break
    return theta 
