#########################################################
# Author:  Jivitesh Singh Dhaliwal
# Date:    09-11-2013
# Program: Main (user interface) for Logistic Regression
#########################################################

# Some parts inspired from Nonnormalizable's NgMachineLearningPython repo. 
# Mainly the use of the pandas library, and scipy's optimize fn. Thanks!

###################
# Import Libraries
###################

from pylab import *                             # Pylab: Numpy+Scipy+matplotlib
from scipy import optimize
import pandas as pd

###################
# Import Functions
###################

from computeCost import *

def __main__():
    '''Perform Logistic Regression using gradient descent or normal equations on data provided by user'''
    ###############
    # Obtain Data
    ###############

    data = pd.read_csv('ex2data1.txt', header = None)
    X = data[[0,1]]
    y = data[2]

    m, n = np.shape(X)

    X['ones'] = ones(m)
    X_array = array(X[['ones', 0, 1]])
    y_array = array(y)

    initial_theta = zeros(n + 1)

    try:
        result_Newton_CG= optimize.minimize(lambda t: cost(t, X_array, y_array),
                                initial_theta, method='Newton-CG', jac=True)
                                # In this optimization, jac = True indicates that the function 
                                # in this case 'cost' returns the jacobian/ gradient as well

    except:
        print 'Unable to perform Newton-CG optimization on data'
        exit(1)

    print 'Optimization performed on function'
    raw_input('Paused ')

    ###########################
    # Perform Machine Learning
    ###########################

    theta = result_Newton_CG.x
    print 'Theta:', theta

    print 'Cost at this theta: ', cost(theta,X_array,y_array)
    raw_input('Paused ')
    p = predict(theta,X_array)

    print 'Training accuracy: ', float(mean(p == y) * 100)

    exit(0)

if __name__ == '__main__':
    __main__()
