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
from performNormalization import *
from fmin import *

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


    result_Newton_CG= optimize.minimize(lambda t: cost(t, X_array, y_array),
                                initial_theta, method='Newton-CG', jac=True)


#    dataFileName = raw_input('Please enter the filename of data file: ')
#    dataDelimiter    = raw_input('Please enter the delimiter : ')
#    
#    try:
#        data = genfromtxt(open(dataFileName, 'r') ,delimiter = dataDelimiter, comments= '""') # Directly imports to 'data' 
#        X = data[:,0:-1]                            # Assuming that the last row of data is the target value
#                                                    # and all others are those of X
#        y = data[:,[-1]]                            # Note the single parenthesis to assert that we are choosing
#                                                    # one single column
##        if shape(X)[1] <= 2:
##            plotdata(X,y)                           # Display data to user if possible
#
#        ################################
#        # Perform Normalization
#        ################################
#
#        X = hstack((ones((shape(X)[0], 1)), X))                  # Add X0 = 1 
#
#    except:
#        print 'Unable to process data'
#        exit(1)                                     # Exit the program with error code 1

    ###############################
    # Assign Theta, alpha, num_iter
    ###############################  
    
#    initialTheta = zeros((shape(X)[1],1))              # The number of columns of X is the no. of parameters
  
#    print 'Cost and gradient at initial theta: ', cost(initialTheta, X, y)
 
    raw_input('Paused ')

    ###########################
    # Perform Machine Learning
    ###########################

    # Using optimization function Without using gradient 

#    theta = optimize.minimize(lambda t: cost(t, X, y), initialTheta, method ='Newton-CG' , jac = True).x

 #   raw_input('Paused')

    theta = result_Newton_CG.x
    print 'Theta:', theta

    print 'Cost at this theta: ', cost(theta,X_array,y_array)
    raw_input('Paused ')
    p = predict(theta,X_array)

    print 'Training accuracy: ', float(mean(p == y) * 100)

    # Using optimization function using gradient 

#    theta = optimize.fmin_bfgs(cost, x0 = initialTheta, fprime = gradDes, args = (X, y)) 
    
#    print 'Theta:', theta 

    exit(0)

if __name__ == '__main__':
    __main__()
