######################################################
# Author:  Jivitesh Singh Dhaliwal
# Date:    07-11-2013
# Program: Main (user interface) for Linear Regression
######################################################

###################
# Import Libraries
###################

from pylab import *                             # Pylab: Numpy+Scipy+matplotlib
from matplotlib import *

###################
# Import Functions
###################

from gradientDescent import *
from normalEquation import *
from performNormalization import *

def __main__():
    '''Perform Linear Regression using gradient descent or normal equations on data provided by user'''
    ###############
    # Obtain Data
    ###############
    print 'Inside main() '
    dataFileName = raw_input('Please enter the filename of data file: ')
    dataDelimiter    = raw_input('Please enter the delimiter : ')
    
    try:
        data = genfromtxt(open(dataFileName, 'r') ,delimiter = dataDelimiter, comments= '""') # Directly imports to 'data' 
        X = data[:,0:-1]                            # Assuming that the last row of data is the target value
                                                    # and all others are those of X
        y = data[:,[-1]]                            # Note the single parenthesis to assert that we are choosing
                                                    # one single column
#        if shape(X)[1] <= 2:
#            plotdata(X,y)                           # Display data to user if possible

        ################################
        # Perform Normalization
        ################################

        (Xnormalized, meanTable,stdTable) = normalizeData(X)
        
        Xnormalized = hstack((ones((shape(Xnormalized)[0], 1)), Xnormalized)) 
        X = hstack((ones((shape(X)[0], 1)), X))                  # Add X0 = 1 

    except:
        print 'Unable to process data'
        exit(1)                                     # Exit the program with error code 1

    ###############################
    # Assign Theta, alpha, num_iter
    ###############################  
    
    initialTheta = zeros((shape(X)[1],1))              # The number of columns of X is the no. of parameters
    alpha = float(raw_input('Enter value of alpha: ')) # Allow user to choose alpha
    numIter = int(raw_input('Iterations: '))          # User specified number of iterations
    
    ###########################
    # Perform Machine Learning
    ###########################
    
    thetaGradDes  = gradientDescent(Xnormalized,y,initialTheta, alpha, numIter) 
    thetaNormalEq = normalEquation(X,y,initialTheta)
    
    print 'Theta observed using Gradient Descent:', thetaGradDes
    print 'Theta observed using Normal Equations:', thetaNormalEq

    exit(0)


if __name__ == '__main__':
    __main__()
