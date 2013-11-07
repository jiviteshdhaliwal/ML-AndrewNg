######################################################
# Author:  Jivitesh Singh Dhaliwal
# Date:    07-11-2013
# Program: Main (user interface) for Linear Regression
######################################################

from pylab import *                             # Pylab: Numpy+Scipy+matplotlib

###############
# Obtain Data
###############

dataFileName = raw_input('Please enter the filename of data file: ')
delimiter    = raw_input('Please enter the delimiter : ')

try:
    data = genfromtxt(open(dataFileName, 'r') ,delim = delimiter, comments= '""') # Directly imports to 'data' 
    X = data[:,0:-1]                            # Assuming that the last row of data is the target value
                                                # and all others are those of X
    y = data[:,[-1]]                            # Note the single parenthesis to assert that we are choosing
                                                # one single column
except:
    print 'Unable to process data'
    exit(1)                                     # Exit the program with error code 1


############################
# Perform Machine Learning 
############################  



