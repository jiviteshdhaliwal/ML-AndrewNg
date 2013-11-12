from scipy import optimize

def advancedOptimization(function, x0, methodName, jacArg):
    return optimize.minimize(function, x0, method = methodName, jac = jacArg).x 
