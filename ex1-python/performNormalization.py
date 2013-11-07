from pylab import *

def normalizeData(X):
    Xnormalized = copy(X)
    meanTable = []
    stdTable = []
    
    for i in range(shape(X)[1]):
        meanTable.append(mean(Xnormalized[:,[i]]))
        stdTable.append(std(Xnormalized[:,[i]]))
        Xnormalized[:, [i]] = (Xnormalized[:, [i]] - mean(Xnormalized[:, [i]]))/std(Xnormalized[:,[i]])
    return (Xnormalized, meanTable, stdTable)
