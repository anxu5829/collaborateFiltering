import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

def selectM(U, list1, list2):
    row = [[i] * len(list2) for i in list1]
    col = np.array(list2 * len(list1)).reshape(len(list1), len(list2))
    return U[row, col]


def userAvg(UPmatrix):
    temp = UPmatrix.sum(axis=1) / (UPmatrix != 0).sum(axis=1)
    temp[np.isnan(temp)] = 0
    return temp

def sim(UPmatrix):
    usermean = userAvg(UPmatrix)
    for useri in np.range(UPmatrix.shape[0]):
        for userj in np.range(UPmatrix.shape[0]):
            pass
def rowCosine(UPmatrix):
    usernum = UPmatrix.shape[0]
    uu = csc_matrix((usernum,usernum))
    for useri in np.arange(usernum):
        for userj in np.arange(usernum):
            if useri==userj :
                uu[useri,userj]=1
            else:
                flag = UPmatrix[useri,] == UPmatrix[userj,]
                innerproduct = np.sum(flag.toarray() * UPmatrix[userj,].toarray() * UPmatrix[useri,].toarray())
                userjLength = np.sqrt(np.sum( flag.toarray() * UPmatrix[userj,].toarray()**2) )
                useriLength = np.sqrt( np.sum( flag.toarray() * UPmatrix[useri,].toarray() ** 2))
                temp = innerproduct/(useriLength*userjLength)
                if np.isnan(temp) == False:
                    uu[useri,userj] = temp

    return uu

