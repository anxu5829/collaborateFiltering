import numpy as np
import pandas as pd


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
