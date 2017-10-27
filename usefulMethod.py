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




def cululateResidual(UPmatrix,UserLatent,ProductLatent):
    ProductLatentTranspose = ProductLatent.transpose()
    UPmatrix_hat =UserLatent.dot(ProductLatentTranspose)
    S = (UPmatrix !=0)
    Error = UPmatrix-UPmatrix_hat
    Residual2 = np.multiply(Error, Error)
    loss = np.sum(np.multiply(S,Residual2))
    return (Error,loss)

def SVD(UPmatrix,k,iterator = 10, alpha = 0.005):
    lenOfUser     = UPmatrix.shape[0]
    lenOfProduct  = UPmatrix.shape[1]
    lenOfGenre    = k
    UserLatent    = np.random.random((lenOfUser,lenOfGenre)) #csc_matrix((lenOfUser, lenOfGenre))
    ProductLatent = np.random.random((lenOfProduct,lenOfGenre)) #csc_matrix((lenOfProduct,lenOfGenre))
    S = (UPmatrix != 0)
    flag = 0
    while flag < iterator:
        for user in np.arange(S.shape[0]):
            for product in np.arange(S.shape[1]):
                if S[user,product] == 1:
                    #lossMatrix,_=cululateResidual(UPmatrix,UserLatent,ProductLatent)
                    #Eij = lossMatrix[user,product]
                    Eij = 0
                    for genr in np.arange(k):
                        Eij = Eij +UserLatent[user,genr]*ProductLatent[product,genr]
                    Eij = UPmatrix[user,product] - Eij
                    for genr in np.arange(k):
                        UserLatentNew=UserLatent[user,genr] + alpha*Eij*ProductLatent[product,genr]
                        ProductLatentNew = ProductLatent[product,genr] + alpha*Eij*UserLatent[user,genr]
                        UserLatent[user,genr] = UserLatentNew
                        ProductLatent[product,genr] = ProductLatentNew
        flag+=1
    return (UserLatent,ProductLatent)






