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
def converCscMatrix(cscMatrix):
    datalist = []
    l = len(cscMatrix.indptr)
    for i in np.arange(len(cscMatrix.indptr)):
        if i!=(len(cscMatrix.indptr)-1):
            datalist.append(cscMatrix.indices[cscMatrix.indptr[i]:cscMatrix.indptr[i+1]])
    # 用pandas 来解决问题
    data = pd.DataFrame(columns=["col","row","data"])
    cols = []
    for i in np.arange(1,len(cscMatrix.indptr)):
        cols.extend([i-1]*(cscMatrix.indptr[i]-cscMatrix.indptr[i-1]))
    data = pd.DataFrame()
    data["col"] = cols
    data["row"] = cscMatrix.indices
    data["data"] = cscMatrix.data
    return data

def cululateResidual(UPmatrix,UserLatent,ProductLatent):
    data = converCscMatrix(UPmatrix)
    lengenre = UserLatent.shape[1]
    Eij = 0
    for index in data.index:
        (col,row,value) = data.loc[index]
        value_hat = 0
        for k in np.arange(lengenre):
            value_hat += UserLatent[row,k]*ProductLatent[col,k]
        Eij += value - value_hat
    return (Eij)

def SVD(UPmatrix,k,iterator = 10, alpha = 0.005):
    lenOfUser     = UPmatrix.shape[0]
    lenOfProduct  = UPmatrix.shape[1]
    lenOfGenre    = k
    UserLatent    = np.random.random((lenOfUser,lenOfGenre)) #csc_matrix((lenOfUser, lenOfGenre))
    ProductLatent = np.random.random((lenOfProduct,lenOfGenre)) #csc_matrix((lenOfProduct,lenOfGenre))
    UserLatent    = csc_matrix(UserLatent)
    ProductLatent = csc_matrix(ProductLatent)
    S = converCscMatrix(UPmatrix)
    flag = 0
    while flag < iterator:
        for index in S.index:
            (col, row, value) = S.loc[index]
            #lossMatrix,_=cululateResidual(UPmatrix,UserLatent,ProductLatent)
            #Eij = lossMatrix[user,product]
            value_hat = 0
            for genr in np.arange(k):
                value_hat = value_hat +UserLatent[row,genr]*ProductLatent[col,genr]
            Eij = value - value_hat
            for genr in np.arange(k):
                UserLatentNew=UserLatent[row,genr] + alpha*Eij*ProductLatent[col,genr]
                ProductLatentNew = ProductLatent[col,genr] + alpha*Eij*UserLatent[row,genr]
                UserLatent[row,genr] = UserLatentNew
                ProductLatent[col,genr] = ProductLatentNew
        flag+=1
    return (UserLatent,ProductLatent)






