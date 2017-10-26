# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:47:45 2017

@author: xuan

target itemCF and userCF 实现

"""

import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")

train = pd.read_csv("train.csv")

"""
注意 index 从0开始
row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csc_matrix((data, (row, col))).toarray()
"""

# extract info of user and product


# notes: df[[a.b]] will extract data with type df
#        df[a] will extract data with type series
msno = train["msno"]

song_id = train["song_id"]

# extract index of user and product which will be used for searching
msno_index = msno.index
song_id_index = song_id.index

# contribute the csc_matrix saving info of u-p relations
# first ：get the unique list of msno and song


msnoU = msno.unique()
songU = song_id.unique()

# use code to substitude the original data
# change it to categorical data is a very important method


# attention : the parameters categories is very important !
# it will set the order of the category

msnoC = msno.astype("category", categories=msnoU).cat.codes
songC = song_id.astype("category", categories=songU).cat.codes

msno_song = csc_matrix((train.target, (msnoC, songC)))

# if you want get the subset of the msno_song
# you may do it like msno_song[1,1:10].toarray()

# get size :msno_song.shape


## contribute the uu matrix

# neglect the songs that no one has listened


"""
    song_listen = np.sum(msno_song,0)

    # when you want to get subset of a sparse matrix using logial index
    # you may use np.array() to package your logical index 

    // 以下做法不可行，废除
    msno_songForUU = msno_song[:,np.array((song_listen!=0).tolist()[0])]
    how to change matrix to array more proficiently
    np.asarray(song_listen!=0).reshape(-1)
    or 
    np.squeeze(np.asarray(song_listen!=0))
    or
    np.asarray(song_listen!=0).squeeze()
"""

# calculate the similarites

"""
    0 建立zero 的 sparseUU 矩阵
    UU = csc_matrix((len(msnoU),len(msnoU)))
    1 思路： 首先获取非0元素的位置

    2 遍历每一列，使得从列中取出uu关系

    3 在UU列表中更新uu子矩阵的关系    

     song4=msno_song[:,4]
     userListenedSong4 = song4.nonzero()[0]


     # how to extract subset of matrix
         method1 UU[[0,1],[0,1]] extract only two values

         # subtract col: UU[[0,1]] get the 0,1 row 提取列： UU[:,1]
         method2 UU[[0,1]][:,[0,1]] extract a submatrix
         UU[userListenedSong4][:,userListenedSong4]


        UU = np.arange(9).reshape((3,3))

        def selectM(U , list1,list2):
            row = [[i]*len(list2) for i in list1]
            col =  np.array(list2*len(list1)).reshape(len(list1),len(list2))
            return U[row,col]



        UU[[0,2]][:,[0,2]]
"""


## contribute the pp matrix

## predict use uu matrix

## predict use pp matrix

## evaluate the result

## use SVD method: random gradient descending method
























