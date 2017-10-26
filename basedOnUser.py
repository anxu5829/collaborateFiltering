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
import usefulMethod




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



## 协同过滤算法：基于用户
    # 1 计算用户平均评分
Umean = usefulMethod.userAvg(msno_song)









## contribute the pp matrix

## predict use uu matrix

## predict use pp matrix

## evaluate the result

## use SVD method: random gradient descending method
























