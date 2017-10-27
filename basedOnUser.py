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
import createUpMatrix
## 协同过滤算法：基于用户
    # 1 计算用户平均评分


workfile = "C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys"

UPmatrix = createUpMatrix.UPmatrix(workfile)


























