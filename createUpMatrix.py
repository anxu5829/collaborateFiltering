
import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np
import usefulMethod

workfile = "C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys"
def msno_song(workfile):

    os.chdir(workfile)
    train = pd.read_csv("train.csv")
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
    return msno_song
