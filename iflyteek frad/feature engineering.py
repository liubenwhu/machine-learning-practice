# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:28:29 2019

@author: Administrator
"""
import pandas as pd

orldata=pd.read_csv('D://frad_train.csv')

orldata.head()

feature=orldata.columns.values.tolist()

orldata.dtypes.value_counts()

sample=orldata.iloc[0:100,:1]

t1=orldata.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
t2=orldata.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)

