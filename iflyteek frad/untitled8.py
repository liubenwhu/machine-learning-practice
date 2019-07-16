# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:27:41 2019

@author: Administrator
"""

import pandas as pd
import numpy as np

from sklearn import datasets

cancer=datasets.load_breast_cancer()

label=pd.DataFrame(cancer.target)

label.describe()

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0,test_size=0.2)
import lightgbm as lgb
train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    

}




#evals_result = {}
#
#clf = lgb.train(params, train_data, valid_sets=[validation_data],  feval=lgb_f1_score, evals_result=evals_result)
#
#lgb.plot_metric(evals_result, metric='f1')
#
def get_f1 (preds,dtrain):
    label=dtrain.get_label()
    preds=np.argmax(preds.reshape(len(label),-1), axis=1)
    f1=f1_score(label,preds,average='weighted')
    return 'f1-score',float(f1),True
#调用与xgb没区别
clf = lgb.train(params,train_data,valid_sets=[validation_data],feval=lgb_f1_score)




y_pred=clf.predict(X_test)
y_pred[y_pred>0.1]=1
y_pred[y_pred<0.1]=0
print(f1_score(y_test, y_pred, average='micro'))