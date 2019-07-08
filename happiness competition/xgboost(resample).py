# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:40:01 2019

@author: Administrator
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def misratiodata(orldata):
    mis=orldata.describe().iloc[0,:]/orldata.iloc[:,1].size
    orldata=orldata.drop(mis[mis<0.5].index,axis=1)
    mis=mis[mis>=0.5]
    mis=mis[mis<1]
    print(mis)
    return orldata

def misratio(orldata):
    mis=orldata.describe().iloc[0,:]/orldata.iloc[:,1].size
    mis=mis[mis<1]
    return mis

orldata=pd.read_csv('C:/Users/Administrator/Desktop/4.2stg/happ_nochar.csv',parse_dates=['survey_time'],encoding='latin-1')

orldata=misratiodata(orldata)

orldata.dtypes.value_counts()#查看类型的 很关键
t1=orldata.select_dtypes('float64').apply(pd.Series.nunique, axis = 0)#查看每个特征的不同取值个数，通过这个来分辨类别特征和连续特征
t2=orldata.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
orldata['happiness'].value_counts()
orldata['happiness'].astype(int).plot.hist();

corrmat = orldata.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

numeric_feats = t1[t1>=20]
numeric_feats=numeric_feats.append(t2[t2>=20])
numeric_feats=numeric_feats.index.values.tolist()
numeric_feats.remove('city')
numeric_feats.remove('province')
numeric_feats.remove('county')

category_feats=t1[t1<20].append(t2[t2<20])
category_feats=category_feats.index.values.tolist()
category_feats.append('city')
category_feats.append('province')
category_feats.append('county')

del t1 
del t2




#后面的只能一个一个操作了
orldata['edu_status']=orldata['edu_status'].fillna(9997)
orldata['edu_yr']=orldata['edu_yr'].fillna(9997)
orldata['hukou_loc']=orldata['hukou_loc'].fillna(9997)
orldata['social_neighbor']=orldata['social_neighbor'].fillna(9997)
orldata['social_friend']=orldata['social_friend'].fillna(9997)
orldata['family_income']=orldata['family_income'].fillna(orldata['family_income'].median)
orldata['minor_child']=orldata['minor_child'].fillna(9997)
orldata['marital_1st']=orldata['marital_1st'].fillna(9997)

mis3=misratio(orldata)
for i,inv in enumerate(mis3.index):
    orldata[inv]=orldata[inv].fillna(9997)

orldata=orldata[~orldata['happiness']<0]
index=orldata.columns.values.tolist()

index.remove('id')
index.remove('happiness')
index.remove('survey_time')
index.remove('family_income')

features=index
#features=['inc_ability','gender','status_peer','work_exper','family_status','health','equity','class','health_problem','family_m','house','depression','learn','relax','edu']
X=orldata[features].values
y=orldata['happiness'].values
X[X<0]=9997
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test  = train_test_split(X,y,test_size=0.1, random_state=0)

from imblearn.over_sampling import SMOTE, ADASYN

from collections import Counter
from pandas import  DataFrame

X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X_train, y_train)

sorted(Counter(y_resampled_smote).items())


X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X_train, y_train)


import xgboost as xgb
dtrain = xgb.DMatrix(X_resampled_smote,label=y_resampled_smote)
dtest = xgb.DMatrix(X_test)
parameters = {
'booster': 'gbtree',
    'objective': 'reg:gamma',  
    'num_class': 1,               
    'gamma': 0.5,                  
    'max_depth': 16,               
    'lambda': 3,                   
    'subsample': 0.7,              
    'colsample_bytree': 0.7,       
    'min_child_weight': 3,
    'silent': 1,                   
    'eta': 0.007,                  
    'seed': 1000,
    'nthread': 4,                  
}
num_round = 50
from datetime import datetime 
start = datetime.now() 
xg = xgb.train(parameters,dtrain,num_round) 
stop = datetime.now()
ypred = xg.predict(dtest)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
#ypred2=xg.predict(dtrain)
#accuracy = np.mean(y_test == ypred) * 100  
#accuracy2=np.mean(y_resampled_smote==ypred2)*100
#print(accuracy)
#print(accuracy2)

#yanzheng=pd.read_csv('C:/Users/Administrator/Desktop/4.2stg/happiness_test_complete.csv',parse_dates=['survey_time'],encoding='latin-1')
#yanzheng2=yanzheng[features].values
#yanzheng2=xgb.DMatrix(yanzheng2)
#a=xg.predict(yanzheng2)
#new=pd.DataFrame(a)
#new.to_csv('C:/Users/Administrator/Desktop/4.2stg/new.csv')