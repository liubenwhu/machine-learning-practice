# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:28:29 2019

@author: Administrator
"""
import pandas as pd
import numpy as np

orldata=pd.read_csv('D://frad_train.csv')
orltest=pd.read_csv('D://frad_test.csv')
orldata=orldata.append(orltest)
del orltest
feature=orldata.columns.values.tolist()
orldata.astype(object)
orldata.dtypes.value_counts()

sample=orldata.iloc[0:100,:]

t1=orldata.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
t2=orldata.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
t3=orldata.select_dtypes('float64').apply(pd.Series.nunique, axis = 0)
t4=orldata.select_dtypes('uint8').apply(pd.Series.nunique, axis = 0)

from sklearn.feature_extraction import FeatureHasher

bin_columns_name=['pkgname','ver','adunitshowid','mediashowid','apptype','city','reqrealip','idfamd5','openudidmd5','model','make','osv']
for i in bin_columns_name:
    fh = FeatureHasher(n_features=5, input_type='string')
    orldata[i]=orldata[i].astype('str')
    hashed_features = fh.fit_transform(orldata[i])
    hashed_features = hashed_features.toarray()
    hashed_features=pd.DataFrame(hashed_features)
    hashed_features.columns=[i+'0',i+'1',i+'2',i+'3',i+'4']
    orldata=orldata.join(hashed_features)
    orldata=orldata.drop(columns=i)
    
oh_columns=['province','dvctype','ntt','carrier','os','orientation','lan']
orldata_oh=pd.get_dummies(orldata[oh_columns].astype('object'))
orldata_oh=orldata_oh.reset_index(drop=True)
orldata=orldata.join(orldata_oh)


orldata=orldata.drop(columns=oh_columns)
orldata=orldata.drop(columns='sid')
label=orldata['label']
del orldata_oh
orldata=orldata.drop(columns=['ip','adidmd5','imeimd5','macmd5'])


orldata['nginxtime']=(orldata['nginxtime']-orldata['nginxtime'].min())/(orldata['nginxtime'].max()-orldata['nginxtime'].min())
orldata['h']=(orldata['h']-orldata['h'].min())/(orldata['h'].max()-orldata['h'].min())
orldata['w']=(orldata['w']-orldata['w'].min())/(orldata['w'].max()-orldata['w'].min())
orldata['ppi']=(orldata['ppi']-orldata['ppi'].min())/(orldata['ppi'].max()-orldata['ppi'].min())

orl_train=orldata[orldata['label']<2].drop(columns='label').values
orl_test=orldata[orldata['label']==2].drop(columns='label').values

del orldata
label=label[label<2].values
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train,X_test,y_train,y_test=train_test_split(orl_train,label,random_state=0,test_size=0.2)
train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数

}
clf=lgb.train(params,train_data,valid_sets=[validation_data])
y_pred=clf.predict(X_test)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0
print(f1_score(y_test, y_pred, average='micro'))

#y=label.values
#X=orldata_v
#del label
#del orldata_v
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
#X_train1,X_train2,y_train1,y_train2=train_test_split(X_train,y_train,random_state=0,test_size=0.2)
#
#from sklearn.metrics import f1_score
#def lgb_f1_score(y_hat, data):
#    y_true = data.get_label()
#    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
#    return 'f1', f1_score(y_true, y_hat), True
#
#
#train_data=lgb.Dataset(X_train1,label=y_train1)
#validation_data=lgb.Dataset(X_train2,label=y_train2)
#
#
#params = {
#    'task': 'train',
#    'boosting_type': 'gbdt',  # 设置提升类型
#    'objective': 'binary', # 目标函数
#    'metric': {'l2', 'auc'},  # 评估函数
#
#}
#clf=lgb.train(params,train_data,valid_sets=[validation_data])
#y_pred=clf.predict(X_test)
#y_pred[y_pred>0.9]=1
#y_pred[y_pred<0.1]=0
#print(f1_score(y_test, y_pred, average='micro'))
#
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, y_pred)

#b=pd.concat([orldata[['sid', 'pkgname']], pd.DataFrame(hashed_features)], 
#          axis=1).iloc[1:6]

#def click_counting(sample, bin_column):
#    clicks = pd.Series(sample[sample['label'] > 0][bin_column].value_counts(), name='label-yes')
#    no_clicks = pd.Series(sample[sample['label'] < 1][bin_column].value_counts(), name='label-no')
#    
#    counts = pd.DataFrame([clicks,no_clicks]).T.fillna('0')
#    counts['total'] = counts['label-yes'].astype('int64') + counts['label-no'].astype('int64')
#    
#    return counts
#
#def bin_counting(counts):
#    counts['N+'] = counts['label-yes'].astype('int64').divide(counts['total'].astype('int64'))
#    counts['N-'] = counts['label-no'].astype('int64').divide(counts['total'].astype('int64'))
#    counts['log_N+'] = counts['N+'].divide(counts['N-'])
#
##    If we wanted to only return bin-counting properties, we would filter here
#    bin_counts = counts.filter(items= ['N+', 'N-', 'log_N+'])
#    return counts, bin_counts
#
#
#label=orldata['label']
#
#bin_columns_name1=['pkgname','ver','adunitshowid','mediashowid','apptype','city','reqrealip','imeimd5','idfamd5']
## bin counts example: device_id
#bin_columns_name2=['ip','openudidmd5','macmd5','model','make','osv','adidmd5']
#bin_columns_name=bin_columns_name1+bin_columns_name2
#
#oh_columns=['province','dvctype','ntt','carrier','os','orientation','lan']
#
#
#for i in bin_columns_name:
#    bin_column = i
#    device_clicks = click_counting(orldata.filter(items= [bin_column, 'label']), bin_column)
#    device_all, device_bin_counts = bin_counting(device_clicks)
#
#    #ind=device_all.index.values.tolist()
#    #ind=pd.DataFrame(ind,columns=['pkgname'])
#    device_all=device_all.reset_index()
#    device_all.rename(columns={'index': i,'N+':'N+'+i}, inplace=True) 
#    
#
#    orldata = pd.merge(orldata, device_all.iloc[:,[0,4]], how='left', on=i)
#    orldata=orldata.drop(columns=i)
#
#orldata.select_dtypes('float64').apply(pd.Series.nunique, axis = 0)
#
#
#orldata_oh=pd.get_dummies(orldata[oh_columns].astype('object'))
#
#orldata=orldata.join(orldata_oh)
#
#orldata=orldata.drop(columns=oh_columns)
#orldata=orldata.drop(columns='sid')
#
#del bin_column
#del bin_columns_name
#del bin_columns_name1
#del bin_columns_name2
#del device_all
#del device_bin_counts
#del device_clicks
#del feature
#del i
#del oh_columns
#del orldata_oh
#del sample
#del t1
#del t2
#
#orldata_v=orldata.values
#del orldata
#from sklearn.preprocessing import MinMaxScaler
#
##区间缩放，返回值为缩放到[0, 1]区间的数据
#orldata_v=MinMaxScaler().fit_transform(orldata_v)
#

#
#
#
#
#
