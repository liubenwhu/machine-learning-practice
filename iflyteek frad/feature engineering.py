# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:28:29 2019

@author: Administrator
"""
import pandas as pd

orldata=pd.read_csv('D://frad_train.csv')


feature=orldata.columns.values.tolist()

orldata.dtypes.value_counts()

sample=orldata.iloc[0:100,:]

t1=orldata.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
t2=orldata.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)

def click_counting(sample, bin_column):
    clicks = pd.Series(sample[sample['label'] > 0][bin_column].value_counts(), name='label-yes')
    no_clicks = pd.Series(sample[sample['label'] < 1][bin_column].value_counts(), name='label-no')
    
    counts = pd.DataFrame([clicks,no_clicks]).T.fillna('0')
    counts['total'] = counts['label-yes'].astype('int64') + counts['label-no'].astype('int64')
    
    return counts

def bin_counting(counts):
    counts['N+'] = counts['label-yes'].astype('int64').divide(counts['total'].astype('int64'))
    counts['N-'] = counts['label-no'].astype('int64').divide(counts['total'].astype('int64'))
    counts['log_N+'] = counts['N+'].divide(counts['N-'])

#    If we wanted to only return bin-counting properties, we would filter here
    bin_counts = counts.filter(items= ['N+', 'N-', 'log_N+'])
    return counts, bin_counts


label=orldata['label']

bin_columns_name1=['pkgname','ver','adunitshowid','mediashowid','apptype','city','reqrealip','imeimd5','idfamd5']
# bin counts example: device_id
bin_columns_name2=['ip','openudidmd5','macmd5','model','make','osv']
bin_columns_name=bin_columns_name1+bin_columns_name2

oh_columns=['province','dvctype','ntt','carrier','os','orientation','lan']


for i in bin_columns_name:
    bin_column = i
    device_clicks = click_counting(orldata.filter(items= [bin_column, 'label']), bin_column)
    device_all, device_bin_counts = bin_counting(device_clicks)

    #ind=device_all.index.values.tolist()
    #ind=pd.DataFrame(ind,columns=['pkgname'])
    device_all=device_all.reset_index()
    device_all.rename(columns={'index': i,'N+':'N+'+i}, inplace=True) 
    

    orldata = pd.merge(orldata, device_all.iloc[:,[0,4]], how='left', on=i)
    orldata=orldata.drop(columns=i)

orldata.select_dtypes('float64').apply(pd.Series.nunique, axis = 0)


orldata_oh=pd.get_dummies(orldata[oh_columns].astype('object'))
