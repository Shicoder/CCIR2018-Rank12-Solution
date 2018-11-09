# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import gc
import datetime as dt

PATH_TO_ORIGINAL_DATA = './data/'
PATH_TO_PROCESSED_DATA = './data/processed/'
import time
print('Loading data from: {}\nOutput directory: {}'.format(PATH_TO_ORIGINAL_DATA, PATH_TO_PROCESSED_DATA))

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'training_set_GRU4Rec.csv', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int32})
data.columns = ['SessionId', 'ItemId','Time']
print('loding finish')
# ttt = time.localtime(1525966602)
# now_date = time.strftime("%Y-%m-%d %H:%M:%S", ttt)
# data['Time'] = pd.to_datetime(data.TimeStr,format='%Y-%m-%dT%H:%M:%S')
# print(data.head())
# data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
# del(data['TimeStr'])

session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
gc.collect()
print('state 1')
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
gc.collect()
print('state 2')
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]
gc.collect()
print('state 3')
# gc.collect()
tmax = data.TimeStr.max()
session_max_times = data.groupby('SessionId').TimeStr.max()
session_train = session_max_times[session_max_times < tmax-86400].index
session_test = session_max_times[session_max_times >= tmax-86400].index
print('train_valid_split')
train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]
test = test[np.in1d(test.ItemId, train.ItemId)]
tslength = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_test.txt', sep='\t', index=False)