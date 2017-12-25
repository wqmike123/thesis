#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 00:34:48 2017

@author: wq
"""
from collections import defaultdict
import pandas as pd
import glob
import sys
sys.path.append('./code/function/')
from preprocess import *
from training import *
import pickle
import glob,os
import matplotlib.pyplot as plt
import gensim
from cnn import *
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as lr
import numpy as np
from rnn import *
#import keras
#reload(sys)  
#sys.setdefaultencoding('utf-8')

data_dir = './wsj/fullnews/'
#%%
vol = pd.read_csv('./marketdata/vixcurrent.csv',header=1)
vol.Date = pd.to_datetime(vol.Date).dt.date

#%%
nov_dir = './wsj/nov/'
basename = 'wsj_bank_related_news_'
for iyr in range(2006,2011):
    bn = basename + str(iyr)
    data = pd.read_csv(data_dir + bn  + '.csv')
    temp = pd.read_csv(data_dir + bn  + '_2.csv')
    data = pd.concat([data,temp],axis = 0)
    temp = pd.read_csv(nov_dir + bn + '.csv')
    data = pd.concat([data,temp],axis = 0)
    data.to_csv('./wsj/fullnews/' + bn + '.csv',index = False,encoding = 'utf-8')
#%%
def getR2(y_actual,factor,isRet = False):
    n = len(y_actual)
    y = np.array(y_actual).reshape((n,1))
    x = np.array(factor).reshape((n,1))
    if isRet:
        n = n-1
        y = np.log(y[1:]/y[:-1])
        x = x[:-1]
    reg = lr()
    reg.fit(x,y)
    return r2_score(y,reg.predict(x))
#%%
def drawVol(sentiment,vol,colname = 'sentiment',plotName = 'sentiment_vol'):
    fig,ax = plt.subplots()
    senti = sentiment.groupby('date')[[colname]].sum()
    res = pd.merge(senti.reset_index().rolling(7).mean(),vol,left_on='date',right_on='Date',how='left').dropna()
    res[[colname,vol.columns[-1]]].plot(secondary_y = colname,ax=ax)
    fig.savefig(plotName + '.png')
    return res
    
#%%
for ij,ifile in enumerate(glob.glob(data_dir+'*.csv')):
    fig,ax = plt.subplots()
    bn = os.path.basename(ifile).split('.')[0]
    data = pd.read_csv(ifile)
    sample = data#.head(20)
    res = []
    for isum in sample.summary:
        if not isinstance(isum,str):
            res.append([])
            continue
        res.append(textTool.sentence2list(isum))#.decode('utf8') for py2
    sample['token'] = res
    
          
    #%% test map to pos/neg dict
    mcdict = textTool.getSentDict('./dictionary/mc_dict.xlsx')
    sample['sentiment'] = sample.token.apply(textTool.word2vec,args=(mcdict,)).apply(sum)
#    sample.to_csv(bn+'.csv')
    textTool.saveData(sample,data_dir+bn+'.pickle')
    
    #%%
    sample['date'] = pd.to_datetime(sample.time).dt.date
    daily_emo = sample.query('sentiment<=-2').groupby('date')[['sentiment']].sum()#.query('sentiment<=-2 or sentiment=>2').
    daily_emo.loc[:,'sentiment'] =- 1 * daily_emo.loc[:,'sentiment'].values
    pd.merge(daily_emo.reset_index().rolling(7).mean(),vol,left_on='date',right_on='Date',how='left').dropna()[['sentiment',vol.columns[-1]]].plot(secondary_y = 'sentiment',ax=ax)
    fig.savefig(data_dir+bn+'.png')
    
    
#%% only positive and negative: predictability and results
data = pd.DataFrame()
for ipickle in glob.glob(data_dir+'*.pickle'):
    temp = textTool.loadData(ipickle)#data_dir+'wsj_bank_related_news_2006.pickle')#pd.read_csv('./wsj_bank_related_news_2006_2.csv')
    data = pd.concat([data,temp],axis=0)
    
train_x,train_y = trainTool.prepareSentimentDataset(data,labelthreshold_neg=-2,labelthreshold_pos=2)
#%%
voc = defaultdict(float)
for ilist in train_x:
    textTool.buildVocab(ilist,voc)

#%%
#w2v = textTool.load_bin_vec('./dictionary/GoogleNews-vectors-negative300.bin',voc,maxl=50000)
#W_embed,word2ind = textTool.get_W(list(voc.keys()),w2v=w2v,buildW=True)
w2v = gensim.models.KeyedVectors.load_word2vec_format('./dictionary/GoogleNews-vectors-negative300.bin',binary=True,limit=200000)
w2v.save_word2vec_format('./dictionary/GoogleNews-vectors-negative300.txt', binary=False)
#%% to dict
w2vDict = {}
for iword in w2v.vocab.keys():
    w2vDict[textTool.stem(iword)] = w2v.wv[iword]
#%%
W,word_id = textTool.get_W(w2vDict,buildW=True)
newcol = []
add = []
for irow in train_x:
    tempcol,tempadd = textTool.word2ind(irow,word_id)
    newcol.append(tempcol)
    add.append(tempadd)
for i,item in enumerate(newcol):
    if len(item)!=50:
        newcol[i] = item[:50]
W = np.vstack((W,np.random.normal(size=(sum(add),300))))

#%%
textTool.saveData(train_x,'./temp_res/train_x.pickle')
textTool.saveData(train_y,'./temp_res/train_y.pickle')
textTool.saveData(W,'./temp_res/W.pickle')
textTool.saveData(w2v,'./temp_res/w2v.pickle')
textTool.saveData(word_id,'./temp_res/word_id.pickle')
textTool.saveData(newcol,'./temp_res/newcol.pickle')
textTool.saveData(w2vDict,'./temp_res/w2vDict.pickle')
textTool.saveData(data,'./temp_res/data.pickle')
#%%
train_x = textTool.loadData('./temp_res/train_x.pickle')
train_y = textTool.loadData('./temp_res/train_y.pickle')
W = textTool.loadData('./temp_res/W.pickle')
w2v = textTool.loadData('./temp_res/w2v.pickle')
newcol = textTool.loadData('./temp_res/newcol.pickle')
w2vDict = textTool.loadData('./temp_res/w2vDict.pickle')
data = textTool.loadData('./temp_res/data.pickle')
word_id = textTool.loadData('./temp_res/word_id.pickle')
#%% tag the samples
#newcol = np.matrix(newcol)
newcol,train_y = textTool.balanceData(newcol,train_y.values)
#%%
truncated = []
for i,irow in enumerate(newcol):
    truncated.append(irow[30:])
truncated = np.array(truncated)
ylabel = np.zeros([len(train_y),2])
ylabel[train_y<=0,0] = 1
ylabel[train_y>0,1] = 1
#%%
model_cnn = cnn(20,140646,trainable =True)#,filters=10000,hidden_dim = 1024)
#classweight = {0:1,1:4}
model_cnn.fit(truncated[:30000],ylabel[:30000,:],truncated[30000:],ylabel[30000:,:])#,class_weight = classweight)
    
#%%
data = pd.DataFrame()
for ipickle in glob.glob(data_dir+'*.pickle'):
    temp = textTool.loadData(ipickle)#data_dir+'wsj_bank_related_news_2006.pickle')#pd.read_csv('./wsj_bank_related_news_2006_2.csv')
    data = pd.concat([data,temp],axis=0)
test_x,test_y = trainTool.prepareSentimentDataset(data,labelthreshold_neg=1,labelthreshold_pos=-1,keep_neutral=True)
#%%
newcol_test_x = []
#add = []
for irow in test_x:
    tempcol,tempadd = textTool.word2ind(irow,word_id,addword = False,padding_len=20)
    newcol_test_x.append(tempcol)
#    add.append(tempadd)
for i,item in enumerate(newcol_test_x):
    if len(item)!=20:
        newcol_test_x[i] = item[:20]
pred_y = model.model.predict(newcol_test_x)
#%% rnn

model_rnn = mylstm(20,140646,embedweight = W,lstm_units = 512,hidden_dims = [256])
model_rnn.fit(truncated[:30000],ylabel[:30000,:],truncated[30000:],ylabel[30000:,:])

pred_y_rnn = model_rnn.predict(newcol_test_x)
#%%
# build index map and W
# delete redundant

#%%

#%% build a simple RNN with gloVec as initial


#%% put two model together at the same layer (one put important words, one put sentence structure)




#%% reinforcement learning: earning as gain (train a trader?)