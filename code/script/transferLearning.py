# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:05:44 2017

@author: wqmike123
"""
import sys
sys.path.append('./code/function/')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import *
from fcnn import *
from lancaster import *
import gensim
from sklearn.model_selection import KFold
from training import *
import os
#%% apply SemEval model to wsj
data1 = textTool.loadData('./labeldata/data1.pickle')
res = []
for isample in data1.title:
    if not isinstance(isample,str):
        res.append([])
        continue
    res.append(textTool.sentence2list(isample.lower(),tokenTool='unitok'))
data1['token2'] = res
word_model = gensim.models.Word2Vec.load('./dictionary/word2vec_models/all_fin_model_lower').wv
#w2v.save_word2vec_format('./dictionary/GoogleNews-vectors-negative300.txt', binary=False)
w2v = word_model

w2vDict = {}
for iword in w2v.vocab.keys():
    w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
    
W,word_id = textTool.get_W(w2vDict,buildW=True)
newcol = []
add = []
#add = []
for irow in data1.token2:
    tempcol,tempadd = textTool.word2ind(irow,word_id,addword = False,padding_len=21,fill0=False)
    newcol.append(tempcol)
    add.append(tempadd)
#    add.append(tempadd)
for i,item in enumerate(newcol):
    if len(item)!=21:
        newcol[i] = item[:21]
W = np.vstack((W,np.zeros(shape=(sum(add),300))))
label_y = data1.sentiment.values
newcol = np.array(newcol)

model_blstm = blstm(21,len(W),embedweight = W,lstm_units = 21,epochs_number=100,trainable = False)
model_blstm.fit(newcol,label_y,earlyStopping = True)

cwd = os.getcwd()
_,news = trainTool.prepareWSJNews(cwd,length = 'full')
pred_x = np.zeros([news.shape[0],newcol.shape[1]])
for i,inews in enumerate(news.news.values):
    pred_x[i] = inews[:21]
    
#%%
pred_y = model_blstm.predict(pred_x)

pred_y = model_blstm.predict(newcol[iktest])

#%%
data = textTool.loadData('C:/Users/wqmike123/Documents/thesis/temp_res/blstm_tagged_news.pickle')

