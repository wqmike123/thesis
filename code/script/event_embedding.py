# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:35:34 2017

@author: wqmike123
"""
import sys
sys.path.append('./code/function/')
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from preprocess import *
from vixPredict import *
import tensorflow as tf
import gensim
import os
from training import *
from newsKneighbor import *
#%% import data
cwd = os.getcwd()
W,news = trainTool.prepareWSJNews(cwd)
target,label = trainTool.prepareSPX(cwd,isClass=True,threshold=0.003,lag = 0)

#%% news embedding
logDir = './temp_res/log/'
# clip value should be scaled or to predict return instead
sess = tf.Session()
model = vixClassify_cnn(27,len(W),embedweight = W.astype(np.float32),trainable = False,lstm_units=512,learning_rate = 0.01,event_embedding=600,pred_dense_dim=64)#,clipvalue=5)
saver = tf.train.Saver()
saver.restore(sess,"./temp_res/log/model_cnn65/model_cnn.ckpt")

#%%

np.random.seed(17)
state_cell = np.random.randn(128).astype(np.float32)
state_hidden = np.random.randn(128).astype(np.float32)
last_date = pd.to_datetime('2006-01-01').date()
news_embedding_list = []
state_hidden_list = [state_hidden]
glb_step = 0
acc_count = 0.
tot_count = 0.
news_list = []
for i,idate in enumerate(target.index):
    x = np.vstack(news[np.logical_and(news.index>=last_date,news.index<idate)].news.values).astype(np.float32)
    if len(x)==0:
        continue
    news_list.append(x)
    feed_dict = {
          model.inputs: x,
          model.dropout: 1.0,
          model.state_cell:state_cell,
          model.state_hidden:state_hidden,
          model.dropout_event:1.0
        }
#    price,state,summary = sess.run(
#            [model.price, model.state,model.summary_op],
#            feed_dict)
    nebd,state = sess.run(
        [model.news_embedding, model.state],
        feed_dict)
    state_cell,state_hidden = state
    state_cell = np.squeeze(state_cell,axis=0)
    state_hidden = np.squeeze(state_hidden,axis=0)
    news_embedding_list.append(nebd)
    state_hidden_list.append(state_hidden)
#        if np.argmax(price) == 2:
#            profit.append(np.log(vol.loc[idate,'VIX Close']/vol.loc[last_date,'VIX Close']))
#        elif np.argmax(price) == 0:
#            profit.append(np.log(vol.loc[idate,'VIX Close']/vol.loc[last_date,'VIX Close'])*-1)
#        else:
#            profit.append(0)
    last_date = idate
#%% closest data
#kn = newsKneighbor(news_list,news_embedding_list)
#textTool.saveData(kn.event,'./temp_res/kn_event_vix_week.pickle')
#textTool.saveData(kn.news,'./temp_res/kn_news_vix_week.pickle')
#kn = newsKneighbor(restore=True,saveDir_news='./temp_res/kn_news_vixfull.pickle')#
knn = kn.find_neighbor(1)

#%%
#with tf.variable_scope('predict',reuse=True):
#    attW = tf.get_variable('attention_W')
#att_W = attW.eval(session=sess)
#textTool.saveData(att_W,'./temp_res/att_Wvix_week.pickle')

#att = attentionRank(news_embedding_list,att_W,state_hidden_list[:-1],news_list)
resNews = att.compAttention(123,False)
print(resNews.sort_values(by='score',ascending=False))
 