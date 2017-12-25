# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:00:08 2017

@author: wqmike123
"""
#%%
import sys
sys.path.append('./code/function/')
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from preprocess import *
from vixPredict import *
import tensorflow as tf
import gensim
from training import *
import os
import glob
#%% read the data and prepare
#data = textTool.loadData('./temp_res/data.pickle')
#
#res = []
#for isample in data['head']:
#    if not isinstance(isample,str):
#        res.append([])
#        continue
#    res.append(textTool.sentence2list(isample.lower(),tokenTool='unitok'))
#data['token'] = res
#
#data.token.apply(len).max()#27
#
#textTool.saveData(data,'./temp_res/vixPred_data.pickle')
#%% 2011 -2017
wkdir = './wsj/news2010/'
data = pd.DataFrame()
for ifile in glob.glob(wkdir + '*'):
    idata = pd.read_csv(ifile)
    idata.date = pd.to_datetime(idata.date,format = '%Y/%m/%d')
    idata = idata.set_index('date')
    data = pd.concat([data,idata],axis=0)
data.to_csv('./wsj/data1116.csv')
#%%
data = pd.read_csv('./wsj/data1116.csv')
res = []
for isample in data['head']:
    if not isinstance(isample,str):
        res.append([])
        continue
    res.append(textTool.sentence2list(isample.lower(),tokenTool='unitok'))
data['token'] = res
textTool.saveData(data,'./temp_res/vixPred_data1116.pickle')#31
#%%
#data.date = data.date.dt.date
#data = data.set_index('date')
cwd = os.getcwd()
W,news = trainTool.prepareWSJNews(cwd)
target = trainTool.prepareSPX(cwd,isClass=True,threshold=0.005)
#%%
#from tensorflow.python.client import device_lib
#
#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#%%
logDir = './temp_res/log/'
# clip value should be scaled or to predict return instead
sess = tf.Session()
model = vixPredict_cnn(27,len(W),embedweight = W.astype(np.float32),trainable = False,learning_rate = 0.01,event_embedding=512,pred_dense_dim=128)#,clipvalue=5)
sess.run(tf.initialize_all_variables())
train_writer = tf.summary.FileWriter(logDir + '/tb/cnn/train',sess.graph)
aux_writer_r = tf.summary.FileWriter(logDir + '/tb/cnn/train/real')
aux_writer_p = tf.summary.FileWriter(logDir + '/tb/cnn/train/pred')
saver = tf.train.Saver()
#%% train
epoch = 20
split_date = pd.to_datetime('2009-01-01').date()
np.random.seed(17)
state_cell_init = np.random.randn(300).astype(np.float32)
state_hidden_init = np.random.randn(300).astype(np.float32)
res = []
glb_step = 0
for ct in range(epoch):
    state_cell = state_cell_init.copy()
    state_hidden = state_hidden_init.copy()
    last_date = pd.to_datetime('2006-01-01').date()
    prlist = []
    for idate in target.index:
        if idate > split_date:
            break
        x = np.vstack(news[np.logical_and(news.index>=last_date,news.index<idate)].news.values)
        if len(x)==0:
            continue
        y = target[idate]
        feed_dict = {
              model.inputs: x,
              model.y: [y],
              model.dropout: 0.8,
              model.state_cell:state_cell,
              model.state_hidden:state_hidden
            }
        price,loss,opt,state,summary = sess.run(
                [model.price, model.loss_op, model.optimize, model.state,model.summary_op],
                feed_dict)
        state_cell,state_hidden = state
        state_cell = np.squeeze(state_cell,axis=0)
        state_hidden = np.squeeze(state_hidden,axis=0)
        prlist.append(price)
        last_date = idate
        train_writer.add_summary(summary,global_step = glb_step)
        summary_op_aux = sess.run(model.summary_op_aux,{model.plot_in:y})
        aux_writer_r.add_summary(summary_op_aux,glb_step)
        summary_op_aux = sess.run(model.summary_op_aux,{model.plot_in:price[0]})
        aux_writer_p.add_summary(summary_op_aux,glb_step)
        glb_step += 1
    res.append(prlist)
    save_path = saver.save(sess, "./temp_res/log/model_cnn/model_cnn"+str(ct)+".ckpt")
    print('epoch %d loss is %f'%(ct,loss))
    

#%%
#res = textTool.loadData('./temp_res/prediction_in.pickle')
#res = np.array(res)
#res = np.squeeze(res,axis=-1)
#signal = res[-1,:].copy()
#signal[signal>0] = 1
#signal[signal<0] = -1
##signal[np.logical_and(signal<=0.05,signal>=-0.05)] = 0
#c
#
##%%
#split_date = pd.to_datetime('2009-01-01').date()
#resdate = []
#prlist = []
#last_date = pd.to_datetime('2006-01-01').date()
#
#for idate in target.index:
#    if idate > split_date:
#        break
#    x = np.vstack(news[np.logical_and(news.index>=last_date,news.index<idate)].news.values)
#    if len(x)==0:
#        continue
#    y = target[idate]
#    prlist.append(y)
#    resdate.append(idate)
#    last_date = idate
#
##%%
#position = pd.DataFrame({'pos':signal,'price':prlist},index = resdate)
#profit = []
#last_date = pd.to_datetime('2006-01-01').date()
#for idate in target.index:
#    if idate > split_date:
#        break
#    x = np.vstack(news[np.logical_and(news.index>=last_date,news.index<idate)].news.values)
#    if len(x)==0:
#        continue
#    y = vol.loc[idate]
#    profit.append((y['VIX Close'] - y['VIX Open'])/y['VIX Open']*position.loc[idate,'pos'])
#    last_date = idate
##%%
#position['profit'] = profit
#position.profit.cumsum().plot()

#%%
#split_date = pd.to_datetime('2009-01-01').date()
sess = tf.Session()
model = vixPredict_cnn(27,len(W),embedweight = W.astype(np.float32),trainable = False,clipvalue=5)
#model = vixPredict(27,len(W),embedweight = W,predict_state_dim=300,lstm_units = 1024,trainable = False,clipvalue=50)
#sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(sess,"./temp_res/model_cnn/model_cnn7.ckpt")
#%%
test_writer = tf.summary.FileWriter(logDir + '/tb/cnn/test',sess.graph)
aux_writer_r = tf.summary.FileWriter(logDir + '/tb/cnn/test/real')
aux_writer_p = tf.summary.FileWriter(logDir + '/tb/cnn/test/pred')
#%%
count = 0

state_cell = state_cell_init.copy()
state_hidden = state_hidden_init.copy()
last_date = pd.to_datetime('2006-01-01').date()
pos = []
profit = []
resdate = []
real = []
glb_step = 0
for idate in target.index:
    x = np.vstack(news[np.logical_and(news.index>=last_date,news.index<idate)].news.values).astype(np.float32)
    if len(x)==0:
        continue
    count += 1
    y = target[idate]
    feed_dict = {
          model.inputs: x,
          model.dropout: 1.0,
          model.y:[y],
          model.state_cell:state_cell,
          model.state_hidden:state_hidden
        }
    price,state,summary = sess.run(
            [model.price, model.state,model.summary_op],
            feed_dict)
    state_cell,state_hidden = state
    state_cell = np.squeeze(state_cell,axis=0)
    state_hidden = np.squeeze(state_hidden,axis=0)
    test_writer.add_summary(summary,global_step = glb_step)
    summary_op_aux = sess.run(model.summary_op_aux,{model.plot_in:y})
    aux_writer_r.add_summary(summary_op_aux,glb_step)
    summary_op_aux = sess.run(model.summary_op_aux,{model.plot_in:price[0]})
    aux_writer_p.add_summary(summary_op_aux,glb_step)    
    glb_step += 1
    if count >= 100:
        pos.append(price[0])
        real.append(y)
        resdate.append(idate)
        if price[0]>0:
            profit.append(np.log(vol.loc[idate,'VIX Close']/vol.loc[last_date,'VIX Close']))
        elif price[0]<0:
            profit.append(np.log(vol.loc[idate,'VIX Close']/vol.loc[last_date,'VIX Close'])*-1)
        else:
            profit.append(0)
    last_date = idate
    if count%100 == 0:
        print('finish %d'%count)

#save_path = saver.save(sess, "./temp_res/modelcnn"+str(ct)+".ckpt")
#print('epoch %d loss is %f'%(ct,loss))
        
#%% test res
sgnl = np.array(pos)
def getSg(sgn,thres):
    sg = sgn.copy()
    sg[sg>thres] = 1
    sg[sg<-thres] = -1
    sg[np.logical_not(np.logical_or(sg==1,sg==-1))] = 0
    return sg
sg = getSg(sgnl,0.05)
plt.plot(np.cumsum(sg*real))
#%%
from sklearn.linear_model import LinearRegression as lr
reg = lr()
reg.fit(np.array(pos).reshape([-1,1]),np.array(real).reshape([-1,1]))
print(reg.score(np.array(pos).reshape([-1,1]),np.array(real).reshape([-1,1])))