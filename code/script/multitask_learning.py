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
from multiTask import *
import tensorflow as tf
import gensim
from training import *
import os
import glob
from sklearn.model_selection import KFold

#%% read the data and prepare
#data = textTool.loadData('./temp_res/allnews_tag.pkl')[['TICKER','TITLE','c2c','o2c','ma1','ma2','ma3']]
#res = []
#for isample in data['TITLE']:
#    if not isinstance(isample,str):
#        res.append([])
#        continue
#    res.append(textTool.sentence2list(isample.lower(),tokenTool='unitok'))
#data['token'] = res
#textTool.saveData(data,'./temp_res/allnews_tokenize.pkl')#31
##%%
#data2 = textTool.loadData('./labeldata/data_token.pickle')
##%%
#word_model = gensim.models.Word2Vec.load('./dictionary/word2vec_models/all_fin_model_lower').wv
#w2v = word_model
##%% to dict
#
#w2vDict = {}
#for iword in w2v.vocab.keys():
#    w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
##%%
#allData = list(data.token.values)+list(data2.token.values)
#outAux = np.zeros([len(allData),2])
#label_y = np.zeros([len(allData),2])
#outAux[0:len(data),0] = 1
#outAux[len(data):,1] = 1 
#label_y[0:len(data),0] = data.ma3.values
#label_y[len(data):,1] = data2.sentiment
##%%
#
#W,word_id = textTool.get_W(w2vDict,buildW=True)
#newcol = []
#add = []
##add = []
#for irow in allData:
#    tempcol,tempadd = textTool.word2ind(irow,word_id,addword = False,padding_len=maxlen,fill0=False)
#    newcol.append(tempcol)
#    add.append(tempadd)
#
#for i,item in enumerate(newcol):
#    if len(item)!=maxlen:
#        newcol[i] = item[:maxlen]
#W = np.vstack((W,np.zeros(shape=(sum(add),300))))
#newcol = np.array(newcol)
#%%
#firmCount = data.groupby('TICKER')['TITLE'].count()
#group50 = firmCount[np.logical_and(firmCount<=50,firmCount>10)].index
#data50 = data.query('TICKER in @group50')
#data50 = data50.drop_duplicates('TITLE')

#%%
#textTool.saveData(newcol,'./temp_res/input_newcol_num.pkl')
#textTool.saveData(outAux,'./temp_res/input_outAux_num.pkl')
#textTool.saveData(label_y,'./temp_res/output_label_num.pkl')
#textTool.saveData(W,'./temp_res/input_W.pkl')
#%%
maxlen = 40
newcol = textTool.loadData('./temp_res/input_newcol_num.pkl')
outAux = textTool.loadData('./temp_res/input_outAux_num.pkl')
label_y = textTool.loadData('./temp_res/output_label_num.pkl')
W = textTool.loadData('./temp_res/input_W.pkl')
#%%
logDir = './temp_res/log/multiTask/'
# clip value should be scaled or to predict return instead
sess = tf.Session()
model = multiTask(maxlen,len(W),embedweight = W.astype(np.float32),trainable = False,
                  learning_rate = 0.01)#,clipvalue=5)
sess.run(tf.initialize_all_variables())
train_writer = tf.summary.FileWriter(logDir + '/tb/cnn/train',sess.graph)
aux_writer_r = tf.summary.FileWriter(logDir + '/tb/cnn/train/return')
aux_writer_p = tf.summary.FileWriter(logDir + '/tb/cnn/train/sentiment')
saver = tf.train.Saver()

#%%
test_x = newcol[-110:,:]
test_aux = outAux[-110:,:]
test_y = label_y[-110:,:]

train_x =  newcol[:-110,:]
train_aux = outAux[:-110,:]
train_y = label_y[:-110,:]

train_shuffle = trainTool.shuffle([train_x,train_aux,train_y])
#train_x = train_shuffle[0]
#train_aux = train_shuffle[0]
#train_y = train_shuffle[0]
#%% train
epoch = 40
batch_size = 32
res = []
glb_step = 0
batch_data = trainTool.batch(train_shuffle,batch_size)
train_x = batch_data[0]
train_aux = batch_data[1]
trian_y = batch_data[2]
for ct in range(epoch):
    for itrain,iaux,ilabel in zip(train_x,train_aux,trian_y):
        feed_dict = {
              model.inputs: itrain,
              model.outputAux:iaux,
              model.y: ilabel,
              model.dropout: 0.8
            }
        predict,loss,opt,summary = sess.run(
                [model.predict, model.loss_op, model.optimize, model.summary_op],
                feed_dict)

        train_writer.add_summary(summary,global_step = glb_step)
        islabel = (iaux[:,0] == 1)
        if sum(islabel) > 0:
            sim = trainTool.cosine(ilabel[islabel,0],predict[islabel,0])
            if not np.isnan(sim):
                summary_op_aux = sess.run(model.summary_op_aux,{model.plot_in:sim})
                aux_writer_r.add_summary(summary_op_aux,glb_step)
        if sum(islabel) < batch_size:
            sim = trainTool.cosine(ilabel[np.logical_not(islabel),1],predict[np.logical_not(islabel),1])
            if not np.isnan(sim):
                summary_op_aux = sess.run(model.summary_op_aux,{model.plot_in:sim})
                aux_writer_p.add_summary(summary_op_aux,glb_step)
        glb_step += 1
        
        
    save_path = saver.save(sess, "./temp_res/log/multi_task/model_cnn"+str(ct)+".ckpt")
    print('epoch %d loss is %f'%(ct,loss))
    
