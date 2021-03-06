#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:11:18 2017

@author: wq
"""
import pandas as pd
import numpy as np
from six.moves import cPickle as pickle
from preprocess import *
import gensim
from preprocess import textTool
#%% dict_based mapp

class trainTool(object):
    
    
    @staticmethod
    def prepareTokenData(name ='vixPred_data' ,full = True,isHead = True,save=True):
        datalist = [textTool.loadData('./temp_res/data.pickle')]
        if full:
            data2 = textTool.loadData('temp_res/vixPred_data1116.pickle').reset_index()
            datalist.append(data2)
        for data in datalist:
            res = []
            if isHead:
                content = 'head'
            else:
                content = 'summary'
            for isample in data[content]:
                if not isinstance(isample,str):
                    res.append([])
                    continue
                res.append(textTool.sentence2list(isample.lower(),tokenTool='unitok'))
            data2['token'] = res
        data = pd.concat(data,axis=0)
        if save:
            #data.token.apply(len).max()#27
        
            textTool.saveData(data,'./temp_res/'+name +'_' +content+'.pickle')
        else:
            return data

    @staticmethod
    def prepareSentimentDataset(df,sentcol='sentiment',listcol='token',labelthreshold_pos = 1,labelthreshold_neg = -1,\
                                keep_neutral = False,train_ratio = 1):
        """
        return the train_data,train_label (, test_data,test_label)
        """
        if not keep_neutral:
            data = df[np.logical_or(df.loc[:,sentcol]>=labelthreshold_pos,df.loc[:,sentcol]<=labelthreshold_neg)]
        else:
            data = df.copy()
        trainindex = len(data.index)*train_ratio
        if train_ratio<1:
            return (data.loc[data.index[:trainindex],listcol],data.loc[data.index[:trainindex],sentcol],\
                    data.loc[data.index[trainindex:],listcol],data.loc[data.index[trainindex:],sentcol])
        else:
            return (data.loc[:,listcol],data.loc[:,sentcol])
        
#    @staticmethod
#    def getEmbeddingW(dictDir):
##        
#    @staticmethod
#    def word2index(wordlist,indexdict):
    @staticmethod
    def prepareVIX(cwd,isClass = False,threshold = 0.02,lag = 1,freq = 'd'):
        vol = pd.read_csv(cwd+'/marketdata/vixcurrent.csv',header=1)
        vol.Date = pd.to_datetime(vol.Date)
        vol = vol.set_index('Date')
        vol = vol['2006-01-01':]
        vol.index = [i.date() for i in vol.index]
        target = vol['VIX Close'].pct_change().shift((lag-1)).dropna()
        if freq != 'd':
            if lag != 0:
                lagbool = True
            else:
                lagbool = False
            target = trainTool.resample(pd.DataFrame(target),freq =freq,lag = lagbool)
            target = target.groupby(target.index).sum()['VIX Close']
        if isClass:
            label = np.zeros([len(target),3])
            label[target.values<-threshold,0] = 1
            label[target.values>threshold,2] = 1
            label[np.logical_not(np.logical_or(target.values>threshold,target.values<-threshold)),1] = 1
            return target,label
        else:
            return target
        
    @staticmethod
    def prepareSPX(cwd,isClass = False,threshold = 0.02,lag = 1,freq = 'd'):
        vol = pd.read_csv(cwd+'/marketdata/spx500.csv')
        vol.Date = pd.to_datetime(vol.Date)
        vol = vol.set_index('Date')
        vol = vol['2006-01-01':]
        vol.index = [i.date() for i in vol.index]
        target = vol['Adj Close'].pct_change().shift((lag-1)).dropna()
        if freq != 'd':
            if lag != 0:
                lagbool = True
            else:
                lagbool = False
            target = trainTool.resample(pd.DataFrame(target),freq =freq,lag = lagbool)
            target = target.groupby(target.index).sum()['Adj Close']        
        if isClass:
            label = np.zeros([len(target),3])
            label[target.values<-threshold,0] = 1
            label[target.values>threshold,2] = 1
            label[np.logical_not(np.logical_or(target.values>threshold,target.values<-threshold)),1] = 1
            return target,label
        else:
            return target        
        
    @staticmethod
    def prepareWSJNews(cwd,file = 'vixPred_data',length = 'full',maxlen = 31):
        data = textTool.loadData(cwd+'/temp_res/'+file+'.pickle')
        if length == 'full':
            data = textTool.loadData(cwd+'/temp_res/'+file+'_full.pickle').reset_index()
            #data = pd.concat([data,data2],axis=0)
        data.date = data.date.dt.date
        data = data.set_index('date')
        data = data.sort_index()
        word_model = gensim.models.Word2Vec.load(cwd+'/dictionary/word2vec_models/all_fin_model_lower').wv
        #w2v.save_word2vec_format('./dictionary/GoogleNews-vectors-negative300.txt', binary=False)
        w2v = word_model
        w2vDict = {}
        for iword in w2v.vocab.keys():
            w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
        W,word_id = textTool.get_W(w2vDict,buildW=True)
        newcol = []
        add = []
        #add = []
        for irow in data.token:
            tempcol,tempadd = textTool.word2ind(irow,word_id,addword = False,padding_len=maxlen,fill0=False)
            newcol.append(tempcol)
            add.append(tempadd)
        #    add.append(tempadd)
        for i,item in enumerate(newcol):
            if len(item)!=maxlen:
                newcol[i] = item[:maxlen]
        W = np.vstack((W,np.zeros(shape=(sum(add),300))))
        newcol = np.array(newcol)
        news = pd.DataFrame(index=data.index)
        news['news'] = newcol.tolist()
        news.news = news.news.apply(np.array)
        return W,news
    
    @staticmethod
    def resample(news,freq = 'W',lag = True):
        start = news.index[0]
        end = news.index[-1]
        if freq == 'm':
            addon = 30
        elif freq == 'W':
            addon = 7
        timeindex = [i.date() for i in pd.date_range(start,(end+pd.offsets.Day(addon)).date(),freq = freq)]
        last_date = (start - pd.offsets.Day(1)).date()
        #news['date'] = start
        for idate in timeindex:
            if lag:
                ind = news[np.logical_and(news.index>=last_date,news.index<idate)].index
                if len(ind)==0:
                    continue
                news.loc[ind,'date'] = idate
            else:
                ind = news[np.logical_and(news.index>last_date,news.index<=idate)].index
                if len(ind)==0:
                    continue
                news.loc[ind,'date'] = idate
            last_date = idate
        news.index = news.date
        news = news.drop('date',axis=1)
        return news

    @staticmethod
    def prepareFakeData(isClass = False,cwd = './',maxlen = 27,threshold = 0.3):
        noisy_data = textTool.loadData(cwd + '/temp_res/vixPred_data_head_full.pickle')
        noisy_data = noisy_data.sort_index()[['head','token']]
        cind = noisy_data.groupby('date').count().query('head>20')
        data = textTool.loadData(cwd+'/labeldata/data_token.pickle').rename(columns = {'title':'head'})[['head','token','sentiment']]
        # random pick to prepare the daily news
        dates = cind.index[:len(data)]
        for i,idate in enumerate(dates):
            rd = np.random.randint(len(noisy_data.loc[idate]))
            noisy_data.loc[idate].iloc[rd]['token'] = data.loc[i,'token']
        noisy_data = noisy_data.loc[dates]
        word_model = gensim.models.Word2Vec.load(cwd+'/dictionary/word2vec_models/all_fin_model_lower').wv
        #w2v.save_word2vec_format('./dictionary/GoogleNews-vectors-negative300.txt', binary=False)
        w2v = word_model
        w2vDict = {}
        for iword in w2v.vocab.keys():
            w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
        W,word_id = textTool.get_W(w2vDict,buildW=True)
        newcol = []
        add = []
        #add = []
        for irow in noisy_data.token:
            tempcol,tempadd = textTool.word2ind(irow,word_id,addword = False,padding_len=maxlen,fill0=False)
            newcol.append(tempcol)
            add.append(tempadd)
        #    add.append(tempadd)
        for i,item in enumerate(newcol):
            if len(item)!=maxlen:
                newcol[i] = item[:maxlen]
        W = np.vstack((W,np.zeros(shape=(sum(add),300))))
        newcol = np.array(newcol)
        news = pd.DataFrame(index=noisy_data.index)
        news['news'] = newcol.tolist()
        news.news = news.news.apply(np.array)
        target = pd.DataFrame({'sentiment':data.sentiment.values},index = dates)['sentiment']
        if isClass:
            label = np.zeros([len(target),3])
            label[target.values<-threshold,0] = 1
            label[target.values>threshold,2] = 1
            label[np.logical_not(np.logical_or(target.values>threshold,target.values<-threshold)),1] = 1
            return W,news,target,label
        else:
            return W,news,target
    @staticmethod
    def shuffle(dataList,seed = 17):
        n = len(dataList[0])
        np.random.seed(seed)
        rn = np.arange(n)
        np.random.shuffle(rn)
        for i,idata in enumerate(dataList):
            dataList[i] = np.array([idata[j] for j in rn])
        return dataList
    @staticmethod
    def batch(dataList,batchsize):
        n = len(dataList[0])
        resList = [[] for i in range(n)]
        nbatch = int(n/batchsize) + 1
        for i,idata in enumerate(dataList):
            for step in range(nbatch):
                offset = (step * batchsize)
                data = idata[offset:offset+batchsize,:]
                resList[i].append(data)
        return resList
    
    @staticmethod
    def cosine(y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        return np.sum(np.dot(y_true,y_pred))/np.sqrt(np.sum(np.square(y_true)))/np.sqrt(np.sum(np.square(y_pred)))