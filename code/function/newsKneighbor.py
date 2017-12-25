# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:49:05 2017

@author: wqmike123
"""
import numpy as np
import gensim
from preprocess import *
import pandas as pd
#%%
class newsKneighbor(object):

    
    def __init__(self,newslist = None, eventlist=None,w2v=None,restore = False, saveDir_news = None,saveDir_event = None):
        if restore:
            self.news = textTool.loadData(saveDir_news)
            self.event = textTool.loadData(saveDir_event)
        else:
            newsMat = np.matrix(newslist[0])
            eventMat = np.matrix(eventlist[0])
            for inews,ievent in zip(newslist[1:],eventlist[1:]):
                newsMat = np.vstack([newsMat,inews])
                eventMat = np.vstack([eventMat,ievent])
            self.news = newsMat
            self.event = eventMat
        if w2v is None:
            w2v = gensim.models.Word2Vec.load('./dictionary/word2vec_models/all_fin_model_lower').wv
        w2vDict = {}
        for iword in w2v.vocab.keys():
            w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
        self.w2v = [''] + list(w2vDict.keys())
    
    def find_neighbor(self,ind,K = 20):
         cross =self.event@self.event[ind,:].transpose()
         l2 = np.sqrt(np.sum(np.square(self.event),axis=1))
         neighbor = np.argsort(cross / l2 / np.sqrt(l2[ind]),axis=0)[-K:]
         res = []
         senLen = self.news[0].shape[-1]
         for inei in neighbor:
             hd = ''
             for iword in range(senLen):
                 hd = hd+' '+ self.w2v[int(self.news[inei][0,0,iword])]
             res.append(hd)
         return res
#%%
class attentionRank(object):
    
    def __init__(self,event,attention,state,news,w2v=None):
        self.event = event
        self.attention = attention
        self.state = state
        self.news = news
        if w2v is None:
            w2v = gensim.models.Word2Vec.load('./dictionary/word2vec_models/all_fin_model_lower').wv
        w2vDict = {}
        for iword in w2v.vocab.keys():
            w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
        self.w2v = [''] + list(w2vDict.keys())
        
    def compAttention(self,ind,softmax=True):
        score = self.event[ind]@self.attention@self.state[ind]
        res = []
        senLen = self.news[0].shape[-1]
        for inei in self.news[ind]:
            hd = ''
            for iword in range(senLen):
                hd = hd+' '+ self.w2v[int(inei[iword])]
            res.append(hd)
        if softmax:
            return pd.DataFrame({'news':res,'score':self.softmax(score)})
        else:
            return pd.DataFrame({'news':res,'score':(score)})
            
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)             
