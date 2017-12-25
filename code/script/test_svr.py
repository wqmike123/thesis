# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:46:00 2017

@author: wqmike123


this script tests the model behavior of SVR with varies features
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
from featureEngineer import *
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

#%% build features
data1 = textTool.loadData('./labeldata/data1.pickle')
ngram_feature_count = linguistic.get_ngrams_feature(data1.title.values,(1,3),weightScheme='tfidf')
verb_feature,_ = linguistic.get_verb_feature(data1.title.values)
ner_feature = linguistic.get_ner_feature(data1.title.values)
opinion_feature,concept_feature = sentiment.get_opinion_score(data1.title.values,istokenlist = False)
#%% dict
dict_dir = 'C:/Users/wqmike123/Documents/thesis/dictionary/'
mc_dict = textTool.getSentDict('./dictionary/mc_dict.xlsx')
lb_dict = textTool.getSentDict(dict_dir + 'lb_dict.xlsx')
vader_dict = textTool.getSentDictValue(dict_dir + 'vader_dict_value.xlsx')
md_dict = textTool.getSentDictValue(dict_dir + 'md_dict_value.xlsx')
inq_dict = textTool.getSentDict(dict_dir + 'inq_dict.xlsx')
swn_dict = textTool.getSentDictValue(dict_dir+'sentiwordnet_dict_value.xlsx')
msol_dict = textTool.getSentDict(dict_dir+'msol_dict.xlsx')
sub_dict = textTool.getSentDict(dict_dir+'subjective_dict.xlsx')
depmood = textTool.getMoodDict(dict_dir+'depechemood.csv')
#%% features
mc_feature = data1.token.apply(sentiment.get_lexicon_feature,args = (mc_dict,))
lb_dict = textTool.getSentDict(dict_dir + 'lb_dict.xlsx')
vader_dict = textTool.getSentDictValue(dict_dir + 'vader_dict_value.xlsx')
md_dict = textTool.getSentDictValue(dict_dir + 'md_dict_value.xlsx')
inq_dict = textTool.getSentDict(dict_dir + 'inq_dict.xlsx')
swn_dict = textTool.getSentDictValue(dict_dir+'sentiwordnet_dict_value.xlsx')
msol_dict = textTool.getSentDict(dict_dir+'msol_dict.xlsx')
sub_dict = textTool.getSentDict(dict_dir+'subjective_dict.xlsx')
depmood = textTool.getMoodDict(dict_dir+'depechemood.csv')
#%% GloVec
word_model = gensim.models.Word2Vec.load('./dictionary/word2vec_models/all_fin_model_lower').wv
#w2v.save_word2vec_format('./dictionary/GoogleNews-vectors-negative300.txt', binary=False)
w2v = word_model
#%% to dict
#w2vDict = {}
#for iword in w2v.vocab.keys():
#    w2vDict[textTool.stem(iword)] = w2v.wv[iword]
w2vDict = {}
for iword in w2v.vocab.keys():
    w2vDict[textTool.unitok_tokens(iword)[0]] = w2v.wv[iword]
#%%
#word_id = textTool.loadData('./temp_res/word_id.pickle')
#w2vDict = textTool.loadData('./temp_res/w2vDict.pickle')
W,word_id = textTool.get_W(w2vDict,buildW=True)
newcol = []
add = []
#add = []
for irow in data1.token:
    tempcol,tempadd = textTool.word2ind(irow,word_id,padding_len=21,addword = False)
    newcol.append(tempcol)
    add.append(tempadd)
#    add.append(tempadd)
for i,item in enumerate(newcol):
    if len(item)!=21:
        newcol[i] = item[:21]
        
#%%
dict_dir = 'C:/Users/wqmike123/Documents/thesis/dictionary/'
mc_dict = textTool.getSentDict('./dictionary/mc_dict.xlsx')
lb_dict = textTool.getSentDict(dict_dir + 'lb_dict.xlsx')
vader_dict = textTool.getSentDict(dict_dir + 'vader_dict.xlsx')
md_dict = textTool.getSentDict(dict_dir + 'md_dict.xlsx')
inq_dict = textTool.getSentDict(dict_dir + 'inq_dict.xlsx')
mc_feature = data1.token.apply(textTool.word2vec,args=(mc_dict,)).apply(sum)
data1['lb_score'] = data1.token.apply(textTool.word2vec,args=(lb_dict,)).apply(sum)
data1['vader_score'] = data1.token.apply(textTool.word2vec,args=(vader_dict,)).apply(sum)
data1['md_score'] = data1.token.apply(textTool.word2vec,args=(md_dict,)).apply(sum)
data1['inq_score'] = data1.token.apply(textTool.word2vec,args=(inq_dict,)).apply(sum)
#%%
glovec_feature = []

for icol in newcol:
    temp = np.zeros(300)
    temp2 = np.array([-np.inf]*300)
    temp3 = np.array([np.inf]*300)
    for i in icol:
        temp+=W[i,:]
        ind = temp2<W[i,:]
        temp2[ind] = W[i,ind]
        ind = temp3>W[i,:]
        temp3[ind] = W[i,ind]
        
    glovec_feature.append(np.concatenate([temp,temp2,temp3]))
glovec_feature = np.array(glovec_feature)
#%%
x = np.concatenate([ngram_feature_count[2],ner_feature,verb_feature,opinion_feature],axis=1)
y = data1.sentiment.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#%%

def cosine(y_true, y_pred):
    return np.sum(np.dot(y_true,y_pred))/np.sqrt(np.sum(np.square(y_true)))/np.sqrt(np.sum(np.square(y_pred)))

for c in [30,20,10,5,1,0.1]:
    model = SVR(C=c,epsilon=0.1)
    model.fit(x_train,y_train)
    print('c is '+str(c)+', cosine similiary is '+str(cosine(y_test,model.predict(x_test)))) 

#%% build model pipeline: parameter tuning and cross validation

