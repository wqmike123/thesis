# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:51:30 2017

@author: wqmike123
"""
import sys
sys.path.append('./code/function/')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import *
from fcnn import *
#%%
def cosine(y_true, y_pred):
    return sum(np.dot(y_true,y_pred))/np.sqrt(sum(np.square(y_true)))/np.sqrt(sum(np.square(y_pred)))
#textTool.saveData(data,'./temp_res/data2.pickle',protocol=2)
#%%
data = textTool.loadData('./temp_res/data.pickle')
#%% word cloud and distribution
crisis_start = '2007-12-01'
crisis_end = '2009-06-01'
fig,axx = plt.subplots(figsize = [8,6])
nNews = data.groupby(by='date')['sentiment'].count()
#nNews.rolling(7).mean().plot(ax = axx)
#data['year'] = data.date.apply(lambda x: x.year)
#data['month'] = data.date.apply(lambda x: x.month)
#data['month'] = data['year'].apply(str).values  + data['month'].apply(str).values
data.set_index('date')['sentiment'].resample('1M').count().plot(ax=axx)
axx.axvspan(pd.to_datetime(crisis_start),pd.to_datetime(crisis_end), color='y', alpha=0.2, lw=0)
axx.set_ylabel('Total Amounts of News Every Month')
axx.set_xlabel('Date')
axx.grid()
fig.savefig('./preanalysis/countMon.png')


#%% read the label data
labeldata_dir = r'C:/Users/wqmike123/Documents/thesis/labeldata/'
data1 = pd.read_json(labeldata_dir+'Headline_Trainingdata.json')
data2 = pd.read_csv(labeldata_dir + 'newsheadline.csv')
data1_test = pd.read_json(labeldata_dir + 'Headline_Trialdata.json')
#%% data1
res = []
for isample in data1.title:
    if not isinstance(isample,str):
        res.append([])
        continue
    res.append(textTool.sentence2list(isample))
data1['token'] = res
#%%data1_test
res = []
for isample in data1_test.title:
    if not isinstance(isample,str):
        res.append([])
        continue
    res.append(textTool.sentence2list(isample))
data1_test['token'] = res
#%%
res = []
for isample in data2.Headline:
    if not isinstance(isample,str):
        res.append([])
        continue
    res.append(textTool.sentence2list(isample))
data2['token'] = res
#%% 
textTool.saveData(data1,'./labeldata/data1.pickle')
textTool.saveData(data2,'./labeldata/data2.pickle')
textTool.saveData(data1_test,'./labeldata/data1_test.pickle')
#%% get dictionary
dict_dir = 'C:/Users/wqmike123/Documents/thesis/dictionary/'
mc_dict = textTool.getSentDict('./dictionary/mc_dict.xlsx')
lb_dict = textTool.getSentDict(dict_dir + 'lb_dict.xlsx')
vader_dict = textTool.getSentDict(dict_dir + 'vader_dict.xlsx')
md_dict = textTool.getSentDict(dict_dir + 'md_dict.xlsx')
inq_dict = textTool.getSentDict(dict_dir + 'inq_dict.xlsx')
#%% liu bing
#pos = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/positive-words.txt',comment=";", encoding='latin-1', header=None, names=["word"])
#neg = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/negative-words.txt',comment=";", encoding='latin-1', header=None, names=["word"])
#pos['Positive'] = 1
#pos['Negative'] = 0
#neg['Positive'] = 0
#neg['Negative'] = 1
#lb_dict = pd.concat([pos,neg],axis=0).rename(columns = {'word':'Word'})
#lb_dict.to_excel(dict_dir + 'lb_dict.xlsx')

#%% vader
#vader_columns = ["word", "vader_score", "std_der", "rates"]
#vader = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/vaderLexicon/vader_sentiment_lexicon.txt',sep="\t", names=vader_columns, encoding="ISO-8859-14")
#vader['Positive'] = 0
#vader['Negative'] = 0
#vader.loc[vader.vader_score>0,'Positive'] = 1
#vader.loc[vader.vader_score<0,'Negative'] = 1
#vader = vader.rename(columns = {'word':'Word'})[['Word','Positive','Negative']]
#vader.to_excel(dict_dir + 'vader_dict.xlsx')

#%% maxdiff
#md = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MaxDiff_Twitter_Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt',\
#                 comment=";", encoding='latin-1', header=None,\
#                               names=["maxdiff_score", "word"], sep="\t")
#md['Negative'] = 0
#md['Positive'] = 0
#md.loc[md.maxdiff_score>0,'Positive'] = 1
#md.loc[md.maxdiff_score<0,'Negative'] = 1
#md = md.rename(columns = {'word':'Word'})[['Word','Positive','Negative']]
#md.to_excel(dict_dir + 'md_dict.xlsx')

##%% harvard
#inq = pd.read_csv('./dictionary/inqtabs.txt',sep="\t")
#poslist = ['Positiv','Active','PosAff','Pleasur','Virtue']
#neglist = ['Negativ','Passive','NegAff','Pain','Vice']
#inq['Positive'] = 0
#inq['Negative'] = 0
#for ipos in poslist:
#    inq.loc[inq[ipos]==ipos,'Positive'] = 1
#for ineg in neglist:
#    inq.loc[inq[ineg]==ineg,'Negative'] = 1
#inq = inq.rename(columns = {'Entry':'Word'})[['Word','Positive','Negative']]
#inq.to_excel(dict_dir + 'inq_dict.xlsx')

#%% prediction accuracy
data1 = textTool.loadData('./labeldata/data1.pickle')
data2 = textTool.loadData('./labeldata/data2.pickle')
#%%
data1['mc_score'] = data1.token.apply(textTool.word2vec,args=(mc_dict,)).apply(sum)
data1['lb_score'] = data1.token.apply(textTool.word2vec,args=(lb_dict,)).apply(sum)
data1['vader_score'] = data1.token.apply(textTool.word2vec,args=(vader_dict,)).apply(sum)
data1['md_score'] = data1.token.apply(textTool.word2vec,args=(md_dict,)).apply(sum)
data1['inq_score'] = data1.token.apply(textTool.word2vec,args=(inq_dict,)).apply(sum)


data2['mc_score'] = data2.token.apply(textTool.word2vec,args=(mc_dict,)).apply(sum)
data2['lb_score'] = data2.token.apply(textTool.word2vec,args=(lb_dict,)).apply(sum)
data2['vader_score'] = data2.token.apply(textTool.word2vec,args=(vader_dict,)).apply(sum)
data2['md_score'] = data2.token.apply(textTool.word2vec,args=(md_dict,)).apply(sum)
data2['inq_score'] = data2.token.apply(textTool.word2vec,args=(inq_dict,)).apply(sum)

data1_test['mc_score'] = data1_test.token.apply(textTool.word2vec,args=(mc_dict,)).apply(sum)
data1_test['lb_score'] = data1_test.token.apply(textTool.word2vec,args=(lb_dict,)).apply(sum)
data1_test['vader_score'] = data1_test.token.apply(textTool.word2vec,args=(vader_dict,)).apply(sum)
data1_test['md_score'] = data1_test.token.apply(textTool.word2vec,args=(md_dict,)).apply(sum)
data1_test['inq_score'] = data1_test.token.apply(textTool.word2vec,args=(inq_dict,)).apply(sum)


def getAcc(dataset,dictname):
    nPos = dataset.sentiment.values > 0
    nNeg = dataset.sentiment.values < 0
    nNeu = dataset.sentiment.values == 0
    pred = dataset[dictname + '_score'].values

    return (round(np.sum(np.logical_and(nPos,pred>0))*100./np.sum(nPos),2),\
            round(np.sum(np.logical_and(nNeg,pred<0))*100./np.sum(nNeg),2),\
            round(np.sum(np.logical_and(nNeu,pred==0))*100./np.sum(nNeu),2),\
            round((np.sum(np.logical_and(nPos,pred>0))+np.sum(np.logical_and(nNeg,pred<0))+\
                  np.sum(np.logical_and(nNeu,pred==0)))*100./len(dataset),2))
    
#%%
word_id = textTool.loadData('./temp_res/word_id.pickle')
data1 = textTool.loadData('./labeldata/data1.pickle')
newcol_test_x = []
#add = []
for irow in data1.token:
    tempcol,tempadd = textTool.word2ind(irow,word_id,addword = False,padding_len=20)
    newcol_test_x.append(tempcol)

#    add.append(tempadd)
for i,item in enumerate(newcol_test_x):
    if len(item)!=20:
        newcol_test_x[i] = item[:20]
        
#%%
model_rnn = mylstm(20,140646,embedweight = W,lstm_units = 512,hidden_dims = [256])
model_rnn.fit(truncated[:30000],ylabel[:30000,:],truncated[30000:],ylabel[30000:,:])
pred_rnn = model_rnn.predict(newcol_test_x)
#%%
train_x = np.array(data1[data1.columns[-5:]].values)
train_y = np.zeros([len(data1),3])
train_y[data1.sentiment>0,2] = 1
train_y[data1.sentiment==0,1] = 1
train_y[data1.sentiment<0,0] = 1
train_y_reg = data1.sentiment.values
test_x = np.array(data1_test[data1_test.columns[-5:]].values)
test_y = np.zeros([len(data1_test),3])
test_y[data1_test.sentiment>0,2] = 1
test_y[data1_test.sentiment==0,1] = 1
test_y[data1_test.sentiment<0,0] = 1
test_y_reg = data1_test.sentiment.values
#model_fc = FCReg(5,hidden_dims = [1024,128,8],batch_size = 5,epochs_number = 200, dropout=0.1, 
#                 learning_rate = 0.1,decay_rate = 1e-4)
model_fc = FCReg(5,hidden_dims = [32,8],batch_size = 5,epochs_number = 200, dropout=0.1, 
                 learning_rate = 0.001,decay_rate = 1e-4)
model_fc.fit(train_x[:1000,:],train_y_reg[:1000],test_x,test_y_reg)#train_x[1000:,:],train_y[1000:,:])
#pred_y = np.argmax(model_fc.predict(train_x[1000:,:]),axis=1)
#act_y = np.argmax(train_y[1000:,:],axis=1)
print('Accuracy: {}'.format(cosine(train_y_reg[1000:],model_fc.predict(train_x[1000:,:]))))

#####################################################
#%% try 15 features:
def toLabel(x,tar):
    ct = 0
    for i in x:
        if i==tar:
            ct+=1
    return ct
for i,ilabel in enumerate([-1,0,1]):
    data1['mc_score'+str(i)] = data1.token.apply(textTool.word2vec,args=(mc_dict,)).apply(toLabel,args=(ilabel,))
    data1['lb_score'+str(i)] = data1.token.apply(textTool.word2vec,args=(lb_dict,)).apply(toLabel,args=(ilabel,))
    data1['vader_score'+str(i)] = data1.token.apply(textTool.word2vec,args=(vader_dict,)).apply(toLabel,args=(ilabel,))
    data1['md_score'+str(i)] = data1.token.apply(textTool.word2vec,args=(md_dict,)).apply(toLabel,args=(ilabel,))
    data1['inq_score'+str(i)] = data1.token.apply(textTool.word2vec,args=(inq_dict,)).apply(toLabel,args=(ilabel,))


    data1_test['mc_score'+str(i)] = data1_test.token.apply(textTool.word2vec,args=(mc_dict,)).apply(toLabel,args=(ilabel,))
    data1_test['lb_score'+str(i)] = data1_test.token.apply(textTool.word2vec,args=(lb_dict,)).apply(toLabel,args=(ilabel,))
    data1_test['vader_score'+str(i)] = data1_test.token.apply(textTool.word2vec,args=(vader_dict,)).apply(toLabel,args=(ilabel,))
    data1_test['md_score'+str(i)] = data1_test.token.apply(textTool.word2vec,args=(md_dict,)).apply(toLabel,args=(ilabel,))
    data1_test['inq_score'+str(i)] = data1_test.token.apply(textTool.word2vec,args=(inq_dict,)).apply(toLabel,args=(ilabel,))
#%%
train_x = np.array(data1[data1.columns[-15:]].values)
train_y = np.zeros([len(data1),3])
train_y[data1.sentiment>0,2] = 1
train_y[data1.sentiment==0,1] = 1
train_y[data1.sentiment<0,0] = 1
train_y_reg = data1.sentiment.values
test_x = np.array(data1_test[data1_test.columns[-15:]].values)
test_y = np.zeros([len(data1_test),3])
test_y[data1_test.sentiment>0,2] = 1
test_y[data1_test.sentiment==0,1] = 1
test_y[data1_test.sentiment<0,0] = 1
test_y_reg = data1_test.sentiment.values
#model_fc = FCReg(5,hidden_dims = [1024,128,8],batch_size = 5,epochs_number = 200, dropout=0.1, 
#                 learning_rate = 0.1,decay_rate = 1e-4)
model_fc = FCReg(15,hidden_dims = [32,8],batch_size = 5,epochs_number = 200, dropout=0.1, 
                 learning_rate = 0.001,decay_rate = 1e-4)
model_fc.fit(train_x[:1000,:],train_y_reg[:1000],test_x,test_y_reg)#train_x[1000:,:],train_y[1000:,:])
#pred_y = np.argmax(model_fc.predict(train_x[1000:,:]),axis=1)
#act_y = np.argmax(train_y[1000:,:],axis=1)
print('Accuracy: {}'.format(cosine(train_y_reg[1000:],model_fc.predict(train_x[1000:,:]))))