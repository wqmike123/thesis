#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 00:50:41 2018

@author: wq
"""
import sys
sys.path.append('./code/function/')
from preprocess import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import glob
from collections import defaultdict
#%%

def plotEventRet(news,wd,contextRange = 14):
    price = pd.read_pickle(wd+news['TICKER'].values[0]+'.pkl')
    price.index = pd.to_datetime(price.index).date
    price = pd.DataFrame(price['Adj Close'])
    price['date'] = price.index
    price = price.drop_duplicates('date')
    context = [i for i in range(-contextRange,0)] + [i for i in range(0,contextRange+1)]
    ret = np.zeros([len(news['date'].unique()),len(context)])
    abret = np.zeros([1,len(context)])
    for i,icon in enumerate(context):
        abret[0,i] = icon*price['Adj Close'].pct_change().mean()#icon*price['Adj Close'].pct_change().mean()#
    for i,idate in enumerate(news['date'].unique()):
        if idate not in price.index:
            ret[i,:] = np.nan
            continue
        for j,icon in enumerate(context):
            iday = (idate+BDay(icon)).date()
            if iday in price.index:
                ret[i,j] = ((price.loc[iday,'Adj Close']/price.loc[idate,'Adj Close']) - abret[0,j]-1)
            else:
                ret[i,j] = np.nan
    return context,np.nanmean(ret,axis=0)

#%%
def mapDir(nlen):
    if nlen<=10:
        return '10'
    for i in range(50,550,50):
        if nlen<=i:
            return str(i)
    if nlen <=1000:
        return '1000'
    else:
        return '1000_plus'
#%%
datasource = pd.read_csv('/home/wq/Documents/eth_study/thesis topic/nlp/companyNews/news.csv')
dataCount = datasource.groupby('TICKER').count()[['TITLE']].sort_values('TITLE')
#%%
data = datasource.query('TICKER=="LPTI"').set_index('PUBLICATION_DATE')
data.index = pd.to_datetime(data.index)
data['date'] = data.index.date
#data = data.groupby('date').count()['TITLE']
#data = pd.DataFrame(data)
data = data.set_index('date')
data = data[['TITLE','TICKER']]
data['date'] = data.index

x,ret = plotEventRet(data,'/home/wq/Documents/eth_study/thesis topic/nlp/marketdata/usstock/htom/')
plt.plot(x,ret)

#%%
allTicker = datasource.TICKER.unique()
resDict = defaultdict(list)
for idir in glob.glob('/home/wq/Documents/eth_study/thesis topic/nlp/marketdata/usstock/*'):
    for ifile in glob.glob(idir+'/*.pkl'):
        tn = os.path.basename(ifile).split('.pkl')[0]
        if tn not in allTicker:
            continue
        data = datasource.query('TICKER==@tn').set_index('PUBLICATION_DATE')
        nNews = len(data)
        dirName = mapDir(nNews)
        data.index = pd.to_datetime(data.index)
        data['date'] = data.index.date
        #data = data.groupby('date').count()['TITLE']
        #data = pd.DataFrame(data)
        data = data.set_index('date')
        data = data[['TITLE','TICKER']]
        data['date'] = data.index
        
        x,ret = plotEventRet(data,idir + '/')
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(x,ret)
        ax.grid()
        ax.set_xlabel('Time to News')
        ax.set_ylabel('Cumulative Abnormal Return')
        fig.tight_layout()
        fig.savefig('/home/wq/Documents/eth_study/thesis topic/nlp/writeup/car/'+dirName+'/'+tn+'.png')
        plt.close(fig)
        resDict[dirName].append(ret)
#textTool.saveData(resDict,'./temp_res/return_after_event.pickle')
#%%
key = '1000'
q_ = 55
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,np.nanpercentile(np.array(resDict[key]),axis=0,q=q_))
ax.grid()
ax.set_xlabel('Time to News')
ax.set_ylabel('Cumulative Abnormal Return')
fig.tight_layout()
fig.savefig('/home/wq/Documents/eth_study/thesis topic/nlp/writeup/carPlot/quantile/'+key+'.png')
plt.close(fig)
#%%

price = pd.read_pickle( '/home/wq/Documents/eth_study/thesis topic/nlp/marketdata/usstock/htom/DXB.pkl')
price.index = pd.to_datetime(price.index).date
price = pd.DataFrame(price['Adj Close'].pct_change())
price['date'] = price.index
data = pd.merge(data,price,how='inner',on='date')



#%% draw plot
data = pd.read_pickle('./labeldata/data1.pickle')
fig,ax = plt.subplots(figsize=(6,4))
data.sentiment.hist(bins=25,ax=ax,figure=fig)
#ax.set_title('Sentiment Distribution: SemEval')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Frequency')
fig.set_tight_layout(tight=True)
fig.savefig('../writeup/sentDist.png')


#%% prepare data set
datasource['datetime'] = pd.to_datetime(datasource['PUBLICATION_DATE'])
datasource['date'] = datasource['datetime'].dt.date


#%% build map
ticker2dir = {}
for idir in glob.glob('/home/wq/Documents/eth_study/thesis topic/nlp/marketdata/usstock/*'):
    for ifile in glob.glob(idir+'/*.pkl'):
        tn = os.path.basename(ifile).split('.pkl')[0]
        ticker2dir[tn] = ifile
#%%
dayList = [1,2,3]
res = pd.DataFrame()
for icom in datasource['TICKER'].unique():
    if icom not in ticker2dir:
        continue
    temp = datasource.query('TICKER==@icom').set_index('date').sort_index()
    price = pd.read_pickle(ticker2dir[icom])
    price.index = pd.to_datetime(price.index).date
    #price = pd.DataFrame(price['Adj Close'])
    price = price[~price.index.duplicated(keep='first')]
    price = price.sort_index()
    price['c2c'] = price['Adj Close'].pct_change()
    price['o2c'] = (price['Close']/price['Open']) - 1
    for ima in dayList:
        price['ma'+str(ima)] = price['c2c'].rolling(ima).sum().shift(-ima)
    price = price.drop(['Open',  'High', 'Low', 'Close', 'Adj Close', 'Volume'],axis=1)
    temp = temp.join(price,how='inner')
    res = pd.concat([res,temp],axis=0)
    
#%%
