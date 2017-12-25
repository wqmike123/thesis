# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:25:57 2017

@author: wqmike123
"""

import pandas as pd
import os
import glob
import sys
sys.path.append('./code/function/')
from featureEngineer import *
#%%
dbdir= 'C:/Users/wqmike123/Documents/database/financial-news-dataset-master/20061020_20131126_bloomberg_news/'
for idate in glob.glob(dbdir + '*'):
    ifile = pd.read_csv(glob.glob(idate+'/*')[0],sep='\n')



#%%
coNews = []
for i in data:
    if 'company' in i['href']:
        coNews.append(i['title'])
#%%
dbdir = 'C:/Users/wqmike123/Documents/database/Reuters-full-data-set-master/Reuters-full-data-set-master/data'
posRes = []
nlpServer =  StanfordCoreNLP(r'C:\Users\wqmike123\Downloads\stanford-corenlp-full-2017-06-09\stanford-corenlp-full-2017-06-09\\')
coNews = []
ts = []
fake = ['PRESS DIGEST','DIARY -']
datelist = []
for idate in glob.glob(dbdir + '/*.pkl'):
    data = pd.read_pickle(idate)
    temdate = os.path.basename(idate).split('.')[0]
    for i in data:
        if 'company' in i['href']:
            flag = False
            for j in fake:
                if j in i['title']:
                    flag = True
                    break
            if flag:
                continue
            coNews.append(i['title'])
            ts.append(i['ts'])
            datelist.append(temdate)

res = pd.DataFrame({'date':datelist,'title':coNews,'time':ts})
#res.to_pickle('./temp_res/reuters_company.pkl')
#%%

for i,isentence in enumerate(coNews):
    temp = linguistic.sentenceNER(isentence,nlpServer)
    org = []
    for i in temp:
        if i[1]=='ORGANIZATION':
            org.append(i[0])
    posRes.append(org)
res['company'] = posRes
#%%
res = pd.read_pickle('./temp_res/reuters_company.pkl')
