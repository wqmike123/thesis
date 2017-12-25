# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:04:45 2017

@author: wqmike123
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
vol = pd.read_csv('./marketdata/vixcurrent.csv',header=1)
vol.Date = pd.to_datetime(vol.Date)
vol = vol.set_index('Date')[['VIX Close']].rename(columns = {'VIX Close':'VIX'})
vol = vol['2006':]
spx = pd.read_csv('./marketdata/spx500.csv')
spx.Date = pd.to_datetime(spx.Date)
spx = spx.set_index('Date')[['Adj Close']].rename(columns = {'Adj Close':'SPX'})
ted = pd.read_csv('./marketdata/TEDRATE.csv',names = ['Date','TED'],header=0)
ted.Date = pd.to_datetime(ted.Date)
ted = ted.set_index('Date')
ted = ted.replace('.',np.nan).astype('float')


data = pd.concat([vol,spx,ted],axis=1).ffill()
#%% normalize
#for icol in data.columns:
data['VIX'] = data['VIX'] / data['VIX'].values[0]
data['TED'] = data['TED'] / data['TED'].values[0]

#%%
def tag(ax,time,event,loc,bcolor,val = 'VIX',ha = 'right',va = 'bottom'):
    x = data.loc[time,:].name
    ax.annotate(
    event,
    xy=(x, data.loc[x,val]), xytext=(loc[0], loc[1]),
    textcoords='offset points', ha=ha, va=va,
    bbox=dict(boxstyle='round,pad=0.5', fc=bcolor, alpha=0.5),
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
fig,ax = plt.subplots(figsize=[12,8])
data['2007'].plot(ax=ax,secondary_y='SPX')
tag(ax,'2007-02-07','HSBC announces losses linked to US subprime mortgages',loc=[-80,5],bcolor='red')
tag(ax,'2007-02-26','Former Fed chair forecasts a possible recession',loc=[-115,25],bcolor='red')
tag(ax,'2007-04-03','Largest subprime lender files for bankruptcy',loc =[-180,40],bcolor='red')
tag(ax,'2007-05-17','Ben Bernanke says mortgage defaults will not harm US',loc=[-30,-120],bcolor= 'green',ha='center')
tag(ax,'2007-06-22','Two Bear Stearns-run hedge funds run into large losses',loc=[-20,-200],bcolor='red',ha='center')
tag(ax,'2007-08-09','BNP freezes three of their funds',loc=[-20,-200],bcolor='red',val='TED',ha='center')
tag(ax,'2007-08-17','Fed cuts lending rates to banks',loc=[-10,100],bcolor='green',val='TED')
tag(ax,'2007-09-13','Northern Rock borrows emergency financial support',loc=[60,-90],bcolor='red',ha='center')
tag(ax,'2007-09-18','Fed cuts interest rate by half',loc=[-20,180],bcolor='green',val='TED')
tag(ax,'2007-10-01','UBS announces 3.4bn losses',loc=[0,-90],bcolor='red',ha='center')
tag(ax,'2007-10-30',"Merrill Lynch's chief resigns for unveiling bad debts",loc=[-20,-85],bcolor='red',ha='center')
tag(ax,'2007-11-20',"IMF approves loans for Iceland",loc=[100,100],bcolor='green',ha='left')
tag(ax,'2007-12-06',"Bush outlines plans to help homeowners facing foreclosure",loc=[60,60],bcolor='green',ha='left')

ax.grid()


