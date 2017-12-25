# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:36:02 2017

@author: wqmike123
"""

import pandas as pd

#%%
def getLiubing(address,isNumber):
    pos = pd.read_csv(address,comment=";", encoding='latin-1', header=None, names=["word"])
    pos['Positive'] = 1
    pos['Negative'] = 0
    return pos



#%%
dictDirAll = {
        'liubing_pos':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/positive-words.txt',
        'liubing_neg':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/negative-words.txt',
        'vader':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/vaderLexicon/vader_sentiment_lexicon.txt',
        'maxdiff':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MaxDiff_Twitter_Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt',
        'harvard':'C:/Users/wqmike123/Documents/thesis/dictionary/inqtabs.txt',
        'msol':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MSOL/MSOL-June15-09.txt',
        'subjective':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff',
        'depechemood_freq':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_freq.txt',
        'depechemood_normfreq':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_normfreq.txt',
        'depechemood_tfidf':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_tfidf.txt',
        'PhraseBank_50':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt',
        'PhraseBank_75':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt',
        'PhraseBank_all':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt',
        'sentiwordnet': 'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/SentiWordNet/SentiWordNet_3.0.0_20130122.txt',
        'mcdonald':'C:/Users/wqmike123/Documents/thesis/dictionary/mc_dict.xlsx'
        }
functionMapAll = {
        'liubing_pos':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/positive-words.txt',
        'liubing_neg':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/negative-words.txt',
        'vader':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/vaderLexicon/vader_sentiment_lexicon.txt',
        'maxdiff':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MaxDiff_Twitter_Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt',
        'harvard':'C:/Users/wqmike123/Documents/thesis/dictionary/inqtabs.txt',
        'msol':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MSOL/MSOL-June15-09.txt',
        'subjective':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff',
        'depechemood_freq':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_freq.txt',
        'depechemood_normfreq':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_normfreq.txt',
        'depechemood_tfidf':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_tfidf.txt',
        'PhraseBank_50':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt',
        'PhraseBank_75':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt',
        'PhraseBank_all':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt',
        'sentiwordnet': 'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/SentiWordNet/SentiWordNet_3.0.0_20130122.txt',
        'mcdonald':'C:/Users/wqmike123/Documents/thesis/dictionary/mc_dict.xlsx'
        }
#%%
class prepareSentimentDict(object):
    
    def __init__(self,dictDir = None,functionMap = None):
        if not dictDir:
            self.dictDir = dictDirAll
        else:
            self.dictDir = dictDir
        if not functionMap:
            self.prepareDict = functionMapAll
        else:
            self.prepareDict = functionMap
    
    def prepareDict(self,isNumber = True):
        
        
    def getDictList(self):
        return self.dictDir.keys()
    
    def getDict(self,dictName,isNumber = True):
        if dictName not in self.dictDir.keys():
            return None
        else:
            return self.prepareDict[dictName](self.dictDir[dictName],isNumber)
        
        
