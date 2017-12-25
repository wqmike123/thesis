# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:47:41 2017

@author: wqmike123
"""

import pandas as pd
#prepare dict
#%%
dict_dir = 'C:/Users/wqmike123/Documents/thesis/dictionary/'
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
        #'PhraseBank_50':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt',
        #'PhraseBank_75':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt',
        #'PhraseBank_all':'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt',
        'sentiwordnet': 'C:/Users/wqmike123/Documents/thesis/dictionary/lexica/SentiWordNet/SentiWordNet_3.0.0_20130122.txt',
        'mcdonald':'C:/Users/wqmike123/Documents/thesis/dictionary/mc_dict.xlsx'
        }
#%% liu bing
pos = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/positive-words.txt',comment=";", encoding='latin-1', header=None, names=["word"])
neg = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/Bing_Liu_Lexicon/negative-words.txt',comment=";", encoding='latin-1', header=None, names=["word"])
pos['Positive'] = 1
pos['Negative'] = 0
neg['Positive'] = 0
neg['Negative'] = 1
lb_dict = pd.concat([pos,neg],axis=0).rename(columns = {'word':'Word'})
lb_dict.to_excel(dict_dir + 'lb_dict.xlsx')

#%% vader
vader_columns = ["word", "vader_score", "std_der", "rates"]
vader = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/vaderLexicon/vader_sentiment_lexicon.txt',sep="\t", names=vader_columns, encoding="ISO-8859-14")
vader['Positive'] = 0
vader['Negative'] = 0
vader.loc[vader.vader_score>0,'Positive'] = 1
vader.loc[vader.vader_score<0,'Negative'] = 1
vader = vader.rename(columns = {'word':'Word'})[['Word','Positive','Negative']]
vader.to_excel(dict_dir + 'vader_dict.xlsx')

#%% vader value
vader_columns = ["word", "vader_score", "std_der", "rates"]
vader = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/vaderLexicon/vader_sentiment_lexicon.txt',sep="\t", names=vader_columns, encoding="ISO-8859-14")
#vader['Positive'] = 0
#vader['Negative'] = 0
#vader.loc[vader.vader_score>0,'Positive'] = 1
#vader.loc[vader.vader_score<0,'Negative'] = 1
vader = vader.rename(columns = {'word':'Word','vader_score':'senti'})[['Word','senti']]
vader.to_excel(dict_dir + 'vader_dict_value.xlsx')


#%% maxdiff
md = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MaxDiff_Twitter_Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt',\
                 comment=";", encoding='latin-1', header=None,\
                               names=["maxdiff_score", "word"], sep="\t")
md['Negative'] = 0
md['Positive'] = 0
md.loc[md.maxdiff_score>0,'Positive'] = 1
md.loc[md.maxdiff_score<0,'Negative'] = 1
md = md.rename(columns = {'word':'Word'})[['Word','Positive','Negative']]
md.to_excel(dict_dir + 'md_dict.xlsx')

#%% maxdiff values
md = pd.read_csv('C:/Users/wqmike123/Documents/thesis/dictionary/lexica/MaxDiff_Twitter_Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt',\
                 comment=";", encoding='latin-1', header=None,\
                               names=["maxdiff_score", "word"], sep="\t")
md = md.rename(columns = {'word':'Word','maxdiff_score':'senti'})[['Word','senti']]
md.to_excel(dict_dir + 'md_dict_value.xlsx')


#%% harvard
inq = pd.read_csv('./dictionary/inqtabs.txt',sep="\t")
poslist = ['Positiv','Active','PosAff','Pleasur','Virtue']
neglist = ['Negativ','Passive','NegAff','Pain','Vice']
inq['Positive'] = 0
inq['Negative'] = 0
for ipos in poslist:
    inq.loc[inq[ipos]==ipos,'Positive'] = 1
for ineg in neglist:
    inq.loc[inq[ineg]==ineg,'Negative'] = 1
inq = inq.rename(columns = {'Entry':'Word'})[['Word','Positive','Negative']]
inq.to_excel(dict_dir + 'inq_dict.xlsx')

#%% sentiwordnet
swn_columns = ["POS", "ID", "PosScore", "NegScore", "SynsetTerms", "Gloss"]
swn = pd.read_csv(dictDirAll['sentiwordnet'], sep="\t", skiprows=27, names=swn_columns)
swn = swn[swn["POS"].str.match("#") == False].reset_index(drop=True)
synset_terms = swn["SynsetTerms"].str.split(expand=True)
swn = pd.merge(swn, synset_terms, left_index=True, right_index=True)

# Write synset terms from columns to rows
swn = pd.melt(swn, id_vars=swn_columns, var_name="to_delete", value_name="SynsetTermWithNumber")
swn = swn.dropna()

# Split numbers of synset terms and join on swn
synset_term_numbers = swn["SynsetTermWithNumber"].str.split("#", expand=True)
synset_term_numbers.columns = ["SynsetTerm", "Number"]
swn = pd.merge(swn, synset_term_numbers, left_index=True, right_index=True)

# Make columns numeric
swn["PosScore"] = pd.to_numeric(swn["PosScore"])
swn["NegScore"] = pd.to_numeric(swn["NegScore"])
swn["Number"] = pd.to_numeric(swn["Number"])

# Rename POS Tags
swn["POS"] = swn["POS"].replace({"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"})

swn = swn.rename(columns = {'SynsetTerm':'Word','PosScore':'Positive','NegScore':'Negative'})[['Word','Positive','Negative']]
swn['senti'] = swn.Positive - swn.Negative
swn[['Word','senti']].to_excel(dict_dir + 'sentiwordnet_dict_value.xlsx')

swn.loc[swn.Positive>0,'Positive'] = 1
swn.loc[swn.Negative>0,'Negative'] = 1
swn[['Word','Positive','Negative']].to_excel(dict_dir + 'sentiwordnet_dict.xlsx')

#%% depechemood_tfidf
dep =  pd.read_csv(dictDirAll['depechemood_tfidf'],sep='\t')
dep_word = dep['Lemma#PoS'].str.split('#',expand=True)
dep_word.columns = ["Word", "PoS"]
dep['Word'] = dep_word['Word']
dep.drop('Lemma#PoS',axis=1).to_csv(dict_dir + 'depechemood.csv')

#%% msol
msol =  pd.read_csv(dictDirAll['msol'],sep=' ',names=['Word','pos'])
msol['Negative'] = 0
msol['Positive'] = 0
msol.loc[msol.pos=='positive','Positive'] = 1
msol.loc[msol.pos=='negative','Negative'] = 1
msol[['Word','Positive','Negative']].to_excel(dict_dir + 'msol_dict.xlsx')

#%% subjective
sub = pd.read_csv(dictDirAll['subjective'],sep=' ',names=['type','length','word1','pos1','stem','pos'])
sub_word = sub.word1.str.split('=',expand=True)
sub_word.columns = ['nm','Word']
sub['Word'] = sub_word.Word
pos = sub.pos.str.split('=',expand=True)
pos.columns = ['nm','pos']
sub['pos'] = pos.pos
sub['Negative'] = 0
sub['Positive'] = 0
sub.loc[sub.pos=='positive','Positive'] = 1
sub.loc[sub.pos=='negative','Negative'] = 1
sub[['Word','Positive','Negative']].to_excel(dict_dir + 'subjective_dict.xlsx')