# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:07:15 2017

@author: wqmike123
"""

from preprocess import *
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from senticnet import Senticnet
#%%

class linguistic(object):
    
    @staticmethod
    def build_n_grams(token_list, n_range):
        '''Given a list of tokens will return a list of tokens that have been
        concatenated with the n closests tokens.'''
        all_n_grams = []
        for tokens in token_list:
            if n_range == (1,1):
                all_n_grams.append(tokens)
            else:
                all_tokens = []
                for n in range(n_range[0], n_range[1] + 1):
                    all_tokens.extend(linguistic.ngrams(tokens, n))
                all_n_grams.append(all_tokens)
    
        return all_n_grams
    
    @staticmethod    
    def ngrams(temp_tokens, n):
        token_copy = list(temp_tokens)
        gram_tokens = []
        while(len(token_copy) >= n):
            n_list = []
            for i in range(0,n):
                n_list.append(token_copy[i])
            token_copy.pop(0)
            gram_tokens.append(' '.join(n_list))
        return gram_tokens
    
    @staticmethod
    def get_ngrams_feature(token_list,n_range,sent = None,weightScheme = 'count',analyzer = 'word'):
        res = {}
        token_list = token_list.copy()
        #if not istokenlist:
        #    for i,isen in enumerate(token_list):
        #        token_list[i] = textTool.unitok_tokens(isen)
        if weightScheme == 'count':
            for i in range(n_range[0],n_range[1]+1):
                mdl = CountVectorizer(analyzer = analyzer,ngram_range = (i,i))
                res[i] = mdl.fit_transform(token_list).toarray()
        elif weightScheme == 'binary':
            for i in range(n_range[0],n_range[1]+1):
                mdl = CountVectorizer(analyzer = analyzer,ngram_range = (i,i),binary = True)
                res[i] = mdl.fit_transform(token_list).toarray()            
        elif weightScheme == 'rf':  
            if not sent:
                raise Exception("Sentiment Value not Provided")
            for i in range(n_range[0],n_range[1]+1):
                mdl = CountVectorizer(analyzer = analyzer,ngram_range = (i,i),binary = True)
                ngram = mdl.fit_transform(token_list).toarray() 
                a = ngram[sent>0,:].sum(axis=0)
                c = ngram[sent<0,:].sum(axis=0)
                a_one = a.copy()
                a_one[a_one<1] = 1
                c_one = c.copy()
                c_one[c_one<1] = 1
                rf = np.max(np.stack([np.log(2+a/c_one),np.log(2+c/a_one)]),axis=0)
                res[i] = ngram * rf
        elif weightScheme == 'tfidf':
            for i in range(n_range[0],n_range[1]+1):
                mdl = CountVectorizer(analyzer = analyzer,ngram_range = (i,i))
                ngram = mdl.fit_transform(token_list).toarray()
                doclen = np.sum(ngram,axis=1)
                tf = (ngram.transpose()/doclen).transpose()
                ngram[ngram>0] = 1
                idf = np.log(ngram.shape[0]/np.sum(ngram,axis=0))
                res[i] = tf*idf
        else:
            raise Exception('Weightscheme not Defined')
        return res                
                
                
    @staticmethod
    def analyzer(x):
        return x            

    @staticmethod
    def get_verb_feature(sentence_list,stem = True,bowDict = None):
        """
            parameter:
            -----------
            sentence_list: list
            a list of sentence (ideally stemmed)
            
            stem: bool
            whether to stem or not
            
            bowDict: defaultDict(int)
            the list map word to index
            
            dictLen: int
            length of the bowDict
            
            Returns: 
            ---------
            list
            a two-dim array, i.e. the bag of words feature vectors
            
            dict
            the bowDict
        """
        
        if not bowDict:
            bowDict = defaultdict(int)
        dictLen = len(bowDict)
        nlpServer =  StanfordCoreNLP(r'C:\Users\wqmike123\Downloads\stanford-corenlp-full-2017-06-09\stanford-corenlp-full-2017-06-09\\')
        verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
        resFeature = []
        for isentence in sentence_list:
            posRes = linguistic.sentencePoS(isentence,nlpServer)
            posResList = []
            for i in posRes:
                if i[1] not in verb:
                    continue
                if stem:
                    i = textTool.stem(i[0])
                else:
                    i = i[0]
                if i in bowDict.keys():
                    posResList.append(bowDict[i])
                else:
                    posResList.append(dictLen)
                    bowDict[i] = dictLen
                    dictLen += 1
            resFeature.append(posResList)
        
        # to bow
        resBow = np.zeros([len(sentence_list),dictLen])
        for i,ipos in enumerate(resFeature):
            for iword in ipos:
                resBow[i][iword] += 1
        return (resBow,bowDict)
                
    @staticmethod
    def get_ner_feature(sentence_list,entity_list = None):
        """
            parameter:
            -----------
            sentence_list: list
            a list of sentence (ideally stemmed)
            
            entity_list: list
            the list of entity to consider
            
            Returns: 
            ---------
            a two-dim array
            whether entity in entity_list appears
        """
        
        if not entity_list:
            entity_list = ['PERSON', 'LOCATION','ORGANIZATION', \
                           'MONEY', 'NUMBER', 'ORDINAL', 'PERCENT',\
                           'DATE', 'TIME', 'DURATION', 'SET']
        entityMap = {}
        for i,ient in enumerate(entity_list):
            entityMap[ient] = i
        nlpServer =  StanfordCoreNLP(r'C:\Users\wqmike123\Downloads\stanford-corenlp-full-2017-06-09\stanford-corenlp-full-2017-06-09\\')
        resBow = np.zeros([len(sentence_list),len(entity_list)])
        for i,isentence in enumerate(sentence_list):
            posRes = linguistic.sentenceNER(isentence,nlpServer)
            for iner in posRes:
                if iner[1] not in entity_list:
                    continue
                resBow[i][entityMap[iner[1]]] = 1
        return resBow
                            
    @staticmethod
    def sentencePoS(sentence,nlpServer = None):
        if not nlpServer:
            nlpServer = StanfordCoreNLP(r'C:\Users\wqmike123\Downloads\stanford-corenlp-full-2017-06-09\stanford-corenlp-full-2017-06-09\\')
        return nlpServer.pos_tag(sentence)

    @staticmethod
    def sentenceNER(sentence,nlpServer = None):
        if not nlpServer:
            nlpServer = StanfordCoreNLP(r'C:\Users\wqmike123\Downloads\stanford-corenlp-full-2017-06-09\stanford-corenlp-full-2017-06-09\\')
        return nlpServer.ner(sentence)

    @staticmethod
    def get_bow_feature(sentence_list,bowDict = None,istokenlist = True, stem=False):
        """
            parameter:
            -----------
            sentence_list: list
            a list of sentence (ideally stemmed)
            
            Returns: 
            ---------
            a two-dim array
            whether entity in entity_list appears
        """
        if not bowDict:
            bowDict = defaultdict(int)
        dictLen = len(bowDict)
        resFeature = []
        for isentence in sentence_list:
            if not istokenlist:
                isentence = textTool.unitok_tokens(isentence)
            posResList = []
            for iword in isentence:
                if stem:
                    iword = textTool.stem(iword)
                else:
                    iword = iword
                if iword in bowDict.keys():
                    posResList.append(bowDict[iword])
                else:
                    posResList.append(dictLen)
                    bowDict[iword] = dictLen
                    dictLen += 1
            resFeature.append(posResList)        
        # to bow
        resBow = np.zeros([len(sentence_list),dictLen])
        for i,ipos in enumerate(resFeature):
            for iword in ipos:
                resBow[i][iword] = 1
        return (resBow,bowDict)           
        
#%%
class sentiment(object):
    
    @staticmethod
    def get_opinion_score(sentence_list,istokenlist = True):
        """ compute the 5-dim opinion vectors
        
        parameter
        ----------
            sentence: list of list of tokens
        
        return
        ----------
            two lists a and b, 
                a: a list of 
                vectors of (aptitude,attention,pleasantness,polarity,sensitivity)
                
                b: a list of 
                BoC vectors for each sentence
            
        """
        
        sn = Senticnet()
        opt_score = []
        concept_list = []
        for isentence in sentence_list:
            if not istokenlist:
                isentence = textTool.unitok_tokens(isentence)
            opt_sen = np.zeros(5)
            con_sen = []
            count = 0
            for itoken in isentence:
                temp1,temp2 = sentiment.word_opinion(itoken,sn,stem=False)
                if len(temp2)==0:
                    temp1,temp2 = sentiment.word_opinion(itoken,sn,stem=True)
                    if len(temp2)==0:
                        continue
                opt_sen += temp1
                con_sen.extend(temp2)
                count += 1
            if count > 0:
                opt_score.append(opt_sen / count)
                concept_list.append(con_sen)
            else:
                opt_score.append(opt_sen)
                concept_list.append([])
        boc,_ = linguistic.get_bow_feature(concept_list)
        
        return (np.array(opt_score),boc)
                
        
        
    
    @staticmethod
    def word_opinion(word,snParser = None,stem = True):
        if stem:
            word = textTool.stem(word)
        if not snParser:
            snParser = Senticnet()
        word = word.replace(" ", "_")
        if word not in snParser.data.keys():
            return (None,[])
        concept_info = snParser.data[word]

        #sentics = {"pleasantness": concept_info[0],
        #           "attention": concept_info[1],
        #           "sensitivity": concept_info[2],
        #           "aptitude": concept_info[3],
        #           "polarity": concept_info[6]}
        
        return (np.array([concept_info[3],concept_info[1],
                          concept_info[0], concept_info[7],
                          concept_info[2]]).astype('float'),
                concept_info[8:])
    
    @staticmethod
    def get_pmi_dict(sentence_list,sentiment,istokenlist=True,stem = False,smoothing = True):
        """" compute the point-wise mutual information and return the word-pmi dict
        """
        bow,res = linguistic.get_bow_feature(sentence_list,istokenlist=istokenlist,stem = stem)
        if smoothing:
            bow = bow + np.ones(bow.shape)
        freq_w = np.sum(bow,axis=0)
        pos_ind = sentiment>0
        freq_w_pos = np.sum(bow[pos_ind,:],axis=0)
        N = len(sentiment)
        freq_pos = sum(pos_ind)
        pmi_w_pos = np.log(freq_w_pos/freq_w*N/freq_pos)
        pmi_w_neg = np.log((freq_w - freq_w_pos)/freq_w*N/(N - freq_pos))
        score = pmi_w_pos - pmi_w_neg
        for i in res.keys():
           res[i] = score[res[i]]
        return res

        
    @staticmethod
    def get_sentiment_lexicon_feature(token_list,senticdict,istokenlist=True,stem = False,isClass = True):
        """ implement in the script (more convenient)
        
            return
            -------
            a 2-dim array, where the columns correspond to
            if isClass:
                (n_pos,n_neg,cum_score,pos_ratio,neg_ratio)
            else:
                (pos_score,neg_score,cum_score,pos_ratio,neg_ratio,max_pos,min_neg)
        """
        #if isClass:
            
        #else:
        pass
    
    @staticmethod
    def get_lexicon_feature(tokenlist,sentiDict,method = 'count'):
        if method == 'count':
            res = np.array(textTool.word2vec(tokenlist,sentiDict))
            return np.array([len(res[res>0]),len(res[res<0])])                
        elif method == 'ratio':
            res = np.array(textTool.word2vec(tokenlist,sentiDict))
            return np.array([len(res[res>0])*1.0/len(res),len(res[res<0])*1.0/len(res)])          
            
        elif method == 'sum':
            res = np.array(textTool.word2vec(tokenlist,sentiDict))
            return np.array([np.sum(res[res>0]),np.sum(res[res<0]),np.sum(res)])              
        elif method == 'maxmin':
            res = np.array(textTool.word2vec(tokenlist,sentiDict))
            return np.array([np.max(res),np.min(res)])           
        else:
            raise Exception("Nonimplement method.")
            
   
    
    
#%%
class domainSpecific(object):
    pass
#%%