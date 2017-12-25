#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:11:18 2017

@author: wq
"""
from collections import defaultdict
import pandas as pd
import ciseau
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from six.moves import cPickle as pickle
import unitok.configs.english
from unitok import unitok as tok
#%% dict_based mapp

class textTool(object):
    
    @staticmethod
    def sentence2list(sentence, stem = True,stopword = True,lower = False, padding=50,clean = True, tokenTool = 'my'):
        """
            preprocess the raw text input
        """
        if tokenTool == 'my':
            sentence = textTool.split(sentence)
            #if replaceword_list:
            #    for i,itoken in enumerate(sentence):
            #        if itoken in replaceword_dict.keys():
            #           sentence[i] = replaceword_dict[itoken] 
            if clean:
                sentence = textTool.clean(sentence,kind='list',TREC=lower)
            if stem:
                sentence = textTool.stem(sentence,'list')
            if stopword:
                sentence = textTool.remove_stopword(sentence)
            return sentence
            if padding>0:
                sentence = textTool.padding(sentence,padding)
        elif tokenTool == 'unitok':
            sentence = textTool.unitok_tokens(sentence)
        else:
            raise Exception("Tool not defined")
        return sentence

    @staticmethod
    def unitok_tokens(text):
        '''Tokenises using unitok http://corpus.tools/wiki/Unitok the text. Given
        a string of text returns a list of strings (tokens) that are sub strings
        of the original text. It does not return any whitespace.
        String -> List of Strings
        '''
        tokens = tok.tokenize(text, unitok.configs.english)
        return [token for tag, token in tokens if token.strip()]
    
    @staticmethod
    def split(sentence):
        return ciseau.tokenize(sentence)
    
    @staticmethod
    def remove_stopword(sentlist):
        en_stop = stopwords.words('english')
        #for sen in res:
        #    sen = [i for i in sen if not i in en_stop]
        sentlist = [i for i in sentlist if not i in en_stop]
        return sentlist
    
    @staticmethod
    def stem(sentlist,kind="sentence"):
        p_stemmer = PorterStemmer()
        if kind=="sentence":
            return p_stemmer.stem(sentlist)
        if kind=="list":
            sentlist = [p_stemmer.stem(i) for i in sentlist]
            return sentlist
        
    @staticmethod
    def clean(string,kind="sentence",TREC = True):
        if kind=="sentence":
            return textTool.cleanstr(string,TREC)
        else:
            res = []
            for i,istr in enumerate(string):
                temp = textTool.cleanstr(istr,TREC)
                if temp != '':
                    res.append(temp)     
            return res
                
    @staticmethod
    def cleanstr(string,TREC):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Every dataset is lower cased except for TREC
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
            string = re.sub(r"\'s", " \'s", string) 
            string = re.sub(r"\'ve", " \'ve", string) 
            string = re.sub(r"n\'t", " n\'t", string) 
            string = re.sub(r"\'re", " \'re", string) 
            string = re.sub(r"\'d", " \'d", string) 
            string = re.sub(r"\'ll", " \'ll", string) 
            string = re.sub(r",", " , ", string) 
            string = re.sub(r"!", " ! ", string) 
            string = re.sub(r"\(", " \( ", string) 
            string = re.sub(r"\)", " \) ", string) 
            string = re.sub(r"\?", " \? ", string) 
            string = re.sub(r"\s{2,}", " ", string)    
            return string.strip() if TREC else string.strip().lower()   
        
    @staticmethod
    def padding(sentlist,length,paddingWord = "<pad>"):
        nsent = len(sentlist)
        if nsent<=length:
            sentlist.extend([paddingWord]*(length-nsent))
        return sentlist[:length]
    
    @staticmethod
    def buildVocab(sentlist,voc=defaultdict(float),vocDoc=defaultdict(float)):
        for iword in sentlist:
            voc[iword] += 1
            vocDoc[iword] += 1 
        
    @staticmethod
    def word2vec(senlist,word2vec,vocDoc=None,learnMissing = False,min_df = 10):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        res = []
        if isinstance(list(word2vec.values())[0],list):
            k = len(list(word2vec.values())[0])
        else:
            k = 1
        if vocDoc:
            for word in senlist:
                if word not in word2vec and vocDoc[word] >= min_df:
                    word2vec[word] = np.random.uniform(-0.25,0.25,k)
                if word in word2vec:
                    res.append(word2vec[word])
            return res
        else:
            for word in senlist:
                if word not in word2vec and learnMissing:
                    word2vec[word] = np.random.uniform(-0.25,0.25,k)
                if word in word2vec:
                    res.append(word2vec[word])
            return res            
    @staticmethod
    def getSentDict(dictDir,stem=True):
        p_stemmer = PorterStemmer()
        mcDict = pd.read_excel(dictDir)
        mcDict.Word = mcDict.Word.apply(str).apply(str.lower)#str.encode('utf-8').
        mcDict.Word = mcDict.Word.apply(textTool.cleanstr,args=(False,))
        if stem:
            mcDict.index = mcDict.Word.apply(p_stemmer.stem)
        else:
            mcDict.index = mcDict.Word
        mcDict['senti'] = 0.
        mcDict.loc[mcDict[mcDict.Negative>0].index,'senti'] = -1.
        mcDict.loc[mcDict[mcDict.Positive>0].index,'senti'] = 1.
        mcDict = mcDict[['senti']].to_dict()['senti']
        return mcDict
    @staticmethod
    def getSentDictValue(dictDir,stem=True):
        p_stemmer = PorterStemmer()
        mcDict = pd.read_excel(dictDir)
        mcDict.Word = mcDict.Word.apply(str).apply(str.lower)#str.encode('utf-8').
        mcDict.Word = mcDict.Word.apply(textTool.cleanstr,args=(False,))
        if stem:
            mcDict.index = mcDict.Word.apply(p_stemmer.stem)
        else:
            mcDict.index = mcDict.Word
        mcDict = mcDict[['senti']].to_dict()['senti']
        return mcDict
    @staticmethod
    def getMoodDict(dictDir,stem=True):
        p_stemmer = PorterStemmer()
        mcDict = pd.read_csv(dictDir)
        mcDict.Word = mcDict.Word.apply(str).apply(str.lower)#str.encode('utf-8').
        mcDict.Word = mcDict.Word.apply(textTool.cleanstr,args=(False,))
        if stem:
            mcDict.index = mcDict.Word.apply(p_stemmer.stem)
        else:
            mcDict.index = mcDict.Word
        return mcDict
    
    @staticmethod
    def saveData(obj,dirf,protocol=pickle.HIGHEST_PROTOCOL):
        with open(dirf, 'wb') as f:
            pickle.dump(obj, f, protocol = protocol)
 
    @staticmethod
    def loadData(dirf):
        with open(dirf, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    @staticmethod
    def get_W(word_vecs, buildW = False,k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        if buildW:
            W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)
            W[0] = np.zeros(k, dtype=np.float32)
        i = 1
        for word in word_vecs:
            if buildW:
                W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map
    
    @staticmethod
    def word2ind(wordlist, word_ind, addword = True,k=300,fill0 = True,padding_len = 50, paddingAtEnd = True):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        if fill0:
            res = [0]*len(wordlist)
            maxind = len(word_ind)
            count = 0
            for i,iword in enumerate(wordlist):
                if iword in word_ind:
                    res[i] = word_ind[iword]
                else:
                    if addword:
                        maxind += 1
                        word_ind[iword] = maxind
                        count += 1
            if paddingAtEnd:
                res = res + [0]*(padding_len - len(wordlist))
            else:
                res = [0]*(padding_len - len(wordlist)) + res
            return [res,count]
        else:
            res = []
            maxind = len(word_ind)
            count = 0
            for i,iword in enumerate(wordlist):
                if iword in word_ind:
                    res.append(word_ind[iword])
                else:
                    if addword:
                        maxind += 1
                        word_ind[iword] = maxind
                        count += 1
            if paddingAtEnd:
                res = res + [0]*(padding_len - len(res))
            else:
                res = [0]*(padding_len - len(res)) + res
            return [res,count]                    
    
    
    @staticmethod 
    def load_bin_vec(fname, vocab,maxl = 50000):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        # coding: utf-8
#        from __future__ import division
#        
#        import struct
#        import sys
        
        FILE_NAME = "GoogleNews-vectors-negative300.bin"
        MAX_VECTORS = 200000 # This script takes a lot of RAM (>2GB for 200K vectors), if you want to use the full 3M embeddings then you probably need to insert the vectors into some kind of database
        FLOAT_SIZE = 4 # 32bit float
        
        vectors = dict()
        
        with open(FILE_NAME, 'rb') as f:
            
            c = None
            
            # read the header
            header = ""
            while c != "\n":
                c = f.read(1)
                header += c
        
            total_num_vectors, vector_len = (int(x) for x in header.split())
            num_vectors = min(MAX_VECTORS, total_num_vectors)
            
            print("Number of vectors: %d/%d" % (num_vectors, total_num_vectors))
            print("Vector size: %d" % vector_len)
        
            while len(vectors) < num_vectors:
        
                word = ""        
                while True:
                    c = f.read(1)
                    if c == " ":
                        break
                    word += c
        
                binary_vector = f.read(FLOAT_SIZE * vector_len)
                vectors[word] = [ struct.unpack_from('f', binary_vector, i)[0] 
                                  for i in xrange(0, len(binary_vector), FLOAT_SIZE) ]
                
                sys.stdout.write("%d%%\r" % (len(vectors) / num_vectors * 100))
                sys.stdout.flush()
        
        import cPickle
        
        print("\nSaving...")
        with open(FILE_NAME[:-3] + "pcl", 'wb') as f:
            cPickle.dump(vectors, f, cPickle.HIGHEST_PROTOCOL)
        return word_vecs
    @staticmethod
    def balanceData(x,y):
        n = len(y)
        if sum(np.logical_or(np.equal(y,0), np.equal(y,1)))<n:
            y[y>0] = 1
            y[y<0] = 0
        n1 = sum(y)
        ratio = n1*1.0/(n-n1)
        if ratio>1:
            tar = 0
            ratio = int(ratio)
        else:
            tar = 1
            ratio = int(1/ratio)
        
        train_y = list(y)+[i for i in y if i==tar]*(ratio-1)
        train_x = list(x)+[i for i,j in zip(x,y) if j==tar]*(ratio-1)
        temp = np.arange(len(train_y))
        np.random.shuffle(temp)
        train_x = [train_x[i] for i in temp]
        train_y = [train_y[i] for i in temp]           
        return (np.array(train_x),np.array(train_y))